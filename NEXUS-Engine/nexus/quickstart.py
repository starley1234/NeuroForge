"""Сквозной сценарий «с нуля до работающего сервиса» одной командой.

    nexus quickstart --scale small

Шаги: проверка окружения → данные (маховик) → обучение токенизатора →
обучение ядра → приёмочные тесты с гейтом → промоушен в production →
подсказка по запуску API. Все артефакты версионируются в реестре.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

SCALES: Dict[str, Dict[str, Any]] = {
    #                      детали  сетка  fem   vocab  seq  шаги  эпохи
    "nano":   dict(samples=8,   grid=14, fem=8,  vocab=1024, seq_len=128, max_steps=8,   preset="tiny"),
    "small":  dict(samples=48,  grid=18, fem=12, vocab=2048, seq_len=256, max_steps=60,  preset="tiny"),
    "medium": dict(samples=256, grid=24, fem=16, vocab=4096, seq_len=512, max_steps=400, preset="tiny"),
    "gpu":    dict(samples=2000, grid=32, fem=24, vocab=16384, seq_len=1024, max_steps=4000,
                   preset="rtx5060-compact"),
}


@dataclass
class QuickstartResult:
    scale: str
    seconds: float
    steps: Dict[str, Any]
    ready: bool

    def to_dict(self) -> Dict[str, Any]:
        return {"scale": self.scale, "seconds": round(self.seconds, 1),
                "ready": self.ready, "steps": self.steps}


def _banner(step: int, total: int, title: str) -> None:
    print(f"\n── [{step}/{total}] {title} " + "─" * max(0, 58 - len(title)), flush=True)


def run_quickstart(
    scale: str = "small",
    model_name: str = "core",
    workdir: str = "artifacts",
    device: str = "auto",
    skip_data: bool = False,
    serve_after: bool = False,
    port: int = 8000,
) -> QuickstartResult:
    from .runtime import describe, gpu_report, pick_device
    device = pick_device(device)
    cfg = dict(SCALES[scale])
    if device.startswith("cuda") and scale in ("nano", "small"):
        print(f"[nexus] обнаружен GPU ({describe(device)}). "
              f"Для полноценного прогона используйте --scale gpu")
    t0 = time.time()
    steps: Dict[str, Any] = {}
    total = 6

    data_dir = os.path.join(workdir, "flywheel")
    tok_path = os.path.join(workdir, "tokenizer", "bpe.json")
    registry_root = os.path.join(workdir, "registry")
    os.makedirs(workdir, exist_ok=True)

    # 1 ── окружение -------------------------------------------------------
    _banner(1, total, "Проверка окружения")
    import torch

    from .config import NexusConfig
    from .eval.vram import estimate
    from .fem.calculix import available as ccx
    from .scad.render import openscad_binary

    gpu = gpu_report()
    env = {
        "torch": torch.__version__,
        "cuda": torch.cuda.is_available(),
        "gpu": gpu.get("gpu"),
        "device": device,
        "openscad": bool(openscad_binary()),
        "calculix": ccx(),
        "vram_estimate_gb": estimate(NexusConfig.rtx5060()).total_gb,
    }
    steps["environment"] = env
    print(f"   torch {env['torch']}, CUDA: {env['cuda']}, устройство: {describe(device)}")
    if gpu.get("problem"):
        print(f"   !  {gpu['problem']}\n      починка: {gpu['fix']}")
    print(f"   OpenSCAD: {'есть' if env['openscad'] else 'нет (встроенный CSG)'}, "
          f"CalculiX: {'есть' if env['calculix'] else 'нет (встроенный МКЭ)'}")

    # 2 ── данные ----------------------------------------------------------
    _banner(2, total, f"Маховик данных: {cfg['samples']} деталей → 3D → FEM")
    if skip_data and os.path.exists(os.path.join(data_dir, "dataset.jsonl")):
        print("   пропущено (данные уже есть)")
        steps["flywheel"] = {"skipped": True}
    else:
        from .data.flywheel import run as flywheel_run
        stats = flywheel_run(cfg["samples"], data_dir, seed=0, grid=cfg["grid"],
                             fem_grid=cfg["fem"], verbose=False)
        steps["flywheel"] = stats.to_dict()
        print(f"   {stats.to_dict()}")

    # 3 ── токенизатор -----------------------------------------------------
    _banner(3, total, f"Обучение BPE-токенизатора (vocab {cfg['vocab']})")
    from .data.bpe import train_from_source
    _, tok_stats = train_from_source(f"flywheel:{data_dir}", vocab_size=cfg["vocab"],
                                     out=tok_path, verbose=False)
    steps["tokenizer"] = tok_stats
    print(f"   {tok_stats['merges']:.0f} мёржей, сжатие "
          f"{tok_stats['compression_bytes_per_token']:.2f} байт/токен → {tok_path}")

    # 4 ── обучение --------------------------------------------------------
    _banner(4, total, f"Обучение ядра ({cfg['max_steps']} шагов, пресет {cfg['preset']})")
    from .training.train_lm import train as train_lm
    out = train_lm(f"flywheel:{data_dir}", model_name, cfg["preset"], seq_len=cfg["seq_len"],
                   registry_root=registry_root, tokenizer_path=tok_path,
                   max_steps=cfg["max_steps"], batch_size=2, grad_accum=4,
                   device=device, amp=device.startswith("cuda"),
                   log_every=max(1, cfg["max_steps"] // 5))
    steps["training"] = {"metrics": out["metrics"], "version": out["version"]["version"]}
    print(f"   версия {out['version']['name']}:v{out['version']['version']:04d}, "
          f"val_loss={out['metrics'].get('val_loss')}")

    # 5 ── приёмка ---------------------------------------------------------
    _banner(5, total, "Приёмочные тесты и промоушен")
    from .eval.suite import SuiteThresholds, evaluate_version
    # порог: модель должна быть лучше равномерного угадывания (ppl < размер словаря)
    thresholds = SuiteThresholds(max_val_ppl=float(cfg["vocab"]), min_unique_ratio=0.02)
    report = evaluate_version(model_name, "latest", None, registry_root,
                              f"flywheel:{data_dir}", seq_len=128,
                              thresholds=thresholds, gate=True, device=device)
    steps["evaluation"] = report.to_dict()
    print(report.summary())

    # 6 ── итог ------------------------------------------------------------
    _banner(6, total, "Готово")
    from .registry import ModelRegistry
    reg = ModelRegistry(registry_root)
    tags = reg.tags(model_name)
    steps["registry"] = {"versions": reg.versions(model_name), "tags": tags}
    ready = "production" in tags
    print(f"   реестр: {registry_root}, версии {reg.versions(model_name)}, теги {tags}")
    print("\n   Запуск API:      nexus serve --registry %s --tokenizer %s" % (registry_root, tok_path))
    print("   Проверка:        curl -s localhost:%d/health" % port)
    print("   Анализ детали:   nexus analyze examples/scad/l_bracket.scad --force 0 0 -400")

    result = QuickstartResult(scale, time.time() - t0, steps, ready)
    report_path = os.path.join(workdir, "quickstart.json")
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(result.to_dict(), fh, indent=2, ensure_ascii=False, default=str)
    print(f"\n   Отчёт: {report_path}  (всего {result.seconds:.0f} с)")

    if serve_after:
        from .serve.app import serve
        serve("0.0.0.0", port, registry_root=registry_root, model=model_name,
              ref="production" if ready else "latest", device=device,
              preset=cfg["preset"], tokenizer=tok_path)
    return result
