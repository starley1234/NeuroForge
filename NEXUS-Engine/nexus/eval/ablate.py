"""Абляции: что именно даёт каждый блок архитектуры.

Идея простая: обучить одинаковым бюджетом несколько вариантов модели, отличающихся
ровно одним компонентом, и сравнить перплексию, скорость и память состояния.
Это единственный честный способ проверить, что TTT-память, скользящее окно и
разреженные эксперты действительно нужны, а не просто присутствуют в схеме.

    nexus ablate --steps 300 --preset tiny

Варианты:
  full          всё включено (TTT + окно внимания + Sparse MoE)
  no_ttt        без Fast-Weights памяти — остаётся обычный трансформер с окном
  no_attention  без окна внимания — только рекуррентная память
  dense_ffn     вместо разреженных экспертов обычный FFN той же ёмкости
  no_reasoning  без латентного цикла рассуждений
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import torch

from ..config import NexusConfig
from ..data.corpora import PackedLMDataset, split_dataset
from ..layers.block import total_state_bytes
from ..model import NexusEngine
from ..runtime import pick_device
from ..training.trainer import TrainConfig, Trainer

VARIANTS: Dict[str, Dict[str, Any]] = {
    "full": {},
    "no_ttt": {"use_ttt": False},
    "no_attention": {"use_attention": False},
    "dense_ffn": {"use_moe": False},
    "no_reasoning": {"_reason": False},
}


@dataclass
class AblationResult:
    variant: str
    params: int
    val_ppl: float
    train_ce: float
    seconds: float
    tokens_per_s: float
    state_kb_4k: float
    state_kb_16k: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@torch.no_grad()
def state_bytes_at(model: NexusEngine, length: int, device: str) -> int:
    """Размер состояния после обработки `length` токенов — проверка O(1)."""
    states = None
    chunk = max(1, model.cfg.attention.window)
    processed = 0
    while processed < length:
        step = min(chunk, length - processed)
        out = model(tokens=torch.randint(4, model.cfg.vocab_size, (1, step), device=device),
                    reason=False, states=states, use_state=True)
        states = out.states  # type: ignore[attr-defined]
        processed += step
    return total_state_bytes(states or [])


def run_ablation(
    variants: Optional[List[str]] = None,
    source: str = "mix:builtin:engineering=0.5,mathgen:2000#seed=3=0.5",
    preset: str = "tiny",
    steps: int = 300,
    seq_len: int = 256,
    batch_size: int = 4,
    lr: float = 3e-4,
    device: str = "auto",
    seed: int = 0,
    workdir: str = "artifacts/ablation",
) -> List[AblationResult]:
    device = pick_device(device)
    names = variants or list(VARIANTS)
    base = {"tiny": NexusConfig.tiny, "small": NexusConfig.small,
            "rtx5060": NexusConfig.rtx5060,
            "rtx5060-compact": NexusConfig.rtx5060_compact}[preset]

    ds = PackedLMDataset(source, seq_len=seq_len, min_blocks=8)
    train_ds, val_ds = split_dataset(ds, 0.15, seed=seed)
    print(f"[ablate] корпус: {ds.stats.to_dict()}, устройство: {device}\n")

    results: List[AblationResult] = []
    for name in names:
        flags = dict(VARIANTS[name])
        reason = flags.pop("_reason", True)
        cfg = base()
        cfg.vocab_size = max(cfg.vocab_size, ds.tok.vocab_size)
        for key, value in flags.items():
            setattr(cfg, key, value)

        torch.manual_seed(seed)
        model = NexusEngine(cfg)
        tcfg = TrainConfig(max_steps=steps, batch_size=batch_size, grad_accum=1, lr=lr,
                           device=device, amp=device.startswith("cuda"),
                           eval_every=max(20, steps // 5), log_every=max(20, steps // 5),
                           reason_every=4 if reason else 0, seed=seed,
                           model_name=f"ablate-{name}", stage="ablation",
                           registry_root=workdir)
        t0 = time.time()
        metrics = Trainer(model, tcfg, train_ds, val_ds).fit()
        elapsed = time.time() - t0
        tokens = steps * batch_size * seq_len

        model.eval()
        res = AblationResult(
            variant=name,
            params=int(sum(p.numel() for p in model.parameters())),
            val_ppl=float(metrics.get("val_ppl", 0.0)),
            train_ce=float(metrics.get("train_ce", 0.0)),
            seconds=round(elapsed, 1),
            tokens_per_s=round(tokens / max(elapsed, 1e-6)),
            state_kb_4k=round(state_bytes_at(model, 4096, device) / 1024, 1),
            state_kb_16k=round(state_bytes_at(model, 16384, device) / 1024, 1),
        )
        results.append(res)
        print(f"[ablate] {name:14s} ppl={res.val_ppl:9.2f} "
              f"{res.tokens_per_s:7d} ток/с  состояние 4k/16k = "
              f"{res.state_kb_4k}/{res.state_kb_16k} КБ\n")
    return results


def format_table(results: List[AblationResult]) -> str:
    header = (f"{'вариант':14s} {'параметры':>10s} {'val_ppl':>10s} {'train_ce':>9s} "
              f"{'ток/с':>8s} {'состояние 4k':>13s} {'16k':>9s}")
    lines = [header, "-" * len(header)]
    best = min((r.val_ppl for r in results if r.val_ppl), default=0.0)
    for r in results:
        mark = "  <- лучшая" if r.val_ppl == best else ""
        lines.append(f"{r.variant:14s} {r.params:10d} {r.val_ppl:10.2f} {r.train_ce:9.3f} "
                     f"{r.tokens_per_s:8d} {r.state_kb_4k:11.1f} КБ {r.state_kb_16k:8.1f}{mark}")
    lines.append("")
    lines.append("Состояние 4k и 16k должны совпадать — это и есть O(1) память по контексту.")
    return "\n".join(lines)
