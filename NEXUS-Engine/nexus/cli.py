"""Единая точка входа: `python -m nexus.cli <команда>`."""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List

import torch


def _preset(name: str):
    from .config import NexusConfig
    return {"tiny": NexusConfig.tiny, "small": NexusConfig.small,
            "rtx5060": NexusConfig.rtx5060,
            "rtx5060-compact": NexusConfig.rtx5060_compact}[name]()


def _cmd_info(args) -> int:
    from .config import NexusConfig
    from .eval.vram import estimate
    from .model import NexusEngine
    from .scad.render import openscad_binary
    from .fem.calculix import available as ccx_available

    cfg = _preset(args.preset)
    report = {"preset": args.preset, "config": {
        "d_latent": cfg.d_latent, "n_layers": cfg.n_layers,
        "window": cfg.attention.window, "moe": f"{cfg.moe.n_shared}+{cfg.moe.n_active}/{cfg.moe.n_experts}",
        "ttt_chunk": cfg.ttt.chunk_size,
    }, "vram_estimate": estimate(cfg).to_dict(),
        "openscad": openscad_binary() or "не найден (используется встроенный CSG-движок)",
        "calculix": "доступен" if ccx_available() else "не найден (используется load-path решатель)"}
    if args.build:
        model = NexusEngine(cfg)
        report["parameters"] = model.parameter_report()
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def _cmd_flywheel(args) -> int:
    from .data.flywheel import run
    stats = run(args.n, args.out, seed=args.seed, grid=args.grid, fem_grid=args.fem_grid,
                use_openscad=args.openscad, prefer_calculix=args.calculix,
                keep_invalid=not args.only_valid)
    print(json.dumps(stats.to_dict(), indent=2, ensure_ascii=False))
    return 0


def _cmd_analyze(args) -> int:
    from .fem.solver import solve
    from .scad.render import render
    with open(args.file, encoding="utf-8") as fh:
        code = fh.read()
    res = render(code, resolution=args.grid, material=args.material,
                 use_openscad=args.openscad, stl_path=args.stl)
    if not res.ok:
        print(json.dumps({"ok": False, "error": res.error}, ensure_ascii=False, indent=2))
        return 1
    fem = solve(res.voxels, tuple(args.force), args.fixture, args.material,
                prefer_calculix=args.calculix, backend=args.fem)
    print(json.dumps({**res.summary(), "fem": fem.to_dict()}, indent=2, ensure_ascii=False))
    return 0


def _cmd_reward(args) -> int:
    from .training.rewards import score_scad
    with open(args.file, encoding="utf-8") as fh:
        code = fh.read()
    rb = score_scad(code, tuple(args.force), args.fixture, args.material,
                    required_sf=args.safety, resolution=args.grid)
    print(json.dumps(rb.to_dict(), indent=2, ensure_ascii=False))
    return 0


def _cmd_pretrain(args) -> int:
    from .config import NexusConfig
    from .training.pretrain import train
    cfg = NexusConfig.tiny() if args.preset == "tiny" else NexusConfig.rtx5060()
    train(args.data, args.out, cfg, epochs=args.epochs, batch_size=args.batch_size,
          seq_len=args.seq_len, lr=args.lr, grad_accum=args.grad_accum,
          device=args.device, max_steps=args.max_steps, eight_bit=args.eight_bit)
    return 0


def _cmd_train_fno(args) -> int:
    from .training.train_fno import train
    train(args.data, args.out, epochs=args.epochs, batch_size=args.batch_size,
          lr=args.lr, device=args.device)
    return 0


def _cmd_rl(args) -> int:
    from .training.grpo import DEFAULT_PROMPTS, GRPOConfig, train
    train(DEFAULT_PROMPTS, args.checkpoint, args.out, args.steps,
          GRPOConfig(group_size=args.group_size, max_new_tokens=args.max_new_tokens),
          device=args.device)
    return 0


def _cmd_needle(args) -> int:
    from .config import NexusConfig
    from .eval.needle import run
    cfg = NexusConfig.tiny() if args.preset == "tiny" else NexusConfig.rtx5060()
    res = run([int(x) for x in args.lengths.split(",")], cfg, args.device)
    print(json.dumps([r.to_dict() for r in res], indent=2, ensure_ascii=False))
    return 0


def _cmd_vram(args) -> int:
    from .config import NexusConfig
    from .eval.vram import estimate
    cfg = NexusConfig.tiny() if args.preset == "tiny" else NexusConfig.rtx5060()
    print(json.dumps(estimate(cfg, args.batch_size, args.chunk).to_dict(),
                     indent=2, ensure_ascii=False))
    return 0


def _cmd_quickstart(args) -> int:
    from .quickstart import run_quickstart
    res = run_quickstart(args.scale, args.model_name, args.workdir, args.device,
                         skip_data=args.skip_data, serve_after=args.serve, port=args.port,
                         samples=args.samples, steps=args.steps, vocab=args.vocab,
                         seq_len=args.seq_len, preset=args.preset,
                         math_samples=args.math)
    return 0 if res.ready else 1


def _cmd_tokenizer(args) -> int:
    from .data.bpe import train_from_source
    _, stats = train_from_source(args.source, args.vocab_size, args.out,
                                 limit=args.limit, max_chars=args.max_chars)
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    return 0


def _cmd_doctor(args) -> int:
    """Диагностика окружения: что установлено, что работает, чего не хватает."""
    import platform
    import shutil
    import torch

    from .config import NexusConfig
    from .eval.vram import estimate
    from .fem.calculix import available as ccx
    from .runtime import gpu_report
    from .scad.render import openscad_binary

    gpu = gpu_report()
    checks = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda_build": gpu.get("torch_cuda_build"),
        "cuda_available": torch.cuda.is_available(),
        "gpu": gpu.get("gpu") or (gpu.get("nvidia_smi") or {}).get("name"),
        "gpu_driver": (gpu.get("nvidia_smi") or {}).get("driver"),
        "gpu_capability": gpu.get("capability"),
        "gpu_vram_gb": gpu.get("vram_gb"),
        "torch_arch_list": gpu.get("arch_list"),
        "selected_device": gpu.get("device"),
        "cpu_threads": torch.get_num_threads(),
        "openscad": openscad_binary() or "нет (используется встроенный CSG-движок)",
        "calculix": "есть" if ccx() else "нет (используется встроенный МКЭ на гексаэдрах)",
        "bitsandbytes": bool(shutil.which("python") and _module_available("bitsandbytes")),
        "transformers": _module_available("transformers"),
        "datasets": _module_available("datasets"),
        "vram_budget_rtx5060_gb": estimate(NexusConfig.rtx5060()).total_gb,
    }
    # быстрый функциональный тест
    try:
        from .scad.render import render
        from .fem.solver import solve
        res = render("cube([20,20,20],center=true);", resolution=12)
        fem = solve(res.voxels, (0, 0, -100.0), backend="hex")
        checks["selftest"] = {"ok": res.ok, "fem_backend": fem.backend,
                              "sigma_max_mpa": round(fem.max_stress_pa / 1e6, 3)}
    except Exception as exc:
        checks["selftest"] = {"ok": False, "error": str(exc)}
    print(json.dumps(checks, indent=2, ensure_ascii=False))

    if gpu.get("problem"):
        print("\n!  " + gpu["problem"])
        print("   Починка:  " + str(gpu.get("fix")))
        print("   Или одной командой:  .\\nexus.ps1 gpu   (Windows)  |  "
              "./nexus.sh gpu   (Linux)")
    elif checks["cuda_available"]:
        print(f"\nOK GPU готов: {checks['gpu']} ({checks['gpu_vram_gb']} ГБ, "
              f"{checks['gpu_capability']}). Обучение: --scale gpu --device cuda")
    return 0


def _module_available(name: str) -> bool:
    import importlib.util
    return importlib.util.find_spec(name) is not None


def _cmd_datasets(args) -> int:
    from .data.catalog import CATALOG, MIXES, filter_catalog, mix_source, summary
    if args.action == "list":
        rows = filter_catalog(args.task, args.commercial_only, args.max_priority)
        for e in rows:
            flag = {True: "коммерч. ок", False: "только research", None: "см. лицензию"}[e.commercial_ok]
            print(f"[{e.priority}] {e.key:24s} {e.task:9s} {e.size:32s} {e.license:28s} {flag}")
            print(f"     источник: {e.source}")
            print(f"     {e.notes}\n")
    elif args.action == "mixes":
        for name in MIXES:
            print(f"{name}: {mix_source(name)}\n")
    elif args.action == "json":
        print(json.dumps(summary(), indent=2, ensure_ascii=False))
    elif args.action == "check":
        try:
            import datasets  # noqa: F401
            print("пакет datasets доступен — источники hf: будут работать")
        except ImportError:
            print("нет пакета datasets: pip install datasets (нужен для hf:-источников)")
        for e in CATALOG:
            if e.source.startswith(("dir:", "file:")):
                path = e.source.split(":", 1)[1]
                print(f"{'есть' if os.path.exists(path) else 'нет '}  {path}  ({e.key})")
    return 0


def _cmd_gen_math(args) -> int:
    from .data.mathgen import GENERATORS, write_jsonl
    if args.list_kinds:
        print("\n".join(sorted(GENERATORS)))
        return 0
    stats = write_jsonl(args.n, args.out, args.seed,
                        args.kinds.split(",") if args.kinds else None)
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    return 0


def _cmd_collect(args) -> int:
    from .data.collect import collect, default_tasks, make_designer
    kwargs = {"seed": args.seed, "device": args.device, "name": args.model_name,
              "ref": args.ref, "registry_root": args.registry}
    if args.teacher == "command":
        if not args.command:
            print("для --teacher command нужен --command", file=sys.stderr)
            return 2
        kwargs["command"] = args.command.split()
    if args.teacher == "hf":
        kwargs["model_id"] = args.hf_model
    designer = make_designer(args.teacher, **kwargs)
    stats = collect(designer, default_tasks(args.n, args.seed), args.n, args.attempts,
                    args.threshold, args.out, seed=args.seed, grid=args.grid)
    print(json.dumps(stats.to_dict(), indent=2, ensure_ascii=False))
    return 0


def _cmd_mcp(args) -> int:
    from .mcp.server import MCPServer
    server = MCPServer(args.workdir)
    if args.list_tools:
        print(json.dumps([t.spec() for t in server.tools.values()], indent=2,
                         ensure_ascii=False))
        return 0
    server.serve_stdio()
    return 0


def _cmd_demo(args) -> int:
    from .demo import run_demo
    run_demo(out_dir=args.out, n_samples=args.n)
    return 0


def _cmd_train_lm(args) -> int:
    from .training.train_lm import train
    out = train(args.source, args.model_name, args.preset, args.resume, args.seq_len,
                args.limit, val_source=args.val_source, tokenizer_path=args.tokenizer,
                registry_root=args.registry, promote=args.promote,
                keep_best=not args.no_keep_best,
                epochs=args.epochs, batch_size=args.batch_size, grad_accum=args.grad_accum,
                lr=args.lr, max_steps=args.max_steps, eval_every=args.eval_every,
                patience=args.patience, device=args.device, amp=args.amp,
                amp_dtype=args.amp_dtype, eight_bit=args.eight_bit)
    print(json.dumps(out, indent=2, ensure_ascii=False, default=str))
    return 0


def _cmd_distill(args) -> int:
    from .training.distill import DistillConfig, HFTeacher, NexusTeacher, distill
    teacher = None
    tokenizer = None
    if args.teacher:
        t = HFTeacher(args.teacher, device=args.device)
        teacher, tokenizer = t, t.adapter
    elif args.teacher_nexus:
        teacher = NexusTeacher.from_registry(args.teacher_nexus, args.teacher_ref,
                                             args.registry, args.device)
    elif args.mode != "sequence":
        print("нужен --teacher (HF) или --teacher-nexus (реестр)", file=sys.stderr)
        return 2
    out = distill(teacher, args.source, args.model_name, args.preset,
                  DistillConfig(args.temperature, args.alpha, args.top_k, args.mode),
                  resume=args.resume, seq_len=args.seq_len, cache_path=args.cache,
                  registry_root=args.registry, tokenizer=tokenizer, promote=args.promote,
                  epochs=args.epochs, batch_size=args.batch_size,
                  grad_accum=args.grad_accum, lr=args.lr, max_steps=args.max_steps,
                  device=args.device)
    print(json.dumps(out, indent=2, ensure_ascii=False, default=str))
    return 0


def _cmd_eval(args) -> int:
    from .eval.suite import SuiteThresholds, evaluate_version
    th = SuiteThresholds(max_val_ppl=args.max_ppl, min_scad_compile_rate=args.min_compile,
                         max_ppl_regression=args.max_regression,
                         max_overfit_gap=args.max_overfit)
    report = evaluate_version(args.model_name, args.ref, args.baseline, args.registry,
                              args.source, args.seq_len, th, args.gate, device=args.device)
    print(report.summary())
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(report.to_dict(), fh, indent=2, ensure_ascii=False)
    return 0 if report.passed else 1


def _fmt(value) -> str:
    return f"{value:.5g}" if isinstance(value, float) else str(value)


def _cmd_registry(args) -> int:
    from .registry import ModelRegistry
    reg = ModelRegistry(args.registry)
    if args.action == "list":
        print(json.dumps(reg.summary(), indent=2, ensure_ascii=False, default=str))
    elif args.action == "show":
        print(json.dumps(reg.get(args.model_name, args.ref).to_dict(), indent=2,
                         ensure_ascii=False, default=str))
    elif args.action == "history":
        print(json.dumps([v.to_dict() for v in reg.history(args.model_name)], indent=2,
                         ensure_ascii=False, default=str))
    elif args.action == "promote":
        print(json.dumps(reg.promote(args.model_name, args.ref, args.tag).to_dict(),
                         indent=2, ensure_ascii=False, default=str))
    elif args.action == "rollback":
        print(json.dumps(reg.rollback(args.model_name, args.tag, args.steps).to_dict(),
                         indent=2, ensure_ascii=False, default=str))
    elif args.action == "compare":
        a_ver, b_ver = reg.get(args.model_name, args.ref), reg.get(args.model_name, args.tag)
        lower_better = ("loss", "ppl", "_ce", "gap", "error", "mse", "latency", "seconds")
        higher_better = ("rate", "accuracy", "reward", "score", "sf", "compile")
        neutral = ("steps", "best_step", "restored_best", "corpus_tokens", "epoch")
        keys = sorted(set(a_ver.metrics) | set(b_ver.metrics))
        print(f"{'метрика':22s} {a_ver.tag:>14s} {b_ver.tag:>14s}   дельта")
        for k in keys:
            va, vb = a_ver.metrics.get(k), b_ver.metrics.get(k)
            delta = ""
            if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
                diff = vb - va
                if k in neutral or abs(diff) < 1e-12:
                    delta = f"{diff:+.4g}"
                elif any(t in k for t in lower_better):
                    delta = f"{diff:+.4g} ({'лучше' if diff < 0 else 'хуже'})"
                elif any(t in k for t in higher_better):
                    delta = f"{diff:+.4g} ({'лучше' if diff > 0 else 'хуже'})"
                else:
                    delta = f"{diff:+.4g}"
            print(f"{k:22s} {_fmt(va):>14s} {_fmt(vb):>14s}   {delta}")
    elif args.action == "prune":
        removed = reg.prune(args.model_name, args.keep, dry_run=args.dry_run)
        print(json.dumps({"removed": removed, "dry_run": args.dry_run}, ensure_ascii=False))
    return 0


def _cmd_serve(args) -> int:
    from .serve.app import serve
    serve(args.host, args.port, registry_root=args.registry, model=args.model_name,
          ref=args.ref, device=args.device, preset=args.preset,
          api_key=args.api_key, rate_limit=args.rate_limit,
          tokenizer=args.tokenizer, compile_model=args.compile)
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser("nexus", description="NEXUS-Engine CLI")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("info", help="конфигурация, параметры, бюджет VRAM, доступные бэкенды")
    p.add_argument("--preset", choices=["tiny", "small", "rtx5060", "rtx5060-compact"], default="rtx5060")
    p.add_argument("--build", action="store_true", help="собрать модель и посчитать параметры")
    p.set_defaults(fn=_cmd_info)

    p = sub.add_parser("flywheel", help="генерация датасета SCAD→3D→FEM")
    p.add_argument("-n", type=int, default=64)
    p.add_argument("--out", default="artifacts/flywheel")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--grid", type=int, default=24)
    p.add_argument("--fem-grid", type=int, default=16)
    p.add_argument("--openscad", action="store_true")
    p.add_argument("--calculix", action="store_true")
    p.add_argument("--only-valid", action="store_true")
    p.set_defaults(fn=_cmd_flywheel)

    p = sub.add_parser("analyze", help="анализ .scad файла: масса, аудит, FEM")
    p.add_argument("file")
    p.add_argument("--grid", type=int, default=32)
    p.add_argument("--material", default="pla")
    p.add_argument("--force", type=float, nargs=3, default=[0.0, 0.0, -200.0])
    p.add_argument("--fixture", default="base", choices=["base", "bore", "face_x"])
    p.add_argument("--stl", default=None)
    p.add_argument("--openscad", action="store_true")
    p.add_argument("--calculix", action="store_true")
    p.add_argument("--fem", choices=["auto", "hex", "loadpath", "calculix"], default="auto")
    p.set_defaults(fn=_cmd_analyze)

    p = sub.add_parser("reward", help="физическая награда для .scad (как в RL)")
    p.add_argument("file")
    p.add_argument("--grid", type=int, default=24)
    p.add_argument("--material", default="pla")
    p.add_argument("--force", type=float, nargs=3, default=[0.0, 0.0, -200.0])
    p.add_argument("--fixture", default="base")
    p.add_argument("--safety", type=float, default=2.0)
    p.set_defaults(fn=_cmd_reward)

    p = sub.add_parser("pretrain", help="фаза 3: предобучение ядра")
    p.add_argument("--data", default="artifacts/flywheel")
    p.add_argument("--out", default="artifacts/checkpoints/core.pt")
    p.add_argument("--preset", choices=["tiny", "rtx5060"], default="tiny")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--eight-bit", action="store_true")
    p.set_defaults(fn=_cmd_pretrain)

    p = sub.add_parser("train-fno", help="фаза 2: обучение FNO-критика")
    p.add_argument("--data", default="artifacts/flywheel")
    p.add_argument("--out", default="artifacts/checkpoints/fno.pt")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="auto")
    p.set_defaults(fn=_cmd_train_fno)

    p = sub.add_parser("rl", help="фаза 4: Physics-RL (GRPO)")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--out", default="artifacts/checkpoints/core_rl.pt")
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--group-size", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--device", default="auto")
    p.set_defaults(fn=_cmd_rl)

    p = sub.add_parser("needle", help="тест O(1) памяти на длинном контексте")
    p.add_argument("--lengths", default="1024,4096,16384")
    p.add_argument("--preset", choices=["tiny", "rtx5060"], default="tiny")
    p.add_argument("--device", default="auto")
    p.set_defaults(fn=_cmd_needle)

    p = sub.add_parser("vram", help="бюджет VRAM")
    p.add_argument("--preset", choices=["tiny", "rtx5060"], default="rtx5060")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--chunk", type=int, default=128)
    p.set_defaults(fn=_cmd_vram)

    p = sub.add_parser("train-lm", help="обучение на стандартном текстовом датасете")
    p.add_argument("--source", default="builtin:engineering",
                   help="builtin:engineering | dir:PATH | jsonl:PATH#text | hf:NAME:split | flywheel:PATH")
    p.add_argument("--model-name", default="core")
    p.add_argument("--preset", choices=["tiny", "small", "rtx5060", "rtx5060-compact"], default="tiny")
    p.add_argument("--resume", default=None, help="версия/тег для дообучения")
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--tokenizer", default=None, help="обученный BPE (artifacts/tokenizer/bpe.json)")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--eval-every", type=int, default=0,
                   help="как часто считать валидацию (шагов)")
    p.add_argument("--patience", type=int, default=0,
                   help="ранняя остановка: сколько проверок терпеть без улучшения")
    p.add_argument("--val-source", default=None,
                   help="отдельный корпус для валидации (честный холдаут)")
    p.add_argument("--no-keep-best", action="store_true",
                   help="сохранить последние веса вместо лучших по валидации")
    p.add_argument("--device", default="auto")
    p.add_argument("--amp", action="store_true", help="смешанная точность на CUDA")
    p.add_argument("--amp-dtype", choices=["auto", "bf16", "fp16"], default="auto",
                   help="auto = bfloat16 на поддерживающих картах (стабильнее fp16)")
    p.add_argument("--eight-bit", action="store_true")
    p.add_argument("--registry", default="artifacts/registry")
    p.add_argument("--promote", action="store_true")
    p.set_defaults(fn=_cmd_train_lm)

    p = sub.add_parser("distill", help="дистилляция из существующей LLM")
    p.add_argument("--teacher", default=None, help="HF-модель, напр. Qwen/Qwen2.5-0.5B")
    p.add_argument("--teacher-nexus", default=None, help="учитель из реестра NEXUS")
    p.add_argument("--teacher-ref", default="production")
    p.add_argument("--mode", choices=["logit", "cached", "sequence"], default="logit")
    p.add_argument("--source", default="builtin:engineering")
    p.add_argument("--model-name", default="core-distill")
    p.add_argument("--preset", choices=["tiny", "small", "rtx5060", "rtx5060-compact"], default="tiny")
    p.add_argument("--resume", default=None)
    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument("--alpha", type=float, default=0.7)
    p.add_argument("--top-k", type=int, default=64)
    p.add_argument("--cache", default="artifacts/cache/teacher_topk.npz")
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--registry", default="artifacts/registry")
    p.add_argument("--promote", action="store_true")
    p.set_defaults(fn=_cmd_distill)

    p = sub.add_parser("eval", help="приёмочные тесты версии модели (quality gate)")
    p.add_argument("--model-name", default="core")
    p.add_argument("--ref", default="latest")
    p.add_argument("--baseline", default=None, help="версия/тег для сравнения")
    p.add_argument("--source", default="builtin:engineering")
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--max-ppl", type=float, default=1e6)
    p.add_argument("--min-compile", type=float, default=0.0)
    p.add_argument("--max-regression", type=float, default=1.10)
    p.add_argument("--max-overfit", type=float, default=1.0,
                   help="допустимый разрыв val_loss − train_ce")
    p.add_argument("--gate", action="store_true", help="повысить до production при успехе")
    p.add_argument("--json", default=None, help="сохранить отчёт в файл")
    p.add_argument("--registry", default="artifacts/registry")
    p.add_argument("--device", default="auto")
    p.set_defaults(fn=_cmd_eval)

    p = sub.add_parser("registry", help="реестр моделей: версии, теги, откат")
    p.add_argument("action", choices=["list", "show", "history", "compare", "promote",
                                      "rollback", "prune"])
    p.add_argument("--model-name", default="core")
    p.add_argument("--ref", default="latest", help="версия A (для compare)")
    p.add_argument("--tag", default="production", help="тег или версия B (для compare)")
    p.add_argument("--steps", type=int, default=1)
    p.add_argument("--keep", type=int, default=5)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--registry", default="artifacts/registry")
    p.set_defaults(fn=_cmd_registry)

    p = sub.add_parser("serve", help="HTTP API (инференс + обучение фоном)")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--model-name", default="core")
    p.add_argument("--ref", default="production")
    p.add_argument("--preset", choices=["tiny", "small", "rtx5060", "rtx5060-compact"], default="tiny")
    p.add_argument("--registry", default="artifacts/registry")
    p.add_argument("--device", default="auto")
    p.add_argument("--api-key", default=None, help="или переменная окружения NEXUS_API_KEY")
    p.add_argument("--rate-limit", type=int, default=120)
    p.add_argument("--tokenizer", default=None)
    p.add_argument("--compile", action="store_true", help="torch.compile для ускорения")
    p.set_defaults(fn=_cmd_serve)

    p = sub.add_parser("quickstart", help="всё за одну команду: данные → токенизатор → обучение → приёмка")
    p.add_argument("--scale", choices=["nano", "small", "medium", "gpu", "gpu-large"],
                   default="small")
    p.add_argument("--model-name", default="core")
    p.add_argument("--workdir", default="artifacts")
    p.add_argument("--samples", type=int, default=None, help="деталей в маховике")
    p.add_argument("--steps", type=int, default=None, help="шагов обучения")
    p.add_argument("--vocab", type=int, default=None, help="размер словаря BPE")
    p.add_argument("--seq-len", type=int, default=None)
    p.add_argument("--preset", choices=["tiny", "small", "rtx5060", "rtx5060-compact"],
                   default=None, help="размер модели")
    p.add_argument("--math", type=int, default=None, help="задач инженерной математики")
    p.add_argument("--device", default="auto")
    p.add_argument("--skip-data", action="store_true", help="использовать уже готовый датасет")
    p.add_argument("--serve", action="store_true", help="сразу поднять API после обучения")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(fn=_cmd_quickstart)

    p = sub.add_parser("doctor", help="диагностика окружения и самопроверка")
    p.set_defaults(fn=_cmd_doctor)

    p = sub.add_parser("train-tokenizer", help="обучить BPE-токенизатор на корпусе")
    p.add_argument("--source", default="builtin:engineering")
    p.add_argument("--vocab-size", type=int, default=4096)
    p.add_argument("--out", default="artifacts/tokenizer/bpe.json")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--max-chars", type=int, default=2_000_000)
    p.set_defaults(fn=_cmd_tokenizer)

    p = sub.add_parser("datasets", help="каталог открытых датасетов и готовые миксы")
    p.add_argument("action", choices=["list", "mixes", "json", "check"], nargs="?",
                   default="list")
    p.add_argument("--task", default=None,
                   choices=["cad-code", "cad-brep", "sim", "math", "code", "text"])
    p.add_argument("--commercial-only", action="store_true")
    p.add_argument("--max-priority", type=int, default=3)
    p.set_defaults(fn=_cmd_datasets)

    p = sub.add_parser("gen-math", help="сгенерировать корпус инженерной математики")
    p.add_argument("-n", type=int, default=10000)
    p.add_argument("--out", default="artifacts/math/engineering_math.jsonl")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--kinds", default=None, help="через запятую; см. --list-kinds")
    p.add_argument("--list-kinds", action="store_true")
    p.set_defaults(fn=_cmd_gen_math)

    p = sub.add_parser("collect", help="сбор данных дистилляцией с физической проверкой")
    p.add_argument("--teacher", choices=["template", "command", "hf", "nexus"],
                   default="template")
    p.add_argument("--command", default=None, help="CLI учителя, напр. 'claude -p'")
    p.add_argument("--hf-model", default=None)
    p.add_argument("-n", type=int, default=16, help="сколько ТЗ обработать")
    p.add_argument("--attempts", type=int, default=3, help="попыток на задачу с фидбеком")
    p.add_argument("--threshold", type=float, default=3.5, help="порог награды для приёма")
    p.add_argument("--grid", type=int, default=20)
    p.add_argument("--out", default="artifacts/collected/dataset.jsonl")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model-name", default="core")
    p.add_argument("--ref", default="production")
    p.add_argument("--registry", default="artifacts/registry")
    p.add_argument("--device", default="auto")
    p.set_defaults(fn=_cmd_collect)

    p = sub.add_parser("mcp", help="MCP-сервер: движок как инструменты для внешней LLM")
    p.add_argument("--workdir", default="artifacts/mcp")
    p.add_argument("--list-tools", action="store_true")
    p.set_defaults(fn=_cmd_mcp)

    p = sub.add_parser("demo", help="сквозная демонстрация всех трёх уровней")
    p.add_argument("--out", default="artifacts/demo")
    p.add_argument("-n", type=int, default=12)
    p.set_defaults(fn=_cmd_demo)
    return ap


def _force_utf8_console() -> None:
    """Windows-консоль по умолчанию не UTF-8: русский текст и рамки ломают вывод."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except Exception:
            pass


def main(argv: List[str] | None = None) -> int:
    _force_utf8_console()
    args = build_parser().parse_args(argv)
    if getattr(args, "device", None) == "auto":
        from .runtime import describe, pick_device
        args.device = pick_device("auto")
        if args.device != "cpu":
            print(f"[nexus] устройство: {describe(args.device)}", flush=True)
    torch.manual_seed(0)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
