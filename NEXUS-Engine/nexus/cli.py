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
    return {"tiny": NexusConfig.tiny, "rtx5060": NexusConfig.rtx5060,
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
                prefer_calculix=args.calculix)
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


def _cmd_demo(args) -> int:
    from .demo import run_demo
    run_demo(out_dir=args.out, n_samples=args.n)
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser("nexus", description="NEXUS-Engine CLI")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("info", help="конфигурация, параметры, бюджет VRAM, доступные бэкенды")
    p.add_argument("--preset", choices=["tiny", "rtx5060", "rtx5060-compact"], default="rtx5060")
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
    p.add_argument("--device", default="cpu")
    p.add_argument("--eight-bit", action="store_true")
    p.set_defaults(fn=_cmd_pretrain)

    p = sub.add_parser("train-fno", help="фаза 2: обучение FNO-критика")
    p.add_argument("--data", default="artifacts/flywheel")
    p.add_argument("--out", default="artifacts/checkpoints/fno.pt")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="cpu")
    p.set_defaults(fn=_cmd_train_fno)

    p = sub.add_parser("rl", help="фаза 4: Physics-RL (GRPO)")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--out", default="artifacts/checkpoints/core_rl.pt")
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--group-size", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--device", default="cpu")
    p.set_defaults(fn=_cmd_rl)

    p = sub.add_parser("needle", help="тест O(1) памяти на длинном контексте")
    p.add_argument("--lengths", default="1024,4096,16384")
    p.add_argument("--preset", choices=["tiny", "rtx5060"], default="tiny")
    p.add_argument("--device", default="cpu")
    p.set_defaults(fn=_cmd_needle)

    p = sub.add_parser("vram", help="бюджет VRAM")
    p.add_argument("--preset", choices=["tiny", "rtx5060"], default="rtx5060")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--chunk", type=int, default=128)
    p.set_defaults(fn=_cmd_vram)

    p = sub.add_parser("demo", help="сквозная демонстрация всех трёх уровней")
    p.add_argument("--out", default="artifacts/demo")
    p.add_argument("-n", type=int, default=12)
    p.set_defaults(fn=_cmd_demo)
    return ap


def main(argv: List[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    torch.manual_seed(0)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
