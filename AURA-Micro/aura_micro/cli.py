from __future__ import annotations

import argparse
import json

import torch

from aura_micro.config import CLASS_NAMES
from aura_micro.download import download_esc50, esc50_ready, list_esc50_items
from aura_micro.engine import load_pipeline, train
from aura_micro.export import memory_report
from aura_micro.model import AuraMicro
from aura_micro.paths import CKPT_PATH
from aura_micro.serve import run_server
from aura_micro.synth import render_scene


def _cmd_download(_: argparse.Namespace) -> None:
    root = download_esc50()
    items = list_esc50_items(root)
    print(f"mapped clips: {len(items)}")


def _cmd_train(ns: argparse.Namespace) -> None:
    if ns.source in ("auto", "esc50") and not esc50_ready() and ns.download:
        download_esc50()
    train(steps=ns.steps, batch=ns.batch, lr=ns.lr, source=ns.source, out=ns.out)


def _cmd_infer(ns: argparse.Namespace) -> None:
    pipe = load_pipeline(ns.ckpt)
    wav, lab = render_scene(pipe.cfg, 0.4, class_id=ns.class_id)
    with torch.no_grad():
        out = pipe(torch.from_numpy(wav).unsqueeze(0))
    print("truth", CLASS_NAMES[lab.class_id], f"r={lab.range_m:.1f}m")
    print(json.dumps(pipe.decode(out)[0], indent=2, ensure_ascii=False))


def _cmd_serve(ns: argparse.Namespace) -> None:
    run_server(host=ns.host, port=ns.port, ckpt=ns.ckpt)


def _cmd_info(_: argparse.Namespace) -> None:
    net = AuraMicro()
    print(json.dumps({"ready_esc50": esc50_ready(), "ckpt": CKPT_PATH.is_file(), **memory_report(net)}, indent=2))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="aura-micro", description="AURA-Micro wearable 3-mic array")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("download", help="Download ESC-50 environmental sounds")
    d.set_defaults(func=_cmd_download)

    t = sub.add_parser("train", help="Train (synth and/or ESC-50)")
    t.add_argument("--steps", type=int, default=60)
    t.add_argument("--batch", type=int, default=8)
    t.add_argument("--lr", type=float, default=2e-3)
    t.add_argument("--source", choices=("auto", "synth", "esc50"), default="auto")
    t.add_argument("--out", default=str(CKPT_PATH))
    t.add_argument("--download", action="store_true", help="fetch ESC-50 if missing")
    t.set_defaults(func=_cmd_train)

    i = sub.add_parser("infer", help="One-shot scene test")
    i.add_argument("--ckpt", default=None)
    i.add_argument("--class-id", dest="class_id", type=int, default=None)
    i.set_defaults(func=_cmd_infer)

    s = sub.add_parser("serve", help="Test console (browser)")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8080)
    s.add_argument("--ckpt", default=None)
    s.set_defaults(func=_cmd_serve)

    n = sub.add_parser("info", help="Budget / dataset status")
    n.set_defaults(func=_cmd_info)
    return p


def main(argv: list[str] | None = None) -> None:
    ns = build_parser().parse_args(argv)
    ns.func(ns)
