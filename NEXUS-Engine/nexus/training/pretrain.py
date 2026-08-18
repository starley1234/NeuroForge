"""Фаза 3: предобучение ядра на коде/AST/датасете маховика."""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader

from ..config import NexusConfig
from ..data.dataset import ScadCorpus
from ..model import NexusEngine


def build_optimizer(model: torch.nn.Module, lr: float, wd: float = 0.01,
                    eight_bit: bool = False):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim < 2 else decay).append(p)
    groups = [{"params": decay, "weight_decay": wd},
              {"params": no_decay, "weight_decay": 0.0}]
    if eight_bit:
        try:  # 8-bit AdamW экономит ~2.8 ГБ на 1.3B активных весах
            import bitsandbytes as bnb
            return bnb.optim.AdamW8bit(groups, lr=lr, betas=(0.9, 0.95))
        except Exception:
            print("[warn] bitsandbytes недоступен, используется fp32 AdamW")
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95))


def cosine_lr(step: int, total: int, base: float, warmup: int = 50) -> float:
    if step < warmup:
        return base * (step + 1) / warmup
    t = (step - warmup) / max(total - warmup, 1)
    return base * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(t, 1.0))))


def train(data: str, out: str, config: Optional[NexusConfig] = None, epochs: int = 1,
          batch_size: int = 2, seq_len: int = 512, lr: float = 3e-4,
          grad_accum: int = 8, device: str = "cpu", max_steps: Optional[int] = None,
          eight_bit: bool = False, log_every: int = 10) -> Dict[str, float]:
    cfg = config or NexusConfig.tiny()
    ds = ScadCorpus(data, seq_len=seq_len)
    if len(ds) == 0:
        raise SystemExit("датасет пуст — сначала запустите flywheel")
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    model = NexusEngine(cfg).to(device)
    opt = build_optimizer(model, lr, eight_bit=eight_bit)
    total = max_steps or (len(dl) * epochs)
    step, t0, last = 0, time.time(), {}

    model.train()
    for _ in range(epochs):
        for batch in dl:
            for g in opt.param_groups:
                g["lr"] = cosine_lr(step, total, lr)
            tokens = batch["tokens"].to(device)
            stats = model.loss(tokens, batch["targets"].to(device), reason=(step % 4 == 0))
            (stats["loss"] / grad_accum).backward()
            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
            last = {k: float(v.detach()) for k, v in stats.items()}
            if step % log_every == 0:
                print(f"step {step:5d}/{total} loss={last['loss']:.4f} "
                      f"lm={last['lm']:.4f} aux={last['aux']:.4f} "
                      f"({time.time() - t0:.1f}s)", flush=True)
            step += 1
            if max_steps and step >= max_steps:
                break
        if max_steps and step >= max_steps:
            break

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    torch.save({"model": model.state_dict(), "config": cfg.to_dict()}, out)
    with open(out + ".json", "w", encoding="utf-8") as fh:
        json.dump({"steps": step, "last": last, "params": model.parameter_report()},
                  fh, indent=2, ensure_ascii=False)
    print(f"чекпойнт сохранён: {out}")
    return last


def main() -> None:
    ap = argparse.ArgumentParser(description="NEXUS pre-train")
    ap.add_argument("--data", default="artifacts/flywheel")
    ap.add_argument("--out", default="artifacts/checkpoints/core.pt")
    ap.add_argument("--preset", choices=["tiny", "rtx5060"], default="tiny")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--eight-bit", action="store_true")
    a = ap.parse_args()
    cfg = NexusConfig.tiny() if a.preset == "tiny" else NexusConfig.rtx5060()
    train(a.data, a.out, cfg, a.epochs, a.batch_size, a.seq_len, a.lr,
          a.grad_accum, a.device, a.max_steps, a.eight_bit)


if __name__ == "__main__":
    main()
