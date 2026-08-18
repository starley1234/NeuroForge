"""
Скрипт обучения ядра NEXUS-Capital на синтетических данных (Фазы 1–3).

Демонстрирует цикл предобучения с функцией потерь L_total:
    L_pred - λ1·Utility + λ2·VaR + L_balance + L_aux

Запуск (CPU):
    python scripts/train.py --steps 50 --batch-size 4

На GPU добавьте --device cuda.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nexus_capital import small_config, build_model
from nexus_capital.training.trainer import NexusTrainer, TrainConfig
from nexus_capital.data.orderbook_stream import collate_book_ticks
from nexus_capital.data.edgar_parser import (
    synthetic_filing, fields_to_tensor, DEFAULT_UNIT_FIELDS,
)
from nexus_capital.data.synthetic_market import (
    generate_stress_dataset, stress_to_tensors,
)


def make_batch(cfg, batch_size: int, T: int = 16, seed: int = 0):
    rng = np.random.default_rng(seed)
    bt = collate_book_ticks(
        batch_size=batch_size, n_levels=cfg.orderbook_levels,
        T=T, n_features=cfg.tick_features, seed=seed)
    fields, masks = [], []
    for _ in range(batch_size):
        rec = synthetic_filing(rng)
        f, m = fields_to_tensor(rec, DEFAULT_UNIT_FIELDS[:cfg.tabular_features])
        fields.append(f)
        masks.append(m)
    tokens = torch.randint(0, cfg.vocab_size, (batch_size, T))
    # Целевые токены «сдвинуты» на одну позицию вправо (упрощённый LM)
    target = torch.randint(0, cfg.vocab_size, (batch_size, 1))
    return {
        "book": bt["book"], "ticks": bt["ticks"],
        "fields": torch.stack(fields), "field_mask": torch.stack(masks),
        "tokens": tokens, "target_tokens": target,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--mc-paths", type=int, default=64)
    args = ap.parse_args()

    cfg = small_config()
    cfg.mc_paths = args.mc_paths
    cfg.mc_horizon = 8

    model = build_model(cfg)
    print(f"Параметров: {model.count_parameters()/1e6:.2f}M")

    tcfg = TrainConfig(lr=args.lr, batch_size=args.batch_size,
                       steps=args.steps, device=args.device)
    trainer = NexusTrainer(model, tcfg)

    t0 = time.time()
    running = {}
    for step in range(args.steps):
        batch = make_batch(cfg, args.batch_size, seed=step)
        logs = trainer.pretrain_step(batch)
        for k, v in logs.items():
            running.setdefault(k, 0.0)
            running[k] += v
        if step % max(1, args.steps // 10) == 0 or step == args.steps - 1:
            avg = {k: v / (step + 1) for k, v in running.items()}
            line = "  ".join(f"{k}={v:.4f}" for k, v in avg.items())
            print(f"[step {step:4d}] {line}")

    dt = time.time() - t0
    print(f"\nОбучение завершено за {dt:.1f}с "
          f"({args.steps/dt:.1f} шаг/с)")


if __name__ == "__main__":
    main()
