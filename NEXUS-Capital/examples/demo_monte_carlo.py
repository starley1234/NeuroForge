"""
Демо латентного Монте-Карло и Neural SDE: генерирует распределение P&L,
считает VaR/Expected Shortfall и строит гистограмму исходов.

Запуск:
    python examples/demo_monte_carlo.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time
import numpy as np
import torch

from nexus_capital import small_config
from nexus_capital.workspace.neural_sde import NeuralSDE
from nexus_capital.workspace.monte_carlo import (
    LatentMonteCarlo, value_at_risk, expected_shortfall,
)
from nexus_capital.utility.utility_layer import expected_utility


def main() -> None:
    torch.manual_seed(0)
    cfg = small_config()
    d = cfg.d_value

    sde = NeuralSDE(d, d_hidden=64)
    mc = LatentMonteCarlo(sde, horizon=cfg.mc_horizon,
                          paths=20000, alpha=cfg.risk_alpha)
    x0 = torch.randn(8, d)

    t0 = time.time()
    with torch.no_grad():
        res = mc(x0, discount=cfg.discount_rate)
    elapsed = (time.time() - t0) * 1000

    print(f"Симуляция {mc.paths} путей, горизонт {mc.horizon}: "
          f"{elapsed:.1f} мс (batch=8)")
    print(f"  E[P&L]     = {res['expected_pnl'].tolist()}")
    print(f"  σ(P&L)     = {res['std_pnl'].tolist()}")
    print(f"  VaR_5%     = {res['var'].tolist()}")
    print(f"  ES_5%      = {res['expected_shortfall'].tolist()}")
    print(f"  P(default) = {res['default_prob'].tolist()}")

    # Сравнение неприятия риска: utility при γ=0.1 vs γ=5
    u_low = expected_utility(res["pnl"], gamma=0.1)
    u_high = expected_utility(res["pnl"], gamma=5.0)
    print(f"\nUtility γ=0.1: {u_low.tolist()}")
    print(f"Utility γ=5.0: {u_high.tolist()}")
    print("(высокое неприятие риска снижает полезность неопределённых исходов)")

    # Сохраняем гистограмму, если matplotlib доступен
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        pnl = res["pnl"][0].numpy()
        ax.hist(pnl, bins=60, color="#4C72B0", alpha=0.75)
        var5 = np.percentile(pnl, 5)
        ax.axvline(var5, color="red", linestyle="--",
                   label=f"VaR 5% = {-var5:.3f}")
        ax.axvline(pnl.mean(), color="green", linestyle="-",
                   label=f"E[P&L] = {pnl.mean():.3f}")
        ax.set_title("NEXUS-Capital: латентное распределение P&L (Neural SDE)")
        ax.set_xlabel("P&L (нормир.)")
        ax.set_ylabel("Частота")
        ax.legend()
        fig.tight_layout()
        path = "examples/mc_pnl_distribution.png"
        fig.savefig(path, dpi=120)
        print(f"\nГистограмма P&L сохранена: {path}")
    except Exception as e:
        print(f"(matplotlib недоступен, график пропущен: {e})")


if __name__ == "__main__":
    main()
