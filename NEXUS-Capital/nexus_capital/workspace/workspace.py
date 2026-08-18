"""
Latent Economic Workspace — Уровень 3.

Связывает Neural SDE, латентный Монте-Карло и теорию игр в единый
контур «размышления о рисках и выгоде». Может многократно применяться
к латентному состоянию (итерации рефлексии), после чего возвращает
обогащённый латентный вектор и словарь метрик риска/полезности.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..core.layers import RMSNorm
from .neural_sde import NeuralSDE
from .monte_carlo import LatentMonteCarlo
from .game_theory import nash_bargaining, bertrand_price


class EconomicWorkspace(nn.Module):
    def __init__(self, d_value: int, sde_hidden: int = 256,
                 mc_paths: int = 10000, mc_horizon: int = 32,
                 alpha: float = 0.05, n_reflect: int = 1):
        super().__init__()
        self.d_value = d_value
        self.n_reflect = n_reflect

        self.sde = NeuralSDE(d_value, d_hidden=sde_hidden)
        self.mc = LatentMonteCarlo(
            self.sde, horizon=mc_horizon, paths=mc_paths, alpha=alpha)

        # Внедрение риск-метрик обратно в латентное пространство
        risk_dim = 5  # expected_pnl, std, var, es, pd
        self.risk_inject = nn.Sequential(
            nn.Linear(risk_dim, d_value),
            nn.SiLU(),
            nn.Linear(d_value, d_value),
        )
        self.reflect = nn.Sequential(
            nn.Linear(d_value, d_value),
            nn.SiLU(),
            nn.Linear(d_value, d_value),
        )
        self.norm = RMSNorm(d_value)

        # Внутренние проекторы для торгов/ценообразования
        self.reservation_a = nn.Linear(d_value, 1)
        self.reservation_b = nn.Linear(d_value, 1)
        self.surplus = nn.Linear(d_value, 1)
        self.elasticity = nn.Linear(d_value, 1)
        self.cost = nn.Linear(d_value, 1)

    def reflect_loop(self, h: torch.Tensor,
                     discount: float | None = None) -> tuple[torch.Tensor, dict]:
        metrics = {}
        for _ in range(self.n_reflect):
            risk = self.mc(h, discount=discount)
            risk_vec = torch.stack([
                risk["expected_pnl"], risk["std_pnl"], risk["var"],
                risk["expected_shortfall"], risk["default_prob"],
            ], dim=-1)
            h = self.norm(h + self.reflect(h) + self.risk_inject(risk_vec))
            metrics = risk
        return h, metrics

    def forward(
        self,
        h: torch.Tensor,
        mode: str = "risk",
        competitor_price: torch.Tensor | None = None,
        market_size: torch.Tensor | None = None,
        discount: float | None = None,
    ) -> dict:
        """
        mode:
          'risk'        — Монте-Карло по риску
          'pricing'     — оптимальная цена Бертрана
          'negotiation' — Nash bargaining (деление излишка)
        """
        h, risk = self.reflect_loop(h, discount=discount)

        out = {"latent": h, "risk": risk}

        if mode == "pricing" and competitor_price is not None:
            cost = F_softplus(self.cost(h)).squeeze(-1)
            elasticity = torch.exp(self.elasticity(h)).squeeze(-1)
            msize = (market_size if market_size is not None
                     else torch.ones_like(cost))
            opt_price = bertrand_price(cost, competitor_price, elasticity, msize)
            out["optimal_price"] = opt_price
            out["cost"] = cost
            out["demand"] = msize * torch.sigmoid(
                elasticity * (competitor_price - opt_price))

        elif mode == "negotiation":
            res_a = self.reservation_a(h).squeeze(-1)
            res_b = self.reservation_b(h).squeeze(-1)
            surplus = F_softplus(self.surplus(h)).squeeze(-1)
            share_a, share_b = nash_bargaining(res_a, res_b, surplus)
            out["negotiation"] = {
                "reservation_a": res_a,
                "reservation_b": res_b,
                "surplus": surplus,
                "share_a": share_a,
                "share_b": share_b,
            }

        return out


def F_softplus(x):
    import torch.nn.functional as F
    return F.softplus(x)
