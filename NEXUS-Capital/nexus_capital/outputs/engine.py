"""
Dual Output Engine — Уровень 4.

Выдаёт два типа выходов:
  1. Дискретные решения / документы:
       • инвестиционные меморандумы, аудит B2B,
       • скрипты переговоров, смарт-контракты/SQL;
  2. Непрерывные экономические векторы:
       • динамические цены в реальном времени,
       • оптимальные веса портфеля / хеджирование,
       • матрицы риска ликвидности и вероятности дефолта.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core.layers import RMSNorm
from ..utility.utility_layer import UtilityLayer


class DiscreteOutputHead(nn.Module):
    """Генеративная LM-голова для документов/решений."""

    def __init__(self, d_value: int, vocab_size: int):
        super().__init__()
        self.norm = RMSNorm(d_value)
        self.proj = nn.Linear(d_value, vocab_size, bias=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.proj(self.norm(h))


class ContinuousEconomicHead(nn.Module):
    """
    Непрерывный вывод: цены, веса портфеля, хеджирование, PD.
    """

    def __init__(self, d_value: int, n_assets: int = 16,
                 gamma: float = 1.0):
        super().__init__()
        self.utility = UtilityLayer(d_value, n_outcomes=32, gamma=gamma)
        # Цена/доходность
        self.price = nn.Sequential(
            nn.Linear(d_value, d_value), nn.SiLU(),
            nn.Linear(d_value, 1))
        # Веса портфеля (симплекс)
        self.weights = nn.Sequential(
            nn.Linear(d_value, d_value), nn.SiLU(),
            nn.Linear(d_value, n_assets))
        # Вероятность дефолта
        self.pd = nn.Sequential(
            nn.Linear(d_value, 64), nn.SiLU(),
            nn.Linear(64, 1))
        # Ликвидность/спред
        self.spread = nn.Sequential(
            nn.Linear(d_value, 64), nn.SiLU(),
            nn.Linear(64, 1))

    def forward(self, h: torch.Tensor) -> dict[str, torch.Tensor]:
        w = torch.softmax(self.weights(h), dim=-1)
        return {
            "price": self.price(h).squeeze(-1),
            "portfolio_weights": w,
            "default_prob": torch.sigmoid(self.pd(h)).squeeze(-1),
            "spread": F.softplus(self.spread(h)).squeeze(-1),
            "utility": self.utility(h),
        }


class DualOutputEngine(nn.Module):
    def __init__(self, d_value: int, vocab_size: int,
                 n_assets: int = 16, gamma: float = 1.0):
        super().__init__()
        self.discrete = DiscreteOutputHead(d_value, vocab_size)
        self.continuous = ContinuousEconomicHead(d_value, n_assets, gamma)

    def forward(self, h: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "logits": self.discrete(h),
            "economic": self.continuous(h),
        }
