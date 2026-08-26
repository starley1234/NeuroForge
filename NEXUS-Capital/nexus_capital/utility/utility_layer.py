"""
Встроенная экономическая функция полезности (фон Нейман–Моргенштерн).

Вместо максимизации только правдоподобия следующего токена P(w_t|w_<t>),
выходной слой оценивает полезность решения:

    U(w, Risk) = E[R(w)] - (γ/2) · Var(R(w)) - Cost(w)

где E[R] — ожидаемая денежная выгода/маржа решения w,
    γ     — коэффициент неприятия риска,
    Cost  — транзакционные издержки/себестоимость.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def expected_utility(
    returns: torch.Tensor,
    cost: torch.Tensor | float = 0.0,
    gamma: float = 1.0,
) -> torch.Tensor:
    """
    returns: (..., P) Monte-Carlo распределение исходов R(w).
    return:  (...) ожидаемая полезность.
    """
    mean = returns.mean(dim=-1)
    var = returns.var(dim=-1, unbiased=False)
    return mean - 0.5 * gamma * var - cost


def certainty_equivalent(
    returns: torch.Tensor, gamma: float = 1.0
) -> torch.Tensor:
    """Гарантированный эквивалент лотереи при CARA/CRRA-аппроксимации."""
    mean = returns.mean(dim=-1)
    var = returns.var(dim=-1, unbiased=False)
    return mean - 0.5 * gamma * var


def risk_adjusted_sharpe(
    returns: torch.Tensor, risk_free: float = 0.0
) -> torch.Tensor:
    mean = returns.mean(dim=-1) - risk_free
    std = returns.std(dim=-1).clamp(min=1e-6)
    return mean / std


class UtilityLayer(nn.Module):
    """
    Слой, переводящий латентное состояние в распределение исходов
    (через набор детерминированных точек) и считающий ожидаемую
    полезность. Может использоваться как value-head для RL/GRPO.
    """

    def __init__(self, d_value: int, n_outcomes: int = 32,
                 gamma: float = 1.0):
        super().__init__()
        self.n_outcomes = n_outcomes
        self.gamma = gamma
        self.return_head = nn.Linear(d_value, n_outcomes)
        self.cost_head = nn.Linear(d_value, 1)
        # Веса исходов (могут быть квантилями Монте-Карло)
        self.log_weights = nn.Parameter(torch.zeros(n_outcomes))

    def forward(self, h: torch.Tensor) -> dict[str, torch.Tensor]:
        returns = self.return_head(h)                    # (..., K)
        cost = F.softplus(self.cost_head(h)).squeeze(-1)  # (...)
        w = F.softmax(self.log_weights, dim=0)
        mean = (returns * w).sum(dim=-1)
        var = ((returns - mean.unsqueeze(-1)) ** 2 * w).sum(dim=-1)
        utility = mean - 0.5 * self.gamma * var - cost
        return {
            "returns": returns,
            "expected_return": mean,
            "variance": var,
            "cost": cost,
            "utility": utility,
        }
