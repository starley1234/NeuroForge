"""
Latent Monte-Carlo: запускает тысячи симуляций Neural SDE внутри
скрытого пространства и считает распределение прибылей/убытков (P&L),
Value-at-Risk, Expected Shortfall и вероятности дефолта.

Время счёта целится в ~3 мс на 10 000 путей благодаря батчированию.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .neural_sde import NeuralSDE


def cashflow_projection(h: torch.Tensor) -> torch.Tensor:
    """
    Из латентного вектора извлекается скалярная проекция денежного потока.
    Использует фиксированное направление (обучаемое снаружи через Linear).
    h: (..., d_value) -> (...) скаляр
    """
    return h[..., 0]  # первая координата после проектора CF


def value_at_risk(pnl: torch.Tensor, alpha: float = 0.05) -> torch.Tensor:
    """
    Параметрический/исторический VaR_α: α-квантиль убытков.
    pnl: (..., P)
    """
    k = max(1, int(alpha * pnl.shape[-1]))
    sorted_pnl, _ = pnl.sort(dim=-1)
    return -sorted_pnl[..., k - 1]


def expected_shortfall(pnl: torch.Tensor, alpha: float = 0.05) -> torch.Tensor:
    k = max(1, int(alpha * pnl.shape[-1]))
    sorted_pnl, _ = pnl.sort(dim=-1)
    return -sorted_pnl[..., :k].mean(dim=-1)


def default_probability(pnl: torch.Tensor,
                        threshold: float = 0.0) -> torch.Tensor:
    return (pnl < threshold).float().mean(dim=-1)


class LatentMonteCarlo(nn.Module):
    def __init__(
        self,
        sde: NeuralSDE,
        horizon: int = 32,
        paths: int = 10000,
        dt: float = 1.0 / 252,
        alpha: float = 0.05,
    ):
        super().__init__()
        self.sde = sde
        self.horizon = horizon
        self.paths = paths
        self.dt = dt
        self.alpha = alpha
        # Проектор латентного состояния в скаляр P&L
        self.cf_head = nn.Linear(sde.d_value, 1, bias=False)

    def pnl_from_state(self, h: torch.Tensor) -> torch.Tensor:
        return self.cf_head(h).squeeze(-1)

    def forward(self, x0: torch.Tensor,
                discount: float | None = None) -> dict[str, torch.Tensor]:
        """
        x0: (B, d_value)
        Возвращает словарь метрик риска.
        """
        sim = self.sde.simulate(
            x0, horizon=self.horizon, dt=self.dt,
            paths=self.paths, return_paths=False)
        final = sim["final"]                            # (B,P,D)
        pnl = self.pnl_from_state(final)                 # (B,P)

        # Дисконтирование (временная стоимость денег)
        if discount is not None:
            pnl = pnl * torch.exp(
                torch.tensor(-discount * self.horizon * self.dt,
                             device=pnl.device, dtype=pnl.dtype))

        with torch.no_grad():
            var = value_at_risk(pnl, self.alpha)
            es = expected_shortfall(pnl, self.alpha)
            pd = default_probability(pnl)

        return {
            "pnl": pnl,
            "expected_pnl": pnl.mean(dim=-1),
            "std_pnl": pnl.std(dim=-1),
            "var": var,
            "expected_shortfall": es,
            "default_prob": pd,
            # True, если SDE откалиброван по реальным возвратам — иначе
            # метрики риска нельзя использовать как реальные оценки рынка.
            "calibrated": bool(self.sde.calibrated.item()),
        }
