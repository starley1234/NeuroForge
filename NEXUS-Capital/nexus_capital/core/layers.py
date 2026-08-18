"""Базовые слои, общие для всех компонентов NEXUS-Capital."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(-1, keepdim=True)
        return self.weight * x * torch.rsqrt(norm + self.eps)


class SwiGLU(nn.Module):
    """SwiGLU feed-forward блок, используемый в экспертах и ядре."""

    def __init__(self, d_in: int, d_hidden: int):
        super().__init__()
        self.w_gate = nn.Linear(d_in, d_hidden, bias=False)
        self.w_up = nn.Linear(d_in, d_hidden, bias=False)
        self.w_down = nn.Linear(d_hidden, d_in, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


def continuous_value_embedding(
    values: torch.Tensor, d_out: int
) -> torch.Tensor:
    """
    Проецирует скалярные денежные величины в непрерывные тензоры стоимости
    без дискретизации в токены. Использует случайные гауссовы признаки
    (Random Fourier Features) поверх логарифма величины, что сохраняет
    метрический масштаб: $1 000 000 и $1000 различаются как разные точки
    континуума, а не как разные строки текста.

    values: (...,) тензор денежных величин (знаковые).
    return: (..., d_out)
    """
    # log1p сохраняет относительный масштаб и устойчив к нулю
    x = torch.sign(values) * torch.log1p(values.abs())
    orig_shape = x.shape
    x = x.reshape(-1, 1)  # (N, 1)

    if not hasattr(continuous_value_embedding, "_proj"):
        continuous_value_embedding._proj = {}
    key = (d_out, values.device, values.dtype)
    proj = continuous_value_embedding._proj.get(key)
    if proj is None:
        weight = torch.randn(1, d_out, device=values.device,
                             dtype=values.dtype) / (d_out ** 0.25)
        bias = 2 * torch.pi * torch.rand(d_out, device=values.device,
                                         dtype=values.dtype)
        proj = (weight, bias)
        continuous_value_embedding._proj[key] = proj
    w, b = proj
    emb = torch.cos(x @ w + b)
    return emb.reshape(*orig_shape, d_out)


class ContinuousProjector(nn.Module):
    """
    Обучаемый проектор скалярных/векторных полей в R^{d_value}.
    Заменяет токенизацию чисел: число -> непрерывный тензор стоимости.
    """

    def __init__(self, d_in: int, d_value: int, use_log: bool = True):
        super().__init__()
        self.use_log = use_log
        self.net = nn.Sequential(
            nn.Linear(d_in, d_value),
            nn.SiLU(),
            nn.Linear(d_value, d_value),
        )
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_log:
            x = torch.sign(x) * torch.log1p(x.abs())
        return self.net(x) * self.scale
