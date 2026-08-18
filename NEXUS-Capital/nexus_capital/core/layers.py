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


def xval_number_embedding(values: torch.Tensor,
                          num_embedding: torch.Tensor) -> torch.Tensor:
    """
    xVal-стиль внедрения вещественного числа (Golkar et al., 2023):
    один обучаемый вектор u умножается на нормализованное число x.
        h_num = x * u
    Это сохраняет алгебраическую структуру: 2·u ≠ 1·u, в отличие от
    дискретной токенизации. Значения пропускаются через sign·log1p
    для устойчивости к масштабу и знаку (прибыль/убыток), что важно
    для денежных величин с тяжёлыми хвостами.

    values:        (...,) скаляр
    num_embedding: (d_out,) обучаемый «числовой» вектор
    return:        (..., d_out)
    """
    x = torch.sign(values) * torch.log1p(values.abs())
    return x.unsqueeze(-1) * num_embedding


# Совместимость со старым именем
continuous_value_embedding = None


class XValNumberEmbedding(nn.Module):
    """
    Обучаемый xVal-эмбеддинг для чисел: один вектор на специальный токен
    <num>, который масштабируется вещественным значением.
    """

    def __init__(self, d_out: int):
        super().__init__()
        self.embedding = nn.Parameter(torch.randn(d_out) * 0.02)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return xval_number_embedding(values, self.embedding)


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
