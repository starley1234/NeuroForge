"""Базовые примитивы энкодеров уровня 1 (Continuous Dynamic Encoders)."""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from ..bus import LatentPacket


class ModalityEncoder(nn.Module, ABC):
    """Лёгкий (10–50M) энкодер модальности → LatentPacket на общую шину.

    Ядро замораживается при подключении новой модальности; учится только это.
    """

    modality: str = "text"

    def __init__(self, d_latent: int):
        super().__init__()
        self.d_latent = d_latent

    @abstractmethod
    def forward(self, *args, **kwargs) -> LatentPacket:  # pragma: no cover - интерфейс
        ...

    def freeze(self) -> "ModalityEncoder":
        for p in self.parameters():
            p.requires_grad_(False)
        return self


class GatedDeltaSSM(nn.Module):
    """Диагональный непрерывный SSM с шагом по РЕАЛЬНОМУ Δt (zero-order hold).

        h(t_k) = exp(A · Δt_k) · h(t_{k-1}) + Δt_k · B · x_k

    Даёт две вещи сразу: (1) физически корректные задержки между событиями,
    (2) «налог на токенизацию» падает — статичный сигнал почти не двигает
    состояние, и downstream-слои видят только дельту новизны.
    """

    def __init__(self, dim: int, state: int = 64):
        super().__init__()
        self.dim = dim
        self.state = state
        self.log_neg_a = nn.Parameter(torch.linspace(-4.0, 1.0, state).repeat(dim, 1))
        self.B = nn.Parameter(torch.randn(dim, state) * 0.02)
        self.C = nn.Parameter(torch.randn(dim, state) * 0.02)
        self.D = nn.Parameter(torch.ones(dim))
        self.gate = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        """x: (B, T, dim), dt: (B, T) в секундах → (B, T, dim)."""
        b, t, d = x.shape
        a = -torch.exp(self.log_neg_a)                      # (d, s), Re(a) < 0
        h = x.new_zeros(b, d, self.state)
        outs = []
        for k in range(t):
            step = dt[:, k].clamp(min=0).view(b, 1, 1)
            decay = torch.exp(a.unsqueeze(0) * step)
            h = decay * h + step * (x[:, k].unsqueeze(-1) * self.B.unsqueeze(0))
            outs.append((h * self.C.unsqueeze(0)).sum(-1) + self.D * x[:, k])
        y = torch.stack(outs, dim=1)
        return y * torch.sigmoid(self.gate(x))


def novelty_delta(x: torch.Tensor) -> torch.Tensor:
    """Дельта новизны: сигнал минус его предыдущий отсчёт (борьба с шумом)."""
    prev = torch.cat([x[:, :1], x[:, :-1]], dim=1)
    return x - prev
