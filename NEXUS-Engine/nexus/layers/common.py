"""Общие примитивы: RMSNorm, SwiGLU, RoPE, непрерывные эмбеддинги времени."""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden: int, bias: bool = False):
        super().__init__()
        self.w_gate = nn.Linear(dim, hidden, bias=bias)
        self.w_up = nn.Linear(dim, hidden, bias=bias)
        self.w_down = nn.Linear(hidden, dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


def build_rope_cache(seq_len: int, head_dim: int, theta: float, device, dtype):
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    pos = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(pos, inv)
    return torch.cos(freqs).to(dtype), torch.sin(freqs).to(dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (B, H, T, D). RoPE поверх последней оси."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos = cos[None, None, : x.shape[-2], :]
    sin = sin[None, None, : x.shape[-2], :]
    o1 = x1 * cos - x2 * sin
    o2 = x1 * sin + x2 * cos
    out = torch.stack((o1, o2), dim=-1).flatten(-2)
    return out.to(x.dtype)


class ContinuousTimeEmbedding(nn.Module):
    """Непрерывная ось времени t (в секундах), а не индекс токена.

    Решает «Time & Causality Blindness»: событие со звуком через 0.4 с после
    вспышки получает физически корректное смещение фазы, независимое от того,
    сколько токенов между ними попало в поток.
    """

    def __init__(self, dim: int, min_period: float = 1e-4, max_period: float = 1e4):
        super().__init__()
        assert dim % 2 == 0
        half = dim // 2
        exponents = torch.linspace(math.log(min_period), math.log(max_period), half)
        self.register_buffer("periods", torch.exp(exponents), persistent=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (..., ) в секундах -> (..., dim)."""
        ang = 2 * math.pi * t.unsqueeze(-1) / self.periods.to(t.device, t.dtype)
        emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)
        return self.proj(emb)


def count_parameters(module: nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad or not trainable_only)
