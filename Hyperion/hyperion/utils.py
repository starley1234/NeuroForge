"""Базовые утилиты: нормализация, ротация, активации."""

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """RMS-нормализация (без центрирования, из Llama)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class RotaryEmbedding(nn.Module):
    """
    Rotary Positional Embedding (RoPE).

    Применяется к части размерности (d_rope), остальная часть
    остаётся без позиционной информации (аналогично DeepSeek MLA).
    """

    def __init__(self, dim: int, base: float = 10_000.0, max_seq_len: int = 4096):
        super().__init__()
        self.dim = dim
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _freqs(self, positions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        freqs = torch.einsum("i,j->ij", positions.float(), self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)  # [seq, d_rope]
        return emb.cos(), emb.sin()

    def forward(
        self, x: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        """x: [..., seq, d_rope] -> повернуть."""
        cos, sin = self._freqs(positions)
        cos = cos.unsqueeze(0).unsqueeze(0)  # [1,1,seq,d]
        sin = sin.unsqueeze(0).unsqueeze(0)
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        rotated = torch.cat([-x2, x1], dim=-1)
        return x * cos + rotated * sin


class SwiGLU(nn.Module):
    """Gated Linear Unit с SiLU (используется в современных LLM)."""

    def __init__(self, dim_in: int, dim_out: int, bias: bool = False):
        super().__init__()
        self.w = nn.Linear(dim_in, dim_out, bias=bias)
        self.v = nn.Linear(dim_in, dim_out, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(self.w(x)) * self.v(x)
