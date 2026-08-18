"""
Local Financial Attention — точный контекст последних W=2048 токенов новостей
и финансовых отчётов. Использует скользящее окно (sliding window) с
относительным позиционным смещением и лёгкой глобальной токен-диагональю
для важных экономических величин (CF, sigma, discount factor).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import RMSNorm


class RelativePositionBias(nn.Module):
    def __init__(self, n_heads: int, window: int):
        super().__init__()
        self.n_heads = n_heads
        self.window = window
        self.bias = nn.Parameter(torch.zeros(n_heads, 2 * window - 1))

    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        positions = torch.arange(seq_len, device=device)
        rel = positions[None, :] - positions[:, None]      # (T, T)
        rel = rel.clamp(-(self.window - 1), self.window - 1) + (self.window - 1)
        return self.bias[:, rel].permute(1, 0, 2)          # (T, H, T)


class LocalFinancialAttention(nn.Module):
    """
    Слайдинговое мультихед-внимание с окном `window` и экономическими
    масштабирующими коэффициентами. Денежные/риск-векторы получают
    дополнительный bias внимания через `value_gate`.
    """

    def __init__(self, d_value: int, n_heads: int, window: int = 2048,
                 dropout: float = 0.0):
        super().__init__()
        assert d_value % n_heads == 0
        self.d_value = d_value
        self.n_heads = n_heads
        self.head_dim = d_value // n_heads
        self.window = window
        self.dropout = dropout

        self.qkv = nn.Linear(d_value, 3 * d_value, bias=False)
        self.out = nn.Linear(d_value, d_value, bias=False)
        self.rel_bias = RelativePositionBias(n_heads, min(window, 512))
        self.value_gate = nn.Sequential(
            nn.Linear(d_value, n_heads),
            nn.Sigmoid(),
        )
        self.norm = RMSNorm(d_value)

    def forward(self, x: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)                   # (B,T,H,Dh)
        q = q.transpose(1, 2)                          # (B,H,T,Dh)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale       # (B,H,T,T)

        # Относительное позиционное смещение (T,H,T) -> (1,H,T,T)
        rel = self.rel_bias(T, x.device).permute(1, 0, 2).unsqueeze(0)
        attn = attn + rel

        # Скользящее окно локальности
        if T > self.window:
            causal = torch.ones(T, T, dtype=torch.bool, device=x.device)
            causal = torch.tril(causal, diagonal=0) & torch.triu(
                torch.ones_like(causal), diagonal=-(self.window - 1))
            attn = attn.masked_fill(~causal.unsqueeze(0).unsqueeze(0),
                                    float("-inf"))
        else:
            causal = torch.tril(torch.ones(T, T, dtype=torch.bool,
                                           device=x.device))
            attn = attn.masked_fill(~causal, float("-inf"))

        # Экономический gate: важные экономические векторы усиливаются
        gate = self.value_gate(x)                      # (B,T,H)
        attn = attn + gate.permute(0, 2, 1).unsqueeze(-1) * 0.1

        attn = F.softmax(attn, dim=-1)
        if self.training and self.dropout > 0:
            attn = F.dropout(attn, p=self.dropout)
        out = attn @ v                                 # (B,H,T,Dh)
        out = out.transpose(1, 2).reshape(B, T, D)
        return self.out(self.norm(out))
