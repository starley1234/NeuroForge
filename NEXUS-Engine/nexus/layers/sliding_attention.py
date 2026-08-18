"""Local Sliding-Window Attention (GQA + RoPE), W = 1024..2048.

Отвечает за высокую точность локального окна: TTT-память держит «бесконечный»
контекст в сжатом виде, а окно даёт точный доступ к недавним токенам.
KV-кэш ограничен W, поэтому VRAM не растёт с длиной последовательности.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import AttentionConfig
from .common import RMSNorm, apply_rope, build_rope_cache


@dataclass
class WindowKVCache:
    k: torch.Tensor  # (B, H_kv, <=W, D)
    v: torch.Tensor
    offset: int = 0

    def trim(self, window: int) -> "WindowKVCache":
        if self.k.shape[2] > window:
            self.k = self.k[:, :, -window:]
            self.v = self.v[:, :, -window:]
        return self


class SlidingWindowAttention(nn.Module):
    def __init__(self, d_model: int, cfg: AttentionConfig):
        super().__init__()
        assert d_model % cfg.n_heads == 0
        assert cfg.n_heads % cfg.n_kv_heads == 0
        self.cfg = cfg
        self.n_heads = cfg.n_heads
        self.n_kv = cfg.n_kv_heads
        self.hd = d_model // cfg.n_heads
        self.rep = cfg.n_heads // cfg.n_kv_heads

        self.norm = RMSNorm(d_model)
        self.to_q = nn.Linear(d_model, cfg.n_heads * self.hd, bias=False)
        self.to_k = nn.Linear(d_model, cfg.n_kv_heads * self.hd, bias=False)
        self.to_v = nn.Linear(d_model, cfg.n_kv_heads * self.hd, bias=False)
        self.out = nn.Linear(cfg.n_heads * self.hd, d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cache: Optional[WindowKVCache] = None,
        return_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[WindowKVCache]]:
        b, t, _ = x.shape
        h = self.norm(x)
        q = self.to_q(h).view(b, t, self.n_heads, self.hd).transpose(1, 2)
        k = self.to_k(h).view(b, t, self.n_kv, self.hd).transpose(1, 2)
        v = self.to_v(h).view(b, t, self.n_kv, self.hd).transpose(1, 2)

        past = cache.k.shape[2] if cache is not None else 0
        cos, sin = build_rope_cache(past + t, self.hd, self.cfg.rope_theta, x.device, x.dtype)
        q = apply_rope(q, cos[past:], sin[past:])
        k = apply_rope(k, cos[past:], sin[past:])

        if cache is not None:
            k = torch.cat([cache.k, k], dim=2)
            v = torch.cat([cache.v, v], dim=2)

        new_cache = WindowKVCache(k, v).trim(self.cfg.window) if return_cache else None

        k_full = k.repeat_interleave(self.rep, dim=1)
        v_full = v.repeat_interleave(self.rep, dim=1)

        s = k_full.shape[2]
        qi = torch.arange(s - t, s, device=x.device).unsqueeze(-1)
        ki = torch.arange(s, device=x.device).unsqueeze(0)
        allowed = (ki <= qi) & (ki > qi - self.cfg.window)
        mask = torch.where(allowed, 0.0, float("-inf")).to(q.dtype)[None, None]

        y = F.scaled_dot_product_attention(q, k_full, v_full, attn_mask=mask)
        y = y.transpose(1, 2).reshape(b, t, self.n_heads * self.hd)
        return self.out(y), new_cache
