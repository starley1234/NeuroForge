"""
Multi-Head Latent Attention (MLA) — из DeepSeek-V2/V3.

Вместо хранения полных K,V на каждую голову, сжимаем их в
низкоразмерное латентное состояние c_kv (одно на токен).
Во время attention K и V восстанавливаются «на лету».

Преимущества:
- KV-кэш сжимается в ~16 раз (типичная конфигурация)
- Качество >= полного MHA благодаря низкоранговой реконструкции
- Ротационные позиционные кодировки (RoPE) отделены от сжатого контента
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from hyperion.utils import RMSNorm, RotaryEmbedding


class MultiHeadLatentAttention(nn.Module):
    """
    MLA-слой.

    dim: входная размерность
    n_heads: число голов
    kv_lora_rank: ранг сжатия KV
    q_lora_rank: ранг сжатия Q (если None, Q не сжимается)
    rope_head_dim: размерность RoPE-части Q/K
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        kv_lora_rank: int,
        q_lora_rank: int,
        rope_head_dim: int,
        max_seq_len: int = 4096,
        dropout: float = 0.0,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.rope_head_dim = rope_head_dim

        self.qk_nope_head_dim = dim // n_heads  # контентная часть Q/K
        self.v_head_dim = dim // n_heads
        self.q_head_dim = self.qk_nope_head_dim + rope_head_dim
        self.scale = 1.0 / math.sqrt(self.q_head_dim)

        # --- Q: down-projection -> norm -> up-projection ---
        self.q_down = nn.Linear(dim, q_lora_rank, bias=False)
        self.q_norm = RMSNorm(q_lora_rank, eps=norm_eps)
        self.q_up = nn.Linear(q_lora_rank, n_heads * self.q_head_dim, bias=False)

        # --- KV: единое сжатие -> up-project на K и V ---
        self.kv_down = nn.Linear(dim, kv_lora_rank + rope_head_dim, bias=False)
        self.kv_norm = RMSNorm(kv_lora_rank, eps=norm_eps)
        self.k_up = nn.Linear(kv_lora_rank, n_heads * self.qk_nope_head_dim, bias=False)
        self.v_up = nn.Linear(kv_lora_rank, n_heads * self.v_head_dim, bias=False)

        self.rope = RotaryEmbedding(rope_head_dim, max_seq_len=max_seq_len)
        self.o_proj = nn.Linear(n_heads * self.v_head_dim, dim, bias=False)
        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        kv_cache: Optional[dict] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[dict]]:
        """
        x: [B, S, dim]
        positions: [S] — абсолютные позиции токенов
        kv_cache: {"c_kv": [B, L, kv_lora_rank], "k_rope": [B, 1, L, rope_head_dim]}
        """
        B, S, _ = x.shape

        # --- Query ---
        q = self.q_norm(self.q_down(x))
        q = self.q_up(q).view(B, S, self.n_heads, self.q_head_dim)
        q_nope, q_rope = q[..., : self.qk_nope_head_dim], q[..., self.qk_nope_head_dim:]
        q_nope = q_nope.transpose(1, 2)          # [B, H, S, d_nope]
        q_rope = q_rope.transpose(1, 2)          # [B, H, S, d_rope]
        q_rope = self.rope(q_rope, positions)

        # --- KV сжатие ---
        kv = self.kv_down(x)                      # [B, S, kv_lora_rank + d_rope]
        c_kv, k_rope_raw = torch.split(kv, [self.kv_lora_rank, self.rope_head_dim], dim=-1)
        c_kv = self.kv_norm(c_kv)
        k_rope = k_rope_raw.unsqueeze(1)          # [B, 1, S, d_rope] (shared по головам)
        k_rope = self.rope(k_rope, positions)

        # --- Кэш ---
        if use_cache:
            if kv_cache is None:
                kv_cache = {}
            if "c_kv" in kv_cache:
                c_kv = torch.cat([kv_cache["c_kv"], c_kv], dim=1)
                k_rope = torch.cat([kv_cache["k_rope"], k_rope], dim=2)
            kv_cache = {"c_kv": c_kv, "k_rope": k_rope}

        # --- Восстановление K, V ---
        L = c_kv.shape[1]
        k_nope = self.k_up(c_kv).view(B, L, self.n_heads, self.qk_nope_head_dim)
        v = self.v_up(c_kv).view(B, L, self.n_heads, self.v_head_dim)
        k_nope = k_nope.transpose(1, 2)           # [B, H, L, d_nope]
        k_rope = k_rope.expand(-1, self.n_heads, -1, -1)
        v = v.transpose(1, 2)                     # [B, H, L, d_v]

        # --- Attention ---
        scores = (
            torch.matmul(q_nope, k_nope.transpose(-2, -1))
            + torch.matmul(q_rope, k_rope.transpose(-2, -1))
        ) * self.scale

        # каузальная маска
        mask = torch.triu(
            torch.ones(S, L, device=x.device, dtype=torch.bool),
            diagonal=L - S + 1,
        )
        scores = scores.masked_fill(mask, float("-inf"))
        probs = F.softmax(scores, dim=-1)
        if self.dropout > 0 and self.training:
            probs = F.dropout(probs, p=self.dropout)

        out = torch.matmul(probs, v)              # [B, H, S, d_v]
        out = out.transpose(1, 2).contiguous().view(B, S, -1)
        out = self.o_proj(out)
        return out, kv_cache
