"""NEXUS Deep Layer Block: TTT → Sliding Attention → Sparse MoE."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from ..config import NexusConfig
from .moe import MoEStats, SparseMoE
from .sliding_attention import SlidingWindowAttention, WindowKVCache
from .ttt import FastWeightMemory, TTTState


@dataclass
class BlockState:
    ttt: Optional[TTTState] = None
    kv: Optional[WindowKVCache] = None

    @property
    def bytes(self) -> int:
        n = self.ttt.bytes if self.ttt else 0
        if self.kv is not None:
            n += (self.kv.k.numel() + self.kv.v.numel()) * self.kv.k.element_size()
        return n


class NexusBlock(nn.Module):
    def __init__(self, cfg: NexusConfig):
        super().__init__()
        self.use_ttt = cfg.use_ttt
        self.use_attention = cfg.use_attention
        self.use_moe = cfg.use_moe
        self.memory = FastWeightMemory(cfg.d_latent, cfg.ttt) if cfg.use_ttt else None
        self.attention = (SlidingWindowAttention(cfg.d_latent, cfg.attention)
                          if cfg.use_attention else None)
        if cfg.use_moe:
            self.moe = SparseMoE(cfg.d_latent, cfg.moe)
        else:                                   # плотный FFN той же ёмкости
            from .common import RMSNorm, SwiGLU
            hidden = max(8, int(cfg.d_latent * cfg.moe.expert_hidden_mult
                                * (cfg.moe.n_active + cfg.moe.n_shared)))
            self.ffn_norm = RMSNorm(cfg.d_latent)
            self.ffn = SwiGLU(cfg.d_latent, hidden)
            self.moe = None

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[BlockState] = None,
        use_state: bool = False,
    ) -> Tuple[torch.Tensor, Optional[BlockState], MoEStats]:
        st = state or BlockState()
        mem_state = kv = None
        if self.memory is not None:
            mem_out, mem_state = self.memory(x, st.ttt, return_state=use_state)
            x = x + mem_out
        if self.attention is not None:
            attn_out, kv = self.attention(x, st.kv, return_cache=use_state)
            x = x + attn_out
        if self.moe is not None:
            moe_out, stats = self.moe(x)
        else:
            moe_out = self.ffn(self.ffn_norm(x))
            zero = x.new_zeros(())
            stats = MoEStats(zero, zero.detach(), zero.detach())
        x = x + moe_out
        new_state = BlockState(mem_state, kv) if use_state else None
        return x, new_state, stats


def total_state_bytes(states: List[BlockState]) -> int:
    return sum(s.bytes for s in states if s is not None)
