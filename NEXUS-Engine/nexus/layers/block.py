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
        self.memory = FastWeightMemory(cfg.d_latent, cfg.ttt)
        self.attention = SlidingWindowAttention(cfg.d_latent, cfg.attention)
        self.moe = SparseMoE(cfg.d_latent, cfg.moe)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[BlockState] = None,
        use_state: bool = False,
    ) -> Tuple[torch.Tensor, Optional[BlockState], MoEStats]:
        st = state or BlockState()
        mem_out, mem_state = self.memory(x, st.ttt, return_state=use_state)
        x = x + mem_out
        attn_out, kv = self.attention(x, st.kv, return_cache=use_state)
        x = x + attn_out
        moe_out, stats = self.moe(x)
        x = x + moe_out
        new_state = BlockState(mem_state, kv) if use_state else None
        return x, new_state, stats


def total_state_bytes(states: List[BlockState]) -> int:
    return sum(s.bytes for s in states if s is not None)
