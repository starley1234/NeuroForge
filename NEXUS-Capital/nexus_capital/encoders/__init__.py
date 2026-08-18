"""Энкодеры 5 классов экономических модальностей NEXUS-Capital."""
import torch
import torch.nn as nn

from .orderbook import OrderBookEncoder
from .structured import StructuredEncoder
from .graph import GraphRiskOperator
from .audio import AudioStressEncoder
from .text import FinancialTextEncoder

__all__ = [
    "OrderBookEncoder",
    "StructuredEncoder",
    "GraphRiskOperator",
    "AudioStressEncoder",
    "FinancialTextEncoder",
    "MultimodalFusion",
]


class MultimodalFusion(nn.Module):
    """
    Сливает эмбеддинги модальностей через cross-attention поверх множества
    «токенов» модальностей. Каждая модальность даёт один вектор; они
    образуют память (B, M, D), а learnable query аккумулирует контекст.
    Это честнее усреднения: модальности имеют разный семантический масштаб
    (стакан vs новость vs скаляр стресса).
    """

    def __init__(self, d_value: int, n_heads: int = 4):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d_value) * 0.02)
        self.cross = nn.MultiheadAttention(
            d_value, n_heads, batch_first=True, bias=False)
        self.norm = nn.LayerNorm(d_value)
        self.proj = nn.Sequential(
            nn.Linear(d_value, d_value),
            nn.SiLU(),
            nn.Linear(d_value, d_value),
        )

    def forward(self, embeddings: dict[str, "torch.Tensor"]) -> torch.Tensor:
        if not embeddings:
            raise ValueError("No modalities provided")
        keys = list(embeddings.keys())
        memory = torch.stack([embeddings[k] for k in keys], dim=1)  # (B,M,D)
        B = memory.shape[0]
        q = self.query.expand(B, -1, -1)
        attn_out, _ = self.cross(q, memory, memory, need_weights=False)
        pooled = self.norm(q + attn_out).squeeze(1)
        return self.proj(pooled)
