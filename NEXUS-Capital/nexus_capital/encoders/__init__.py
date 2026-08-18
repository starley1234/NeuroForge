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
    Сливает эмбеддинги всех доступных модальностей в единый вектор
    для Value Bus с обучаемыми весами важности (gating).
    """

    def __init__(self, d_value: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(d_value, d_value),
            nn.Sigmoid(),
        )
        self.proj = nn.Sequential(
            nn.Linear(d_value, d_value),
            nn.SiLU(),
            nn.Linear(d_value, d_value),
        )

    def forward(self, embeddings: dict[str, "torch.Tensor"]) -> torch.Tensor:
        if not embeddings:
            raise ValueError("No modalities provided")
        keys = list(embeddings.keys())
        stacked = torch.stack([embeddings[k] for k in keys], dim=1)
        gate = self.gate(stacked)
        fused = (stacked * gate).sum(dim=1) / gate.sum(dim=1).clamp(min=1e-6)
        return self.proj(fused)
