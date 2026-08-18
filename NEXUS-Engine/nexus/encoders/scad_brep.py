"""B-Rep / CSG Graph Neural Operator — ключевая модальность (инженерия)."""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from ..bus import LatentPacket
from ..geometry.brep import NODE_FEATURE_DIM, BRepGraph
from .base import ModalityEncoder


class BRepGNOEncoder(ModalityEncoder):
    """Message passing по CSG/B-Rep графу + инварианты масс-свойств.

    В отличие от «2D-проекций», энкодер видит топологию: что из чего вычтено,
    где перемычка, где внутренняя полость.
    """

    modality = "brep"

    def __init__(self, d_latent: int, width: int = 256, layers: int = 4):
        super().__init__(d_latent)
        self.embed = nn.Linear(NODE_FEATURE_DIM, width)
        self.messages = nn.ModuleList(
            nn.Sequential(nn.Linear(width * 2, width), nn.SiLU(), nn.Linear(width, width))
            for _ in range(layers)
        )
        self.norms = nn.ModuleList(nn.LayerNorm(width) for _ in range(layers))
        self.proj = nn.Linear(width, d_latent)
        self.mass_proj = nn.Linear(10, d_latent)   # V, m, centroid, inertia, bbox

    def encode_graphs(self, graphs: Sequence[BRepGraph], device=None):
        width = max(g.n_nodes for g in graphs)
        feats = np.zeros((len(graphs), width, NODE_FEATURE_DIM), dtype=np.float32)
        adj = np.zeros((len(graphs), width, width), dtype=np.float32)
        mask = np.zeros((len(graphs), width), dtype=np.float32)
        for b, g in enumerate(graphs):
            feats[b, : g.n_nodes] = g.nodes
            mask[b, : g.n_nodes] = 1.0
            if g.n_edges:
                adj[b, g.edges[0], g.edges[1]] = 1.0
            adj[b, np.arange(g.n_nodes), np.arange(g.n_nodes)] = 1.0
        t = lambda a: torch.as_tensor(a, device=device)
        return t(feats), t(adj), t(mask)

    def forward(self, node_features: torch.Tensor, adjacency: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                mass_vector: Optional[torch.Tensor] = None) -> LatentPacket:
        h = self.embed(node_features)
        deg = adjacency.sum(-1, keepdim=True).clamp(min=1.0)
        for mlp, norm in zip(self.messages, self.norms):
            agg = torch.bmm(adjacency, h) / deg
            h = norm(h + mlp(torch.cat([h, agg], dim=-1)))
        x = self.proj(h)
        if mask is not None:
            x = x * mask.unsqueeze(-1)
        if mass_vector is not None:
            x = x + self.mass_proj(mass_vector).unsqueeze(1)
        b, t, _ = x.shape
        time = torch.zeros(b, t, device=x.device, dtype=x.dtype)
        return LatentPacket(x, self.modality, time=time)


class FieldEncoder(ModalityEncoder):
    """Тензорные поля FEM/CFD (σ, скорость, температура) → латенты."""

    modality = "stress"

    def __init__(self, d_latent: int, in_ch: int = 1, width: int = 128):
        super().__init__(d_latent)
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, width // 2, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv3d(width // 2, width, 3, stride=2, padding=1), nn.SiLU(),
        )
        self.proj = nn.Linear(width, d_latent)

    def forward(self, field: torch.Tensor, modality: Optional[str] = None) -> LatentPacket:
        b = field.shape[0]
        h = self.net(field)
        tokens = h.flatten(2).transpose(1, 2)               # (B, T, W)
        x = self.proj(tokens)
        time = torch.zeros(b, x.shape[1], device=x.device, dtype=x.dtype)
        return LatentPacket(x, modality or self.modality, time=time)
