"""Dual Output Engine (уровень 3).

Дискретная ветвь: токены (OpenSCAD, текст, Python/C++, спецификации).
Непрерывная ветвь: траектории манипуляторов / G-код, тензорные поля FEM/CFD,
скалярные физические предсказания (масса, σ_max, коэффициент запаса).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from .config import NexusConfig
from .layers.common import RMSNorm


@dataclass
class NexusOutput:
    logits: torch.Tensor                     # (B, T, V)
    actions: Optional[torch.Tensor] = None   # (B, T, action_dim)
    field: Optional[torch.Tensor] = None     # (B, 1, G, G, G)
    physics: Optional[torch.Tensor] = None   # (B, 3): масса, σ_max, запас
    aux_loss: Optional[torch.Tensor] = None
    reasoning: Optional[object] = None


class DiscreteHead(nn.Module):
    def __init__(self, cfg: NexusConfig, embedding: Optional[nn.Embedding] = None):
        super().__init__()
        self.norm = RMSNorm(cfg.d_latent)
        self.proj = nn.Linear(cfg.d_latent, cfg.vocab_size, bias=False)
        if embedding is not None and cfg.tie_embeddings:
            self.proj.weight = embedding.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.norm(x))


class ContinuousHead(nn.Module):
    """Траектории/G-код + тензорные поля + скалярная физика."""

    def __init__(self, cfg: NexusConfig):
        super().__init__()
        g = cfg.field_grid
        self.grid = g
        self.norm = RMSNorm(cfg.d_latent)
        self.action = nn.Sequential(
            nn.Linear(cfg.d_latent, cfg.d_latent // 2), nn.SiLU(),
            nn.Linear(cfg.d_latent // 2, cfg.action_dim),
        )
        self.field_seed = nn.Linear(cfg.d_latent, 64 * (g // 4) ** 3)
        self.field_net = nn.Sequential(
            nn.ConvTranspose3d(64, 32, 4, stride=2, padding=1), nn.SiLU(),
            nn.ConvTranspose3d(32, 16, 4, stride=2, padding=1), nn.SiLU(),
            nn.Conv3d(16, 1, 3, padding=1),
        )
        self.physics = nn.Linear(cfg.d_latent, 3)

    def forward(self, x: torch.Tensor, pooled: Optional[torch.Tensor] = None):
        h = self.norm(x)
        actions = self.action(h)
        p = pooled if pooled is not None else h.mean(dim=1)
        g4 = self.grid // 4
        seed = self.field_seed(p).view(-1, 64, g4, g4, g4)
        return actions, self.field_net(seed), self.physics(p)
