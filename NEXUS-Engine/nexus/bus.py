"""Unified Latent Bus — общая шина уровня 2.

Каждый токен любой модальности, попадая в ядро, несёт с собой физические
инварианты, а не только «семантику патча»:

    • координаты (x, y, z) в метрах — общая мировая рамка;
    • непрерывное время t (сек) — а не индекс токена;
    • масса / плотность материала;
    • вектор силы или потока (Fx, Fy, Fz);
    • идентификатор модальности.

Именно это лечит «2D-Projection Fallacy» и «Time Blindness»: звук, точка
облака, вершина B-Rep и отсчёт ЭКГ живут в одном метрическом пространстве.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn

from .layers.common import ContinuousTimeEmbedding, RMSNorm

MODALITIES: Dict[str, int] = {
    "text": 0, "code": 1, "ast": 2,
    "image": 3, "video": 4, "audio": 5,
    "scad": 6, "brep": 7, "cfd": 8, "stress": 9,
    "gaussian": 10, "event": 11, "tactile": 12, "imu": 13,
    "ecg": 14, "eeg": 15, "emg": 16, "gsr": 17,
    "thought": 18, "query": 19,
}
N_MODALITIES = 32


@dataclass
class LatentPacket:
    """Пакет уровня 1 → шина уровня 2."""
    features: torch.Tensor                    # (B, T, d_latent)
    modality: str = "text"
    time: Optional[torch.Tensor] = None       # (B, T) секунды
    position: Optional[torch.Tensor] = None   # (B, T, 3) метры
    mass: Optional[torch.Tensor] = None       # (B, T) кг
    force: Optional[torch.Tensor] = None      # (B, T, 3) Н
    meta: Dict[str, object] = field(default_factory=dict)

    @property
    def shape(self):
        return tuple(self.features.shape)


class UnifiedLatentBus(nn.Module):
    """Сливает пакеты разных модальностей в один каузальный поток латентов."""

    def __init__(self, d_latent: int, max_time: float = 1e4):
        super().__init__()
        self.d_latent = d_latent
        self.max_time = max_time
        self.time_emb = ContinuousTimeEmbedding(d_latent, max_period=max_time)
        self.modality_emb = nn.Embedding(N_MODALITIES, d_latent)
        self.invariants = nn.Linear(7, d_latent)  # xyz, mass, Fxyz
        self.norm = RMSNorm(d_latent)

    def encode_packet(self, p: LatentPacket) -> torch.Tensor:
        x = p.features
        b, t, _ = x.shape
        dev, dt = x.device, x.dtype

        time = p.time if p.time is not None else torch.zeros(b, t, device=dev, dtype=dt)
        pos = p.position if p.position is not None else torch.zeros(b, t, 3, device=dev, dtype=dt)
        mass = p.mass if p.mass is not None else torch.zeros(b, t, device=dev, dtype=dt)
        force = p.force if p.force is not None else torch.zeros(b, t, 3, device=dev, dtype=dt)

        inv = torch.cat([pos, mass.unsqueeze(-1), force], dim=-1).to(dt)
        mod_id = torch.full((b, t), MODALITIES.get(p.modality, 0), device=dev, dtype=torch.long)

        return self.norm(
            x
            + self.time_emb(time.to(dt))
            + self.invariants(inv)
            + self.modality_emb(mod_id)
        )

    def forward(self, packets: Sequence[LatentPacket], sort_by_time: bool = True) -> torch.Tensor:
        """Возвращает (B, ΣT, d_latent), упорядоченный по физическому времени."""
        assert packets, "нужен хотя бы один пакет"
        chunks: List[torch.Tensor] = []
        stamps: List[torch.Tensor] = []
        for p in packets:
            chunks.append(self.encode_packet(p))
            b, t, _ = p.features.shape
            stamps.append(
                p.time if p.time is not None
                else torch.zeros(b, t, device=p.features.device, dtype=p.features.dtype)
            )
        x = torch.cat(chunks, dim=1)
        if not sort_by_time or len(packets) == 1:
            return x
        ts = torch.cat(stamps, dim=1)
        order = torch.argsort(ts, dim=1, stable=True)
        return torch.gather(x, 1, order.unsqueeze(-1).expand_as(x))
