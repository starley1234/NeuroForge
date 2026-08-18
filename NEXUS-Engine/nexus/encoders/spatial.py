"""Пространство и физика (Embodied AI): 3DGS, событийные камеры, тактильность, IMU."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from ..bus import LatentPacket
from .base import GatedDeltaSSM, ModalityEncoder, novelty_delta


class PointSSMEncoder(ModalityEncoder):
    """3D Gaussian Splatting / облака точек: (x,y,z,scale,opacity,rgb)."""

    modality = "gaussian"

    def __init__(self, d_latent: int, in_dim: int = 10, width: int = 192):
        super().__init__(d_latent)
        self.point_mlp = nn.Sequential(
            nn.Linear(in_dim, width), nn.SiLU(), nn.Linear(width, width)
        )
        self.ssm = GatedDeltaSSM(width, state=32)
        self.proj = nn.Linear(width, d_latent)

    def forward(self, points: torch.Tensor, t0: float = 0.0) -> LatentPacket:
        """points: (B, P, in_dim), первые 3 канала — координаты в метрах."""
        h = self.point_mlp(points)
        b, p, _ = h.shape
        dt = torch.full((b, p), 1e-3, device=points.device, dtype=h.dtype)
        y = self.ssm(h, dt)
        pos = points[..., :3]
        time = torch.full((b, p), t0, device=points.device, dtype=h.dtype)
        return LatentPacket(self.proj(y), self.modality, time=time, position=pos)


class EventODEEncoder(ModalityEncoder):
    """DVS / событийная камера: асинхронный поток (x, y, полярность, t)."""

    modality = "event"

    def __init__(self, d_latent: int, width: int = 128):
        super().__init__(d_latent)
        self.embed = nn.Linear(3, width)
        self.ssm = GatedDeltaSSM(width, state=48)
        self.proj = nn.Linear(width, d_latent)

    def forward(self, events: torch.Tensor, timestamps: torch.Tensor) -> LatentPacket:
        """events: (B, E, 3) = (x_norm, y_norm, polarity); timestamps: (B, E) сек."""
        h = self.embed(events)
        dt = torch.diff(timestamps, dim=1, prepend=timestamps[:, :1])
        y = self.ssm(h, dt)
        pos = torch.cat([events[..., :2], torch.zeros_like(events[..., :1])], dim=-1)
        return LatentPacket(self.proj(y), self.modality, time=timestamps, position=pos)


class ProprioceptionEncoder(ModalityEncoder):
    """Тактильность (GelSight), IMU, моменты в приводах — непрерывный Neural-ODE фильтр."""

    modality = "imu"

    def __init__(self, d_latent: int, in_dim: int = 12, width: int = 128):
        super().__init__(d_latent)
        self.embed = nn.Linear(in_dim, width)
        self.ssm = GatedDeltaSSM(width, state=32)
        self.proj = nn.Linear(width, d_latent)

    def forward(self, signal: torch.Tensor, timestamps: torch.Tensor,
                force: Optional[torch.Tensor] = None,
                modality: Optional[str] = None) -> LatentPacket:
        h = self.embed(signal)
        dt = torch.diff(timestamps, dim=1, prepend=timestamps[:, :1])
        y = self.ssm(novelty_delta(h), dt)
        return LatentPacket(self.proj(y), modality or self.modality,
                            time=timestamps, force=force)


class BiophysicsEncoder(ProprioceptionEncoder):
    """ЭКГ/ЭЭГ/ЭМГ/GSR/микротремор — оценка нагрузки и усталости оператора."""

    modality = "ecg"

    def __init__(self, d_latent: int, channels: int = 8, width: int = 96):
        super().__init__(d_latent, in_dim=channels, width=width)
