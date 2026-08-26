"""Spatio-Temporal SSM для аудио и видеопотока (класс модальностей 2).

Вместо «нарезки на патчи и спектрограммы» — непрерывный SSM по реальному Δt.
Видео обрабатывается как поток пространственных признаков с временными метками
кадров (переменный FPS допускается по построению).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from ..bus import LatentPacket
from .base import GatedDeltaSSM, ModalityEncoder, novelty_delta


class AudioSSMEncoder(ModalityEncoder):
    modality = "audio"

    def __init__(self, d_latent: int, hop: int = 160, width: int = 256, sample_rate: int = 16000):
        super().__init__(d_latent)
        self.hop = hop
        self.sample_rate = sample_rate
        self.frontend = nn.Conv1d(1, width, kernel_size=hop * 2, stride=hop, padding=hop // 2)
        self.ssm = GatedDeltaSSM(width, state=48)
        self.proj = nn.Linear(width, d_latent)

    def forward(self, waveform: torch.Tensor, t0: float = 0.0) -> LatentPacket:
        """waveform: (B, N) моно-сигнал."""
        x = self.frontend(waveform.unsqueeze(1)).transpose(1, 2)      # (B, T, W)
        b, t, _ = x.shape
        step = self.hop / self.sample_rate
        time = t0 + torch.arange(t, device=x.device, dtype=x.dtype) * step
        dt = torch.full((b, t), step, device=x.device, dtype=x.dtype)
        y = self.ssm(novelty_delta(x), dt)
        return LatentPacket(self.proj(y), self.modality, time=time.expand(b, t))


class VideoSSMEncoder(ModalityEncoder):
    modality = "video"

    def __init__(self, d_latent: int, in_ch: int = 3, width: int = 192):
        super().__init__(d_latent)
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, width // 2, 5, stride=4, padding=2), nn.SiLU(),
            nn.Conv2d(width // 2, width, 3, stride=2, padding=1), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.ssm = GatedDeltaSSM(width, state=32)
        self.proj = nn.Linear(width, d_latent)

    def forward(self, frames: torch.Tensor, timestamps: Optional[torch.Tensor] = None) -> LatentPacket:
        """frames: (B, T, C, H, W); timestamps: (B, T) сек (могут быть неравномерны)."""
        b, t = frames.shape[:2]
        feats = self.stem(frames.flatten(0, 1)).flatten(1).view(b, t, -1)
        if timestamps is None:
            timestamps = (torch.arange(t, device=frames.device, dtype=feats.dtype) / 30.0).expand(b, t)
        dt = torch.diff(timestamps, dim=1, prepend=timestamps[:, :1])
        y = self.ssm(novelty_delta(feats), dt)
        return LatentPacket(self.proj(y), self.modality, time=timestamps)
