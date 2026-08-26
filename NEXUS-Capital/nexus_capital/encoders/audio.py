"""
Audio-Wav Stress Encoder для записи звонков (Earnings Calls).

Из мел-спектрограммы извлекает маркеры скрытого стресса топ-менеджеров:
микротремор речи (вариативность частоты основного тона), паузы, темп.
Лёгкая свёрточная пирамида сжимает сигнал в латентный вектор стресса.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core.layers import RMSNorm


class AudioStressEncoder(nn.Module):
    def __init__(self, d_value: int, n_mels: int = 64,
                 d_hidden: int = 256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(5, 3), stride=(2, 1),
                      padding=(2, 1)),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=(5, 3), stride=(2, 1),
                      padding=(2, 1)),
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=(5, 3), stride=(2, 1),
                      padding=(2, 1)),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, d_value),
        )
        # Голова стресса: 0..1
        self.stress_head = nn.Sequential(
            RMSNorm(d_value),
            nn.Linear(d_value, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )

    def forward(self, mels: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        mels: (B, T, n_mels) — мел-спектрограмма, нормированная.
        """
        x = mels.unsqueeze(1)               # (B,1,T,n_mels)
        h = self.conv(x)
        emb = self.proj(h)
        stress = torch.sigmoid(self.stress_head(emb)).squeeze(-1)
        return {"embedding": emb, "stress": stress}
