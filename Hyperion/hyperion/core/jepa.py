"""
JEPA — Joint Embedding Predictive Architecture.

Вместо предсказания дискретных токенов (cross-entropy), модель
предсказывает НЕПРЕРЫВНЫЕ ЭМБЕДДИНГИ следующих элементов
последовательности. Обучение происходит в латентном пространстве:

    L_JEPA = || predictor(z_1..t) - target_encoder(z_{t+1..t+k}) ||²

Преимущества (VL-JEPA / LLM-JEPA):
- 50% меньше обучаемых параметров при том же качестве
- Устойчивость к переобучению
- Фокус на семантике, а не на поверхностной вариативности

Здесь:
- JEPAPredictor: общий слой, предсказывающий эмбеддинги k шагов вперёд
- JEPALoss: функция потерь с экспоненциальным затуханием дальности
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from hyperion.utils import RMSNorm


class JEPAPredictor(nn.Module):
    """
    Предсказывает эмбеддинги следующих k элементов из текущего состояния.

    predictor: [B, S, dim] -> [B, S, k, pred_dim]
    """

    def __init__(
        self,
        dim: int,
        pred_dim: int = 512,
        k: int = 4,
        hidden_mult: float = 2.0,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.dim = dim
        self.pred_dim = pred_dim
        self.k = k
        hidden = int(dim * hidden_mult)

        self.norm = RMSNorm(dim, eps=norm_eps)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden, bias=False),
            nn.SiLU(),
            nn.Linear(hidden, hidden, bias=False),
            nn.SiLU(),
        )
        self.head = nn.Linear(hidden, k * pred_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, S, dim] -> [B, S, k, pred_dim]."""
        h = self.net(self.norm(x))
        out = self.head(h)
        B, S, _ = out.shape
        return out.view(B, S, self.k, self.pred_dim)


class JEPALoss(nn.Module):
    """
    Потеря в пространстве эмбеддингов.

    Считает MSE между предсказанными эмбеддингами и целевыми
    (закодированными) эмбеддингами будущих элементов, с
    экспоненциальным затуханием веса по горизонту k.
    """

    def __init__(self, decay: float = 0.9):
        super().__init__()
        self.decay = decay

    def forward(
        self,
        preds: torch.Tensor,          # [B, S, k, D]
        targets: torch.Tensor,        # [B, S, k, D]
        valid: Optional[torch.Tensor] = None,  # [B, S, k] маска
    ) -> torch.Tensor:
        # веса: decay^i для шага i
        weights = torch.tensor(
            [self.decay ** i for i in range(preds.shape[2])],
            device=preds.device, dtype=preds.dtype,
        ).view(1, 1, -1, 1)

        diff = (preds - targets.detach()).pow(2) * weights
        if valid is not None:
            diff = diff * valid.unsqueeze(-1)
            n = valid.sum().clamp(min=1)
        else:
            n = preds.numel() // preds.shape[-1]
        return diff.sum() / n
