"""
MTP — Multi-Token Prediction (из DeepSeek-V3).

Во время обучения модель предсказывает НЕСКОЛЬКО токенов вперёд
(не только следующий). Это даёт:
- Плотный градиентный сигнал (быстрее конвергенция)
- «Планирование» вперёд в представлениях
- На инференсе MTP-модули можно использовать для
  спекулятивного декодирования (acceptance ~85-90%)

Важно: MTP-модули отбрасываются при обычном инференсе,
основная модель не замедляется.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from hyperion.utils import RMSNorm, SwiGLU


class MTPModule(nn.Module):
    """
    Один MTP-слой: предсказывает токен на глубине depth.

    Формула (DeepSeek-V3):
        h' = RMSNorm(concat(Linear(h_t), Embedding(x_{t+d})))
        logits = Linear(act(Linear(h')))
    """

    def __init__(
        self,
        dim: int,
        vocab_size: int,
        depth: int,
        hidden_mult: float = 1.5,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.depth = depth
        self.dim = dim

        # Вход: [h_t; e_{t+depth}] -> dim
        self.in_norm = RMSNorm(dim, eps=norm_eps)
        self.in_proj = nn.Linear(dim + dim, dim, bias=False)
        self.ff = SwiGLU(dim, int(dim * hidden_mult))
        self.out_proj = nn.Linear(int(dim * hidden_mult), vocab_size, bias=False)

    def forward(
        self,
        h_t: torch.Tensor,        # [B, S, dim]
        emb_next: torch.Tensor,   # [B, S, dim] — эмбеддинг токена на глубине d
    ) -> torch.Tensor:
        h = self.in_norm(h_t)
        x = torch.cat([h, emb_next], dim=-1)
        x = self.in_proj(x)
        x = self.ff(x)
        return self.out_proj(x)   # [B, S, vocab]


class MTPStack(nn.Module):
    """
    Стек MTP-модулей глубиной mtp_depth.

    Используется ТОЛЬКО на обучении (или для спекулятивного
    декодирования на инференсе).
    """

    def __init__(
        self,
        dim: int,
        vocab_size: int,
        embedding: nn.Embedding,
        mtp_depth: int = 1,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.mtp_depth = mtp_depth
        self.embedding = embedding  # разделяемый с основной моделью
        self.mtp_modules = nn.ModuleList([
            MTPModule(dim, vocab_size, depth=d, norm_eps=norm_eps)
            for d in range(1, mtp_depth + 1)
        ])
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(
        self,
        h: torch.Tensor,          # [B, S, dim] — выход backbone
        input_ids: torch.Tensor,  # [B, S] — исходные токены
    ) -> torch.Tensor:
        """
        Возвращает суммарную MTP-потерю.

        Для глубины d: цель = input_ids[:, d+depth:], источник = h[:, :S-d].
        """
        B, S, _ = h.shape
        total = 0.0
        for d, mod in enumerate(self.mtp_modules, start=1):
            if S <= d + 1:
                continue
            # эмбеддинги будущих токенов (сдвиг на d): e_{t+d}
            emb_next = self.embedding(input_ids[:, d: S - 1])  # [B, S-d-1, dim]
            src = h[:, : S - d - 1]                            # [B, S-d-1, dim]
            logits = mod(src, emb_next)                        # [B, S-d-1, vocab]
            targets = input_ids[:, d + 1:]                     # [B, S-d-1]
            loss = self.loss_fn(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
            total = total + loss / self.mtp_depth
        return total
