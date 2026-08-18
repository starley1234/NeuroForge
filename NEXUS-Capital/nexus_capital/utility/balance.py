"""
Балансовые тождества и голова реконструкции полей отчётности.

Чтобы L_balance реально действовал при обучении (а не проверял уже
поданные на вход числа), нужен предсказатель полей из латентности.
Голова FieldReconstructionHead восстанавливает лог-масштабы полей,
а balance_loss проверяет:
    gross_profit = revenue - cogs
    assets = liabilities + equity
только по ПРЕДСКАЗАННЫМ значениям.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..encoders.structured import DEFAULT_UNIT_FIELDS


def log_transform(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(x.abs())


def invert_log(y: torch.Tensor) -> torch.Tensor:
    return torch.sign(y) * (torch.expm1(y.abs()))


class FieldReconstructionHead(nn.Module):
    """
    Предсказывает лог-трансформированные значения финансовых полей из
    латентного вектора. Маска используется только в loss (отсутствующие
    поля не штрафуются).
    """

    def __init__(self, d_value: int, n_fields: int, field_names: list[str] | None = None):
        super().__init__()
        self.n_fields = n_fields
        self.field_names = field_names or DEFAULT_UNIT_FIELDS[:n_fields]
        self.net = nn.Sequential(
            nn.Linear(d_value, d_value),
            nn.SiLU(),
            nn.Linear(d_value, n_fields),
        )
        # Индексы ключевых полей для тождеств
        self._idx = {f: i for i, f in enumerate(self.field_names)}

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: (..., d_value) -> (..., n_fields) в лог-пространстве."""
        return self.net(h)

    def reconstruction_loss(self, h: torch.Tensor, target_fields: torch.Tensor,
                            mask: torch.Tensor) -> torch.Tensor:
        pred = self.forward(h)
        tgt = log_transform(target_fields)
        se = (pred - tgt) ** 2
        if mask is not None:
            return (se * mask).sum() / mask.sum().clamp(min=1.0)
        return se.mean()

    def balance_loss(self, h: torch.Tensor) -> torch.Tensor:
        """
        Штраф за нарушение балансовых тождеств по ПРЕДСКАЗАННЫМ полям.
        Возвращает скаляр.
        """
        pred_log = self.forward(h)
        loss = pred_log.new_zeros(())
        idx = self._idx
        if {"revenue", "cogs", "gross_profit"} <= idx.keys():
            rev = invert_log(pred_log[..., idx["revenue"]])
            cogs = invert_log(pred_log[..., idx["cogs"]])
            gp = invert_log(pred_log[..., idx["gross_profit"]])
            loss = loss + F.mse_loss(gp, rev - cogs)
        if {"assets", "liabilities", "equity"} <= idx.keys():
            a = invert_log(pred_log[..., idx["assets"]])
            l = invert_log(pred_log[..., idx["liabilities"]])
            e = invert_log(pred_log[..., idx["equity"]])
            loss = loss + F.mse_loss(a, l + e)
        return loss

    def predict_fields(self, h: torch.Tensor) -> torch.Tensor:
        """Возвращает поля в натуральном масштабе."""
        return invert_log(self.forward(h))
