"""
Structured Tabular / AST Embedder.

Кодирует корпоративную отчётность (10-K/10-Q, РСБУ), таблицы юнит-экономики,
ERP/CRM-транзакции и складские остатки. Числа не токенизируются, а
проецируются в непрерывные тензоры стоимости, сохраняя масштаб и знаки
(margin > 0 vs убыток).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core.layers import ContinuousProjector, RMSNorm


# Стандартные поля юнит-экономики / P&L
DEFAULT_UNIT_FIELDS = [
    "revenue", "cogs", "gross_profit", "opex", "ebitda", "net_income",
    "cash_flow", "cac", "ltv", "churn", "arr", "mrr",
    "inventory", "receivables", "payables", "debt", "equity",
]


class StructuredEncoder(nn.Module):
    def __init__(self, d_value: int, n_fields: int = 64,
                 d_hidden: int = 256):
        super().__init__()
        self.n_fields = n_fields
        # Полевое эмбеддинг-вектор (имя колонки)
        self.field_emb = nn.Parameter(
            torch.randn(n_fields, d_value) * 0.02)
        # Числовой проектор: значение + его лог-знаковая трансформация
        self.value_proj = ContinuousProjector(2, d_value, use_log=False)
        self.encoder = nn.TransformerEncoderLayer(
            d_model=d_value, nhead=4, dim_feedforward=d_hidden,
            batch_first=True, activation="gelu", norm_first=True,
        )
        self.norm = RMSNorm(d_value)

    def forward(self, fields: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        fields: (B, n_fields) — числовые значения полей (0 если поле отсутствует).
        mask:   (B, n_fields) — 1 если поле валидно.
        return: (B, d_value)
        """
        B, F = fields.shape
        sign = torch.sign(fields)
        log_mag = torch.log1p(fields.abs())
        v = torch.stack([sign * log_mag, log_mag], dim=-1)  # (B,F,2)
        v_emb = self.value_proj(v)                          # (B,F,D)
        h = v_emb + self.field_emb[:F].unsqueeze(0)
        if mask is not None:
            h = h * mask.unsqueeze(-1)
        h = self.encoder(h)
        if mask is not None:
            h = h * mask.unsqueeze(-1)
            h = h.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
        else:
            h = h.mean(dim=1)
        return self.norm(h)
