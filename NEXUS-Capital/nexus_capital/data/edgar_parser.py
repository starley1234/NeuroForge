"""
Парсер финансовой отчётности (10-K/10-Q SEC EDGAR и упрощённых РСБU-таблиц).

Извлекает ключевые поля юнит-экономики и баланса в непрерывные числовые
поля без потери масштаба. Может работать с:
  • CSV/Excel таблицами;
  • словарями;
  • сырым текстом SEC-файлов (через регулярные выражения).

В offline-режиме синтетически генерирует правдоподобные отчёты для обучения.
"""
from __future__ import annotations

import re
import math
import numpy as np
import torch

from ..encoders.structured import DEFAULT_UNIT_FIELDS


# Поля, которые умеет извлекать парсер
FIELD_PATTERNS = {
    "revenue":     r"(total\s+)?revenues?|net\s+sales|выручка",
    "cogs":        r"cost\s+of\s+(goods|sales|revenue)|себестоимость",
    "gross_profit":r"gross\s+profit|валовая\s+прибыль",
    "opex":        r"(total\s+)?operating\s+expenses|операционные\s+расходы",
    "ebitda":      r"ebitda",
    "net_income":  r"net\s+income|чистая\s+прибыль",
    "cash_flow":   r"(net\s+)?cash\s+flow\s+from\s+operations|операционный\s+денежный\s+поток",
    "cac":         r"customer\s+acquisition\s+cost|cac",
    "ltv":         r"lifetime\s+value|ltv",
    "churn":       r"churn\s+rate|отток",
    "arr":         r"annual\s+recurring\s+revenue|arr",
    "mrr":         r"monthly\s+recurring\s+revenue|mrr",
    "inventory":   r"inventor(y|ies)|запасы",
    "receivables": r"accounts?\s+receivable|дебиторская",
    "payables":    r"accounts?\s+payable|кредиторская",
    "debt":        r"(total\s+)?debt|долг",
    "equity":      r"(total\s+)?(shareholders?['']?\s+)?equity|капитал",
}

# Числа в финансовом формате: '$1,234.56', '(1,234)' = убыток, '-10%'
_NUM_RE = re.compile(
    r"\(?-?\$?\d{1,3}(?:,\d{3})*(?:\.\d+)?%?\)?")


def _to_float(s: str) -> float:
    neg = s.startswith("(") and s.endswith(")")
    s = s.replace("$", "").replace(",", "").replace("(", "").replace(")", "")
    s = s.replace("%", "").strip()
    try:
        v = float(s)
    except ValueError:
        return float("nan")
    return -v if neg else v


def parse_text_filing(text: str) -> dict[str, float]:
    """Извлекает известные поля из неразмеченного текста SEC-файла."""
    out: dict[str, float] = {}
    lower = text.lower()
    for field, pat in FIELD_PATTERNS.items():
        for m in re.finditer(pat, lower):
            window = text[m.end():m.end() + 60]
            nm = _NUM_RE.search(window.replace("\n", " "))
            if nm:
                v = _to_float(nm.group(0))
                if not math.isnan(v):
                    out[field] = v
                    break
    return out


def fields_to_tensor(
    record: dict[str, float],
    field_order: list[str] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Конвертирует словарь полей в (fields, mask) тензоры для StructuredEncoder.
    """
    order = field_order or DEFAULT_UNIT_FIELDS
    vals = np.zeros(len(order), dtype=np.float32)
    mask = np.zeros(len(order), dtype=np.float32)
    for i, f in enumerate(order):
        if f in record and not math.isnan(record[f]):
            vals[i] = float(record[f])
            mask[i] = 1.0
    return torch.from_numpy(vals), torch.from_numpy(mask)


def synthetic_filing(rng: np.random.Generator | None = None,
                     profitable: bool | None = None) -> dict[str, float]:
    """
    Генерирует правдоподобный синтетический отчёт для Data Flywheel.
    Соблюдает балансовые тождества:
        gross_profit = revenue - cogs
        assets ≈ liabilities + equity
    """
    rng = rng or np.random.default_rng()
    revenue = float(rng.lognormal(mean=18, sigma=1.0))  # ~$10M-$1B
    cogs_ratio = rng.uniform(0.3, 0.85)
    cogs = revenue * cogs_ratio
    gross_profit = revenue - cogs
    opex = gross_profit * rng.uniform(0.4, 1.1)
    ebitda = gross_profit - opex
    net_income = ebitda * rng.uniform(0.5, 0.9)
    if profitable is False:
        net_income = -abs(net_income) * rng.uniform(1.0, 2.0)
        ebitda = -abs(ebitda)

    assets = revenue * rng.uniform(0.8, 3.0)
    liabilities = assets * rng.uniform(0.3, 0.8)
    equity = assets - liabilities

    return {
        "revenue": revenue,
        "cogs": cogs,
        "gross_profit": gross_profit,
        "opex": opex,
        "ebitda": ebitda,
        "net_income": net_income,
        "cash_flow": net_income * rng.uniform(0.5, 1.5),
        "cac": float(rng.lognormal(mean=5, sigma=0.8)),
        "ltv": float(rng.lognormal(mean=7, sigma=0.8)),
        "churn": rng.uniform(0.01, 0.15),
        "arr": revenue * rng.uniform(0.2, 0.9),
        "mrr": revenue / 12,
        "inventory": cogs * rng.uniform(0.05, 0.3),
        "receivables": revenue * rng.uniform(0.05, 0.25),
        "payables": cogs * rng.uniform(0.1, 0.3),
        "debt": liabilities * rng.uniform(0.2, 0.7),
        "equity": equity,
        # дополнительные балансовые поля
        "assets": assets,
        "liabilities": liabilities,
    }
