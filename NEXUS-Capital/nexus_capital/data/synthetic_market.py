"""
Synthetic Market Engine — генератор рыночных стрессов для Data Flywheel.

Симулирует шоки:
    • инфляция +10%,
    • разрыв логистики,
    • рост ключевой ставки,
    • санкции / эмбарго,
    • демпинг конкурентов,
    • паника/эйфория сентимента.

Каждый сценарий применяется к базовому отчёту и макро-состоянию,
в результате получается обучающий кортеж:
    { рыночный контекст -> сценарий -> итог P&L / риск }.
"""
from __future__ import annotations

import numpy as np
import torch

from .edgar_parser import synthetic_filing


# Шаблоны стрессов: множители для полей отчёта
STRESS_SHOCKS = {
    "inflation_surge": {
        "cogs": 1.12, "opex": 1.08, "inventory": 1.05,
        "macro_inflation": 0.10,
    },
    "supply_chain_break": {
        "cogs": 1.25, "inventory": 0.60, "revenue": 0.85,
        "macro_shock": 1.0,
    },
    "rate_hike": {
        "debt": 1.15, "cash_flow": 0.80, "macro_rate": 0.04,
    },
    "sanctions": {
        "revenue": 0.70, "receivables": 1.30, "payables": 0.80,
        "default_prob": 0.08,
    },
    "competitor_dump": {
        "revenue": 0.82, "gross_profit": 0.75, "elasticity": -2.5,
    },
    "demand_boom": {
        "revenue": 1.25, "inventory": 0.70, "gross_profit": 1.20,
    },
    "liquidity_crisis": {
        "cash_flow": 0.50, "receivables": 1.40, "debt": 1.20,
        "spread": 0.05,
    },
}


def apply_shock(report: dict[str, float],
                shock: str | dict,
                rng: np.random.Generator | None = None) -> dict[str, float]:
    """Применяет шок к отчёту, возвращает модифицированную копию."""
    rng = rng or np.random.default_rng()
    params = STRESS_SHOCKS[shock] if isinstance(shock, str) else shock
    out = dict(report)
    # Случайная амплитуда шока 0.7..1.3 для разнообразия
    amp = rng.uniform(0.7, 1.3)
    for k, v in params.items():
        if k in out and isinstance(v, (int, float)) and abs(v) < 5:
            if v > 0 and v < 2:
                out[k] = out[k] * (1.0 + (v - 1.0) * amp)
            else:
                out[k] = out[k] * v
        else:
            out[k] = v
    # Пересчёт gross_profit для сохранения правдоподобия
    if "revenue" in out and "cogs" in out:
        out["gross_profit"] = out["revenue"] - out["cogs"]
    return out


def macro_state(dim: int = 8,
                rng: np.random.Generator | None = None) -> np.ndarray:
    """Вектор макро-состояния: ставка, инфляция, VIX, спреды и т.д."""
    rng = rng or np.random.default_rng()
    return np.array([
        rng.uniform(0.0, 0.10),    # ключевая ставка
        rng.uniform(0.0, 0.08),    # инфляция
        rng.uniform(10.0, 40.0),   # VIX
        rng.uniform(0.0, 0.05),    # кредитный спред
        rng.uniform(0.5, 1.2),     # FX
        rng.uniform(0.0, 0.03),    # CDS
        rng.normal(0.0, 0.01),     # изменение ВВП
        rng.uniform(0.0, 1.0),     # сентимент
    ], dtype=np.float32)[:dim]


def generate_stress_dataset(
    n: int = 500, seed: int = 42
) -> list[dict]:
    """
    Генерирует датасет обучающих кортежей:
      {context (report, macro, shock_name) -> pnl_after, risk_after}
    """
    rng = np.random.default_rng(seed)
    data = []
    for _ in range(n):
        base = synthetic_filing(rng)
        shock = rng.choice(list(STRESS_SHOCKS.keys()))
        stressed = apply_shock(base, shock, rng)
        m = macro_state(rng=rng)
        # Простой расчёт P&L после шока
        pnl = (stressed["gross_profit"] - stressed["opex"])
        risk = rng.lognormal(mean=8, sigma=0.5)  # VaR proxy
        data.append({
            "base": base,
            "stressed": stressed,
            "macro": m,
            "shock": shock,
            "pnl": pnl,
            "var": risk,
        })
    return data


def stress_to_tensors(batch: list[dict], n_fields: int = 16
                      ) -> dict[str, torch.Tensor]:
    from .edgar_parser import fields_to_tensor, DEFAULT_UNIT_FIELDS
    fields = []
    masks = []
    pnls = []
    vars_ = []
    macros = []
    for item in batch:
        f, m = fields_to_tensor(item["stressed"], DEFAULT_UNIT_FIELDS[:n_fields])
        fields.append(f)
        masks.append(m)
        pnls.append(item["pnl"])
        vars_.append(item["var"])
        macros.append(torch.from_numpy(item["macro"]))
    return {
        "fields": torch.stack(fields),
        "mask": torch.stack(masks),
        "pnl": torch.tensor(pnls, dtype=torch.float32),
        "var": torch.tensor(vars_, dtype=torch.float32),
        "macro": torch.stack(macros),
    }
