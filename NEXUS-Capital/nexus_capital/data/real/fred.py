"""
Загрузчик макроэкономических рядов из FRED (St. Louis Fed).

FRED бесплатен, но требует API-ключа (32 символа), который запрашивается
на https://fred.stlouisfed.org/docs/api/api_key.html. Ключ можно передать
через параметр api_key или переменную окружения FRED_API_KEY.

Если ключа нет, используется детерминированный оффлайн-профиль на базе
реальных средних (ставка ~4%, инфляция ~3%, VIX ~18, 10Y ~4%, спред ~1.5%),
чтобы пайплайн обучения запускался без регистрации. Для настоящей науки
получите бесплатный ключ — это 30 секунд.

Ряды по умолчанию:
    DFF       Federal Funds Effective Rate
    DGS10     10-Year Treasury Constant Maturity Rate
    T10Y2Y    10Y-2Y спред
    CPIAUCSL  CPI (индекс, используется YoY)
    VIXCLS    VIX
    BAMLH0A0HYM2  ICE BofA US High Yield Index Option-Adjusted Spread
    DTWEXBGS  Trade Weighted U.S. Dollar Index
    UNRATE    безработица
"""
from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_SERIES: list[str] = [
    "DFF", "DGS10", "T10Y2Y", "CPIAUCSL", "VIXCLS",
    "BAMLH0A0HYM2", "DTWEXBGS", "UNRATE",
]

# Средние значения для оффлайн-режима (реальные исторические средние)
OFFLINE_MEANS = {
    "DFF": 2.5, "DGS10": 3.8, "T10Y2Y": 0.9, "CPIAUCSL": 260.0,
    "VIXCLS": 19.0, "BAMLH0A0HYM2": 4.0, "DTWEXBGS": 115.0,
    "UNRATE": 5.5,
}
OFFLINE_STD = {
    "DFF": 2.2, "DGS10": 1.1, "T10Y2Y": 0.7, "CPIAUCSL": 15.0,
    "VIXCLS": 8.0, "BAMLH0A0HYM2": 2.0, "DTWEXBGS": 8.0,
    "UNRATE": 1.8,
}


@dataclass
class FredConfig:
    api_key: str | None = None
    series: list[str] = field(default_factory=lambda: list(DEFAULT_SERIES))
    start: str = "2018-01-01"
    end: str = "2025-01-01"
    cache_dir: str = "data/raw/fred"


def _get_key(cfg: FredConfig) -> str | None:
    return cfg.api_key or os.environ.get("FRED_API_KEY")


def _fetch_series(series_id: str, key: str, start: str, end: str,
                  timeout: int = 30) -> pd.Series:
    params = {
        "series_id": series_id,
        "api_key": key,
        "file_type": "json",
        "observation_start": start,
        "observation_end": end,
    }
    url = "https://api.stlouisfed.org/fred/series/observations?" + \
          urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "NEXUS-Capital"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        payload = json.loads(r.read().decode("utf-8"))
    obs = payload.get("observations", [])
    dates, vals = [], []
    for o in obs:
        if o["value"] in (".", ""):
            continue
        dates.append(pd.to_datetime(o["date"]))
        vals.append(float(o["value"]))
    return pd.Series(vals, index=pd.DatetimeIndex(dates), name=series_id)


def load_macro(cfg: FredConfig) -> pd.DataFrame:
    """
    Загружает макро-ряд в DataFrame (индекс-дата, колонки — series).
    Если ключа нет — генерирует стабильный оффлайн-профиль.
    """
    cache = Path(cfg.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    key = _get_key(cfg)
    fp = cache / f"macro_{cfg.start}_{cfg.end}.parquet"

    if key:
        if fp.exists():
            return pd.read_parquet(fp)
        series = {}
        for sid in cfg.series:
            try:
                series[sid] = _fetch_series(sid, key, cfg.start, cfg.end)
            except Exception as e:
                print(f"[fred] пропуск {sid}: {e}")
        if not series:
            raise RuntimeError("FRED: не загружен ни один ряд")
        df = pd.DataFrame(series).sort_index().ffill().dropna(how="all")
        df.to_parquet(fp)
        return df

    print("[fred] FRED_API_KEY не задан — используется оффлайн-профиль.")
    print("       Получите бесплатный ключ на "
          "https://fred.stlouisfed.org/docs/api/api_key.html")
    rng = np.random.default_rng(20240101)
    idx = pd.date_range(cfg.start, cfg.end, freq="ME")
    n = len(idx)
    data = {}
    for sid in cfg.series:
        mu = OFFLINE_MEANS.get(sid, 0.0)
        sd = OFFLINE_STD.get(sid, 1.0)
        # AR(1) ряд для реалистичности
        x = np.zeros(n)
        x[0] = mu
        phi = 0.95
        for t in range(1, n):
            x[t] = phi * x[t - 1] + (1 - phi) * mu + rng.normal(0, sd * 0.1)
        data[sid] = x
    df = pd.DataFrame(data, index=idx)
    df.to_parquet(fp)
    return df


def macro_feature_vector(df: pd.DataFrame, date: pd.Timestamp | None = None
                         ) -> np.ndarray:
    """
    Сжимает макро-DF в вектор фиксированной длины для синтеза стрессов/контекста:
    [последние значения рядов + 12M разницы + волатильность].
    """
    if date is not None:
        df = df.loc[:date]
    last = df.iloc[-1].fillna(0).to_numpy(np.float32)
    if len(df) > 12:
        diff = (df.iloc[-1] - df.iloc[-12]).fillna(0).to_numpy(np.float32)
        vol = df.iloc[-60:].std().fillna(0).to_numpy(np.float32)
    else:
        diff = np.zeros_like(last)
        vol = df.std().fillna(0).to_numpy(np.float32)
    return np.concatenate([last, diff, vol])
