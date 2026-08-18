"""
Интегрированный набор данных NEXUS-Capital поверх реальных источников:
    • Binance klines -> тиковые признаки + синтетический L2 стакан
    • SEC EDGAR -> поля юнит-экономики/баланса
    • FRED -> макро-вектор (ставка, инфляция, VIX, спреды, FX)

Скрипт подготовки кэширует всё в data/raw и data/processed и возвращает
батчи в формате, который понимает NexusCapital.forward.

Режимы задачи:
    - "invariants": предсказание экономических инвариантов
      (CF, sigma, discount, elasticity) из рыночно-корпоративного контекста;
    - "risk": обучение с L_total (предсказание + Utility/VaR), где в качестве
      target используется следующий бар доходности;
    - "pricing": динамическое ценообразование (цель — следующий mid).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .binance import (
    BinanceConfig, download_klines, klines_to_tick_features,
    synthetic_book_from_klines,
)
from .sec_edgar import EdgarConfig, load_filings, to_field_tensor
from .fred import FredConfig, load_macro, macro_feature_vector


@dataclass
class RealDataConfig:
    binance: BinanceConfig | None = None
    edgar: EdgarConfig | None = None
    fred: FredConfig | None = None
    tick_seq_len: int = 64
    n_fields: int = 16
    n_tick_features: int = 8
    n_levels: int = 16
    task: Literal["invariants", "risk", "pricing"] = "risk"
    processed_dir: str = "data/processed"
    max_samples: int = 10000
    # Хронологический сплит: первые val_frac по времени — train, последние — val.
    # Это критично для предотвращения утечки будущего.
    val_frac: float = 0.1


class RealMarketDataset(Dataset):
    """
    Каждый пример = срез тиков Binance + (опц.) поля отчётности случайной
    компании + макро-вектор. Целевая переменная — лог-доходность следующего
    бара (для risk/pricing) или инварианты.
    """

    def __init__(self, cfg: RealDataConfig):
        self.cfg = cfg
        self.tick_features: np.ndarray | None = None
        self.klines: pd.DataFrame | None = None
        self.fields: np.ndarray | None = None
        self.field_mask: np.ndarray | None = None
        self.macro_vec: np.ndarray | None = None
        self.targets: np.ndarray | None = None
        self._load_all()

    # ── загрузка ─────────────────────────────────────────────────
    def _offline_klines(self, n: int) -> pd.DataFrame:
        """Детерминированный геометрический броуновский OHLCV для оффлайна."""
        rng = np.random.default_rng(20240101)
        dt = 1.0 / (24 * 60)  # 1m бары
        mu, sigma = 0.00002, 0.002
        rets = rng.normal(mu, sigma, n)
        price = 40000 * np.exp(np.cumsum(rets))
        high = price * (1 + np.abs(rng.normal(0, 0.001, n)))
        low = price * (1 - np.abs(rng.normal(0, 0.001, n)))
        open_ = np.concatenate([[price[0]], price[:-1]])
        volume = rng.lognormal(2.0, 0.8, n)
        taker = volume * rng.uniform(0.3, 0.7, n)
        trades = rng.integers(50, 2000, n)
        times = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
        return pd.DataFrame({
            "open_time": times, "open": open_, "high": high, "low": low,
            "close": price, "volume": volume, "quote_volume": volume * price,
            "trades": trades, "taker_buy_base": taker,
        })

    def _load_all(self) -> None:
        Path(self.cfg.processed_dir).mkdir(parents=True, exist_ok=True)

        # 1) Binance klines
        if self.cfg.binance is not None:
            try:
                df = download_klines(self.cfg.binance)
                df = df.sort_values("open_time").reset_index(drop=True)
                self.klines = df
                self.tick_features = klines_to_tick_features(
                    df, self.cfg.n_tick_features)
                print(f"[dataset] klines: {len(df)} баров, "
                      f"features {self.tick_features.shape}")
            except Exception as e:
                print(f"[dataset] Binance недоступен ({e}); "
                      f"генерирую оффлайн-OHLCV.")
                df = self._offline_klines(20000)
                self.klines = df
                self.tick_features = klines_to_tick_features(
                    df, self.cfg.n_tick_features)
                print(f"[dataset] offline klines: {len(df)} баров")

        # 2) SEC EDGAR (опционально; может быть медленно при первом запуске)
        if self.cfg.edgar is not None:
            try:
                filings = load_filings(self.cfg.edgar)
                f, m, order = to_field_tensor(
                    filings, n_fields=self.cfg.n_fields)
                self.fields = f
                self.field_mask = m
                print(f"[dataset] EDGAR: {len(f)} компаний/подач, "
                      f"полей {len(order)}")
            except Exception as e:
                print(f"[dataset] EDGAR недоступен ({e}); "
                      f"использую синтетические поля.")
                self.fields = None

        # 3) FRED макро
        if self.cfg.fred is not None:
            macro_df = load_macro(self.cfg.fred)
            self.macro_vec = macro_feature_vector(macro_df).astype(np.float32)
            print(f"[dataset] macro vector dim={self.macro_vec.shape[0]}")

        # 4) Цели по тикам (следующая доходность close->close)
        if self.tick_features is not None:
            log_p = self.tick_features[:, 0]
            fwd = np.diff(log_p, append=log_p[-1])
            self.targets = fwd.astype(np.float32)

    # ── Dataset API ───────────────────────────────────────────────
    def _n_windows(self) -> int:
        if self.tick_features is None:
            return 0
        return max(0, len(self.tick_features) - self.cfg.tick_seq_len - 1)

    def __len__(self) -> int:
        return min(self._n_windows(), self.cfg.max_samples)

    def chronological_split(self, val_frac: float | None = None
                            ) -> tuple["RealMarketDataset", "RealMarketDataset"]:
        """
        Хронологическое (не случайное!) разбиение train/val. Первые окна —
        train, последние — val, без пересечения и без утечки будущего.
        """
        frac = self.cfg.val_frac if val_frac is None else val_frac
        n = self._n_windows()
        n_train = int(n * (1.0 - frac))
        # Делаем легковесные обёртки, ссылающиеся на те же массивы
        train_ds = _ChronoSubset(self, 0, n_train)
        val_ds = _ChronoSubset(self, n_train, n)
        return train_ds, val_ds

    def raw_log_returns(self) -> np.ndarray | None:
        """Сырые лог-доходности close->close для калибровки Neural SDE.
        Это НЕ кумулятивный log-price, а именно покоординатные разности."""
        if self.tick_features is None:
            return None
        # tick_features[:, 0] = log(close/base), поэтому доходности = diff
        log_p = self.tick_features[:, 0]
        return np.diff(log_p, prepend=log_p[0]).astype(np.float32)

    def __getitem__(self, idx: int) -> dict:
        s = idx
        e = s + self.cfg.tick_seq_len
        ticks = self.tick_features[s:e]
        book = synthetic_book_from_klines(
            self.klines, n_levels=self.cfg.n_levels, row=e - 1)

        # Поля отчётности: реальные из EDGAR либо, если не загрузились,
        # детерминированный «синтетический» профиль, чтобы пример был валиден.
        if self.fields is not None and len(self.fields) > 0:
            row = idx % len(self.fields)
            fields = self.fields[row]
            mask = self.field_mask[row]
        else:
            fields = np.zeros(self.cfg.n_fields, dtype=np.float32)
            mask = np.zeros(self.cfg.n_fields, dtype=np.float32)
            # Используем log-масштаб цены как прокси выручки (без expm1)
            fields[0] = np.float32(
                np.log1p(self.klines["close"].iloc[e]))
            mask[0] = 1.0

        sample = {
            "book": torch.from_numpy(book),
            "ticks": torch.from_numpy(ticks),
            "fields": torch.from_numpy(fields),
            "field_mask": torch.from_numpy(mask),
        }
        if self.macro_vec is not None:
            sample["macro"] = torch.from_numpy(self.macro_vec)
        if self.targets is not None:
            sample["target_return"] = torch.tensor(
                self.targets[e], dtype=torch.float32)
        return sample


class _ChronoSubset(Dataset):
    """Хронологический срез RealMarketDataset (не шафлит окна)."""

    def __init__(self, parent: "RealMarketDataset", start: int, end: int):
        self.parent = parent
        self.start = start
        self.end = max(start, end)

    def __len__(self) -> int:
        return self.end - self.start

    def __getitem__(self, i: int) -> dict:
        return self.parent[self.start + i]


def collate_real(batch: list[dict]) -> dict:
    """Собирает список примеров в батч-тензоры."""
    out: dict[str, torch.Tensor] = {}
    for key in ("book", "ticks", "fields", "field_mask"):
        if key in batch[0]:
            out[key] = torch.stack([b[key] for b in batch])
    if "macro" in batch[0]:
        out["macro"] = torch.stack([b["macro"] for b in batch])
    if "target_return" in batch[0]:
        out["target_return"] = torch.stack(
            [b["target_return"] for b in batch])
    return out


def default_config(download_sec: bool = True,
                   sec_years: tuple[int, ...] = (2023, 2024)) -> RealDataConfig:
    """Разумная конфигурация «из коробки»."""
    bc = BinanceConfig(
        symbol="BTCUSDT", interval="1m", market="spot",
        start_month="2024-01", end_month="2024-03",
    )
    ec = EdgarConfig(years=list(sec_years), quarters=[1, 2, 3, 4],
                    cache_dir="data/raw/sec") if download_sec else None
    fc = FredConfig(start="2018-01-01", end="2025-01-01",
                    cache_dir="data/raw/fred")
    return RealDataConfig(binance=bc, edgar=ec, fred=fc,
                          n_fields=16, n_levels=16, tick_seq_len=64)
