"""
Тесты коннекторов реальных данных.

Сеть в CI/песочнице может быть недоступна, поэтому все тесты используют
оффлайн-режим (FRED без ключа) и детерминированные пути Binance/EDGAR.
Сами сетевые загрузчики покрываются юнит-проверками парсинга и URL.
"""
import numpy as np
import pandas as pd
import pytest

from nexus_capital.data.real.binance import (
    klines_to_tick_features, synthetic_book_from_klines, kline_url,
    aggtrades_url, KLINE_COLS,
)
from nexus_capital.data.real.fred import (
    FredConfig, load_macro, macro_feature_vector, DEFAULT_SERIES,
)
from nexus_capital.data.real.real_dataset import (
    RealDataConfig, RealMarketDataset, collate_real,
)


def _fake_klines(n=500, seed=0):
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0, 0.002, n)
    price = 100 * np.exp(np.cumsum(rets))
    return pd.DataFrame({
        "open_time": pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC"),
        "open": np.concatenate([[price[0]], price[:-1]]),
        "high": price * 1.001,
        "low": price * 0.999,
        "close": price,
        "volume": rng.lognormal(2, 0.5, n),
        "quote_volume": price * rng.lognormal(2, 0.5, n),
        "trades": rng.integers(10, 500, n),
        "taker_buy_base": rng.lognormal(2, 0.5, n) * 0.5,
    })


def test_kline_url_patterns():
    url = kline_url("BTCUSDT", "1m", 2024, 1, "spot")
    assert "data.binance.vision" in url
    assert "BTCUSDT-1m-2024-01.zip" in url
    url2 = aggtrades_url("ETHUSDT", 2023, 12)
    assert "aggTrades" in url2 and "ETHUSDT" in url2


def test_klines_to_tick_features_shape_and_finite():
    df = _fake_klines()
    feats = klines_to_tick_features(df, n_features=8)
    assert feats.shape == (len(df), 8)
    assert np.isfinite(feats).all()
    # side должен быть ±1/0
    assert set(np.unique(feats[:, 2])).issubset({-1.0, 0.0, 1.0})


def test_synthetic_book_from_klines():
    df = _fake_klines()
    book = synthetic_book_from_klines(df, n_levels=16)
    assert book.shape == (64,)  # bid_p, bid_sz, ask_p, ask_sz
    assert np.isfinite(book).all()
    # bid ниже ask
    mid = df["close"].iloc[len(df) // 2]
    assert book[15] <= mid <= book[2 * 16]


def test_fred_offline_profile(monkeypatch):
    # Без ключа и без сети — оффлайн AR(1)-профиль
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    cfg = FredConfig(series=DEFAULT_SERIES[:3], start="2020-01-01",
                     end="2021-01-01", cache_dir="/tmp/fred_test")
    df = load_macro(cfg)
    assert df.shape[1] == 3
    assert len(df) > 0
    vec = macro_feature_vector(df)
    assert np.isfinite(vec).all()
    assert vec.shape[0] == 9  # 3 last + 3 diff + 3 vol


def test_real_dataset_offline(monkeypatch, tmp_path):
    # Binance без сети -> offline klines; FRED без ключа -> offline
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    cfg = RealDataConfig(
        binance=None, edgar=None,
        fred=FredConfig(start="2020-01-01", end="2021-01-01",
                        cache_dir=str(tmp_path / "fred")),
        tick_seq_len=32, n_fields=16, n_tick_features=8, n_levels=16,
        processed_dir=str(tmp_path / "proc"), max_samples=50,
    )
    # binance=None — пропускаем рынок? Нет, зададим вручную offline через None:
    # для теста соберём датасет только из макро + синтетики невозможно,
    # поэтому проверим, что при недоступности сети он падает в offline.
    cfg.binance = None
    ds = RealMarketDataset(cfg)
    # Без тиков датасет пуст
    assert len(ds) == 0


def test_real_dataset_with_offline_market(monkeypatch, tmp_path):
    monkeypatch.delenv("FRED_API_KEY", raising=False)

    class _OfflineBinance:
        symbol = "TESTUSDT"; interval = "1m"; market = "spot"
        start_month = "2024-01"; end_month = "2024-01"
        cache_dir = str(tmp_path / "bn")

    cfg = RealDataConfig(
        binance=_OfflineBinance(), edgar=None,
        fred=FredConfig(start="2020-01-01", end="2021-01-01",
                        cache_dir=str(tmp_path / "fred")),
        tick_seq_len=16, n_fields=16, n_levels=8,
        processed_dir=str(tmp_path / "proc"), max_samples=20,
    )
    ds = RealMarketDataset(cfg)
    # Сеть недоступна -> offline OHLCV
    assert len(ds) == 20
    s = ds[0]
    assert s["ticks"].shape == (16, 8)
    assert s["book"].shape == (32,)
    assert torch_is_finite(s["ticks"])
    assert "target_return" in s


def test_collate_real(monkeypatch, tmp_path):
    import torch
    monkeypatch.delenv("FRED_API_KEY", raising=False)

    class _OB:
        symbol = "T"; interval = "1m"; market = "spot"
        start_month = end_month = "2024-01"
        cache_dir = str(tmp_path / "bn")

    cfg = RealDataConfig(
        binance=_OB(), edgar=None,
        fred=FredConfig(start="2020-01-01", end="2021-01-01",
                        cache_dir=str(tmp_path / "fred")),
        tick_seq_len=8, n_fields=16, n_levels=8,
        processed_dir=str(tmp_path / "p"), max_samples=6,
    )
    ds = RealMarketDataset(cfg)
    batch = collate_real([ds[i] for i in range(4)])
    assert batch["ticks"].shape[0] == 4
    assert batch["book"].shape[1] == 32
    assert batch["fields"].shape == (4, 16)


def torch_is_finite(t):
    import torch
    return bool(torch.isfinite(t).all())
