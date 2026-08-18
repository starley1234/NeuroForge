"""
Загрузчик реальных рыночных данных с Binance Public Data (data.binance.vision).

Без API-ключа. Качает ежемесячные/ежедневные ZIP-архивы с klines и
aggTrades прямым HTTPS, парсит CSV и формирует признаки, совместимые с
OrderBookEncoder:
    [log(price/base), log1p(volume), side, spread_proxy, depth_imbalance,
     mid_move, realized_vol, log1p(trade_count)]

Полноценных L2-стаканов в публичном дампе нет, поэтому L2 восстанавливается
приближённо из OHLCV (синтетический book вокруг mid). Для настоящих L2/L3
используйте Tardis.dev или биржевые веб-сокеты — см. sources.md.

URL-шаблоны:
    https://data.binance.vision/data/spot/monthly/klines/SYMBOL/1m/SYMBOL-1m-YYYY-MM.zip
    https://data.binance.vision/data/spot/daily/klines/SYMBOL/1m/SYMBOL-1m-YYYY-MM-DD.zip
    https://data.binance.vision/data/spot/monthly/aggTrades/SYMBOL/SYMBOL-aggTrades-YYYY-MM.zip
"""
from __future__ import annotations

import io
import zipfile
import urllib.request
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


KLINE_COLS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trades", "taker_buy_base",
    "taker_buy_quote", "ignore",
]
AGG_COLS = [
    "agg_trade_id", "price", "quantity", "first_trade_id",
    "last_trade_id", "transact_time", "is_buyer_maker",
]

_UA = "NEXUS-Capital/0.1 (research; contact@neuroforge)"


def _http_get(url: str, timeout: int = 30, retries: int = 3) -> bytes:
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": _UA})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            last = e
            import time
            time.sleep(1.5 * (attempt + 1))
    raise last  # type: ignore[misc]


def _read_zipped_csv(data: bytes, cols: list[str]) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        name = zf.namelist()[0]
        with zf.open(name) as f:
            df = pd.read_csv(f, header=None)
    if df.shape[1] == len(cols):
        df.columns = cols
    else:
        df = df.iloc[:, :len(cols)]
        df.columns = cols
    return df


def month_range(start: str, end: str) -> list[tuple[int, int]]:
    s = date.fromisoformat(start + "-01")
    e = date.fromisoformat(end + "-01")
    out = []
    y, m = s.year, s.month
    while (y, m) <= (e.year, e.month):
        out.append((y, m))
        m += 1
        if m > 12:
            m, y = 1, y + 1
    return out


@dataclass
class BinanceConfig:
    symbol: str = "BTCUSDT"
    interval: str = "1m"
    market: str = "spot"          # spot | um (usd-m futures)
    start_month: str = "2024-01"  # YYYY-MM
    end_month: str = "2024-01"
    cache_dir: str = "data/raw/binance"


def kline_url(symbol: str, interval: str, year: int, month: int,
              market: str = "spot", freq: str = "monthly") -> str:
    stem_month = f"{symbol}-{interval}-{year}-{month:02d}"
    if market == "spot":
        base = "data/spot"
    elif market == "um":
        base = "data/futures/um"
    else:
        raise ValueError(f"unknown market {market}")
    return (f"https://data.binance.vision/{base}/{freq}/klines/"
            f"{symbol}/{interval}/{stem_month}.zip")


def aggtrades_url(symbol: str, year: int, month: int,
                  market: str = "spot") -> str:
    stem = f"{symbol}-aggTrades-{year}-{month:02d}"
    base = "data/spot" if market == "spot" else "data/futures/um"
    return (f"https://data.binance.vision/{base}/monthly/aggTrades/"
            f"{symbol}/{stem}.zip")


def download_klines(cfg: BinanceConfig) -> pd.DataFrame:
    """Качает и склеивает klines за диапазон месяцев (с кэшем на диск)."""
    cache = Path(cfg.cache_dir) / cfg.symbol / "klines" / cfg.interval
    cache.mkdir(parents=True, exist_ok=True)
    frames = []
    for (y, m) in month_range(cfg.start_month, cfg.end_month):
        fp = cache / f"{cfg.symbol}-{cfg.interval}-{y}-{m:02d}.parquet"
        if fp.exists():
            frames.append(pd.read_parquet(fp))
            continue
        url = kline_url(cfg.symbol, cfg.interval, y, m, cfg.market)
        try:
            raw = _http_get(url)
        except Exception as e:
            print(f"[binance] пропуск {y}-{m:02d}: {e}")
            continue
        df = _read_zipped_csv(raw, KLINE_COLS)
        for c in ("open", "high", "low", "close", "volume", "quote_volume"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["trades"] = pd.to_numeric(df["trades"], errors="coerce")
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df = df.dropna(subset=["close"])
        df.to_parquet(fp)
        frames.append(df)
        print(f"[binance] {cfg.symbol} {cfg.interval} {y}-{m:02d}: "
              f"{len(df)} баров")
    if not frames:
        raise RuntimeError("Не удалось скачать ни одного месяца klines")
    return pd.concat(frames, ignore_index=True).sort_values("open_time")


def download_aggtrades(symbol: str, year: int, month: int,
                       cache_dir: str = "data/raw/binance",
                       market: str = "spot") -> pd.DataFrame:
    cache = Path(cache_dir) / symbol / "aggTrades"
    cache.mkdir(parents=True, exist_ok=True)
    fp = cache / f"{symbol}-aggTrades-{year}-{month:02d}.parquet"
    if fp.exists():
        return pd.read_parquet(fp)
    url = aggtrades_url(symbol, year, month, market)
    raw = _http_get(url)
    df = _read_zipped_csv(raw, AGG_COLS)
    for c in ("price", "quantity"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["transact_time"] = pd.to_datetime(df["transact_time"], unit="ms",
                                         utc=True)
    df = df.dropna(subset=["price"])
    df.to_parquet(fp)
    print(f"[binance] aggTrades {symbol} {year}-{month:02d}: {len(df)} строк")
    return df


def klines_to_tick_features(df: pd.DataFrame, n_features: int = 8
                            ) -> np.ndarray:
    """
    Преобразует OHLCV-бар в 8-мерный тик-признак для OrderBookEncoder:
        [log(close/base), log1p(volume), side(taker_sign), spread_proxy,
         depth_imbalance_proxy, log_ret, realized_vol_proxy, log1p(trades)]
    """
    close = df["close"].to_numpy(dtype=np.float64)
    base = close[0] if close[0] > 0 else 1.0
    log_p = np.log(close / base).astype(np.float32)
    vol = np.log1p(df["volume"].to_numpy(dtype=np.float64)).astype(np.float32)
    # знак: преобладание покупок тейкеров
    taker = df["taker_buy_base"].to_numpy(dtype=np.float64)
    total = df["volume"].to_numpy(dtype=np.float64)
    side = np.sign(taker / np.maximum(total, 1e-9) - 0.5).astype(np.float32)
    spread = ((df["high"] - df["low"]) / close).astype(np.float32)
    depth_imb = (taker - (total - taker)) / np.maximum(total, 1e-9)
    depth_imb = depth_imb.astype(np.float32)
    log_ret = np.diff(np.log(np.maximum(close, 1e-9)), prepend=np.log(base))
    log_ret = log_ret.astype(np.float64)
    rv = pd.Series(log_ret).rolling(10, min_periods=1).std().fillna(0.0)
    rv = rv.to_numpy(dtype=np.float32)
    log_ret = log_ret.astype(np.float32)
    trades = np.log1p(df["trades"].fillna(0).to_numpy(np.float64)).astype(
        np.float32)
    arr = np.stack([log_p, vol, side, spread, depth_imb, log_ret, rv, trades],
                   axis=-1)
    if n_features != 8:
        # дублируем/обрезаем до нужного числа фич
        if n_features < 8:
            arr = arr[:, :n_features]
        else:
            pad = np.tile(arr[:, -1:], (1, n_features - 8))
            arr = np.concatenate([arr, pad], axis=1)
    return arr


def synthetic_book_from_klines(df: pd.DataFrame, n_levels: int = 50,
                               row: int | None = None) -> np.ndarray:
    """
    Строит приближённый L2-стакан вокруг mid из OHLCV-баров:
        [bid_p, bid_sz, ask_p, ask_sz] по n_levels уровням.
    """
    rng = np.random.default_rng(0 if row is None else row)
    if row is None:
        row = len(df) // 2
    mid = float(df["close"].iloc[row])
    rng_mid = float(df["high"].iloc[row] - df["low"].iloc[row]) / 2
    tick = max(mid * 1e-4, 0.01)
    x = np.arange(1, n_levels + 1)
    bid_p = mid - tick * x
    ask_p = mid + tick * x
    base_sz = float(df["volume"].iloc[row]) / n_levels
    bid_sz = rng.exponential(base_sz, n_levels) * np.exp(-0.05 * x)
    ask_sz = rng.exponential(base_sz, n_levels) * np.exp(-0.05 * x)
    return np.concatenate([bid_p, bid_sz, ask_p, ask_sz]).astype(np.float32)
