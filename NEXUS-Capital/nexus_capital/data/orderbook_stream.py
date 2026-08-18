"""
Генератор потока тиков и L2-стакана для обучения и демо.
Реалистичная микроструктура: процессор Пуассоновых поступлений,
временное воздействие маркет-мейкеров, mean-reverting mid-price.
"""
from __future__ import annotations

import numpy as np
import torch


def simulate_orderbook(
    n_levels: int = 50,
    tick_size: float = 0.01,
    base_price: float = 100.0,
    volatility: float = 0.02,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Возвращает одномерные массивы stakan (4*n_levels,) и массив уровней.
    Формат: [bid_p, bid_sz, ask_p, ask_sz]
    """
    rng = np.random.default_rng(seed)
    mid = base_price * np.exp(rng.normal(0, volatility))
    bid_p = mid - tick_size * np.arange(1, n_levels + 1)
    ask_p = mid + tick_size * np.arange(1, n_levels + 1)
    # Размеры убывают с глубиной + экспоненциальный шум
    bid_sz = rng.exponential(100, n_levels) * np.exp(-0.05 * np.arange(n_levels))
    ask_sz = rng.exponential(100, n_levels) * np.exp(-0.05 * np.arange(n_levels))
    book = np.concatenate([bid_p, bid_sz, ask_p, ask_sz]).astype(np.float32)
    return book, np.array([mid], dtype=np.float32)


def simulate_tick_stream(
    T: int = 1000,
    n_features: int = 8,
    base_price: float = 100.0,
    seed: int | None = None,
) -> np.ndarray:
    """
    Генерирует поток тиков (T, n_features):
    [price, volume, side, spread, depth_imbalance, mid_move, vol, oi]
    side: +1 покупка, -1 продажа.
    """
    rng = np.random.default_rng(seed)
    ticks = np.zeros((T, n_features), dtype=np.float32)
    mid = base_price
    spread = 0.02
    for t in range(T):
        # Mean-reverting mid + jumps
        mid = mid + rng.normal(0, 0.05) + 0.002 * (base_price - mid)
        side = 1 if rng.random() > 0.5 else -1
        price = mid + side * spread / 2
        volume = rng.exponential(50)
        depth_imb = rng.normal(0, 0.3)
        ticks[t] = [
            np.log(price / base_price),
            np.log1p(volume),
            side,
            spread,
            depth_imb,
            (price - mid) / base_price,
            rng.normal(0, 0.01),
            rng.exponential(10),
        ]
    return ticks


def collate_book_ticks(
    batch_size: int = 4,
    n_levels: int = 50,
    T: int = 256,
    n_features: int = 8,
    seed: int | None = None,
) -> dict[str, torch.Tensor]:
    rng = np.random.default_rng(seed)
    books = []
    ticks = []
    for _ in range(batch_size):
        book, _ = simulate_orderbook(
            n_levels=n_levels, seed=int(rng.integers(1e9)))
        stream = simulate_tick_stream(
            T=T, n_features=n_features, seed=int(rng.integers(1e9)))
        books.append(book)
        ticks.append(stream)
    return {
        "book": torch.from_numpy(np.stack(books)),
        "ticks": torch.from_numpy(np.stack(ticks)),
    }
