"""Коннекторы реальных открытых данных: Binance, SEC EDGAR, FRED."""
from .binance import (
    BinanceConfig, download_klines, download_aggtrades,
    klines_to_tick_features, synthetic_book_from_klines,
)
from .sec_edgar import EdgarConfig, load_filings, to_field_tensor, TAG_MAP
from .fred import FredConfig, load_macro, macro_feature_vector, DEFAULT_SERIES
from .real_dataset import (
    RealDataConfig, RealMarketDataset, collate_real, default_config,
)

__all__ = [
    "BinanceConfig", "download_klines", "download_aggtrades",
    "klines_to_tick_features", "synthetic_book_from_klines",
    "EdgarConfig", "load_filings", "to_field_tensor", "TAG_MAP",
    "FredConfig", "load_macro", "macro_feature_vector", "DEFAULT_SERIES",
    "RealDataConfig", "RealMarketDataset", "collate_real", "default_config",
]
