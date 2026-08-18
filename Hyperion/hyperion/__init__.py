"""
Hyperion — гибридная мультимодальная LLM архитектура.

Собирает ключевые инновации 2024-2026:
- MLA (Multi-Head Latent Attention) — сжатый KV-кэш
- Mamba-3 (State Space Model) — линейная сложность
- Titans Neural Memory — бесконечный контекст
- JEPA — обучение в пространстве эмбеддингов
- MoSE — slimmable mixture-of-experts
- MTP — multi-token prediction
"""

__version__ = "0.1.0"

from hyperion.config import HyperionConfig
from hyperion.model import HyperionModel

__all__ = ["HyperionConfig", "HyperionModel"]
