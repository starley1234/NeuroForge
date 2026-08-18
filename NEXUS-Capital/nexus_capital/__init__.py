"""
NEXUS-Capital (VALUEX) — Value-Augmented Latent Universal Exchange
and Economic Reasoning Engine.

Экономически заземлённая мультимодальная архитектура: непрерывные
энкодеры модальностей, единое пространство стоимости с TTT-памятью,
локальным финансовым вниманием и разреженным MoE, латентный
Monte-Carlo/теория игр и двухрежимный вывод.
"""
from .core.config import NexusConfig, small_config, rtx5060_config
from .core.loader import load_config, named_config
from .core.tokenizer import NexusTokenizer
from .models.nexus_model import NexusCapital

__version__ = "0.1.0"
__all__ = [
    "NexusConfig", "NexusCapital", "NexusTokenizer",
    "small_config", "rtx5060_config",
    "load_config", "named_config", "build_model", "__version__",
]


def build_model(cfg: NexusConfig | None = None,
                tokenizer: "NexusTokenizer | None" = None) -> "NexusCapital":
    cfg = cfg or small_config()
    return NexusCapital(cfg, tokenizer=tokenizer)
