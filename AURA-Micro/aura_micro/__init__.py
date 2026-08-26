"""AURA-Micro: wearable 3-microphone ultra-low-power acoustic network."""

from aura_micro.config import CLASS_NAMES, MODIFIER_NAMES, AuraConfig
from aura_micro.frontend import HardwareDSPFrontEnd
from aura_micro.model import AuraMicro
from aura_micro.pipeline import AuraPipeline

__all__ = [
    "AuraConfig",
    "CLASS_NAMES",
    "MODIFIER_NAMES",
    "AuraMicro",
    "AuraPipeline",
    "HardwareDSPFrontEnd",
]
__version__ = "0.2.0"
