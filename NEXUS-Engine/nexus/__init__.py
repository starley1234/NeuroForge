"""NEXUS-Engine — Non-linear Episodic eXtensible Unified System
for Physical Intelligence & Engineering.

Трёхуровневая архитектура UniPhysical-Latent Framework:
  1. Continuous Dynamic Encoders (SSM / Neural-ODE / B-Rep GNO)
  2. Physical World Core: Unified Latent Bus + TTT + Sliding Attention + Sparse MoE
     + Latent Reasoning Workspace с FNO-суррогатом физики
  3. Dual Output Engine: дискретные токены и непрерывные поля/действия
"""
from .config import (AttentionConfig, FNOConfig, MoEConfig, NexusConfig,
                     ReasoningConfig, TTTConfig)
from .bus import LatentPacket, UnifiedLatentBus
from .model import NexusEngine

__version__ = "0.1.0"
__all__ = [
    "NexusEngine", "NexusConfig", "TTTConfig", "AttentionConfig", "MoEConfig",
    "ReasoningConfig", "FNOConfig", "LatentPacket", "UnifiedLatentBus",
]
