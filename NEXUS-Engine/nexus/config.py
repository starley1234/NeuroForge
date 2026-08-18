"""Конфигурация NEXUS-Engine (UniPhysical-Latent Framework).

Все дефолты подобраны под целевой бюджет RTX 5060 / 16 ГБ VRAM
(см. nexus.eval.vram и docs/vram_budget.md).
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, Any
import json


@dataclass
class TTTConfig:
    """Linear Fast-Weights Memory (Test-Time Training) — O(1) память."""
    chunk_size: int = 128            # L: длина чанка префикса
    key_dim: int = 64                # ключ/запрос быстрых весов
    value_dim: int = 64
    base_lr: float = 1.0             # η_t (модулируется сетью)
    base_decay: float = 0.02         # α_t (модулируется сетью)
    n_heads: int = 8


@dataclass
class AttentionConfig:
    """Локальное скользящее окно высокой точности."""
    window: int = 1024               # W = 1024..2048
    n_heads: int = 12
    n_kv_heads: int = 4              # GQA
    rope_theta: float = 10000.0


@dataclass
class MoEConfig:
    """Fine-Grained Sparse MoE: 1 общий + k активных из n."""
    n_experts: int = 32
    n_active: int = 4
    n_shared: int = 1
    expert_hidden_mult: float = 1.3333  # мелкозернистые эксперты (доля от d_latent)
    router_z_loss: float = 1e-3
    load_balance_loss: float = 1e-2


@dataclass
class ReasoningConfig:
    """Latent Reasoning Workspace: мышление без генерации токенов."""
    max_steps: int = 8
    min_steps: int = 1
    halt_threshold: float = 0.95     # Adaptive Pondering
    ponder_cost: float = 1e-2
    use_fno_critic: bool = True


@dataclass
class FNOConfig:
    """Суррогат FEM/CFD (Fourier Neural Operator), ~30M параметров."""
    modes: int = 12
    width: int = 48
    depth: int = 4
    grid: int = 32                   # воксельная сетка поля
    in_channels: int = 4             # occupancy + 3 компоненты нагрузки
    out_channels: int = 1            # von Mises σ


@dataclass
class NexusConfig:
    d_latent: int = 1536
    n_layers: int = 24
    vocab_size: int = 4096
    max_time: float = 1e4            # верхняя граница непрерывной оси t (сек)
    dropout: float = 0.0
    tie_embeddings: bool = True
    action_dim: int = 12             # непрерывные действия / траектории
    field_grid: int = 16             # выход тензорных полей FEM/CFD

    ttt: TTTConfig = field(default_factory=TTTConfig)
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    moe: MoEConfig = field(default_factory=MoEConfig)
    reasoning: ReasoningConfig = field(default_factory=ReasoningConfig)
    fno: FNOConfig = field(default_factory=FNOConfig)

    # ---------------------------------------------------------------- utils
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NexusConfig":
        sub = {
            "ttt": TTTConfig,
            "attention": AttentionConfig,
            "moe": MoEConfig,
            "reasoning": ReasoningConfig,
            "fno": FNOConfig,
        }
        kwargs = {k: v for k, v in data.items() if k not in sub}
        for name, klass in sub.items():
            if name in data:
                kwargs[name] = klass(**data[name])
        return cls(**kwargs)

    @classmethod
    def load(cls, path: str) -> "NexusConfig":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    # --------------------------------------------------------------- presets
    @classmethod
    def tiny(cls) -> "NexusConfig":
        """Отладочная конфигурация (CPU, юнит-тесты, CI)."""
        return cls(
            d_latent=96,
            n_layers=3,
            vocab_size=512,
            action_dim=6,
            field_grid=8,
            ttt=TTTConfig(chunk_size=16, key_dim=16, value_dim=16, n_heads=2),
            attention=AttentionConfig(window=32, n_heads=4, n_kv_heads=2),
            moe=MoEConfig(n_experts=8, n_active=2, n_shared=1),
            reasoning=ReasoningConfig(max_steps=3),
            fno=FNOConfig(modes=4, width=12, depth=2, grid=8),
        )

    @classmethod
    def rtx5060(cls) -> "NexusConfig":
        """Целевой профиль: ~1.36B активных параметров, ~7.7B суммарных весов MoE."""
        return cls()

    @classmethod
    def rtx5060_compact(cls) -> "NexusConfig":
        """Вариант под таблицу VRAM из спецификации: ~3.6B суммарных весов MoE.

        Меньше экспертов (14 вместо 32) при той же активной ёмкости — меньше
        оффлоада на CPU, но ниже суммарная память знаний.
        """
        cfg = cls()
        cfg.moe = MoEConfig(n_experts=14, n_active=4, n_shared=1, expert_hidden_mult=1.3333)
        return cfg
