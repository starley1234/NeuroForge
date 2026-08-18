"""
NEXUS-Capital configuration.

Определяет размерность единого экономического латентного пространства
(d_value), параметры TTT-памяти, локального внимания, разреженного MoE,
Neural SDE и симулятора. Значения по умолчанию соответствуют бюджетному
варианту для RTX 5060 (16 ГБ VRAM).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class NexusConfig:
    # ── Единое пространство стоимости ──────────────────────────────
    d_value: int = 1536                 # размерность латентного экономического пространства
    n_layers: int = 24                  # число блоков ядра
    n_heads: int = 12
    d_ff: int = 4096
    dropout: float = 0.0

    # ── Уровень 1: энкодеры модальностей ───────────────────────────
    orderbook_levels: int = 50          # L2 уровней в стакане
    tick_features: int = 8              # признаки тика (price, volume, side, ...)
    tabular_features: int = 64          # юнит-экономика, P&L, ERP поля
    graph_node_dim: int = 64
    audio_n_mels: int = 64
    text_vocab_size: int = 250001       # XLM-RoBERTa sentencepiece (~250k)
    text_embed_dim: int = 256           # embedding до проекции в d_value
    text_seq_len: int = 2048            # W = 2048 (Local Financial Attention)
    text_n_pos: int = 2048

    # ── TTT быстрая память (Market Regime) ─────────────────────────
    ttt_rank: int = 128                 # ранги быстрой матрицы M_t
    ttt_alpha: float = 0.95             # (1 - alpha) коэффициент затухания
    ttt_eta: float = 0.05               # шаг быстрого градиента
    ttt_mem_dim: int = 256

    # ── Sparse MoE ────────────────────────────────────────────────
    n_experts: int = 8
    experts_per_token: int = 2
    expert_ff: int = 2048
    expert_names: List[str] = field(default_factory=lambda: [
        "risk_var", "pricing_unit_econ", "macro_fx", "legal_compliance",
        "credit_default", "liquidity", "derivatives", "sentiment",
    ])

    # ── Neural SDE / латентный Монте-Карло ────────────────────────
    sde_hidden: int = 256
    sde_steps: int = 64
    mc_paths: int = 10000
    mc_horizon: int = 32

    # ── Функция полезности и функция потерь ───────────────────────
    gamma_risk: float = 1.0             # неприятие риска (CRRA)
    risk_alpha: float = 0.05            # уровень VaR (5%)
    lambda_utility: float = 1.0
    lambda_var: float = 0.5
    discount_rate: float = 0.05         # безрисковая ставка для e^{-rt}

    # ── Обучение ──────────────────────────────────────────────────
    vocab_size: int = 32000
    max_seq_len: int = 2048
    pad_token_id: int = 0
    dtype: str = "float32"              # float32|float16|bfloat16

    # ── Бюджет параметров (для отчётов) ───────────────────────────
    def estimate_params(self) -> Dict[str, float]:
        core = 12 * self.d_value ** 2 * self.n_layers / 1e9
        experts = (self.n_experts * 3 * self.d_value * self.expert_ff) / 1e9
        sde = (self.d_value * self.sde_hidden * 4) / 1e9
        return {
            "core_B": core,
            "moe_total_B": experts,
            "sde_M": sde * 1000,
            "active_experts": self.experts_per_token,
        }

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "NexusConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "NexusConfig":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


# Предустановленные конфигурации
def small_config() -> NexusConfig:
    """Облегчённая конфигурация для CPU/CI и быстрых тестов."""
    return NexusConfig(
        d_value=128, n_layers=2, n_heads=4, d_ff=256,
        text_seq_len=128, text_n_pos=128, max_seq_len=128,
        ttt_rank=16, ttt_mem_dim=32,
        n_experts=4, experts_per_token=1, expert_ff=128,
        sde_hidden=32, mc_paths=64, mc_horizon=8,
        orderbook_levels=16, tabular_features=16, graph_node_dim=16,
        text_vocab_size=1024, vocab_size=1024,
    )


def rtx5060_config() -> NexusConfig:
    """Целевая конфигурация MVP для RTX 5060 16 ГБ."""
    return NexusConfig()  # defaults подобраны под спецификацию
