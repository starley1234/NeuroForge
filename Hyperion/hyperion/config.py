"""Конфигурация модели Hyperion."""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any


@dataclass
class HyperionConfig:
    # --- Базовые размерности ---
    dim: int = 768                    # размерность модели
    n_layers: int = 12                # всего слоёв backbone
    vocab_size: int = 32_000          # размер словаря

    # --- MLA (Multi-Head Latent Attention) ---
    mla_heads: int = 12               # число голов внимания
    mla_kv_lora_rank: int = 96        # ранг сжатия KV
    mla_q_lora_rank: Optional[int] = None  # ранг сжатия Q (None = no compression)
    mla_rope_head_dim: int = 32       # размерность RoPE-части

    # --- Mamba-3 (State Space Model) ---
    mamba_state_dim: int = 64         # размерность состояния SSM
    mamba_conv_kernel: int = 4        # kernel свёртки

    # --- Titans Neural Memory ---
    memory_dim: int = 192             # размерность нейронной памяти
    memory_depth: int = 2             # глубина MLP памяти
    memory_surprise_threshold: float = 0.3  # порог KL-удивления
    memory_decay: float = 0.99        # коэффициент затухания

    # --- MoSE (Slimmable Mixture-of-Experts) ---
    mose_experts: int = 8             # число экспертов
    mose_top_k: int = 2               # активных экспертов на токен
    mose_shared_experts: int = 1      # всегда активных экспертов
    mose_nested_widths: int = 3       # число вложенных ширин

    # --- JEPA ---
    jepa_enabled: bool = True         # включать JEPA-предсказатель
    jepa_pred_dim: int = 512          # размерность предсказания

    # --- MTP (Multi-Token Prediction) ---
    mtp_depth: int = 1                # глубина MTP (сколько токенов вперёд)

    # --- Пропорции слоёв ---
    ssm_ratio: float = 0.5            # доля Mamba-3 слоёв
    attn_ratio: float = 0.25          # доля MLA слоёв
    memory_ratio: float = 0.25        # доля Titans Memory слоёв

    # --- Обучение ---
    dropout: float = 0.0
    norm_eps: float = 1e-6
    init_std: float = 0.02

    # --- Вспомогательное ---
    max_seq_len: int = 4096           # макс. длина последовательности
    pad_token_id: int = 0

    def __post_init__(self):
        assert self.ssm_ratio + self.attn_ratio + self.memory_ratio == 1.0, \
            "ssm_ratio + attn_ratio + memory_ratio должны давать 1.0"
        if self.mla_q_lora_rank is None:
            self.mla_q_lora_rank = self.dim // 4

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
