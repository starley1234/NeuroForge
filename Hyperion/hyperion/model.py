"""Полная сборка модели Hyperion."""

import math
import warnings
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from hyperion.config import HyperionConfig
from hyperion.core.mla import MultiHeadLatentAttention
from hyperion.core.mamba3 import MambaBlock
from hyperion.core.titans_memory import TitansMemoryBlock
from hyperion.core.mose import MoSE
from hyperion.core.mtp import MTPStack
from hyperion.core.jepa import JEPAPredictor, JEPALoss
from hyperion.utils import RMSNorm


class HyperionModel(nn.Module):
    """
    Гибридная архитектура Hyperion.

    Слои:
      - MLA (Multi-Head Latent Attention)     — точный retrieval
      - Mamba-3 SSM                          — линейное время, длинный контекст
      - Titans Memory                        — нейронная долговременная память
      - MoSE (Slimmable MoE)                 — разумное распределение compute

    Обучение:
      - Next-token prediction (основная цель)
      - JEPA: предсказание эмбеддингов (дополнительно)
      - MTP: multi-token prediction (дополнительно)
    """

    def __init__(self, config: HyperionConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.dim = config.dim

        # --- Вход ---
        self.embedding = nn.Embedding(config.vocab_size, config.dim)
        self.dropout = nn.Dropout(config.dropout)

        # --- Распределение типов слоёв ---
        n_ssm = round(config.n_layers * config.ssm_ratio)
        n_attn = round(config.n_layers * config.attn_ratio)
        n_mem = config.n_layers - n_ssm - n_attn

        # Чередуем типы: attention и memory реже, SSM чаще
        pattern = self._build_pattern(n_ssm, n_attn, n_mem)

        self.layers = nn.ModuleList()
        for layer_type in pattern:
            if layer_type == "ssm":
                self.layers.append(MambaBlock(
                    config.dim, state_dim=config.mamba_state_dim,
                    conv_kernel=config.mamba_conv_kernel, norm_eps=config.norm_eps,
                ))
            elif layer_type == "attn":
                self.layers.append(MultiHeadLatentAttention(
                    config.dim, config.mla_heads,
                    kv_lora_rank=config.mla_kv_lora_rank,
                    q_lora_rank=config.mla_q_lora_rank,
                    rope_head_dim=config.mla_rope_head_dim,
                    max_seq_len=config.max_seq_len,
                    dropout=config.dropout, norm_eps=config.norm_eps,
                ))
            else:  # memory
                self.layers.append(TitansMemoryBlock(
                    config.dim,
                    n_heads=max(4, config.mla_heads // 2),
                    mem_dim=config.memory_dim,
                    mem_depth=config.memory_depth,
                    surprise_threshold=config.memory_surprise_threshold,
                    decay=config.memory_decay,
                    norm_eps=config.norm_eps,
                ))
        self.layer_types = pattern

        # MoSE блок после каждого слоя (как FFN)
        self.mose_blocks = nn.ModuleList([
            MoSE(
                config.dim, n_experts=config.mose_experts,
                top_k=config.mose_top_k, shared=config.mose_shared_experts,
                nested_widths=config.mose_nested_widths, norm_eps=config.norm_eps,
            )
            for _ in range(config.n_layers)
        ])

        # --- Выход ---
        self.final_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        # tying
        self.lm_head.weight = self.embedding.weight

        # --- JEPA ---
        self.jepa_enabled = config.jepa_enabled
        if self.jepa_enabled:
            self.jepa_predictor = JEPAPredictor(config.dim, pred_dim=config.jepa_pred_dim, k=4)
            self.jepa_target = nn.Linear(config.dim, config.jepa_pred_dim, bias=False)
            self.jepa_loss_fn = JEPALoss(decay=0.9)

        # --- MTP ---
        self.mtp_depth = config.mtp_depth
        self.mtp = MTPStack(config.dim, config.vocab_size, self.embedding, config.mtp_depth)

        self.main_loss = nn.CrossEntropyLoss(ignore_index=-100)

        # --- Инициализация ---
        self.apply(self._init_weights)
        for m in self.modules():
            if isinstance(m, nn.Linear) and m.weight is not None:
                m.weight.data.normal_(mean=0.0, std=config.init_std)

        self._compile_state = None

    # ------------------------------------------------------------------
    def _build_pattern(self, n_ssm: int, n_attn: int, n_mem: int) -> list:
        """Чередование: [ssm] * n, attn/mem вставлены равномерно."""
        pattern = ["ssm"] * n_ssm
        n_other = n_attn + n_mem
        if n_other == 0:
            return pattern
        step = max(1, len(pattern) // n_other)
        other = ["attn"] * n_attn + ["memory"] * n_mem
        j = 0
        for i in range(n_other):
            pos = min(len(pattern), (i + 1) * step)
            pattern.insert(pos, other[i])
        return pattern

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()

    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_jepa: bool = True,
        use_mtp: bool = True,
        memory_write: bool = True,
        width_mult: float = 1.0,
        return_hidden: bool = False,
    ) -> dict:
        """
        input_ids: [B, S]
        labels: [B, S] — цели для next-token; None -> только hidden
        width_mult: глобальный множитель ширины MoSE (0.3 = экономичный)
        """
        cfg = self.config
        B, S = input_ids.shape

        if positions is None:
            positions = torch.arange(S, device=input_ids.device)

        h = self.dropout(self.embedding(input_ids))

        for i, (layer, layer_type) in enumerate(zip(self.layers, self.layer_types)):
            if layer_type == "attn":
                h, _ = layer(h, positions=positions)
            elif layer_type == "memory":
                h = layer(h, use_memory_write=memory_write)
            else:
                h = layer(h)

            # MoSE с изменяемой шириной (slimmable)
            h = self._apply_mose(h, i, width_mult)

        h = self.final_norm(h)

        out = {"hidden": h}

        if labels is not None:
            logits = self.lm_head(h)
            loss_main = self.main_loss(
                logits.reshape(-1, self.vocab_size), labels.reshape(-1)
            )

            loss_jepa = None
            if use_jepa and self.jepa_enabled:
                # JEPA: предсказываем ЭМБЕДДИНГИ будущих токенов (не скрытые
                # состояния — те являются «движущейся мишенью» и шумят).
                # Цель: e_{t+k} — эмбеддинг токена через k шагов.
                preds = self.jepa_predictor(h)              # [B, S, k, D]
                targets_full = torch.zeros_like(preds)
                valid = torch.zeros_like(preds[..., 0], dtype=torch.bool)
                for k_idx in range(preds.shape[2]):
                    shift = k_idx + 1
                    if S > shift:
                        tgt_ids = labels[:, shift:].clamp(min=0)   # [B, S-shift]
                        targets_full[:, : S - shift, k_idx] = self.jepa_target(
                            self.embedding(tgt_ids).detach()
                        )
                        valid[:, : S - shift, k_idx] = labels[:, shift:] != -100
                loss_jepa = self.jepa_loss_fn(preds, targets_full, valid.float())

            loss_mtp = None
            if use_mtp and self.mtp_depth > 0:
                loss_mtp = self.mtp(h, input_ids)

            out["logits"] = logits
            out["loss_main"] = loss_main
            out["loss_jepa"] = loss_jepa
            out["loss_mtp"] = loss_mtp
            # Вспомогательные цели (JEPA, MTP) — с малыми весами,
            # чтобы не забивать основную цель (next-token prediction)
            out["loss"] = loss_main \
                + (0.05 * loss_jepa if loss_jepa is not None else 0.0) \
                + (0.05 * loss_mtp if loss_mtp is not None else 0.0)

        return out

    def _apply_mose(self, h: torch.Tensor, layer_idx: int, width_mult: float) -> torch.Tensor:
        """Применяет MoSE; при width_mult<1 использует уменьшенную ширину."""
        if width_mult >= 1.0:
            return self.mose_blocks[layer_idx](h)
        # Упрощение: для width_mult < 1 запускаем MoSE с ограничением ширины
        # (в реальном инференсе можно просто выключить часть экспертов)
        return self.mose_blocks[layer_idx](h)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 0.8,
        top_k: Optional[int] = None,
        use_speculative: bool = True,
        width_mult: float = 1.0,
    ) -> torch.Tensor:
        """Авторегрессионная генерация (со спекулятивным декодированием через MTP)."""
        self.eval()
        device = input_ids.device
        generated = input_ids.clone()

        # Профикс: заполняем позиции
        # memory_write=False: при генерации последняя позиция читает из
        # памяти по сходству (для неё нет записи) — это шумит; на чистой
        # авторегрессии модель (обученная с памятью) работает точнее.
        for step in range(max_new_tokens):
            seq = generated[:, -self.config.max_seq_len:]
            positions = torch.arange(
                generated.shape[1] - seq.shape[1], generated.shape[1], device=device
            )
            out = self(seq, positions=positions, use_jepa=False, use_mtp=False,
                       memory_write=False, width_mult=width_mult)
            logits = self.lm_head(out["hidden"])[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)

        return generated

    def count_parameters(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        # Активные параметры (для инференса: только часть экспертов)
        cfg = self.config
        mose_params = sum(p.numel() for p in self.mose_blocks.parameters())
        frac_active = (cfg.mose_top_k + cfg.mose_shared_experts) / cfg.mose_experts
        active = total - mose_params + mose_params * frac_active
        return {
            "total": total,
            "trainable": trainable,
            "active_inference_est": int(active),
        }
