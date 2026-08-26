"""
Unified Value Bus — единое непрерывное латентное экономическое пространство
R^{d_value}.

Все 5 классов модальностей проецируются сюда. Пространство поддерживает
инварианты:
    • CF  — денежные потоки (лог-амплитуда)
    • sigma — векторы риска/волатильности
    • e^{-rt} — дисконтный фактор (временная стоимость денег)
    • E_d — эластичность спроса
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import NexusConfig
from .layers import RMSNorm
from .ttt_memory import FastWeightMemory
from .attention import LocalFinancialAttention
from .moe import FinancialSparseMoE


class EconomicInvariantHead(nn.Module):
    """
    Считывает из латентного состояния физико-экономические инварианты:
    денежный поток CF, волатильность sigma, дисконт-фактор и эластичность.
    Используется как вспомогательный обучающий сигнал и как вход в
    функцию полезности / VaR.
    """

    def __init__(self, d_value: int):
        super().__init__()
        self.cf = nn.Linear(d_value, 1)         # свободный денежный поток
        self.sigma = nn.Linear(d_value, 1)      # волатильность (>0)
        self.discount = nn.Linear(d_value, 1)   # ставка дисконтирования
        self.elasticity = nn.Linear(d_value, 1) # эластичность спроса

    def forward(self, h: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "cf": self.cf(h).squeeze(-1),
            "sigma": F.softplus(self.sigma(h)).squeeze(-1),
            "discount": torch.sigmoid(self.discount(h)).squeeze(-1),
            "elasticity": self.elasticity(h).squeeze(-1),
        }


class NexusCoreBlock(nn.Module):
    """Один блок ядра: TTT -> Local Attention -> MoE."""

    def __init__(self, cfg: NexusConfig, block_id: int):
        super().__init__()
        self.block_id = block_id
        self.fast_memory = FastWeightMemory(
            d_value=cfg.d_value,
            d_mem=cfg.ttt_mem_dim,
            rank=cfg.ttt_rank,
            alpha=cfg.ttt_alpha,
            eta=cfg.ttt_eta,
        )
        self.attn = LocalFinancialAttention(
            d_value=cfg.d_value,
            n_heads=cfg.n_heads,
            window=cfg.text_seq_len,
            dropout=cfg.dropout,
        )
        self.moe = FinancialSparseMoE(
            d_value=cfg.d_value,
            n_experts=cfg.n_experts,
            experts_per_token=cfg.experts_per_token,
            d_ff=cfg.expert_ff,
            expert_names=cfg.expert_names,
        )
        self.norm1 = RMSNorm(cfg.d_value)
        self.norm2 = RMSNorm(cfg.d_value)
        self.norm3 = RMSNorm(cfg.d_value)

    def forward(
        self, x: torch.Tensor, target: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # TTT быстрая память рыночного режима
        h = self.norm1(x)
        ttt_out, state = self.fast_memory(h, target=target)
        x = x + ttt_out

        # Локальное финансовое внимание
        h = self.norm2(x)
        x = x + self.attn(h)

        # Sparse MoE
        h = self.norm3(x)
        moe_out, aux_loss = self.moe(h)
        x = x + moe_out

        return x, {"aux_loss": aux_loss, "ttt_state": state}


class UnifiedValueBus(nn.Module):
    """
    Стек из n_layers блоков ядра. Принимает уже спроецированные в R^{d_value}
    эмбеддинги модальностей и возвращает латентное состояние + инварианты.
    """

    def __init__(self, cfg: NexusConfig):
        super().__init__()
        self.cfg = cfg
        self.blocks = nn.ModuleList([
            NexusCoreBlock(cfg, i) for i in range(cfg.n_layers)
        ])
        self.final_norm = RMSNorm(cfg.d_value)
        self.invariants = EconomicInvariantHead(cfg.d_value)
        # Позиционное кодирование для дискретных токенов текста
        pe = torch.zeros(cfg.text_n_pos, cfg.d_value)
        pos = torch.arange(0, cfg.text_n_pos).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, cfg.d_value, 2).float()
                        * (-math.log(10000.0) / cfg.d_value))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pos_emb", pe, persistent=False)

    def forward(
        self,
        h: torch.Tensor,
        target: torch.Tensor | None = None,
        use_pos: bool = True,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        B, T, D = h.shape
        if use_pos and T <= self.cfg.text_n_pos:
            h = h + self.pos_emb[:T].unsqueeze(0)

        aux_losses = []
        states = []
        for block in self.blocks:
            h, info = block(h, target=target)
            aux_losses.append(info["aux_loss"])
            states.append(info["ttt_state"])

        h = self.final_norm(h)
        inv = self.invariants(h)
        total_aux = torch.stack(aux_losses).mean() if aux_losses else h.new_zeros(())
        return h, {
            "invariants": inv,
            "aux_loss": total_aux,
            "ttt_states": states,
        }

    def reset_ttt_state(self) -> None:
        for block in self.blocks:
            block.fast_memory.reset_state()
