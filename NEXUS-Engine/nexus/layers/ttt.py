"""Linear Fast-Weights Memory (Test-Time Training).

Реализация уровня 2.1 спецификации:

    M_t = (1 - α_t) · M_{t-1} - η_t · ∇L_t ,
    L_t = ½ ‖ M_{t-1} k_t − v_t ‖²   ⇒   ∇L_t = (M_{t-1} k_t − v_t) k_tᵀ

Память — матрица фиксированного размера (d_v × d_k) на голову, поэтому
стоимость хранения контекста O(1) по длине последовательности (в отличие от
KV-кэша, растущего как O(N)). α_t и η_t предсказываются сетью на каждый токен
(data-dependent gating), что даёт «дельту новизны»: статический шум почти не
меняет память, новое событие — меняет сильно.

Вычисления идут чанками длины L (см. TTTConfig.chunk_size): внутри чанка все
операции векторизованы, между чанками — строгая рекуррентность. Внутри чанка
используется стандартная chunk-wise аппроксимация M_{t-1} ≈ M_0 в члене ошибки
(точная для L = 1, режим потокового инференса).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import TTTConfig
from .common import RMSNorm


@dataclass
class TTTState:
    """Состояние быстрых весов: (B, H, d_v, d_k) + последний вход.

    Размер не зависит от длины контекста — это и есть O(1) память.
    """
    memory: torch.Tensor
    last_x: Optional[torch.Tensor] = None      # (B, 1, d_model), для дельты новизны

    def detach(self) -> "TTTState":
        return TTTState(self.memory.detach(),
                        None if self.last_x is None else self.last_x.detach())

    @property
    def bytes(self) -> int:
        n = self.memory.numel() * self.memory.element_size()
        if self.last_x is not None:
            n += self.last_x.numel() * self.last_x.element_size()
        return n


class FastWeightMemory(nn.Module):
    def __init__(self, d_model: int, cfg: TTTConfig):
        super().__init__()
        self.cfg = cfg
        self.h = cfg.n_heads
        self.dk = cfg.key_dim
        self.dv = cfg.value_dim

        self.norm = RMSNorm(d_model)
        self.to_q = nn.Linear(d_model, self.h * self.dk, bias=False)
        self.to_k = nn.Linear(d_model, self.h * self.dk, bias=False)
        self.to_v = nn.Linear(d_model, self.h * self.dv, bias=False)
        self.to_gate = nn.Linear(d_model, self.h * 2, bias=True)  # α_t, η_t
        self.out = nn.Linear(self.h * self.dv, d_model, bias=False)
        self.out_norm = RMSNorm(self.h * self.dv)

        nn.init.zeros_(self.to_gate.bias)

    # ------------------------------------------------------------------ util
    def init_state(self, batch: int, device=None, dtype=None) -> TTTState:
        device = device or self.to_q.weight.device
        dtype = dtype or self.to_q.weight.dtype
        return TTTState(torch.zeros(batch, self.h, self.dv, self.dk, device=device, dtype=dtype))

    def _split(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        b, t, _ = x.shape
        return x.view(b, t, self.h, dim).transpose(1, 2)  # (B,H,T,D)

    # --------------------------------------------------------------- forward
    def forward(
        self,
        x: torch.Tensor,
        state: Optional[TTTState] = None,
        return_state: bool = False,
    ) -> Tuple[torch.Tensor, Optional[TTTState]]:
        b, t, _ = x.shape
        h = self.norm(x)
        q = F.normalize(self._split(self.to_q(h), self.dk), dim=-1)
        k = F.normalize(self._split(self.to_k(h), self.dk), dim=-1)
        v = self._split(self.to_v(h), self.dv)

        gates = self.to_gate(h).view(b, t, self.h, 2).permute(0, 2, 1, 3)  # (B,H,T,2)
        alpha = self.cfg.base_decay * torch.sigmoid(gates[..., 0])         # забывание
        eta = self.cfg.base_lr * torch.sigmoid(gates[..., 1])              # скорость TTT

        # «Дельта новизны»: статический сигнал почти не пишется в память —
        # это и есть отказ от налога на токенизацию (Tokenization Tax).
        prev = x[:, :-1]
        first = state.last_x if (state is not None and state.last_x is not None) else x[:, :1]
        prev = torch.cat([first.to(x.dtype), prev], dim=1)
        novelty = torch.tanh((x - prev).norm(dim=-1) / math.sqrt(x.shape[-1]))  # (B,T)
        eta = eta * novelty.unsqueeze(1)
        decay = (1.0 - alpha * novelty.unsqueeze(1)).clamp(min=1e-4)       # a_t

        mem = (state or self.init_state(b, x.device, x.dtype)).memory
        outputs = []
        L = max(1, self.cfg.chunk_size)
        for start in range(0, t, L):
            end = min(start + L, t)
            y, mem = self._chunk(
                q[:, :, start:end], k[:, :, start:end], v[:, :, start:end],
                decay[:, :, start:end], eta[:, :, start:end], mem,
            )
            outputs.append(y)

        y = torch.cat(outputs, dim=2)                     # (B,H,T,dv)
        y = y.transpose(1, 2).reshape(b, t, self.h * self.dv)
        y = self.out(self.out_norm(y))
        new_state = TTTState(mem, x[:, -1:].detach() if not self.training else x[:, -1:])
        return (y, new_state) if return_state else (y, None)

    # ------------------------------------------------------------ chunk step
    def _chunk(self, q, k, v, decay, eta, mem):
        """Один чанк: чтение памяти + рекуррентное обновление быстрых весов."""
        # Кумулятивные произведения затуханий (в лог-пространстве — устойчиво).
        log_a = torch.log(decay)                       # (B,H,L)
        cum = torch.cumsum(log_a, dim=-1)              # A_t
        cum_prev = cum - log_a                         # A_{t-1}

        # 1) вклад «старой» памяти в чтение: A_{t-1} · M_0 q_t
        read_old = torch.einsum("bhvk,bhtk->bhtv", mem, q) * torch.exp(cum_prev).unsqueeze(-1)

        # 2) ошибка предсказания памяти и апдейт-векторы u_t = -η_t (M_0 k_t - v_t)
        pred = torch.einsum("bhvk,bhtk->bhtv", mem, k)
        u = -eta.unsqueeze(-1) * (pred - v)

        # 3) внутричанковое каузальное чтение: Σ_{i<t} (A_{t-1}/A_i)(q_t·k_i) u_i
        attn = torch.einsum("bhtk,bhsk->bhts", q, k)
        ratio = torch.exp(cum_prev.unsqueeze(-1) - cum.unsqueeze(-2))  # (B,H,T,S)
        mask = torch.ones_like(attn, dtype=torch.bool).tril(diagonal=-1)
        w = torch.where(mask, attn * ratio, torch.zeros_like(attn))
        read_new = torch.einsum("bhts,bhsv->bhtv", w, u)

        # 4) новое состояние памяти
        total = cum[..., -1]                                           # A_L
        scale = torch.exp(total.unsqueeze(-1) - cum)                   # A_L / A_t
        mem_new = mem * torch.exp(total)[..., None, None] + torch.einsum(
            "bhtv,bhtk->bhvk", u * scale.unsqueeze(-1), k
        )
        return read_old + read_new, mem_new

    # ------------------------------------------------------------- streaming
    @torch.no_grad()
    def step(self, x_t: torch.Tensor, state: TTTState) -> Tuple[torch.Tensor, TTTState]:
        """Строгий O(1) шаг инференса на один токен (точное правило дельты)."""
        y, new_state = self.forward(x_t.unsqueeze(1), state, return_state=True)
        return y.squeeze(1), new_state
