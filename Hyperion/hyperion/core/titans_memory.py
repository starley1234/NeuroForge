"""
Titans Neural Long-Term Memory (вдохновлено Google Research, 2025).

Три ветви памяти:
1. Core (рабочая) — обычное внимание на ближнем окне
2. Neural Memory (долговременная) — MLP, обучаемый на инференсе
   через ассоциативную функцию потерь; запись срабатывает
   только при «удивлении» (KL-порог), забывание — через decay
3. Persistent — постоянные обучаемые токены (pretraining knowledge)

Стабильность:
- Ключи/значения нормируются к единичной норме (память хранит
  направления, а не амплитуды) — предотвращает расходимость.
- Матрица памяти M спектрально ограничена (||M|| <= max_norm),
  поэтому чтение v̂ = M·k всегда ограничено.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from hyperion.utils import RMSNorm


class NeuralMemory(nn.Module):
    """
    MLP-память, которая учится на тестовом времени (test-time learning).

    Идея (Titans):
      h_{t+1} = h_t - lr * ∇L(h_t),  где L = ||h_t k_t^T - v_t||²
    Память обновляется «удивлением»: запись происходит, только если
    предсказание отклоняется от наблюдения сильнее порога.
    """

    def __init__(
        self,
        dim: int,
        mem_dim: int,
        depth: int = 2,
        surprise_threshold: float = 0.3,
        decay: float = 0.95,
        lr: float = 0.05,
        max_mem_norm: float = 2.0,
        next_step_assoc: bool = True,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.mem_dim = mem_dim
        self.surprise_threshold = surprise_threshold
        self.decay = decay
        self.lr = lr
        self.max_mem_norm = max_mem_norm
        self.next_step_assoc = next_step_assoc

        # Учимся строить key/value из входа
        self.k_proj = nn.Linear(dim, mem_dim, bias=False)
        self.v_proj = nn.Linear(dim, mem_dim, bias=False)

        # Deep memory: стек MLP-слоёв, обрабатывающих ассоциативный вывод
        layers = []
        d = mem_dim
        for i in range(depth):
            layers.append(nn.Linear(d, d))
            if i < depth - 1:
                layers.append(nn.GELU())
        self.mlp = nn.Sequential(*layers)

        self.out_proj = nn.Linear(mem_dim, dim, bias=False)
        self.norm = RMSNorm(dim, eps=norm_eps)

        # Ассоциативная матрица M (состояние памяти)
        self.register_buffer("M", torch.zeros(mem_dim, mem_dim), persistent=False)

    # ------------------------------------------------------------------
    def _surprise(self, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Относительная мера удивления (масштаб-инвариантная):
        ||pred - v||² / (||v||² + eps)   [k, v уже единичной нормы]
        При M=0, pred=0 -> surprise ≈ 1 (запись происходит).
        Когда память выучила ассоциацию, pred ≈ v -> surprise ≈ 0.
        """
        pred = F.linear(k, self.M.t())            # [B, S, mem]
        denom = v.pow(2).sum(-1, keepdim=True) + 1e-6
        return (pred - v).pow(2).sum(-1, keepdim=True) / denom

    @torch.no_grad()
    def write(self, k: torch.Tensor, v: torch.Tensor) -> None:
        """
        Пакетная запись в память (только «удивляющие» записи).

        next_step_assoc=True: храним ассоциацию «текущий -> следующий»
        (k_t, v_{t+1}) — тогда чтение k_t возвращает v_{t+1},
        что прямо помогает предсказанию следующего токена.
        """
        k = F.normalize(k, dim=-1)
        v = F.normalize(v, dim=-1)
        if self.next_step_assoc and k.shape[1] > 1:
            k_src, v_tgt = k[:, :-1], v[:, 1:]
        else:
            k_src, v_tgt = k, v

        surprise = self._surprise(k_src, v_tgt)
        mask = surprise > self.surprise_threshold

        # Правильный градиент ассоциативной цели L = ||k·M - v||²:
        #   ∂L/∂M = kᵀ(k·M - v)
        pred = F.linear(k_src, self.M.t())            # pred = k·M
        err = pred - v_tgt
        grad = k_src.transpose(-1, -2) @ err          # [B, mem, mem]
        update = grad.mean(0) * mask.float().mean()
        self.M.data = self.decay * self.M.data - self.lr * update

        # Спектральная стабилизация: ||M|| <= max_mem_norm
        n = self.M.norm()
        if n > self.max_mem_norm:
            self.M.data *= self.max_mem_norm / n

    def read(self, k: torch.Tensor) -> torch.Tensor:
        """
        Чтение: v̂ = M·k  (+ рефайнмент MLP по остаточному каналу).

        МЛП добавляет выразительность поверх сырой ассоциативной выборки,
        но НЕ может заглушить её: v̂ = raw + mlp(raw). Даже с
        необученным MLP сырая выборка сохраняется.
        """
        k = F.normalize(k, dim=-1)
        v_raw = F.linear(k, self.M.t())          # сырое ассоциативное чтение
        v = v_raw + self.mlp(v_raw)              # остаточный рефайнмент
        return v

    def reset(self):
        self.M.data.zero_()

    def forward(
        self, x: torch.Tensor, write: bool = True
    ) -> torch.Tensor:
        """
        x: [B, S, dim]
        write=True — обучаться на текущем входе (test-time learning).
        """
        residual = x
        x = self.norm(x)

        k = self.k_proj(x)
        v = self.v_proj(x)
        if write:
            self.write(k, v)
        out = self.read(k)
        return self.out_proj(out) + residual


class TitansMemoryBlock(nn.Module):
    """
    Полный блок памяти (core + neural + persistent).

    Core: локальное (скользящее окно) внимание — рабочая память.
    Neural: долговременная память (см. NeuralMemory).
    Persistent: обучаемые токены, прибавляемые к выходу.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int = 8,
        window_size: int = 512,
        mem_dim: int = 192,
        mem_depth: int = 2,
        surprise_threshold: float = 0.3,
        decay: float = 0.95,
        n_persistent: int = 8,
        dropout: float = 0.0,
        norm_eps: float = 1e-6,
        next_step_assoc: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.n_heads = n_heads
        head_dim = dim // n_heads
        self.head_dim = head_dim

        # Core: локальное внимание
        self.core_q = nn.Linear(dim, dim, bias=False)
        self.core_k = nn.Linear(dim, dim, bias=False)
        self.core_v = nn.Linear(dim, dim, bias=False)
        self.core_o = nn.Linear(dim, dim, bias=False)

        # Persistent tokens
        self.persistent = nn.Parameter(torch.randn(n_persistent, head_dim) * 0.02)
        self.persistent_proj = nn.Linear(head_dim, dim, bias=False)

        # Neural memory
        self.memory = NeuralMemory(
            dim, mem_dim, depth=mem_depth,
            surprise_threshold=surprise_threshold, decay=decay,
            next_step_assoc=next_step_assoc, norm_eps=norm_eps,
        )
        # Нормируем объединённую ветку ПЕРЕД добавлением к residual:
        # без этого сигнал памяти (маленький) тонет в residual-потоке
        # (большом), и модель учится игнорировать память.
        self.branch_norm = RMSNorm(dim, eps=norm_eps)
        self.dropout = dropout

    def _core_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Скользящее окно внимания [B, S, dim] -> [B, S, dim]."""
        B, S, D = x.shape
        q = self.core_q(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.core_k(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.core_v(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # каузальная + оконная маска
        causal = torch.triu(torch.ones(S, S, device=x.device, dtype=torch.bool), diagonal=1)
        window = torch.arange(S, device=x.device).unsqueeze(1) - torch.arange(S, device=x.device).unsqueeze(0)
        outside = window > self.window_size
        mask = causal | outside
        scores = scores.masked_fill(mask, float("-inf"))
        probs = F.softmax(scores, dim=-1)
        out = torch.matmul(probs, v)
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        return self.core_o(out)

    def forward(
        self, x: torch.Tensor, use_memory_write: bool = True, **kwargs
    ) -> torch.Tensor:
        B, S, D = x.shape
        residual = x

        # 1. Core (рабочая память)
        core_out = self._core_attention(x)

        # 2. Persistent: добавляем усреднённые persistent-токены
        pers = self.persistent.mean(0, keepdim=True).expand(B, S, -1)
        pers_out = self.persistent_proj(pers)

        # 3. Neural memory (долговременная)
        mem_out = self.memory(x, write=use_memory_write)

        branch = self.branch_norm(core_out + pers_out + mem_out)
        return residual + branch
