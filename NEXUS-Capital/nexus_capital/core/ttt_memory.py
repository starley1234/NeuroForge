"""
Linear Fast-Weight Memory (Test-Time Training) для адаптации к рыночному режиму.

Идея: рыночный режим (тренд/кризис/боковик) меняется за часы, а базовые веса
модели обучаются на месяцах данных. Вместо переобучения вводится быстрая
матрица M_t, которая обновляется на лету градиентом от ошибки рыночного
прогноза:

    M_t = (1 - alpha_t) M_{t-1} - eta_t ∇_M || M_{t-1} k_market - v_spread ||^2

При линейном выходе v = M k градиент имеет замкнутую форму:

    ∇_M ||M k - v||^2 = 2 (M k - v) k^T

Поэтому обновление — O(d_mem * d_value) без обратного прохода графа и
с константной памятью на тик. Это позволяет стримить 100k+ событий без
роста VRAM.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import RMSNorm


class TTTRegimeMemory(nn.Module):
    """
    Линейная быстрая память с self-contained состоянием M_t.

    Параметры:
        d_value: размерность латентного пространства.
        d_mem:   внутренняя размерность памяти (ранг M).
        rank:    если задан, M факторизуется как A B (low-rank) для экономии.
        alpha:   коэффициент затухания старого состояния.
        eta:     шаг быстрого градиента (learning rate на инфлайте).
    """

    def __init__(
        self,
        d_value: int,
        d_mem: int = 256,
        rank: int | None = 128,
        alpha: float = 0.95,
        eta: float = 0.05,
    ):
        super().__init__()
        self.d_value = d_value
        self.d_mem = d_mem
        self.rank = rank
        self.alpha = alpha
        self.eta = eta

        if rank is not None and rank < d_mem:
            # Факторизованная память M ≈ A B
            self.A = nn.Parameter(torch.randn(d_mem, rank) * 0.02)
            self.B = nn.Parameter(torch.randn(rank, d_value) * 0.02)
            self.factorized = True
        else:
            self.M = nn.Parameter(torch.randn(d_mem, d_value) * 0.02)
            self.factorized = False

        self.k_proj = nn.Linear(d_value, d_mem, bias=False)
        self.v_proj = nn.Linear(d_value, d_value, bias=False)
        self.q_proj = nn.Linear(d_value, d_mem, bias=False)
        self.out = nn.Linear(d_value, d_value, bias=False)
        self.norm = RMSNorm(d_value)

        # Регистрируем ненулевой буфер состояния (M_t); инициализируется нулём
        self.register_buffer("state", None, persistent=False)

    def _base_M(self) -> torch.Tensor:
        if self.factorized:
            return self.A @ self.B
        return self.M

    def init_state(self, batch: int, device: torch.device,
                   dtype: torch.dtype) -> torch.Tensor:
        M = self._base_M().to(dtype)
        # M: (d_mem, d_value) -> (batch, d_mem, d_value)
        return M.unsqueeze(0).expand(batch, -1, -1).clone()

    def reset_state(self) -> None:
        self.state = None

    def forward(
        self,
        x: torch.Tensor,
        target: torch.Tensor | None = None,
        update: bool = True,
        state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x:      (B, T, d_value) или (B, d_value)
        target: (B, T, d_value) — целевое значение v_spread для TTT-обновления.
                Если None, в качестве цели используется проекция самого x
                (self-supervised рыночный тик).
        update: выполнять ли TTT-обновление памяти.
        state:  внешнее состояние (B, d_mem, d_value). Если None,
                используется внутреннее с инициализацией.

        Возвращает:
            out:   (B, T, d_value)
            state: обновлённое быстрейшее состояние M_t
        """
        squeeze = False
        if x.dim() == 2:
            x = x.unsqueeze(1)
            if target is not None:
                target = target.unsqueeze(1)
            squeeze = True

        B, T, D = x.shape
        device, dtype = x.device, x.dtype

        if state is None:
            if self.state is None or self.state.shape[0] != B:
                state = self.init_state(B, device, dtype)
            else:
                state = self.state

        M_t = state
        outputs = []

        k_seq = self.k_proj(x)   # (B, T, d_mem)
        q_seq = self.q_proj(x)   # (B, T, d_mem)
        if target is None:
            v_target = self.v_proj(x)  # self-supervised цель
        else:
            v_target = target

        for t in range(T):
            k = k_seq[:, t]               # (B, d_mem)
            q = q_seq[:, t]               # (B, d_mem)
            v = v_target[:, t]            # (B, d_value)

            # Считывание: v_pred = M_t^T q, где M_t (B, d_mem, d_value)
            v_pred = torch.bmm(M_t.transpose(1, 2),
                               q.unsqueeze(-1)).squeeze(-1)     # (B, d_value)
            out_t = self.out(self.norm(v_pred))
            outputs.append(out_t)

            if update:
                # Замкнутый градиент: ∇_M ||M^T q - v||^2 = q (M^T q - v)^T
                err = v_pred - v                                   # (B, d_value)
                # M_t: (B, d_mem, d_value); q: (B, d_mem)
                # err^T по dim-value: (B, d_value)
                grad = q.unsqueeze(-1) * err.unsqueeze(1)          # (B, d_mem, d_value)
                M_t = (1.0 - self.alpha) * M_t - self.eta * grad
                # Дополнительно лёгкая pull-back регуляризация к базовым весам,
                # чтобы быстрая память не улетала на бесконечность.
                M_t = M_t + self.alpha * self._base_M().to(dtype).unsqueeze(0)

        out = torch.stack(outputs, dim=1)
        if self.state is None or not self.training:
            self.state = M_t.detach()
        else:
            self.state = M_t.detach()

        if squeeze:
            out = out.squeeze(1)
        return out, M_t


class RegimeClassifier(nn.Module):
    """
    Лёгкий классификатор рыночного режима по состоянию M_t.
    Используется для индикации: 'trend_up', 'trend_down',
    'sideways', 'liquidity_crisis'.
    """

    REGIMES = ["trend_up", "trend_down", "sideways", "liquidity_crisis"]

    def __init__(self, d_mem: int, d_value: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_mem * d_value, 256),
            nn.SiLU(),
            nn.Linear(256, len(self.REGIMES)),
        )

    def forward(self, M_t: torch.Tensor) -> torch.Tensor:
        flat = M_t.reshape(M_t.shape[0], -1)
        return self.net(flat)
