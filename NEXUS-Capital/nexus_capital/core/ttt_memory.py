"""
Linear Fast-Weight Memory (онлайн-адаптация к рыночному режиму).

ВАЖНО о терминологии. Этот слой вдохновлён Test-Time Training
(Sun et al., 2024, «Learning to Learn at Test Time»): скрытое состояние
само является маленькой моделью и обновляется шагом градиентного спуска
по self-supervised ошибке на каждом шаге последовательности.

Данная реализация — облегчённая аппроксимация TTT-Linear, а не точная
реализация Sun et al.:
  * одношаговое обновление (mini-batch size 1) вместо многотокового;
  * обучаемый per-head learning rate вместо отдельного мини-оптимизатора;
  * low-rank факторизация памяти A·B для экономии VRAM;
  * pull-back регуляризация к базовым весам для стабильности.
Полноценный TTT-Linear требует кастомных ядер и параллельной рекуррентности
(см. github.com/test-time-training/ttt-lm). Здесь приоритет —
понятность, O(1) память на тик и возможность стриминга 100k+ событий.

Обновление:
    M_t = (1-α) M_{t-1} - η_t · q_t (M_{t-1}^T q_t - v_t)^T
    v_pred = M_t^T q
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .layers import RMSNorm


class FastWeightMemory(nn.Module):
    """
    Линейная быстрая память с self-contained состоянием M_t и обучаемым
    per-head шагом η_t.
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
        # базовый (медленный) learning rate; фактический обучаемый
        self.log_eta = nn.Parameter(torch.tensor(eta).log())

        if rank is not None and 0 < rank < d_mem:
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

        self.register_buffer("state", None, persistent=False)

    @property
    def eta(self) -> torch.Tensor:
        return self.log_eta.exp().clamp(max=1.0)

    def _base_M(self) -> torch.Tensor:
        return self.A @ self.B if self.factorized else self.M

    def init_state(self, batch: int, device: torch.device,
                   dtype: torch.dtype) -> torch.Tensor:
        M = self._base_M().to(dtype)
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
        k_seq = self.k_proj(x)
        q_seq = self.q_proj(x)
        v_target = target if target is not None else self.v_proj(x)
        eta = self.eta.to(dtype)

        for t in range(T):
            k = k_seq[:, t]
            q = q_seq[:, t]
            v = v_target[:, t]
            # Нормализация q для численной стабильности градиента
            q = F_layer_norm(q)
            v_pred = torch.bmm(M_t.transpose(1, 2),
                               q.unsqueeze(-1)).squeeze(-1)
            outputs.append(self.out(self.norm(v_pred)))
            if update:
                err = v_pred - v
                # Градиент с нормализацией по величине состояния
                grad = q.unsqueeze(-1) * err.unsqueeze(1)
                M_t = (1.0 - self.alpha) * M_t - eta * grad
                M_t = M_t + self.alpha * self._base_M().to(dtype).unsqueeze(0)

        out = torch.stack(outputs, dim=1)
        self.state = M_t.detach()
        if squeeze:
            out = out.squeeze(1)
        return out, M_t


# Лёгкий functional RMS-норм без отдельного модуля в hot-path
def F_layer_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


# Совместимость со старым именем
TTTRegimeMemory = FastWeightMemory


class RegimeClassifier(nn.Module):
    """Классификатор рыночного режима по состоянию M_t."""

    REGIMES = ["trend_up", "trend_down", "sideways", "liquidity_crisis"]

    def __init__(self, d_mem: int, d_value: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_mem * d_value, 256),
            nn.SiLU(),
            nn.Linear(256, len(self.REGIMES)),
        )

    def forward(self, M_t: torch.Tensor) -> torch.Tensor:
        return self.net(M_t.reshape(M_t.shape[0], -1))
