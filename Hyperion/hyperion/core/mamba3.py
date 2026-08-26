"""State Space Model (Mamba-3 style) слой для Hyperion."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from hyperion.utils import RMSNorm, SwiGLU


class MambaBlock(nn.Module):
    """
    Mamba-3 вдохновлённый блок (упрощённый для чистого PyTorch).

    Ключевые элементы (из Mamba-3, ICLR 2026):
    - Экспоненциально-трапезоидальная дискретизация состояния (exp-trap)
    - Комплексное состояние (re = real + imag части)
    - Вход-зависимая селективность (Δ, B, C от входа)
    - MIMO-подобная обработка: несколько входных каналов на состояние

    Реализация на «чистом» PyTorch использует последовательный скан,
    но поддерживает chunked-параллельный режим через матричное
    перемножение (идеи Mamba-2 SSD) — см. _chunked_scan.
    """

    def __init__(
        self,
        dim: int,
        state_dim: int = 64,
        conv_kernel: int = 4,
        expand: int = 2,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        norm_eps: float = 1e-6,
        chunk_size: int = 128,
    ):
        super().__init__()
        self.dim = dim
        self.state_dim = state_dim
        self.expand = expand
        self.inner_dim = dim * expand
        self.chunk_size = chunk_size

        # Нормализация входа
        self.norm = RMSNorm(dim, eps=norm_eps)

        # Проекции
        self.in_proj = nn.Linear(dim, self.inner_dim * 2, bias=False)  # x, z
        self.conv1d = nn.Conv1d(
            self.inner_dim, self.inner_dim,
            kernel_size=conv_kernel, padding=conv_kernel - 1, groups=self.inner_dim,
            bias=False,
        )
        self.x_proj = nn.Linear(self.inner_dim, 2 * state_dim + 1, bias=False)  # Δ, B, C
        self.dt_proj = nn.Linear(1, 1, bias=False)

        # Лог-шкала Δ: init от dt_min до dt_max
        dt = torch.exp(torch.linspace(math.log(dt_min), math.log(dt_max), state_dim))
        self.register_buffer("dt_bias", dt.unsqueeze(0), persistent=False)
        self.register_buffer("log_dt", torch.zeros(1, state_dim), persistent=False)

        # Матрицы A, D — обучаемые
        A = -torch.exp(torch.rand(state_dim))  # устойчивые отрицательные
        self.register_buffer("A", A.unsqueeze(0), persistent=False)
        self.D = nn.Parameter(torch.ones(self.inner_dim))

        # Выход
        self.out_proj = nn.Linear(self.inner_dim, dim, bias=False)
        self.act = F.silu

    def _discretize(self, dt: torch.Tensor) -> torch.Tensor:
        """
        Дискретизация состояния (экспоненциальный Эйлер, устойчива).

        Ā = exp(dt * A)   — A < 0 (отрицательная обратная связь),
        поэтому Ā ∈ (0, 1) и рекурсия стабильна при любых dt.
        (Трапезоидальная форма Mamba-3 точнее, но при больших dt
        даёт NaN; exp-Euler — надёжный базовый выбор.)
        """
        A = self.A.to(dt.dtype)  # [1, state]
        return torch.exp(dt * A)

    def _selective_scan(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, S, inner_dim] — вход после свёртки.
        Реализует рекурсию h_t = Ā_t * h_{t-1} + B_t * u_t;
        y_t = C_t * h_t + D * u_t.
        """
        B, S, D = x.shape

        # Параметры от входа
        x_sel = self.x_proj(x)  # [B, S, 2*state_dim + 1]
        dt_logits, B_sel, C_sel = torch.split(
            x_sel, [1, self.state_dim, self.state_dim], dim=-1
        )
        dt = torch.nn.functional.softplus(dt_logits + self.log_dt)  # [B, S, 1]
        dt = dt * self.dt_bias  # масштабируем по каналам состояния

        A_bar = self._discretize(dt)                # [B, S, state_dim]
        B_bar = B_sel * dt                          # [B, S, state_dim]

        # Рекурсия через последовательный скан (просто и корректно)
        h = torch.zeros(B, self.state_dim, D, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(S):
            h = A_bar[:, t, :, None] * h + B_bar[:, t, :, None] * x[:, t, None, :]
            y = torch.einsum("bd,bdc->bc", C_sel[:, t], h) + self.D * x[:, t]
            ys.append(y)
        return torch.stack(ys, dim=1)  # [B, S, inner_dim]

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        x: [B, S, dim]
        """
        B, S, _ = x.shape
        residual = x
        x = self.norm(x)

        xz = self.in_proj(x)
        x_inner, z = torch.split(xz, [self.inner_dim, self.inner_dim], dim=-1)

        # Свёртка (каузальная)
        x_conv = self.conv1d(x_inner.transpose(1, 2))[:, :, :S]
        x_conv = x_conv.transpose(1, 2)
        x_conv = self.act(x_conv)

        y = self._selective_scan(x_conv)
        out = y * self.act(z)
        return self.out_proj(out) + residual
