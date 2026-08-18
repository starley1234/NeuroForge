"""
Continuous SDE/SSM encoder биржевого стакана L2/L3 и потока тиков.

Вместо токенизации стакан моделируется как непрерывная динамика:
    dX = (A - A^T) X dt + B dW  (state-space, S4-подобный),
где X — состояние ликвидности, а dW — винеров процесс тиковых ударов.
Это даёт линейную O(L) сложность по числу уровней и естественно
описывает проскальзывание (slippage) и баланс спроса/предложения.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core.layers import ContinuousProjector, RMSNorm


class SSMHiPPO(nn.Module):
    """
    Диагонализованный SSM (S4-подобный) с комплексной HiPPO-матрицей.
    Обеспечивает непрерывное представление последовательности тиков
    с O(T) сложностью и константной памятью на шаг.
    """

    def __init__(self, d_model: int, d_state: int = 64):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # Лог-слои A для устойчивости
        log_A_real = torch.logspace(0, 1.0, d_state).log().neg()
        A_imag = torch.linspace(0.1, 1.0, d_state)
        self.log_A_real = nn.Parameter(log_A_real)
        self.A_imag = nn.Parameter(A_imag)

        self.B = nn.Parameter(torch.randn(d_state) * 0.02)
        self.C = nn.Parameter(torch.randn(d_model, d_state, 2)
                              / d_state ** 0.5)
        self.D = nn.Parameter(torch.ones(d_model))
        self.dt = nn.Parameter(torch.ones(d_model) * 0.01)

    def _kernel(self, L: int, device, dtype):
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag
        dt = F.softplus(self.dt).unsqueeze(-1)       # (d_model, 1)
        dA = torch.exp(A.unsqueeze(0) * dt)           # (d_model, d_state)
        dB = self.B.unsqueeze(0) * dt                 # (d_model, d_state)
        # Свёртка через степенной ряд
        powers = dA.unsqueeze(-1) ** torch.arange(
            L, device=device).reshape(1, 1, -1)
        K = (dB.unsqueeze(-1) * powers).sum(dim=1)    # (d_state, L)
        K = K.to(torch.complex64)
        C = torch.view_as_complex(self.C)             # (d_model, d_state)
        kernel = C @ K                                # (d_model, L)
        return kernel.real

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        # u: (B, T, d_model)
        B, T, D = u.shape
        dt = F.softplus(self.dt)                      # (d_model,)
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag  # (N,)
        dA = torch.exp(A.unsqueeze(0) * dt.unsqueeze(-1))   # (d_model, N)
        dB = self.B.unsqueeze(0) * dt.unsqueeze(-1)         # (d_model, N)
        C = torch.view_as_complex(self.C)                    # (d_model, N)

        h = torch.zeros(B, D, dA.shape[-1], dtype=torch.complex64,
                        device=u.device)
        ys = []
        for t in range(T):
            # input per channel
            inp = u[:, t].unsqueeze(-1)                  # (B, D, 1)
            h = h * dA.unsqueeze(0) + inp * dB.unsqueeze(0)
            y_t = torch.real((h * C.unsqueeze(0)).sum(-1))  # (B, D)
            ys.append(y_t)
        y = torch.stack(ys, dim=1)                        # (B, T, D)
        return y + self.D * u


class OrderBookEncoder(nn.Module):
    """
    Кодирует L2 стакан: массив (bid_prices, bid_sizes, ask_prices, ask_sizes)
    и последовательность тиков в непрерывный латентный вектор R^{d_value}.
    """

    def __init__(self, d_value: int, levels: int = 50,
                 tick_features: int = 8, d_state: int = 64):
        super().__init__()
        self.levels = levels
        # 4 поля стакана × levels + признаки тиков
        book_dim = 4 * levels
        self.book_proj = ContinuousProjector(book_dim, d_value // 2,
                                             use_log=True)
        self.tick_proj = nn.Linear(tick_features, d_value // 2, bias=False)
        self.ssm = SSMHiPPO(d_value // 2, d_state=d_state)
        self.fuse = nn.Sequential(
            RMSNorm(d_value),
            nn.Linear(d_value, d_value),
            nn.SiLU(),
            nn.Linear(d_value, d_value),
        )

    def forward(
        self,
        book: torch.Tensor,
        ticks: torch.Tensor,
    ) -> torch.Tensor:
        """
        book:  (B, 4*levels) — [bid_p, bid_sz, ask_p, ask_sz]
        ticks: (B, T, tick_features)
        return: (B, d_value)
        """
        b_emb = self.book_proj(book)                  # (B, d//2)
        t_emb = self.tick_proj(ticks)                 # (B, T, d//2)
        t_emb = self.ssm(t_emb)                        # (B, T, d//2)
        t_emb = t_emb.mean(dim=1)                      # (B, d//2)
        h = torch.cat([b_emb, t_emb], dim=-1)         # (B, d_value)
        return self.fuse(h)
