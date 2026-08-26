"""
Financial Sparse Mixture-of-Experts.

8 специализированных экспертов:
    [Risk/VaR, Pricing & Unit-Econ, Macro/FX, Legal/Compliance,
     Credit/Default, Liquidity, Derivatives, Sentiment]

На каждый токен маршрутизатор (Top-K gating) выбирает `experts_per_token`
(по умолчанию 2) эксперта. Это позволяет держать суммарно 3.5B весов MoE,
но на каждый форвард-проход активировать лишь ~2 из них — укладываясь
в бюджет 16 ГБ VRAM.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import SwiGLU, RMSNorm


class FinancialExpert(nn.Module):
    """Один экономический эксперт — SwiGLU + нормализация."""

    def __init__(self, d_value: int, d_ff: int, name: str = "expert"):
        super().__init__()
        self.name = name
        self.ffn = SwiGLU(d_value, d_ff)
        self.norm = RMSNorm(d_value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.ffn(x))


class FinancialSparseMoE(nn.Module):
    """
    Top-k разреженный MoE с нагрузочным балансировочным штрафом
    (load-balancing loss) и маршрутизацией по важности.
    """

    def __init__(
        self,
        d_value: int,
        n_experts: int = 8,
        experts_per_token: int = 2,
        d_ff: int = 2048,
        expert_names: list[str] | None = None,
        jitter_noise: float = 0.0,
    ):
        super().__init__()
        self.d_value = d_value
        self.n_experts = n_experts
        self.k = experts_per_token
        self.jitter_noise = jitter_noise

        default_names = ["risk_var", "pricing_unit_econ", "macro_fx",
                         "legal_compliance", "credit_default", "liquidity",
                         "derivatives", "sentiment"]
        names = expert_names or default_names
        if len(names) < n_experts:
            names = names + [f"expert_{i}" for i in range(len(names), n_experts)]
        names = names[:n_experts]
        self.experts = nn.ModuleList([
            FinancialExpert(d_value, d_ff, name=names[i])
            for i in range(n_experts)
        ])
        self.gate = nn.Linear(d_value, n_experts, bias=False)
        self.shared = SwiGLU(d_value, d_ff)  # общий лёгкий эксперт
        self.norm = RMSNorm(d_value)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, T, d_value)
        Возвращает:
            out: (B, T, d_value)
            aux_loss: скаляр балансировочного штрафа для добавления в L_total
        """
        B, T, D = x.shape
        flat = x.reshape(-1, D)
        N = flat.shape[0]

        if self.training and self.jitter_noise > 0:
            flat = flat * (1.0 + self.jitter_noise * torch.randn_like(flat))

        logits = self.gate(flat)                    # (N, E)
        probs = F.softmax(logits, dim=-1)

        topk_p, topk_idx = probs.topk(self.k, dim=-1)  # (N, k)
        topk_p = topk_p / topk_p.sum(dim=-1, keepdim=True)

        out = torch.zeros_like(flat)

        # Векторизованный вызов экспертов по корзинам
        for e in range(self.n_experts):
            mask = (topk_idx == e)                  # (N, k)
            if not mask.any():
                continue
            # Все токены, выбравшие этого эксперта (с учётом k позиций)
            rows = mask.nonzero(as_tuple=False)
            token_ids = rows[:, 0]
            weights = topk_p[token_ids, rows[:, 1]]  # (M,)
            x_e = flat[token_ids]                     # (M, D)
            y_e = self.experts[e](x_e) * weights.unsqueeze(-1)
            out.index_add_(0, token_ids, y_e)

        out = out.reshape(B, T, D)
        # Общий остаточный путь
        out = out + self.shared(x)
        out = self.norm(out + x)

        # Load-balancing aux loss (Switch Transformer):
        # f_e = доля токенов у эксперта; P_e = средняя вероятность гейта
        with torch.no_grad():
            one_hot = F.one_hot(topk_idx, num_classes=self.n_experts).float()
            tokens_per_expert = one_hot.sum(dim=(0, 1)) / (N * self.k)
        mean_prob = probs.mean(dim=0)
        aux_loss = (self.n_experts *
                    (tokens_per_expert * mean_prob).sum())

        return out, aux_loss
