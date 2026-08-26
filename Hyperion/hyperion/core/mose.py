"""
MoSE — Mixture of Slimmable Experts (arXiv 2602.06154).

Каждый эксперт имеет ВЛОЖЕННУЮ (slimmable) структуру:
можно выполнить его на различной внутренней ширине без
переобучения. Это даёт непрерывный спектр accuracy/compute.

Помимо выбора экспертов (routing), мы выбираем и ширину
каждого выбранного эксперта (width selection) — зависит от
уверенности роутера: уверен -> шире, не уверен -> уже.
"""

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from hyperion.utils import RMSNorm




class SlimmableExpert(nn.Module):
    """
    Эксперт с вложенными ширинами.

    Внутренняя размерность expert_dim, а вложенные подмодули
    используют первые k*step размерностей для k=1..nested_widths.
    """

    def __init__(self, dim: int, expert_dim: int, nested_widths: int = 3, act=F.silu):
        super().__init__()
        self.dim = dim
        self.nested_widths = nested_widths
        self.step = max(1, math.ceil(expert_dim / nested_widths))
        self.expert_dim = self.step * nested_widths  # округляем вверх до кратности
        self.act = act

        self.w1 = nn.Linear(dim, self.expert_dim, bias=False)
        self.w2 = nn.Linear(self.expert_dim, self.expert_dim, bias=False)
        self.w3 = nn.Linear(self.expert_dim, dim, bias=False)

    def forward(self, x: torch.Tensor, width_idx: int = -1) -> torch.Tensor:
        """width_idx: 0..nested_widths-1 (0 = минимальная ширина)."""
        if width_idx < 0:
            width_idx = self.nested_widths - 1
        d_use = self.step * (width_idx + 1)

        # Обрезаем веса: используем только подпространство первых d_use
        w1 = self.w1.weight[:d_use]
        b1 = self.w1.bias[:d_use] if self.w1.bias is not None else None
        h = F.linear(x, w1, b1)
        h = self.act(h)

        w2 = self.w2.weight[:d_use, :d_use]
        b2 = self.w2.bias[:d_use] if self.w2.bias is not None else None
        h = F.linear(h, w2, b2)
        h = self.act(h)

        w3 = self.w3.weight[:, :d_use]
        return F.linear(h, w3)


class MoSE(nn.Module):
    """
    Mixture of Slimmable Experts блок.

    dim: размерность модели
    n_experts: общее число экспертов
    top_k: активных экспертов на токен
    shared: число всегда-активных (shared) экспертов
    nested_widths: число вложенных ширин
    """

    def __init__(
        self,
        dim: int,
        n_experts: int = 8,
        top_k: int = 2,
        shared: int = 1,
        nested_widths: int = 3,
        expert_mult: float = 2.0,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.dim = dim
        self.n_experts = n_experts
        self.top_k = top_k
        self.shared = shared
        self.nested_widths = nested_widths
        expert_dim = int(dim * expert_mult)

        self.norm = RMSNorm(dim, eps=norm_eps)
        self.router = nn.Linear(dim, n_experts, bias=False)
        self.experts = nn.ModuleList([
            SlimmableExpert(dim, expert_dim, nested_widths) for _ in range(n_experts)
        ])
        self.shared_experts = nn.ModuleList([
            SlimmableExpert(dim, expert_dim, nested_widths) for _ in range(shared)
        ])

    def _select_widths(self, routing_scores: torch.Tensor, topk_idx: torch.Tensor) -> torch.Tensor:
        """
        Выбор ширины по уверенности роутера:
        уверенность = сигмоид(logit) в [0, 1]
        ширина = round((conf - 1/n) * (nested_widths - 1) / (1 - 1/n))
        """
        conf = torch.sigmoid(routing_scores.gather(-1, topk_idx))  # [B*S, k]
        # нормализуем: (conf - min) / (max - min) -> [0, 1]
        conf_norm = (conf - 1.0 / self.n_experts) / (1.0 - 1.0 / self.n_experts + 1e-6)
        widths = torch.clamp(conf_norm * (self.nested_widths - 1), min=0, max=self.nested_widths - 1)
        return widths.round().long()  # [B*S, k]

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        x: [B, S, dim]
        Возвращает: [B, S, dim]
        """
        B, S, D = x.shape
        residual = x
        x = self.norm(x)
        flat = x.view(B * S, D)

        # Shared эксперты (всегда активны на полной ширине)
        shared_out = sum(e(flat) for e in self.shared_experts)

        # Routing
        logits = self.router(flat)  # [B*S, n_experts]
        topk_scores, topk_idx = torch.topk(logits, self.top_k, dim=-1)
        probs = F.softmax(logits, dim=-1)
        weights = probs.gather(-1, topk_idx) / probs.gather(-1, topk_idx).sum(-1, keepdim=True)

        widths = self._select_widths(logits, topk_idx)

        # Выполнение выбранных экспертов на выбранной ширине
        out = torch.zeros_like(flat)
        for i in range(self.top_k):
            idx = topk_idx[:, i]
            w = widths[:, i]
            w_scaled = weights[:, i].unsqueeze(-1)
            expert_out = torch.zeros_like(flat)
            for e_idx in range(self.n_experts):
                mask = idx == e_idx
                if mask.any():
                    expert_out[mask] = self.experts[e_idx](flat[mask], w[mask].max())
            out += w_scaled * expert_out

        out = out + shared_out
        return out.view(B, S, D) + residual
