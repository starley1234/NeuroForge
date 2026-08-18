"""Fine-Grained Sparse MoE: 1 общий эксперт + top-k активных из N.

Мелкозернистые эксперты (hidden = 0.5·d) дают низкий FLOPs при большой
суммарной ёмкости весов: 32 эксперта × 0.5 hidden ≈ 3.6B весов при 1.3B
активных параметров на токен.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import MoEConfig
from .common import RMSNorm, SwiGLU


@dataclass
class MoEStats:
    aux_loss: torch.Tensor
    entropy: torch.Tensor
    expert_load: torch.Tensor


class SparseMoE(nn.Module):
    def __init__(self, d_model: int, cfg: MoEConfig):
        super().__init__()
        self.cfg = cfg
        hidden = max(8, int(d_model * cfg.expert_hidden_mult))
        self.norm = RMSNorm(d_model)
        self.router = nn.Linear(d_model, cfg.n_experts, bias=False)
        self.experts = nn.ModuleList(SwiGLU(d_model, hidden) for _ in range(cfg.n_experts))
        self.shared = nn.ModuleList(SwiGLU(d_model, hidden) for _ in range(cfg.n_shared))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, MoEStats]:
        b, t, d = x.shape
        h = self.norm(x)
        flat = h.reshape(-1, d)

        logits = self.router(flat)                                  # (N, E)
        probs = F.softmax(logits, dim=-1)
        topv, topi = torch.topk(probs, self.cfg.n_active, dim=-1)
        topv = topv / topv.sum(dim=-1, keepdim=True)

        # под автокастом эксперты возвращают half/bfloat16 — держим один dtype
        sample_dtype = self.experts[0].w_down.weight.dtype if self.experts else flat.dtype
        with torch.no_grad():
            probe_dtype = torch.promote_types(flat.dtype, sample_dtype)
        out = torch.zeros(flat.shape, dtype=probe_dtype, device=flat.device)
        flat = flat.to(probe_dtype)
        for e, expert in enumerate(self.experts):
            hit = (topi == e)
            if not hit.any():
                continue
            tok = hit.any(dim=-1).nonzero(as_tuple=True)[0]
            w = (topv * hit).sum(dim=-1)[tok].unsqueeze(-1)
            out.index_add_(0, tok, (expert(flat[tok]) * w).to(out.dtype))

        for expert in self.shared:
            out = out + expert(flat).to(out.dtype)

        # Auxiliary losses: балансировка нагрузки + z-loss стабильности роутера.
        load = torch.zeros(self.cfg.n_experts, device=x.device, dtype=probs.dtype)
        load.index_add_(0, topi.reshape(-1), torch.ones_like(topi.reshape(-1), dtype=probs.dtype))
        frac_tokens = load / load.sum().clamp(min=1)
        frac_prob = probs.mean(dim=0)
        balance = self.cfg.n_experts * (frac_tokens * frac_prob).sum()
        z_loss = torch.logsumexp(logits, dim=-1).pow(2).mean()
        entropy = -(probs.clamp_min(1e-9).log() * probs).sum(-1).mean()
        aux = self.cfg.load_balance_loss * balance + self.cfg.router_z_loss * z_loss

        return out.view(b, t, d).to(x.dtype), MoEStats(aux, entropy.detach(),
                                                      frac_tokens.detach())

    @property
    def active_param_fraction(self) -> float:
        return (self.cfg.n_active + self.cfg.n_shared) / (self.cfg.n_experts + self.cfg.n_shared)
