"""Latent Reasoning Workspace — мышление в скрытых векторах.

Вместо 10 000 текстовых токенов CoT состояние h циркулирует k раз через
рекуррентный контур; на каждом шаге:

  1. трансформация состояния «мыслительным» блоком ядра;
  2. (опционально) обращение к FNO-суррогату: мгновенная проверка прочности /
     аэродинамики предполагаемой геометрии;
  3. Adaptive Pondering — халт-голова оценивает уверенность и останавливает
     цикл, как только накопленная вероятность остановки превышает порог.

Стоимость: k · d вместо тысяч токенов; градиент течёт через все шаги.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn

from ..config import ReasoningConfig
from ..layers.common import RMSNorm, SwiGLU


@dataclass
class ReasoningTrace:
    steps: int
    halt_probs: List[float]
    ponder_cost: torch.Tensor
    physics_scores: List[float]

    def to_dict(self):
        return {
            "steps": self.steps,
            "halt_probs": [round(p, 4) for p in self.halt_probs],
            "ponder_cost": float(self.ponder_cost.detach()),
            "physics_scores": [round(s, 4) for s in self.physics_scores],
        }


class LatentReasoner(nn.Module):
    def __init__(self, d_latent: int, cfg: ReasoningConfig):
        super().__init__()
        self.cfg = cfg
        self.norm = RMSNorm(d_latent)
        self.update = nn.GRUCell(d_latent, d_latent)
        self.think = SwiGLU(d_latent, d_latent * 2)
        self.halt = nn.Linear(d_latent, 1)
        self.physics_gate = nn.Linear(1, d_latent)
        nn.init.zeros_(self.physics_gate.weight)
        nn.init.zeros_(self.physics_gate.bias)

    def forward(
        self,
        h: torch.Tensor,
        physics_probe: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        max_steps: Optional[int] = None,
    ) -> Tuple[torch.Tensor, ReasoningTrace]:
        """h: (B, d_latent) — сжатое состояние рабочей области."""
        steps = max_steps or self.cfg.max_steps
        state = h
        acc_halt = torch.zeros(h.shape[0], device=h.device, dtype=h.dtype)
        weighted = torch.zeros_like(h)
        remainder = torch.ones_like(acc_halt)
        halts: List[float] = []
        scores: List[float] = []
        ponder = torch.zeros((), device=h.device, dtype=h.dtype)
        used = 0

        for k in range(steps):
            used = k + 1
            thought = self.think(self.norm(state))
            if physics_probe is not None and self.cfg.use_fno_critic:
                score = physics_probe(state).view(-1, 1).to(state.dtype)
                thought = thought + self.physics_gate(score)
                scores.append(float(score.mean().detach()))
            state = self.update(thought, state)

            p = torch.sigmoid(self.halt(self.norm(state))).squeeze(-1)
            halts.append(float(p.mean().detach()))
            contribution = torch.where(
                (acc_halt + p > self.cfg.halt_threshold) | (k == steps - 1),
                remainder, p,
            )
            weighted = weighted + contribution.unsqueeze(-1) * state
            acc_halt = acc_halt + p
            remainder = (remainder - contribution).clamp(min=0.0)
            ponder = ponder + self.cfg.ponder_cost
            if bool((remainder <= 1e-4).all()) and k + 1 >= self.cfg.min_steps:
                break

        return weighted, ReasoningTrace(used, halts, ponder, scores)
