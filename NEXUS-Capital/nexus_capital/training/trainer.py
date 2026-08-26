"""
Базовый тренер NEXUS-Capital.

Поддерживает:
  • обучение ядра на синтетических данных (предобучение фазы 3);
  • стриминг тиков с TTT;
  • Value-RL (GRPO) на P&L (фаза 4);
  • смешанную точность и градиентный клиппинг.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn

from ..models.nexus_model import NexusCapital
from ..core.config import NexusConfig
from .grpo import grpo_loss, group_relative_advantage, economic_reward


@dataclass
class TrainConfig:
    lr: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    batch_size: int = 4
    steps: int = 100
    log_every: int = 10
    device: str = "cpu"
    use_amp: bool = False
    # GaLore / 8-bit placeholder (реализуется через bitsandbytes при наличии)
    use_8bit: bool = False


class NexusTrainer:
    def __init__(self, model: NexusCapital, tcfg: TrainConfig | None = None):
        self.model = model
        self.cfg = tcfg or TrainConfig()
        self.device = torch.device(self.cfg.device)
        self.model.to(self.device)
        self.opt = self._build_optimizer()
        self.scaler = torch.cuda.amp.GradScaler(
            enabled=self.cfg.use_amp and self.device.type == "cuda")
        self.step_count = 0

    def _build_optimizer(self):
        if self.cfg.use_8bit:
            try:
                import bitsandbytes as bnb  # type: ignore
                return bnb.optim.AdamW8bit(
                    self.model.parameters(), lr=self.cfg.lr,
                    weight_decay=self.cfg.weight_decay)
            except ImportError:
                pass  # fallback
        return torch.optim.AdamW(
            self.model.parameters(), lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay)

    def pretrain_step(self, batch: dict) -> dict:
        """
        Один шаг предобучения. Ожидает ключи, совместимые с
        NexusCapital.forward. Минимально — fields/tokens + target_tokens.
        """
        self.model.train()
        batch = {k: v.to(self.device) if torch.is_tensor(v) else v
                 for k, v in batch.items()}
        self.opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(
            enabled=self.cfg.use_amp and self.device.type == "cuda",
            dtype=torch.bfloat16,
        ):
            out = self.model(**batch)
            loss = out.get("loss")
            if loss is None:
                raise ValueError("Batch must include target_tokens for pretrain")
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.opt)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                       self.cfg.grad_clip)
        self.scaler.step(self.opt)
        self.scaler.update()
        self.step_count += 1
        return {k: float(v) for k, v in out["loss_logs"].items()}

    @torch.no_grad()
    def stream_evaluate(self, tick_batch: dict) -> list[dict]:
        self.model.eval()
        return self.model.stream_ticks(
            tick_batch["book"].to(self.device),
            tick_batch["ticks"].to(self.device),
        )

    def grpo_step(
        self,
        contexts: list[dict],
        old_logprobs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        group_size: int = 4,
    ) -> dict:
        """
        Один шаг GRPO. actions — индексы решений (цены/сделки).
        contexts передаются в модель для получения новых логитов.
        """
        self.model.train()
        advantages = group_relative_advantage(rewards, group_size)
        advantages = advantages.to(self.device)
        old_logprobs = old_logprobs.to(self.device)

        self.opt.zero_grad(set_to_none=True)
        # В реальном сценарии actions сэмплируются из политики;
        # здесь используем контекст для получения логитов.
        new_logprobs = []
        for ctx in contexts:
            ctx = {k: v.to(self.device) if torch.is_tensor(v) else v
                   for k, v in ctx.items()}
            out = self.model(**ctx)
            logp = torch.log_softmax(out["logits"][:, -1, :], dim=-1)
            # Индекс действия берётся как первый доступный
            new_logprobs.append(logp[:, actions[:logp.shape[0]]].diag())
        new_logprobs = torch.cat(new_logprobs)
        loss, logs = grpo_loss(new_logprobs, old_logprobs, advantages)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                       self.cfg.grad_clip)
        self.opt.step()
        self.step_count += 1
        return {k: float(v) for k, v in logs.items()}
