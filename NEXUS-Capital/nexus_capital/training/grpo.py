"""
Value-RL: Group Relative Policy Optimization (GRPO), адаптированный под
экономические награды.

Награды (согласно дорожной карте):
    +2.0  рост маржи/P&L,
    +1.5  снижение риска/VaR,
    +1.0  точность баланса (Assets = L+E, GP = Rev - COGS),
    -5.0  штраф за дефолт.

GRPO не требует отдельной value-сети: преимущество нормируется внутри
группы сэмплированных решений (group-relative advantage).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def economic_reward(
    pnl: torch.Tensor,
    var: torch.Tensor,
    balance_error: torch.Tensor,
    default: torch.Tensor,
    w_pnl: float = 2.0,
    w_var: float = 1.5,
    w_balance: float = 1.0,
    w_default: float = 5.0,
) -> torch.Tensor:
    """
    pnl:            (G,) P&L решения в долларах (нормированный)
    var:            (G,) VaR решения
    balance_error:  (G,) ошибка балансового тождества (0 = идеально)
    default:        (G,) 1 если дефолт
    """
    pnl_r = torch.tanh(pnl / (pnl.abs().median() + 1e-6)) * w_pnl
    var_r = -torch.tanh(var / (var.abs().median() + 1e-6)) * w_var
    bal_r = torch.exp(-balance_error.abs()) * w_balance
    def_r = -default.float() * w_default
    return pnl_r + var_r + bal_r + def_r


def grpo_loss(
    new_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    clip_eps: float = 0.2,
    kl_coef: float = 0.01,
) -> tuple[torch.Tensor, dict]:
    """
    Стандартный PPO/GRPO clipped objective с KL-штрафом.
    Все тензоры (G,).
    """
    ratio = torch.exp(new_logprobs - old_logprobs)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    policy_loss = -torch.min(unclipped, clipped).mean()
    kl = (old_logprobs - new_logprobs).mean()
    total = policy_loss + kl_coef * kl
    return total, {
        "policy_loss": policy_loss.detach(),
        "kl": kl.detach(),
        "ratio": ratio.mean().detach(),
    }


def group_relative_advantage(rewards: torch.Tensor,
                             group_size: int) -> torch.Tensor:
    """Нормирует награды внутри каждой группы (G/group_size групп)."""
    G = rewards.shape[0]
    rewards = rewards.reshape(-1, group_size)
    mean = rewards.mean(dim=1, keepdim=True)
    std = rewards.std(dim=1, keepdim=True).clamp(min=1e-6)
    adv = (rewards - mean) / std
    return adv.reshape(G)
