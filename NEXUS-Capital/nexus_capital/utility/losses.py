"""
Совокупная функция потерь NEXUS-Capital с прямой оптимизацией полезности:

    L_total = L_pred - λ1 · Utility(x) + λ2 · VaR_α(x)
            + L_balance + L_aux

где L_pred — предсказательная потеря (NLL по токенам или MSE по числам),
Utility — функция полезности фон Неймана-Моргенштерна (максимизируем,
        поэтому знак минус),
VaR    — Value-at-Risk (штраф за хвостовой риск),
L_balance — вспомогательная потеря на инвариантах (CF, sigma, discount,
            эластичность) и на корректности баланса,
L_aux  — балансировка экспертов MoE и прочие мелочи.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utility_layer import expected_utility


class NexusLoss(nn.Module):
    def __init__(
        self,
        lambda_utility: float = 1.0,
        lambda_var: float = 0.5,
        lambda_balance: float = 0.5,
        gamma_risk: float = 1.0,
        alpha: float = 0.05,
    ):
        super().__init__()
        self.lambda_utility = lambda_utility
        self.lambda_var = lambda_var
        self.lambda_balance = lambda_balance
        self.gamma_risk = gamma_risk
        self.alpha = alpha

    def predictive_loss(
        self, logits: torch.Tensor, targets: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Кроссэнтропия по токенам дискретного вывода."""
        V = logits.shape[-1]
        loss = F.cross_entropy(
            logits.reshape(-1, V), targets.reshape(-1), reduction="none"
        ).reshape_as(targets)
        if mask is not None:
            loss = (loss * mask).sum() / mask.sum().clamp(min=1)
        else:
            loss = loss.mean()
        return loss

    def numeric_loss(
        self, pred: torch.Tensor, target: torch.Tensor,
    ) -> torch.Tensor:
        """Лог-чувствительная MSE для непрерывных экономических величин."""
        pred_log = torch.sign(pred) * torch.log1p(pred.abs())
        target_log = torch.sign(target) * torch.log1p(target.abs())
        return F.mse_loss(pred_log, target_log)

    def balance_loss(
        self, pred_fields: dict[str, torch.Tensor],
        target_fields: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Проверка балансового тождества:
            Assets = Liabilities + Equity,
            Gross Profit = Revenue - COGS.
        Оба словаря содержат одномерные тензоры (B,).
        """
        loss = pred_fields.new_zeros(())
        if ("assets" in pred_fields and "liabilities" in pred_fields
                and "equity" in pred_fields):
            lhs = pred_fields["assets"]
            rhs = target_fields["liabilities"] + target_fields["equity"]
            loss = loss + self.numeric_loss(lhs, rhs)
        if ("revenue" in pred_fields and "cogs" in pred_fields
                and "gross_profit" in pred_fields):
            lhs = pred_fields["gross_profit"]
            rhs = pred_fields["revenue"] - pred_fields["cogs"]
            tgt = (target_fields["revenue"] - target_fields["cogs"]
                   if "gross_profit" not in target_fields
                   else target_fields["gross_profit"])
            loss = loss + self.numeric_loss(lhs, tgt)
        return loss

    def forward(
        self,
        pred_loss: torch.Tensor,
        returns: torch.Tensor | None = None,
        cost: torch.Tensor | float = 0.0,
        var: torch.Tensor | None = None,
        balance: torch.Tensor | None = None,
        aux_loss: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        utility = returns.new_zeros(()) if returns is not None else pred_loss.new_zeros(())
        var_term = pred_loss.new_zeros(())

        if returns is not None:
            utility = expected_utility(returns, cost=cost, gamma=self.gamma_risk)
            if var is None:
                # VaR из эмпирического распределения
                k = max(1, int(self.alpha * returns.shape[-1]))
                sorted_r, _ = returns.sort(dim=-1)
                var = -sorted_r[..., k - 1].mean()
            var_term = var.mean() if torch.is_tensor(var) else torch.tensor(
                var, device=pred_loss.device, dtype=pred_loss.dtype)

        total = pred_loss
        total = total - self.lambda_utility * utility.mean()
        total = total + self.lambda_var * var_term
        if balance is not None:
            total = total + self.lambda_balance * balance
        if aux_loss is not None:
            total = total + 0.01 * aux_loss

        logs = {
            "total": total.detach(),
            "pred": pred_loss.detach(),
            "utility": utility.detach().mean() if torch.is_tensor(utility)
                       else torch.tensor(utility),
            "var": var_term.detach() if torch.is_tensor(var_term)
                   else torch.tensor(var_term),
        }
        if balance is not None:
            logs["balance"] = balance.detach()
        if aux_loss is not None:
            logs["aux"] = aux_loss.detach()
        return total, logs
