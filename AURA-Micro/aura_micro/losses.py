from __future__ import annotations

import torch
import torch.nn.functional as F


def multitask_loss(pred: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    cls = F.cross_entropy(pred["logits"], batch["class_id"])
    doa = (
        F.mse_loss(pred["sin_theta"], batch["sin_theta"])
        + F.mse_loss(pred["cos_theta"], batch["cos_theta"])
        + F.mse_loss(pred["sin_phi"], batch["sin_phi"])
        + F.mse_loss(pred["cos_phi"], batch["cos_phi"])
    )
    dyn = F.smooth_l1_loss(pred["range_m"], batch["range_m"]) + 0.25 * F.smooth_l1_loss(
        pred["vr"], batch["vr"]
    )
    mod = F.binary_cross_entropy_with_logits(pred["modifiers"], batch["modifiers"])
    total = cls + 0.5 * doa + 0.15 * dyn + 0.35 * mod
    return {"total": total, "cls": cls, "doa": doa, "dyn": dyn, "mod": mod}
