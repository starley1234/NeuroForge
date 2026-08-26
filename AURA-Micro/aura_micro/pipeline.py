"""End-to-end: 3-ch PCM -> DSP -> AURA-Micro heads."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from aura_micro.config import CLASS_NAMES, MODIFIER_NAMES, AuraConfig
from aura_micro.frontend import HardwareDSPFrontEnd
from aura_micro.model import AuraMicro


class AuraPipeline(nn.Module):
    def __init__(self, cfg: AuraConfig | None = None):
        super().__init__()
        self.cfg = cfg or AuraConfig()
        self.dsp = HardwareDSPFrontEnd(self.cfg)
        self.net = AuraMicro(self.cfg)

    def forward(self, wav: torch.Tensor, state=None):
        feat = self.dsp(wav)
        return self.net(feat, state)

    @torch.no_grad()
    def decode(self, out: dict[str, torch.Tensor]) -> list[dict]:
        probs = torch.softmax(out["logits"], dim=-1)
        mods = torch.sigmoid(out["modifiers"])
        results = []
        b = probs.size(0)
        for i in range(b):
            cid = int(probs[i].argmax())
            theta = math.atan2(float(out["sin_theta"][i]), float(out["cos_theta"][i]))
            phi = math.atan2(float(out["sin_phi"][i]), float(out["cos_phi"][i]))
            active = [
                MODIFIER_NAMES[j]
                for j, v in enumerate(mods[i].tolist())
                if v > 0.5
            ]
            results.append(
                {
                    "class": CLASS_NAMES[cid],
                    "confidence": float(probs[i, cid]),
                    "azimuth_deg": math.degrees(theta),
                    "elevation_deg": math.degrees(phi),
                    "range_m": float(out["range_m"][i]),
                    "vr_mps": float(out["vr"][i]),
                    "modifiers": active,
                }
            )
        return results
