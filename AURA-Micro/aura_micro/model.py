"""AURA-Micro network: Sinc stem + D-DS-CNN + FastGRNN + 4 heads."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from aura_micro.config import AuraConfig


class LearnableSincConv(nn.Module):
    """Band-pass temporal stem (fixed-window FIR + depthwise)."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 15):
        super().__init__()
        self.proj = nn.Conv1d(in_ch, out_ch, kernel_size=1)
        self.fir = nn.Conv1d(out_ch, out_ch, kernel_size=kernel, padding=kernel // 2, groups=out_ch)
        self.dw = nn.Conv1d(out_ch, out_ch, kernel_size=5, padding=2, groups=out_ch)
        self.pw = nn.Conv1d(out_ch, out_ch, kernel_size=1)
        self.act = nn.Hardswish()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.fir(self.proj(x)))
        return self.act(self.pw(self.dw(y)))


class DilatedDSBlock(nn.Module):
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.dw = nn.Conv1d(
            channels,
            channels,
            kernel_size=5,
            padding=2 * dilation,
            dilation=dilation,
            groups=channels,
        )
        self.pw = nn.Conv1d(channels, channels, kernel_size=1)
        self.norm = nn.GroupNorm(1, channels)
        self.act = nn.Hardswish()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.act(self.norm(self.pw(self.dw(x))))


class FastGRNNCell(nn.Module):
    """FastGRNN: z = σ(Wx + Uh), h' = tanh(Wx + Uh), h = (ζ(1-z)+ν)⊙h' + z⊙h.

    Hidden state is 64 floats (~256 B fp32, 64 B int8) — cyclic RAM budget.
    """

    def __init__(self, input_size: int, hidden: int):
        super().__init__()
        self.W = nn.Linear(input_size, hidden, bias=False)
        self.U = nn.Linear(hidden, hidden, bias=False)
        self.bz = nn.Parameter(torch.zeros(hidden))
        self.bh = nn.Parameter(torch.zeros(hidden))
        self.zeta = nn.Parameter(torch.tensor(0.9))
        self.nu = nn.Parameter(torch.tensor(0.1))
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.orthogonal_(self.U.weight)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        pre = torch.clamp(self.W(x) + self.U(h), -8.0, 8.0)
        z = torch.sigmoid(pre + self.bz)
        hp = torch.tanh(pre + self.bh)
        zeta = torch.sigmoid(self.zeta)
        nu = torch.sigmoid(self.nu)
        gate = zeta * (1.0 - z) + nu
        return torch.clamp(gate * hp + z * h, -4.0, 4.0)


class FastGRNN(nn.Module):
    def __init__(self, input_size: int, hidden: int):
        super().__init__()
        self.hidden = hidden
        self.cell = FastGRNNCell(input_size, hidden)

    def forward(self, x: torch.Tensor, h0: torch.Tensor | None = None) -> torch.Tensor:
        # x: [B, T, C]
        b, t, _ = x.shape
        h = x.new_zeros(b, self.hidden) if h0 is None else h0
        outs = []
        for i in range(t):
            h = self.cell(x[:, i], h)
            outs.append(h)
        return torch.stack(outs, dim=1), h


class MultiTaskHeads(nn.Module):
    def __init__(self, hidden: int, n_classes: int, n_modifiers: int):
        super().__init__()
        self.cls = nn.Sequential(nn.Linear(hidden, hidden), nn.Hardswish(), nn.Linear(hidden, n_classes))
        self.doa = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.Hardswish(), nn.Linear(hidden // 2, 4))
        self.dyn = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.Hardswish(), nn.Linear(hidden // 2, 2))
        self.mod = nn.Sequential(nn.Linear(hidden, hidden), nn.Hardswish(), nn.Linear(hidden, n_modifiers))

    def forward(self, h: torch.Tensor) -> dict[str, torch.Tensor]:
        doa = self.doa(h)
        dyn = self.dyn(h)
        return {
            "logits": self.cls(h),
            "sin_theta": torch.tanh(doa[:, 0]),
            "cos_theta": torch.tanh(doa[:, 1]),
            "sin_phi": torch.tanh(doa[:, 2]),
            "cos_phi": torch.tanh(doa[:, 3]),
            "range_m": torch.clamp(F.softplus(dyn[:, 0]), 0.0, 80.0),
            "vr": torch.clamp(dyn[:, 1], -20.0, 20.0),
            "modifiers": self.mod(h),
        }


class AuraMicro(nn.Module):
    def __init__(self, cfg: AuraConfig | None = None):
        super().__init__()
        self.cfg = cfg or AuraConfig()
        c0 = self.cfg.stem_out
        self.input_norm = nn.GroupNorm(1, self.cfg.feat_dim)
        self.stem = LearnableSincConv(self.cfg.feat_dim, c0)
        blocks = []
        ch = c0
        for out_ch, dil in zip(self.cfg.dds_channels, self.cfg.dilations):
            if out_ch != ch:
                blocks.append(nn.Conv1d(ch, out_ch, kernel_size=1))
                ch = out_ch
            blocks.append(DilatedDSBlock(ch, dilation=dil))
        self.backbone = nn.Sequential(*blocks)
        self.rnn = FastGRNN(ch, self.cfg.hidden)
        self.heads = MultiTaskHeads(self.cfg.hidden, self.cfg.n_classes, self.cfg.n_modifiers)

    def forward(
        self,
        features: torch.Tensor,
        state: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """features: [B, T, feat_dim] from HardwareDSPFrontEnd."""
        x = torch.nan_to_num(features, nan=0.0, posinf=8.0, neginf=-8.0)
        x = x.transpose(1, 2)  # [B, C, T]
        x = self.input_norm(x)
        x = self.stem(x)
        x = self.backbone(x)
        x = x.transpose(1, 2)
        seq, h = self.rnn(x, state)
        out = self.heads(h)
        out["state"] = h
        out["sequence"] = seq
        return out

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def int8_flash_kb(self) -> float:
        return self.count_parameters() / 1024.0

    def within_budget(self) -> bool:
        return self.int8_flash_kb() <= self.cfg.int8_budget_kb + 80.0  # headroom
