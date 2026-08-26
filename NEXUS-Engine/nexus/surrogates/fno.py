"""FNO-суррогат FEM/CFD: мгновенная оценка поля напряжений/скоростей.

Вход:  (B, C, N, N, N) — occupancy + компоненты нагрузки (и, опционально, поля
       граничных условий CFD).
Выход: (B, 1, N, N, N) — нормированное поле фон Мизеса (или модуль скорости).

Оператор Фурье учит отображение между функциональными пространствами, поэтому
одна сеть обобщается на разные геометрии и разрешения — именно то, что нужно
внутри Latent Reasoning Loop как «мгновенный физический критик».
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import FNOConfig


class SpectralConv3d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, modes: int):
        super().__init__()
        self.modes = modes
        scale = 1.0 / (in_ch * out_ch)
        shape = (in_ch, out_ch, modes, modes, modes)
        self.w1 = nn.Parameter(scale * torch.randn(*shape, dtype=torch.cfloat))
        self.w2 = nn.Parameter(scale * torch.randn(*shape, dtype=torch.cfloat))

    @staticmethod
    def _mul(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixyz,ioxyz->boxyz", x, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, nx, ny, nz = x.shape
        m = min(self.modes, nx, ny, nz // 2 + 1)
        ft = torch.fft.rfftn(x.float(), dim=(-3, -2, -1))
        out = torch.zeros(b, self.w1.shape[1], nx, ny, nz // 2 + 1,
                          dtype=torch.cfloat, device=x.device)
        out[:, :, :m, :m, :m] = self._mul(ft[:, :, :m, :m, :m], self.w1[:, :, :m, :m, :m])
        out[:, :, -m:, :m, :m] = self._mul(ft[:, :, -m:, :m, :m], self.w2[:, :, :m, :m, :m])
        return torch.fft.irfftn(out, s=(nx, ny, nz), dim=(-3, -2, -1)).to(x.dtype)


class FNO3d(nn.Module):
    def __init__(self, cfg: FNOConfig):
        super().__init__()
        self.cfg = cfg
        self.lift = nn.Conv3d(cfg.in_channels + 3, cfg.width, 1)
        self.spectral = nn.ModuleList(SpectralConv3d(cfg.width, cfg.width, cfg.modes)
                                      for _ in range(cfg.depth))
        self.local = nn.ModuleList(nn.Conv3d(cfg.width, cfg.width, 1) for _ in range(cfg.depth))
        self.norms = nn.ModuleList(nn.GroupNorm(4, cfg.width) for _ in range(cfg.depth))
        self.head = nn.Sequential(
            nn.Conv3d(cfg.width, cfg.width * 2, 1), nn.GELU(),
            nn.Conv3d(cfg.width * 2, cfg.out_channels, 1),
        )
        self.scalar_head = nn.Linear(cfg.width, 2)   # (log σ_max, log σ_mean)

    @staticmethod
    def _coords(x: torch.Tensor) -> torch.Tensor:
        b, _, nx, ny, nz = x.shape
        ax = [torch.linspace(0, 1, s, device=x.device, dtype=x.dtype) for s in (nx, ny, nz)]
        grid = torch.stack(torch.meshgrid(*ax, indexing="ij"), dim=0)
        return grid.unsqueeze(0).expand(b, 3, nx, ny, nz)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.lift(torch.cat([x, self._coords(x)], dim=1))
        for spec, loc, norm in zip(self.spectral, self.local, self.norms):
            h = h + F.gelu(norm(spec(h) + loc(h)))
        field = self.head(h)
        scalars = self.scalar_head(h.mean(dim=(-3, -2, -1)))
        return field, scalars


class FEMCritic(nn.Module):
    """Обёртка-критик: поле → скалярная оценка прочности для RL/reasoning."""

    def __init__(self, cfg: FNOConfig):
        super().__init__()
        self.fno = FNO3d(cfg)

    def forward(self, occupancy: torch.Tensor, load: torch.Tensor):
        """occupancy: (B,1,N,N,N); load: (B,3) — вектор силы (нормированный)."""
        b, _, nx, ny, nz = occupancy.shape
        load_field = load.view(b, 3, 1, 1, 1).expand(b, 3, nx, ny, nz)
        x = torch.cat([occupancy, load_field], dim=1)
        field, scalars = self.fno(x)
        return field, scalars

    @torch.no_grad()
    def safety_factor(self, occupancy: torch.Tensor, load: torch.Tensor,
                      yield_pa: torch.Tensor) -> torch.Tensor:
        _, scalars = self(occupancy, load)
        sigma_max = torch.exp(scalars[:, 0]).clamp(min=1e-3)
        return yield_pa / sigma_max
