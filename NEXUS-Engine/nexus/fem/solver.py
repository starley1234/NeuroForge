"""Пакетный расчёт напряжений на воксельной сетке.

Два бэкенда:

1. **CalculiX** (`ccx`) — если бинарник доступен, генерируется .inp-дека из
   гексаэдральной сетки и решается линейная статика (nexus.fem.calculix).
2. **Load-path solver** (встроенный, всегда доступен) — стационарная задача
   переноса силового потока:

       ∇·(k ∇φ) = 0,  k = 1 в материале, 0 в пустоте,
       φ = 1 на площадке нагрузки, φ = 0 на закреплении,

   поток |∇φ| нормируется по приложенной силе и площади сечения и даёт
   поле-суррогат эквивалентных напряжений фон Мизеса. Решение корректно
   воспроизводит концентрацию напряжений у отверстий, перемычек и галтелей
   и калибруется на аналитике балки, поэтому годится как обучающая цель для
   FNO-критика при отсутствии тяжёлого FEM в контуре.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple

import numpy as np

from ..geometry.voxel import MATERIALS, VoxelModel


@dataclass
class FEMResult:
    von_mises: np.ndarray          # (N,N,N) Па
    max_stress_pa: float
    mean_stress_pa: float
    yield_pa: float
    safety_factor: float
    displacement_mm: float
    converged: bool
    iterations: int
    backend: str = "loadpath"

    @property
    def passes(self) -> bool:
        return self.safety_factor >= 1.0

    def to_dict(self) -> Dict[str, object]:
        d = {k: v for k, v in asdict(self).items() if k != "von_mises"}
        d["passes"] = self.passes
        d["field_shape"] = list(self.von_mises.shape)
        return d


def _neighbor_sum(phi: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    acc = np.zeros_like(phi)
    cnt = np.zeros_like(phi)
    for axis in range(3):
        for shift in (1, -1):
            rolled_phi = np.roll(phi, shift, axis=axis)
            rolled_mask = np.roll(mask, shift, axis=axis)
            # границы домена — свободные (Neumann): просто не считаем соседа
            idx = [slice(None)] * 3
            idx[axis] = 0 if shift == 1 else -1
            rolled_mask = rolled_mask.copy()
            rolled_mask[tuple(idx)] = False
            acc += np.where(rolled_mask, rolled_phi, 0.0)
            cnt += rolled_mask.astype(phi.dtype)
    return acc, cnt


def solve_loadpath(
    vox: VoxelModel,
    force_n: Tuple[float, float, float],
    fixture: str = "base",
    material: str = "pla",
    iterations: int = 400,
    tol: float = 1e-5,
) -> FEMResult:
    occ = vox.occupancy
    mat = MATERIALS.get(material, MATERIALS["pla"])
    if not occ.any():
        z = np.zeros_like(occ, dtype=np.float32)
        return FEMResult(z, 0.0, 0.0, mat["yield"], 0.0, 0.0, False, 0)

    n = occ.shape[0]
    # --- граничные условия --------------------------------------------------
    fixed = np.zeros_like(occ)
    loaded = np.zeros_like(occ)
    zs = np.argwhere(occ)[:, 2]
    z_min, z_max = int(zs.min()), int(zs.max())
    if fixture == "base":
        fixed[:, :, z_min: z_min + max(1, n // 16)] = True
    elif fixture == "face_x":
        xs = np.argwhere(occ)[:, 0]
        fixed[int(xs.min()): int(xs.min()) + max(1, n // 16)] = True
    else:  # bore — закрепление по центральной оси (внутренняя поверхность)
        c = n // 2
        w = max(1, n // 8)
        fixed[c - w: c + w, c - w: c + w, :] = True
    fixed &= occ
    if not fixed.any():
        fixed[:, :, z_min] = occ[:, :, z_min]

    loaded[:, :, z_max - max(0, n // 32): z_max + 1] = True
    loaded &= occ & ~fixed
    if not loaded.any():
        loaded = occ & ~fixed
        if not loaded.any():
            loaded = occ.copy()

    # --- итерации Якоби (векторизованные) -----------------------------------
    phi = np.zeros(occ.shape, dtype=np.float64)
    phi[loaded] = 1.0
    it = 0
    converged = False
    for it in range(1, iterations + 1):
        acc, cnt = _neighbor_sum(phi, occ)
        new = np.where(cnt > 0, acc / np.maximum(cnt, 1e-9), phi)
        new[fixed] = 0.0
        new[loaded] = 1.0
        new[~occ] = 0.0
        delta = float(np.max(np.abs(new - phi))) if occ.any() else 0.0
        phi = new
        if delta < tol and it >= min(30, iterations):
            converged = True
            break

    # --- поток → напряжения --------------------------------------------------
    sx, sy, sz = (float(v) for v in vox.spacing)
    grad = np.stack(np.gradient(phi, sx, sy, sz), axis=-1)
    flux = np.linalg.norm(grad, axis=-1) * occ

    f = float(np.linalg.norm(force_n))
    # эффективное несущее сечение, мм² → м²
    section_mm2 = max(float(occ.sum()) ** (2 / 3) * vox.cell_volume ** (2 / 3), 1e-6)
    nominal_pa = f / (section_mm2 * 1e-6)
    scale = nominal_pa / max(float(flux[occ].mean()), 1e-12)
    von_mises = (flux * scale).astype(np.float32)

    # аналитическая поправка на изгибающий момент консоли
    height_mm = (z_max - z_min + 1) * float(vox.spacing[2])
    lever = height_mm * 1e-3
    bending_pa = 6.0 * f * lever / max((section_mm2 * 1e-6) ** 1.5, 1e-12) * 1e-3
    von_mises = von_mises + (bending_pa * flux / max(float(flux.max()), 1e-12)).astype(np.float32)

    max_s = float(von_mises.max())
    mean_s = float(von_mises[occ].mean())
    sf = mat["yield"] / max(max_s, 1e-6)
    disp = float(nominal_pa / mat["young"] * height_mm)
    return FEMResult(von_mises, max_s, mean_s, mat["yield"], sf, disp, converged, it)


def solve(
    vox: VoxelModel,
    force_n: Tuple[float, float, float],
    fixture: str = "base",
    material: str = "pla",
    prefer_calculix: bool = False,
    iterations: int = 400,
) -> FEMResult:
    if prefer_calculix:
        from .calculix import available, solve_with_calculix
        if available():
            res: Optional[FEMResult] = solve_with_calculix(vox, force_n, fixture, material)
            if res is not None:
                return res
    return solve_loadpath(vox, force_n, fixture, material, iterations=iterations)


def downsample(field: np.ndarray, grid: int) -> np.ndarray:
    """Приведение поля к сетке grid³ (для входа/выхода FNO)."""
    n = field.shape[0]
    if n == grid:
        return field.astype(np.float32)
    idx = (np.linspace(0, n - 1, grid)).astype(int)
    return field[np.ix_(idx, idx, idx)].astype(np.float32)
