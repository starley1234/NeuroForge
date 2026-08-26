"""Воксельная дискретизация SDF + масс-инерционные свойства + аудит.

Здесь считаются физические инварианты, которые уходят на Unified Latent Bus:
объём, масса, центр масс, тензор инерции, площадь поверхности, замкнутость
(manifold), внутренние полости, минимальная толщина стенки, свесы для 3D-печати
и достижимость фрезой для ЧПУ.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple

import numpy as np

from .csg import Node

# кг/м³
MATERIALS: Dict[str, Dict[str, float]] = {
    "pla":      {"density": 1240.0, "yield": 50e6,  "young": 3.5e9},
    "petg":     {"density": 1270.0, "yield": 53e6,  "young": 2.1e9},
    "abs":      {"density": 1040.0, "yield": 40e6,  "young": 2.3e9},
    "alu6061":  {"density": 2700.0, "yield": 276e6, "young": 68.9e9},
    "steel304": {"density": 8000.0, "yield": 215e6, "young": 193e9},
    "ti6al4v":  {"density": 4430.0, "yield": 880e6, "young": 113.8e9},
}


@dataclass
class VoxelModel:
    occupancy: np.ndarray      # (N, N, N) bool
    origin: np.ndarray         # (3,) мм
    spacing: np.ndarray        # (3,) мм на воксель по каждой оси

    def __post_init__(self) -> None:
        self.spacing = np.broadcast_to(np.asarray(self.spacing, float), (3,)).copy()

    @property
    def resolution(self) -> int:
        return int(self.occupancy.shape[0])

    @property
    def cell_volume(self) -> float:
        return float(np.prod(self.spacing))

    @property
    def min_spacing(self) -> float:
        return float(np.min(self.spacing))

    def grid_points(self) -> np.ndarray:
        n = self.resolution
        ax = [self.origin[i] + (np.arange(n) + 0.5) * self.spacing[i] for i in range(3)]
        return np.stack(np.meshgrid(*ax, indexing="ij"), axis=-1)


def voxelize(node: Node, resolution: int = 48, padding: float = 0.08,
             isotropic: bool = False) -> VoxelModel:
    """Дискретизация SDF на сетку resolution³.

    По умолчанию шаг вокселя анизотропный (свой по каждой оси): иначе тонкие
    пластины толщиной 2 мм в габарите 120 мм просто исчезают из сетки.
    """
    lo, hi = node.bounds()
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    size = np.maximum(hi - lo, 1e-6)
    pad = np.maximum(size * padding, 1e-6)
    lo, hi = lo - pad, hi + pad
    if isotropic:
        extent = float(np.max(hi - lo))
        center = (lo + hi) / 2
        lo = center - extent / 2
        hi = center + extent / 2
    origin = lo
    spacing = (hi - lo) / resolution

    ax = [origin[i] + (np.arange(resolution) + 0.5) * spacing[i] for i in range(3)]
    pts = np.stack(np.meshgrid(*ax, indexing="ij"), axis=-1)
    d = node.sdf(pts.reshape(-1, 3)).reshape(resolution, resolution, resolution)
    return VoxelModel(d <= 0.0, origin, spacing)


@dataclass
class MassProperties:
    volume_mm3: float
    mass_g: float
    centroid_mm: Tuple[float, float, float]
    inertia_g_mm2: Tuple[float, float, float]
    surface_mm2: float
    bbox_mm: Tuple[float, float, float]
    material: str

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def mass_properties(vox: VoxelModel, material: str = "pla") -> MassProperties:
    mat = MATERIALS.get(material, MATERIALS["pla"])
    occ = vox.occupancy
    n_solid = int(occ.sum())
    cell = vox.cell_volume
    volume = n_solid * cell
    mass_g = volume * 1e-9 * mat["density"] * 1000.0  # мм³ → м³ → кг → г

    if n_solid == 0:
        return MassProperties(0.0, 0.0, (0, 0, 0), (0, 0, 0), 0.0, (0, 0, 0), material)

    idx = np.argwhere(occ).astype(np.float64)
    coords = vox.origin + (idx + 0.5) * vox.spacing[None, :]
    centroid = coords.mean(axis=0)
    rel = coords - centroid
    m_i = mass_g / n_solid
    ixx = float(m_i * np.sum(rel[:, 1] ** 2 + rel[:, 2] ** 2))
    iyy = float(m_i * np.sum(rel[:, 0] ** 2 + rel[:, 2] ** 2))
    izz = float(m_i * np.sum(rel[:, 0] ** 2 + rel[:, 1] ** 2))

    surface = 0.0
    for axis in range(3):
        a = np.moveaxis(occ, axis, 0)
        faces = int((a[0]).sum() + (a[-1]).sum() + (a[:-1] != a[1:]).sum())
        others = [vox.spacing[i] for i in range(3) if i != axis]
        surface += faces * others[0] * others[1]

    bbox = tuple(float(x) for x in (coords.max(0) - coords.min(0) + vox.spacing))  # мм
    return MassProperties(
        float(volume), float(mass_g), tuple(float(c) for c in centroid),
        (ixx, iyy, izz), float(surface), bbox, material,
    )


# ------------------------------------------------------------------- аудит
def _label_components(mask: np.ndarray) -> Tuple[np.ndarray, int]:
    """6-связные компоненты без scipy (итеративный BFS по маске)."""
    labels = np.zeros(mask.shape, dtype=np.int32)
    cur = 0
    shape = mask.shape
    for seed in map(tuple, np.argwhere(mask)):
        if labels[seed]:
            continue
        cur += 1
        stack = [seed]
        labels[seed] = cur
        while stack:
            x, y, z = stack.pop()
            for dx, dy, dz in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
                nx, ny, nz = x + dx, y + dy, z + dz
                if 0 <= nx < shape[0] and 0 <= ny < shape[1] and 0 <= nz < shape[2]:
                    if mask[nx, ny, nz] and not labels[nx, ny, nz]:
                        labels[nx, ny, nz] = cur
                        stack.append((nx, ny, nz))
    return labels, cur


def _run_lengths(mask: np.ndarray, axis: int) -> np.ndarray:
    """Длины сплошных отрезков материала вдоль оси (в вокселях)."""
    a = np.moveaxis(mask, axis, -1).reshape(-1, mask.shape[axis]).astype(np.int8)
    pad = np.pad(a, ((0, 0), (1, 1)))
    d = np.diff(pad, axis=1)
    starts = np.argwhere(d == 1)
    ends = np.argwhere(d == -1)
    if len(starts) == 0:
        return np.zeros(0, dtype=np.int64)
    return ends[:, 1] - starts[:, 1]


def min_wall_thickness(vox: "VoxelModel", percentile: float = 2.0) -> float:
    """Толщина стенки: перцентиль длин сплошных отрезков по каждой оси, в мм.

    Перцентиль (а не строгий минимум) отсекает шум дискретизации на скруглениях.
    """
    best = float("inf")
    for axis in range(3):
        runs = _run_lengths(vox.occupancy, axis)
        if runs.size == 0:
            continue
        thickness = float(np.percentile(runs, percentile)) * float(vox.spacing[axis])
        best = min(best, thickness)
    return 0.0 if best == float("inf") else best


def _erode(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    for axis in range(3):
        a = np.moveaxis(out, axis, 0)
        shifted_up = np.concatenate([np.zeros_like(a[:1]), a[:-1]], axis=0)
        shifted_dn = np.concatenate([a[1:], np.zeros_like(a[:1])], axis=0)
        a = a & shifted_up & shifted_dn
        out = np.moveaxis(a, 0, axis)
    return out


@dataclass
class GeometryAudit:
    manifold: bool
    n_solid_components: int
    n_internal_cavities: int
    min_wall_mm: float
    printable_overhang_ratio: float
    touches_bbox: bool
    thin_feature_warning: bool
    cnc_reachable_ratio: float
    empty: bool

    @property
    def ok(self) -> bool:
        return (not self.empty) and self.manifold and not self.thin_feature_warning

    def to_dict(self) -> Dict[str, object]:
        d = asdict(self)
        d["ok"] = self.ok
        return d


def audit(vox: VoxelModel, min_wall_mm: float = 1.2,
          max_overhang_deg: float = 45.0) -> GeometryAudit:
    occ = vox.occupancy
    if not occ.any():
        return GeometryAudit(False, 0, 0, 0.0, 0.0, False, True, 0.0, True)

    _, n_solid = _label_components(occ)

    # полости: компоненты пустоты, не связанные с внешней границей
    void = ~occ
    labels, n_void = _label_components(void)
    border = set()
    for axis in range(3):
        a = np.moveaxis(labels, axis, 0)
        border.update(np.unique(a[0]).tolist())
        border.update(np.unique(a[-1]).tolist())
    border.discard(0)
    cavities = max(0, n_void - len([b for b in border if b > 0]))

    min_wall = min_wall_thickness(vox)

    # свесы: доля нижних граней без опоры под ними (упрощённая модель)
    below = np.concatenate([np.zeros_like(occ[:, :, :1]), occ[:, :, :-1]], axis=2)
    unsupported = occ & ~below
    overhang_ratio = float(unsupported.sum()) / float(occ.sum())

    # достижимость фрезой сверху (ЧПУ 3-осевой): луч вниз по +Z
    top_visible = np.cumsum(occ[:, :, ::-1], axis=2)[:, :, ::-1]
    reachable = float(((top_visible <= 1) & occ).sum()) / float(occ.sum())

    touches = any(
        np.moveaxis(occ, ax, 0)[i].any() for ax in range(3) for i in (0, -1)
    )
    return GeometryAudit(
        manifold=(n_solid == 1) and not touches,
        n_solid_components=n_solid,
        n_internal_cavities=cavities,
        min_wall_mm=float(min_wall),
        printable_overhang_ratio=overhang_ratio,
        touches_bbox=bool(touches),
        thin_feature_warning=bool(min_wall < min_wall_mm),
        cnc_reachable_ratio=reachable,
        empty=False,
    )


def occupancy_tensor(node: Node, resolution: int = 32,
                     material: str = "pla") -> Tuple[np.ndarray, MassProperties]:
    vox = voxelize(node, resolution)
    return vox.occupancy.astype(np.float32), mass_properties(vox, material)
