"""Линейная статика на воксельной сетке: настоящий МКЭ без сборки матрицы.

Трилинейные 8-узловые гексаэдры (C3D8). Поскольку все воксели одинаковы,
матрица жёсткости элемента KE считается **один раз**, а произведение K·u
выполняется поэлементно (matrix-free) — это классический приём для воксельных
сеток и топологической оптимизации: памяти O(N), сборки глобальной матрицы нет.

Решатель: метод сопряжённых градиентов с якобиевым предобуславливателем.
В систему входят **только заполненные воксели**; узлы, не принадлежащие ни
одному материальному элементу, закрепляются — это убирает нулевые моды и
на порядок улучшает обусловленность по сравнению с «мягким» заполнением
пустоты. Дополнительно добавляется тихоновская регуляризация 1e-9·diag(K),
чтобы плавающие острова (несвязанные куски детали) не ломали сходимость.

Выход: поле эквивалентных напряжений фон Мизеса по элементам, максимальное
перемещение и коэффициент запаса. Всё на numpy, без внешних зависимостей.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Optional, Tuple

import numpy as np

from ..geometry.voxel import MATERIALS, VoxelModel

GAUSS = (-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0))
CORNERS = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                    [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], dtype=float)


def _elastic_matrix(young: float, poisson: float) -> np.ndarray:
    e, nu = young, poisson
    c = e / ((1 + nu) * (1 - 2 * nu))
    D = np.zeros((6, 6))
    D[:3, :3] = c * (nu * np.ones((3, 3)) + (1 - 2 * nu) * np.eye(3))
    D[3:, 3:] = c * (1 - 2 * nu) / 2 * np.eye(3)
    return D


def _shape_derivs(xi: float, eta: float, zeta: float) -> np.ndarray:
    """dN/dξ для 8 узлов, форма (3, 8)."""
    s = CORNERS
    p = np.array([xi, eta, zeta])
    out = np.zeros((3, 8))
    for a in range(3):
        b, c = [i for i in range(3) if i != a]
        out[a] = 0.125 * s[:, a] * (1 + s[:, b] * p[b]) * (1 + s[:, c] * p[c])
    return out


def _b_matrix(dn_dx: np.ndarray) -> np.ndarray:
    """B (6×24) из производных функций формы в физических координатах (3×8)."""
    B = np.zeros((6, 24))
    for n in range(8):
        dx, dy, dz = dn_dx[:, n]
        col = 3 * n
        B[0, col] = dx
        B[1, col + 1] = dy
        B[2, col + 2] = dz
        B[3, col] = dy
        B[3, col + 1] = dx
        B[4, col + 1] = dz
        B[4, col + 2] = dy
        B[5, col] = dz
        B[5, col + 2] = dx
    return B


@lru_cache(maxsize=8)
def element_stiffness(hx: float, hy: float, hz: float, young: float,
                      poisson: float) -> Tuple[np.ndarray, np.ndarray]:
    """KE (24×24) и B в центре элемента (6×24) для кирпича hx×hy×hz."""
    D = _elastic_matrix(young, poisson)
    scale = np.array([2.0 / hx, 2.0 / hy, 2.0 / hz])[:, None]
    det_j = (hx * hy * hz) / 8.0
    KE = np.zeros((24, 24))
    for xi in GAUSS:
        for eta in GAUSS:
            for zeta in GAUSS:
                B = _b_matrix(_shape_derivs(xi, eta, zeta) * scale)
                KE += B.T @ D @ B * det_j
    B_center = _b_matrix(_shape_derivs(0.0, 0.0, 0.0) * scale)
    return KE, B_center


def _dof_map(shape: Tuple[int, int, int]) -> np.ndarray:
    """Матрица степеней свободы: (n_elem, 24)."""
    nx, ny, nz = shape
    node_shape = (nx + 1, ny + 1, nz + 1)
    ex, ey, ez = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    base = np.stack([ex.ravel(), ey.ravel(), ez.ravel()], axis=1)
    dofs = np.zeros((base.shape[0], 24), dtype=np.int64)
    for n, (dx, dy, dz) in enumerate(((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
                                      (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1))):
        node = np.ravel_multi_index(
            (base[:, 0] + dx, base[:, 1] + dy, base[:, 2] + dz), node_shape)
        dofs[:, 3 * n: 3 * n + 3] = np.stack([3 * node, 3 * node + 1, 3 * node + 2], axis=1)
    return dofs


@dataclass
class HexFEMResult:
    von_mises: np.ndarray
    max_stress_pa: float
    mean_stress_pa: float
    max_displacement_mm: float
    compliance: float
    iterations: int
    residual: float
    converged: bool


def solve_hex_fem(
    vox: VoxelModel,
    force_n: Tuple[float, float, float],
    fixture: str = "base",
    material: str = "pla",
    poisson: float = 0.33,
    max_iter: int = 600,
    tol: float = 1e-6,
    regularization: float = 1e-9,
) -> HexFEMResult:
    occ = vox.occupancy
    mat = MATERIALS.get(material, MATERIALS["pla"])
    young = mat["young"]
    shape = occ.shape
    nx, ny, nz = shape
    n_nodes = (nx + 1) * (ny + 1) * (nz + 1)
    n_dof = 3 * n_nodes

    hx, hy, hz = (float(s) * 1e-3 for s in vox.spacing)     # мм → м
    KE, B_center = element_stiffness(hx, hy, hz, young, poisson)
    D = _elastic_matrix(young, poisson)

    edof_all = _dof_map(shape)
    solid_mask = occ.reshape(-1).astype(bool)
    edof = edof_all[solid_mask]                      # только материальные элементы

    # --- граничные условия -------------------------------------------------
    solid_idx = np.argwhere(occ)
    if solid_idx.size == 0:
        z = np.zeros(shape, dtype=np.float32)
        return HexFEMResult(z, 0.0, 0.0, 0.0, 0.0, 0, 0.0, False)
    z_min, z_max = int(solid_idx[:, 2].min()), int(solid_idx[:, 2].max())
    x_min = int(solid_idx[:, 0].min())

    node_grid = np.arange(n_nodes).reshape(nx + 1, ny + 1, nz + 1)
    if fixture == "face_x":
        fixed_nodes = node_grid[x_min: x_min + 2].ravel()
    elif fixture == "bore":
        c0, c1 = nx // 2 - max(1, nx // 8), nx // 2 + max(1, nx // 8)
        fixed_nodes = node_grid[c0:c1, c0:c1, :].ravel()
    else:
        fixed_nodes = node_grid[:, :, z_min: z_min + 2].ravel()
    loaded_nodes = node_grid[:, :, z_max: z_max + 2].ravel()

    active = np.zeros(n_dof, dtype=bool)
    active[edof.ravel()] = True                      # узлы материала
    fixed = ~active
    fixed[np.concatenate([3 * fixed_nodes, 3 * fixed_nodes + 1, 3 * fixed_nodes + 2])] = True

    f = np.zeros(n_dof)
    material_nodes = np.unique(edof.ravel() // 3)
    loaded_nodes = np.intersect1d(loaded_nodes, material_nodes)
    load_nodes = np.setdiff1d(loaded_nodes, fixed_nodes)
    if load_nodes.size == 0:
        load_nodes = material_nodes
    for axis, value in enumerate(force_n):
        f[3 * load_nodes + axis] = value / max(load_nodes.size, 1)
    f[fixed] = 0.0

    # --- matrix-free K·u ---------------------------------------------------
    diag = np.bincount(edof.ravel(), weights=np.tile(np.diag(KE), edof.shape[0]),
                       minlength=n_dof)
    reg = regularization * float(diag.max() if diag.size else 1.0)

    def matvec(u: np.ndarray) -> np.ndarray:
        ue = u[edof]                                  # (n_solid_elem, 24)
        fe = ue @ KE
        out = np.bincount(edof.ravel(), weights=fe.ravel(), minlength=n_dof)
        out += reg * u
        out[fixed] = 0.0
        return out

    diag = diag + reg
    diag[diag <= 0] = 1.0
    m_inv = 1.0 / diag
    m_inv[fixed] = 0.0

    # --- предобусловленный CG ---------------------------------------------
    u = np.zeros(n_dof)
    r = f - matvec(u)
    r[fixed] = 0.0
    z = m_inv * r
    p = z.copy()
    rz = float(r @ z)
    f_norm = float(np.linalg.norm(f)) or 1.0
    it, residual, converged = 0, float(np.linalg.norm(r)) / f_norm, False
    for it in range(1, max_iter + 1):
        ap = matvec(p)
        denom = float(p @ ap)
        if abs(denom) < 1e-30:
            break
        alpha = rz / denom
        u += alpha * p
        r -= alpha * ap
        residual = float(np.linalg.norm(r)) / f_norm
        if residual < tol:
            converged = True
            break
        z = m_inv * r
        rz_new = float(r @ z)
        p = z + (rz_new / rz) * p
        rz = rz_new

    # --- напряжения по элементам ------------------------------------------
    strain = u[edof] @ B_center.T                      # (n_solid_elem, 6)
    stress = strain @ D.T
    sxx, syy, szz, sxy, syz, sxz = stress.T
    vm = np.sqrt(0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
                 + 3.0 * (sxy ** 2 + syz ** 2 + sxz ** 2))
    field_flat = np.zeros(solid_mask.size, dtype=np.float32)
    field_flat[solid_mask] = vm
    field = field_flat.reshape(shape)

    return HexFEMResult(
        von_mises=field,
        max_stress_pa=float(field.max()),
        mean_stress_pa=float(field[occ].mean()) if occ.any() else 0.0,
        max_displacement_mm=float(np.abs(u).max() * 1e3),
        compliance=float(f @ u),
        iterations=it,
        residual=residual,
        converged=converged,
    )
