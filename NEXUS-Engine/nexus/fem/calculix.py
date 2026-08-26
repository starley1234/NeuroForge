"""Адаптер CalculiX (`ccx`): экспорт воксельной сетки в .inp и разбор .dat.

Если CalculiX не установлен, `available()` возвращает False и вызывающий код
падает обратно на встроенный load-path решатель (nexus.fem.solver).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from typing import Dict, Optional, Tuple

import numpy as np

from ..geometry.voxel import MATERIALS, VoxelModel


def available() -> bool:
    return shutil.which("ccx") is not None or shutil.which("ccx_2.21") is not None


def _binary() -> Optional[str]:
    return shutil.which("ccx") or shutil.which("ccx_2.21")


def build_inp(vox: VoxelModel, force_n: Tuple[float, float, float],
              fixture: str = "base", material: str = "pla") -> str:
    """Сгенерировать деку линейной статики C3D8 из воксельной модели."""
    occ = vox.occupancy
    n = occ.shape[0]
    mat = MATERIALS.get(material, MATERIALS["pla"])
    nodes: Dict[Tuple[int, int, int], int] = {}
    lines_n, lines_e = [], []

    def nid(i, j, k) -> int:
        key = (i, j, k)
        if key not in nodes:
            nodes[key] = len(nodes) + 1
            x, y, z = vox.origin + np.array([i, j, k]) * vox.spacing
            lines_n.append(f"{nodes[key]}, {x:.5f}, {y:.5f}, {z:.5f}")
        return nodes[key]

    eid = 0
    for i, j, k in map(tuple, np.argwhere(occ)):
        eid += 1
        c = [nid(i, j, k), nid(i + 1, j, k), nid(i + 1, j + 1, k), nid(i, j + 1, k),
             nid(i, j, k + 1), nid(i + 1, j, k + 1), nid(i + 1, j + 1, k + 1), nid(i, j + 1, k + 1)]
        lines_e.append(f"{eid}, " + ", ".join(str(x) for x in c))

    zs = np.argwhere(occ)[:, 2]
    z_min, z_max = int(zs.min()), int(zs.max())
    fixed = [v for (i, j, k), v in nodes.items() if k <= z_min + (0 if fixture != "base" else 1)]
    loaded = [v for (i, j, k), v in nodes.items() if k >= z_max]
    fx, fy, fz = (c / max(len(loaded), 1) for c in force_n)

    parts = [
        "*NODE, NSET=NALL", *lines_n,
        "*ELEMENT, TYPE=C3D8, ELSET=EALL", *lines_e,
        "*MATERIAL, NAME=MAT", "*ELASTIC",
        f"{mat['young']:.4e}, 0.3",
        "*SOLID SECTION, ELSET=EALL, MATERIAL=MAT",
        "*NSET, NSET=FIX", ", ".join(str(v) for v in fixed) or "1",
        "*NSET, NSET=LOAD", ", ".join(str(v) for v in loaded) or "1",
        "*STEP", "*STATIC", "*BOUNDARY", "FIX, 1, 3, 0.0", "*CLOAD",
        f"LOAD, 1, {fx:.5f}", f"LOAD, 2, {fy:.5f}", f"LOAD, 3, {fz:.5f}",
        "*EL PRINT, ELSET=EALL", "S", "*END STEP",
    ]
    return "\n".join(parts) + "\n"


def solve_with_calculix(vox: VoxelModel, force_n, fixture="base", material="pla",
                        timeout: int = 600):
    from .solver import FEMResult

    binary = _binary()
    if binary is None:
        return None
    with tempfile.TemporaryDirectory() as tmp:
        job = os.path.join(tmp, "job")
        with open(job + ".inp", "w", encoding="utf-8") as fh:
            fh.write(build_inp(vox, force_n, fixture, material))
        try:
            proc = subprocess.run([binary, "job"], cwd=tmp, capture_output=True,
                                  text=True, timeout=timeout)
        except (subprocess.TimeoutExpired, OSError):
            return None
        dat = job + ".dat"
        if proc.returncode != 0 or not os.path.exists(dat):
            return None
        stresses = _parse_dat(dat)
    if stresses is None or stresses.size == 0:
        return None

    mat = MATERIALS.get(material, MATERIALS["pla"])
    field = np.zeros(vox.occupancy.shape, dtype=np.float32)
    coords = np.argwhere(vox.occupancy)
    m = min(len(coords), len(stresses))
    field[tuple(coords[:m].T)] = stresses[:m]
    max_s = float(field.max())
    return FEMResult(
        field, max_s, float(field[vox.occupancy].mean()), mat["yield"],
        mat["yield"] / max(max_s, 1e-6), 0.0, True, 1, backend="calculix",
    )


def _parse_dat(path: str) -> Optional[np.ndarray]:
    vals = []
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        collect = False
        for line in fh:
            if "stresses" in line.lower():
                collect = True
                continue
            if collect:
                parts = line.split()
                if len(parts) >= 8:
                    try:
                        s = [float(x) for x in parts[2:8]]
                    except ValueError:
                        continue
                    sxx, syy, szz, sxy, sxz, syz = s
                    vm = np.sqrt(0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
                                 + 3 * (sxy ** 2 + sxz ** 2 + syz ** 2))
                    vals.append(vm)
                elif not parts:
                    collect = bool(vals) and collect
    return np.asarray(vals, dtype=np.float32) if vals else None
