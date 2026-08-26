"""Headless-рендер OpenSCAD.

Если в системе установлен бинарник `openscad`, он используется для экспорта
STL/CSG. Если нет — включается встроенный SDF/CSG-движок (nexus.geometry),
поэтому маховик данных работает в любой среде, включая CI без GUI.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..geometry.brep import BRepGraph, build_graph, voxel_surface_triangles, write_stl
from ..geometry.csg import Node
from ..geometry.voxel import GeometryAudit, MassProperties, VoxelModel, audit, mass_properties, voxelize
from .parser import ScadSyntaxError, parse_scad


def openscad_binary() -> Optional[str]:
    return shutil.which("openscad") or shutil.which("openscad-nightly")


@dataclass
class RenderResult:
    ok: bool
    error: str = ""
    tree: Optional[Node] = None
    voxels: Optional[VoxelModel] = None
    mass: Optional[MassProperties] = None
    audit_report: Optional[GeometryAudit] = None
    graph: Optional[BRepGraph] = None
    backend: str = "internal"

    def summary(self) -> dict:
        return {
            "ok": self.ok,
            "error": self.error,
            "backend": self.backend,
            "mass": self.mass.to_dict() if self.mass else None,
            "audit": self.audit_report.to_dict() if self.audit_report else None,
            "graph_nodes": self.graph.n_nodes if self.graph else 0,
        }


def compile_scad(source: str) -> RenderResult:
    """Только проверка синтаксиса/сборки CSG (быстрая награда для RL)."""
    try:
        tree = parse_scad(source)
    except (ScadSyntaxError, ValueError, ZeroDivisionError, RecursionError) as exc:
        return RenderResult(False, f"{type(exc).__name__}: {exc}")
    return RenderResult(True, tree=tree)


def render(source: str, resolution: int = 40, material: str = "pla",
           use_openscad: bool = False, stl_path: Optional[str] = None) -> RenderResult:
    """Полный проход: SCAD → CSG → воксели → масс-свойства → аудит → B-Rep граф."""
    res = compile_scad(source)
    if not res.ok:
        return res

    if use_openscad and openscad_binary():
        ok, err = _openscad_check(source)
        res.backend = "openscad"
        if not ok:
            return RenderResult(False, err, backend="openscad")

    vox = voxelize(res.tree, resolution=resolution)
    res.voxels = vox
    res.mass = mass_properties(vox, material)
    res.audit_report = audit(vox)
    res.graph = build_graph(res.tree)
    if stl_path:
        verts, tris = voxel_surface_triangles(vox.occupancy, vox.origin, vox.spacing)
        if len(tris):
            write_stl(stl_path, verts, tris)
    return res


def _openscad_check(source: str) -> tuple[bool, str]:
    binary = openscad_binary()
    if not binary:
        return True, ""
    with tempfile.TemporaryDirectory() as tmp:
        scad = os.path.join(tmp, "model.scad")
        out = os.path.join(tmp, "model.stl")
        with open(scad, "w", encoding="utf-8") as fh:
            fh.write(source)
        try:
            proc = subprocess.run([binary, "-o", out, scad], capture_output=True,
                                  text=True, timeout=120)
        except (subprocess.TimeoutExpired, OSError) as exc:
            return False, f"openscad: {exc}"
        if proc.returncode != 0:
            return False, proc.stderr.strip()[:500]
    return True, ""


def occupancy_field(result: RenderResult, grid: int = 32) -> np.ndarray:
    """Перевыборка occupancy на кубическую сетку заданного размера."""
    assert result.tree is not None
    return voxelize(result.tree, resolution=grid).occupancy.astype(np.float32)
