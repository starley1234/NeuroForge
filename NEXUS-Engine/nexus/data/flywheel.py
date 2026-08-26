"""OpenSCAD Data Flywheel — замкнутый контур самообучения.

    Базовые скрипты → Domain Randomization → Headless-рендер (CSG/STL/B-Rep)
        → Геометрический аудит (manifold, стенки, печать, ЧПУ)
        → Пакетный FEM (CalculiX / load-path) → Обучающий кортеж

Кортеж: { ТЗ+нагрузки → OpenSCAD код → 3D-форма → FEM тензор }.
"""
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

import numpy as np

from ..fem.solver import downsample, solve
from ..geometry.voxel import MATERIALS
from ..scad.generator import ScadSample, generate
from ..scad.render import render


@dataclass
class FlywheelSample:
    spec: str
    code: str
    template: str
    material: str
    params: Dict[str, float]
    load: Dict[str, object]
    mass: Dict[str, object]
    audit: Dict[str, object]
    fem: Dict[str, object]
    occupancy: np.ndarray
    stress: np.ndarray
    valid: bool
    reject_reason: str = ""

    def record(self, field_id: Optional[int] = None) -> Dict[str, object]:
        rec = {
            "spec": self.spec, "code": self.code, "template": self.template,
            "material": self.material, "params": self.params, "load": self.load,
            "mass": self.mass, "audit": self.audit, "fem": self.fem,
            "valid": self.valid, "reject_reason": self.reject_reason,
        }
        if field_id is not None:
            rec["field_id"] = field_id
        return rec


@dataclass
class FlywheelStats:
    total: int = 0
    compiled: int = 0
    manifold: int = 0
    fem_passed: int = 0
    rejected: Dict[str, int] = field(default_factory=dict)

    def reject(self, reason: str) -> None:
        self.rejected[reason] = self.rejected.get(reason, 0) + 1

    def to_dict(self) -> Dict[str, object]:
        return {
            "total": self.total, "compiled": self.compiled, "manifold": self.manifold,
            "fem_passed": self.fem_passed, "rejected": self.rejected,
            "yield_rate": round(self.manifold / max(self.total, 1), 4),
        }


def process(sample: ScadSample, grid: int = 24, fem_grid: int = 16,
            use_openscad: bool = False, fem_iterations: int = 250,
            prefer_calculix: bool = False) -> FlywheelSample:
    res = render(sample.code, resolution=grid, material=sample.material,
                 use_openscad=use_openscad)
    empty = np.zeros((fem_grid,) * 3, dtype=np.float32)
    if not res.ok or res.voxels is None:
        return FlywheelSample(sample.spec, sample.code, sample.template, sample.material,
                              sample.params, sample.load.__dict__, {}, {}, {},
                              empty, empty, False, res.error or "compile_error")

    audit_report = res.audit_report
    fem = solve(res.voxels, sample.load.force_n, sample.load.fixture, sample.material,
                prefer_calculix=prefer_calculix, iterations=fem_iterations)

    occ = downsample(res.voxels.occupancy.astype(np.float32), fem_grid)
    sigma = downsample(fem.von_mises, fem_grid)

    valid = bool(audit_report and audit_report.ok)
    reason = "" if valid else (
        "empty" if (audit_report and audit_report.empty) else
        "non_manifold" if (audit_report and not audit_report.manifold) else "thin_wall"
    )
    return FlywheelSample(
        sample.spec, sample.code, sample.template, sample.material, sample.params,
        {"force_n": list(sample.load.force_n), "fixture": sample.load.fixture,
         "safety_factor": sample.load.safety_factor,
         "temperature_c": sample.load.temperature_c},
        res.mass.to_dict() if res.mass else {},
        audit_report.to_dict() if audit_report else {},
        fem.to_dict(), occ, sigma, valid, reason,
    )


def run(n: int, out_dir: str, seed: int = 0, grid: int = 24, fem_grid: int = 16,
        keep_invalid: bool = True, use_openscad: bool = False,
        prefer_calculix: bool = False, templates: Optional[List[str]] = None,
        verbose: bool = True) -> FlywheelStats:
    os.makedirs(out_dir, exist_ok=True)
    stats = FlywheelStats()
    records: List[Dict[str, object]] = []
    occs: List[np.ndarray] = []
    sigmas: List[np.ndarray] = []

    for i, sample in enumerate(generate(n, seed=seed, templates=templates)):
        fs = process(sample, grid=grid, fem_grid=fem_grid, use_openscad=use_openscad,
                     prefer_calculix=prefer_calculix)
        stats.total += 1
        if fs.mass:
            stats.compiled += 1
        else:
            stats.reject(fs.reject_reason or "compile_error")
        if fs.valid:
            stats.manifold += 1
            if fs.fem.get("passes"):
                stats.fem_passed += 1
        elif fs.reject_reason:
            stats.reject(fs.reject_reason)

        if fs.valid or keep_invalid:
            records.append(fs.record(field_id=len(occs)))
            occs.append(fs.occupancy)
            sigmas.append(fs.stress)
        if verbose and (i + 1) % 25 == 0:
            print(f"  [flywheel] {i + 1}/{n}  manifold={stats.manifold}", flush=True)

    with open(os.path.join(out_dir, "dataset.jsonl"), "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    np.savez_compressed(
        os.path.join(out_dir, "fields.npz"),
        occupancy=np.stack(occs) if occs else np.zeros((0, fem_grid, fem_grid, fem_grid), np.float32),
        stress=np.stack(sigmas) if sigmas else np.zeros((0, fem_grid, fem_grid, fem_grid), np.float32),
    )
    with open(os.path.join(out_dir, "stats.json"), "w", encoding="utf-8") as fh:
        json.dump(stats.to_dict(), fh, indent=2, ensure_ascii=False)
    return stats
