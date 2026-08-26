"""Физические награды для RL-фазы (без обучаемого критика-сети).

    +1.0  код компилируется в валидный CSG
    +1.5  геометрия manifold и печатаема / фрезеруема
    +2.0  запас прочности по FEM ≥ требуемого
    −штрафы за тонкие стенки, лишнюю массу и провал по прочности
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple

from ..fem.solver import solve
from ..scad.render import render


@dataclass
class RewardBreakdown:
    compile: float = 0.0
    manifold: float = 0.0
    strength: float = 0.0
    mass_penalty: float = 0.0
    wall_penalty: float = 0.0
    total: float = 0.0
    info: Optional[Dict[str, object]] = None

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def score_scad(
    code: str,
    force_n: Tuple[float, float, float] = (0.0, 0.0, -200.0),
    fixture: str = "base",
    material: str = "pla",
    required_sf: float = 2.0,
    mass_budget_g: float = 200.0,
    resolution: int = 20,
    fem_iterations: int = 120,
) -> RewardBreakdown:
    res = render(code, resolution=resolution, material=material)
    rb = RewardBreakdown()
    if not res.ok or res.voxels is None or res.audit_report is None:
        rb.total = -1.0
        rb.info = {"error": res.error}
        return rb

    rb.compile = 1.0
    a = res.audit_report
    if a.ok:
        rb.manifold = 1.5
    elif a.manifold:
        rb.manifold = 0.5
    if a.thin_feature_warning:
        rb.wall_penalty = -0.5
    if a.printable_overhang_ratio > 0.35:
        rb.wall_penalty -= 0.25

    fem = solve(res.voxels, force_n, fixture, material, iterations=fem_iterations)
    if fem.safety_factor >= required_sf:
        rb.strength = 2.0
    elif fem.safety_factor >= 1.0:
        rb.strength = 1.0 * (fem.safety_factor / max(required_sf, 1e-6))
    else:
        rb.strength = -1.0

    mass_g = res.mass.mass_g if res.mass else 0.0
    if mass_g > mass_budget_g:
        rb.mass_penalty = -min(1.0, (mass_g - mass_budget_g) / max(mass_budget_g, 1e-6))

    rb.total = rb.compile + rb.manifold + rb.strength + rb.mass_penalty + rb.wall_penalty
    rb.info = {
        "mass_g": round(mass_g, 3),
        "safety_factor": round(fem.safety_factor, 3),
        "sigma_max_mpa": round(fem.max_stress_pa / 1e6, 3),
        "min_wall_mm": round(a.min_wall_mm, 3),
        "fem_backend": fem.backend,
    }
    return rb
