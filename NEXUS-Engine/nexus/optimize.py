"""Автоматическая оптимизация параметрической детали под нагрузку.

Что делает: берёт параметрический OpenSCAD-скрипт (такой, какие уже лежат в
библиотеке шаблонов), заданную нагрузку и требуемый запас прочности, перебирает
значения параметров с **настоящим расчётом МКЭ** на каждом шаге и возвращает
самый лёгкий вариант, который держит.

Это классический generative design, но без облака и без лицензии: один расчёт
занимает доли секунды, поэтому 40–100 вариантов перебираются за десятки секунд
на обычном ПК.

    nexus optimize part.scad --force 0 0 -350 --safety 2.0 --budget 60

Меняются только «силовые» параметры (толщины, стенки, рёбра, бобышки) — размеры
посадочных мест и отверстий остаются нетронутыми, иначе деталь перестанет
подходить по месту.
"""
from __future__ import annotations

import json
import math
import os
import random
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Имена параметров, которые влияют на прочность и массу, но не на присоединительные
# размеры. Их можно менять без риска, что деталь перестанет подходить по месту.
STRENGTH_PATTERNS = (
    "thick", "wall", "rib", "web", "floor", "shell", "boss", "flange_th", "gusset",
    "толщ", "стенк", "ребр", "дно", "бобыш", "усил", "_th", "_t",
)
# Эти трогать нельзя: посадки, отверстия, межосевые расстояния, резьбы.
# Габариты и присоединительные размеры: их менять нельзя — деталь перестанет
# подходить по месту или соответствовать запросу пользователя.
INTERFACE_PATTERNS = (
    "hole", "bore", "pcd", "screw", "shaft", "pipe", "inner_d", "clearance",
    "distance", "height", "width", "length", "leg", "depth", "outer_d", "diameter",
    "отверст", "вал", "резьб", "межос", "высот", "ширин", "длин", "диамет",
    "d1", "d2", "id", "od",
)

ASSIGN_RE = re.compile(r"^(?P<indent>[ \t]*)(?P<name>{name})\s*=\s*(?P<value>[^;]+);",
                       re.MULTILINE)


@dataclass
class Knob:
    """Параметр, который разрешено крутить."""
    name: str
    base: float
    low: float
    high: float
    comment: str = ""

    def clamp(self, value: float) -> float:
        return max(self.low, min(self.high, value))


@dataclass
class Candidate:
    params: Dict[str, float]
    mass_g: float
    safety_factor: float
    min_wall_mm: float
    ok: bool
    score: float
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class OptimizeResult:
    baseline: Candidate
    best: Candidate
    code: str
    knobs: List[Dict[str, Any]]
    evaluations: int
    seconds: float
    history: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def mass_saved_pct(self) -> float:
        if self.baseline.mass_g <= 0:
            return 0.0
        return round(100.0 * (self.baseline.mass_g - self.best.mass_g)
                     / self.baseline.mass_g, 1)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "baseline": self.baseline.to_dict(),
            "best": self.best.to_dict(),
            "mass_saved_pct": self.mass_saved_pct,
            "knobs": self.knobs,
            "evaluations": self.evaluations,
            "seconds": round(self.seconds, 1),
        }

    def summary(self) -> str:
        b, n = self.baseline, self.best
        lines = [
            f"{'параметр':22s} {'было':>10s} {'стало':>10s}",
            "-" * 44,
        ]
        for name, value in n.params.items():
            lines.append(f"{name:22s} {b.params.get(name, float('nan')):10.3g} "
                         f"{value:10.3g}")
        lines += [
            "-" * 44,
            f"{'масса, г':22s} {b.mass_g:10.2f} {n.mass_g:10.2f}"
            f"   ({self.mass_saved_pct:+.1f} %)",
            f"{'запас прочности':22s} {b.safety_factor:10.2f} {n.safety_factor:10.2f}",
            f"{'мин. стенка, мм':22s} {b.min_wall_mm:10.2f} {n.min_wall_mm:10.2f}",
            f"\nвариантов просчитано: {self.evaluations} за {self.seconds:.1f} с",
        ]
        return "\n".join(lines)


# --------------------------------------------------------------- параметры
def substitute(code: str, params: Dict[str, float]) -> str:
    """Подставить новые значения параметров в исходник."""
    out = code
    for name, value in params.items():
        pattern = re.compile(ASSIGN_RE.pattern.format(name=re.escape(name)), re.MULTILINE)
        formatted = f"{value:.4g}"
        out = pattern.sub(lambda m: f"{m.group('indent')}{name} = {formatted};", out, count=1)
    return out


def pick_knobs(code: str, span: float = 0.6, only: Optional[Sequence[str]] = None,
               limit: int = 6) -> List[Knob]:
    """Выбрать параметры, которые можно безопасно крутить."""
    from .data.ingest import extract_parameters

    knobs: List[Knob] = []
    for param in extract_parameters(code, limit=80):
        name, raw = param["name"], param["value"].strip()
        try:
            base = float(raw)
        except ValueError:
            continue
        if base <= 0:
            continue
        low_name = name.lower()
        if only is not None:
            if name not in only:
                continue
        else:
            if any(p in low_name for p in INTERFACE_PATTERNS):
                continue
            if not any(p in low_name for p in STRENGTH_PATTERNS):
                continue
        knobs.append(Knob(name=name, base=base,
                          low=max(base * (1 - span), 0.4),
                          high=base * (1 + span),
                          comment=param.get("comment", "")))
        if len(knobs) >= limit:
            break
    return knobs


# ------------------------------------------------------------------ оценка
def evaluate(code: str, params: Dict[str, float], force_n: Sequence[float],
             fixture: str, material: str, required_sf: float, min_wall_mm: float,
             grid: int, mass_budget_g: Optional[float]) -> Candidate:
    """Один вариант: рендер → аудит → МКЭ → сводная оценка (меньше — лучше)."""
    from .fem.solver import solve
    from .scad.render import render

    variant = substitute(code, params)
    res = render(variant, resolution=grid, material=material)
    if not res.ok or res.voxels is None or res.audit_report is None:
        return Candidate(params, 0.0, 0.0, 0.0, False, 1e9, res.error or "render_error")

    audit = res.audit_report
    fem = solve(res.voxels, tuple(force_n), fixture, material, iterations=200)
    mass = res.mass.mass_g if res.mass else 0.0
    sf = fem.safety_factor

    penalty = 0.0
    if not audit.manifold:
        penalty += 1e6
    if audit.min_wall_mm < min_wall_mm:
        penalty += 1e4 * (min_wall_mm - audit.min_wall_mm)
    if sf < required_sf:
        penalty += 1e3 * (required_sf - sf)
    if mass_budget_g and mass > mass_budget_g:
        penalty += 10 * (mass - mass_budget_g)

    ok = penalty == 0.0
    return Candidate(dict(params), mass, sf, audit.min_wall_mm, ok, mass + penalty)


# --------------------------------------------------------------- поиск
def optimize(
    code: str,
    force_n: Sequence[float] = (0.0, 0.0, -200.0),
    fixture: str = "base",
    material: str = "pla",
    required_sf: float = 2.0,
    min_wall_mm: float = 1.2,
    budget: int = 40,
    grid: int = 16,
    verify_grid: int = 26,
    span: float = 0.6,
    only: Optional[Sequence[str]] = None,
    seed: int = 0,
    mass_budget_g: Optional[float] = None,
    verbose: bool = True,
) -> OptimizeResult:
    """Найти самый лёгкий вариант детали, который держит нагрузку.

    Стратегия: случайный поиск для разведки + покоординатный спуск для
    доводки. Дёшево, устойчиво и не требует градиентов — на каждом шаге
    выполняется настоящий расчёт, а не предсказание.
    """
    rng = random.Random(seed)
    knobs = pick_knobs(code, span=span, only=only)
    if not knobs:
        raise ValueError("не нашёл силовых параметров: укажите их явно через --params")

    t0 = time.time()
    base_params = {k.name: k.base for k in knobs}
    baseline = evaluate(code, base_params, force_n, fixture, material, required_sf,
                        min_wall_mm, grid, mass_budget_g)
    if verbose:
        print(f"[optimize] крутим: {', '.join(k.name for k in knobs)}")
        print(f"[optimize] исходно: масса {baseline.mass_g:.1f} г, "
              f"запас {baseline.safety_factor:.2f}, "
              f"{'проходит' if baseline.ok else 'НЕ проходит'}")

    best = baseline
    history: List[Dict[str, Any]] = []
    evaluations = 1

    # 1) разведка случайными точками
    explore = max(4, budget // 3)
    for _ in range(explore):
        trial = {k.name: k.clamp(k.base * rng.uniform(1 - span, 1 + span)) for k in knobs}
        cand = evaluate(code, trial, force_n, fixture, material, required_sf,
                        min_wall_mm, grid, mass_budget_g)
        evaluations += 1
        history.append({"stage": "explore", "score": cand.score, "mass": cand.mass_g,
                        "sf": cand.safety_factor})
        if cand.score < best.score:
            best = cand

    # 2) покоординатный спуск от лучшей точки
    step = 0.25
    while evaluations < budget and step > 0.03:
        improved = False
        for knob in knobs:
            for direction in (1.0, -1.0):
                if evaluations >= budget:
                    break
                trial = dict(best.params)
                trial[knob.name] = knob.clamp(trial[knob.name] * (1 + direction * step))
                cand = evaluate(code, trial, force_n, fixture, material, required_sf,
                                min_wall_mm, grid, mass_budget_g)
                evaluations += 1
                history.append({"stage": f"descent:{knob.name}", "score": cand.score,
                                "mass": cand.mass_g, "sf": cand.safety_factor})
                if cand.score < best.score:
                    best, improved = cand, True
        if not improved:
            step /= 2

    # 3) финальная проверка лучшего варианта на мелкой сетке
    if verify_grid > grid:
        refined = evaluate(code, best.params, force_n, fixture, material, required_sf,
                           min_wall_mm, verify_grid, mass_budget_g)
        evaluations += 1
        if refined.ok or not best.ok:
            best = refined
        baseline = evaluate(code, base_params, force_n, fixture, material, required_sf,
                            min_wall_mm, verify_grid, mass_budget_g)
        evaluations += 1

    result = OptimizeResult(
        baseline=baseline, best=best, code=substitute(code, best.params),
        knobs=[asdict(k) for k in knobs], evaluations=evaluations,
        seconds=time.time() - t0, history=history,
    )
    if verbose:
        print(f"[optimize] итог: масса {best.mass_g:.1f} г "
              f"({result.mass_saved_pct:+.1f} %), запас {best.safety_factor:.2f}")
    return result


def optimize_file(path: str, out_path: Optional[str] = None, **kwargs) -> OptimizeResult:
    with open(path, encoding="utf-8") as fh:
        code = fh.read()
    result = optimize(code, **kwargs)
    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(result.code)
        with open(os.path.splitext(out_path)[0] + "_report.json", "w",
                  encoding="utf-8") as fh:
            json.dump(result.to_dict(), fh, indent=2, ensure_ascii=False)
    return result
