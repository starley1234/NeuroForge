"""Генератор параметрических вариаций OpenSCAD (Domain Randomization).

Первое звено маховика данных: из небольшого набора шаблонов деталей
порождаются десятки тысяч валидных вариантов с рандомизацией толщин, фасок,
диаметров отверстий и рёбер жёсткости, а также нагрузочного случая (ТЗ).
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, asdict, field
from typing import Callable, Dict, List, Tuple

MATERIALS = ["pla", "petg", "abs", "alu6061", "steel304"]


@dataclass
class LoadCase:
    """Физическое ТЗ: где закреплено и какая сила приложена."""
    force_n: Tuple[float, float, float]
    fixture: str = "base"           # base | bore | face_x
    safety_factor: float = 2.0
    temperature_c: float = 20.0


@dataclass
class ScadSample:
    template: str
    params: Dict[str, float]
    code: str
    spec: str
    load: LoadCase
    material: str

    def to_json(self) -> Dict[str, object]:
        d = asdict(self)
        d["load"] = asdict(self.load)
        return d

    def dumps(self) -> str:
        return json.dumps(self.to_json(), ensure_ascii=False)


# --------------------------------------------------------------- шаблоны
def _l_bracket(p: Dict[str, float]) -> str:
    return f"""// L-bracket, параметрический
w = {p['width']:.2f}; h = {p['height']:.2f}; t = {p['thickness']:.2f};
d = {p['hole_d']:.2f}; rib = {p['rib']:.2f};
difference() {{
  union() {{
    cube([w, t, h], center=false);
    cube([w, {p['leg']:.2f}, t], center=false);
    translate([w/2 - rib/2, 0, 0]) cube([rib, {p['leg']:.2f}, {p['leg']:.2f}], center=false);
  }}
  translate([w/2, t/2, h*0.72]) rotate([90, 0, 0]) translate([0, 0, -t]) cylinder(h=t*3, r=d/2);
  translate([w/2, {p['leg']:.2f}*0.65, t/2]) cylinder(h=t*3, r=d/2, center=true);
}}
"""


def _flange(p: Dict[str, float]) -> str:
    n = int(p["bolts"])
    holes = "\n".join(
        f"  rotate([0,0,{i * 360 / n:.2f}]) translate([{p['bolt_circle']/2:.2f},0,0]) "
        f"cylinder(h={p['thickness']*3:.2f}, r={p['hole_d']/2:.2f}, center=true);"
        for i in range(n)
    )
    return f"""// Круглый фланец с болтовой окружностью
difference() {{
  union() {{
    cylinder(h={p['thickness']:.2f}, r={p['outer_d']/2:.2f});
    translate([0,0,{p['thickness']:.2f}]) cylinder(h={p['hub_h']:.2f}, r={p['hub_d']/2:.2f});
  }}
  translate([0,0,-1]) cylinder(h={p['thickness'] + p['hub_h'] + 2:.2f}, r={p['bore_d']/2:.2f});
{holes}
}}
"""


def _plate(p: Dict[str, float]) -> str:
    return f"""// Монтажная плита с окном облегчения
difference() {{
  cube([{p['length']:.2f}, {p['width']:.2f}, {p['thickness']:.2f}], center=true);
  cube([{p['length']*0.5:.2f}, {p['width']*0.5:.2f}, {p['thickness']*3:.2f}], center=true);
  translate([{p['length']*0.36:.2f}, {p['width']*0.32:.2f}, 0])
    cylinder(h={p['thickness']*3:.2f}, r={p['hole_d']/2:.2f}, center=true);
  translate([{-p['length']*0.36:.2f}, {-p['width']*0.32:.2f}, 0])
    cylinder(h={p['thickness']*3:.2f}, r={p['hole_d']/2:.2f}, center=true);
}}
"""


def _standoff(p: Dict[str, float]) -> str:
    return f"""// Стойка-проставка с внутренней резьбовой бобышкой
difference() {{
  union() {{
    cylinder(h={p['height']:.2f}, r={p['outer_d']/2:.2f});
    cylinder(h={p['flange_h']:.2f}, r={p['outer_d']*0.75:.2f});
  }}
  translate([0,0,-1]) cylinder(h={p['height'] + 2:.2f}, r={p['bore_d']/2:.2f});
}}
"""


def _bearing_block(p: Dict[str, float]) -> str:
    return f"""// Корпус подшипника с рёбрами жёсткости
difference() {{
  union() {{
    cube([{p['width']:.2f}, {p['depth']:.2f}, {p['thickness']:.2f}], center=false);
    translate([{p['width']/2:.2f}, {p['depth']/2:.2f}, 0])
      cylinder(h={p['height']:.2f}, r={p['boss_d']/2:.2f});
    translate([{p['width']/2 - p['rib']/2:.2f}, 0, 0])
      cube([{p['rib']:.2f}, {p['depth']:.2f}, {p['height']*0.6:.2f}], center=false);
  }}
  translate([{p['width']/2:.2f}, {p['depth']/2:.2f}, -1])
    cylinder(h={p['height'] + 2:.2f}, r={p['bore_d']/2:.2f});
}}
"""


Sampler = Callable[[random.Random], Dict[str, float]]


def _rng_params(spec: Dict[str, Tuple[float, float]], rng: random.Random) -> Dict[str, float]:
    return {k: round(rng.uniform(a, b), 2) for k, (a, b) in spec.items()}


TEMPLATES: Dict[str, Tuple[Callable[[Dict[str, float]], str], Dict[str, Tuple[float, float]], str]] = {
    "l_bracket": (_l_bracket, {
        "width": (24, 60), "height": (26, 70), "thickness": (2.4, 7.0),
        "hole_d": (3.2, 8.5), "rib": (3.0, 9.0), "leg": (20, 55),
    }, "кронштейн L-образный, крепление к стене, консольная нагрузка"),
    "flange": (_flange, {
        "outer_d": (36, 96), "bore_d": (8, 28), "thickness": (3.0, 9.0),
        "hub_d": (16, 44), "hub_h": (4.0, 18.0), "bolt_circle": (26, 78),
        "hole_d": (3.2, 7.0), "bolts": (3, 8),
    }, "фланец вал-корпус, осевая тяга и момент затяжки"),
    "plate": (_plate, {
        "length": (40, 120), "width": (30, 90), "thickness": (2.0, 8.0),
        "hole_d": (3.0, 8.0),
    }, "монтажная плита с окном облегчения, распределённая нагрузка"),
    "standoff": (_standoff, {
        "height": (10, 60), "outer_d": (8, 22), "bore_d": (2.5, 10.0),
        "flange_h": (1.5, 5.0),
    }, "стойка-проставка платы, сжатие и продольный изгиб"),
    "bearing_block": (_bearing_block, {
        "width": (36, 90), "depth": (24, 60), "thickness": (5.0, 14.0),
        "height": (18, 55), "boss_d": (22, 52), "bore_d": (10, 32), "rib": (4.0, 12.0),
    }, "корпус подшипника, радиальная нагрузка от вала"),
}


def _fix_params(name: str, p: Dict[str, float]) -> Dict[str, float]:
    """Ремонт заведомо невалидных комбинаций (bore > outer и т. п.)."""
    if name == "flange":
        p["bolts"] = float(max(3, min(8, round(p["bolts"]))))
        p["bore_d"] = min(p["bore_d"], p["hub_d"] * 0.7)
        p["bolt_circle"] = min(max(p["bolt_circle"], p["hub_d"] + 3 * p["hole_d"]),
                               p["outer_d"] - 2.2 * p["hole_d"])
    if name == "standoff":
        p["bore_d"] = min(p["bore_d"], p["outer_d"] * 0.6)
    if name == "bearing_block":
        p["bore_d"] = min(p["bore_d"], p["boss_d"] * 0.65)
        p["boss_d"] = min(p["boss_d"], min(p["width"], p["depth"]) * 0.9)
    return p


def sample(template: str, rng: random.Random) -> ScadSample:
    fn, spec, description = TEMPLATES[template]
    params = _fix_params(template, _rng_params(spec, rng))
    code = fn(params)
    material = rng.choice(MATERIALS)
    force = (
        round(rng.uniform(-400, 400), 1),
        round(rng.uniform(-400, 400), 1),
        round(rng.uniform(-900, -20), 1),
    )
    load = LoadCase(force, rng.choice(["base", "bore", "face_x"]),
                    round(rng.uniform(1.2, 3.5), 2), round(rng.uniform(-20, 90), 1))
    text = (
        f"<task>Спроектируй деталь: {description}. "
        f"Материал: {material}. Нагрузка F = {force} Н, крепление: {load.fixture}, "
        f"коэффициент запаса {load.safety_factor}, температура {load.temperature_c} °C. "
        f"Ограничения: минимальная стенка 1.2 мм, деталь должна быть manifold."
    )
    return ScadSample(template, params, code, text, load, material)


def generate(n: int, seed: int = 0, templates: List[str] | None = None) -> List[ScadSample]:
    rng = random.Random(seed)
    names = templates or list(TEMPLATES)
    return [sample(rng.choice(names), rng) for _ in range(n)]
