"""Deterministic product-family generators used before trained weights are promoted."""
from __future__ import annotations
import re
from .ir import Constraint, Node, Op, Parameter, Program, calc, param


def _find(prompt: str, patterns: list[str], default: float) -> float:
    for pattern in patterns:
        match = re.search(pattern, prompt, re.IGNORECASE)
        if match: return float(match.group(1).replace(",", "."))
    return default


def _fastener(prompt: str, default: float = 4) -> float:
    return _find(prompt, [r"\b[мm]\s*(\d+(?:[.,]\d+)?)"], default)


def camera_pipe_bracket(prompt: str) -> Program:
    """Generate an adjustable split clamp with an M-size bolt lug."""
    tube = _find(prompt, [
        r"(\d+(?:[.,]\d+)?)\s*(?:мм|mm)?\s*(?:труб\w*|pipe|tube)",
        r"(?:труб\w*|pipe|tube)(?:\s+\w+){0,2}?\s*(?:диаметр\w*|[ø⌀d])?\s*(\d+(?:[.,]\d+)?)",
        r"[ø⌀]\s*(\d+(?:[.,]\d+)?)",
    ], 25)
    screw = _fastener(prompt)
    ps = (
        Parameter("tube_d", tube, 8, 80, 0.5, label="Диаметр трубы"),
        Parameter("wall", 4, 2, 12, 0.5, label="Толщина стенки"),
        Parameter("clamp_width", 20, 8, 50, 1, label="Ширина хомута"),
        Parameter("split_gap", 3, 1, 10, 0.5, label="Зазор разреза"),
        Parameter("screw_d", screw + 0.4, 2, 12, 0.1, label="Отверстие под винт"),
        Parameter("lug_length", 18, 10, 40, 1, label="Длина проушины"),
        Parameter("lug_height", 14, 14, 30, 1, label="Высота проушины"),
    )
    outer = calc("add", param("tube_d"), calc("mul", 2, param("wall")))
    nodes = (
        Node("result", Op.DIFFERENCE, ("body", "bore", "split", "bolt_hole")),
        Node("body", Op.UNION, ("ring", "lug")),
        Node("ring", Op.CYLINDER, args={"d": outer, "h": param("clamp_width")}),
        Node("lug", Op.BOX, args={"size": [param("lug_length"), param("lug_height"), param("clamp_width")]},
             translate=(calc("add", calc("div", outer, 2), calc("div", param("lug_length"), 2), -1), 0, 0)),
        Node("bore", Op.CYLINDER, args={"d": param("tube_d"), "h": calc("add", param("clamp_width"), 2)}),
        Node("split", Op.BOX, args={"size": [calc("add", param("wall"), param("lug_length"), 2), param("split_gap"), calc("add", param("clamp_width"), 2)]},
             translate=(calc("add", calc("div", outer, 2), calc("div", param("lug_length"), 2)), 0, 0), role="clearance"),
        Node("bolt_hole", Op.CYLINDER, args={"d": param("screw_d"), "h": calc("add", param("lug_height"), 2)},
             translate=(calc("add", calc("div", outer, 2), calc("div", param("lug_length"), 2)), 0, 0), rotate=(90, 0, 0), role="fastener_clearance"),
    )
    constraints = (
        Constraint("bolt_ligament", calc("add", param("screw_d"), 2), "le", param("lug_height"), "Вокруг отверстия должно оставаться не менее 1 мм материала"),
        Constraint("split_inside_lug", param("split_gap"), "lt", param("lug_height"), "Разрез должен быть уже проушины"),
    )
    return Program("result", nodes, ps, {
        "name": "Camera pipe split clamp", "family": "split_pipe_clamp", "source_prompt": prompt,
        "assumptions": ["FDM prototype", "diametral screw clearance +0.4 mm", "load/FEM not evaluated"],
    }, constraints=constraints)


def mounting_plate(prompt: str) -> Program:
    """Rectangular plate with four symmetric fastener holes."""
    dims = re.search(r"(\d+(?:[.,]\d+)?)\s*[xх×]\s*(\d+(?:[.,]\d+)?)\s*[xх×]\s*(\d+(?:[.,]\d+)?)", prompt, re.I)
    length, width, thickness = (float(x.replace(",", ".")) for x in dims.groups()) if dims else (80.0, 40.0, 4.0)
    hole = _fastener(prompt) + 0.4
    ps = (
        Parameter("plate_length", length, 25, 300, 1, label="Длина пластины"),
        Parameter("plate_width", width, 25, 200, 1, label="Ширина пластины"),
        Parameter("thickness", thickness, 2, 20, 0.5, label="Толщина"),
        Parameter("hole_d", hole, 2, 14, 0.1, label="Диаметр отверстий"),
        Parameter("edge_offset", 8, 4, 17, 0.5, label="Отступ центров от края"),
    )
    x = calc("sub", calc("div", param("plate_length"), 2), param("edge_offset"))
    y = calc("sub", calc("div", param("plate_width"), 2), param("edge_offset"))
    nodes: list[Node] = [Node("result", Op.DIFFERENCE, ("plate", "hole_1", "hole_2", "hole_3", "hole_4")),
                         Node("plate", Op.BOX, args={"size": [param("plate_length"), param("plate_width"), param("thickness")]})]
    for index, (sx, sy) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1)), 1):
        nodes.append(Node(f"hole_{index}", Op.CYLINDER, args={"d": param("hole_d"), "h": calc("add", param("thickness"), 2)},
                          translate=(calc("mul", sx, x), calc("mul", sy, y), 0), role="fastener_clearance"))
    constraints = (
        Constraint("holes_fit_length", calc("add", calc("mul", 2, param("edge_offset")), param("hole_d")), "lt", param("plate_length"), "Отверстия не помещаются по длине"),
        Constraint("holes_fit_width", calc("add", calc("mul", 2, param("edge_offset")), param("hole_d")), "lt", param("plate_width"), "Отверстия не помещаются по ширине"),
        Constraint("edge_ligament", param("edge_offset"), "gt", calc("div", param("hole_d"), 2), "Центр отверстия слишком близко к краю"),
    )
    return Program("result", tuple(nodes), ps, {
        "name": "Four-hole mounting plate", "family": "mounting_plate", "source_prompt": prompt,
        "assumptions": ["four symmetric through holes", "diametral screw clearance +0.4 mm", "sharp external edges"],
    }, constraints=constraints)


def bushing(prompt: str) -> Program:
    """Simple printable cylindrical bushing."""
    compact = re.search(r"(?:втулк\w*|bushing)\s*(\d+(?:[.,]\d+)?)\s*[/xх×]\s*(\d+(?:[.,]\d+)?)", prompt, re.I)
    if compact:
        outer, inner = (float(x.replace(",", ".")) for x in compact.groups())
    else:
        outer = _find(prompt, [r"(?:наружн\w*|outer)\s*(?:диаметр|diameter|[dø⌀])?\s*(\d+(?:[.,]\d+)?)"], 20)
        inner = _find(prompt, [r"(?:внутренн\w*|inner|отверсти\w*|bore)\s*(?:диаметр|diameter|[dø⌀])?\s*(\d+(?:[.,]\d+)?)"], 8)
    height = _find(prompt, [r"(?:высот\w*|длин\w*|height|length)\s*(\d+(?:[.,]\d+)?)"], 15)
    ps = (
        Parameter("outer_d", outer, inner + 2.4, 100, 0.5, label="Наружный диаметр"),
        Parameter("inner_d", inner, 1, outer - 2.4, 0.5, label="Внутренний диаметр"),
        Parameter("height", height, 2, 100, 0.5, label="Высота"),
    )
    nodes = (
        Node("result", Op.DIFFERENCE, ("outer", "bore")),
        Node("outer", Op.CYLINDER, args={"d": param("outer_d"), "h": param("height")}),
        Node("bore", Op.CYLINDER, args={"d": param("inner_d"), "h": calc("add", param("height"), 2)}, role="clearance"),
    )
    constraints = (Constraint("minimum_wall", param("outer_d"), "ge", calc("add", param("inner_d"), 2.4), "Радиальная стенка должна быть не тоньше 1.2 мм"),)
    return Program("result", nodes, ps, {
        "name": "Cylindrical bushing", "family": "bushing", "source_prompt": prompt,
        "assumptions": ["1.2 mm minimum radial wall", "straight through bore"],
    }, constraints=constraints)


def generate(prompt: str) -> Program:
    lower = prompt.lower()
    if any(word in lower for word in ("пластин", "plate", "монтажная планка")):
        return mounting_plate(prompt)
    if any(word in lower for word in ("втул", "bushing", "sleeve")):
        return bushing(prompt)
    if any(word in lower for word in ("труб", "pipe", "tube", "хомут", "clamp", "кронштейн")):
        return camera_pipe_bracket(prompt)
    raise ValueError("Поддерживаются: хомут для трубы, монтажная пластина и цилиндрическая втулка")
