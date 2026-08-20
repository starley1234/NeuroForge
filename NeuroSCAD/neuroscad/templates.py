"""Deterministic baseline generators used before a trained model is available."""
from __future__ import annotations
import re
from .ir import Node, Op, Parameter, Program, calc, param


def _find(prompt: str, patterns: list[str], default: float) -> float:
    for pattern in patterns:
        match = re.search(pattern, prompt, re.IGNORECASE)
        if match: return float(match.group(1).replace(",", "."))
    return default


def camera_pipe_bracket(prompt: str) -> Program:
    """Generate an adjustable split clamp with an M-size bolt lug."""
    tube = _find(prompt, [r"(?:труб\w*|pipe|tube)(?:\s+\w+){0,2}?\s*(?:диаметр\w*|[ø⌀d])?\s*(\d+(?:[.,]\d+)?)", r"[ø⌀]\s*(\d+(?:[.,]\d+)?)"], 25)
    screw = _find(prompt, [r"[мm]\s*(\d+(?:[.,]\d+)?)"], 4)
    ps = (
        Parameter("tube_d", tube, 8, 80, 0.5, label="Диаметр трубы"),
        Parameter("wall", 4, 2, 12, 0.5, label="Толщина стенки"),
        Parameter("clamp_width", 20, 8, 50, 1, label="Ширина хомута"),
        Parameter("split_gap", 3, 1, 10, 0.5, label="Зазор разреза"),
        Parameter("screw_d", screw + 0.4, 2, 12, 0.1, label="Отверстие под винт"),
        Parameter("lug_length", 18, 10, 40, 1, label="Длина проушины"),
        Parameter("lug_height", 12, 8, 30, 1, label="Высота проушины"),
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
    return Program("result", nodes, ps, {
        "name": "Camera pipe split clamp", "source_prompt": prompt,
        "assumptions": ["FDM prototype", "diametral screw clearance +0.4 mm", "load/FEM not evaluated"],
    })


def generate(prompt: str) -> Program:
    lower = prompt.lower()
    if any(word in lower for word in ("труб", "pipe", "tube", "хомут", "clamp", "кронштейн")):
        return camera_pipe_bracket(prompt)
    raise ValueError("MVP recognises pipe/clamp bracket requests only")
