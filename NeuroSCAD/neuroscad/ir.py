"""Typed, serialisable CSG-IR.

The IR deliberately separates topology, dimensions and presentation metadata.
It is small enough for constrained decoding but expressive enough for an MVP.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
import re
from typing import Any, Mapping


class IRError(ValueError):
    """Raised when a program violates the grammar or engineering constraints."""


class Op(str, Enum):
    BOX = "box"
    CYLINDER = "cylinder"
    SPHERE = "sphere"
    UNION = "union"
    DIFFERENCE = "difference"
    INTERSECTION = "intersection"


Expr = float | int | dict[str, Any]


def param(name: str) -> dict[str, str]:
    return {"param": name}


def calc(op: str, *args: Expr) -> dict[str, Any]:
    if op not in {"add", "sub", "mul", "div", "max", "min"}:
        raise IRError(f"unknown expression operator: {op}")
    return {"op": op, "args": list(args)}


@dataclass(frozen=True)
class Parameter:
    name: str
    default: float
    minimum: float
    maximum: float
    step: float = 0.1
    unit: str = "mm"
    label: str = ""
    group: str = "Dimensions"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Parameter":
        return cls(
            name=str(data["name"]), default=float(data["default"]),
            minimum=float(data["minimum"]), maximum=float(data["maximum"]),
            step=float(data.get("step", 0.1)), unit=str(data.get("unit", "mm")),
            label=str(data.get("label", "")), group=str(data.get("group", "Dimensions")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "default": self.default, "minimum": self.minimum,
            "maximum": self.maximum, "step": self.step, "unit": self.unit,
            "label": self.label, "group": self.group,
        }


@dataclass(frozen=True)
class Node:
    id: str
    op: Op
    children: tuple[str, ...] = ()
    args: Mapping[str, Any] = field(default_factory=dict)
    translate: tuple[Expr, Expr, Expr] = (0, 0, 0)
    rotate: tuple[Expr, Expr, Expr] = (0, 0, 0)
    role: str = "solid"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Node":
        return cls(
            id=str(data["id"]), op=Op(data["op"]),
            children=tuple(data.get("children", ())), args=dict(data.get("args", {})),
            translate=tuple(data.get("translate", (0, 0, 0))),
            rotate=tuple(data.get("rotate", (0, 0, 0))),
            role=str(data.get("role", "solid")),
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"id": self.id, "op": self.op.value}
        if self.children: result["children"] = list(self.children)
        if self.args: result["args"] = dict(self.args)
        if self.translate != (0, 0, 0): result["translate"] = list(self.translate)
        if self.rotate != (0, 0, 0): result["rotate"] = list(self.rotate)
        if self.role != "solid": result["role"] = self.role
        return result


@dataclass(frozen=True)
class Program:
    root: str
    nodes: tuple[Node, ...]
    parameters: tuple[Parameter, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    version: str = "0.1"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Program":
        return cls(
            root=str(data["root"]), nodes=tuple(Node.from_dict(n) for n in data["nodes"]),
            parameters=tuple(Parameter.from_dict(p) for p in data.get("parameters", ())),
            metadata=dict(data.get("metadata", {})), version=str(data.get("version", "0.1")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version, "root": self.root,
            "parameters": [p.to_dict() for p in self.parameters],
            "nodes": [n.to_dict() for n in self.nodes], "metadata": dict(self.metadata),
        }


def evaluate(expr: Expr, values: Mapping[str, float]) -> float:
    if isinstance(expr, bool):
        raise IRError("boolean is not a numeric expression")
    if isinstance(expr, (int, float)):
        result = float(expr)
        if not math.isfinite(result): raise IRError("expression must be finite")
        return result
    if not isinstance(expr, dict):
        raise IRError(f"invalid expression: {expr!r}")
    if "param" in expr:
        name = str(expr["param"])
        if name not in values: raise IRError(f"unknown parameter: {name}")
        return float(values[name])
    op, args = expr.get("op"), [evaluate(x, values) for x in expr.get("args", [])]
    if len(args) < 2: raise IRError(f"{op} requires at least two operands")
    if op == "add": return sum(args)
    if op == "sub": return args[0] - sum(args[1:])
    if op == "mul":
        out = 1.0
        for x in args: out *= x
        return out
    if op == "div":
        out = args[0]
        for x in args[1:]: out /= x
        return out
    if op == "max": return max(args)
    if op == "min": return min(args)
    raise IRError(f"unknown expression operator: {op}")


def validate(program: Program, overrides: Mapping[str, float] | None = None) -> list[str]:
    """Validate graph shape and dimensions; return non-fatal warnings."""
    if program.version != "0.1": raise IRError(f"unsupported IR version: {program.version}")
    if len(program.nodes) > 512: raise IRError("program exceeds 512-node safety limit")
    params = {p.name: p for p in program.parameters}
    if len(params) != len(program.parameters): raise IRError("duplicate parameter name")
    if any(not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", p.name) for p in program.parameters):
        raise IRError("parameter names must be safe OpenSCAD identifiers")
    values = {p.name: p.default for p in program.parameters}
    for p in program.parameters:
        if not p.minimum <= p.default <= p.maximum: raise IRError(f"{p.name}: default outside range")
        if p.step <= 0 or p.minimum >= p.maximum: raise IRError(f"{p.name}: invalid range")
    for name, value in (overrides or {}).items():
        if name not in params: raise IRError(f"unknown override: {name}")
        if not params[name].minimum <= value <= params[name].maximum:
            raise IRError(f"{name}: override outside range")
        values[name] = float(value)

    nodes = {n.id: n for n in program.nodes}
    if len(nodes) != len(program.nodes): raise IRError("duplicate node id")
    if program.root not in nodes: raise IRError("root node does not exist")
    primitive = {Op.BOX, Op.CYLINDER, Op.SPHERE}
    operators = {Op.UNION, Op.DIFFERENCE, Op.INTERSECTION}
    state: dict[str, int] = {}
    def visit(node_id: str, depth: int = 0) -> None:
        if depth > 128: raise IRError("program exceeds 128-level depth limit")
        if node_id not in nodes: raise IRError(f"missing child node: {node_id}")
        if state.get(node_id) == 1: raise IRError("CSG graph contains a cycle")
        if state.get(node_id) == 2: return
        state[node_id] = 1
        n = nodes[node_id]
        if n.op in primitive and n.children: raise IRError(f"{n.id}: primitive cannot have children")
        if n.op in operators and len(n.children) < 2: raise IRError(f"{n.id}: operator needs >=2 children")
        required = {Op.BOX: ("size",), Op.CYLINDER: ("d", "h"), Op.SPHERE: ("d",)}.get(n.op, ())
        for key in required:
            if key not in n.args: raise IRError(f"{n.id}: missing {key}")
            raw = n.args[key]
            seq = raw if isinstance(raw, list) else [raw]
            if n.op == Op.BOX and key == "size" and len(seq) != 3:
                raise IRError(f"{n.id}: box size must have three components")
            if any(evaluate(x, values) <= 0 for x in seq): raise IRError(f"{n.id}: {key} must be positive")
        if len(n.translate) != 3 or len(n.rotate) != 3:
            raise IRError(f"{n.id}: transforms must have three components")
        for x in (*n.translate, *n.rotate): evaluate(x, values)
        for child in n.children: visit(child, depth + 1)
        state[node_id] = 2
    visit(program.root)
    unreachable = sorted(set(nodes) - set(state))
    return ([f"unreachable nodes: {', '.join(unreachable)}"] if unreachable else [])
