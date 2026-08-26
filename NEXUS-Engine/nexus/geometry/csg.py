"""CSG-ядро: дерево конструктивной геометрии + аналитические SDF.

Это «инвариантное физическое пространство» уровня 2 в его геометрической
части: деталь описывается не набором 2D-проекций, а функцией расстояния
f(x, y, z), из которой строго выводятся объём, масса, центр масс, тензор
инерции, внутренние полости и толщины стенок.

Реализовано на numpy, без внешних CAD-зависимостей.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import math
import numpy as np

Vec3 = Tuple[float, float, float]


# --------------------------------------------------------------------- узлы
@dataclass
class Node:
    def sdf(self, p: np.ndarray) -> np.ndarray:  # pragma: no cover - интерфейс
        raise NotImplementedError

    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:  # pragma: no cover
        raise NotImplementedError

    def children(self) -> Sequence["Node"]:
        return ()

    @property
    def kind(self) -> str:
        return type(self).__name__.lower()

    def params(self) -> List[float]:
        return []


@dataclass
class Cube(Node):
    size: Vec3 = (1.0, 1.0, 1.0)
    center: bool = True

    def _half(self) -> np.ndarray:
        return np.asarray(self.size, dtype=np.float64) / 2.0

    def _offset(self) -> np.ndarray:
        return np.zeros(3) if self.center else self._half()

    def sdf(self, p: np.ndarray) -> np.ndarray:
        q = np.abs(p - self._offset()) - self._half()
        outside = np.linalg.norm(np.maximum(q, 0.0), axis=-1)
        inside = np.minimum(np.max(q, axis=-1), 0.0)
        return outside + inside

    def bounds(self):
        o, h = self._offset(), self._half()
        return o - h, o + h

    def params(self):
        return [*self.size, float(self.center)]


@dataclass
class Sphere(Node):
    r: float = 1.0

    def sdf(self, p: np.ndarray) -> np.ndarray:
        return np.linalg.norm(p, axis=-1) - self.r

    def bounds(self):
        return np.full(3, -self.r), np.full(3, self.r)

    def params(self):
        return [self.r]


@dataclass
class Cylinder(Node):
    h: float = 1.0
    r1: float = 1.0
    r2: float | None = None
    center: bool = True

    def sdf(self, p: np.ndarray) -> np.ndarray:
        r2 = self.r1 if self.r2 is None else self.r2
        z = p[..., 2] - (0.0 if self.center else self.h / 2.0)
        t = np.clip((z + self.h / 2) / max(self.h, 1e-9), 0.0, 1.0)
        r = self.r1 + (r2 - self.r1) * t
        d_radial = np.linalg.norm(p[..., :2], axis=-1) - r
        d_axial = np.abs(z) - self.h / 2
        outside = np.sqrt(np.maximum(d_radial, 0) ** 2 + np.maximum(d_axial, 0) ** 2)
        inside = np.minimum(np.maximum(d_radial, d_axial), 0.0)
        return outside + inside

    def bounds(self):
        r = max(self.r1, self.r1 if self.r2 is None else self.r2)
        z0 = -self.h / 2 if self.center else 0.0
        return np.array([-r, -r, z0]), np.array([r, r, z0 + self.h])

    def params(self):
        return [self.h, self.r1, self.r1 if self.r2 is None else self.r2, float(self.center)]


# ------------------------------------------------------------ преобразования
@dataclass
class Transform(Node):
    child: Node = field(default_factory=Cube)
    translate: Vec3 = (0.0, 0.0, 0.0)
    rotate: Vec3 = (0.0, 0.0, 0.0)     # градусы, XYZ
    scale: Vec3 = (1.0, 1.0, 1.0)

    def matrix(self) -> np.ndarray:
        rx, ry, rz = (math.radians(a) for a in self.rotate)
        cx, sx, cy, sy, cz, sz = (
            math.cos(rx), math.sin(rx), math.cos(ry), math.sin(ry), math.cos(rz), math.sin(rz)
        )
        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        return Rz @ Ry @ Rx

    def sdf(self, p: np.ndarray) -> np.ndarray:
        s = np.asarray(self.scale, dtype=np.float64)
        q = (p - np.asarray(self.translate, dtype=np.float64)) @ self.matrix()
        return self.child.sdf(q / s) * float(np.min(np.abs(s)))

    def bounds(self):
        lo, hi = self.child.bounds()
        corners = np.array(np.meshgrid(*zip(lo, hi), indexing="ij")).reshape(3, -1).T
        corners = corners * np.asarray(self.scale)
        corners = corners @ self.matrix().T + np.asarray(self.translate)
        return corners.min(0), corners.max(0)

    def children(self):
        return (self.child,)

    def params(self):
        return [*self.translate, *self.rotate, *self.scale]


# ---------------------------------------------------------------- булевы ops
@dataclass
class BooleanOp(Node):
    op: str = "union"
    nodes: List[Node] = field(default_factory=list)

    def sdf(self, p: np.ndarray) -> np.ndarray:
        if not self.nodes:
            return np.full(p.shape[:-1], 1e6)
        d = self.nodes[0].sdf(p)
        for n in self.nodes[1:]:
            e = n.sdf(p)
            if self.op == "union":
                d = np.minimum(d, e)
            elif self.op == "difference":
                d = np.maximum(d, -e)
            elif self.op == "intersection":
                d = np.maximum(d, e)
            else:
                raise ValueError(f"неизвестная операция: {self.op}")
        return d

    def bounds(self):
        bs = [n.bounds() for n in self.nodes] or [(np.zeros(3), np.zeros(3))]
        if self.op == "difference":
            return bs[0]
        lo = np.min([b[0] for b in bs], axis=0)
        hi = np.max([b[1] for b in bs], axis=0)
        if self.op == "intersection":
            lo = np.max([b[0] for b in bs], axis=0)
            hi = np.min([b[1] for b in bs], axis=0)
        return lo, hi

    def children(self):
        return tuple(self.nodes)

    @property
    def kind(self) -> str:
        return self.op


def union(*nodes: Node) -> BooleanOp:
    return BooleanOp("union", list(nodes))


def difference(*nodes: Node) -> BooleanOp:
    return BooleanOp("difference", list(nodes))


def intersection(*nodes: Node) -> BooleanOp:
    return BooleanOp("intersection", list(nodes))


def evaluate(node: Node, points: np.ndarray) -> np.ndarray:
    """SDF в точках (..., 3)."""
    return node.sdf(np.asarray(points, dtype=np.float64))


def walk(node: Node):
    yield node
    for c in node.children():
        yield from walk(c)
