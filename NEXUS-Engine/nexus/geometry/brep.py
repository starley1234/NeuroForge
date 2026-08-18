"""B-Rep / CSG-граф: структурное представление детали для GNO-энкодера.

Граф несёт топологию (кто из кого вычитается, что с чем объединяется) и
геометрические инварианты каждого узла (габариты, объём, параметры примитива).
Это то, чего принципиально нет у «нарезки 3D на 2D-картинки».
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from .csg import BooleanOp, Cube, Cylinder, Node, Sphere, Transform

KINDS = ["cube", "sphere", "cylinder", "transform", "union", "difference", "intersection", "other"]
MAX_PARAMS = 12
NODE_FEATURE_DIM = len(KINDS) + MAX_PARAMS + 6 + 2  # kind + params + bbox + (volume, depth)


@dataclass
class BRepGraph:
    nodes: np.ndarray          # (V, NODE_FEATURE_DIM)
    edges: np.ndarray          # (2, E) — направленные, оба направления
    kinds: List[str]

    @property
    def n_nodes(self) -> int:
        return int(self.nodes.shape[0])

    @property
    def n_edges(self) -> int:
        return int(self.edges.shape[1])


def _kind_id(node: Node) -> int:
    k = node.kind
    return KINDS.index(k) if k in KINDS else KINDS.index("other")


def _features(node: Node, depth: int) -> np.ndarray:
    f = np.zeros(NODE_FEATURE_DIM, dtype=np.float32)
    f[_kind_id(node)] = 1.0
    p = np.asarray(node.params(), dtype=np.float32)[:MAX_PARAMS]
    f[len(KINDS): len(KINDS) + len(p)] = p
    lo, hi = node.bounds()
    off = len(KINDS) + MAX_PARAMS
    f[off: off + 3] = np.asarray(lo, dtype=np.float32)
    f[off + 3: off + 6] = np.asarray(hi, dtype=np.float32)
    f[off + 6] = float(np.prod(np.maximum(np.asarray(hi) - np.asarray(lo), 0.0)))
    f[off + 7] = float(depth)
    return f


def build_graph(root: Node) -> BRepGraph:
    nodes: List[np.ndarray] = []
    kinds: List[str] = []
    src: List[int] = []
    dst: List[int] = []

    def visit(n: Node, depth: int) -> int:
        idx = len(nodes)
        nodes.append(_features(n, depth))
        kinds.append(n.kind)
        for child in n.children():
            cid = visit(child, depth + 1)
            src.extend([idx, cid])
            dst.extend([cid, idx])
        return idx

    visit(root, 0)
    edges = np.asarray([src, dst], dtype=np.int64) if src else np.zeros((2, 0), dtype=np.int64)
    return BRepGraph(np.stack(nodes), edges, kinds)


# ------------------------------------------------------------ экспорт сеток
def voxel_surface_triangles(occupancy: np.ndarray, origin: np.ndarray,
                            spacing) -> Tuple[np.ndarray, np.ndarray]:
    """Грани поверхности воксельной модели → (вершины, треугольники)."""
    verts: List[Tuple[float, float, float]] = []
    tris: List[Tuple[int, int, int]] = []
    vmap: dict = {}

    step = np.broadcast_to(np.asarray(spacing, float), (3,))

    def vid(i, j, k):
        key = (i, j, k)
        if key not in vmap:
            vmap[key] = len(verts)
            verts.append(tuple(origin + np.array([i, j, k]) * step))
        return vmap[key]

    nx, ny, nz = occupancy.shape
    dirs = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    for i, j, k in map(tuple, np.argwhere(occupancy)):
        for d in dirs:
            n = (i + d[0], j + d[1], k + d[2])
            inside = 0 <= n[0] < nx and 0 <= n[1] < ny and 0 <= n[2] < nz
            if inside and occupancy[n]:
                continue
            if d[0]:
                x = i + (1 if d[0] > 0 else 0)
                quad = [(x, j, k), (x, j + 1, k), (x, j + 1, k + 1), (x, j, k + 1)]
            elif d[1]:
                y = j + (1 if d[1] > 0 else 0)
                quad = [(i, y, k), (i + 1, y, k), (i + 1, y, k + 1), (i, y, k + 1)]
            else:
                z = k + (1 if d[2] > 0 else 0)
                quad = [(i, j, z), (i + 1, j, z), (i + 1, j + 1, z), (i, j + 1, z)]
            a, b, c, e = (vid(*q) for q in quad)
            tris.extend([(a, b, c), (a, c, e)])
    return np.asarray(verts, dtype=np.float32), np.asarray(tris, dtype=np.int64)


def write_stl(path: str, verts: np.ndarray, tris: np.ndarray, name: str = "nexus") -> str:
    lines = [f"solid {name}"]
    for t in tris:
        a, b, c = verts[t[0]], verts[t[1]], verts[t[2]]
        n = np.cross(b - a, c - a)
        norm = np.linalg.norm(n)
        n = n / norm if norm > 1e-12 else np.zeros(3)
        lines.append(f"  facet normal {n[0]:.6e} {n[1]:.6e} {n[2]:.6e}")
        lines.append("    outer loop")
        for v in (a, b, c):
            lines.append(f"      vertex {v[0]:.6e} {v[1]:.6e} {v[2]:.6e}")
        lines.append("    endloop")
        lines.append("  endfacet")
    lines.append(f"endsolid {name}")
    text = "\n".join(lines)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return path
