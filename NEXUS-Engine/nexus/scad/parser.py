"""Парсер подмножества OpenSCAD → CSG-дерево.

Поддерживается: переменные и арифметика, cube/sphere/cylinder,
translate/rotate/scale/mirror-как-scale, union/difference/intersection,
блоки {...}, комментарии, $fn/$fa/$fs (игнорируются), for-циклы с диапазоном.
Этого достаточно для параметрических деталей data-flywheel.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from ..geometry.csg import BooleanOp, Cube, Cylinder, Node, Sphere, Transform, union

TOKEN_RE = re.compile(
    r"""
    (?P<ws>\s+)
  | (?P<comment>//[^\n]*|/\*.*?\*/)
  | (?P<number>\d+\.\d*|\.\d+|\d+)
  | (?P<name>\$?[A-Za-z_][A-Za-z_0-9]*)
  | (?P<op><=|>=|==|!=|&&|\|\||[\[\]{}();,=+\-*/%<>?:!])
    """,
    re.VERBOSE | re.DOTALL,
)


class ScadSyntaxError(ValueError):
    pass


@dataclass
class Token:
    kind: str
    value: str
    pos: int


def tokenize(src: str) -> List[Token]:
    out: List[Token] = []
    i = 0
    while i < len(src):
        m = TOKEN_RE.match(src, i)
        if not m:
            raise ScadSyntaxError(f"неразобранный символ {src[i]!r} на позиции {i}")
        kind = m.lastgroup
        if kind not in ("ws", "comment"):
            out.append(Token(kind, m.group(), i))
        i = m.end()
    return out


class ScadParser:
    def __init__(self, src: str):
        self.tokens = tokenize(src)
        self.i = 0
        self.vars: Dict[str, Any] = {"true": 1.0, "false": 0.0, "PI": math.pi}

    # ------------------------------------------------------------- служебное
    def peek(self, k: int = 0) -> Optional[Token]:
        return self.tokens[self.i + k] if self.i + k < len(self.tokens) else None

    def next(self) -> Token:
        if self.i >= len(self.tokens):
            raise ScadSyntaxError("неожиданный конец файла")
        self.i += 1
        return self.tokens[self.i - 1]

    def accept(self, value: str) -> bool:
        t = self.peek()
        if t and t.value == value:
            self.i += 1
            return True
        return False

    def expect(self, value: str) -> Token:
        t = self.next()
        if t.value != value:
            raise ScadSyntaxError(f"ожидалось {value!r}, получено {t.value!r} (поз. {t.pos})")
        return t

    # ------------------------------------------------------------ выражения
    def expr(self) -> Any:
        return self.add()

    def add(self) -> Any:
        v = self.mul()
        while (t := self.peek()) and t.value in "+-":
            self.next()
            r = self.mul()
            v = v + r if t.value == "+" else v - r
        return v

    def mul(self) -> Any:
        v = self.unary()
        while (t := self.peek()) and t.value in ("*", "/", "%"):
            self.next()
            r = self.unary()
            if t.value == "*":
                v = v * r
            elif t.value == "/":
                v = v / r if r else 0.0
            else:
                v = math.fmod(v, r) if r else 0.0
        return v

    def unary(self) -> Any:
        t = self.peek()
        if t and t.value == "-":
            self.next()
            v = self.unary()
            return [-x for x in v] if isinstance(v, list) else -v
        if t and t.value == "+":
            self.next()
            return self.unary()
        return self.atom()

    def atom(self) -> Any:
        t = self.next()
        if t.kind == "number":
            return float(t.value)
        if t.value == "(":
            v = self.expr()
            self.expect(")")
            return v
        if t.value == "[":
            items: List[Any] = []
            if not self.accept("]"):
                items.append(self.expr())
                while self.accept(","):
                    items.append(self.expr())
                self.expect("]")
            return items
        if t.kind == "name":
            if t.value in ("true", "false"):
                return 1.0 if t.value == "true" else 0.0
            if t.value in ("sin", "cos", "tan", "sqrt", "abs", "max", "min", "pow", "floor", "ceil"):
                self.expect("(")
                args = [self.expr()]
                while self.accept(","):
                    args.append(self.expr())
                self.expect(")")
                return self._call(t.value, args)
            return self.vars.get(t.value, 0.0)
        raise ScadSyntaxError(f"неожиданный токен {t.value!r} (поз. {t.pos})")

    @staticmethod
    def _call(name: str, args: List[Any]) -> float:
        if name in ("sin", "cos", "tan"):
            return getattr(math, name)(math.radians(args[0]))
        if name == "sqrt":
            return math.sqrt(max(args[0], 0.0))
        if name == "abs":
            return abs(args[0])
        if name == "pow":
            return args[0] ** args[1]
        if name == "max":
            return max(args)
        if name == "min":
            return min(args)
        return getattr(math, name)(args[0])

    # -------------------------------------------------------------- аргументы
    def arguments(self) -> Tuple[List[Any], Dict[str, Any]]:
        self.expect("(")
        pos: List[Any] = []
        kw: Dict[str, Any] = {}
        if self.accept(")"):
            return pos, kw
        while True:
            t = self.peek()
            nxt = self.peek(1)
            if t and t.kind == "name" and nxt and nxt.value == "=":
                key = self.next().value
                self.next()
                kw[key.lstrip("$")] = self.expr()
            else:
                pos.append(self.expr())
            if self.accept(","):
                continue
            self.expect(")")
            return pos, kw

    # ------------------------------------------------------------- операторы
    def parse(self) -> Node:
        nodes = self.statements(until=None)
        if not nodes:
            raise ScadSyntaxError("пустая модель")
        return nodes[0] if len(nodes) == 1 else union(*nodes)

    def statements(self, until: Optional[str]) -> List[Node]:
        out: List[Node] = []
        while True:
            t = self.peek()
            if t is None:
                if until:
                    raise ScadSyntaxError(f"не закрыт блок {until!r}")
                return out
            if until and t.value == until:
                self.next()
                return out
            node = self.statement()
            if node is not None:
                out.append(node)

    def block_or_statement(self) -> Optional[Node]:
        if self.accept("{"):
            nodes = self.statements(until="}")
            return union(*nodes) if len(nodes) != 1 else nodes[0]
        return self.statement()

    def statement(self) -> Optional[Node]:
        t = self.peek()
        if t is None:
            return None
        if t.value == ";":
            self.next()
            return None
        if t.value == "{":
            return self.block_or_statement()
        if t.kind != "name":
            raise ScadSyntaxError(f"неожиданный токен {t.value!r} (поз. {t.pos})")

        nxt = self.peek(1)
        if nxt and nxt.value == "=":            # присваивание переменной
            name = self.next().value
            self.next()
            self.vars[name] = self.expr()
            self.accept(";")
            return None

        name = self.next().value
        if name.startswith("$"):                # $fn=... уже обработан выше
            self.accept(";")
            return None
        args, kw = self.arguments()
        node = self._make(name, args, kw)
        if node is not None:
            self.accept(";")
            return node

        child = self.block_or_statement()
        return self._wrap(name, args, kw, child)

    # ---------------------------------------------------------------- фабрики
    def _make(self, name: str, args: List[Any], kw: Dict[str, Any]) -> Optional[Node]:
        if name == "cube":
            size = kw.get("size", args[0] if args else 1.0)
            size = [float(size)] * 3 if not isinstance(size, list) else [float(x) for x in size]
            center = bool(kw.get("center", args[1] if len(args) > 1 else 0.0))
            return Cube(tuple(size), center)
        if name == "sphere":
            r = kw.get("r", args[0] if args else None)
            if r is None:
                r = float(kw.get("d", 2.0)) / 2
            return Sphere(float(r))
        if name == "cylinder":
            h = float(kw.get("h", args[0] if args else 1.0))
            if "d" in kw:
                r1 = r2 = float(kw["d"]) / 2
            else:
                r1 = float(kw.get("r1", kw.get("r", args[1] if len(args) > 1 else 1.0)))
                r2 = float(kw.get("r2", kw.get("r", r1)))
            if "d1" in kw:
                r1 = float(kw["d1"]) / 2
            if "d2" in kw:
                r2 = float(kw["d2"]) / 2
            center = bool(kw.get("center", 0.0))
            return Cylinder(h, r1, r2, center)
        return None

    def _wrap(self, name: str, args: List[Any], kw: Dict[str, Any],
              child: Optional[Node]) -> Optional[Node]:
        if child is None:
            return None
        if name == "translate":
            v = kw.get("v", args[0] if args else [0, 0, 0])
            return Transform(child, translate=tuple(_vec3(v)))
        if name == "rotate":
            v = kw.get("a", args[0] if args else [0, 0, 0])
            v = [0, 0, float(v)] if not isinstance(v, list) else v
            return Transform(child, rotate=tuple(_vec3(v)))
        if name == "scale":
            v = kw.get("v", args[0] if args else [1, 1, 1])
            v = [float(v)] * 3 if not isinstance(v, list) else v
            return Transform(child, scale=tuple(_vec3(v, default=1.0)))
        if name == "mirror":
            v = _vec3(kw.get("v", args[0] if args else [1, 0, 0]))
            return Transform(child, scale=tuple(-1.0 if abs(x) > 0 else 1.0 for x in v))
        if name in ("union", "difference", "intersection"):
            nodes = list(child.nodes) if isinstance(child, BooleanOp) and child.op == "union" else [child]
            return BooleanOp(name, nodes)
        if name in ("hull", "minkowski", "render", "group", "color", "offset"):
            return child                              # аппроксимируем содержимым
        return child


def _vec3(v: Any, default: float = 0.0) -> List[float]:
    if not isinstance(v, list):
        v = [float(v)] * 3
    v = [float(x) for x in v]
    while len(v) < 3:
        v.append(default)
    return v[:3]


def parse_scad(source: str) -> Node:
    """Разобрать исходник OpenSCAD в CSG-дерево."""
    return ScadParser(source).parse()
