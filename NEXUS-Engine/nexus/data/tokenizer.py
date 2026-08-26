"""AST-BPE токенизатор: байтовая база + словарь инженерных мёржей.

Обратимый (lossless) токенизатор: 256 байт + служебные токены + жадные мёржи
частых конструкций OpenSCAD/Python/математики. Не требует обучения, поэтому
data-flywheel и юнит-тесты работают из коробки.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

SPECIALS: List[str] = [
    "<pad>", "<bos>", "<eos>", "<unk>",
    "<task>", "<scad>", "<brep>", "<fem>", "<cfd>",
    "<thought>", "<endthought>", "<action>", "<field>",
]

MERGES: List[str] = [
    # OpenSCAD
    "module ", "function ", "difference()", "union()", "intersection()",
    "translate([", "rotate([", "scale([", "mirror([", "linear_extrude(",
    "rotate_extrude(", "cube([", "cylinder(", "sphere(", "polyhedron(",
    "hull()", "minkowski()", "$fn=", "center=true", "center=false",
    "r=", "d=", "h=", "true", "false", "for(", "if(", "else",
    # Python / инженерия
    "def ", "return ", "import ", "numpy", "stress", "von_mises",
    "load", "material", "thickness", "fillet", "chamfer", "rib",
    # разметка
    "\n    ", "\n  ", "\n", "  ", ", ", " = ", ");", "};", "{\n",
]


class ASTBPETokenizer:
    def __init__(self, merges: Sequence[str] = tuple(MERGES)):
        self.specials = list(SPECIALS)
        self.merges = sorted(set(merges), key=len, reverse=True)
        self.special_to_id: Dict[str, int] = {s: 256 + i for i, s in enumerate(self.specials)}
        base = 256 + len(self.specials)
        self.merge_to_id: Dict[str, int] = {m: base + i for i, m in enumerate(self.merges)}
        self.id_to_merge = {v: k for k, v in self.merge_to_id.items()}
        self.id_to_special = {v: k for k, v in self.special_to_id.items()}
        self.vocab_size = base + len(self.merges)

    # ------------------------------------------------------------------ API
    @property
    def pad_id(self) -> int:
        return self.special_to_id["<pad>"]

    @property
    def bos_id(self) -> int:
        return self.special_to_id["<bos>"]

    @property
    def eos_id(self) -> int:
        return self.special_to_id["<eos>"]

    def special(self, name: str) -> int:
        return self.special_to_id[name]

    def encode(self, text: str, bos: bool = False, eos: bool = False) -> List[int]:
        ids: List[int] = [self.bos_id] if bos else []
        i, n = 0, len(text)
        while i < n:
            for sp, sid in self.special_to_id.items():
                if text.startswith(sp, i):
                    ids.append(sid)
                    i += len(sp)
                    break
            else:
                for m in self.merges:
                    if text.startswith(m, i):
                        ids.append(self.merge_to_id[m])
                        i += len(m)
                        break
                else:
                    ids.extend(text[i].encode("utf-8"))
                    i += 1
        if eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        out = bytearray()
        for tid in ids:
            if tid < 256:
                out.append(tid)
            elif tid in self.id_to_special:
                out.extend(self.id_to_special[tid].encode("utf-8"))
            elif tid in self.id_to_merge:
                out.extend(self.id_to_merge[tid].encode("utf-8"))
        return out.decode("utf-8", errors="replace")


DEFAULT_TOKENIZER = ASTBPETokenizer()
