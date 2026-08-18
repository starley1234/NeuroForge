"""Загрузка стандартных текстовых корпусов и упаковка в LM-блоки.

Поддерживаемые источники (строка-спецификация `source`):

===========================  ==================================================
`builtin:engineering`        встроенный мини-корпус (работает офлайн, для CI)
`dir:PATH`                   все *.txt/*.md/*.scad/*.py в каталоге рекурсивно
`file:PATH`                  один текстовый файл
`jsonl:PATH#field`           JSONL, поле `field` (по умолчанию `text`)
`flywheel:PATH`              артефакты маховика (ТЗ + код + сводка FEM)
`hf:NAME[/CONFIG][:SPLIT]#F` HuggingFace `datasets` (если библиотека доступна)
===========================  ==================================================
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Iterable, Iterator, List, Optional, Sequence

import torch
from torch.utils.data import Dataset

from .tokenizer import DEFAULT_TOKENIZER, ASTBPETokenizer

TEXT_EXTENSIONS = (".txt", ".md", ".scad", ".py", ".c", ".cpp", ".h", ".json")

BUILTIN_ENGINEERING = [
    "Кронштейн воспринимает изгибающий момент M = F·L; напряжение σ = M·y/I, "
    "где I — момент инерции сечения. Для прямоугольного сечения I = b·h³/12.",
    "Коэффициент запаса прочности n = σ_т / σ_max. Для ответственных узлов n ≥ 2.0, "
    "для декоративных деталей достаточно n ≥ 1.2.",
    "module bracket(w=40, h=45, t=4) { difference() { cube([w, t, h]); "
    "translate([w/2, -1, h*0.7]) rotate([-90,0,0]) cylinder(h=t+2, r=2.5); } }",
    "Минимальная толщина стенки для FDM-печати PLA соплом 0.4 мм составляет 1.2 мм "
    "(три периметра). Свесы более 45° требуют поддержек.",
    "difference() { cylinder(h=6, r=35); cylinder(h=20, r=10, center=true); }",
    "Плотность материалов, кг/м³: PLA 1240, PETG 1270, ABS 1040, алюминий 6061 — 2700, "
    "сталь 304 — 8000, титан Ti-6Al-4V — 4430.",
    "Von Mises stress combines principal stresses: sqrt(0.5*((s1-s2)^2 + (s2-s3)^2 + (s3-s1)^2)).",
    "Для фрезеровки на 3-осевом станке все поверхности должны быть достижимы инструментом "
    "сверху; внутренние углы имеют радиус не меньше радиуса фрезы.",
    "def safety_factor(yield_pa, sigma_max_pa):\n    return yield_pa / max(sigma_max_pa, 1e-6)",
    "Резьба M5 требует отверстия под нарезание диаметром 4.2 мм, сквозного — 5.5 мм.",
    "Момент затяжки болта M6 класса 8.8 составляет примерно 10 Н·м при коэффициенте трения 0.14.",
    "translate([0,0,10]) rotate([0,90,0]) cylinder(h=20, r1=5, r2=3, center=true);",
    "Тепловое расширение алюминия: 23·10⁻⁶ 1/К. При Δt = 80 К деталь длиной 200 мм "
    "удлиняется на 0.37 мм — это учитывают в посадках.",
    "Посадка с натягом H7/p6 применяется для установки подшипника в корпус; "
    "зазорная H7/g6 — для скользящих соединений.",
]


# ---------------------------------------------------------------- источники
def _iter_dir(path: str) -> Iterator[str]:
    for root, _, files in os.walk(path):
        for fn in sorted(files):
            if fn.lower().endswith(TEXT_EXTENSIONS):
                with open(os.path.join(root, fn), encoding="utf-8", errors="ignore") as fh:
                    text = fh.read().strip()
                if text:
                    yield text


def _iter_jsonl(path: str, field: str = "text") -> Iterator[str]:
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            value = rec.get(field)
            if isinstance(value, str) and value:
                yield value


def _iter_flywheel(path: str) -> Iterator[str]:
    ds = os.path.join(path, "dataset.jsonl") if os.path.isdir(path) else path
    with open(ds, encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            fem = rec.get("fem", {})
            yield (rec.get("spec", "") + "<scad>" + rec.get("code", "") + "<fem>" +
                   json.dumps({"mass_g": round(rec.get("mass", {}).get("mass_g", 0.0), 3),
                               "sigma_max_mpa": round(fem.get("max_stress_pa", 0.0) / 1e6, 3),
                               "safety_factor": round(fem.get("safety_factor", 0.0), 3)},
                              ensure_ascii=False))


def _iter_hf(spec: str) -> Iterator[str]:
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:  # pragma: no cover - зависит от окружения
        raise RuntimeError(
            "источник hf: требует пакет `datasets` (pip install datasets)"
        ) from exc
    field = "text"
    if "#" in spec:
        spec, field = spec.split("#", 1)
    split = "train"
    if ":" in spec:
        spec, split = spec.rsplit(":", 1)
    name, config = (spec.split("/", 1) + [None])[:2] if "/" in spec else (spec, None)
    ds = load_dataset(name, config, split=split, streaming=True)
    for rec in ds:
        value = rec.get(field)
        if isinstance(value, str) and value.strip():
            yield value


def iter_texts(source: str, limit: Optional[int] = None) -> Iterator[str]:
    """Единая точка входа: спецификация источника → поток текстов."""
    scheme, _, rest = source.partition(":")
    if not rest and scheme not in ("builtin",):
        scheme, rest = ("dir" if os.path.isdir(source) else "file"), source

    if scheme == "builtin":
        stream: Iterable[str] = BUILTIN_ENGINEERING
    elif scheme == "dir":
        stream = _iter_dir(rest)
    elif scheme == "file":
        with open(rest, encoding="utf-8", errors="ignore") as fh:
            stream = [fh.read()]
    elif scheme == "jsonl":
        path, _, fld = rest.partition("#")
        stream = _iter_jsonl(path, fld or "text")
    elif scheme == "flywheel":
        stream = _iter_flywheel(rest)
    elif scheme == "hf":
        stream = _iter_hf(rest)
    else:
        raise ValueError(f"неизвестный источник данных: {source!r}")

    for i, text in enumerate(stream):
        if limit is not None and i >= limit:
            return
        yield text


# ------------------------------------------------------------------ датасет
@dataclass
class CorpusStats:
    documents: int
    tokens: int
    blocks: int
    seq_len: int

    def to_dict(self):
        return self.__dict__


class PackedLMDataset(Dataset):
    """Классическая LM-упаковка: тексты склеиваются и режутся на блоки seq_len+1."""

    def __init__(self, source: str, tokenizer: Optional[ASTBPETokenizer] = None,
                 seq_len: int = 512, limit: Optional[int] = None,
                 min_blocks: int = 1, repeat_if_small: bool = True):
        self.tok = tokenizer or DEFAULT_TOKENIZER
        self.seq_len = seq_len
        ids: List[int] = []
        docs = 0
        for text in iter_texts(source, limit=limit):
            ids.extend(self.tok.encode(text, bos=True, eos=True))
            docs += 1
        if not ids:
            raise ValueError(f"источник {source!r} не дал ни одного токена")
        if repeat_if_small:
            need = (seq_len + 1) * min_blocks
            while len(ids) < need:
                ids.extend(ids[: max(1, need - len(ids))])
        n_blocks = max(1, (len(ids) - 1) // seq_len)
        self.data = torch.tensor(ids[: n_blocks * seq_len + 1], dtype=torch.long)
        self.n_blocks = n_blocks
        self.stats = CorpusStats(docs, len(ids), n_blocks, seq_len)

    def __len__(self) -> int:
        return self.n_blocks

    def __getitem__(self, i: int):
        chunk = self.data[i * self.seq_len: i * self.seq_len + self.seq_len + 1]
        return {"tokens": chunk[:-1].clone(), "targets": chunk[1:].clone()}


def split_dataset(ds: Dataset, val_fraction: float = 0.1, seed: int = 0):
    from torch.utils.data import random_split
    n_val = max(1, int(len(ds) * val_fraction)) if len(ds) > 1 else 0
    if n_val == 0:
        return ds, None
    return random_split(ds, [len(ds) - n_val, n_val],
                        generator=torch.Generator().manual_seed(seed))
