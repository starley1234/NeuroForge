"""Байтовый BPE-токенизатор с обучением на корпусе (в стиле minbpe).

Зачем: ручные мёржи в `ASTBPETokenizer` дают избыточно длинные
последовательности. Обученный на своём корпусе BPE сокращает длину в 2–4 раза,
то есть во столько же раз дешевле обучение и инференс.

Особенности:
* байтовая база (256) — токенизатор всегда lossless, `decode(encode(x)) == x`;
* регексное пред-разбиение (как в GPT-2/4), чтобы мёржи не перескакивали
  границы слов и не съедали пробелы вместе с кодом;
* служебные токены (`<task>`, `<scad>`, `<fem>`, …) с фиксированными id;
* формат хранения — обычный JSON, без внешних зависимостей.
"""
from __future__ import annotations

import json
import os
import re
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

SPLIT_PATTERN = re.compile(
    r"'(?:[sdmt]|ll|ve|re)|[^\r\n\w]?\w+|\d{1,3}| ?[^\s\w]+[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"
)

SPECIALS: List[str] = [
    "<pad>", "<bos>", "<eos>", "<unk>",
    "<task>", "<scad>", "<brep>", "<fem>", "<cfd>",
    "<thought>", "<endthought>", "<action>", "<field>",
]


def _pair_counts(chunks: List[List[int]], counts: Optional[Counter] = None) -> Counter:
    counts = counts if counts is not None else Counter()
    for chunk in chunks:
        for pair in zip(chunk, chunk[1:]):
            counts[pair] += 1
    return counts


def _merge(chunk: List[int], pair: Tuple[int, int], new_id: int) -> List[int]:
    out: List[int] = []
    i = 0
    while i < len(chunk):
        if i < len(chunk) - 1 and chunk[i] == pair[0] and chunk[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(chunk[i])
            i += 1
    return out


class BPETokenizer:
    """Обучаемый байтовый BPE. Совместим по API с `ASTBPETokenizer`."""

    def __init__(self, merges: Optional[Dict[Tuple[int, int], int]] = None,
                 specials: Sequence[str] = tuple(SPECIALS)):
        self.merges: Dict[Tuple[int, int], int] = dict(merges or {})
        self.specials = list(specials)
        self._rebuild()

    # ------------------------------------------------------------- служебное
    def _rebuild(self) -> None:
        self.vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        for (a, b), idx in sorted(self.merges.items(), key=lambda kv: kv[1]):
            self.vocab[idx] = self.vocab[a] + self.vocab[b]
        base = 256 + len(self.merges)
        self.special_to_id = {s: base + i for i, s in enumerate(self.specials)}
        self.id_to_special = {v: k for k, v in self.special_to_id.items()}
        self.vocab_size = base + len(self.specials)
        self._cache: Dict[str, List[int]] = {}

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

    # -------------------------------------------------------------- обучение
    def train(self, texts: Iterable[str], vocab_size: int = 8192,
              verbose: bool = False, max_chars: Optional[int] = None) -> "BPETokenizer":
        """Обучить мёржи на корпусе. vocab_size включает 256 байт и спецтокены."""
        target_merges = max(0, vocab_size - 256 - len(self.specials))
        chunks: List[List[int]] = []
        total = 0
        for text in texts:
            for piece in SPLIT_PATTERN.findall(text):
                raw = piece.encode("utf-8")
                total += len(raw)
                chunks.append(list(raw))
            if max_chars and total >= max_chars:
                break
        if not chunks:
            raise ValueError("пустой корпус для обучения токенизатора")

        self.merges = {}
        for i in range(target_merges):
            counts = _pair_counts(chunks)
            if not counts:
                break
            pair, freq = counts.most_common(1)[0]
            if freq < 2:
                break
            new_id = 256 + i
            self.merges[pair] = new_id
            chunks = [_merge(c, pair, new_id) if len(c) > 1 else c for c in chunks]
            if verbose and (i + 1) % 200 == 0:
                print(f"  [bpe] мёрж {i + 1}/{target_merges} freq={freq}", flush=True)
        self._rebuild()
        return self

    # ------------------------------------------------------------- кодирование
    def _encode_piece(self, piece: str) -> List[int]:
        if piece in self._cache:
            return self._cache[piece]
        ids = list(piece.encode("utf-8"))
        while len(ids) >= 2:
            pairs = set(zip(ids, ids[1:]))
            best = min(pairs, key=lambda p: self.merges.get(p, float("inf")))
            if best not in self.merges:
                break
            ids = _merge(ids, best, self.merges[best])
        if len(self._cache) < 100_000:
            self._cache[piece] = ids
        return ids

    def encode(self, text: str, bos: bool = False, eos: bool = False) -> List[int]:
        ids: List[int] = [self.bos_id] if bos else []
        rest = text
        while rest:
            hit = None
            for sp in self.specials:                       # спецтокены не режем
                pos = rest.find(sp)
                if pos != -1 and (hit is None or pos < hit[0]):
                    hit = (pos, sp)
            if hit is None:
                for piece in SPLIT_PATTERN.findall(rest):
                    ids.extend(self._encode_piece(piece))
                break
            pos, sp = hit
            for piece in SPLIT_PATTERN.findall(rest[:pos]):
                ids.extend(self._encode_piece(piece))
            ids.append(self.special_to_id[sp])
            rest = rest[pos + len(sp):]
        if eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        out = bytearray()
        for tid in ids:
            if tid in self.id_to_special:
                out.extend(self.id_to_special[tid].encode("utf-8"))
            elif tid in self.vocab:
                out.extend(self.vocab[tid])
        return out.decode("utf-8", errors="replace")

    # -------------------------------------------------------------- хранение
    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = {
            "version": 1,
            "specials": self.specials,
            "merges": [[a, b, idx] for (a, b), idx in
                       sorted(self.merges.items(), key=lambda kv: kv[1])],
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False)
        return path

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        with open(path, encoding="utf-8") as fh:
            payload = json.load(fh)
        merges = {(a, b): idx for a, b, idx in payload["merges"]}
        return cls(merges, payload.get("specials", SPECIALS))

    # ------------------------------------------------------------- статистика
    def compression(self, text: str) -> float:
        """Во сколько раз короче байтовой длины (чем больше, тем лучше)."""
        return len(text.encode("utf-8")) / max(len(self.encode(text)), 1)


def train_from_source(source: str = "builtin:engineering", vocab_size: int = 4096,
                      out: str = "artifacts/tokenizer/bpe.json",
                      limit: Optional[int] = None, max_chars: Optional[int] = 2_000_000,
                      verbose: bool = True) -> Tuple[BPETokenizer, Dict[str, float]]:
    from .corpora import iter_texts
    texts = list(iter_texts(source, limit=limit))
    tok = BPETokenizer().train(texts, vocab_size=vocab_size, verbose=verbose,
                               max_chars=max_chars)
    tok.save(out)
    sample = "\n".join(texts[: min(20, len(texts))])
    stats = {
        "vocab_size": float(tok.vocab_size),
        "merges": float(len(tok.merges)),
        "documents": float(len(texts)),
        "compression_bytes_per_token": round(tok.compression(sample), 3),
    }
    return tok, stats


def load_tokenizer(path: Optional[str]):
    """BPE из файла, если он есть; иначе — встроенный AST-BPE."""
    from .tokenizer import DEFAULT_TOKENIZER
    if path and os.path.exists(path):
        return BPETokenizer.load(path)
    return DEFAULT_TOKENIZER
