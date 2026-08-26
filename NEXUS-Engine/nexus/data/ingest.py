"""Импорт пользовательских данных «ТЗ → OpenSCAD» в обучающий корпус.

Источники:

===========================  ==================================================
`sql:dump.sql`               дамп MySQL (`INSERT INTO ... VALUES (...)`)
`mysql://user:pass@host/db`  прямое подключение (нужен pymysql)
`jsonl:file.jsonl`           уже выгруженные записи
`csv:file.csv`               экспорт из phpMyAdmin/Excel
===========================  ==================================================

Что делает импорт:

1. читает записи и приводит поля к единому виду (`spec`, `code`, `image`, …);
2. **проверяет каждый скрипт нашим движком**: компиляция CSG, manifold, стенки,
   масса, МКЭ — то, чего нет ни в одном публичном датасете;
3. выбрасывает дубликаты (по хешу нормализованного кода) и мусор;
4. вытаскивает таблицу параметров из шапки скрипта (`name = value; // комментарий`);
5. при желании дописывает недостающие ТЗ внешней LLM (`--enrich-command`);
6. пишет `dataset.jsonl` (готов для `nexus train-lm`), `rejects.jsonl` и `stats.json`.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import subprocess
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence

# ------------------------------------------------------------------ парсинг
_TUPLE_START = re.compile(r"\(")


def parse_sql_dump(text: str, table: Optional[str] = None) -> List[List[Any]]:
    """Разобрать `INSERT INTO ... VALUES (...), (...);` в список кортежей.

    Понимает экранирование MySQL (`\\'`, `\\"`, `\\\\`, `\\n`, `''`), NULL и числа.
    Написан вручную, чтобы не тянуть зависимость от SQL-парсера.
    """
    rows: List[List[Any]] = []
    pos = 0
    pattern = re.compile(r"INSERT\s+INTO\s+`?(?P<table>\w+)`?[^;]*?VALUES", re.IGNORECASE)
    while True:
        match = pattern.search(text, pos)
        if not match:
            return rows
        pos = match.end()
        if table and match.group("table") != table:
            continue
        pos = _parse_value_tuples(text, pos, rows)


def _parse_value_tuples(text: str, pos: int, rows: List[List[Any]]) -> int:
    n = len(text)
    while pos < n:
        while pos < n and text[pos] in " \t\r\n,":
            pos += 1
        if pos >= n or text[pos] == ";":
            return pos + 1
        if text[pos] != "(":
            return pos
        pos += 1
        row: List[Any] = []
        field_chars: List[str] = []
        quote: Optional[str] = None
        quoted = False
        while pos < n:
            ch = text[pos]
            if quote:
                if ch == "\\" and pos + 1 < n:
                    field_chars.append(_unescape(text[pos + 1]))
                    pos += 2
                    continue
                if ch == quote:
                    if pos + 1 < n and text[pos + 1] == quote:   # '' внутри строки
                        field_chars.append(quote)
                        pos += 2
                        continue
                    quote = None
                    pos += 1
                    continue
                field_chars.append(ch)
                pos += 1
                continue
            if ch in "'\"":
                quote = ch
                quoted = True
                field_chars = []          # пробелы перед кавычкой — не часть значения
                pos += 1
                continue
            if ch == ",":
                row.append(_finish_field(field_chars, quoted))
                field_chars, quoted = [], False
                pos += 1
                continue
            if ch == ")":
                row.append(_finish_field(field_chars, quoted))
                pos += 1
                break
            field_chars.append(ch)
            pos += 1
        rows.append(row)
    return pos


def _finish_field(chars: List[str], quoted: bool) -> Any:
    """Строковые значения отдаём как есть, остальные — приводим к числу/NULL."""
    raw = "".join(chars)
    return raw if quoted else _convert(raw)


def _unescape(ch: str) -> str:
    return {"n": "\n", "t": "\t", "r": "\r", "0": "\0", "\\": "\\",
            "'": "'", '"': '"', "b": "\b", "Z": "\x1a"}.get(ch, ch)


def _convert(raw: str) -> Any:
    value = raw.strip()
    if value.upper() == "NULL":
        return None
    if re.fullmatch(r"-?\d+", value):
        return int(value)
    if re.fullmatch(r"-?\d*\.\d+", value):
        return float(value)
    return raw


# ------------------------------------------------------------------ загрузка
DEFAULT_COLUMNS = [
    "stl_item_id", "stl_item_status", "stl_item_code_basis", "stl_item_code",
    "stl_item_favorite", "stl_item_public", "stl_item_img", "stl_item_update",
    "user_id", "guest_id", "answer_id", "created_at",
]

FIELD_MAP = {
    "spec": ("stl_item_code_basis", "spec", "description", "prompt", "task"),
    "code": ("stl_item_code", "code", "scad", "source"),
    "image": ("stl_item_img", "image", "img", "thumbnail"),
    "status": ("stl_item_status", "status"),
    "item_id": ("stl_item_id", "id", "item_id"),
    "created": ("created_at", "created", "date"),
}


def _normalize(record: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for target, candidates in FIELD_MAP.items():
        for name in candidates:
            if name in record and record[name] not in (None, ""):
                out[target] = record[name]
                break
    out.setdefault("spec", "")
    out.setdefault("code", "")
    return out


def load_records(source: str, columns: Sequence[str] = tuple(DEFAULT_COLUMNS),
                 table: str = "stl_items", limit: Optional[int] = None
                 ) -> Iterator[Dict[str, Any]]:
    """Единая точка входа: строка-источник → поток нормализованных записей."""
    scheme, _, rest = source.partition(":")
    if scheme == "sql" or (not rest and source.endswith(".sql")):
        path = rest or source
        with open(path, encoding="utf-8", errors="ignore") as fh:
            text = fh.read()
        for i, row in enumerate(parse_sql_dump(text, table)):
            if limit and i >= limit:
                return
            yield _normalize(dict(zip(columns, row)))
    elif scheme == "jsonl":
        with open(rest, encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                if limit and i >= limit:
                    return
                if line.strip():
                    yield _normalize(json.loads(line))
    elif scheme == "csv":
        with open(rest, encoding="utf-8", newline="") as fh:
            for i, row in enumerate(csv.DictReader(fh)):
                if limit and i >= limit:
                    return
                yield _normalize(row)
    elif source.startswith("mysql://"):
        yield from _load_mysql(source, table, limit)
    else:
        raise ValueError(f"неизвестный источник: {source!r} "
                         f"(ожидается sql:, jsonl:, csv: или mysql://)")


def _load_mysql(dsn: str, table: str, limit: Optional[int]) -> Iterator[Dict[str, Any]]:
    try:
        import pymysql  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("для mysql:// нужен pymysql (pip install pymysql) "
                           "или выгрузите дамп и используйте sql:dump.sql") from exc
    from urllib.parse import urlparse
    url = urlparse(dsn)
    conn = pymysql.connect(host=url.hostname or "localhost", port=url.port or 3306,
                           user=url.username or "root", password=url.password or "",
                           database=(url.path or "/").lstrip("/"), charset="utf8mb4",
                           cursorclass=pymysql.cursors.DictCursor)
    try:
        with conn.cursor() as cur:
            cur.execute(f"SELECT * FROM `{table}`" + (f" LIMIT {int(limit)}" if limit else ""))
            for row in cur.fetchall():
                yield _normalize(row)
    finally:
        conn.close()


# --------------------------------------------------------------- обогащение
PARAM_RE = re.compile(
    r"^\s*(?P<name>[A-Za-z_]\w*)\s*=\s*(?P<value>[^;]+?);\s*(?://\s*(?P<comment>.*))?$",
    re.MULTILINE)


def extract_parameters(code: str, limit: int = 40) -> List[Dict[str, str]]:
    """Таблица параметров из шапки скрипта — ценнейшая часть для обучения."""
    params: List[Dict[str, str]] = []
    for m in PARAM_RE.finditer(code):
        value = m.group("value").strip()
        if len(value) > 80 or value.startswith(("function", "[for")):
            continue
        params.append({"name": m.group("name"), "value": value,
                       "comment": (m.group("comment") or "").strip()})
        if len(params) >= limit:
            break
    return params


def caption_with_llm(code: str, command: Sequence[str], timeout: int = 120) -> str:
    """Попросить внешнюю LLM написать ТЗ по коду (когда описания нет)."""
    prompt = ("Ниже код OpenSCAD. Напиши на русском одно техническое задание "
              "(2-4 предложения), из которого этот код мог бы быть создан: назначение "
              "детали, ключевые размеры и особенности. Без кода и пояснений.\n\n"
              + code[:6000])
    proc = subprocess.run(list(command), input=prompt, capture_output=True,
                          text=True, timeout=timeout)
    return proc.stdout.strip()


# ------------------------------------------------------------------ импорт
@dataclass
class IngestStats:
    total: int = 0
    accepted: int = 0
    duplicates: int = 0
    enriched: int = 0
    rejected: Dict[str, int] = field(default_factory=dict)
    mean_reward: float = 0.0
    seconds: float = 0.0

    def reject(self, reason: str) -> None:
        self.rejected[reason] = self.rejected.get(reason, 0) + 1

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["acceptance_rate"] = round(self.accepted / max(self.total, 1), 3)
        return d


def _code_hash(code: str) -> str:
    normalized = re.sub(r"\s+", " ", re.sub(r"//[^\n]*|/\*.*?\*/", "", code, flags=re.S))
    return hashlib.sha256(normalized.strip().encode("utf-8")).hexdigest()[:16]


def ingest(
    source: str,
    out_dir: str = "artifacts/ingest",
    table: str = "stl_items",
    limit: Optional[int] = None,
    statuses: Optional[Sequence[str]] = None,
    grid: int = 20,
    material: str = "pla",
    force_n: Sequence[float] = (0.0, 0.0, -200.0),
    validate: bool = True,
    min_code_len: int = 40,
    enrich_command: Optional[Sequence[str]] = None,
    min_spec_len: int = 30,
    val_fraction: float = 0.1,
    verbose: bool = True,
) -> IngestStats:
    os.makedirs(out_dir, exist_ok=True)
    from ..mcp.server import NexusTools
    tools = NexusTools(os.path.join(out_dir, "_work"))

    stats = IngestStats()
    seen: Dict[str, int] = {}
    rewards: List[float] = []
    t0 = time.time()

    train_path = os.path.join(out_dir, "dataset.jsonl")
    val_path = os.path.join(out_dir, "val.jsonl")
    rej_path = os.path.join(out_dir, "rejects.jsonl")

    with open(train_path, "w", encoding="utf-8") as train_fh, \
            open(val_path, "w", encoding="utf-8") as val_fh, \
            open(rej_path, "w", encoding="utf-8") as rej_fh:
        for record in load_records(source, table=table, limit=limit):
            stats.total += 1
            code = (record.get("code") or "").strip()
            spec = (record.get("spec") or "").strip()
            status = str(record.get("status") or "")

            if statuses and status not in statuses:
                stats.reject("status")
                continue
            if len(code) < min_code_len:
                stats.reject("too_short")
                rej_fh.write(json.dumps({**record, "reason": "too_short"},
                                        ensure_ascii=False) + "\n")
                continue

            digest = _code_hash(code)
            if digest in seen:
                stats.duplicates += 1
                stats.reject("duplicate")
                continue
            seen[digest] = stats.total

            if len(spec) < min_spec_len and enrich_command:
                try:
                    spec = caption_with_llm(code, enrich_command) or spec
                    stats.enriched += 1
                except Exception as exc:                       # LLM может упасть
                    if verbose:
                        print(f"  [ingest] обогащение не удалось: {exc}")

            report: Dict[str, Any] = {}
            if validate:
                report = tools.score_design(code, list(force_n), "base", material,
                                            grid=grid)
                if report.get("compile", 0) <= 0:
                    stats.reject("compile_error")
                    rej_fh.write(json.dumps(
                        {"item_id": record.get("item_id"), "reason": "compile_error",
                         "error": (report.get("info") or {}).get("error"), "code": code},
                        ensure_ascii=False) + "\n")
                    continue
                rewards.append(float(report.get("total", 0.0)))

            info = report.get("info") or {}
            params = extract_parameters(code)
            text = (("<task>" + spec if spec else "<task>") + "<scad>" + code +
                    ("<fem>" + json.dumps(info, ensure_ascii=False) if info else ""))
            out = {
                "item_id": record.get("item_id"),
                "spec": spec,
                "code": code,
                "image": record.get("image"),
                "status": status,
                "params": params,
                "reward": report.get("total"),
                "physics": info,
                "code_hash": digest,
                "text": text,
            }
            target = val_fh if (stats.accepted % max(int(1 / max(val_fraction, 1e-6)), 1) == 0
                                and val_fraction > 0) else train_fh
            target.write(json.dumps(out, ensure_ascii=False) + "\n")
            stats.accepted += 1
            if verbose and stats.accepted % 50 == 0:
                print(f"  [ingest] принято {stats.accepted}/{stats.total}", flush=True)

    stats.mean_reward = round(sum(rewards) / len(rewards), 3) if rewards else 0.0
    stats.seconds = round(time.time() - t0, 1)
    with open(os.path.join(out_dir, "stats.json"), "w", encoding="utf-8") as fh:
        json.dump(stats.to_dict(), fh, indent=2, ensure_ascii=False)
    return stats
