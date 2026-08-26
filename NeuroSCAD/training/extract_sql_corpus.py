"""Extract OpenSCAD and prompts from a phpMyAdmin stl_items SQL dump.

The dump is parsed as text and is never executed. User/guest identifiers and
answer IDs are intentionally not copied into training sidecars.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

_INSERT = re.compile(r"INSERT\s+INTO\s+`stl_items`\s*\((.*?)\)\s*VALUES\s*", re.I | re.S)
_COLUMNS = re.compile(r"`([^`]+)`")
_ESCAPES = {"0": "\0", "n": "\n", "r": "\r", "t": "\t", "b": "\b", "Z": "\x1a", "\\": "\\", "'": "'", '"': '"'}


def _quoted(text: str, index: int) -> tuple[str, int]:
    assert text[index] == "'"; index += 1; output: list[str] = []
    while index < len(text):
        char = text[index]
        if char == "'":
            if index + 1 < len(text) and text[index + 1] == "'":
                output.append("'"); index += 2; continue
            return "".join(output), index + 1
        if char == "\\" and index + 1 < len(text):
            index += 1; output.append(_ESCAPES.get(text[index], text[index])); index += 1
        else:
            output.append(char); index += 1
    raise ValueError("unterminated SQL string")


def _value(text: str, index: int) -> tuple[Any, int]:
    while index < len(text) and text[index].isspace(): index += 1
    if text[index] == "'": return _quoted(text, index)
    end = index
    while end < len(text) and text[end] not in ",)": end += 1
    raw = text[index:end].strip()
    if raw.upper() == "NULL": return None, end
    try: return int(raw), end
    except ValueError:
        try: return float(raw), end
        except ValueError: return raw, end


def parse_tuples(text: str, index: int) -> tuple[list[list[Any]], int]:
    rows: list[list[Any]] = []
    while index < len(text):
        while index < len(text) and (text[index].isspace() or text[index] == ","): index += 1
        if index >= len(text) or text[index] == ";": return rows, index + 1
        if text[index] != "(": raise ValueError(f"expected tuple at offset {index}")
        index += 1; row: list[Any] = []
        while index < len(text):
            value, index = _value(text, index); row.append(value)
            while index < len(text) and text[index].isspace(): index += 1
            if text[index] == ",": index += 1; continue
            if text[index] == ")": index += 1; break
            raise ValueError(f"expected comma or closing parenthesis at offset {index}")
        rows.append(row)
    return rows, index


def iter_items(sql: str) -> Iterator[dict[str, Any]]:
    position = 0
    while match := _INSERT.search(sql, position):
        columns = _COLUMNS.findall(match.group(1)); rows, position = parse_tuples(sql, match.end())
        for values in rows:
            if len(values) != len(columns): raise ValueError(f"column/value mismatch: {len(columns)} != {len(values)}")
            yield dict(zip(columns, values))


def extract(sql_path: Path, output: Path, license_name: str = "owner-supplied", include_deleted: bool = False) -> dict[str, int]:
    sql = sql_path.read_text(encoding="utf-8", errors="strict"); output.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter(); seen: set[int] = set()
    for item in iter_items(sql):
        item_id = int(item["stl_item_id"]); status = str(item.get("stl_item_status") or "")
        if item_id in seen: counts["duplicate_id"] += 1; continue
        seen.add(item_id)
        if status == "удалено" and not include_deleted: counts["deleted_skipped"] += 1; continue
        code = item.get("stl_item_code"); prompt = item.get("stl_item_code_basis")
        if not isinstance(code, str) or not code.strip(): counts["empty_code"] += 1; continue
        stem = f"stl_item_{item_id}"; (output / f"{stem}.scad").write_text(code.rstrip() + "\n", encoding="utf-8")
        sidecar = {
            "prompt": prompt if isinstance(prompt, str) and prompt.strip() else None,
            "license": license_name, "group_id": stem,
            "tags": ["sql-import"],
            "source": {"table": "stl_items", "stl_item_id": item_id, "status": status,
                       "public": bool(item.get("stl_item_public")), "favorite": bool(item.get("stl_item_favorite")),
                       "image": item.get("stl_item_img"), "created_at": item.get("created_at")},
        }
        (output / f"{stem}.json").write_text(json.dumps(sidecar, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        counts["extracted"] += 1
    manifest = {"format": "neuroscad-sql-extract-v1", "source": str(sql_path), "counts": dict(counts),
                "privacy": "user_id, guest_id and answer_id intentionally omitted"}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return dict(counts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True); parser.add_argument("--license", default="owner-supplied")
    parser.add_argument("--include-deleted", action="store_true"); args = parser.parse_args()
    print(json.dumps(extract(args.input, args.output, args.license, args.include_deleted), ensure_ascii=False, indent=2))

if __name__ == "__main__": main()
