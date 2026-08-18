"""Fetch ESC-50 (Karol Piczak) — 2000 clips, 50 classes, CC BY-NC."""

from __future__ import annotations

import csv
import io
import zipfile
from pathlib import Path
from urllib.request import urlopen

from aura_micro.catalog import map_esc50
from aura_micro.paths import ESC50_DIR

# Pinned snapshot (~600 MB wav). Fallback: GitHub archive.
ESC50_URLS = (
    "https://github.com/karoldvl/ESC-50/archive/master.zip",
    "https://github.com/karolpiczak/ESC-50/archive/refs/heads/master.zip",
)


def esc50_ready(root: Path | None = None) -> bool:
    root = root or ESC50_DIR
    audio = _audio_dir(root)
    return audio is not None and any(audio.glob("*.wav"))


def _audio_dir(root: Path) -> Path | None:
    if (root / "audio").is_dir():
        return root / "audio"
    for p in root.glob("**/audio"):
        if p.is_dir():
            return p
    return None


def _meta_path(root: Path) -> Path | None:
    if (root / "meta" / "esc50.csv").is_file():
        return root / "meta" / "esc50.csv"
    hits = list(root.glob("**/meta/esc50.csv"))
    return hits[0] if hits else None


def list_esc50_items(root: Path | None = None) -> list[tuple[Path, int, str]]:
    root = root or ESC50_DIR
    audio = _audio_dir(root)
    meta = _meta_path(root)
    if audio is None or meta is None:
        return []
    items: list[tuple[Path, int, str]] = []
    with meta.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cid = map_esc50(row["category"])
            if cid is None:
                continue
            path = audio / row["filename"]
            if path.is_file():
                items.append((path, cid, row["category"]))
    return items


def download_esc50(root: Path | None = None, timeout: int = 180) -> Path:
    root = root or ESC50_DIR
    if esc50_ready(root):
        return root
    root.mkdir(parents=True, exist_ok=True)
    last_err: Exception | None = None
    for url in ESC50_URLS:
        try:
            print(f"downloading ESC-50 from {url} …")
            with urlopen(url, timeout=timeout) as resp:
                blob = resp.read()
            with zipfile.ZipFile(io.BytesIO(blob)) as zf:
                zf.extractall(root)
            if esc50_ready(root):
                print(f"ESC-50 ready at {root}")
                return root
        except Exception as exc:  # noqa: BLE001 — network fallbacks
            last_err = exc
            print(f"failed: {exc}")
    raise RuntimeError(f"could not download ESC-50: {last_err}")
