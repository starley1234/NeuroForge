"""Рендер OpenSCAD в изображения: выборка для визуальной модальности.

Два бэкенда:

* **OpenSCAD CLI** — если бинарник есть, снимает деталь с нескольких ракурсов
  (`--camera`, `--viewall`, `--autocenter`) в PNG нужного размера;
* **встроенный растеризатор** — воксель + Z-буфер + освещение по Ламберту,
  чистый numpy и своя запись PNG (zlib), работает без OpenSCAD и без Pillow.

Результат: каталог с картинками и `manifest.jsonl`, где к каждому изображению
привязаны код, ТЗ, параметры и физика детали. Это готовый датасет для обучения
энкодера «картинка → код» (`nexus.encoders.audio_visual.VideoSSMEncoder` или
любой другой визуальный вход).
"""
from __future__ import annotations

import json
import math
import os
import shutil
import struct
import subprocess
import tempfile
import time
import zlib
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# Стандартные ракурсы: (поворот вокруг X, поворот вокруг Z)
DEFAULT_VIEWS: List[Tuple[str, Tuple[float, float]]] = [
    ("iso", (55.0, 25.0)),
    ("front", (90.0, 0.0)),
    ("right", (90.0, 90.0)),
    ("top", (0.0, 0.0)),
    ("iso_back", (55.0, 205.0)),
    ("bottom", (180.0, 0.0)),
]


# ------------------------------------------------------------------- PNG
def write_png(path: str, rgb: np.ndarray) -> str:
    """Сохранить массив (H, W, 3) uint8 в PNG без внешних библиотек."""
    height, width, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(height))

    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    header = struct.pack(">2I5B", width, height, 8, 2, 0, 0, 0)
    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header)
           + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(png)
    return path


# ------------------------------------------------------- встроенный рендер
def _rotation(rx_deg: float, rz_deg: float) -> np.ndarray:
    rx, rz = math.radians(rx_deg), math.radians(rz_deg)
    cx, sx, cz, sz = math.cos(rx), math.sin(rx), math.cos(rz), math.sin(rz)
    return np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]]) @ \
           np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])


def render_internal(code: str, out_path: str, view: Tuple[float, float] = (55.0, 25.0),
                    size: int = 384, resolution: int = 56,
                    background: Tuple[int, int, int] = (24, 26, 32),
                    color: Tuple[int, int, int] = (215, 190, 120)) -> Optional[str]:
    """Ортографическая проекция воксельной модели с затенением по Ламберту."""
    from ..geometry.voxel import voxelize
    from ..scad.parser import parse_scad

    try:
        tree = parse_scad(code)
        vox = voxelize(tree, resolution=resolution)
    except Exception:
        return None
    occ = vox.occupancy
    if not occ.any():
        return None

    idx = np.argwhere(occ).astype(np.float64)
    pts = vox.origin + (idx + 0.5) * vox.spacing[None, :]
    pts -= pts.mean(axis=0)

    rot = _rotation(*view)
    cam = pts @ rot.T                                    # x — вправо, y — вглубь, z — вверх
    scale = size * 0.42 / max(np.abs(cam[:, [0, 2]]).max(), 1e-6)
    px = np.clip((cam[:, 0] * scale + size / 2).astype(int), 0, size - 1)
    py = np.clip((size / 2 - cam[:, 2] * scale).astype(int), 0, size - 1)
    depth = cam[:, 1]

    zbuf = np.full((size, size), np.inf)
    normals = np.zeros((size, size, 3))
    voxel_normals = _surface_normals(occ)[tuple(np.argwhere(occ).T)] @ rot.T

    radius = max(1, int(scale * float(vox.spacing.max()) / 2))
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            xs = np.clip(px + dx, 0, size - 1)
            ys = np.clip(py + dy, 0, size - 1)
            better = depth < zbuf[ys, xs]
            zbuf[ys[better], xs[better]] = depth[better]
            normals[ys[better], xs[better]] = voxel_normals[better]

    light = np.array([0.4, -0.7, 0.6])
    light = light / np.linalg.norm(light)
    shade = np.clip(normals @ light, 0.0, 1.0) * 0.75 + 0.25
    image = np.zeros((size, size, 3), dtype=np.uint8)
    image[:, :] = background
    mask = np.isfinite(zbuf)
    for c in range(3):
        image[..., c] = np.where(mask, np.clip(shade * color[c], 0, 255), background[c])
    return write_png(out_path, image)


def _surface_normals(occ: np.ndarray) -> np.ndarray:
    """Грубые нормали: градиент занятости, сглаженный соседями."""
    filled = occ.astype(np.float32)
    gx, gy, gz = np.gradient(filled)
    normals = np.stack([-gx, -gy, -gz], axis=-1)
    norm = np.linalg.norm(normals, axis=-1, keepdims=True)
    normals = np.divide(normals, np.maximum(norm, 1e-6))
    flat = norm[..., 0] < 1e-6
    normals[flat] = np.array([0.0, 0.0, 1.0])
    return normals


# ------------------------------------------------------------- OpenSCAD CLI
def openscad_binary() -> Optional[str]:
    from ..scad.render import openscad_binary as find
    return find()


def render_openscad(code: str, out_path: str, view: Tuple[float, float] = (55.0, 25.0),
                    size: int = 512, colorscheme: str = "Tomorrow Night",
                    timeout: int = 180, extra_args: Sequence[str] = ()) -> Optional[str]:
    binary = openscad_binary()
    if not binary:
        return None
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        scad = os.path.join(tmp, "model.scad")
        with open(scad, "w", encoding="utf-8") as fh:
            fh.write(code)
        camera = f"0,0,0,{view[0]},0,{view[1]},0"          # dist=0 + --viewall
        cmd = [binary, "-o", out_path, f"--imgsize={size},{size}",
               f"--camera={camera}", "--viewall", "--autocenter",
               f"--colorscheme={colorscheme}", *extra_args, scad]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except (OSError, subprocess.TimeoutExpired):
            return None
    return out_path if proc.returncode == 0 and os.path.exists(out_path) else None


def export_stl(code: str, out_path: str, timeout: int = 300) -> Optional[str]:
    """STL через OpenSCAD (точная сетка) либо через встроенный воксельный движок."""
    binary = openscad_binary()
    if binary:
        with tempfile.TemporaryDirectory() as tmp:
            scad = os.path.join(tmp, "model.scad")
            with open(scad, "w", encoding="utf-8") as fh:
                fh.write(code)
            try:
                proc = subprocess.run([binary, "-o", out_path, scad],
                                      capture_output=True, text=True, timeout=timeout)
                if proc.returncode == 0 and os.path.exists(out_path):
                    return out_path
            except (OSError, subprocess.TimeoutExpired):
                pass
    from ..scad.render import render as internal_render
    res = internal_render(code, resolution=48, stl_path=out_path)
    return out_path if res.ok and os.path.exists(out_path) else None


# --------------------------------------------------------------- датасет
@dataclass
class RenderStats:
    items: int = 0
    images: int = 0
    failed: int = 0
    backend: str = "internal"
    seconds: float = 0.0
    errors: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def render_views(code: str, out_dir: str, stem: str,
                 views: Sequence[Tuple[str, Tuple[float, float]]] = tuple(DEFAULT_VIEWS),
                 size: int = 384, backend: str = "auto",
                 resolution: int = 56) -> List[Dict[str, Any]]:
    """Снять деталь с нескольких ракурсов. Возвращает список описаний картинок."""
    use_openscad = backend == "openscad" or (backend == "auto" and openscad_binary())
    out: List[Dict[str, Any]] = []
    for name, angles in views:
        path = os.path.join(out_dir, f"{stem}_{name}.png")
        made = render_openscad(code, path, angles, size) if use_openscad else None
        engine = "openscad"
        if made is None:
            made = render_internal(code, path, angles, size, resolution)
            engine = "internal"
        if made:
            out.append({"image": os.path.relpath(made, out_dir), "view": name,
                        "angles": list(angles), "size": size, "renderer": engine})
    return out


def build_vision_dataset(
    records: Iterable[Dict[str, Any]],
    out_dir: str = "artifacts/vision",
    size: int = 384,
    backend: str = "auto",
    views: Optional[Sequence[str]] = None,
    resolution: int = 56,
    with_stl: bool = False,
    limit: Optional[int] = None,
    verbose: bool = True,
) -> RenderStats:
    """Из записей `ingest` сделать выборку «изображение → код»."""
    os.makedirs(out_dir, exist_ok=True)
    chosen = [v for v in DEFAULT_VIEWS if not views or v[0] in views]
    stats = RenderStats(backend="openscad" if (backend != "internal" and openscad_binary())
                        else "internal")
    t0 = time.time()
    manifest_path = os.path.join(out_dir, "manifest.jsonl")

    with open(manifest_path, "w", encoding="utf-8") as fh:
        for i, record in enumerate(records):
            if limit and i >= limit:
                break
            code = (record.get("code") or "").strip()
            if not code:
                continue
            stats.items += 1
            stem = f"item_{record.get('item_id') or i:06d}"
            try:
                images = render_views(code, out_dir, stem, chosen, size, backend, resolution)
            except Exception as exc:                       # рендер не должен ронять прогон
                stats.failed += 1
                stats.errors[type(exc).__name__] = stats.errors.get(type(exc).__name__, 0) + 1
                continue
            if not images:
                stats.failed += 1
                stats.errors["no_image"] = stats.errors.get("no_image", 0) + 1
                continue

            stl_rel = None
            if with_stl:
                stl_path = os.path.join(out_dir, f"{stem}.stl")
                stl_rel = export_stl(code, stl_path)
                stl_rel = os.path.relpath(stl_rel, out_dir) if stl_rel else None

            stats.images += len(images)
            fh.write(json.dumps({
                "item_id": record.get("item_id"),
                "spec": record.get("spec", ""),
                "code": code,
                "params": record.get("params", []),
                "physics": record.get("physics", {}),
                "images": images,
                "stl": stl_rel,
                # готовая обучающая строка для мультимодальной пары
                "text": ("<task>" + (record.get("spec") or "") + "<scad>" + code),
            }, ensure_ascii=False) + "\n")
            if verbose and stats.items % 20 == 0:
                print(f"  [vision] {stats.items} деталей, {stats.images} картинок",
                      flush=True)

    stats.seconds = round(time.time() - t0, 1)
    with open(os.path.join(out_dir, "stats.json"), "w", encoding="utf-8") as fh:
        json.dump(stats.to_dict(), fh, indent=2, ensure_ascii=False)
    return stats


def load_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def load_image(path: str) -> np.ndarray:
    """Прочитать PNG обратно в массив (H, W, 3) float32 [0..1] — для обучения."""
    with open(path, "rb") as fh:
        data = fh.read()
    pos, width, height, idat = 8, 0, 0, b""
    while pos < len(data):
        length = struct.unpack(">I", data[pos:pos + 4])[0]
        tag = data[pos + 4:pos + 8]
        body = data[pos + 8:pos + 8 + length]
        if tag == b"IHDR":
            width, height = struct.unpack(">2I", body[:8])
        elif tag == b"IDAT":
            idat += body
        pos += 12 + length
    raw = zlib.decompress(idat)
    stride = width * 3
    out = np.zeros((height, width, 3), dtype=np.uint8)
    prev = np.zeros(stride, dtype=np.uint8)
    for y in range(height):
        start = y * (stride + 1)
        filt = raw[start]
        line = np.frombuffer(raw[start + 1:start + 1 + stride], dtype=np.uint8).copy()
        if filt == 1:
            for i in range(3, stride):
                line[i] = (int(line[i]) + int(line[i - 3])) % 256
        elif filt == 2:
            line = ((line.astype(int) + prev.astype(int)) % 256).astype(np.uint8)
        elif filt != 0:                                    # 3/4 в нашем writer не бывают
            raise ValueError(f"неподдерживаемый PNG-фильтр {filt}")
        out[y] = line.reshape(width, 3)
        prev = line
    return out.astype(np.float32) / 255.0
