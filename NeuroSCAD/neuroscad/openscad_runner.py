"""Controlled OpenSCAD execution for trusted offline dataset preparation.

Raw SCAD is never accepted by the public API. This runner is for owner-supplied
training files and must still be executed inside the restricted Docker profile.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

MAX_SOURCE_BYTES = 512 * 1024
_FORBIDDEN_STRICT = {
    "external_include": re.compile(r"(?mi)^\s*(?:include|use)\s*<"),
    "mesh_import": re.compile(r"(?i)\b(?:import|surface)\s*\("),
}
_PARAMETER = re.compile(
    r"(?m)^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*"
    r"(-?\d+(?:\.\d+)?)\s*;\s*(?://\s*\[\s*"
    r"(-?\d+(?:\.\d+)?)\s*:\s*(-?\d+(?:\.\d+)?)\s*:\s*"
    r"(-?\d+(?:\.\d+)?)\s*\])?"
)

CAMERAS: dict[str, str] = {
    "iso_front": "0,0,0,55,0,45,200",
    "iso_back": "0,0,0,55,0,225,200",
    "front": "0,0,0,90,0,0,200",
    "right": "0,0,0,90,0,90,200",
    "back": "0,0,0,90,0,180,200",
    "left": "0,0,0,90,0,270,200",
    "top": "0,0,0,0,0,0,200",
    "bottom": "0,0,0,180,0,0,200",
}


def audit_source(code: str, strict: bool = True) -> list[str]:
    issues: list[str] = []
    if len(code.encode("utf-8")) > MAX_SOURCE_BYTES: issues.append("source_too_large")
    if "\x00" in code: issues.append("nul_byte")
    if strict:
        issues.extend(name for name, pattern in _FORBIDDEN_STRICT.items() if pattern.search(code))
    return issues


def extract_customizer_parameters(code: str) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for match in _PARAMETER.finditer(code):
        item: dict[str, Any] = {"name": match.group(1), "default": float(match.group(2))}
        if match.group(3) is not None:
            item.update(minimum=float(match.group(3)), step=float(match.group(4)), maximum=float(match.group(5)))
        result.append(item)
    return result


def openscad_version() -> str | None:
    executable = shutil.which("openscad")
    if not executable: return None
    result = subprocess.run([executable, "--version"], capture_output=True, text=True, timeout=10)
    return (result.stdout or result.stderr).strip() or None


def _run(command: list[str], cwd: Path, timeout: int) -> subprocess.CompletedProcess[str]:
    env = {"PATH": os.environ.get("PATH", ""), "HOME": str(cwd), "LC_ALL": "C.UTF-8", "QT_QPA_PLATFORM": "offscreen"}
    return subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout)


def compile_file(source: Path, target: Path, timeout: int = 60) -> dict[str, Any]:
    executable = shutil.which("openscad")
    if not executable: raise RuntimeError("OpenSCAD CLI is not installed")
    target.parent.mkdir(parents=True, exist_ok=True)
    command = [executable, "--hardwarnings", "-o", str(target), str(source)]
    result = _run(command, source.parent, timeout)
    if result.returncode or not target.exists() or target.stat().st_size == 0:
        raise RuntimeError((result.stderr or result.stdout or "OpenSCAD produced no output")[-8000:])
    return {"bytes": target.stat().st_size, "log": (result.stderr or result.stdout)[-4000:]}


def render_views(source: Path, output_dir: Path, views: tuple[str, ...] = tuple(CAMERAS), size: int = 512, timeout: int = 60) -> list[str]:
    executable = shutil.which("openscad")
    if not executable: raise RuntimeError("OpenSCAD CLI is not installed")
    output_dir.mkdir(parents=True, exist_ok=True); outputs: list[str] = []
    for view in views:
        if view not in CAMERAS: raise ValueError(f"unknown camera: {view}")
        target = output_dir / f"{view}.png"
        command = [executable, "--render", "--autocenter", "--viewall", "--projection", "o",
                   "--imgsize", f"{size},{size}", "--camera", CAMERAS[view], "-o", str(target), str(source)]
        result = _run(command, source.parent, timeout)
        if result.returncode or not target.exists():
            raise RuntimeError(f"render {view} failed: {(result.stderr or result.stdout)[-4000:]}")
        outputs.append(str(target))
    return outputs


def mesh_metrics(stl: Path) -> dict[str, Any]:
    try:
        import trimesh  # type: ignore[import-not-found]
    except ImportError:
        return {"analysis": "unavailable"}
    mesh = trimesh.load_mesh(stl, force="mesh")
    return {
        "watertight": bool(mesh.is_watertight), "volume_mm3": float(mesh.volume),
        "area_mm2": float(mesh.area), "components": len(mesh.split(only_watertight=False)),
        "bounds_mm": mesh.bounds.tolist(), "center_mass_mm": mesh.center_mass.tolist(),
        "vertices": len(mesh.vertices), "faces": len(mesh.faces),
    }
