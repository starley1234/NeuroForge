"""Layered validation: IR always; OpenSCAD and mesh checks when installed."""
from __future__ import annotations
import json, shutil, subprocess, tempfile, time
from pathlib import Path
from typing import Any, Mapping
from .compiler import compile_openscad
from .ir import IRError, Program, validate


def validate_program(program: Program, overrides: Mapping[str, float] | None = None, render: bool = False) -> dict[str, Any]:
    started = time.perf_counter()
    checks: dict[str, Any] = {}
    try:
        warnings = validate(program, overrides)
        checks["ir"] = {"status": "pass", "warnings": warnings}
    except IRError as exc:
        return {"valid": False, "level": "ir", "checks": {"ir": {"status": "fail", "error": str(exc)}}}
    code = compile_openscad(program)
    openscad = shutil.which("openscad")
    if not render:
        checks["geometry"] = {"status": "not_run", "reason": "render=false"}
    elif not openscad:
        checks["geometry"] = {"status": "unavailable", "reason": "OpenSCAD CLI is not installed"}
    else:
        with tempfile.TemporaryDirectory(prefix="neuroscad-") as tmp:
            source, stl = Path(tmp) / "model.scad", Path(tmp) / "model.stl"
            source.write_text(code, encoding="utf-8")
            result = subprocess.run([openscad, "--hardwarnings", "--backend", "manifold", "-o", str(stl), str(source)], capture_output=True, text=True, timeout=30)
            if result.returncode:
                checks["geometry"] = {"status": "fail", "log": (result.stderr or result.stdout)[-4000:]}
            else:
                checks["geometry"] = {"status": "pass", "bytes": stl.stat().st_size}
                try:
                    import trimesh  # optional
                    mesh = trimesh.load_mesh(stl, force="mesh")
                    checks["mesh"] = {"status": "pass" if mesh.is_watertight and mesh.volume > 0 else "fail", "watertight": bool(mesh.is_watertight), "volume_mm3": float(mesh.volume), "area_mm2": float(mesh.area), "center_mass_mm": mesh.center_mass.tolist()}
                except ImportError:
                    checks["mesh"] = {"status": "unavailable", "reason": "install neuroscad[validation]"}
    failed = any(c["status"] == "fail" for c in checks.values())
    level = "mesh" if checks.get("mesh", {}).get("status") == "pass" else ("geometry" if checks.get("geometry", {}).get("status") == "pass" else "ir")
    return {"valid": not failed, "level": level, "checks": checks, "elapsed_ms": round((time.perf_counter() - started) * 1000, 3)}
