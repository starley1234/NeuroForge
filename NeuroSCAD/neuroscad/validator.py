"""Layered validation: IR always; OpenSCAD and mesh checks when installed."""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping

from .compiler import compile_openscad
from .ir import IRError, Program, validate

_RENDER_TIMEOUT_SECONDS = int(os.getenv("NEUROSCAD_RENDER_TIMEOUT", "30"))


def _openscad_command(executable: str, source: Path, target: Path, manifold: bool = True) -> list[str]:
    command = [executable, "--hardwarnings"]
    if manifold:
        command += ["--backend", "manifold"]
    return command + ["-o", str(target), str(source)]


def render_stl(program: Program, overrides: Mapping[str, float] | None = None) -> tuple[bytes, dict[str, Any]]:
    """Render STL in an isolated temporary directory and return bytes + metrics.

    OS-level sandboxing belongs to the container/runtime. No user-provided SCAD is
    accepted here: only code emitted from validated IR reaches the executable.
    """
    executable = shutil.which("openscad")
    if not executable:
        raise RuntimeError("OpenSCAD CLI is not installed")
    code = compile_openscad(program, overrides)
    with tempfile.TemporaryDirectory(prefix="neuroscad-") as tmp:
        root = Path(tmp)
        source, target = root / "model.scad", root / "model.stl"
        source.write_text(code, encoding="utf-8")
        env = {"PATH": os.environ.get("PATH", ""), "HOME": tmp, "LC_ALL": "C.UTF-8"}
        result = subprocess.run(
            _openscad_command(executable, source, target), capture_output=True,
            text=True, timeout=_RENDER_TIMEOUT_SECONDS, cwd=tmp, env=env,
        )
        # OpenSCAD 2021 packages do not expose --backend; remain deployable there.
        if result.returncode and ("backend" in result.stderr.lower() or "unknown option" in result.stderr.lower()):
            result = subprocess.run(
                _openscad_command(executable, source, target, manifold=False),
                capture_output=True, text=True, timeout=_RENDER_TIMEOUT_SECONDS,
                cwd=tmp, env=env,
            )
        if result.returncode or not target.exists():
            log = (result.stderr or result.stdout or "OpenSCAD produced no STL")[-4000:]
            raise RuntimeError(log)
        payload = target.read_bytes()
        metrics: dict[str, Any] = {"bytes": len(payload)}
        try:
            import trimesh  # type: ignore[import-not-found]
            mesh = trimesh.load_mesh(target, force="mesh")
            metrics.update({
                "watertight": bool(mesh.is_watertight),
                "volume_mm3": float(mesh.volume), "area_mm2": float(mesh.area),
                "center_mass_mm": mesh.center_mass.tolist(),
            })
        except ImportError:
            metrics["mesh_analysis"] = "unavailable"
        return payload, metrics


def validate_program(
    program: Program,
    overrides: Mapping[str, float] | None = None,
    render: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    checks: dict[str, Any] = {}
    try:
        warnings = validate(program, overrides)
        checks["ir"] = {"status": "pass", "warnings": warnings}
    except IRError as exc:
        return {
            "valid": False, "level": "ir",
            "checks": {"ir": {"status": "fail", "error": str(exc)}},
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
        }

    if not render:
        checks["geometry"] = {"status": "not_run", "reason": "render=false"}
    elif not shutil.which("openscad"):
        checks["geometry"] = {"status": "unavailable", "reason": "OpenSCAD CLI is not installed"}
    else:
        try:
            _, metrics = render_stl(program, overrides)
            checks["geometry"] = {"status": "pass", "bytes": metrics["bytes"]}
            if "watertight" in metrics:
                mesh_ok = metrics["watertight"] and metrics["volume_mm3"] > 0
                checks["mesh"] = {"status": "pass" if mesh_ok else "fail", **metrics}
            else:
                checks["mesh"] = {"status": "unavailable", "reason": "install neuroscad[validation]"}
        except (RuntimeError, subprocess.TimeoutExpired) as exc:
            checks["geometry"] = {"status": "fail", "error": str(exc)[-4000:]}

    failed = any(check["status"] == "fail" for check in checks.values())
    level = "mesh" if checks.get("mesh", {}).get("status") == "pass" else (
        "geometry" if checks.get("geometry", {}).get("status") == "pass" else "ir"
    )
    return {
        "valid": not failed, "level": level, "checks": checks,
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
    }
