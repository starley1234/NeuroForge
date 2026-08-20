"""Production HTTP adapter: typed requests, safe compiler and downloadable artifacts."""
from __future__ import annotations

import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import Response
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel, ConfigDict, Field
except ImportError as exc:
    raise RuntimeError("Install API dependencies: pip install -e '.[api]'") from exc

from .compiler import compile_openscad
from .ir import IRError, Program
from .templates import generate
from .validator import render_stl, validate_parameter_sweep, validate_program

VERSION = "0.4.0"
app = FastAPI(
    title="NeuroSCAD", version=VERSION,
    docs_url="/api/docs", openapi_url="/api/openapi.json", redoc_url=None,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GenerateRequest(StrictModel):
    prompt: str = Field(min_length=3, max_length=2000)
    validate_geometry: bool = False


class ProgramRequest(StrictModel):
    ir: dict[str, Any]
    overrides: dict[str, float] = Field(default_factory=dict)
    validate_geometry: bool = False


def _program(data: dict[str, Any]) -> Program:
    try:
        return Program.from_dict(data)
    except (KeyError, TypeError, ValueError, IRError) as exc:
        raise HTTPException(422, f"Invalid CSG-IR: {exc}") from exc


@app.middleware("http")
async def operational_headers(request: Request, call_next):
    request_id = request.headers.get("x-request-id", str(uuid.uuid4()))[:128]
    started = time.perf_counter()
    response = await call_next(request)
    response.headers["x-request-id"] = request_id
    response.headers["x-process-time-ms"] = f"{(time.perf_counter() - started) * 1000:.2f}"
    response.headers["x-content-type-options"] = "nosniff"
    response.headers["referrer-policy"] = "no-referrer"
    response.headers["content-security-policy"] = "default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'; img-src 'self' data:"
    return response


@app.get("/health/live")
def liveness() -> dict[str, str]:
    return {"status": "ok", "version": VERSION}


@app.get("/health/ready")
def readiness() -> dict[str, Any]:
    return {"status": "ready", "version": VERSION, "openscad": shutil.which("openscad") is not None}


@app.post("/v1/generate")
def generate_model(request: GenerateRequest) -> dict[str, Any]:
    try:
        program = generate(request.prompt)
        code = compile_openscad(program)
    except (ValueError, IRError) as exc:
        raise HTTPException(422, str(exc)) from exc
    return {
        "ir": program.to_dict(), "openscad": code,
        "validation": validate_program(program, render=request.validate_geometry),
    }


@app.post("/v1/compile")
def compile_model(request: ProgramRequest) -> dict[str, Any]:
    program = _program(request.ir)
    try:
        code = compile_openscad(program, request.overrides)
        report = validate_program(program, request.overrides, request.validate_geometry)
    except IRError as exc:
        raise HTTPException(422, str(exc)) from exc
    return {"openscad": code, "validation": report}


@app.post("/v1/validate/sweep")
def validate_sweep(request: ProgramRequest) -> dict[str, Any]:
    program = _program(request.ir)
    return validate_parameter_sweep(program, render=request.validate_geometry)


@app.post("/v1/export/scad")
def export_scad(request: ProgramRequest) -> Response:
    program = _program(request.ir)
    try:
        payload = compile_openscad(program, request.overrides).encode("utf-8")
    except IRError as exc:
        raise HTTPException(422, str(exc)) from exc
    return Response(payload, media_type="text/plain; charset=utf-8", headers={"content-disposition": "attachment; filename=neuroscad.scad"})


@app.post("/v1/export/stl")
def export_stl(request: ProgramRequest) -> Response:
    program = _program(request.ir)
    try:
        payload, metrics = render_stl(program, request.overrides)
    except IRError as exc:
        raise HTTPException(422, str(exc)) from exc
    except (RuntimeError, subprocess.TimeoutExpired) as exc:
        raise HTTPException(503, f"Render unavailable: {exc}") from exc
    return Response(payload, media_type="model/stl", headers={
        "content-disposition": "attachment; filename=neuroscad.stl",
        "x-neuroscad-watertight": str(metrics.get("watertight", "unknown")).lower(),
    })


_WEB = Path(__file__).resolve().parent / "web"
if _WEB.is_dir():
    app.mount("/", StaticFiles(directory=_WEB, html=True), name="web")
