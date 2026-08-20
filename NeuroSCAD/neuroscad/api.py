"""Optional FastAPI adapter. Core modules do not depend on a web framework."""
from __future__ import annotations
try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field
except ImportError as exc:  # clear error instead of mysterious import failure
    raise RuntimeError("Install API dependencies: pip install -e '.[api]'") from exc
from .compiler import compile_openscad
from .templates import generate
from .validator import validate_program

app = FastAPI(title="NeuroSCAD", version="0.1.0")
class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=3, max_length=2000)
    validate_geometry: bool = False
@app.get("/health")
def health(): return {"status": "ok", "version": "0.1.0"}
@app.post("/v1/generate")
def generate_model(request: GenerateRequest):
    try: program = generate(request.prompt)
    except ValueError as exc: raise HTTPException(422, str(exc)) from exc
    return {"ir": program.to_dict(), "openscad": compile_openscad(program), "validation": validate_program(program, render=request.validate_geometry)}
