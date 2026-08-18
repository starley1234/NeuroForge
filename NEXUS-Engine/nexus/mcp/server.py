"""MCP-сервер NEXUS: движок геометрии и прочности как инструменты для LLM.

Зачем: сильная внешняя модель (Claude, GPT, Qwen) умеет писать OpenSCAD, но не
умеет проверить результат. Через MCP она получает наши инструменты — компиляцию
CSG, аудит печати/ЧПУ, настоящий МКЭ и физическую награду — и может итеративно
исправлять деталь. Каждый такой диалог, прошедший проверку, становится
обучающим примером: это и есть дистилляция «с верификацией», без ручной
разметки (см. `nexus.data.collect`).

Транспорт — stdio, протокол JSON-RPC 2.0, без внешних зависимостей.
Совместим с Claude Desktop / Kiro / OpenAI Agents SDK:

    {"mcpServers": {"nexus": {"command": "python", "args": ["-m", "nexus.mcp.server"]}}}
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

PROTOCOL_VERSION = "2024-11-05"
SERVER_INFO = {"name": "nexus-engine", "version": "0.1.0"}


# ─────────────────────────────────────────────────────────── инструменты
@dataclass
class Tool:
    name: str
    description: str
    schema: Dict[str, Any]
    handler: Callable[[Dict[str, Any]], Dict[str, Any]]

    def spec(self) -> Dict[str, Any]:
        return {"name": self.name, "description": self.description,
                "inputSchema": self.schema}


def _num_array(desc: str, default: List[float]) -> Dict[str, Any]:
    return {"type": "array", "items": {"type": "number"}, "minItems": 3,
            "maxItems": 3, "description": desc, "default": default}


class NexusTools:
    """Прикладная логика инструментов (используется и MCP, и сборщиком данных)."""

    def __init__(self, workdir: str = "artifacts/mcp"):
        self.workdir = workdir
        os.makedirs(workdir, exist_ok=True)
        self.samples_path = os.path.join(workdir, "samples.jsonl")

    # -- геометрия ---------------------------------------------------------
    def scad_compile(self, code: str) -> Dict[str, Any]:
        from ..scad.render import compile_scad
        res = compile_scad(code)
        return {"ok": res.ok, "error": res.error}

    def scad_analyze(self, code: str, material: str = "pla", grid: int = 24,
                     stl_path: Optional[str] = None) -> Dict[str, Any]:
        from ..scad.render import render
        res = render(code, resolution=grid, material=material, stl_path=stl_path)
        if not res.ok:
            return {"ok": False, "error": res.error}
        return {"ok": True, "mass": res.mass.to_dict(), "audit": res.audit_report.to_dict(),
                "graph_nodes": res.graph.n_nodes, "stl": stl_path}

    def fem_analyze(self, code: str, force_n: Optional[List[float]] = None,
                    fixture: str = "base", material: str = "pla", grid: int = 24,
                    backend: str = "auto") -> Dict[str, Any]:
        from ..fem.solver import solve
        from ..scad.render import render
        res = render(code, resolution=grid, material=material)
        if not res.ok:
            return {"ok": False, "error": res.error}
        fem = solve(res.voxels, tuple(force_n or [0.0, 0.0, -200.0]), fixture,
                    material, backend=backend)
        return {"ok": True, "mass_g": round(res.mass.mass_g, 3),
                "audit": res.audit_report.to_dict(), "fem": fem.to_dict()}

    def score_design(self, code: str, force_n: Optional[List[float]] = None,
                     fixture: str = "base", material: str = "pla",
                     required_sf: float = 2.0, mass_budget_g: float = 200.0,
                     grid: int = 20) -> Dict[str, Any]:
        from ..training.rewards import score_scad
        rb = score_scad(code, tuple(force_n or [0.0, 0.0, -200.0]), fixture, material,
                        required_sf=required_sf, mass_budget_g=mass_budget_g,
                        resolution=grid)
        return rb.to_dict()

    def propose_variation(self, template: str = "l_bracket", seed: int = 0) -> Dict[str, Any]:
        import random
        from ..scad.generator import TEMPLATES, sample
        if template not in TEMPLATES:
            return {"ok": False, "error": f"нет шаблона {template!r}",
                    "available": list(TEMPLATES)}
        s = sample(template, random.Random(seed))
        return {"ok": True, "template": s.template, "params": s.params,
                "code": s.code, "spec": s.spec, "material": s.material,
                "load": {"force_n": list(s.load.force_n), "fixture": s.load.fixture}}

    def math_problem(self, kind: Optional[str] = None, seed: int = 0) -> Dict[str, Any]:
        from ..data.mathgen import GENERATORS, generate
        if kind and kind not in GENERATORS:
            return {"ok": False, "error": f"нет типа {kind!r}", "available": list(GENERATORS)}
        s = generate(1, seed=seed, kinds=[kind] if kind else None)[0]
        return {"ok": True, **s.to_dict()}

    def record_sample(self, spec: str, code: str, tags: Optional[List[str]] = None,
                      verify: bool = True, force_n: Optional[List[float]] = None,
                      material: str = "pla") -> Dict[str, Any]:
        """Записать проверенный пример в обучающий корпус."""
        record: Dict[str, Any] = {"spec": spec, "code": code, "tags": tags or [],
                                  "material": material}
        if verify:
            check = self.score_design(code, force_n, material=material)
            record["reward"] = check
            if check.get("total", -1) <= 0:
                return {"ok": False, "error": "пример не прошёл проверку", "reward": check}
        with open(self.samples_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        return {"ok": True, "path": self.samples_path, "reward": record.get("reward")}

    def engine_info(self) -> Dict[str, Any]:
        from ..config import NexusConfig
        from ..eval.vram import estimate
        from ..fem.calculix import available as ccx
        from ..scad.render import openscad_binary
        return {"openscad": bool(openscad_binary()), "calculix": ccx(),
                "fem_backends": ["hex", "loadpath"] + (["calculix"] if ccx() else []),
                "materials": list(__import__("nexus.geometry.voxel", fromlist=["MATERIALS"]).MATERIALS),
                "vram_estimate_rtx5060_gb": estimate(NexusConfig.rtx5060()).total_gb}


def build_tools(tools: NexusTools) -> List[Tool]:
    code_prop = {"type": "string", "description": "исходник OpenSCAD"}
    material_prop = {"type": "string", "default": "pla",
                     "description": "pla | petg | abs | alu6061 | steel304 | ti6al4v"}
    return [
        Tool("scad_compile", "Проверить, что код OpenSCAD собирается в валидное CSG-дерево "
                             "(быстро, без геометрии).",
             {"type": "object", "properties": {"code": code_prop}, "required": ["code"]},
             lambda a: tools.scad_compile(a["code"])),
        Tool("scad_analyze", "Геометрия детали: объём, масса, центр масс, тензор инерции, "
                             "manifold, толщина стенки, свесы для печати, достижимость фрезой.",
             {"type": "object", "properties": {
                 "code": code_prop, "material": material_prop,
                 "grid": {"type": "integer", "default": 24, "description": "разрешение вокселизации"},
                 "stl_path": {"type": "string", "description": "куда сохранить STL (опционально)"}},
              "required": ["code"]},
             lambda a: tools.scad_analyze(a["code"], a.get("material", "pla"),
                                          int(a.get("grid", 24)), a.get("stl_path"))),
        Tool("fem_analyze", "Прочностной расчёт: настоящий МКЭ на гексаэдрах — поле напряжений "
                            "фон Мизеса, максимальное перемещение, коэффициент запаса.",
             {"type": "object", "properties": {
                 "code": code_prop, "material": material_prop,
                 "force_n": _num_array("вектор силы в ньютонах", [0, 0, -200]),
                 "fixture": {"type": "string", "enum": ["base", "bore", "face_x"],
                             "default": "base"},
                 "grid": {"type": "integer", "default": 24},
                 "backend": {"type": "string", "enum": ["auto", "hex", "loadpath", "calculix"],
                             "default": "auto"}},
              "required": ["code"]},
             lambda a: tools.fem_analyze(a["code"], a.get("force_n"), a.get("fixture", "base"),
                                         a.get("material", "pla"), int(a.get("grid", 24)),
                                         a.get("backend", "auto"))),
        Tool("score_design", "Сводная физическая оценка детали: +1 компиляция, +1.5 manifold и "
                             "технологичность, +2 запас прочности, штрафы за массу и тонкие стенки.",
             {"type": "object", "properties": {
                 "code": code_prop, "material": material_prop,
                 "force_n": _num_array("вектор силы в ньютонах", [0, 0, -200]),
                 "fixture": {"type": "string", "default": "base"},
                 "required_sf": {"type": "number", "default": 2.0},
                 "mass_budget_g": {"type": "number", "default": 200.0}},
              "required": ["code"]},
             lambda a: tools.score_design(a["code"], a.get("force_n"), a.get("fixture", "base"),
                                          a.get("material", "pla"),
                                          float(a.get("required_sf", 2.0)),
                                          float(a.get("mass_budget_g", 200.0)))),
        Tool("propose_variation", "Выдать параметрический шаблон детали с ТЗ и нагрузкой "
                                  "(l_bracket, flange, plate, standoff, bearing_block).",
             {"type": "object", "properties": {
                 "template": {"type": "string", "default": "l_bracket"},
                 "seed": {"type": "integer", "default": 0}}},
             lambda a: tools.propose_variation(a.get("template", "l_bracket"),
                                               int(a.get("seed", 0)))),
        Tool("math_problem", "Сгенерировать инженерную задачу с проверенным ответом "
                             "(изгиб, затяжка, расширение, размерные цепи, устойчивость).",
             {"type": "object", "properties": {
                 "kind": {"type": "string", "description": "тип задачи, необязательно"},
                 "seed": {"type": "integer", "default": 0}}},
             lambda a: tools.math_problem(a.get("kind"), int(a.get("seed", 0)))),
        Tool("record_sample", "Сохранить проверенный пример «ТЗ → код» в обучающий корпус. "
                              "Пример записывается только если проходит физическую проверку.",
             {"type": "object", "properties": {
                 "spec": {"type": "string", "description": "техническое задание"},
                 "code": code_prop, "material": material_prop,
                 "force_n": _num_array("вектор силы в ньютонах", [0, 0, -200]),
                 "tags": {"type": "array", "items": {"type": "string"}}},
              "required": ["spec", "code"]},
             lambda a: tools.record_sample(a["spec"], a["code"], a.get("tags"),
                                           True, a.get("force_n"),
                                           a.get("material", "pla"))),
        Tool("engine_info", "Что доступно в движке: бэкенды FEM, материалы, оценка VRAM.",
             {"type": "object", "properties": {}}, lambda a: tools.engine_info()),
    ]


# ─────────────────────────────────────────────────────────── JSON-RPC слой
class MCPServer:
    def __init__(self, workdir: str = "artifacts/mcp"):
        self.tools_impl = NexusTools(workdir)
        self.tools = {t.name: t for t in build_tools(self.tools_impl)}
        self.initialized = False

    # -- обработка сообщений ----------------------------------------------
    def handle(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        method = message.get("method")
        msg_id = message.get("id")
        params = message.get("params") or {}

        if method == "initialize":
            self.initialized = True
            client_version = params.get("protocolVersion", PROTOCOL_VERSION)
            return self._ok(msg_id, {
                "protocolVersion": client_version if client_version else PROTOCOL_VERSION,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": SERVER_INFO,
                "instructions": (
                    "Инструменты движка NEXUS: проектируй деталь на OpenSCAD, "
                    "проверяй её scad_analyze и fem_analyze, повышай score_design "
                    "и сохраняй удачные варианты через record_sample."),
            })
        if method in ("notifications/initialized", "initialized"):
            return None
        if method == "ping":
            return self._ok(msg_id, {})
        if method == "tools/list":
            return self._ok(msg_id, {"tools": [t.spec() for t in self.tools.values()]})
        if method == "tools/call":
            return self._call_tool(msg_id, params)
        if method in ("resources/list", "prompts/list"):
            key = method.split("/")[0]
            return self._ok(msg_id, {key: []})
        if msg_id is None:
            return None
        return self._error(msg_id, -32601, f"неизвестный метод: {method}")

    def _call_tool(self, msg_id: Any, params: Dict[str, Any]) -> Dict[str, Any]:
        name = params.get("name")
        arguments = params.get("arguments") or {}
        tool = self.tools.get(name)
        if tool is None:
            return self._error(msg_id, -32602, f"нет инструмента {name!r}")
        try:
            result = tool.handler(arguments)
            is_error = bool(isinstance(result, dict) and result.get("ok") is False)
        except Exception as exc:  # инструмент не должен ронять сервер
            result = {"ok": False, "error": f"{type(exc).__name__}: {exc}",
                      "trace": traceback.format_exc(limit=3)}
            is_error = True
        text = json.dumps(result, ensure_ascii=False, indent=2, default=str)
        return self._ok(msg_id, {"content": [{"type": "text", "text": text}],
                                 "isError": is_error})

    @staticmethod
    def _ok(msg_id: Any, result: Dict[str, Any]) -> Dict[str, Any]:
        return {"jsonrpc": "2.0", "id": msg_id, "result": result}

    @staticmethod
    def _error(msg_id: Any, code: int, message: str) -> Dict[str, Any]:
        return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": message}}

    # -- цикл stdio --------------------------------------------------------
    def serve_stdio(self, stdin=None, stdout=None) -> None:
        stdin = stdin or sys.stdin
        stdout = stdout or sys.stdout
        for line in stdin:
            line = line.strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError as exc:
                response = self._error(None, -32700, f"некорректный JSON: {exc}")
            else:
                response = self.handle(message)
            if response is not None:
                stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
                stdout.flush()


def main() -> None:
    ap = argparse.ArgumentParser(description="MCP-сервер NEXUS-Engine (stdio)")
    ap.add_argument("--workdir", default="artifacts/mcp")
    ap.add_argument("--list-tools", action="store_true", help="напечатать инструменты и выйти")
    a = ap.parse_args()
    server = MCPServer(a.workdir)
    if a.list_tools:
        print(json.dumps([t.spec() for t in server.tools.values()], indent=2,
                         ensure_ascii=False))
        return
    server.serve_stdio()


if __name__ == "__main__":
    main()
