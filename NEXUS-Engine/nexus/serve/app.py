"""HTTP API NEXUS-Engine — только стандартная библиотека, без зависимостей.

Запуск: ``nexus serve --host 0.0.0.0 --port 8000``

===================================  =========================================
`GET  /`                             мини-панель (HTML) со списком эндпойнтов
`GET  /health`                       статус, загруженные модели, аптайм
`GET  /v1/models`                    реестр: версии, теги, метрики
`POST /v1/generate`                  генерация текста/кода
`POST /v1/design`                    ТЗ → SCAD → геометрия → FEM → награда
`POST /v1/analyze`                   анализ готового SCAD (масса, аудит, FEM)
`POST /v1/reward`                    физическая награда (как в RL)
`POST /v1/reload`                    перечитать модель из реестра
`POST /v1/registry/promote`          назначить тег (production/staging)
`POST /v1/registry/rollback`         откатить тег на предыдущую версию
`POST /v1/eval`                      приёмочные тесты версии (+ gate)
`POST /v1/jobs`                      запустить обучение фоновой задачей
`GET  /v1/jobs`, `/v1/jobs/{id}`     список задач и лог конкретной задачи
`POST /v1/jobs/{id}/cancel`          остановить задачу
===================================  =========================================
"""
from __future__ import annotations

import argparse
import json
import re
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict, Optional, Tuple

from ..registry import ModelRegistry
from .jobs import JobManager
from .service import InferenceService

INDEX_HTML = """<!doctype html><html lang="ru"><meta charset="utf-8">
<title>NEXUS-Engine API</title>
<style>
 body{font:15px/1.55 system-ui,sans-serif;margin:0;background:#0d1117;color:#e6edf3}
 main{max-width:900px;margin:0 auto;padding:32px 20px}
 h1{font-size:22px;margin:0 0 4px} .sub{color:#8b949e;margin-bottom:24px}
 table{border-collapse:collapse;width:100%;margin-bottom:24px}
 td,th{border-bottom:1px solid #21262d;padding:7px 8px;text-align:left;vertical-align:top}
 code{background:#161b22;padding:2px 6px;border-radius:5px;color:#79c0ff}
 .m{color:#7ee787;font-weight:600} pre{background:#161b22;padding:12px;border-radius:8px;overflow:auto}
 .ok{color:#7ee787}.warn{color:#d29922}
</style><main>
<h1>NEXUS-Engine API</h1>
<div class="sub">UniPhysical-Latent Framework — инференс, инженерный анализ, обучение</div>
<div id="status"></div>
<table><tr><th>Метод</th><th>Путь</th><th>Назначение</th></tr>
<tr><td class="m">GET</td><td><code>/health</code></td><td>статус сервиса</td></tr>
<tr><td class="m">GET</td><td><code>/v1/models</code></td><td>реестр моделей: версии, теги, метрики</td></tr>
<tr><td class="m">POST</td><td><code>/v1/generate</code></td><td>{"prompt","max_new_tokens","temperature"}</td></tr>
<tr><td class="m">POST</td><td><code>/v1/design</code></td><td>{"spec","material","force"} → код + FEM + награда</td></tr>
<tr><td class="m">POST</td><td><code>/v1/analyze</code></td><td>{"code","material","force","fixture"}</td></tr>
<tr><td class="m">POST</td><td><code>/v1/reward</code></td><td>{"code","force","required_sf"}</td></tr>
<tr><td class="m">POST</td><td><code>/v1/reload</code></td><td>{"model","ref"} — горячая перезагрузка</td></tr>
<tr><td class="m">POST</td><td><code>/v1/registry/promote</code></td><td>{"model","ref","tag"}</td></tr>
<tr><td class="m">POST</td><td><code>/v1/registry/rollback</code></td><td>{"model","tag"}</td></tr>
<tr><td class="m">POST</td><td><code>/v1/eval</code></td><td>{"model","ref","baseline","gate"}</td></tr>
<tr><td class="m">POST</td><td><code>/v1/jobs</code></td><td>{"args":["train-lm","--max-steps","20"]}</td></tr>
</table>
<pre>curl -s localhost:PORT/v1/analyze -d '{"code":"cube([20,20,4],center=true);","force":[0,0,-300]}'</pre>
<script>
fetch('/health').then(r=>r.json()).then(h=>{
  document.getElementById('status').innerHTML =
    `<p class="ok">● ${h.status} · устройство ${h.device} · аптайм ${h.uptime_s} с · запросов ${h.requests}</p>`;
}).catch(()=>{});
</script></main></html>"""


class NexusAPI:
    """Маршрутизация и бизнес-логика (без привязки к HTTP-серверу)."""

    def __init__(self, service: InferenceService, jobs: Optional[JobManager] = None):
        self.service = service
        self.jobs = jobs or JobManager()
        self.registry: ModelRegistry = service.registry
        self.routes: Dict[Tuple[str, str], Callable[[Dict[str, Any], Dict[str, str]], Any]] = {
            ("GET", "/health"): lambda b, p: self.service.health(),
            ("GET", "/v1/models"): lambda b, p: self.service.models(),
            ("POST", "/v1/generate"): lambda b, p: self.service.generate(**b),
            ("POST", "/v1/design"): lambda b, p: self.service.design(**b),
            ("POST", "/v1/analyze"): lambda b, p: self.service.analyze(**b),
            ("POST", "/v1/reward"): lambda b, p: self.service.reward(**b),
            ("POST", "/v1/reload"): lambda b, p: self.service.reload(b.get("model"), b.get("ref")),
            ("POST", "/v1/registry/promote"): lambda b, p: self._promote(b),
            ("POST", "/v1/registry/rollback"): lambda b, p: self._rollback(b),
            ("POST", "/v1/eval"): lambda b, p: self._eval(b),
            ("POST", "/v1/jobs"): lambda b, p: self.jobs.submit(b.get("args", [])).to_dict(),
            ("GET", "/v1/jobs"): lambda b, p: {"jobs": self.jobs.list()},
        }
        self.patterns = [
            ("GET", re.compile(r"^/v1/jobs/(?P<job_id>[a-f0-9]+)$"), self._job_status),
            ("POST", re.compile(r"^/v1/jobs/(?P<job_id>[a-f0-9]+)/cancel$"), self._job_cancel),
        ]

    # ------------------------------------------------------------- обработчики
    def _promote(self, body: Dict[str, Any]) -> Dict[str, Any]:
        mv = self.registry.promote(body.get("model", self.service.default_model),
                                   body.get("ref", "latest"), body.get("tag", "production"),
                                   reason=body.get("reason", "api"))
        self.service.reload(mv.name, body.get("tag", "production"))
        return mv.to_dict()

    def _rollback(self, body: Dict[str, Any]) -> Dict[str, Any]:
        mv = self.registry.rollback(body.get("model", self.service.default_model),
                                    body.get("tag", "production"), int(body.get("steps", 1)))
        self.service.reload(mv.name, body.get("tag", "production"))
        return mv.to_dict()

    def _eval(self, body: Dict[str, Any]) -> Dict[str, Any]:
        from ..eval.suite import evaluate_version
        report = evaluate_version(
            name=body.get("model", self.service.default_model),
            ref=body.get("ref", "latest"),
            baseline_ref=body.get("baseline"),
            registry_root=self.registry.root,
            source=body.get("source", "builtin:engineering"),
            gate=bool(body.get("gate", False)),
            device=self.service.device,
        )
        return report.to_dict()

    def _job_status(self, body: Dict[str, Any], params: Dict[str, str]) -> Dict[str, Any]:
        job = self.jobs.get(params["job_id"])
        if job is None:
            raise KeyError("задача не найдена")
        return {**job.to_dict(), "log": self.jobs.logs(params["job_id"], tail=int(body.get("tail", 80)))}

    def _job_cancel(self, body: Dict[str, Any], params: Dict[str, str]) -> Dict[str, Any]:
        job = self.jobs.cancel(params["job_id"])
        if job is None:
            raise KeyError("задача не найдена")
        return job.to_dict()

    # ------------------------------------------------------------- диспетчер
    def dispatch(self, method: str, path: str, body: Dict[str, Any]) -> Tuple[int, Any]:
        handler = self.routes.get((method, path))
        if handler is not None:
            return 200, handler(body, {})
        for m, pattern, fn in self.patterns:
            match = pattern.match(path)
            if match and m == method:
                return 200, fn(body, match.groupdict())
        return 404, {"error": f"нет маршрута {method} {path}"}


def make_handler(api: NexusAPI):
    class Handler(BaseHTTPRequestHandler):
        server_version = "NEXUS/0.1"
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):  # компактный лог
            print(f"[api] {self.address_string()} {fmt % args}", flush=True)

        def _send(self, code: int, payload: Any, content_type="application/json") -> None:
            if content_type == "application/json":
                data = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
            else:
                data = payload.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", f"{content_type}; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "*")
            self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
            self.end_headers()
            self.wfile.write(data)

        def do_OPTIONS(self):  # noqa: N802
            self._send(204, {})

        def do_GET(self):  # noqa: N802
            path = self.path.split("?")[0]
            if path in ("/", "/index.html"):
                port = self.server.server_address[1]
                return self._send(200, INDEX_HTML.replace("PORT", str(port)), "text/html")
            self._handle("GET", path, {})

        def do_POST(self):  # noqa: N802
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length).decode("utf-8") if length else ""
            try:
                body = json.loads(raw) if raw else {}
            except json.JSONDecodeError as exc:
                return self._send(400, {"error": f"некорректный JSON: {exc}"})
            self._handle("POST", self.path.split("?")[0], body)

        def _handle(self, method: str, path: str, body: Dict[str, Any]) -> None:
            try:
                code, payload = api.dispatch(method, path, body)
            except TypeError as exc:
                code, payload = 400, {"error": f"неверные параметры: {exc}"}
            except KeyError as exc:
                code, payload = 404, {"error": str(exc)}
            except Exception as exc:  # pragma: no cover
                traceback.print_exc()
                code, payload = 500, {"error": f"{type(exc).__name__}: {exc}"}
            self._send(code, payload)

    return Handler


def create_server(host: str = "0.0.0.0", port: int = 8000,
                  registry_root: str = "artifacts/registry",
                  model: str = "core", ref: str = "production",
                  device: str = "cpu", preset: str = "tiny"):
    service = InferenceService(registry_root, model, ref, device, preset)
    api = NexusAPI(service)
    server = ThreadingHTTPServer((host, port), make_handler(api))
    server.daemon_threads = True
    return server, api


def serve(host: str = "0.0.0.0", port: int = 8000, **kwargs) -> None:
    server, _ = create_server(host, port, **kwargs)
    print(f"[api] NEXUS-Engine слушает http://{host}:{port} (Ctrl+C — стоп)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[api] остановка")
    finally:
        server.server_close()


def main() -> None:
    ap = argparse.ArgumentParser(description="HTTP API NEXUS-Engine")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--registry", default="artifacts/registry")
    ap.add_argument("--model", default="core")
    ap.add_argument("--ref", default="production")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--preset", choices=["tiny", "rtx5060", "rtx5060-compact"], default="tiny")
    a = ap.parse_args()
    serve(a.host, a.port, registry_root=a.registry, model=a.model, ref=a.ref,
          device=a.device, preset=a.preset)


if __name__ == "__main__":
    main()
