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
`POST /v1/generate/stream`           потоковая генерация (SSE)
`POST /v1/generate/batch`            батч-генерация по списку промптов
`GET  /metrics`                      метрики в формате Prometheus
`GET  /docs`, `/redoc`               интерактивная документация (OpenAPI 3.1)
`GET  /openapi.json`                 машинная спецификация API
===================================  =========================================

Защита: если задана переменная окружения ``NEXUS_API_KEY``, все запросы, кроме
``/health`` и ``/``, требуют заголовок ``Authorization: Bearer <key>``
(или ``X-API-Key``). Плюс простой rate-limit по IP (по умолчанию 120 запросов
в минуту, меняется флагом ``--rate-limit``).
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
from .openapi import REDOC_HTML, SWAGGER_HTML, build_spec
from .service import InferenceService

API_KEY_ENV = "NEXUS_API_KEY"

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
<tr><td class="m">GET</td><td><code><a href="/docs">/docs</a></code></td><td>Swagger UI: можно дёргать методы прямо из браузера</td></tr>
<tr><td class="m">GET</td><td><code><a href="/openapi.json">/openapi.json</a></code></td><td>спецификация OpenAPI 3.1</td></tr>
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
            ("POST", "/v1/generate/batch"): lambda b, p: self.service.generate_batch(**b),
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


class RateLimiter:
    """Скользящее окно на 60 с по IP-адресу."""

    def __init__(self, limit_per_minute: int = 120):
        self.limit = limit_per_minute
        self.hits: Dict[str, list] = {}
        self._lock = threading.Lock()

    def allow(self, client: str) -> bool:
        if self.limit <= 0:
            return True
        import time
        now = time.time()
        with self._lock:
            bucket = [t for t in self.hits.get(client, []) if now - t < 60.0]
            if len(bucket) >= self.limit:
                self.hits[client] = bucket
                return False
            bucket.append(now)
            self.hits[client] = bucket
            return True


def make_handler(api: NexusAPI, api_key: Optional[str] = None,
                 limiter: Optional[RateLimiter] = None):
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

        # ---------------------------------------------------------- защита
        def _authorized(self, path: str) -> bool:
            if not api_key or path in ("/", "/index.html", "/health", "/docs", "/docs/",
                                       "/redoc", "/redoc/", "/openapi.json", "/favicon.ico"):
                return True
            header = self.headers.get("Authorization", "")
            token = header[7:] if header.startswith("Bearer ") else self.headers.get("X-API-Key", "")
            return token == api_key

        def _rate_ok(self) -> bool:
            return limiter.allow(self.client_address[0]) if limiter else True

        def do_GET(self):  # noqa: N802
            path = self.path.split("?")[0]
            if path == "/favicon.ico":
                return self._send(204, {})
            if path in ("/", "/index.html"):
                port = self.server.server_address[1]
                return self._send(200, INDEX_HTML.replace("PORT", str(port)), "text/html")
            if not self._authorized(path):
                return self._send(401, {"error": "нужен Authorization: Bearer <NEXUS_API_KEY>"})
            if not self._rate_ok():
                return self._send(429, {"error": "превышен лимит запросов"})
            if path == "/metrics":
                return self._send(200, api.service.metrics(), "text/plain")
            if path in ("/docs", "/docs/"):
                return self._send(200, SWAGGER_HTML, "text/html")
            if path in ("/redoc", "/redoc/"):
                return self._send(200, REDOC_HTML, "text/html")
            if path == "/openapi.json":
                port = self.server.server_address[1]
                return self._send(200, build_spec(port))
            self._handle("GET", path, {})

        def do_POST(self):  # noqa: N802
            path = self.path.split("?")[0]
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length).decode("utf-8") if length else ""
            if not self._authorized(path):
                return self._send(401, {"error": "нужен Authorization: Bearer <NEXUS_API_KEY>"})
            if not self._rate_ok():
                return self._send(429, {"error": "превышен лимит запросов"})
            try:
                body = json.loads(raw) if raw else {}
            except json.JSONDecodeError as exc:
                return self._send(400, {"error": f"некорректный JSON: {exc}"})
            if path == "/v1/generate/stream":
                return self._stream(body)
            self._handle("POST", path, body)

        def _stream(self, body: Dict[str, Any]) -> None:
            """Server-Sent Events: токены отдаются по мере генерации."""
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            try:
                for chunk in api.service.generate_stream(**body):
                    payload = json.dumps(chunk, ensure_ascii=False)
                    self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                return
            except Exception as exc:  # pragma: no cover
                err = json.dumps({"error": str(exc)}, ensure_ascii=False)
                self.wfile.write(f"data: {err}\n\n".encode("utf-8"))
            self.close_connection = True

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
                  device: str = "cpu", preset: str = "tiny",
                  api_key: Optional[str] = None, rate_limit: int = 120,
                  tokenizer: Optional[str] = None, compile_model: bool = False):
    import os
    service = InferenceService(registry_root, model, ref, device, preset,
                               tokenizer_path=tokenizer, compile_model=compile_model)
    api = NexusAPI(service)
    key = api_key or os.environ.get(API_KEY_ENV) or None
    handler = make_handler(api, key, RateLimiter(rate_limit))
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    return server, api


def serve(host: str = "0.0.0.0", port: int = 8000, **kwargs) -> None:
    import os
    server, _ = create_server(host, port, **kwargs)
    protected = bool(kwargs.get("api_key") or os.environ.get(API_KEY_ENV))
    shown = "localhost" if host in ("0.0.0.0", "::") else host
    print(f"[api] NEXUS-Engine слушает http://{host}:{port} "
          f"({'с ключом' if protected else 'без авторизации'}; Ctrl+C — стоп)", flush=True)
    print(f"[api] документация: http://{shown}:{port}/docs   "
          f"спецификация: http://{shown}:{port}/openapi.json", flush=True)
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
    ap.add_argument("--preset", choices=["tiny", "small", "rtx5060", "rtx5060-compact"], default="tiny")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--rate-limit", type=int, default=120)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--compile", action="store_true")
    a = ap.parse_args()
    serve(a.host, a.port, registry_root=a.registry, model=a.model, ref=a.ref,
          device=a.device, preset=a.preset, api_key=a.api_key,
          rate_limit=a.rate_limit, tokenizer=a.tokenizer, compile_model=a.compile)


if __name__ == "__main__":
    main()
