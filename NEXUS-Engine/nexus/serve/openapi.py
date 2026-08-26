"""Спецификация OpenAPI 3.1 и страницы документации (Swagger UI / ReDoc).

Генерируется из одного описания, чтобы `/docs` не расходился с реальными
маршрутами. Без внешних зависимостей: сам JSON собирается здесь, а UI грузится
с CDN (если интернета нет — работает `/openapi.json` и мини-панель на `/`).
"""
from __future__ import annotations

from typing import Any, Dict

VECTOR3 = {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3,
           "example": [0, 0, -300]}
MATERIALS = ["pla", "petg", "abs", "alu6061", "steel304", "ti6al4v"]

SCAD_EXAMPLE = "difference(){cube([40,40,6],center=true); cylinder(h=20,r=5,center=true);}"


def _body(schema: Dict[str, Any], example: Dict[str, Any] | None = None) -> Dict[str, Any]:
    content: Dict[str, Any] = {"schema": schema}
    if example:
        content["example"] = example
    return {"required": True, "content": {"application/json": content}}


def _ok(description: str, example: Dict[str, Any] | None = None) -> Dict[str, Any]:
    content: Dict[str, Any] = {"schema": {"type": "object"}}
    if example:
        content["example"] = example
    return {"200": {"description": description,
                    "content": {"application/json": content}},
            "400": {"description": "неверные параметры или JSON"},
            "401": {"description": "нужен Authorization: Bearer (если задан NEXUS_API_KEY)"},
            "429": {"description": "превышен лимит запросов"}}


def build_spec(port: int = 8000, version: str = "0.1.0") -> Dict[str, Any]:
    generate_schema = {
        "type": "object",
        "required": ["prompt"],
        "properties": {
            "prompt": {"type": "string", "example": "<task>Кронштейн, алюминий, 350 Н"},
            "max_new_tokens": {"type": "integer", "default": 64},
            "temperature": {"type": "number", "default": 0.8},
            "top_k": {"type": "integer", "default": 40},
            "top_p": {"type": "number", "default": 1.0, "description": "nucleus-отсечка"},
            "stop_at_eos": {"type": "boolean", "default": True,
                            "description": "остановиться на токене конца текста"},
            "model": {"type": "string", "description": "имя модели в реестре"},
            "ref": {"type": "string", "description": "версия или тег (production, latest, v0003)"},
        },
    }
    analyze_schema = {
        "type": "object",
        "required": ["code"],
        "properties": {
            "code": {"type": "string", "description": "исходник OpenSCAD", "example": SCAD_EXAMPLE},
            "material": {"type": "string", "enum": MATERIALS, "default": "pla"},
            "force": dict(VECTOR3, description="вектор силы, Н"),
            "fixture": {"type": "string", "enum": ["base", "bore", "face_x"], "default": "base"},
            "grid": {"type": "integer", "default": 24, "description": "разрешение вокселизации"},
            "calculix": {"type": "boolean", "default": False},
        },
    }
    reward_schema = {
        "type": "object",
        "required": ["code"],
        "properties": {
            "code": {"type": "string", "example": SCAD_EXAMPLE},
            "material": {"type": "string", "enum": MATERIALS, "default": "pla"},
            "force": dict(VECTOR3, description="вектор силы, Н"),
            "fixture": {"type": "string", "default": "base"},
            "required_sf": {"type": "number", "default": 2.0},
            "grid": {"type": "integer", "default": 20},
        },
    }
    design_schema = {
        "type": "object",
        "required": ["spec"],
        "properties": {
            "spec": {"type": "string", "example": "<task>Кронштейн под 350 Н, алюминий"},
            "max_new_tokens": {"type": "integer", "default": 96},
            "material": {"type": "string", "enum": MATERIALS, "default": "pla"},
            "force": dict(VECTOR3, description="вектор силы, Н"),
        },
    }
    return {
        "openapi": "3.1.0",
        "info": {
            "title": "NEXUS-Engine API",
            "version": version,
            "description": (
                "Инференс модели, инженерный анализ (масса, аудит печати/ЧПУ, МКЭ), "
                "управление версиями моделей и фоновое обучение.\n\n"
                "Если задана переменная окружения `NEXUS_API_KEY`, все методы кроме "
                "`/health`, `/`, `/docs` требуют заголовок `Authorization: Bearer <key>`."),
        },
        "servers": [{"url": f"http://localhost:{port}"}],
        "tags": [
            {"name": "service", "description": "статус и метрики"},
            {"name": "inference", "description": "генерация и сквозное проектирование"},
            {"name": "engineering", "description": "геометрия, прочность, награды"},
            {"name": "registry", "description": "версии моделей, теги, откат"},
            {"name": "jobs", "description": "фоновое обучение"},
        ],
        "components": {
            "securitySchemes": {
                "bearerAuth": {"type": "http", "scheme": "bearer"},
                "apiKey": {"type": "apiKey", "in": "header", "name": "X-API-Key"},
            }
        },
        "paths": {
            "/health": {"get": {"tags": ["service"], "summary": "Статус сервиса",
                                "responses": _ok("аптайм, устройство, загруженные модели")}},
            "/metrics": {"get": {"tags": ["service"], "summary": "Метрики Prometheus",
                                 "responses": {"200": {"description": "text/plain"}}}},
            "/v1/models": {"get": {"tags": ["registry"], "summary": "Реестр моделей",
                                   "responses": _ok("версии, теги, метрики")}},
            "/v1/generate": {
                "post": {"tags": ["inference"], "summary": "Сгенерировать текст или код",
                         "requestBody": _body(generate_schema,
                                              {"prompt": "<task>Кронштейн 350 Н",
                                               "max_new_tokens": 64}),
                         "responses": _ok("сгенерированный текст и версия модели")}},
            "/v1/generate/stream": {
                "post": {"tags": ["inference"],
                         "summary": "Потоковая генерация (Server-Sent Events)",
                         "requestBody": _body(generate_schema),
                         "responses": {"200": {"description": "text/event-stream"}}}},
            "/v1/generate/batch": {
                "post": {"tags": ["inference"], "summary": "Батч-генерация",
                         "requestBody": _body({
                             "type": "object", "required": ["prompts"],
                             "properties": {
                                 "prompts": {"type": "array", "items": {"type": "string"},
                                             "example": ["<task>плита", "<task>фланец"]},
                                 "max_new_tokens": {"type": "integer", "default": 64}}}),
                         "responses": _ok("список ответов и время на промпт")}},
            "/v1/design": {
                "post": {"tags": ["inference"],
                         "summary": "ТЗ → код → геометрия → МКЭ → награда",
                         "requestBody": _body(design_schema),
                         "responses": _ok("генерация, анализ и оценка")}},
            "/v1/analyze": {
                "post": {"tags": ["engineering"],
                         "summary": "Анализ детали: масса, аудит, прочность",
                         "requestBody": _body(analyze_schema,
                                              {"code": SCAD_EXAMPLE, "material": "alu6061",
                                               "force": [0, 0, -400]}),
                         "responses": _ok("масс-инерционные свойства, аудит, поля МКЭ")}},
            "/v1/reward": {
                "post": {"tags": ["engineering"], "summary": "Физическая награда (как в RL)",
                         "requestBody": _body(reward_schema, {"code": SCAD_EXAMPLE}),
                         "responses": _ok("составляющие награды и итог")}},
            "/v1/reload": {
                "post": {"tags": ["registry"], "summary": "Горячая перезагрузка модели",
                         "requestBody": _body({"type": "object", "properties": {
                             "model": {"type": "string", "default": "core"},
                             "ref": {"type": "string", "default": "production"}}}),
                         "responses": _ok("какая версия загружена")}},
            "/v1/registry/promote": {
                "post": {"tags": ["registry"], "summary": "Назначить тег версии",
                         "requestBody": _body({"type": "object", "properties": {
                             "model": {"type": "string", "default": "core"},
                             "ref": {"type": "string", "default": "latest"},
                             "tag": {"type": "string", "default": "production"}}}),
                         "responses": _ok("метаданные версии")}},
            "/v1/registry/rollback": {
                "post": {"tags": ["registry"], "summary": "Откатить тег на предыдущую версию",
                         "requestBody": _body({"type": "object", "properties": {
                             "model": {"type": "string", "default": "core"},
                             "tag": {"type": "string", "default": "production"},
                             "steps": {"type": "integer", "default": 1}}}),
                         "responses": _ok("версия после отката")}},
            "/v1/eval": {
                "post": {"tags": ["registry"], "summary": "Приёмочные тесты версии",
                         "requestBody": _body({"type": "object", "properties": {
                             "model": {"type": "string", "default": "core"},
                             "ref": {"type": "string", "default": "latest"},
                             "baseline": {"type": "string"},
                             "gate": {"type": "boolean", "default": False}}}),
                         "responses": _ok("отчёт с проверками")}},
            "/v1/jobs": {
                "get": {"tags": ["jobs"], "summary": "Список фоновых задач",
                        "responses": _ok("задачи и их статусы")},
                "post": {"tags": ["jobs"], "summary": "Запустить обучение фоном",
                         "requestBody": _body({"type": "object", "required": ["args"],
                                               "properties": {"args": {
                                                   "type": "array",
                                                   "items": {"type": "string"},
                                                   "example": ["train-lm", "--max-steps", "100"]}}}),
                         "responses": _ok("идентификатор задачи")}},
            "/v1/jobs/{job_id}": {
                "get": {"tags": ["jobs"], "summary": "Статус задачи и хвост лога",
                        "parameters": [{"name": "job_id", "in": "path", "required": True,
                                        "schema": {"type": "string"}}],
                        "responses": _ok("статус, код возврата, лог")}},
            "/v1/jobs/{job_id}/cancel": {
                "post": {"tags": ["jobs"], "summary": "Остановить задачу",
                         "parameters": [{"name": "job_id", "in": "path", "required": True,
                                         "schema": {"type": "string"}}],
                         "responses": _ok("состояние задачи")}},
        },
    }


SWAGGER_HTML = """<!doctype html><html lang="ru"><head><meta charset="utf-8">
<title>NEXUS-Engine API — документация</title>
<link rel="stylesheet" href="https://unpkg.com/swagger-ui-dist@5/swagger-ui.css">
<style>body{margin:0}#fallback{font:14px system-ui;padding:24px;color:#444}</style>
</head><body>
<div id="swagger"></div>
<div id="fallback">Swagger UI грузится с CDN. Если интернета нет — спецификация
доступна напрямую: <a href="/openapi.json">/openapi.json</a>, а короткий список
методов — на <a href="/">главной</a>.</div>
<script src="https://unpkg.com/swagger-ui-dist@5/swagger-ui-bundle.js" crossorigin></script>
<script>
window.addEventListener('load', function () {
  if (window.SwaggerUIBundle) {
    document.getElementById('fallback').style.display = 'none';
    SwaggerUIBundle({url: '/openapi.json', dom_id: '#swagger', tryItOutEnabled: true,
                     defaultModelsExpandDepth: -1});
  }
});
</script></body></html>"""

REDOC_HTML = """<!doctype html><html lang="ru"><head><meta charset="utf-8">
<title>NEXUS-Engine API — ReDoc</title><style>body{margin:0}</style></head><body>
<redoc spec-url="/openapi.json"></redoc>
<script src="https://cdn.redoc.ly/redoc/latest/bundles/redoc.standalone.js"></script>
</body></html>"""
