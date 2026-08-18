# HTTP API

```bash
nexus serve --host 0.0.0.0 --port 8000 --model-name core --ref production
```

Только стандартная библиотека Python, без FastAPI/uvicorn. Открытый `GET /`
отдаёт мини-панель со списком эндпойнтов и живым статусом.

Интерактивная документация: **http://localhost:8000/docs** (Swagger UI, можно
дёргать методы прямо из браузера), спецификация — `/openapi.json`, ReDoc — `/redoc`.
Эти три адреса доступны без API-ключа.

## Эндпойнты

| Метод | Путь | Тело / результат |
| :-- | :-- | :-- |
| GET | `/health` | статус, устройство, аптайм, загруженные модели |
| GET | `/v1/models` | реестр: версии, теги, метрики |
| POST | `/v1/generate` | `{prompt, max_new_tokens, temperature, top_k, model, ref}` |
| POST | `/v1/design` | `{spec, material, force}` → код + геометрия + FEM + награда |
| POST | `/v1/analyze` | `{code, material, force, fixture, grid, calculix}` |
| POST | `/v1/reward` | `{code, force, fixture, material, required_sf}` |
| POST | `/v1/reload` | `{model, ref}` — горячая перезагрузка весов |
| POST | `/v1/registry/promote` | `{model, ref, tag}` |
| POST | `/v1/registry/rollback` | `{model, tag, steps}` |
| POST | `/v1/eval` | `{model, ref, baseline, gate}` — приёмочные тесты |
| POST | `/v1/generate/stream` | SSE: токены по мере генерации |
| POST | `/v1/generate/batch` | `{prompts:[...]}` — один прогон на несколько запросов |
| GET | `/metrics` | метрики в формате Prometheus |
| GET | `/docs`, `/redoc`, `/openapi.json` | документация и спецификация OpenAPI 3.1 |
| POST | `/v1/jobs` | `{args:["train-lm","--max-steps","100"]}` → фоновая задача |
| GET | `/v1/jobs`, `/v1/jobs/{id}` | список задач, статус и хвост лога |
| POST | `/v1/jobs/{id}/cancel` | остановить задачу |

## Примеры

```bash
curl -s localhost:8000/health

curl -s -X POST localhost:8000/v1/generate \
  -d '{"prompt":"<task>Кронштейн, алюминий, 350 Н","max_new_tokens":128}'

curl -s -X POST localhost:8000/v1/analyze \
  -d '{"code":"difference(){cube([40,40,6],center=true); cylinder(h=20,r=5,center=true);}",
       "material":"alu6061","force":[0,0,-400]}'

curl -s -X POST localhost:8000/v1/jobs \
  -d '{"args":["train-lm","--source","dir:./corpus","--epochs","1"]}'
curl -s localhost:8000/v1/jobs/<id>
```

## Защита и производительность

```bash
NEXUS_API_KEY=secret nexus serve --rate-limit 300 --compile
curl -s localhost:8000/v1/models -H "Authorization: Bearer secret"
curl -N -X POST localhost:8000/v1/generate/stream -d '{"prompt":"<task>вал","max_new_tokens":64}'
```

* `NEXUS_API_KEY` (или `--api-key`) включает проверку `Authorization: Bearer` /
  `X-API-Key`; `/health` и `/` остаются открытыми для проб живости.
* `--rate-limit N` — скользящее окно 60 с по IP (0 — выключить).
* `--compile` — `torch.compile` модели при загрузке.
* `/metrics` — счётчик запросов, аптайм, число загруженных моделей,
  квантили латентности генерации, версия каждой активной модели.

## Пределы запроса

| Ограничение | По умолчанию | Переменная |
| :-- | :-- | :-- |
| размер тела | 2 МБ (иначе 413) | `NEXUS_MAX_BODY_BYTES` |
| длина кода | 200 000 символов | `NEXUS_MAX_CODE_CHARS` |
| сетка вокселизации | 6…64 | `NEXUS_MAX_GRID` |
| длина генерации | 2048 токенов | `NEXUS_MAX_NEW_TOKENS` |
| промптов в батче | 32 | — |
| параллельных тяжёлых операций | 4 (иначе 503 + `Retry-After`) | `NEXUS_MAX_CONCURRENCY` |

Ошибки приходят с машиночитаемым полем `code`: `validation_error`,
`bad_parameters`, `not_found`, `payload_too_large`, `busy`, `internal_error`.

## Поведение и ошибки

* Если запрошенной версии нет в реестре, поднимается модель из пресета и ответ
  помечается `"untrained": true` — сервис не падает на «пустой» инсталляции.
* `400` — некорректный JSON или неизвестные параметры, `404` — нет маршрута
  или задачи, `500` — ошибка обработки (в теле `{"error": "..."}`).
* Кэш моделей — `max_cached` версий, вытесняется самая старая; `promote` и
  `rollback` автоматически перечитывают активную модель.
* Аутентификации нет: сервис рассчитан на закрытый контур. Наружу — за
  reverse-proxy с TLS и авторизацией.
