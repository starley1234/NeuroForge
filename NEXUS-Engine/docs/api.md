# HTTP API

```bash
nexus serve --host 0.0.0.0 --port 8000 --model-name core --ref production
```

Только стандартная библиотека Python, без FastAPI/uvicorn. Открытый `GET /`
отдаёт мини-панель со списком эндпойнтов и живым статусом.

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

## Поведение и ошибки

* Если запрошенной версии нет в реестре, поднимается модель из пресета и ответ
  помечается `"untrained": true` — сервис не падает на «пустой» инсталляции.
* `400` — некорректный JSON или неизвестные параметры, `404` — нет маршрута
  или задачи, `500` — ошибка обработки (в теле `{"error": "..."}`).
* Кэш моделей — `max_cached` версий, вытесняется самая старая; `promote` и
  `rollback` автоматически перечитывают активную модель.
* Аутентификации нет: сервис рассчитан на закрытый контур. Наружу — за
  reverse-proxy с TLS и авторизацией.
