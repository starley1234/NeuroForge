# Production deployment

## Один сервер

```bash
cp .env.example .env
docker compose up --build -d
curl http://localhost:8000/health/ready
```

UI: `http://server:8000`, OpenAPI: `/api/docs`. Контейнер запускается не от root, read-only, без capabilities, с `no-new-privileges`, лимитами RAM/CPU/PID и отдельным `/tmp`.

## Что обязательно перед публичным интернетом

1. TLS и reverse proxy (Caddy/Traefik/nginx).
2. Ограничение тела запроса до 256 KB и rate limit, например 30 generate/min и 5 STL renders/min на пользователя.
3. Аутентификация и quotas, если сервис не публичный.
4. Централизованные stdout logs, метрики p50/p95 и alerts по 5xx/render timeout.
5. Реплики API должны быть stateless; тяжёлый render лучше вынести в очередь worker-ов.
6. Зафиксировать digest базового Docker image и выполнять image/SBOM scan в release pipeline.

## Kubernetes ориентир

- API deployment: 2+ replicas, readiness `/health/ready`, liveness `/health/live`.
- Render worker pool: CPU limits 2–4 cores/job, 512 MB temporary volume, hard timeout.
- NetworkPolicy без исходящего трафика для renderer.
- HPA по latency/queue depth, не только CPU.
- Не размещать непроверенные model checkpoints автоматически: promotion gate должен быть отдельным CI job.

## Security model

Пользователь не передаёт SCAD/Python. API принимает только ограниченный CSG-IR, проверяет типы, глубину, количество узлов и диапазоны, затем компилирует своим emitter. OpenSCAD всё равно считается недоверенной нативной зависимостью и запускается внутри ограниченного контейнера.

## Backup и state

Текущий API stateless — резервировать нечего. При добавлении проектов хранить IR как канонический артефакт в PostgreSQL/object storage, а STL считать восстанавливаемым cache. Для каждого артефакта сохранять IR schema version, compiler version и model provenance.
