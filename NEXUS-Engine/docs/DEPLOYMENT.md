# Развёртывание в продакшене

Три сценария: **Docker** (проще всего), **systemd на сервере** (полный контроль),
**рядом с существующим сайтом** (наш случай — PHP-сайт вызывает локальный API).
В конце — эксплуатация: обновления, откат, бэкапы, мониторинг, безопасность.

---

## 1. Требования

| Что | Минимум | Рекомендуется |
| :-- | :-- | :-- |
| Python | 3.10 | 3.11 или 3.12 |
| RAM | 4 ГБ (только движок) | 16 ГБ (движок + модель) |
| Диск | 5 ГБ | 50 ГБ (реестр версий, датасеты, рендеры) |
| GPU | не нужен для анализа и МКЭ | RTX с 12+ ГБ для обучения и быстрой генерации |
| Внешнее | ничего | `openscad` (точный рендер), `ccx` (CalculiX) |

Движок геометрии и МКЭ работает на CPU. GPU нужен только для обучения и
ускорения генерации.

---

## 2. Docker (рекомендуемый путь)

```bash
git clone <repo> && cd NEXUS-Engine
export NEXUS_API_KEY="$(openssl rand -hex 24)"

docker compose --profile train up train    # данные + первая версия модели
docker compose up -d                       # API на :8000
docker compose logs -f nexus
curl -s localhost:8000/health
```

Тома и переменные:

```yaml
services:
  nexus:
    volumes: ["./artifacts:/data"]          # реестр, датасеты — переживают пересборку
    environment:
      NEXUS_API_KEY: "${NEXUS_API_KEY}"     # без него API открыт
      NEXUS_MAX_CONCURRENCY: "4"            # тяжёлых запросов одновременно
      NEXUS_MAX_GRID: "48"                  # предел сетки вокселизации
      NEXUS_MAX_BODY_BYTES: "2097152"       # 2 МБ на запрос
    deploy:
      resources:
        limits: { cpus: "4.0", memory: 8G }
    restart: unless-stopped
```

Проверка живости уже встроена в образ (`HEALTHCHECK` → `/health`).

---

## 3. systemd на сервере

```bash
sudo useradd -r -s /usr/sbin/nologin nexus
sudo mkdir -p /opt/nexus /var/lib/nexus
sudo chown -R nexus:nexus /opt/nexus /var/lib/nexus

sudo -u nexus git clone <repo> /opt/nexus
cd /opt/nexus && sudo -u nexus ./nexus.sh setup
```

`/etc/systemd/system/nexus-api.service`:

```ini
[Unit]
Description=NEXUS-Engine API
After=network-online.target

[Service]
Type=simple
User=nexus
WorkingDirectory=/opt/nexus
Environment="NEXUS_REGISTRY=/var/lib/nexus/registry"
Environment="NEXUS_MAX_CONCURRENCY=4"
EnvironmentFile=/etc/nexus/api.env          # здесь NEXUS_API_KEY
ExecStart=/opt/nexus/.venv/bin/python -m nexus.cli serve \
          --host 127.0.0.1 --port 8000 \
          --registry /var/lib/nexus/registry \
          --model-name core --ref production \
          --tokenizer /var/lib/nexus/tokenizer/bpe.json \
          --rate-limit 300
Restart=always
RestartSec=5
# защита процесса
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/var/lib/nexus
MemoryMax=8G
CPUQuota=400%

[Install]
WantedBy=multi-user.target
```

```bash
sudo install -d -m 750 /etc/nexus
printf 'NEXUS_API_KEY=%s\n' "$(openssl rand -hex 24)" | sudo tee /etc/nexus/api.env
sudo chmod 640 /etc/nexus/api.env && sudo chown root:nexus /etc/nexus/api.env
sudo systemctl daemon-reload && sudo systemctl enable --now nexus-api
sudo systemctl status nexus-api
```

Логи: `journalctl -u nexus-api -f`.

---

## 4. nginx и TLS

Сервис слушает `127.0.0.1` — наружу его пускает nginx:

```nginx
server {
    listen 443 ssl http2;
    server_name api.example.com;

    ssl_certificate     /etc/letsencrypt/live/api.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/api.example.com/privkey.pem;

    client_max_body_size 2m;          # совпадает с NEXUS_MAX_BODY_BYTES

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_read_timeout 300s;      # МКЭ и генерация бывают долгими
        proxy_buffering off;          # чтобы работал SSE-стриминг
        proxy_set_header X-Real-IP $remote_addr;
    }

    location /metrics {               # метрики — только для мониторинга
        allow 10.0.0.0/8;
        deny all;
        proxy_pass http://127.0.0.1:8000/metrics;
    }
}
```

---

## 5. Рядом с существующим сайтом

Самый простой и безопасный вариант: API слушает только localhost, PHP ходит к
нему напрямую, наружу ничего не открывается.

```php
function nexus_analyze(string $scad, array $force = [0, 0, -150]): array {
    $payload = json_encode([
        'code' => $scad, 'material' => 'pla', 'force' => $force, 'grid' => 20,
    ], JSON_UNESCAPED_UNICODE);

    $ch = curl_init('http://127.0.0.1:8000/v1/analyze');
    curl_setopt_array($ch, [
        CURLOPT_POST => true,
        CURLOPT_POSTFIELDS => $payload,
        CURLOPT_RETURNTRANSFER => true,
        CURLOPT_TIMEOUT => 60,
        CURLOPT_HTTPHEADER => [
            'Content-Type: application/json',
            'Authorization: Bearer ' . getenv('NEXUS_API_KEY'),
        ],
    ]);
    $raw = curl_exec($ch);
    $code = curl_getinfo($ch, CURLINFO_HTTP_CODE);
    curl_close($ch);

    if ($code === 503) return ['ok' => false, 'busy' => true];   // повторить позже
    if ($code !== 200) return ['ok' => false, 'error' => $raw];
    return json_decode($raw, true);
}
```

Коды ответов, которые стоит обработать:

| Код | Что значит | Что делать |
| :-- | :-- | :-- |
| 200 | всё хорошо | показать бейдж |
| 400 | некорректный вход (`validation_error`) | показать текст ошибки |
| 401 | нет или неверный ключ | проверить `NEXUS_API_KEY` |
| 413 | тело больше лимита | обрезать код или поднять `NEXUS_MAX_BODY_BYTES` |
| 429 | превышен rate-limit | повторить с задержкой |
| 503 | все слоты заняты (`busy`) | повторить через `Retry-After` секунд |

---

## 6. Настройки через переменные окружения

| Переменная | По умолчанию | Смысл |
| :-- | :-- | :-- |
| `NEXUS_API_KEY` | пусто (API открыт) | обязательный ключ `Authorization: Bearer` |
| `NEXUS_REGISTRY` | `artifacts/registry` | где лежат версии моделей |
| `NEXUS_MAX_CONCURRENCY` | 4 | тяжёлых операций одновременно; выше — риск OOM |
| `NEXUS_MAX_GRID` | 64 | предел сетки вокселизации в запросе |
| `NEXUS_MAX_BODY_BYTES` | 2 МБ | предел размера запроса |
| `NEXUS_MAX_CODE_CHARS` | 200 000 | предел длины кода |
| `NEXUS_MAX_NEW_TOKENS` | 2048 | предел длины генерации |
| `NEXUS_REQUEST_TIMEOUT_S` | 120 | сколько ждать свободный слот |
| `PYTHONUTF8` | — | ставьте `1` на Windows |

Все пределы применяются и в API, и в MCP-сервере: один запрос не может съесть
машину.

---

## 7. Обновление и откат

```bash
# обновление кода
cd /opt/nexus && sudo -u nexus git pull
sudo -u nexus ./.venv/bin/pip install -e ".[dev]"
sudo -u nexus ./.venv/bin/python -m pytest -q          # 141 тест, ~1 минута
sudo systemctl restart nexus-api

# обновление модели: новая версия не переключает трафик сама
nexus eval --model-name core --ref latest --baseline production --gate \
  && curl -X POST localhost:8000/v1/reload -H "Authorization: Bearer $KEY" \
          -d '{"model":"core","ref":"production"}'

# откат за секунды
nexus registry rollback --model-name core --tag production
curl -X POST localhost:8000/v1/reload -d '{"model":"core","ref":"production"}'
```

Правило: **код обновляется рестартом сервиса, модель — переключением тега.**
Одно не мешает другому.

---

## 8. Бэкапы и целостность

Ценное лежит в двух местах: реестр моделей и датасеты.

```bash
# бэкап (реестр небольшой: веса + метаданные)
tar czf /backup/nexus-registry-$(date +%F).tgz -C /var/lib/nexus registry tokenizer

# проверка целостности после восстановления или переноса
nexus registry verify --registry /var/lib/nexus/registry
# {"ok": true, "checked": 12, "problems": []}
```

`verify` сверяет sha256 каждой версии, наличие файлов и то, что теги указывают на
существующие версии. Код возврата `1` при проблемах — можно ставить в cron.

Чистка старого (версии под тегами не трогаются):

```bash
nexus registry prune --model-name core --keep 10 --dry-run
```

---

## 9. Мониторинг

```bash
curl -s localhost:8000/health     # аптайм, устройство, загруженные версии, отказы
curl -s localhost:8000/metrics    # формат Prometheus
```

`prometheus.yml`:

```yaml
scrape_configs:
  - job_name: nexus
    static_configs: [{ targets: ["127.0.0.1:8000"] }]
```

Что стоит держать на графике и в алертах:

| Метрика | Алерт |
| :-- | :-- |
| `nexus_rejected_busy_total` растёт | не хватает слотов → поднять `NEXUS_MAX_CONCURRENCY` или ресурсы |
| `nexus_generation_latency_ms{quantile="0.95"}` | деградация ответа |
| `nexus_requests_total` не растёт | сайт перестал ходить к API |
| `nexus_model_version` изменилась неожиданно | кто-то переключил тег |
| `up == 0` | сервис лежит |

---

## 10. Безопасность

* Слушать `127.0.0.1`, наружу — только через nginx с TLS.
* `NEXUS_API_KEY` обязателен, если API доступен не только с localhost;
  `/health`, `/docs`, `/openapi.json` намеренно открыты для проб живости.
* Rate-limit: `--rate-limit 300` (запросов в минуту с одного IP).
* Фоновые задачи (`/v1/jobs`) запускают процессы обучения — **не открывайте этот
  маршрут наружу**, ограничьте его в nginx по IP или отключите отдельным
  сервисом без jobs.
* Сервис исполняет только собственный парсер OpenSCAD (без `eval` и без запуска
  чужого кода); внешний `openscad` вызывается лишь при явном флаге.
* Запускать под отдельным пользователем с `ProtectSystem=strict`, как в
  примере systemd выше.

---

## 11. Проверка после развёртывания

```bash
nexus doctor                                        # окружение и самотест
curl -s localhost:8000/health | jq .status          # ok
curl -s -X POST localhost:8000/v1/analyze \
  -H "Authorization: Bearer $NEXUS_API_KEY" -H "Content-Type: application/json" \
  -d '{"code":"cube([20,20,4],center=true);","force":[0,0,-150]}' | jq .fem.safety_factor
nexus registry verify                               # целостность моделей
nexus registry list                                 # какие версии и теги активны
```

Если все четыре команды отвечают ожидаемо — сервис в рабочем состоянии.
