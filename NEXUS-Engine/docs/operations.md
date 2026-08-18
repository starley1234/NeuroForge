# Эксплуатация

Установка на сервере (systemd, nginx, Docker, бэкапы, мониторинг) —
[DEPLOYMENT.md](DEPLOYMENT.md). Здесь — жизненный цикл моделей.

## Жизненный цикл модели

```
train-lm / distill / pretrain / rl
        │  сохраняет НОВУЮ версию (перезапись невозможна)
        ▼
artifacts/registry/<model>/v0007/{model.pt, meta.json}
        │
        ├── nexus eval --ref latest --baseline production   приёмочные тесты
        │        └── --gate: тег production переключается только при успехе
        ▼
nexus serve                     инференс идёт по тегу production
        │
        └── nexus registry rollback --tag production        мгновенный откат
```

## Реестр

```bash
nexus registry list                              # все модели, версии, теги
nexus registry history --model-name core         # метрики и родитель каждой версии
nexus registry show --model-name core --ref production
nexus registry promote --model-name core --ref v0007 --tag production
nexus registry rollback --model-name core --tag production
nexus registry prune --model-name core --keep 5 --dry-run
nexus registry verify                            # sha256 всех версий и теги
```

Гарантии:

* версия пишется во временный каталог и появляется атомарно (`os.replace`);
* номер версии монотонный, перезапись существующей запрещена в API;
* `meta.json`: конфиг, метрики, родительская версия, датасет, стадия,
  git-коммит, sha256 весов, размер;
* `tags.json` — указатели, `tags.log` — журнал переключений (кто, когда, откуда);
* `load()` сверяет sha256 и падает на повреждённом файле;
* `prune` никогда не трогает версии под тегами.

## Приёмочные тесты дообученной модели

```bash
nexus eval --model-name core --ref latest --baseline production \
           --max-ppl 40 --min-compile 0.3 --max-regression 1.05 \
           --json artifacts/eval/core_v7.json --gate
```

Проверяется: конечность логитов, перплексия, вырожденность генерации,
детерминизм greedy, латентность, постоянство размера TTT-состояния, доля
компилируемых SCAD-генераций и регрессия к базовой версии.
Код возврата `0` — прошло, `1` — нет (удобно для CI).

## Типовые сценарии

```bash
# ночное дообучение с автоматическим гейтом и откатом при провале
nexus train-lm --source dir:/data/corpus --resume production --model-name core
nexus eval --model-name core --ref latest --baseline production --gate || \
  nexus registry rollback --model-name core --tag production

# обучение прямо через API, инференс не останавливается
curl -X POST localhost:8000/v1/jobs -d '{"args":["train-lm","--epochs","1"]}'
curl -X POST localhost:8000/v1/reload -d '{"model":"core","ref":"production"}'
```

## Docker

```bash
docker compose --profile train up train   # данные + первая версия модели в ./artifacts
docker compose up -d                      # API на :8000
docker compose logs -f nexus
```

Реестр монтируется томом `./artifacts:/data`, поэтому версии переживают
пересоздание контейнера. Ключ API — переменная `NEXUS_API_KEY`.

## Каталоги

| Путь | Содержимое | В git |
| :-- | :-- | :-- |
| `artifacts/registry/` | версии моделей и теги | нет |
| `artifacts/flywheel/` | датасет SCAD → 3D → FEM | нет |
| `artifacts/jobs/` | логи фоновых задач | нет |
| `artifacts/cache/` | кэш логитов учителя | нет |
| `artifacts/eval/` | отчёты приёмочных тестов | нет |
| `artifacts/tokenizer/` | обученный BPE (`bpe.json`) | нет |

Переменная `NEXUS_REGISTRY` меняет корень реестра по умолчанию.

## Мониторинг

`GET /health` — аптайм, устройство, число запросов, загруженные версии.
Логи API идут в stdout (`[api] ...`), логи задач — в `artifacts/jobs/<id>.log`.
Метрики каждой версии лежат в `meta.json` и доступны через `/v1/models`.
