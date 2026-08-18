# Импорт своих данных и выборка для визуальной модальности

У вас есть таблица `stl_items` с парами «ТЗ → OpenSCAD → картинка». Это лучший
источник данных из возможных: реальные запросы живых людей. Ниже — как
превратить её в обучающий корпус двумя командами.

---

## 1. Импорт: `nexus ingest`

```bash
# из дампа MySQL (самый простой путь: экспорт таблицы в .sql)
nexus ingest sql:stl_items.sql --out artifacts/ingest

# напрямую из базы (нужен pymysql)
nexus ingest mysql://user:pass@localhost/mysite --table stl_items --out artifacts/ingest

# только готовые записи, с проверкой прочности на алюминии
nexus ingest sql:dump.sql --statuses "готово,в работе" --material alu6061 --force 0 0 -300
```

Что происходит с каждой записью:

| Шаг | Что делает |
| :-- | :-- |
| разбор | понимает экранирование MySQL (`\n`, `\'`, `NULL`), кириллицу, многострочный код |
| нормализация | `stl_item_code_basis` → `spec`, `stl_item_code` → `code`, `stl_item_img` → `image` |
| дедупликация | по хешу кода без комментариев и пробелов — правки одного скрипта не дублируются |
| **валидация движком** | компиляция CSG, manifold, толщина стенки, масса, МКЭ, итоговая награда |
| параметры | из шапки вытаскивается таблица `имя = значение; // комментарий` |
| обогащение | если ТЗ пустое или короче 30 символов — можно дописать внешней LLM |
| разделение | `dataset.jsonl` (обучение) и `val.jsonl` (валидация, по умолчанию 10 %) |

На выходе:

```
artifacts/ingest/
  dataset.jsonl   # spec, code, params, physics, reward, text (готов для train-lm)
  val.jsonl       # честный холдаут
  rejects.jsonl   # что не прошло и почему (битый код виден сразу)
  stats.json      # сколько принято, сколько дублей, средняя награда
```

Проверка на трёх записях вашего формата: принято 2, дубли отброшены, битый
скрипт ушёл в `rejects.jsonl` с текстом ошибки, средняя награда 4.5 из 5.

### Дописать недостающие ТЗ внешней LLM

```bash
nexus ingest sql:dump.sql --enrich-command "claude -p"
# или: --enrich-command "ollama run qwen2.5-coder:7b"
```

Модель получает код и пишет по нему техническое задание. Работает только для
записей, где описания нет или оно слишком короткое — остальные не трогаются.

### Обучение на своём корпусе

```bash
nexus train-tokenizer --source "jsonl:artifacts/ingest/dataset.jsonl#text" --vocab-size 16384
nexus train-lm --source "mix:jsonl:artifacts/ingest/dataset.jsonl#text=0.6,flywheel:artifacts/flywheel=0.25,mathgen:20000=0.15" \
               --val-source "jsonl:artifacts/ingest/val.jsonl#text" \
               --tokenizer artifacts/tokenizer/bpe.json --preset small \
               --seq-len 1024 --max-steps 20000 --eval-every 500 --patience 4 \
               --device cuda --amp
```

Смесь важна: только на своих данных модель переобучится под их стиль, а маховик
и математика дают разнообразие и физику.

Флаг `--physics` дополнительно подаёт массу и нагрузку **числом** на общую шину
(а не только текстом `<fem>`) — подробности в
[PHYSICS_EXPLAINED.md](PHYSICS_EXPLAINED.md):

```bash
nexus train-lm --source "jsonl:artifacts/ingest/dataset.jsonl#text" --physics ...
```

### Чистить ли базу от кривых моделей

Нет: импорт сам отбраковывает некомпилируемое и дедуплицирует, а `rejects.jsonl`
— готовый список сломанных моделей сайта. Брак ещё и полезен: пара «сломанный код
→ замечание → исправление» учит модель чинить ошибки.

---

## 2. Картинки: `nexus render-dataset`

```bash
nexus render-dataset --input artifacts/ingest/dataset.jsonl --out artifacts/vision \
                     --size 512 --views iso,front,right,top --stl
```

Два бэкенда, выбираются автоматически:

* **OpenSCAD CLI** (у вас установлен) — честный рендер через `--camera --viewall
  --autocenter`, 6 стандартных ракурсов, любой размер;
* **встроенный растеризатор** — воксель + Z-буфер + освещение, чистый numpy,
  своя запись PNG. Работает без OpenSCAD и без Pillow, если бинарника нет.

Результат:

```
artifacts/vision/
  item_007077_iso.png  item_007077_front.png  …   # по 6 картинок на деталь
  item_007077.stl                                  # если указан --stl
  manifest.jsonl   # image ↔ code ↔ spec ↔ params ↔ physics
  stats.json
```

Каждая строка манифеста — готовая мультимодальная пара:

```json
{"item_id": 7077, "spec": "Прижимной фланец…", "code": "…",
 "images": [{"image": "item_007077_iso.png", "view": "iso", "angles": [55, 25],
             "renderer": "openscad"}],
 "params": [{"name": "flange_outer_d", "value": "110.0", "comment": "внешний диаметр"}],
 "physics": {"mass_g": 210.4, "safety_factor": 3.1}, "text": "<task>…<scad>…"}
```

Прочитать картинку обратно в массив для обучения:

```python
from nexus.data.vision import load_image
image = load_image("artifacts/vision/item_007077_iso.png")   # (H, W, 3) float32 0..1
```

### Зачем это нужно

С такой выборкой обучается визуальная модальность по схеме из архитектуры:
ядро замораживается, учится только лёгкий энкодер «картинка → латент».

```python
model = NexusEngine(NexusConfig.small())
model.freeze_core()                                   # ядро не трогаем
model.register_encoder("image", VideoSSMEncoder(model.cfg.d_latent))
```

Задачи, которые открываются: «фото детали → код», «чертёж → модель»,
«исправь деталь по картинке».

---

## 3. Практический порядок для вашего сайта

```bash
# 1. выгрузить таблицу
mysqldump -u user -p mysite stl_items > stl_items.sql

# 2. импорт с проверкой (несколько минут на тысячу записей)
nexus ingest sql:stl_items.sql --out artifacts/ingest --grid 20

# 3. посмотреть, что отбраковано и почему
head -3 artifacts/ingest/rejects.jsonl
cat artifacts/ingest/stats.json

# 4. картинки для визуальной модальности
nexus render-dataset --input artifacts/ingest/dataset.jsonl --out artifacts/vision --size 512

# 5. обучение
nexus train-tokenizer --source "jsonl:artifacts/ingest/dataset.jsonl#text" --vocab-size 16384
nexus train-lm --source "jsonl:artifacts/ingest/dataset.jsonl#text" \
               --val-source "jsonl:artifacts/ingest/val.jsonl#text" \
               --tokenizer artifacts/tokenizer/bpe.json --preset small --device cuda --amp
```

Полезные мелочи:

* `--limit 100` — прогнать сначала сотню записей и посмотреть статистику;
* `--no-validate` — быстрый импорт без физики (в 10 раз быстрее, но без меток);
* `--grid 14` — грубее сетка, быстрее проверка; `--grid 28` — точнее масса и σ;
* `rejects.jsonl` полезен сам по себе: это список сломанных моделей на сайте.
