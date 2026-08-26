# NeuroSCAD

**Engineering intent → typed CSG → validated manufacturing artifacts.**

NeuroSCAD преобразует техническое задание в параметрическую модель, но не доверяет генератору: CSG-IR проверяется строгой грамматикой, OpenSCAD создаётся детерминированным компилятором, а уровень фактической проверки явно указывается в отчёте.

> Версия `0.4.1`: production-oriented foundation. Сейчас работают три семейства: хомуты, монтажные пластины и втулки. Обученных нейросетевых весов пока нет: созданы CAD pipeline, эталонные генераторы и training pipeline. Подробно: [что является моделью](docs/MODEL.md).

## Возможности

- русское/английское ТЗ → параметризованный CSG-IR;
- безопасный AST: ссылки, арность, циклы, диапазоны, глубина и лимит узлов;
- инженерные constraints между параметрами и pairwise slider sweep;
- параметрические выражения и интерактивные sliders;
- deterministic OpenSCAD, скачивание SCAD/STL/IR;
- OpenSCAD render с совместимостью старого CLI и Trimesh metrology;
- browser UI, FastAPI/OpenAPI, request IDs и security headers;
- non-root/read-only Docker deployment;
- воспроизводимый synthetic data engine, LoRA training profile и promotion gate;
- ingestion 500+ owner OpenSCAD scripts: audit, compile, STL metrics и 8 renders;
- resumable teacher annotation и сборка distillation/edit датасета;
- исполняемая оценка предсказанного OpenSCAD;
- реестр внешних датасетов с лицензионной политикой.

## Запуск одной командой

```bash
cd NeuroSCAD
docker compose up --build -d
```

Открыть `http://localhost:8000`. Проверить состояние:

```bash
curl http://localhost:8000/health/ready
```

Без Docker:

```bash
pip install -e '.[api,validation]'
uvicorn neuroscad.api:app --host 0.0.0.0 --port 8000
```

## CLI

```bash
python3 -m neuroscad.cli generate \
  'Кронштейн для камеры на трубу 25 мм с фиксацией винтом М4' \
  --ir examples/camera_bracket.json --scad examples/camera_bracket.scad
python3 -m neuroscad.cli validate examples/camera_bracket.json --render
python3 -m neuroscad.cli validate examples/camera_bracket.json --sweep
```

## Дистилляция на собственных OpenSCAD

```bash
# 0. Если источник — phpMyAdmin SQL dump, безопасно извлечь код без запуска SQL
python3 -m training.extract_sql_corpus --input data/stl_items.sql \
  --output data/sql-extracted --license owned

# 1. Проверить, скомпилировать и отрендерить исходные скрипты
python3 -m training.ingest_openscad --input data/sql-extracted \
  --output data/openscad-corpus --license owned

# 2. Получить grounded описания и планы от мультимодального teacher
export TEACHER_API_KEY=...
python3 -m training.annotate_teacher --corpus data/openscad-corpus \
  --output data/openscad-corpus/annotations.jsonl \
  --endpoint https://provider.example/v1/chat/completions --model teacher-model

# 3. Собрать SFT train/validation/test и обучить 3B LoRA
python3 -m training.build_distillation --corpus data/openscad-corpus \
  --annotations data/openscad-corpus/annotations.jsonl --output data/distilled
pip install -e '.[training]'
python3 -m training.train --config training/openscad_config.example.json
```

Модель не активируется автоматически. Подробный недорогой план: [LOW_COST_DISTILLATION.md](docs/LOW_COST_DISTILLATION.md). Синтетический CSG-IR pipeline остаётся доступен через `training.prepare`.

## Проверка проекта

```bash
make test
```

Тесты не требуют CUDA, OpenSCAD, FastAPI или доступа к сети.

## Структура

```text
neuroscad/ir.py             typed IR и static validation
neuroscad/compiler.py       deterministic OpenSCAD emitter
neuroscad/templates.py      текущий production fallback
neuroscad/validator.py      IR → geometry → mesh validation
neuroscad/api.py            HTTP API и artifacts
neuroscad/web/              responsive browser workbench
training/                   prepare, LoRA SFT, evaluation gate
data/catalog.json           dataset/license registry
schema/                     machine-readable IR contract
docs/                       architecture, datasets, training, deployment
Dockerfile / compose        ограниченное runtime-окружение
```

## Что означает `valid`

- `level: ir` — программа структурно корректна;
- `level: geometry` — OpenSCAD успешно получил STL;
- `level: mesh` — дополнительно проверены watertight и положительный объём.

Ни один из уровней не доказывает прочность. Для неё нужны материал, направления слоёв, закрепления, нагрузки, FEM convergence и инженерное утверждение.

## Документация

- [Что мы называем моделью и чего реально ждать](docs/MODEL.md)
- [Недорогая дистилляция на собственных OpenSCAD](docs/LOW_COST_DISTILLATION.md)
- [Разбор предоставленного SQL-примера](docs/SAMPLE_SQL_REPORT.md)
- [Целевая coarse-to-fine flow-архитектура](docs/MODEL_ARCHITECTURE_V2.md)
- [Архитектурные решения](docs/ARCHITECTURE.md)
- [Выбор датасетов и лицензии](docs/DATASETS.md)
- [Обучение и promotion](docs/TRAINING.md)
- [Production deployment](docs/DEPLOYMENT.md)
- [Исследованные наработки](docs/RESEARCH.md)
