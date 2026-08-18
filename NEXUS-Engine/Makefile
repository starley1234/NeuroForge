PY ?= python

.PHONY: help install test demo flywheel ingest vision datasets gen-math collect mcp pretrain fno rl train-lm distill guide eval ablate serve registry needle vram

help:
	@grep -E '^[a-z-]+:.*?##' $(MAKEFILE_LIST) | sed 's/:.*##/ —/'

install:  ## поставить пакет в editable-режиме
	$(PY) -m pip install -e ".[dev]"

test:  ## юнит-тесты (CPU, ~30 с)
	$(PY) -m pytest -q

demo:  ## сквозная демонстрация всех уровней
	$(PY) -m nexus.cli demo --out artifacts/demo -n 12

flywheel:  ## сгенерировать датасет SCAD → 3D → FEM
	$(PY) -m nexus.cli flywheel -n 256 --out artifacts/flywheel --grid 24 --fem-grid 16

pretrain:  ## фаза 3: предобучение ядра
	$(PY) -m nexus.cli pretrain --data artifacts/flywheel --preset tiny --epochs 2

fno:  ## фаза 2: обучение FNO-суррогата FEM
	$(PY) -m nexus.cli train-fno --data artifacts/flywheel --epochs 30

rl:  ## фаза 4: Physics-RL (GRPO)
	$(PY) -m nexus.cli rl --steps 10

ingest:  ## импорт своих данных: make ingest SRC=sql:dump.sql
	$(PY) -m nexus.cli ingest $(SRC) --out artifacts/ingest

vision:  ## рендер картинок для визуальной модальности
	$(PY) -m nexus.cli render-dataset --input artifacts/ingest/dataset.jsonl --out artifacts/vision

datasets:  ## каталог открытых датасетов и рекомендованные миксы
	$(PY) -m nexus.cli datasets list

gen-math:  ## сгенерировать корпус верифицированной инженерной математики
	$(PY) -m nexus.cli gen-math -n 20000

collect:  ## дистилляция с физической проверкой (офлайн-учитель по умолчанию)
	$(PY) -m nexus.cli collect --teacher template -n 32 --attempts 3

mcp:  ## MCP-сервер: движок как инструменты для внешней LLM
	$(PY) -m nexus.cli mcp

train-lm:  ## обучение на стандартном датасете (встроенный корпус)
	$(PY) -m nexus.cli train-lm --source builtin:engineering --epochs 2

distill:  ## самодистилляция из последней версии в реестре
	$(PY) -m nexus.cli distill --teacher-nexus core --teacher-ref latest --mode logit

guide:  ## открыть регламент обучения
	@echo docs/TRAINING_GUIDE.md

eval:  ## приёмочные тесты последней версии против production
	$(PY) -m nexus.cli eval --model-name core --ref latest --baseline production

serve:  ## HTTP API на 0.0.0.0:8000
	$(PY) -m nexus.cli serve --host 0.0.0.0 --port 8000

registry:  ## содержимое реестра моделей
	$(PY) -m nexus.cli registry list

ablate:  ## A/B: что даёт каждый блок архитектуры
	$(PY) -m nexus.cli ablate --steps 300

needle:  ## тест O(1) памяти на длинном контексте
	$(PY) -m nexus.cli needle --lengths 1024,8192,32768

vram:  ## бюджет VRAM для RTX 5060
	$(PY) -m nexus.cli vram --preset rtx5060
