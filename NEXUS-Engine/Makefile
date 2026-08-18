PY ?= python

.PHONY: help install test demo flywheel pretrain fno rl needle vram lint

help:
	@grep -E '^[a-z-]+:.*?##' $(MAKEFILE_LIST) | sed 's/:.*##/ —/'

install:  ## поставить пакет в editable-режиме
	$(PY) -m pip install -e ".[dev]"

test:  ## юнит-тесты (CPU, < 10 с)
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

needle:  ## тест O(1) памяти на длинном контексте
	$(PY) -m nexus.cli needle --lengths 1024,8192,32768

vram:  ## бюджет VRAM для RTX 5060
	$(PY) -m nexus.cli vram --preset rtx5060
