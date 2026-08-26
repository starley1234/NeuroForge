# NEXUS-Capital — простые команды
# Использование:  make <цель>
# Все команды запускаются из корня проекта.

PYTHON ?= python3
PIP    ?= pip3

.PHONY: help install verify quick test train train-real serve demo clean

help: ## Показать список команд
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## Установить зависимости
	$(PIP) install -r requirements.txt

verify: ## Полная проверка (окружение, модель, тесты, обучение)
	$(PYTHON) verify.py

quick: ## Быстрая проверка без обучения (~10 сек)
	$(PYTHON) verify.py --quick

test: ## Запустить все юнит-тесты
	$(PYTHON) -m pytest tests/ -q -W ignore

train: ## Короткое обучение на синтетике
	$(PYTHON) scripts/train.py --steps 50 --batch-size 4

train-real: ## Обучение на реальных данных (Binance/FRED)
	$(PYTHON) scripts/train_real_data.py \
		--steps 100 --text-steps 50 --batch-size 4 --mc-paths 64

serve: ## Запустить веб-UI и API на http://127.0.0.1:8000
	$(PYTHON) serve.py

demo: ## Прогнать демо-пайплайн всех 4 уровней
	$(PYTHON) examples/demo_pipeline.py

clean: ## Удалить кэш и артефакты
	rm -rf .pytest_cache __pycache__ **/__pycache__ checkpoints data/raw data/processed
	find . -name "*.pyc" -delete
