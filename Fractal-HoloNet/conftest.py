"""Корневой conftest: добавляет репозиторий в sys.path, чтобы тесты могли
импортировать пакет `fractal_holonet` и `research.*` независимо от способа
запуска pytest (python -m pytest или бинарник pytest)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
