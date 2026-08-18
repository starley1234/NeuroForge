"""
Реестр обученных моделей Fractal-HoloNet.

Каталог реестра (по умолчанию <repo>/registry, переопределяется через
переменную окружения FH_REGISTRY_DIR):

    registry/
      registry.json          # индекс: {"version", "active", "models": {id: meta}}
      <model_id>/
        config.json
        pytorch_model.pt
        tokenizer.json       # legacy-словарь или HF byte-level BPE
        metadata.json        # имя, дата, git-коммит, статус, метрики

Правила:
  * каждый запуск обучения получает уникальный id и СВОЙ каталог —
    чекпоинты никогда не перезаписываются;
  * метаданные и метрики обновляются отдельными вызовами;
  * "active" указывает модель, которую будет обслуживать API при
    FH_MODEL_ID=active или после POST /v1/models/activate.
"""
import json
import os
import shutil
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fractal_holonet.core import ProductionFractalHoloNet
from fractal_holonet.tokenizer import load_tokenizer

_LOCK = threading.Lock()

REGISTRY_VERSION = 1


def get_registry_dir() -> Path:
    env = os.getenv("FH_REGISTRY_DIR")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[1] / "registry"


def git_commit_short() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return "unknown"


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_index() -> Dict[str, Any]:
    path = get_registry_dir() / "registry.json"
    if not path.exists():
        return {"version": REGISTRY_VERSION, "active": None, "models": {}}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data.setdefault("models", {})
    data.setdefault("active", None)
    return data


def _save_index(data: Dict[str, Any]):
    registry_dir = get_registry_dir()
    registry_dir.mkdir(parents=True, exist_ok=True)
    with open(registry_dir / "registry.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _resolve_architecture(config: Dict[str, Any]):
    """Возвращает класс модели по конфигу (v1 text / multimodal / v2)."""
    if "input_signal_dim" in config:
        from fractal_holonet.multimodal import MultimodalFractalHoloNet, MultimodalSignalConfig
        return MultimodalFractalHoloNet, MultimodalSignalConfig
    if "n_read_iters" in config or "use_slow_memory" in config:
        from research.arch_v2_core import ElasticHoloNet, ElasticHoloConfig
        return ElasticHoloNet, ElasticHoloConfig
    return ProductionFractalHoloNet, None


def create_model(
    model_id: Optional[str] = None,
    name: str = "",
    architecture: str = "fractal-holonet-v1",
    status: str = "initialized",
    tokenizer_type: str = "",
    tokenizer_vocab: int = 0,
    config_summary: Optional[Dict[str, Any]] = None,
    notes: str = "",
) -> str:
    """Регистрирует новую модель и возвращает её id."""
    with _LOCK:
        data = _load_index()
        if model_id is None:
            ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
            model_id = f"fhnet-{ts}"
        if model_id in data["models"]:
            raise ValueError(f"Model id already exists: {model_id}")
        meta = {
            "id": model_id,
            "name": name or model_id,
            "architecture": architecture,
            "status": status,
            "created_at": _utc_now(),
            "updated_at": _utc_now(),
            "git_commit": git_commit_short(),
            "tokenizer_type": tokenizer_type,
            "tokenizer_vocab": tokenizer_vocab,
            "config_summary": config_summary or {},
            "metrics": {},
            "notes": notes,
        }
        data["models"][model_id] = meta
        _save_index(data)
        (get_registry_dir() / model_id).mkdir(parents=True, exist_ok=True)
        return model_id


def update_model(model_id: str, **fields) -> None:
    with _LOCK:
        data = _load_index()
        if model_id not in data["models"]:
            raise KeyError(f"Unknown model: {model_id}")
        data["models"][model_id].update(fields)
        data["models"][model_id]["updated_at"] = _utc_now()
        _save_index(data)


def record_metrics(model_id: str, metrics: Dict[str, Any]) -> None:
    with _LOCK:
        data = _load_index()
        if model_id not in data["models"]:
            raise KeyError(f"Unknown model: {model_id}")
        meta = data["models"][model_id]
        meta.setdefault("metrics", {}).update(metrics)
        meta["updated_at"] = _utc_now()
        _save_index(data)
    # продублировать в каталог модели
    model_dir = get_registry_dir() / model_id
    model_dir.mkdir(parents=True, exist_ok=True)
    with open(model_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(meta["metrics"], f, ensure_ascii=False, indent=2)


def get_model(model_id: str) -> Dict[str, Any]:
    data = _load_index()
    if model_id not in data["models"]:
        raise KeyError(f"Unknown model: {model_id}")
    return data["models"][model_id]


def list_models() -> List[Dict[str, Any]]:
    data = _load_index()
    active_id = data.get("active")
    models = []
    for m in data["models"].values():
        entry = dict(m)
        entry["active"] = entry["id"] == active_id
        models.append(entry)
    models.sort(key=lambda m: m.get("created_at", ""), reverse=True)
    return models


def activate(model_id: Optional[str]) -> None:
    """Делает модель активной (None — сброс)."""
    with _LOCK:
        data = _load_index()
        if model_id is not None and model_id not in data["models"]:
            raise KeyError(f"Unknown model: {model_id}")
        data["active"] = model_id
        _save_index(data)


def get_active() -> Optional[str]:
    return _load_index().get("active")


def model_dir(model_id: str) -> Path:
    data = _load_index()
    if model_id not in data["models"]:
        raise KeyError(f"Unknown model: {model_id}")
    return get_registry_dir() / model_id


def save_model_artifacts(
    model_id: str,
    model,
    tokenizer,
    status: str = "trained",
) -> Path:
    """Сохраняет веса/конфиг/токенизатор в каталог модели реестра."""
    directory = model_dir(model_id)
    directory.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(directory))
    tokenizer.save(os.path.join(str(directory), "tokenizer.json"))
    update_model(model_id, status=status)
    return directory


def load_model(model_id: str, map_location: str = "cpu"):
    """Загружает (model, tokenizer) из реестра с автоопределением архитектуры."""
    directory = model_dir(model_id)
    config = json.load(open(directory / "config.json", encoding="utf-8"))
    cls, cfg_cls = _resolve_architecture(config)
    if cfg_cls is not None:
        config = cfg_cls.load(str(directory / "config.json"))
    model = cls.from_pretrained(str(directory), map_location=map_location)
    model.eval()
    tokenizer = load_tokenizer(str(directory))
    return model, tokenizer


def import_existing(
    model_id: str,
    name: str,
    checkpoint_dir: str,
    architecture: str,
    notes: str = "",
    metrics: Optional[Dict[str, Any]] = None,
    tokenizer_type: str = "",
    tokenizer_vocab: int = 0,
    config_summary: Optional[Dict[str, Any]] = None,
    copy_artifacts: bool = True,
) -> str:
    """Импортирует существующий чекпоинт в реестр (копирование файлов)."""
    checkpoint_dir = Path(checkpoint_dir)
    create_model(
        model_id=model_id,
        name=name,
        architecture=architecture,
        status="imported",
        tokenizer_type=tokenizer_type,
        tokenizer_vocab=tokenizer_vocab,
        config_summary=config_summary,
        notes=notes,
    )
    directory = model_dir(model_id)
    if copy_artifacts:
        for fname in ("config.json", "pytorch_model.pt", "tokenizer.json"):
            src = checkpoint_dir / fname
            if src.exists():
                shutil.copy2(str(src), str(directory / fname))
    if metrics:
        record_metrics(model_id, metrics)
    return model_id
