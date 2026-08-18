"""Сохранение/загрузка чекпоинтов модели и конфигурации."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from ..core.config import NexusConfig
from ..models.nexus_model import NexusCapital


def save_checkpoint(
    model: NexusCapital,
    out_dir: str | Path,
    optimizer: torch.optim.Optimizer | None = None,
    step: int = 0,
    metrics: dict | None = None,
    tokenizer_name: str | None = None,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.cfg.save(str(out_dir / "config.json"))
    payload = {
        "model": model.state_dict(),
        "step": step,
        "metrics": metrics or {},
        "tokenizer_name": tokenizer_name or getattr(
            model.tokenizer, "name", "offline-bpe"),
        "vocab_size": model.cfg.vocab_size,
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    torch.save(payload, out_dir / "pytorch_model.pt")
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics or {}, f, ensure_ascii=False, indent=2)
    return out_dir


def load_checkpoint(
    path: str | Path,
    map_location: str | torch.device = "cpu",
    tokenizer=None,
) -> tuple[NexusCapital, dict]:
    path = Path(path)
    cfg = NexusConfig.load(str(path / "config.json"))
    if tokenizer is None:
        from ..core.tokenizer import NexusTokenizer
        tokenizer = NexusTokenizer.default(prefer_offline=True)
    model = NexusCapital(cfg, tokenizer=tokenizer)
    payload = torch.load(path / "pytorch_model.pt", map_location=map_location)
    model.load_state_dict(payload["model"])
    return model, payload
