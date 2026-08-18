"""Обучение на стандартном текстовом датасете (LM-режим).

Источник задаётся строкой (см. `nexus.data.corpora`): `builtin:engineering`,
`dir:./corpus`, `jsonl:data.jsonl#text`, `hf:wikitext/wikitext-2-raw-v1:train`,
`flywheel:artifacts/flywheel`.

Результат всегда сохраняется **новой версией** в реестр моделей.
"""
from __future__ import annotations

import argparse
import json
from typing import Dict, Optional

from ..config import NexusConfig
from ..data.corpora import PackedLMDataset, split_dataset
from ..registry import ModelRegistry
from .trainer import TrainConfig, Trainer, resume_or_new


def train(
    source: str = "builtin:engineering",
    model_name: str = "core",
    preset: str = "tiny",
    resume: Optional[str] = None,
    seq_len: int = 512,
    limit: Optional[int] = None,
    val_fraction: float = 0.1,
    tokenizer_path: Optional[str] = None,
    registry_root: str = "artifacts/registry",
    promote: bool = False,
    **train_kwargs,
) -> Dict[str, object]:
    cfg = {"tiny": NexusConfig.tiny, "small": NexusConfig.small,
           "rtx5060": NexusConfig.rtx5060,
           "rtx5060-compact": NexusConfig.rtx5060_compact}[preset]()

    from ..data.bpe import load_tokenizer
    tokenizer = load_tokenizer(tokenizer_path)
    ds = PackedLMDataset(source, tokenizer=tokenizer, seq_len=seq_len, limit=limit,
                         min_blocks=8)
    cfg.vocab_size = max(cfg.vocab_size, ds.tok.vocab_size)
    train_ds, val_ds = split_dataset(ds, val_fraction)
    print(f"[data] {source}: {ds.stats.to_dict()}")

    registry = ModelRegistry(registry_root)
    model, parent = resume_or_new(registry, model_name, cfg, resume,
                                  train_kwargs.get("device", "cpu"))
    tcfg = TrainConfig(model_name=model_name, stage="lm", dataset=source,
                       tokenizer_path=tokenizer_path, registry_root=registry_root,
                       **train_kwargs)
    trainer = Trainer(model, tcfg, train_ds, val_ds)
    metrics = trainer.fit()
    metrics["corpus_tokens"] = float(ds.stats.tokens)
    version = trainer.save(metrics, parent=parent)
    if promote:
        registry.promote(model_name, version.version, "production", reason="train_lm --promote")
    return {"metrics": metrics, "version": version.to_dict()}


def main() -> None:
    ap = argparse.ArgumentParser(description="Обучение NEXUS на стандартном датасете")
    ap.add_argument("--source", default="builtin:engineering")
    ap.add_argument("--model-name", default="core")
    ap.add_argument("--preset", choices=["tiny", "small", "rtx5060", "rtx5060-compact"], default="tiny")
    ap.add_argument("--resume", default=None, help="версия/тег для продолжения (latest, v0003)")
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--tokenizer", default=None, help="путь к обученному BPE (bpe.json)")
    ap.add_argument("--limit", type=int, default=None, help="макс. документов из источника")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--eval-every", type=int, default=0)
    ap.add_argument("--patience", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--eight-bit", action="store_true")
    ap.add_argument("--registry", default="artifacts/registry")
    ap.add_argument("--promote", action="store_true", help="сразу пометить как production")
    a = ap.parse_args()
    out = train(a.source, a.model_name, a.preset, a.resume, a.seq_len, a.limit,
                tokenizer_path=a.tokenizer, registry_root=a.registry, promote=a.promote,
                epochs=a.epochs, batch_size=a.batch_size, grad_accum=a.grad_accum,
                lr=a.lr, max_steps=a.max_steps, eval_every=a.eval_every,
                patience=a.patience, device=a.device, amp=a.amp, eight_bit=a.eight_bit)
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
