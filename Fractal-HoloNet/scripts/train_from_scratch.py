"""
Каноническое обучение модели С НУЛЯ с регистрацией в реестре моделей.

Особенности:
  * конфиг модели строится ИЗ токенизатора (vocab_size/pad/bos/eos — единый
    источник истины; никакого рассинхрона конфиг vs словарь);
  * cosine LR с warmup, фиксированный seed, train/val сплит;
  * каждая модель получает уникальный id и свой каталог в registry/
    (чекпоинты никогда не перезаписываются);
  * метрики (val loss/ppl, params, время) записываются в реестр;
  * --activate делает модель активной для API.

Запуск:
  python scripts/train_from_scratch.py \
      --tokenizer registry/tokenizers/bpe-8192 \
      --epochs 4 --batch-size 64 --block-size 96

  # или с байтовым токенизатором:
  python scripts/train_from_scratch.py --tokenizer byte --epochs 4
"""
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from fractal_holonet.core import ProductionFractalHoloNet
from fractal_holonet.datasets import ByteDataset, get_russian_corpus
from fractal_holonet.tokenizer import (
    SimpleProductionTokenizer,
    BpeTokenizer,
    build_config_for_tokenizer,
)
import fractal_holonet.registry as registry


def load_train_tokenizer(spec: str):
    """spec: путь к каталогу с tokenizer.json (BPE) или 'byte' (легаси)."""
    if spec.lower() == "byte":
        return SimpleProductionTokenizer()
    path = Path(spec) / "tokenizer.json"
    if not path.exists():
        raise FileNotFoundError(f"tokenizer.json не найден: {path}")
    return BpeTokenizer.load(str(path))


def build_corpus(max_chars: int = 0) -> str:
    """Русский корпус + TinyShakespeare (опционально обрезается)."""
    ru = get_russian_corpus()
    bench_path = ROOT / "data" / "data_benchmark.txt"
    bench = bench_path.read_text(encoding="utf-8") if bench_path.exists() else ""
    corpus = ru + "\n" + bench
    if max_chars and len(corpus) > max_chars:
        corpus = corpus[:max_chars]
    return corpus


def make_scheduler(optimizer, base_lr: float, warmup_steps: int, total_steps: int):
    def step_lr(step: int):
        if step < warmup_steps:
            lr = base_lr * (step + 1) / max(1, warmup_steps)
        else:
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            lr = base_lr * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        for g in optimizer.param_groups:
            g["lr"] = lr
        return lr

    return step_lr


def evaluate(model, loader, criterion, device):
    model.eval()
    total, batches = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x, states=None, use_step=False)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            total += loss.item()
            batches += 1
    return total / max(1, batches)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="registry/tokenizers/bpe-8192",
                    help="каталог BPE-токенизатора или 'byte'")
    ap.add_argument("--model-id", default=None)
    ap.add_argument("--name", default=None)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--d-ff", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--block-size", type=int, default=96)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--warmup-ratio", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-chars", type=int, default=0, help="ограничение корпуса (0 = весь)")
    ap.add_argument("--val-ratio", type=float, default=0.05)
    ap.add_argument("--no-activate", action="store_true")
    ap.add_argument("--notes", default="")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cpu")

    print("=" * 74)
    print("  ОБУЧЕНИЕ МОДЕЛИ С НУЛЯ (Fractal-HoloNet v1) + РЕЕСТР")
    print("=" * 74)

    # 1. Токенизатор и конфиг из токенизатора
    tokenizer = load_train_tokenizer(args.tokenizer)
    tokenizer_type = "byte" if isinstance(tokenizer, SimpleProductionTokenizer) else "bpe"
    config = build_config_for_tokenizer(
        tokenizer,
        d_model=args.d_model,
        n_layers=args.layers,
        d_ff=args.d_ff,
        dropout=0.02,
    )
    model = ProductionFractalHoloNet(config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Токенизатор: {tokenizer_type} | vocab={tokenizer.vocab_size} | "
          f"pad={tokenizer.pad_token_id} eos={tokenizer.eos_token_id}")
    print(f"Архитектура: d={args.d_model}, layers={args.layers}, d_ff={args.d_ff} | "
          f"params={n_params:,}")

    # 2. Корпус и сплит
    corpus = build_corpus(args.max_chars)
    token_ids = tokenizer.encode(corpus, add_bos=False)
    split = int(len(token_ids) * (1.0 - args.val_ratio))
    train_ids, val_ids = token_ids[:split], token_ids[split:]
    print(f"Корпус: {len(corpus):,} символов -> {len(token_ids):,} токенов "
          f"(train {len(train_ids):,} / val {len(val_ids):,})")

    train_ds = ByteDataset(train_ids, block_size=args.block_size, pad_token_id=tokenizer.pad_token_id)
    val_ds = ByteDataset(val_ids, block_size=args.block_size, pad_token_id=tokenizer.pad_token_id)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    # 3. Регистрация в реестре
    model_id = registry.create_model(
        model_id=args.model_id,
        name=args.name or f"Fractal-HoloNet v1 ({tokenizer_type}, vocab {tokenizer.vocab_size})",
        architecture="fractal-holonet-v1",
        status="training",
        tokenizer_type=tokenizer_type,
        tokenizer_vocab=tokenizer.vocab_size,
        config_summary={
            "d_model": args.d_model, "n_layers": args.layers, "d_ff": args.d_ff,
            "vocab_size": config.vocab_size, "block_size": args.block_size,
        },
        notes=args.notes or f"Обучение с нуля. seed={args.seed}, lr={args.lr}, epochs={args.epochs}",
    )
    print(f"Реестр: модель зарегистрирована как '{model_id}' (статус: training)")

    # 4. Обучение
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    total_steps = args.epochs * len(train_loader)
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))
    set_lr = make_scheduler(optimizer, args.lr, warmup_steps, total_steps)

    t0 = time.time()
    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        total, batches = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            set_lr(global_step)
            optimizer.zero_grad()
            logits, _ = model(x, states=None, use_step=False)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()
            batches += 1
            global_step += 1
        train_loss = total / max(1, batches)
        val_loss = evaluate(model, val_loader, criterion, device)
        print(f"  epoch {epoch+1}/{args.epochs} | train {train_loss:.4f} | "
              f"val {val_loss:.4f} | ppl {math.exp(min(val_loss, 20)):.2f} | "
              f"lr {set_lr(global_step-1):.2e}")
    train_sec = time.time() - t0

    # 5. Сохранение + метрики в реестре
    registry.save_model_artifacts(model_id, model, tokenizer, status="trained")
    metrics = {
        "params": n_params,
        "train_loss": round(train_loss, 4),
        "val_loss": round(val_loss, 4),
        "val_ppl": round(math.exp(min(val_loss, 20)), 2),
        "train_sec": round(train_sec, 1),
        "corpus_chars": len(corpus),
        "corpus_tokens": len(token_ids),
        "epochs": args.epochs,
        "seed": args.seed,
        "lr": args.lr,
    }
    registry.record_metrics(model_id, metrics)
    if not args.no_activate:
        registry.activate(model_id)
        print(f"Модель '{model_id}' активирована (API: FH_MODEL_ID={model_id} или /v1/models/activate).")

    print(f"\n✅ Обучение завершено за {train_sec:.0f}s | id={model_id}")
    print(f"   Чекпоинт: {registry.model_dir(model_id)}")
    print(f"   Метрики: {json.dumps(metrics, ensure_ascii=False, indent=2)}")

    # 6. Демо генерации
    print("\n--- Демо генерации (greedy) ---")
    model.eval()
    for prompt in [
        "Искусственный интеллект — это",
        "Голографическая память нейросети сохраняет",
        "First Citizen:\nWe are accounted poor citizens",
    ]:
        ids = torch.tensor([tokenizer.encode(prompt, add_bos=False)], dtype=torch.long)
        gen = model.generate(ids, max_new_tokens=32, temperature=1.0, top_k=1)
        new_ids = gen[0, ids.size(1):].tolist()
        print(f"  [{prompt[:44]}]\n    -> {tokenizer.decode(new_ids, skip_special=True)}")


if __name__ == "__main__":
    main()
