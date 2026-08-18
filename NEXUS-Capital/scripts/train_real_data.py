"""
Обучение NEXUS-Capital на реальных открытых данных.

Порядок действий (Фазы 2–3 дорожной карты):
  1) Скачать реальные данные:
       - Binance spot 1m klines (без ключа, прямые ZIP с data.binance.vision)
       - SEC EDGAR XBRL bulk (без ключа, public domain, при --with-sec)
       - FRED макро (бесплатный ключ в FRED_API_KEY; без ключа — оффлайн-профиль)
  2) Подготовить батчи: тики + L2-стакан + поля отчётности + макро
  3) Обучить ядро с совокупной функцией потерь
       L_total = L_pred - λ1·Utility + λ2·VaR + L_balance + L_aux
  4) Оценить метрики риска (VaR, ES, P(default), Sharpe) на hold-out.

Примеры:
    # Минимальный запуск на CPU, без SEC (быстро):
    python scripts/train_real_data.py --steps 50 --batch-size 4

    # С реальной отчётностью SEC (первый кач ~сотни МБ):
    python scripts/train_real_data.py --with-sec --sec-years 2023 2024 \\
        --steps 200 --batch-size 4

    # GPU:
    python scripts/train_real_data.py --device cuda --steps 500 \\
        --batch-size 8 --mc-paths 256

Для FRED задайте переменную окружения:
    export FRED_API_KEY=ваш_ключ_с_fred.stlouisfed.org
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nexus_capital import small_config, build_model
from nexus_capital.core.config import NexusConfig
from nexus_capital.core.loader import load_config
from nexus_capital.core.tokenizer import NexusTokenizer
from nexus_capital.data.real import (
    BinanceConfig, EdgarConfig, FredConfig, RealDataConfig,
    RealMarketDataset, collate_real,
)
from nexus_capital.data.real.text_dataset import (
    FinancialTextDataset, TextDataConfig, collate_text,
)
from nexus_capital.data.real.corpus import CorpusConfig
from nexus_capital.training.trainer import NexusTrainer, TrainConfig
from nexus_capital.workspace.monte_carlo import value_at_risk


def build_data_config(args) -> RealDataConfig:
    bc = BinanceConfig(
        symbol=args.symbol, interval=args.interval, market="spot",
        start_month=args.start_month, end_month=args.end_month,
        cache_dir=str(Path(args.data_dir) / "binance"),
    )
    ec = None
    if args.with_sec:
        ec = EdgarConfig(
            years=list(args.sec_years), quarters=[1, 2, 3, 4],
            cache_dir=str(Path(args.data_dir) / "sec"),
        )
    fc = FredConfig(
        start=args.fred_start, end=args.fred_end,
        cache_dir=str(Path(args.data_dir) / "fred"),
    )
    return RealDataConfig(
        binance=bc, edgar=ec, fred=fc,
        tick_seq_len=args.seq_len, n_fields=16,
        n_tick_features=8, n_levels=args.book_levels,
        task=args.task,
        processed_dir=str(Path(args.data_dir) / "processed"),
        max_samples=args.max_samples,
    )


def build_model_config(args) -> NexusConfig:
    if args.config and Path(args.config).exists():
        cfg = load_config(args.config)
    else:
        cfg = small_config()
    # Подгоняем размерности под данные
    cfg.orderbook_levels = args.book_levels
    cfg.tick_features = 8
    cfg.tabular_features = 16
    cfg.text_seq_len = max(args.seq_len, cfg.text_seq_len)
    cfg.text_n_pos = cfg.text_seq_len
    cfg.max_seq_len = cfg.text_seq_len
    cfg.mc_paths = args.mc_paths
    cfg.mc_horizon = args.mc_horizon
    return cfg


def make_train_step(trainer: NexusTrainer, cfg: NexusConfig, device):
    """Возвращает функцию шага обучения на реальном батче."""
    def step(batch: dict) -> dict:
        b = {k: v.to(device) if torch.is_tensor(v) else v
             for k, v in batch.items()}
        model = trainer.model
        model.train()

        # Кодируем рыночные данные через модель
        fused = model.encode_modalities(
            book=b["book"], ticks=b["ticks"],
            fields=b["fields"], field_mask=b["field_mask"],
        ).unsqueeze(1)
        h, bus_info = model.bus(fused, use_pos=False)

        # Прогноз следующей доходности — линейная головa поверх латентности
        # (обучаем «непрерывный вывод» цен)
        pred_ret = model.output.continuous.price(h).squeeze(-1)  # (B,1)
        target = b["target_return"].unsqueeze(-1)                # (B,1)
        # Лог-чувствительный MSE
        pred_loss = torch.nn.functional.mse_loss(pred_ret, target)

        # Риск через воркспейс (Монте-Карло) — для Utility/VaR члена
        ws = model.workspace(
            h.squeeze(1), mode="risk", discount=cfg.discount_rate)
        returns = ws["risk"]["pnl"]
        cost = model.output.continuous.utility(h.squeeze(1))["cost"]

        total, logs = model.loss_fn(
            pred_loss, returns=returns, cost=cost,
            aux_loss=bus_info["aux_loss"])

        trainer.opt.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), trainer.cfg.grad_clip)
        trainer.opt.step()
        trainer.step_count += 1
        return {k: float(v) for k, v in logs.items()}
    return step


@torch.no_grad()
def evaluate(model, loader, device, n_batches: int = 10) -> dict:
    model.eval()
    preds, targets = [], []
    risks = []
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        b = {k: v.to(device) if torch.is_tensor(v) else v
             for k, v in batch.items()}
        fused = model.encode_modalities(
            book=b["book"], ticks=b["ticks"],
            fields=b["fields"], field_mask=b["field_mask"],
        ).unsqueeze(1)
        h, _ = model.bus(fused, use_pos=False)
        pred = model.output.continuous.price(h).squeeze(-1)
        preds.append(pred.cpu().numpy().reshape(-1))
        targets.append(b["target_return"].cpu().numpy().reshape(-1))
        ws = model.workspace(h.squeeze(1), mode="risk")
        risks.append(ws["risk"]["pnl"].cpu().numpy())

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    risks = np.concatenate(risks, axis=0)
    mse = float(np.mean((preds - targets) ** 2))
    # Направление (sign accuracy)
    sign_acc = float(np.mean(np.sign(preds) == np.sign(targets)))
    # Эмпирический Sharpe предсказания
    if preds.std() > 1e-9:
        sharpe = float(preds.mean() / preds.std())
    else:
        sharpe = 0.0
    pnl_t = torch.from_numpy(risks)
    var5 = float(value_at_risk(pnl_t, 0.05).mean())
    return {
        "mse": mse, "sign_acc": sign_acc, "sharpe_pred": sharpe,
        "var5_mean": var5,
        "pred_mean": float(preds.mean()), "pred_std": float(preds.std()),
    }


def train_text(model, tokenizer, args, languages=("en", "ru"),
               device=torch.device("cpu")) -> dict:
    """
    Этап двуязычного (RU+EN) LM-обучения на финансовом корпусе.
    Использует предобученный токенизатор XLM-R (словарь НЕ обучается).
    """
    from torch.utils.data import DataLoader

    ccfg = CorpusConfig(
        use_sec=args.with_sec_text,
        use_cbr=True,
        cache_dir=str(Path(args.data_dir) / "corpus"),
    )
    tcfg = TextDataConfig(
        max_length=args.text_max_len,
        languages=languages,
        corpus=ccfg,
        val_frac=0.1,
    )
    train_ds = FinancialTextDataset(tokenizer, tcfg, split="train")
    val_ds = FinancialTextDataset(tokenizer, tcfg, split="val")
    train_loader = DataLoader(
        train_ds, batch_size=args.text_batch_size, shuffle=True,
        collate_fn=lambda b: collate_text(b, tokenizer.pad_id),
        num_workers=args.workers, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.text_batch_size, shuffle=False,
        collate_fn=lambda b: collate_text(b, tokenizer.pad_id),
        num_workers=args.workers,
    )

    n_en = sum(1 for item in train_ds.samples if item[0] == "en")
    n_ru = sum(1 for item in train_ds.samples if item[0] == "ru")
    print(f"  корпус: train={len(train_ds)} (EN={n_en}, RU={n_ru}), "
          f"val={len(val_ds)}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.text_lr,
                            weight_decay=0.01)
    model.train()
    data_iter = iter(train_loader)
    running: dict[str, float] = {}
    t0 = time.time()
    for step in range(args.text_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)
        batch = {k: v.to(device) if torch.is_tensor(v) else v
                 for k, v in batch.items()}
        out = model.text_forward(
            input_ids=batch["input_ids"],
            labels=batch["labels"],
            attention_mask=batch["attention_mask"],
            number_values=batch["number_values"],
            number_mask=batch["number_mask"],
        )
        opt.zero_grad(set_to_none=True)
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        for k, v in out["loss_logs"].items():
            running[k] = running.get(k, 0.0) + float(v)
        if step % max(1, args.text_steps // 5) == 0 or step == args.text_steps - 1:
            avg = {k: v / (step + 1) for k, v in running.items()}
            line = "  ".join(f"{k}={v:.4f}" for k, v in avg.items())
            print(f"    [text step {step:4d}] {line}")

    dt = time.time() - t0
    # Оценка перплексии на валидации
    model.eval()
    total_loss, n = 0.0, 0
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v
                     for k, v in batch.items()}
            out = model.text_forward(
                input_ids=batch["input_ids"], labels=batch["labels"],
                attention_mask=batch["attention_mask"],
                number_values=batch["number_values"],
                number_mask=batch["number_mask"],
                add_economic_loss=False,
            )
            total_loss += float(out["loss"]) * batch["input_ids"].shape[0]
            n += batch["input_ids"].shape[0]
    val_loss = total_loss / max(1, n)
    perplexity = float(min(np.exp(min(val_loss, 20)), 1e6))
    print(f"  text val_loss={val_loss:.4f}  perplexity={perplexity:.2f}  "
          f"({args.text_steps/dt:.1f} шаг/с)")
    model.train()
    return {"val_loss": val_loss, "perplexity": perplexity,
            "steps_per_sec": args.text_steps / dt}


def main() -> None:
    ap = argparse.ArgumentParser(description="NEXUS-Capital на реальных данных")
    ap.add_argument("--config", type=str, default="",
                    help="YAML-конфиг модели (необязательно)")
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--interval", default="1m")
    ap.add_argument("--start-month", default="2024-01")
    ap.add_argument("--end-month", default="2024-03")
    ap.add_argument("--with-sec", action="store_true",
                    help="скачать и использовать SEC EDGAR (10-K/10-Q)")
    ap.add_argument("--sec-years", type=int, nargs="+", default=[2023, 2024])
    ap.add_argument("--fred-start", default="2018-01-01")
    ap.add_argument("--fred-end", default="2025-01-01")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--book-levels", type=int, default=16)
    ap.add_argument("--task", default="risk",
                    choices=["invariants", "risk", "pricing"])
    ap.add_argument("--max-samples", type=int, default=20000)

    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--mc-paths", type=int, default=128)
    ap.add_argument("--mc-horizon", type=int, default=8)
    ap.add_argument("--workers", type=int, default=0)

    # Двуязычное обучение (русский + английский)
    ap.add_argument("--languages", nargs="+", default=["en", "ru"],
                    choices=["en", "ru"],
                    help="языки финансового корпуса")
    ap.add_argument("--text-steps", type=int, default=100,
                    help="число шагов LM-обучения RU+EN (0 = пропустить)")
    ap.add_argument("--text-batch-size", type=int, default=8)
    ap.add_argument("--text-max-len", type=int, default=192)
    ap.add_argument("--text-lr", type=float, default=3e-4)
    ap.add_argument("--with-sec-text", action="store_true",
                    help="подтянуть реальный текст SEC 10-K/10-Q для EN")
    ap.add_argument("--offline-tokenizer", action="store_true",
                    help="принудительно использовать встроенный BPE "
                         "(без скачивания XLM-R)")
    args = ap.parse_args()

    print("=" * 70)
    print("NEXUS-Capital — обучение на реальных открытых данных")
    print("=" * 70)

    # 1) Данные
    print("\n[1/4] Подготовка данных...")
    dcfg = build_data_config(args)
    dataset = RealMarketDataset(dcfg)
    if len(dataset) == 0:
        print("Не удалось собрать датасет (проверьте доступ к интернету/источникам).")
        sys.exit(1)

    n_train = int(len(dataset) * 0.9)
    n_val = len(dataset) - n_train
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, collate_fn=collate_real, drop_last=True)
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, collate_fn=collate_real)
    print(f"  train={n_train}  val={n_val}  примеров")

    device = torch.device(args.device)

    # 2) Модель с предобученным токенизатором (RU+EN)
    print("\n[2/4] Загрузка токенизатора и сборка модели...")
    tokenizer = NexusTokenizer.default(prefer_offline=args.offline_tokenizer)
    mcfg = build_model_config(args)
    model = build_model(mcfg, tokenizer=tokenizer)
    print(f"  токенизатор: {tokenizer.name}, vocab={tokenizer.vocab_size}")
    print(f"  параметров: {model.count_parameters()/1e6:.2f}M")
    print(f"  d_value={mcfg.d_value}, layers={mcfg.n_layers}, "
          f"experts={mcfg.n_experts}/{mcfg.experts_per_token}, "
          f"mc_paths={mcfg.mc_paths}")

    # 3) Двуязычное LM-обучение (русский + английский)
    if args.text_steps > 0:
        print(f"\n[3/4] Двуязычное LM-обучение ({', '.join(args.languages)}), "
              f"{args.text_steps} шагов...")
        text_metrics = train_text(
            model, tokenizer, args,
            languages=tuple(args.languages),
            device=device,
        )
        print("  Итог текстового этапа:")
        for k, v in text_metrics.items():
            print(f"    {k:18s} = {v:.4f}")
    else:
        print("\n[3/4] Текстовый этап пропущен (--text-steps 0).")

    tcfg = TrainConfig(lr=args.lr, batch_size=args.batch_size,
                       steps=args.steps, device=args.device)
    trainer = NexusTrainer(model, tcfg)
    step_fn = make_train_step(trainer, mcfg, device)

    # 4) Обучение на рыночных данных
    print(f"\n[4/4] Обучение на рыночных данных ({args.steps} шагов)...")
    t0 = time.time()
    running: dict[str, float] = {}
    data_iter = iter(train_loader)
    for step in range(args.steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)
        logs = step_fn(batch)
        for k, v in logs.items():
            running[k] = running.get(k, 0.0) + v
        if step % max(1, args.steps // 10) == 0 or step == args.steps - 1:
            avg = {k: v / (step + 1) for k, v in running.items()}
            line = "  ".join(f"{k}={v:.4f}" for k, v in avg.items())
            print(f"  [step {step:4d}] {line}")

    dt = time.time() - t0
    print(f"\nОбучение: {dt:.1f}с ({args.steps/dt:.1f} шаг/с)")

    # 4) Оценка
    print("\nОценка на hold-out...")
    metrics = evaluate(model, val_loader, device)
    for k, v in metrics.items():
        print(f"  {k:14s} = {v:.6f}")

    # 5) Сохранение
    from nexus_capital.training.checkpoint import save_checkpoint
    out = save_checkpoint(
        model, Path("checkpoints/nexus_real"),
        optimizer=trainer.opt, step=trainer.step_count, metrics=metrics,
        tokenizer_name=tokenizer.name)
    print(f"\nЧекпоинт сохранён: {out}")


if __name__ == "__main__":
    main()
