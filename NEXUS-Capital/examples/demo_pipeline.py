"""
Полный демо-прогон NEXUS-Capital на синтетических данных (CPU).

Прогоняет все 4 уровня архитектуры:
  1. кодирование 5 модальностей,
  2. Value Bus с TTT/вниманием/MoE,
  3. латентный Монте-Карло + теорию игр,
  4. двухрежимный вывод (дискретный + непрерывный).

Запуск:
    python examples/demo_pipeline.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time

import numpy as np
import torch

from nexus_capital import small_config, build_model
from nexus_capital.data.orderbook_stream import collate_book_ticks
from nexus_capital.data.edgar_parser import (
    synthetic_filing, fields_to_tensor, DEFAULT_UNIT_FIELDS,
    parse_text_filing,
)
from nexus_capital.data.synthetic_market import (
    generate_stress_dataset, stress_to_tensors, apply_shock,
)
from nexus_capital.data.self_play import run_b2b_negotiation


def banner(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def main() -> None:
    torch.manual_seed(0)
    np.random.seed(0)

    cfg = small_config()
    cfg.mc_paths = 512          # больше путей — стабильнее метрики риска
    cfg.mc_horizon = 16
    print("Конфигурация:")
    print(f"  d_value          = {cfg.d_value}")
    print(f"  n_layers         = {cfg.n_layers}")
    print(f"  n_experts        = {cfg.n_experts} (активны {cfg.experts_per_token})")
    print(f"  MC paths/horizon = {cfg.mc_paths}/{cfg.mc_horizon}")

    model = build_model(cfg)
    n_params = model.count_parameters() / 1e6
    print(f"  параметров       = {n_params:.2f}M")
    model.eval()

    # ── Уровень 1: модальности ────────────────────────────────────
    banner("УРОВЕНЬ 1 — кодирование модальностей")
    B = 2
    book_ticks = collate_book_ticks(
        batch_size=B, n_levels=cfg.orderbook_levels, T=32,
        n_features=cfg.tick_features)
    print(f"L2 стакан: {tuple(book_ticks['book'].shape)}, "
          f"тики: {tuple(book_ticks['ticks'].shape)}")

    rng = np.random.default_rng(42)
    fields, masks = [], []
    for _ in range(B):
        rec = synthetic_filing(rng)
        f, m = fields_to_tensor(rec, DEFAULT_UNIT_FIELDS[:cfg.tabular_features])
        fields.append(f)
        masks.append(m)
    fields = torch.stack(fields)
    masks = torch.stack(masks)
    print(f"Юнит-экономика: {tuple(fields.shape)} полей")

    tokens = torch.randint(0, cfg.vocab_size, (B, 32))
    print(f"Текст/новости:  {tuple(tokens.shape)} токенов")

    # ── Уровни 2-4: risk-режим ────────────────────────────────────
    banner("УРОВНИ 2–4 — оценка риска (risk)")
    t0 = time.time()
    with torch.no_grad():
        out = model(
            book=book_ticks["book"], ticks=book_ticks["ticks"],
            fields=fields, field_mask=masks, tokens=tokens,
            workspace_mode="risk",
        )
    dt = (time.time() - t0) * 1000
    risk = out["workspace"]["risk"]
    print(f"Время форварда ({cfg.mc_paths} MC-путей): {dt:.1f} мс")
    print(f"  E[P&L]      = {risk['expected_pnl'].tolist()}")
    print(f"  σ(P&L)      = {risk['std_pnl'].tolist()}")
    print(f"  VaR_5%      = {risk['var'].tolist()}")
    print(f"  ES_5%       = {risk['expected_shortfall'].tolist()}")
    print(f"  P(default)  = {risk['default_prob'].tolist()}")

    inv = out["invariants"]
    print(f"Инварианты Value Bus:")
    print(f"  CF     = {inv['cf'].squeeze(-1).tolist()}")
    print(f"  sigma  = {inv['sigma'].squeeze(-1).tolist()}")
    print(f"  disc.  = {inv['discount'].squeeze(-1).tolist()}")
    print(f"  E_d    = {inv['elasticity'].squeeze(-1).tolist()}")

    eco = out["economic"]
    print("Непрерывный вывод:")
    print(f"  оптимальная цена     = {eco['price'].squeeze(-1).tolist()}")
    print(f"  P(default)           = {eco['default_prob'].tolist()}")
    print(f"  сумма весов портфеля = {eco['portfolio_weights'].sum(-1).tolist()}")
    print(f"  спред ликвидности    = {eco['spread'].tolist()}")
    print(f"  ожидаемая полезность = "
          f"{eco['utility']['utility'].tolist()}")

    # ── Режим ценообразования Бертрана ────────────────────────────
    banner("УРОВЕНЬ 3 — динамическое ценообразование (Bertrand)")
    comp_price = torch.tensor([99.0, 149.0])
    with torch.no_grad():
        out_p = model(
            book=book_ticks["book"], ticks=book_ticks["ticks"],
            fields=fields, field_mask=masks, tokens=tokens,
            workspace_mode="pricing", competitor_price=comp_price,
        )
    ws = out_p["workspace"]
    for i in range(B):
        print(f"  конкурент={comp_price[i].item():.2f}  ->  "
              f"opt_price={ws['optimal_price'][i].item():.2f}  "
              f"(cost={ws['cost'][i].item():.2f}, "
              f"demand={ws['demand'][i].item():.2f})")

    # ── Режим переговоров (Nash bargaining) ───────────────────────
    banner("УРОВЕНЬ 3 — B2B переговоры (Nash bargaining)")
    with torch.no_grad():
        out_n = model(
            book=book_ticks["book"], ticks=book_ticks["ticks"],
            fields=fields, field_mask=masks, tokens=tokens,
            workspace_mode="negotiation",
        )
    neg = out_n["workspace"]["negotiation"]
    for i in range(B):
        print(f"  резерв продавца={neg['reservation_a'][i].item():.2f}  "
              f"покупателя={neg['reservation_b'][i].item():.2f}  "
              f"излишек={neg['surplus'][i].item():.2f}")
        print(f"     -> доля продавца={neg['share_a'][i].item():.2f}  "
              f"покупателя={neg['share_b'][i].item():.2f}")

    # ── Data Flywheel ─────────────────────────────────────────────
    banner("DATA FLYWEEL — синтетические стрессы и self-play")
    stress = generate_stress_dataset(n=200, seed=7)
    batch = stress_to_tensors(stress[:5], n_fields=cfg.tabular_features)
    print(f"Сгенерировано стресс-сценариев: {len(stress)}")
    shocks, counts = np.unique([s["shock"] for s in stress],
                               return_counts=True)
    for name, c in zip(shocks, counts):
        print(f"  {name:22s}: {c}")
    print(f"Тензор полей батча: {tuple(batch['fields'].shape)}")

    ep = run_b2b_negotiation(market_price=120, seller_cost=80,
                             buyer_value=150, seed=11)
    print(f"\nB2B эпизод: deal={ep['deal']}, "
          f"seller_pnl={ep['seller_pnl']:.2f}, "
          f"buyer_pnl={ep['buyer_pnl']:.2f}")

    # ── Парсер EDGAR ──────────────────────────────────────────────
    banner("ПАРСЕР EDGAR — извлечение чисел из текста 10-K")
    sample = ("Total net sales $4,521.3 million. Cost of sales $2,100.5. "
              "Gross profit $2,420.8. Operating expenses $1,800.2. "
              "Net income $(120.4).")
    parsed = parse_text_filing(sample)
    for k, v in parsed.items():
        print(f"  {k:15s} = {v:,.2f}")

    banner("ГОТОВО")
    print("NEXUS-Capital успешно прогнал все 4 уровня на CPU.")


if __name__ == "__main__":
    main()
