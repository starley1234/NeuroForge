"""Сквозные тесты полной модели NEXUS-Capital."""
import torch
import numpy as np

from nexus_capital import small_config, build_model
from nexus_capital.data.orderbook_stream import collate_book_ticks
from nexus_capital.data.edgar_parser import (
    synthetic_filing, fields_to_tensor, DEFAULT_UNIT_FIELDS,
    parse_text_filing,
)
from nexus_capital.data.self_play import (
    generate_self_play_dataset, run_b2b_negotiation, run_market_episode,
)
from nexus_capital.data.synthetic_market import (
    generate_stress_dataset, stress_to_tensors, apply_shock,
)
from nexus_capital.utility.utility_layer import expected_utility
from nexus_capital.utility.losses import NexusLoss
from nexus_capital.training.grpo import (
    economic_reward, group_relative_advantage, grpo_loss,
)


def _tiny_cfg():
    cfg = small_config()
    cfg.mc_paths = 32
    cfg.mc_horizon = 4
    cfg.tabular_features = 16
    cfg.orderbook_levels = 8
    return cfg


def _batch(cfg, b=2, T=8):
    bt = collate_book_ticks(b, n_levels=cfg.orderbook_levels,
                            T=T, n_features=cfg.tick_features)
    rng = np.random.default_rng(0)
    fields, masks = [], []
    for _ in range(b):
        f, m = fields_to_tensor(synthetic_filing(rng),
                                DEFAULT_UNIT_FIELDS[:cfg.tabular_features])
        fields.append(f)
        masks.append(m)
    return {
        "book": bt["book"], "ticks": bt["ticks"],
        "fields": torch.stack(fields), "field_mask": torch.stack(masks),
        "tokens": torch.randint(0, cfg.vocab_size, (b, T)),
    }


def test_full_model_risk_forward():
    cfg = _tiny_cfg()
    m = build_model(cfg)
    batch = _batch(cfg)
    out = m(**batch)
    assert out["logits"].shape[0] == 2
    assert out["economic"]["price"].shape == (2, 1)
    assert out["economic"]["portfolio_weights"].sum(-1).allclose(
        torch.ones(2, 1), atol=1e-4)
    assert (out["economic"]["default_prob"] >= 0).all()
    assert (out["economic"]["default_prob"] <= 1).all()


def test_full_model_with_loss_and_backward():
    cfg = _tiny_cfg()
    m = build_model(cfg)
    batch = _batch(cfg)
    batch["target_tokens"] = torch.randint(0, cfg.vocab_size, (2, 1))
    out = m(**batch)
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert all(torch.isfinite(g).all() for g in grads)


def test_pricing_mode_outputs_price():
    cfg = _tiny_cfg()
    m = build_model(cfg)
    batch = _batch(cfg)
    batch["workspace_mode"] = "pricing"
    batch["competitor_price"] = torch.tensor([100.0, 90.0])
    out = m(**batch)
    p = out["workspace"]["optimal_price"]
    assert p.shape == (2,)
    assert (p > 0).all()


def test_streaming_constant_state():
    cfg = _tiny_cfg()
    m = build_model(cfg)
    bt = collate_book_ticks(1, n_levels=cfg.orderbook_levels,
                            T=100, n_features=cfg.tick_features)
    res = m.stream_ticks(bt["book"], bt["ticks"], window=20)
    assert len(res) == 5
    m.reset_streaming()


def test_edgar_parser_text():
    text = ("Total revenue $1,234.56 million. Cost of goods sold $500.0. "
            "Net income $(200.0).")
    rec = parse_text_filing(text)
    assert abs(rec["revenue"] - 1234.56) < 0.01
    assert abs(rec["cogs"] - 500.0) < 0.01
    assert rec["net_income"] < 0  # убыток в скобках


def test_synthetic_filing_balances():
    rng = np.random.default_rng(1)
    for _ in range(20):
        f = synthetic_filing(rng)
        assert abs(f["gross_profit"] - (f["revenue"] - f["cogs"])) < 1e-3


def test_stress_dataset():
    data = generate_stress_dataset(n=20)
    batch = stress_to_tensors(data[:4], n_fields=16)
    assert batch["fields"].shape == (4, 16)
    assert batch["pnl"].shape == (4,)


def test_apply_shock_changes_report():
    rng = np.random.default_rng(0)
    base = synthetic_filing(rng)
    stressed = apply_shock(base, "inflation_surge", rng)
    assert stressed["cogs"] > base["cogs"]


def test_self_play_deal_has_welfare():
    ep = run_b2b_negotiation(seed=1)
    if ep["deal"]:
        assert ep["seller_pnl"] + ep["buyer_pnl"] <= ep["welfare"] + 1e-4
    ds = generate_self_play_dataset(20)
    assert len(ds) >= 20


def test_market_episode():
    ep = run_market_episode(n_steps=30, seed=1)
    assert "mm_pnl" in ep and "arb_pnl" in ep


def test_utility_layer():
    returns = torch.randn(1000)
    u = expected_utility(returns, cost=0.5, gamma=1.0)
    assert u.ndim == 0
    # Большее неприятие риска снижает полезность неопределённой лотереи
    u1 = expected_utility(returns, gamma=0.1)
    u2 = expected_utility(returns, gamma=5.0)
    assert u2 < u1


def test_total_loss_components():
    loss_fn = NexusLoss()
    pred = torch.tensor(2.0)
    returns = torch.randn(100)
    total, logs = loss_fn(pred, returns=returns)
    assert torch.isfinite(total)
    assert "utility" in logs and "var" in logs


def test_grpo_reward_and_advantage():
    G = 8
    rewards = economic_reward(
        pnl=torch.randn(G),
        var=torch.rand(G).abs(),
        balance_error=torch.rand(G),
        default=(torch.rand(G) > 0.8).float(),
    )
    assert rewards.shape == (G,)
    adv = group_relative_advantage(rewards, group_size=4)
    assert adv.shape == (G,)
    # Нулевое среднее преимущества внутри группы
    assert abs(adv.reshape(-1, 4).mean().item()) < 1e-5

    old_lp = torch.randn(G)
    new_lp = old_lp + 0.01 * torch.randn(G)
    loss, logs = grpo_loss(new_lp, old_lp, adv)
    assert torch.isfinite(loss)
    assert "kl" in logs
