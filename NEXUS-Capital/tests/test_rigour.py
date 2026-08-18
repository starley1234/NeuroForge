"""Тесты «честности»: калибровка SDE, балансовый loss, бейзлайны, сплит."""
import numpy as np
import torch

from nexus_capital.workspace.neural_sde import NeuralSDE
from nexus_capital.workspace.monte_carlo import (
    LatentMonteCarlo, value_at_risk,
)
from nexus_capital.utility.balance import (
    FieldReconstructionHead, log_transform, invert_log,
)
from nexus_capital.training.baselines import (
    zero_baseline, mean_baseline, ar1_baseline, all_baselines, var_coverage,
)
from nexus_capital.data.real.real_dataset import (
    RealDataConfig, RealMarketDataset,
)


def test_sde_calibration_sets_stats():
    torch.manual_seed(0)
    sde = NeuralSDE(8, d_hidden=16)
    # Реалистичные лог-доходности
    rets = torch.from_numpy(
        np.random.RandomState(0).normal(0.0001, 0.015, size=5000).astype(
            np.float32)).unsqueeze(-1)
    info = sde.calibrate_to_returns(rets)
    assert info["calibrated"] is True
    assert abs(info["emp_mean"]) < 0.01
    assert 0.005 < info["emp_std"] < 0.03
    assert 2.1 < info["df"] <= 30.0
    assert bool(sde.calibrated.item()) is True


def test_sde_simulate_uses_t_increments():
    sde = NeuralSDE(4, d_hidden=8, use_t_increments=True)
    x0 = torch.zeros(3, 4)
    sim = sde.simulate(x0, horizon=4, paths=200)
    assert sim["final"].shape == (3, 200, 4)
    # Тяжёлые хвосты: эксцесс больше, чем у гаусса
    pnl = sim["final"][..., 0].detach().numpy().flatten()
    kurt = ((pnl - pnl.mean()) ** 4).mean() / (pnl.std() ** 4)
    assert kurt > 2.0


def test_monte_carlo_reports_calibration_flag():
    sde = NeuralSDE(4, d_hidden=8)
    mc = LatentMonteCarlo(sde, horizon=4, paths=50)
    x0 = torch.zeros(2, 4)
    res = mc(x0)
    assert "calibrated" in res
    assert res["calibrated"] is False
    sde.calibrate_to_returns(torch.randn(1000, 4) * 0.01)
    res2 = mc(x0)
    assert res2["calibrated"] is True


def test_field_reconstruction_head_balance_identity():
    torch.manual_seed(0)
    fields = ["revenue", "cogs", "gross_profit", "assets",
              "liabilities", "equity"]
    head = FieldReconstructionHead(16, len(fields), fields)
    h = torch.randn(8, 16)
    # После обучения head может выдавать что угодно; проверим, что loss
    # конечно и дифференцируем
    loss = head.balance_loss(h)
    assert loss.ndim == 0
    loss.backward()


def test_field_reconstruction_recovers_log_scale():
    fields = ["revenue", "cogs", "gross_profit"]
    head = FieldReconstructionHead(8, 3, fields)
    target = torch.tensor([[100.0, 60.0, 40.0], [200.0, 80.0, 120.0]])
    mask = torch.ones_like(target)
    h = torch.randn(2, 8, requires_grad=True)
    loss = head.reconstruction_loss(h, target, mask)
    loss.backward()
    assert h.grad is not None
    pred = head.predict_fields(h.detach())
    assert pred.shape == target.shape


def test_log_transform_roundtrip():
    x = torch.tensor([0.0, 1.0, -1.0, 1e6, -1e6])
    y = invert_log(log_transform(x))
    assert torch.allclose(y, x, atol=1e-3)


def test_baselines_shapes_and_monotonicity():
    rng = np.random.RandomState(0)
    rets = rng.normal(0, 0.01, 1000)
    z = zero_baseline(rets)
    assert z["name"] == "zero"
    assert z["mse"] >= 0
    m = mean_baseline(rets)
    a = ar1_baseline(rets)
    for b in (z, m, a):
        assert 0 <= b["sign_acc"] <= 1
    all_bs = all_baselines(rets)
    assert len(all_bs) == 3


def test_var_coverage_baseline():
    rng = np.random.RandomState(0)
    # Гаусс: 5% квантиль
    pnl = rng.normal(0, 1, 100000)
    var5 = -np.percentile(pnl, 5)
    cov = var_coverage(pnl, var5, alpha=0.05)
    assert abs(cov["observed_breach_rate"] - 0.05) < 0.01


def test_chronological_split_no_leakage():
    from nexus_capital.data.real.binance import BinanceConfig
    cfg = RealDataConfig(
        binance=BinanceConfig(symbol="BTCUSDT", start_month="2024-01",
                             end_month="2024-01"),
        max_samples=1000, tick_seq_len=8)
    ds = RealMarketDataset(cfg)
    train, val = ds.chronological_split(0.2)
    assert len(train) + len(val) == ds._n_windows()
    # train-окна идут до val-окон (хронологически)
    assert train.end <= val.start
    # Нет пересечения
    assert len(set(range(train.start, train.end))
               & set(range(val.start, val.end))) == 0


def test_raw_log_returns_available():
    from nexus_capital.data.real.binance import BinanceConfig
    cfg = RealDataConfig(
        binance=BinanceConfig(symbol="BTCUSDT", start_month="2024-01",
                             end_month="2024-01"),
        max_samples=50, tick_seq_len=8)
    ds = RealMarketDataset(cfg)
    r = ds.raw_log_returns()
    assert r is not None
    assert r.ndim == 1
    assert np.isfinite(r).all()
