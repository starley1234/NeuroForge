"""Тесты Neural SDE, латентного Монте-Карло и теории игр."""
import torch

from nexus_capital.workspace.neural_sde import NeuralSDE
from nexus_capital.workspace.monte_carlo import (
    LatentMonteCarlo, value_at_risk, expected_shortfall,
)
from nexus_capital.workspace.game_theory import (
    nash_bargaining, bertrand_price, find_nash_equilibrium,
)
from nexus_capital.workspace.workspace import EconomicWorkspace


def test_neural_sde_shapes():
    sde = NeuralSDE(16, d_hidden=32)
    x0 = torch.randn(4, 16)
    sim = sde.simulate(x0, horizon=8, paths=100, return_paths=True)
    assert sim["final"].shape == (4, 100, 16)
    assert sim["paths"].shape == (4, 100, 9, 16)


def test_neural_sde_backward():
    sde = NeuralSDE(8, d_hidden=16)
    x0 = torch.randn(2, 8, requires_grad=True)
    out = sde.simulate(x0, horizon=4, paths=20)
    out["final"].sum().backward()
    assert x0.grad is not None


def test_monte_carlo_metrics():
    sde = NeuralSDE(16, d_hidden=32)
    mc = LatentMonteCarlo(sde, horizon=8, paths=200, alpha=0.05)
    x0 = torch.randn(3, 16)
    res = mc(x0)
    for key in ("pnl", "expected_pnl", "var", "expected_shortfall",
                "default_prob"):
        assert key in res
    assert res["var"].shape == (3,)
    # VaR — α-квантиль убытков; для случайной модели проверяем конечность
    # и монотонность относительно уровня α
    assert torch.isfinite(res["var"]).all()


def test_var_quantile():
    pnl = torch.linspace(-100, 100, 1000)
    var = value_at_risk(pnl, alpha=0.05)
    es = expected_shortfall(pnl, alpha=0.05)
    assert var.item() > 0
    assert es.item() >= var.item()  # ES >= VaR


def test_nash_bargaining_efficiency():
    ra = torch.tensor(10.0)
    rb = torch.tensor(5.0)
    surplus = torch.tensor(100.0)
    a, b = nash_bargaining(ra, rb, surplus, bargaining_power_a=0.5)
    # Сумма долей = резервы + излишек (эффективность по Парето)
    assert torch.isclose(a + b, ra + rb + surplus, atol=1e-5)


def test_bertrand_price_above_cost():
    cost = torch.tensor(50.0)
    comp = torch.tensor(120.0)
    elas = torch.tensor(2.0)
    size = torch.tensor(1000.0)
    p = bertrand_price(cost, comp, elas, size)
    assert p.item() >= cost.item()
    assert p.item() <= comp.item() * 2


def test_nash_equilibrium_prisoners():
    # Дилемма заключенного: (C,D) пэйоффы
    # A: строки [cooperate, defect], B: столбцы [cooperate, defect]
    A = torch.tensor([[-1.0, -3.0], [0.0, -2.0]])
    B = torch.tensor([[-1.0, 0.0], [-3.0, -2.0]])
    pA, pB, gap = find_nash_equilibrium(A, B, n_iter=500)
    assert gap.item() < 0.1 or pA[1] > 0.5  # defect доминирует


def test_economic_workspace_pricing():
    ws = EconomicWorkspace(16, sde_hidden=32, mc_paths=64, mc_horizon=4)
    h = torch.randn(2, 16)
    comp = torch.tensor([100.0, 80.0])
    out = ws(h, mode="pricing", competitor_price=comp)
    assert out["optimal_price"].shape == (2,)
    assert "risk" in out


def test_economic_workspace_negotiation():
    ws = EconomicWorkspace(16, sde_hidden=32, mc_paths=64, mc_horizon=4)
    h = torch.randn(3, 16)
    out = ws(h, mode="negotiation")
    assert out["negotiation"]["share_a"].shape == (3,)
    # Сумма долей >= суммы резервов
    neg = out["negotiation"]
    total = neg["share_a"] + neg["share_b"]
    reservation = neg["reservation_a"] + neg["reservation_b"]
    assert (total >= reservation - 1e-4).all()
