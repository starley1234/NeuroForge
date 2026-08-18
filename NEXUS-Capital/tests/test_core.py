"""Тесты ядра NEXUS-Capital: TTT, внимание, MoE, Value Bus, utility."""
import pytest
import torch

from nexus_capital.core.config import small_config
from nexus_capital.core.ttt_memory import TTTRegimeMemory
from nexus_capital.core.attention import LocalFinancialAttention
from nexus_capital.core.moe import FinancialSparseMoE
from nexus_capital.core.value_bus import UnifiedValueBus
from nexus_capital.core.layers import continuous_value_embedding, ContinuousProjector


def test_continuous_value_embedding_preserves_scale():
    v = torch.tensor([1.0, 1000.0, 1_000_000.0, -500.0])
    emb = continuous_value_embedding(v, 32)
    assert emb.shape == (4, 32)
    # Нуль даёт детерминированное представление
    z = continuous_value_embedding(torch.tensor(0.0), 16)
    assert z.shape == (16,)
    assert torch.isfinite(emb).all()


def test_projector():
    proj = ContinuousProjector(4, 16)
    x = torch.randn(3, 4)
    assert proj(x).shape == (3, 16)


def test_ttt_memory_updates_state():
    ttt = TTTRegimeMemory(d_value=32, d_mem=16, rank=8, alpha=0.9, eta=0.1)
    x = torch.randn(2, 5, 32)
    out, state = ttt(x)
    assert out.shape == (2, 5, 32)
    assert state.shape == (2, 16, 32)
    # При стриминге состояние должно меняться
    out2, state2 = ttt(x, state=state)
    assert not torch.allclose(state, state2, atol=1e-6)


def test_ttt_memory_constant_memory_long_stream():
    ttt = TTTRegimeMemory(d_value=16, d_mem=8, rank=4)
    x = torch.randn(1, 100, 16)
    out, _ = ttt(x)
    assert out.shape == (1, 100, 16)


def test_local_attention():
    attn = LocalFinancialAttention(32, n_heads=4, window=8)
    x = torch.randn(2, 20, 32)
    assert attn(x).shape == (2, 20, 32)


def test_sparse_moe_routing():
    moe = FinancialSparseMoE(32, n_experts=4, experts_per_token=2, d_ff=64)
    x = torch.randn(2, 6, 32)
    out, aux = moe(x)
    assert out.shape == (2, 6, 32)
    assert aux.ndim == 0
    assert torch.isfinite(aux)


def test_value_bus_invariants():
    cfg = small_config()
    cfg.d_value = 32
    bus = UnifiedValueBus(cfg)
    x = torch.randn(2, 4, 32)
    h, info = bus(x)
    assert h.shape == (2, 4, 32)
    inv = info["invariants"]
    assert inv["sigma"].shape == (2, 4)
    assert (inv["sigma"] >= 0).all()
    assert (inv["discount"] >= 0).all() and (inv["discount"] <= 1).all()


def test_value_bus_backward():
    cfg = small_config()
    cfg.d_value = 32
    bus = UnifiedValueBus(cfg)
    x = torch.randn(2, 3, 32, requires_grad=True)
    h, _ = bus(x)
    h.sum().backward()
    assert x.grad is not None
