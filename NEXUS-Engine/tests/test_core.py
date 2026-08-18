import torch

from nexus import LatentPacket, NexusConfig, NexusEngine
from nexus.config import TTTConfig
from nexus.layers import FastWeightMemory, SparseMoE, SlidingWindowAttention
from nexus.reasoning import LatentReasoner


def tiny():
    return NexusConfig.tiny()


def test_ttt_state_is_constant_size():
    cfg = TTTConfig(chunk_size=8, key_dim=8, value_dim=8, n_heads=2)
    mem = FastWeightMemory(32, cfg)
    sizes = []
    for length in (16, 256, 2048):
        _, state = mem(torch.randn(1, length, 32), return_state=True)
        sizes.append(state.bytes)
    assert len(set(sizes)) == 1  # O(1) память


def test_ttt_chunking_equivalent_to_streaming():
    torch.manual_seed(0)
    cfg = TTTConfig(chunk_size=1, key_dim=8, value_dim=8, n_heads=2)
    mem = FastWeightMemory(16, cfg).eval()
    x = torch.randn(1, 12, 16)
    with torch.no_grad():
        full, _ = mem(x, return_state=True)
        state, outs = None, []
        for t in range(x.shape[1]):
            y, state = mem(x[:, t: t + 1], state=state, return_state=True)
            outs.append(y)
    assert torch.allclose(full, torch.cat(outs, dim=1), atol=1e-5)


def test_sliding_window_limits_kv_cache():
    from nexus.config import AttentionConfig
    cfg = AttentionConfig(window=16, n_heads=4, n_kv_heads=2)
    attn = SlidingWindowAttention(32, cfg)
    cache = None
    for _ in range(6):
        _, cache = attn(torch.randn(1, 8, 32), cache, return_cache=True)
    assert cache.k.shape[2] <= cfg.window


def test_moe_routes_sparsely():
    cfg = tiny().moe
    moe = SparseMoE(48, cfg)
    y, stats = moe(torch.randn(2, 10, 48))
    assert y.shape == (2, 10, 48)
    assert stats.aux_loss.requires_grad
    assert float(stats.expert_load.sum()) == 1.0


def test_latent_reasoner_halts_adaptively():
    cfg = tiny().reasoning
    r = LatentReasoner(32, cfg)
    out, trace = r(torch.randn(4, 32))
    assert out.shape == (4, 32)
    assert 1 <= trace.steps <= cfg.max_steps


def test_engine_forward_and_backward():
    cfg = tiny()
    model = NexusEngine(cfg)
    tokens = torch.randint(4, cfg.vocab_size, (2, 24))
    stats = model.loss(tokens, tokens, continuous=False)
    stats["loss"].backward()
    assert torch.isfinite(stats["loss"])
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads


def test_dual_output_shapes():
    cfg = tiny()
    model = NexusEngine(cfg)
    tokens = torch.randint(4, cfg.vocab_size, (1, 16))
    occ = torch.rand(1, 1, cfg.fno.grid, cfg.fno.grid, cfg.fno.grid)
    out = model(tokens=tokens, occupancy=occ, load=torch.randn(1, 3), continuous=True)
    assert out.logits.shape == (1, 16, cfg.vocab_size)
    assert out.actions.shape == (1, 16, cfg.action_dim)
    assert out.field.shape == (1, 1, cfg.field_grid, cfg.field_grid, cfg.field_grid)
    assert out.reasoning.physics_scores  # FNO-критик участвовал в рассуждении


def test_multimodal_bus_orders_by_time():
    cfg = tiny()
    model = NexusEngine(cfg)
    a = LatentPacket(torch.randn(1, 3, cfg.d_latent), "audio",
                     time=torch.tensor([[0.5, 0.6, 0.7]]))
    b = LatentPacket(torch.randn(1, 2, cfg.d_latent), "event",
                     time=torch.tensor([[0.1, 0.2]]))
    fused = model.encode([a, b])
    assert fused.shape == (1, 5, cfg.d_latent)


def test_generation_runs():
    cfg = tiny()
    model = NexusEngine(cfg)
    out = model.generate(torch.randint(4, cfg.vocab_size, (1, 5)), max_new_tokens=4)
    assert out.shape == (1, 9)


def test_novelty_gating_ignores_static_signal():
    """Статический сигнал почти не пишется в память — отказ от «налога на токенизацию»."""
    from nexus.config import TTTConfig
    torch.manual_seed(0)
    mem = FastWeightMemory(32, TTTConfig(chunk_size=8, key_dim=16, value_dim=16, n_heads=2)).eval()
    static = torch.randn(1, 1, 32).repeat(1, 64, 1)
    dynamic = torch.randn(1, 64, 32)
    with torch.no_grad():
        _, s_static = mem(static, return_state=True)
        _, s_dynamic = mem(dynamic, return_state=True)
    assert s_static.memory.norm() < 0.2 * s_dynamic.memory.norm()


def test_needle_recall_survives_long_context():
    from nexus.eval.needle import memory_recall
    assert memory_recall(d_model=64, context_len=4096) > 0.3
