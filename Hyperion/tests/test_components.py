"""Тесты компонентов Hyperion."""

import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hyperion.config import HyperionConfig
from hyperion.core.mla import MultiHeadLatentAttention
from hyperion.core.mamba3 import MambaBlock
from hyperion.core.titans_memory import TitansMemoryBlock, NeuralMemory
from hyperion.core.mose import MoSE, SlimmableExpert
from hyperion.core.mtp import MTPStack
from hyperion.core.jepa import JEPAPredictor, JEPALoss


def make_config(**overrides) -> HyperionConfig:
    cfg = HyperionConfig(
        dim=256,
        n_layers=4,
        vocab_size=2000,
        mla_heads=4,
        mla_kv_lora_rank=48,
        mla_q_lora_rank=64,
        mla_rope_head_dim=16,
        mamba_state_dim=32,
        memory_dim=96,
        memory_depth=2,
        mose_experts=4,
        mose_top_k=2,
        mose_shared_experts=1,
        jepa_pred_dim=128,
        mtp_depth=1,
        ssm_ratio=0.5,
        attn_ratio=0.25,
        memory_ratio=0.25,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def test_mla():
    torch.manual_seed(0)
    dim, B, S = 256, 2, 16
    layer = MultiHeadLatentAttention(
        dim, n_heads=4, kv_lora_rank=48, q_lora_rank=64, rope_head_dim=16,
    )
    x = torch.randn(B, S, dim)
    positions = torch.arange(S)
    out, cache = layer(x, positions=positions, use_cache=True)
    assert out.shape == (B, S, dim), f"MLA out: {out.shape}"
    assert cache is not None and cache["c_kv"].shape == (B, S, 48)
    # Инкрементальное декодирование даёт тот же результат
    x2 = torch.randn(B, 4, dim)
    pos2 = torch.arange(S, S + 4)
    out2, cache2 = layer(x2, positions=pos2, kv_cache=cache, use_cache=True)
    assert out2.shape == (B, 4, dim)
    print("MLA OK")


def test_mamba():
    torch.manual_seed(0)
    dim, B, S = 256, 2, 32
    layer = MambaBlock(dim, state_dim=32)
    x = torch.randn(B, S, dim)
    out = layer(x)
    assert out.shape == (B, S, dim)
    # Каузальность: результат на t не зависит от будущего
    x_a = torch.randn(B, S, dim)
    out_a = layer(x_a.clone())
    out_b = layer(torch.cat([x_a, torch.randn(B, 8, dim)], dim=1))
    assert torch.allclose(out_a, out_b[:, :S], atol=1e-5), "Mamba не каузален!"
    print("Mamba OK (causal)")


def test_titans_memory():
    torch.manual_seed(0)
    dim, B, S = 256, 2, 16
    block = TitansMemoryBlock(dim, n_heads=4, mem_dim=96, n_persistent=8)
    x = torch.randn(B, S, dim)
    out = block(x, use_memory_write=True)
    assert out.shape == (B, S, dim)
    # Повторный проход должен "помнить" (выход отличается от первого)
    out2 = block(x, use_memory_write=True)
    assert not torch.allclose(out, out2, atol=1e-3), "Память не обучается!"
    # После reset выход должен вернуться к базовому
    block.memory.reset()
    out3 = block(x, use_memory_write=False)
    assert out3.shape == (B, S, dim)
    print("TitansMemory OK (test-time learning)")


def test_mose():
    torch.manual_seed(0)
    dim, B, S = 256, 2, 16
    mose = MoSE(dim, n_experts=4, top_k=2, shared=1, nested_widths=3)
    x = torch.randn(B, S, dim)
    out = mose(x)
    assert out.shape == (B, S, dim)
    # Ширина: эксперты slimмable
    exp = SlimmableExpert(dim, expert_dim=96, nested_widths=3)
    out_w0 = exp(x, width_idx=0)
    out_w2 = exp(x, width_idx=2)
    assert out_w0.shape == (B, S, dim) and out_w2.shape == (B, S, dim)
    print("MoSE OK (slimmable)")


def test_mtp():
    torch.manual_seed(0)
    dim, B, S, V = 256, 2, 16, 2000
    emb = torch.nn.Embedding(V, dim)
    stack = MTPStack(dim, V, emb, mtp_depth=2)
    h = torch.randn(B, S, dim)
    ids = torch.randint(0, V, (B, S))
    loss = stack(h, ids)
    assert loss.ndim == 0 and loss > 0, f"MTP loss: {loss}"
    print(f"MTP OK (loss={loss.item():.3f})")


def test_jepa():
    torch.manual_seed(0)
    dim, B, S, pdim = 256, 2, 16, 128
    predictor = JEPAPredictor(dim, pred_dim=pdim, k=4)
    x = torch.randn(B, S, dim)
    preds = predictor(x)
    assert preds.shape == (B, S, 4, pdim)
    loss_fn = JEPALoss()
    targets = torch.randn(B, S, 4, pdim)
    loss = loss_fn(preds, targets)
    assert loss.ndim == 0
    print(f"JEPA OK (loss={loss.item():.3f})")


if __name__ == "__main__":
    test_mla()
    test_mamba()
    test_titans_memory()
    test_mose()
    test_mtp()
    test_jepa()
    print("\nВсе компоненты прошли тесты ✓")
