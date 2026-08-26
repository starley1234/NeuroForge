"""Интеграционный тест: полная модель Hyperion."""

import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hyperion.config import HyperionConfig
from hyperion.model import HyperionModel


def build_model() -> HyperionModel:
    cfg = HyperionConfig(
        dim=256,
        n_layers=4,
        vocab_size=2000,
        mla_heads=4,
        mla_kv_lora_rank=48,
        mla_rope_head_dim=16,
        mamba_state_dim=32,
        mamba_conv_kernel=4,
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
        dropout=0.0,
        max_seq_len=512,
    )
    return HyperionModel(cfg)


def test_forward_loss():
    torch.manual_seed(0)
    model = build_model()
    B, S = 2, 32
    ids = torch.randint(0, model.vocab_size, (B, S))
    labels = ids.clone()
    labels[:, -1] = -100  # последний токен не обучаем

    out = model(ids, labels=labels)
    assert out["loss"].ndim == 0, "loss не скаляр"
    assert torch.isfinite(out["loss"]), f"loss не конечен: {out['loss']}"
    assert out["logits"].shape == (B, S, model.vocab_size)
    assert out["loss_jepa"] is not None and torch.isfinite(out["loss_jepa"])
    assert out["loss_mtp"] is not None and torch.isfinite(out["loss_mtp"])
    print(f"forward+loss OK: main={out['loss_main'].item():.3f}, "
          f"jepa={out['loss_jepa'].item():.3f}, mtp={out['loss_mtp'].item():.3f}")


def test_backward():
    torch.manual_seed(0)
    model = build_model()
    B, S = 2, 16
    ids = torch.randint(0, model.vocab_size, (B, S))
    out = model(ids, labels=ids.clone())
    out["loss"].backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0, "нет градиентов"
    norm = sum(g.norm().item() ** 2 for g in grads) ** 0.5
    assert torch.isfinite(torch.tensor(norm)), f"grad norm не конечен: {norm}"
    print(f"backward OK (grad norm={norm:.3f})")


def test_generate():
    torch.manual_seed(0)
    model = build_model()
    model.eval()
    B, S = 1, 8
    ids = torch.randint(0, model.vocab_size, (B, S))
    with torch.no_grad():
        generated = model.generate(ids, max_new_tokens=10, temperature=0.8)
    assert generated.shape == (B, S + 10), f"gen shape: {generated.shape}"
    assert (generated[:, S:] >= 0).all() and (generated[:, S:] < model.vocab_size).all()
    print(f"generate OK -> {generated.shape}")


def test_memory_persistence():
    """Память Titans должна сохраняться между вызовами."""
    torch.manual_seed(0)
    model = build_model()
    model.eval()
    B, S = 1, 16
    ids = torch.randint(0, model.vocab_size, (B, S))
    with torch.no_grad():
        h1 = model(ids, use_jepa=False, use_mtp=False, memory_write=True)["hidden"]
        h2 = model(ids, use_jepa=False, use_mtp=False, memory_write=True)["hidden"]
    assert not torch.allclose(h1, h2, atol=1e-3), "память не обновляется между вызовами"
    print("memory persistence OK")


def test_parameter_counts():
    model = build_model()
    counts = model.count_parameters()
    total_m = counts["total"] / 1e6
    active_m = counts["active_inference_est"] / 1e6
    print(f"Параметры: total={total_m:.1f}M, активные (инференс)={active_m:.1f}M")
    assert counts["active_inference_est"] < counts["total"]
    assert total_m < 50, f"слишком большая модель для теста: {total_m}M"


if __name__ == "__main__":
    test_forward_loss()
    test_backward()
    test_generate()
    test_memory_persistence()
    test_parameter_counts()
    print("\nИнтеграционный тест пройден ✓")
