"""
Tests for ELAST-HOLO v2 core and the autonomous self-training loop.
Run: python -m pytest tests/test_v2.py -v
"""
import os
import time
import torch
import torch.nn as nn

from research.arch_v2_core import (
    ElasticHoloNet,
    ElasticHoloConfig,
    ElasticSignalNet,
)
from fractal_holonet.tokenizer import SimpleProductionTokenizer


def make_model(**kw):
    cfg = dict(vocab_size=64, d_model=32, n_layers=2, d_ff=64, n_read_iters=2)
    cfg.update(kw)
    return ElasticHoloNet(ElasticHoloConfig(**cfg))


def test_stream_equals_sequence():
    torch.manual_seed(0)
    m = make_model().eval()
    x = torch.randint(0, 64, (2, 12))
    with torch.no_grad():
        logits_seq, states_seq = m(x)
        logits_step, states_step = m(x[:, :1])
        for t in range(1, x.size(1)):
            lo, states_step = m(x[:, t : t + 1], states=states_step, use_step=True)
            logits_step = torch.cat([logits_step, lo], dim=1)
    assert torch.allclose(logits_seq, logits_step, atol=1e-4)
    for s1, s2 in zip(states_seq, states_step):
        for a, b in zip(s1, s2):
            assert torch.allclose(a, b, atol=1e-4)


def test_generate_and_save_load(tmp_path):
    torch.manual_seed(1)
    m = make_model(vocab_size=300)
    m.save_pretrained(str(tmp_path))
    m2 = ElasticHoloNet.from_pretrained(str(tmp_path))
    ids = torch.randint(0, 300, (1, 6))
    with torch.no_grad():
        l1, _ = m(ids)
        l2, _ = m2(ids)
    assert torch.allclose(l1, l2)
    gen = m.generate(ids, max_new_tokens=8, temperature=0.8)
    assert gen.size(1) == ids.size(1) + 8


def test_circulant_mixing_mode():
    torch.manual_seed(2)
    m = make_model(use_circulant=True).eval()
    x = torch.randint(0, 64, (1, 5))
    with torch.no_grad():
        logits, _ = m(x)
    assert logits.shape == (1, 5, 64)


def test_signal_net_and_dt():
    torch.manual_seed(0)
    m = ElasticSignalNet(
        ElasticHoloConfig(vocab_size=0, d_model=24, n_layers=1, d_ff=48, n_read_iters=1),
        input_signal_dim=1, output_signal_dim=1,
    ).eval()
    sig = torch.randn(2, 16, 1)
    dt = torch.rand(2, 16) * 0.5 + 0.5
    with torch.no_grad():
        pred, anom, states = m.forward_continuous(sig, dt=dt)
        fc = m.forecast_stream(sig, forecast_steps=6, dt=dt)
    assert pred.shape == (2, 16, 1)
    assert anom.shape == (2, 16, 1)
    assert fc.shape == (2, 6, 1)
    assert len(states) == 1


def test_mqar_quick_learning():
    torch.manual_seed(7)
    from research.benchmarks.mqar_benchmark import make_mqar, query_accuracy

    x, y = make_mqar(320, seq_len=24, num_pairs=2, vocab_size=64, seed=0)
    m = ElasticHoloNet(
        ElasticHoloConfig(
            vocab_size=64, d_model=48, n_layers=2, d_ff=96,
            n_read_iters=3, use_slow_memory=False,
        )
    )
    acc0 = query_accuracy(m, x, y)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=0.01)
    crit = nn.CrossEntropyLoss(reduction="none")
    for _ in range(30):
        m.train()
        for i in range(0, x.size(0), 32):
            xb, yb = x[i : i + 32], y[i : i + 32]
            opt.zero_grad()
            logits, _ = m(xb)
            l = crit(logits.view(-1, logits.size(-1)), yb.view(-1)).view(xb.size(0), -1)
            w = torch.ones_like(l)
            w[:, -1] = 8.0
            loss = (l * w).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
    acc1 = query_accuracy(m, x, y)
    assert acc1 > acc0 + 0.2, f"recall did not improve: {acc0:.3f} -> {acc1:.3f}"
    assert acc1 > 0.25, f"recall too low after training: {acc1:.3f}"


def test_self_train_loop_gate(tmp_path):
    from fractal_holonet.self_train import SelfTrainLoop, SyntheticTeacher

    from fractal_holonet.core import ProductionFractalHoloNet, FractalHoloNetConfig

    torch.manual_seed(3)
    tokenizer = SimpleProductionTokenizer()
    student = ProductionFractalHoloNet(
        FractalHoloNetConfig(vocab_size=300, d_model=64, n_layers=2, d_ff=128)
    )
    loop = SelfTrainLoop(
        student,
        tokenizer,
        teacher_client=SyntheticTeacher(),
        save_dir=str(tmp_path),
        eval_text="Фрактальная голографическая память и линейная сложность O(N). " * 40,
    )
    r = loop.run_round(prompts=["Что такое фазовый резонанс?"], epochs=2, batch_size=2)
    assert r["round"] == 1
    assert r["accepted"] is True  # first round always establishes the baseline
    assert r["loss_before"] > 0 and r["loss_after"] > 0
    assert os.path.exists(os.path.join(str(tmp_path), "config.json"))
    assert os.path.exists(os.path.join(str(tmp_path), "pytorch_model.pt"))


def test_self_train_service_daemon(tmp_path):
    from fractal_holonet.self_train import SelfTrainService
    from fractal_holonet.core import ProductionFractalHoloNet, FractalHoloNetConfig

    # pre-create a small checkpoint so daemon rounds stay fast
    student = ProductionFractalHoloNet(
        FractalHoloNetConfig(vocab_size=300, d_model=64, n_layers=2, d_ff=128)
    )
    student.save_pretrained(str(tmp_path))
    SimpleProductionTokenizer().save(os.path.join(str(tmp_path), "tokenizer.json"))

    svc = SelfTrainService(
        checkpoint_dir=str(tmp_path),
        save_dir=str(tmp_path),
        interval_sec=0.05,
        epochs=1,
        batch_size=2,
        curriculum=False,
    )
    assert svc.start() is True
    assert svc.start() is False  # already running
    time.sleep(1.5)
    svc.stop()
    st = svc.status()
    assert st["running"] is False
    assert st["rounds"] >= 1
    assert os.path.exists(os.path.join(str(tmp_path), "config.json"))
