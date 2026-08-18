from __future__ import annotations

import math

import torch

from aura_micro.config import AuraConfig
from aura_micro.export import memory_report
from aura_micro.frontend import HardwareDSPFrontEnd
from aura_micro.losses import multitask_loss
from aura_micro.model import AuraMicro
from aura_micro.physics import itd_seconds, mic_positions
from aura_micro.pipeline import AuraPipeline
from aura_micro.synth import SyntheticAuraDataset, render_scene


def test_mic_triangle():
    cfg = AuraConfig()
    m = mic_positions(cfg)
    d01 = ((m[0] - m[1]) ** 2).sum() ** 0.5
    d12 = ((m[1] - m[2]) ** 2).sum() ** 0.5
    d20 = ((m[2] - m[0]) ** 2).sum() ** 0.5
    assert abs(d01 - cfg.mic_baseline_m) < 1e-9
    assert abs(d12 - cfg.mic_baseline_m) < 1e-6
    assert abs(d20 - cfg.mic_baseline_m) < 1e-6


def test_itd_sign_left_right():
    cfg = AuraConfig()
    left = itd_seconds(__import__("numpy").array([-1.0, 0.0]), cfg)
    right = itd_seconds(__import__("numpy").array([2.0, 0.0]), cfg)
    # ITD12 = tau(M1)-tau(M2). Source left of M1: closer to M1 => tau1 < tau2 => ITD12 < 0
    assert left[0] < 0
    assert right[0] > 0


def test_frontend_shape():
    cfg = AuraConfig()
    dsp = HardwareDSPFrontEnd(cfg)
    wav = torch.randn(2, 3, int(cfg.sample_rate * 0.32))
    feat = dsp(wav)
    assert feat.shape[0] == 2
    assert feat.shape[2] == cfg.feat_dim
    assert feat.shape[1] > 8


def test_model_budget_and_forward():
    cfg = AuraConfig()
    net = AuraMicro(cfg)
    rep = memory_report(net)
    assert rep["int8_flash_kb"] <= 500.0
    assert net.cfg.hidden <= 128
    x = torch.randn(3, 24, cfg.feat_dim)
    out = net(x)
    assert out["logits"].shape == (3, cfg.n_classes)
    assert out["modifiers"].shape == (3, cfg.n_modifiers)
    assert out["state"].shape == (3, cfg.hidden)


def test_state_is_tiny():
    cfg = AuraConfig()
    assert cfg.hidden == 64  # 64 B INT8 cyclic state


def test_pipeline_and_loss():
    cfg = AuraConfig()
    pipe = AuraPipeline(cfg)
    ds = SyntheticAuraDataset(cfg, size=4, duration_s=0.3, seed=1)
    batch = {k: torch.stack([ds[i][k] for i in range(4)]) for k in ds[0]}
    out = pipe(batch["wav"])
    losses = multitask_loss(out, batch)
    assert torch.isfinite(losses["total"])
    decoded = pipe.decode(out)
    assert len(decoded) == 4
    assert "class" in decoded[0]


def test_synth_three_channels():
    wav, lab = render_scene(AuraConfig(), 0.2)
    assert wav.shape[0] == 3
    assert wav.shape[1] > 1000
    assert 0 <= lab.class_id < 30


def test_esc50_mapping():
    from aura_micro.catalog import map_esc50

    assert map_esc50("siren") is not None
    assert map_esc50("glass_breaking") is not None
    assert map_esc50("pig") is None


def test_cli_info():
    from aura_micro.cli import main

    main(["info"])
