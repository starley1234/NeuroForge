import json
import os

import torch

from nexus.config import NexusConfig
from nexus.data.flywheel import run as flywheel_run
from nexus.data.tokenizer import DEFAULT_TOKENIZER
from nexus.encoders import (AudioSSMEncoder, BRepGNOEncoder, BiophysicsEncoder,
                            EventODEEncoder, PointSSMEncoder)
from nexus.eval.vram import estimate
from nexus.scad.render import render
from nexus.training.rewards import score_scad


def test_tokenizer_is_lossless():
    text = "difference() { cube([10,10,2], center=true); cylinder(h=5, r=2); }\n<task>Нагрузка 300 Н"
    ids = DEFAULT_TOKENIZER.encode(text)
    assert DEFAULT_TOKENIZER.decode(ids) == text


def test_reward_prefers_valid_part():
    good = "difference(){ cube([30,30,6], center=true); cylinder(h=20, r=3, center=true); }"
    bad = "cube([30,30,0.2], center=true);"
    r_good = score_scad(good, (0, 0, -150.0), resolution=16, fem_iterations=60)
    r_bad = score_scad(bad, (0, 0, -150.0), resolution=16, fem_iterations=60)
    assert r_good.compile == 1.0
    assert r_good.total > r_bad.total


def test_reward_penalizes_broken_code():
    r = score_scad("difference( cube([1,1,1)", resolution=8)
    assert r.total < 0


def test_flywheel_end_to_end(tmp_path):
    out = str(tmp_path / "fly")
    stats = flywheel_run(4, out, seed=2, grid=12, fem_grid=8, verbose=False)
    assert stats.total == 4
    assert os.path.exists(os.path.join(out, "dataset.jsonl"))
    assert os.path.exists(os.path.join(out, "fields.npz"))

    from nexus.data.dataset import FieldDataset, ScadCorpus
    fields = FieldDataset(out)
    assert len(fields) == 4
    occ, load, sigma, scal = fields[0]
    assert occ.shape == (1, 8, 8, 8) and sigma.shape == (1, 8, 8, 8)

    corpus = ScadCorpus(out, seq_len=128, only_valid=False)
    assert len(corpus) == 4
    assert corpus[0]["tokens"].shape == (128,)


def test_pretrain_and_fno_smoke(tmp_path):
    data = str(tmp_path / "fly")
    flywheel_run(4, data, seed=4, grid=12, fem_grid=8, verbose=False)

    from nexus.training.pretrain import train as pretrain
    ckpt = str(tmp_path / "core.pt")
    cfg = NexusConfig.tiny()
    stats = pretrain(data, ckpt, cfg, epochs=1, batch_size=2, seq_len=64,
                     grad_accum=1, max_steps=2, log_every=100)
    assert os.path.exists(ckpt) and stats["lm"] > 0

    from nexus.config import FNOConfig
    from nexus.training.train_fno import train as train_fno
    fno_ckpt = str(tmp_path / "fno.pt")
    hist = train_fno(data, fno_ckpt, FNOConfig(grid=8, modes=3, width=8, depth=2),
                     epochs=2, batch_size=2, log_every=100)
    assert os.path.exists(fno_ckpt) and "val_mse" in hist


def test_grpo_step_runs(tmp_path):
    from nexus.training.grpo import GRPOConfig, grpo_step
    from nexus.model import NexusEngine
    cfg = NexusConfig.tiny()
    model = NexusEngine(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
    stats = grpo_step(model, None, "<task>кронштейн 200 Н",
                      GRPOConfig(group_size=2, max_new_tokens=8), opt,
                      reward_kwargs={"resolution": 8, "fem_iterations": 20})
    assert "reward_mean" in stats and "compile_rate" in stats


def test_encoders_emit_packets():
    d = 32
    brep = BRepGNOEncoder(d, width=32, layers=2)
    res = render("cube([10,10,4], center=true);", resolution=10)
    nodes, adj, mask = brep.encode_graphs([res.graph])
    p = brep(nodes, adj, mask)
    assert p.features.shape[-1] == d

    assert AudioSSMEncoder(d, width=16)(torch.randn(1, 1600)).features.shape[-1] == d
    assert PointSSMEncoder(d, width=16)(torch.randn(1, 8, 10)).features.shape[-1] == d
    ts = torch.cumsum(torch.rand(1, 8) * 1e-3, dim=1)
    assert EventODEEncoder(d, width=16)(torch.rand(1, 8, 3), ts).features.shape[-1] == d
    assert BiophysicsEncoder(d, channels=4, width=16)(torch.randn(1, 8, 4), ts).features.shape[-1] == d


def test_vram_budget_fits_16gb():
    budget = estimate(NexusConfig.rtx5060()).to_dict()
    assert budget["fits_16gb"] is True
    assert budget["total_gb"] < 16.0


def test_needle_state_does_not_grow():
    from nexus.eval.needle import run
    res = run([256, 1024], NexusConfig.tiny())
    assert res[0].state_bytes == res[1].state_bytes
    assert res[1].kv_bytes_equivalent > res[0].kv_bytes_equivalent


def test_presets_hit_target_budget():
    tight = estimate(NexusConfig.rtx5060()).to_dict()
    compact = estimate(NexusConfig.rtx5060_compact()).to_dict()
    assert 10.0 < tight["total_gb"] < 16.0
    assert compact["total_gb"] < tight["total_gb"]
    assert compact["fits_16gb"] and tight["fits_16gb"]


def test_freeze_core_keeps_new_encoder_trainable():
    from nexus.model import NexusEngine
    cfg = NexusConfig.tiny()
    model = NexusEngine(cfg).freeze_core()
    enc = AudioSSMEncoder(cfg.d_latent, width=16)
    model.register_encoder("audio", enc)
    assert all(not p.requires_grad for p in model.blocks.parameters())
    assert all(p.requires_grad for p in enc.parameters())
