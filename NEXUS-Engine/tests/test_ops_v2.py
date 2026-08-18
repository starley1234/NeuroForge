import json
import os

import numpy as np
import pytest
import torch


# ------------------------------------------------------------------- BPE
def test_bpe_is_lossless_and_compresses():
    from nexus.data.bpe import BPETokenizer
    from nexus.data.corpora import BUILTIN_ENGINEERING
    from nexus.data.tokenizer import DEFAULT_TOKENIZER

    tok = BPETokenizer().train(BUILTIN_ENGINEERING, vocab_size=1024)
    text = "difference() { cube([10,10,2], center=true); } <task>Нагрузка 300 Н"
    ids = tok.encode(text, bos=True, eos=True)
    assert tok.decode(ids[1:-1]) == text                 # обратимость
    assert len(tok.encode(text)) < len(DEFAULT_TOKENIZER.encode(text))
    assert ids[0] == tok.bos_id and ids[-1] == tok.eos_id


def test_bpe_special_tokens_not_split():
    from nexus.data.bpe import BPETokenizer
    tok = BPETokenizer().train(["a" * 200 + "<scad>" + "b" * 200], vocab_size=400)
    ids = tok.encode("<scad>cube();")
    assert ids[0] == tok.special("<scad>")


def test_bpe_save_load_roundtrip(tmp_path):
    from nexus.data.bpe import BPETokenizer, load_tokenizer
    from nexus.data.corpora import BUILTIN_ENGINEERING
    tok = BPETokenizer().train(BUILTIN_ENGINEERING, vocab_size=600)
    path = str(tmp_path / "bpe.json")
    tok.save(path)
    again = load_tokenizer(path)
    text = "cylinder(h=10, r=3);"
    assert again.encode(text) == tok.encode(text)
    assert again.vocab_size == tok.vocab_size


def test_training_with_custom_tokenizer_records_it(tmp_path):
    from nexus.data.bpe import train_from_source
    from nexus.registry import ModelRegistry
    from nexus.training.train_lm import train
    tok_path = str(tmp_path / "bpe.json")
    train_from_source("builtin:engineering", 512, tok_path, verbose=False)
    root = str(tmp_path / "registry")
    train("builtin:engineering", "core", "tiny", seq_len=64, tokenizer_path=tok_path,
          registry_root=root, max_steps=2, batch_size=2, grad_accum=1, log_every=100)
    mv = ModelRegistry(root).get("core", "latest")
    assert mv.tokenizer == "tokenizer.json" and os.path.exists(mv.tokenizer_path)


# ------------------------------------------------------------------- МКЭ
def test_hex_fem_matches_analytical_tension():
    """Растяжение бруса: σ = F/A, δ = F·L/(E·A)."""
    from nexus.fem.hex_fem import solve_hex_fem
    from nexus.geometry import Cube, voxelize
    vox = voxelize(Cube((10.0, 10.0, 40.0)), resolution=16, padding=0.0)
    r = solve_hex_fem(vox, (0, 0, -500.0), "base", "alu6061")
    area = 10e-3 * 10e-3
    sigma = 500 / area
    delta = 500 * 40e-3 / (68.9e9 * area) * 1000
    assert r.converged
    assert r.mean_stress_pa == pytest.approx(sigma, rel=0.25)
    assert r.max_displacement_mm == pytest.approx(delta, rel=0.35)


def test_hex_fem_finds_stress_concentration_at_hole():
    from nexus.fem.hex_fem import solve_hex_fem
    from nexus.scad import parse_scad
    from nexus.geometry import voxelize
    plain = voxelize(parse_scad("cube([40,40,8], center=true);"), resolution=20, padding=0.0)
    holed = voxelize(parse_scad(
        "difference(){cube([40,40,8],center=true); cylinder(h=20,r=6,center=true);}"),
        resolution=20, padding=0.0)
    a = solve_hex_fem(plain, (0, 0, -400.0), "base", "alu6061")
    b = solve_hex_fem(holed, (0, 0, -400.0), "base", "alu6061")
    assert b.max_stress_pa > a.max_stress_pa          # концентрация у отверстия


def test_hex_fem_stress_scales_linearly():
    from nexus.fem.hex_fem import solve_hex_fem
    from nexus.geometry import Cube, voxelize
    vox = voxelize(Cube((10.0, 10.0, 20.0)), resolution=12, padding=0.0)
    low = solve_hex_fem(vox, (0, 0, -100.0), material="steel304")
    high = solve_hex_fem(vox, (0, 0, -1000.0), material="steel304")
    assert high.max_stress_pa == pytest.approx(10 * low.max_stress_pa, rel=0.05)


def test_solver_backend_selection():
    from nexus.fem.solver import solve
    from nexus.geometry import Cube, voxelize
    vox = voxelize(Cube((10.0, 10.0, 10.0)), resolution=10)
    assert solve(vox, (0, 0, -100.0), backend="hex").backend == "hex"
    assert solve(vox, (0, 0, -100.0), backend="loadpath").backend == "loadpath"
    assert solve(vox, (0, 0, -100.0), backend="auto").backend == "hex"


# ------------------------------------------------------- парсер: for-циклы
def test_parser_supports_for_loops():
    from nexus.geometry import mass_properties, voxelize
    from nexus.scad import parse_scad
    tree = parse_scad("for (i = [0:2:6]) translate([i*10,0,0]) cube([5,5,5]);")
    props = mass_properties(voxelize(tree, resolution=24))
    assert props.volume_mm3 > 0
    single = mass_properties(voxelize(parse_scad("cube([5,5,5]);"), resolution=24))
    assert props.volume_mm3 > 3 * single.volume_mm3 * 0.5

    from nexus.scad import render
    res = render(open("examples/scad/flange.scad").read(), resolution=16)
    assert res.ok and res.mass.mass_g > 0


# ------------------------------------------------ инициализация и обучение
def test_initial_loss_is_near_ln_vocab():
    import math
    import torch.nn.functional as F
    from nexus import NexusConfig, NexusEngine
    cfg = NexusConfig.tiny()
    model = NexusEngine(cfg)
    tokens = torch.randint(4, cfg.vocab_size, (2, 32))
    logits = model(tokens=tokens, reason=False).logits
    loss = float(F.cross_entropy(logits.reshape(-1, cfg.vocab_size), tokens.reshape(-1)).detach())
    assert abs(loss - math.log(cfg.vocab_size)) < 1.5


# ------------------------------------------------------------- quickstart
def test_quickstart_nano_end_to_end(tmp_path):
    from nexus.quickstart import run_quickstart
    res = run_quickstart("nano", "core", str(tmp_path), device="cpu")
    steps = res.steps
    assert steps["flywheel"]["total"] == 8
    assert steps["tokenizer"]["merges"] > 0
    assert steps["training"]["version"] == 1
    assert "checks" in steps["evaluation"]
    assert os.path.exists(os.path.join(str(tmp_path), "quickstart.json"))
    assert os.path.exists(os.path.join(str(tmp_path), "tokenizer", "bpe.json"))


# --------------------------------------------------------- API: защита и SSE
def test_api_requires_key_and_limits_rate(tmp_path):
    import json as js
    import threading
    import urllib.error
    import urllib.request

    from nexus import NexusConfig, NexusEngine
    from nexus.registry import ModelRegistry
    from nexus.serve.app import create_server

    reg = ModelRegistry(str(tmp_path / "registry"))
    model = NexusEngine(NexusConfig.tiny())
    reg.save("core", model.state_dict(), model.cfg.to_dict())
    server, _ = create_server("127.0.0.1", 0, registry_root=str(tmp_path / "registry"),
                              model="core", ref="latest", api_key="secret", rate_limit=3)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()

    def get(path, key=None):
        req = urllib.request.Request(f"http://127.0.0.1:{port}{path}")
        if key:
            req.add_header("Authorization", f"Bearer {key}")
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as exc:
            return exc.code, b""

    try:
        assert get("/health")[0] == 200                       # health открыт
        assert get("/v1/models")[0] == 401                    # без ключа нельзя
        assert get("/v1/models", "secret")[0] == 200
        code, body = get("/metrics", "secret")
        assert code == 200 and b"nexus_requests_total" in body
        codes = [get("/v1/models", "secret")[0] for _ in range(6)]
        assert 429 in codes                                   # rate-limit сработал
    finally:
        server.shutdown()
        server.server_close()


def test_service_stream_and_batch(tmp_path):
    from nexus import NexusConfig, NexusEngine
    from nexus.registry import ModelRegistry
    from nexus.serve.service import InferenceService
    reg = ModelRegistry(str(tmp_path / "registry"))
    model = NexusEngine(NexusConfig.tiny())
    reg.save("core", model.state_dict(), model.cfg.to_dict())
    svc = InferenceService(str(tmp_path / "registry"), "core", "latest")

    chunks = list(svc.generate_stream("<task>тест", max_new_tokens=4))
    assert len(chunks) == 5 and chunks[-1]["done"] and "full_text" in chunks[-1]

    batch = svc.generate_batch(["<task>a", "<task>bb"], max_new_tokens=4)
    assert batch["count"] == 2 and len(batch["texts"]) == 2

    metrics = svc.metrics()
    assert "nexus_generation_latency_ms" in metrics and "nexus_models_loaded" in metrics


# ─────────────────────────────────────────────────── выбор устройства и GPU
def test_pick_device_prefers_explicit_and_falls_back():
    from nexus.runtime import pick_device
    assert pick_device("cpu") == "cpu"
    assert pick_device("cuda:1") == "cuda:1"
    assert pick_device("auto") in ("cpu", "cuda", "mps")


def test_gpu_report_has_diagnosis_fields():
    from nexus.runtime import gpu_report
    rep = gpu_report()
    assert {"torch", "cuda_available", "device", "problem", "fix"} <= set(rep)
    assert isinstance(rep["cuda_available"], bool)


def test_fix_command_matches_gpu_generation():
    from nexus.runtime import _fix_command
    assert "cu128" in _fix_command({"name": "NVIDIA GeForce RTX 5060 Ti"})
    assert "cu126" in _fix_command({"name": "NVIDIA GeForce RTX 4070"})
    assert "--index-url" in _fix_command({"name": "RTX 5090"})


def test_trainer_resolves_auto_device(tmp_path):
    from nexus.config import NexusConfig
    from nexus.data.corpora import PackedLMDataset
    from nexus.model import NexusEngine
    from nexus.training.trainer import TrainConfig, Trainer
    ds = PackedLMDataset("builtin:engineering", seq_len=32, min_blocks=4)
    cfg = TrainConfig(device="auto", max_steps=1, batch_size=1, grad_accum=1,
                      log_every=100, registry_root=str(tmp_path))
    tr = Trainer(NexusEngine(NexusConfig.tiny()), cfg, ds)
    assert tr.cfg.device in ("cpu", "cuda", "mps")


def test_small_preset_fits_and_is_trainable():
    """Пресет small (~200M) — первый осмысленный размер для 16 ГБ."""
    from nexus.config import NexusConfig
    from nexus.eval.vram import estimate
    cfg = NexusConfig.small()
    assert cfg.d_latent == 768 and cfg.n_layers == 12
    budget = estimate(cfg).to_dict()
    assert budget["fits_16gb"] and budget["total_gb"] < 4.0


def test_quickstart_overrides_scale_defaults(tmp_path):
    from nexus.quickstart import run_quickstart
    res = run_quickstart("nano", "core", str(tmp_path), device="cpu",
                         samples=4, steps=3, vocab=600, seq_len=64, math_samples=20)
    assert res.steps["flywheel"]["total"] == 4
    assert res.steps["training"]["metrics"]["steps"] <= 3
    assert res.steps["tokenizer"]["vocab_size"] <= 620


# ─────────────────────────────────── смешанная точность и устройства (GPU-путь)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_model_is_finite_under_autocast(dtype):
    """Под fp16/bf16 автокастом ни forward, ни градиенты не должны давать NaN."""
    import torch.nn.functional as F
    from nexus.config import NexusConfig
    from nexus.model import NexusEngine
    torch.manual_seed(0)
    cfg = NexusConfig.tiny()
    model = NexusEngine(cfg)
    tokens = torch.randint(4, cfg.vocab_size, (2, 64))
    with torch.autocast("cpu", dtype=dtype):
        out = model(tokens=tokens, reason=True)
        loss = F.cross_entropy(out.logits.reshape(-1, cfg.vocab_size).float(),
                               tokens.reshape(-1))
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_ttt_runs_in_fp32_under_autocast():
    from nexus.config import TTTConfig
    from nexus.layers.ttt import FastWeightMemory
    mem = FastWeightMemory(32, TTTConfig(chunk_size=8, key_dim=16, value_dim=16, n_heads=2))
    x = torch.randn(1, 24, 32)
    with torch.autocast("cpu", dtype=torch.float16):
        y, state = mem(x, return_state=True)
    assert torch.isfinite(y).all() and torch.isfinite(state.memory).all()
    assert state.memory.dtype == torch.float32        # состояние всегда в fp32


def test_moe_keeps_single_dtype_under_autocast():
    from nexus.config import NexusConfig
    from nexus.layers.moe import SparseMoE
    moe = SparseMoE(48, NexusConfig.tiny().moe)
    x = torch.randn(2, 8, 48)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        y, stats = moe(x)
    assert y.shape == x.shape and torch.isfinite(y).all()


def test_suite_uses_model_device():
    """Тензоры приёмки создаются на устройстве модели (иначе падало на CUDA)."""
    from nexus.config import NexusConfig
    from nexus.eval.suite import SuiteThresholds, model_device, run_suite
    from nexus.model import NexusEngine
    model = NexusEngine(NexusConfig.tiny())
    assert model_device(model) == next(model.parameters()).device
    rep = run_suite(model, "core", 1, seq_len=64, n_scad_samples=1,
                    thresholds=SuiteThresholds(max_val_ppl=1e9, min_unique_ratio=0.0))
    assert rep.passed


def test_trainer_survives_nan_loss(tmp_path):
    """NaN-шаги пропускаются, а серия NaN останавливает обучение с понятной ошибкой."""
    from nexus.config import NexusConfig
    from nexus.data.corpora import PackedLMDataset
    from nexus.model import NexusEngine
    from nexus.training.trainer import TrainConfig, Trainer

    ds = PackedLMDataset("builtin:engineering", seq_len=32, min_blocks=4)
    cfg = TrainConfig(max_steps=6, batch_size=1, grad_accum=1, log_every=100,
                      max_nan_steps=3, registry_root=str(tmp_path))
    model = NexusEngine(NexusConfig.tiny())

    def broken_loss(m, batch):
        return {"loss": torch.tensor(float("nan"), requires_grad=True)}

    tr = Trainer(model, cfg, ds, loss_fn=broken_loss)
    with pytest.raises(RuntimeError, match="NaN"):
        tr.fit()


def test_amp_dtype_resolution():
    from nexus.training.trainer import _resolve_amp_dtype
    assert _resolve_amp_dtype("fp16", "cuda") is torch.float16
    assert _resolve_amp_dtype("bf16", "cuda") is torch.bfloat16
    assert _resolve_amp_dtype("auto", "cpu") is torch.float16


# ────────────────────────────────── инкрементальная генерация и подсказки
def test_incremental_generation_matches_full_recompute():
    """Кэш состояния не меняет результат жадной генерации (chunk_size=1)."""
    from nexus.config import NexusConfig, TTTConfig
    from nexus.model import NexusEngine
    cfg = NexusConfig.tiny()
    cfg.ttt = TTTConfig(chunk_size=1, key_dim=16, value_dim=16, n_heads=2)
    torch.manual_seed(0)
    model = NexusEngine(cfg).eval()
    ids = torch.randint(4, cfg.vocab_size, (1, 8))
    a = model.generate(ids.clone(), 10, temperature=0.0, use_cache=False)
    b = model.generate(ids.clone(), 10, temperature=0.0, use_cache=True)
    assert torch.equal(a, b)


def test_generation_stops_at_eos():
    from nexus.config import NexusConfig
    from nexus.model import NexusEngine
    cfg = NexusConfig.tiny()
    model = NexusEngine(cfg).eval()
    ids = torch.randint(4, cfg.vocab_size, (1, 4))
    first = int(model.generate(ids.clone(), 1, temperature=0.0)[0, -1])

    stopped = model.generate(ids.clone(), 32, temperature=0.0, eos_id=first)
    assert stopped.shape[1] == ids.shape[1] + 1          # остановились на первом же токене
    full = model.generate(ids.clone(), 32, temperature=0.0)
    assert full.shape[1] == ids.shape[1] + 32            # без eos генерируем всё


def test_stream_yields_tokens_incrementally():
    from nexus.config import NexusConfig
    from nexus.model import NexusEngine
    cfg = NexusConfig.tiny()
    model = NexusEngine(cfg).eval()
    ids = torch.randint(4, cfg.vocab_size, (1, 6))
    tokens = list(model.stream(ids, max_new_tokens=5, temperature=0.7))
    assert len(tokens) == 5 and all(t.shape == (1, 1) for t in tokens)


def test_service_reports_quality_hint_for_undertrained_model(tmp_path):
    from nexus.config import NexusConfig
    from nexus.model import NexusEngine
    from nexus.registry import ModelRegistry
    from nexus.serve.service import InferenceService
    reg = ModelRegistry(str(tmp_path / "reg"))
    model = NexusEngine(NexusConfig.tiny())
    reg.save("core", model.state_dict(), model.cfg.to_dict(),
             metrics={"val_ppl": 480.0, "steps": 50})
    svc = InferenceService(str(tmp_path / "reg"), "core", "latest")
    out = svc.generate("<task>тест", max_new_tokens=4)
    assert out["quality_hint"] and "недообучена" in out["quality_hint"]

    reg.save("core", model.state_dict(), model.cfg.to_dict(),
             metrics={"val_ppl": 12.0, "steps": 20000})
    svc2 = InferenceService(str(tmp_path / "reg"), "core", "latest")
    assert svc2.generate("<task>тест", max_new_tokens=4)["quality_hint"] is None
