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
