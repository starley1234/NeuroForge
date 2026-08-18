"""
Тесты токенизаторов (byte + BPE) и реестра обученных моделей.
"""
import json
import os
from pathlib import Path

import pytest
import torch

from fractal_holonet.core import ProductionFractalHoloNet
from fractal_holonet.tokenizer import (
    SimpleProductionTokenizer,
    BpeTokenizer,
    load_tokenizer,
    build_config_for_tokenizer,
)
from fractal_holonet.distillation import DistillationDataset
import fractal_holonet.registry as registry


# ---------------------------------------------------------------------------
# Токенизаторы
# ---------------------------------------------------------------------------
def test_legacy_tokenizer_properties():
    tok = SimpleProductionTokenizer()
    assert tok.vocab_size == 260  # 4 спец + 256 байт
    assert tok.pad_token_id == 0
    assert tok.bos_token_id == 1
    assert tok.eos_token_id == 2
    text = "Привет, мир!"
    ids = tok.encode(text)
    assert tok.decode(ids) == text


def test_bpe_tokenizer_roundtrip_and_specials(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(
        "Искусственный интеллект и голографическая память. " * 100 +
        "Fractal-HoloNet O(N) context. " * 50,
        encoding="utf-8",
    )
    # Внимание: vocab_size включает байтовый алфавит (256) + 4 спец-токена,
    # поэтому merges = vocab_size - 260.
    tok = BpeTokenizer.train_from_files([str(corpus)], vocab_size=512)
    assert 260 < tok.vocab_size <= 512
    text = "Голографическая память O(N) и искусственный интеллект!"
    ids = tok.encode(text, add_bos=True)
    assert ids[0] == tok.bos_token_id
    assert tok.decode(ids, skip_special=True) == text
    # сжатие: кириллица короче, чем по байтам
    assert len(tok.encode(text)) < len(text.encode("utf-8"))
    # сохранение/загрузка
    path = tmp_path / "tokenizer.json"
    tok.save(str(path))
    tok2 = BpeTokenizer.load(str(path))
    assert tok2.encode(text) == tok.encode(text)


def test_load_tokenizer_autodetect(tmp_path):
    # legacy-формат
    legacy_dir = tmp_path / "legacy"
    legacy_dir.mkdir()
    SimpleProductionTokenizer().save(str(legacy_dir / "tokenizer.json"))
    assert isinstance(load_tokenizer(str(legacy_dir)), SimpleProductionTokenizer)

    # HF byte-level BPE
    corpus = tmp_path / "c.txt"
    corpus.write_text(("токен токен токены токенизация " * 50), encoding="utf-8")
    bpe_dir = tmp_path / "bpe"
    bpe_dir.mkdir()
    BpeTokenizer.train_from_files([str(corpus)], vocab_size=512, path=str(bpe_dir / "tokenizer.json"))
    assert isinstance(load_tokenizer(str(bpe_dir)), BpeTokenizer)

    # отсутствует -> legacy по умолчанию
    empty = tmp_path / "empty"
    empty.mkdir()
    assert isinstance(load_tokenizer(str(empty)), SimpleProductionTokenizer)


def test_build_config_from_tokenizer():
    tok = SimpleProductionTokenizer()
    cfg = build_config_for_tokenizer(tok, d_model=64, n_layers=2, d_ff=128)
    assert cfg.vocab_size == tok.vocab_size  # 260, а не жёсткие 300
    assert cfg.pad_token_id == 0 and cfg.eos_token_id == 2
    model = ProductionFractalHoloNet(cfg)
    assert model.token_emb.num_embeddings == 260


def test_distillation_dataset_uses_real_eos_token():
    tok = SimpleProductionTokenizer()
    pairs = [{"prompt": "Что такое фаза?", "response": "Фаза — это угол."}]
    ds = DistillationDataset(pairs, tok, block_size=256)
    x, y = ds[0]
    prefix = tok.encode("User: Что такое фаза?\nAssistant: Фаза — это угол.\n")
    assert x[len(prefix)].item() == tok.eos_token_id  # настоящий спец-токен
    # а не литеральная строка "<eos>" из пяти байтов
    literal = tok.encode("<eos>")
    seq = x.tolist()
    assert not _contains_sublist(seq, literal)


def _contains_sublist(seq, sub):
    for i in range(len(seq) - len(sub) + 1):
        if seq[i : i + len(sub)] == sub:
            return True
    return False


# ---------------------------------------------------------------------------
# Реестр моделей
# ---------------------------------------------------------------------------
@pytest.fixture
def tmp_registry(tmp_path, monkeypatch):
    monkeypatch.setenv("FH_REGISTRY_DIR", str(tmp_path))
    return tmp_path


def _make_tiny_model():
    from fractal_holonet.core import FractalHoloNetConfig

    torch.manual_seed(0)
    tok = SimpleProductionTokenizer()
    cfg = build_config_for_tokenizer(tok, d_model=64, n_layers=2, d_ff=128)
    return ProductionFractalHoloNet(cfg), tok


def test_registry_lifecycle(tmp_registry):
    model, tok = _make_tiny_model()
    mid = registry.create_model(name="test model", tokenizer_type="byte", tokenizer_vocab=260)
    assert registry.get_model(mid)["status"] == "initialized"
    assert mid in [m["id"] for m in registry.list_models()]

    registry.save_model_artifacts(mid, model, tok, status="trained")
    registry.record_metrics(mid, {"val_loss": 0.5, "params": 100})
    meta = registry.get_model(mid)
    assert meta["status"] == "trained"
    assert meta["metrics"]["val_loss"] == 0.5

    # артефакты на месте и модель загружается
    assert (registry.model_dir(mid) / "pytorch_model.pt").exists()
    loaded, loaded_tok = registry.load_model(mid)
    assert isinstance(loaded, ProductionFractalHoloNet)
    assert loaded_tok.vocab_size == 260

    # активация
    registry.activate(mid)
    assert registry.get_active() == mid
    assert any(m["active"] for m in registry.list_models())
    registry.activate(None)
    assert registry.get_active() is None


def test_registry_import_existing(tmp_registry, tmp_path):
    model, tok = _make_tiny_model()
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    model.save_pretrained(str(ckpt))
    tok.save(str(ckpt / "tokenizer.json"))

    mid = registry.import_existing(
        "imported-x", "Imported", str(ckpt), architecture="fractal-holonet-v1",
        metrics={"val_loss": 0.9},
    )
    meta = registry.get_model(mid)
    assert meta["status"] == "imported"
    assert meta["metrics"]["val_loss"] == 0.9
    assert (registry.model_dir(mid) / "pytorch_model.pt").exists()


def test_serve_models_endpoint():
    from fastapi.testclient import TestClient
    from fractal_holonet.serve import app

    client = TestClient(app)
    r = client.get("/v1/models")
    assert r.status_code == 200
    body = r.json()
    assert "models" in body and "active" in body
