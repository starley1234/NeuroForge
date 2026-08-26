"""Тесты мультиязычного токенизатора (RU+EN) и текстовых модулей."""
import os
import numpy as np
import pytest
import torch

# Принудительно оффлайн, чтобы тесты не зависели от доступа к HF Hub
os.environ["NEXUS_OFFLINE_TOKENIZER"] = "1"

from nexus_capital.core.tokenizer import (  # noqa: E402
    NexusTokenizer, _parse_number, NUM, PAD, BOS, EOS,
)


@pytest.fixture(scope="module")
def tokenizer():
    return NexusTokenizer.offline()


def test_parse_number_variants():
    assert _parse_number("1,234.56") == pytest.approx(1234.56)
    assert _parse_number("1 234,56") == pytest.approx(1234.56, abs=0.01)
    assert _parse_number("$5.2M") == pytest.approx(5_200_000.0)
    assert _parse_number("1.5B") == pytest.approx(1_500_000_000.0)
    assert _parse_number("20%") == pytest.approx(20.0)
    assert _parse_number("(120.4)") == pytest.approx(-120.4)
    assert _parse_number("\u221210") == pytest.approx(-10.0)
    assert _parse_number("нет числа") == 0.0


def test_tokenizer_bilingual_encoding(tokenizer):
    en = "Revenue grew 20 percent in Q1."
    ru = "Выручка выросла на 20 процентов в первом квартале."
    e_en = tokenizer.encode(en)
    e_ru = tokenizer.encode(ru)
    assert len(e_en.ids) > 2
    assert len(e_ru.ids) > 2
    assert BOS in [tokenizer.id_to_token(i) for i in e_en.ids[:1]]
    # Число 20 извлечено в обоих языках
    assert any(e_en.number_mask)
    assert any(e_ru.number_mask)
    assert 20.0 in [round(v) for v in e_en.number_values if v != 0.0]
    assert 20.0 in [round(v) for v in e_ru.number_values if v != 0.0]


def test_number_replaced_with_special_token(tokenizer):
    text = "Net income $1,234.56 million."
    enc = tokenizer.encode(text)
    num_positions = [i for i, m in enumerate(enc.number_mask) if m]
    assert len(num_positions) >= 1
    for i in num_positions:
        assert enc.ids[i] == tokenizer.num_id


def test_encode_batch_padding(tokenizer):
    texts = ["Short text.", "A somewhat longer financial sentence about revenue."]
    batch = tokenizer.encode_batch(texts, max_length=32)
    assert batch["input_ids"].shape[0] == 2
    assert batch["input_ids"].shape[1] <= 32
    # Паддинг заполнен pad_id
    assert (batch["attention_mask"].sum(dim=1) > 0).all()


def test_tokenizer_roundtrip_decodable(tokenizer):
    text = "Gross profit equals revenue minus cost of goods sold."
    enc = tokenizer.encode(text)
    decoded = tokenizer.decode(enc.ids, skip_special_tokens=True)
    assert isinstance(decoded, str)
    assert len(decoded) > 0


def test_default_factory_offline(monkeypatch):
    monkeypatch.setenv("NEXUS_OFFLINE_TOKENIZER", "1")
    tok = NexusTokenizer.default()
    assert tok.name.startswith("offline") or "xlm" in tok.name
    assert tok.vocab_size > 100


def test_text_encoder_with_pretrained_vocab(tokenizer):
    from nexus_capital.encoders.text import FinancialTextEncoder
    enc = FinancialTextEncoder(
        d_value=64, vocab_size=tokenizer.vocab_size,
        d_embed=32, max_len=128, n_heads=4, n_layers=1,
        pad_id=tokenizer.pad_id, num_id=tokenizer.num_id,
    )
    batch = tokenizer.encode_batch(
        ["Revenue rose 12 percent.", "Выручка выросла на 12 процентов."],
        max_length=32,
    )
    # pooled
    pooled = enc(
        batch["input_ids"],
        number_values=batch["number_values"],
        number_mask=batch["number_mask"],
        attention_mask=batch["attention_mask"].bool(),
    )
    assert pooled.shape == (2, 64)
    assert torch.isfinite(pooled).all()
    # sequence
    seq = enc(
        batch["input_ids"],
        number_values=batch["number_values"],
        number_mask=batch["number_mask"],
        attention_mask=batch["attention_mask"].bool(),
        return_sequence=True,
    )
    assert seq.shape[0] == 2 and seq.shape[-1] == 64


def test_model_text_forward_uses_tokenizer():
    from nexus_capital import small_config, build_model
    cfg = small_config()
    cfg.mc_paths = 8
    cfg.mc_horizon = 2
    tokenizer = NexusTokenizer.offline()
    model = build_model(cfg, tokenizer=tokenizer)
    assert model.cfg.vocab_size == tokenizer.vocab_size
    batch = tokenizer.encode_batch(
        ["Assets equal liabilities plus equity.",
         "Активы равны обязательствам плюс капитал."],
        max_length=32,
    )
    out = model.text_forward(
        input_ids=batch["input_ids"],
        labels=batch["input_ids"],
        attention_mask=batch["attention_mask"].bool(),
        number_values=batch["number_values"],
        number_mask=batch["number_mask"],
        add_economic_loss=False,
    )
    assert out["logits"].shape[-1] == tokenizer.vocab_size
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert all(torch.isfinite(g).all() for g in grads)


def test_corpus_bilingual_loaded():
    from nexus_capital.data.real.corpus import load_corpus, CorpusConfig
    cfg = CorpusConfig(use_sec=False, use_cbr=False)
    corpus = load_corpus(cfg, languages=("en", "ru"))
    assert "en" in corpus and "ru" in corpus
    assert len(corpus["en"]) > 0
    assert len(corpus["ru"]) > 0
    # В русском корпусе есть кириллица
    assert any(any("а" <= ch <= "я" for ch in s.lower())
               for s in corpus["ru"])
    # В английском — латиница
    assert any(any("a" <= ch <= "z" for ch in s.lower())
               for s in corpus["en"])


def test_financial_text_dataset():
    from nexus_capital.data.real.text_dataset import (
        FinancialTextDataset, TextDataConfig, collate_text,
    )
    from nexus_capital.data.real.corpus import CorpusConfig
    tok = NexusTokenizer.offline()
    cfg = TextDataConfig(
        max_length=48,
        corpus=CorpusConfig(use_sec=False, use_cbr=False),
        languages=("en", "ru"),
    )
    ds = FinancialTextDataset(tok, cfg, split="train")
    assert len(ds) > 0
    s = ds[0]
    assert s["input_ids"].shape == s["labels"].shape
    assert s["attention_mask"].shape[0] == s["input_ids"].shape[0]
    batch = collate_text([ds[i] for i in range(min(4, len(ds)))], tok.pad_id)
    assert batch["input_ids"].shape[0] >= 1
    assert batch["lang"].shape[0] == batch["input_ids"].shape[0]
