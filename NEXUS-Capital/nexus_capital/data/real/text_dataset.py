"""
Набор данных для двуязычного (RU+EN) финансового LM-обучения NEXUS-Capital.

Каждый пример — токенизированное предложение на русском или английском.
Цель — next-token prediction на той же позиции (сдвиг на 1 токен), что
соответствует предобучению языковой головы. Числа в тексте сохраняются
отдельно как непрерывные значения и внедряются в латентность через
<num>-механизм FinancialTextEncoder.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch.utils.data import Dataset

from ...core.tokenizer import NexusTokenizer
from .corpus import CorpusConfig, load_corpus, train_val_split


@dataclass
class TextDataConfig:
    max_length: int = 256
    languages: tuple[str, ...] = ("en", "ru")
    corpus: CorpusConfig | None = None
    val_frac: float = 0.05
    seed: int = 42


class FinancialTextDataset(Dataset):
    def __init__(self, tokenizer: NexusTokenizer,
                 cfg: TextDataConfig, split: Literal["train", "val"] = "train"):
        self.tokenizer = tokenizer
        self.cfg = cfg
        ccfg = cfg.corpus or CorpusConfig()
        corpus = load_corpus(ccfg, languages=cfg.languages)
        train_c, val_c = train_val_split(corpus, cfg.val_frac)
        part = train_c if split == "train" else val_c
        self.samples: list[tuple[str, str]] = []
        for lang, sents in part.items():
            for s in sents:
                self.samples.append((lang, s))
        if not self.samples:
            raise RuntimeError("Пустой корпус — нет предложений для обучения")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        lang, text = self.samples[idx]
        enc = self.tokenizer.encode(text, max_length=self.cfg.max_length)
        ids = torch.tensor(enc.ids, dtype=torch.long)
        attn = torch.tensor(enc.attention_mask, dtype=torch.long)
        num_vals = torch.tensor(enc.number_values, dtype=torch.float32)
        num_mask = torch.tensor(enc.number_mask, dtype=torch.bool)
        # next-token targets: сдвиг на 1 (используется в trainer)
        return {
            "input_ids": ids[:-1] if len(ids) > 1 else ids,
            "labels": ids[1:] if len(ids) > 1 else ids,
            "attention_mask": attn[:-1] if len(attn) > 1 else attn,
            "number_values": num_vals[:-1] if len(num_vals) > 1 else num_vals,
            "number_mask": num_mask[:-1] if len(num_mask) > 1 else num_mask,
            "lang": 0 if lang == "en" else 1,
        }


def collate_text(batch: list[dict], pad_id: int) -> dict[str, torch.Tensor]:
    max_len = max(b["input_ids"].shape[0] for b in batch)
    B = len(batch)
    input_ids = torch.full((B, max_len), pad_id, dtype=torch.long)
    labels = torch.full((B, max_len), -100, dtype=torch.long)  # ignore_index
    attn = torch.zeros(B, max_len, dtype=torch.long)
    num_vals = torch.zeros(B, max_len, dtype=torch.float32)
    num_mask = torch.zeros(B, max_len, dtype=torch.bool)
    lang = torch.zeros(B, dtype=torch.long)
    for i, b in enumerate(batch):
        n = b["input_ids"].shape[0]
        input_ids[i, :n] = b["input_ids"]
        labels[i, :n] = b["labels"]
        attn[i, :n] = b["attention_mask"]
        num_vals[i, :n] = b["number_values"]
        num_mask[i, :n] = b["number_mask"]
        lang[i] = b["lang"]
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attn,
        "number_values": num_vals,
        "number_mask": num_mask,
        "lang": lang,
    }
