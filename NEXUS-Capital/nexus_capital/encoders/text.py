"""
Text/News encoder для новостей (Bloomberg/Reuters), пресс-релизов и
финансовой отчётности на русском и английском.

Токенизация выполняется ПРЕДОБУЧЕННЫМ мультиязычным SentencePiece
(XLM-RoBERTa, ~250k токенов, покрывает RU+EN) — словарь не обучается
заново. Чтобы не раздувать память под 250k × d_value, embedding имеет
собственную размерность d_embed и проецируется в d_value обучаемой
линейной картой.

Числа в тексте детектируются до токенизации (см. core/tokenizer.py),
заменяются спецтокеном <num> и их вещественные значения внедряются как
непрерывные тензоры стоимости — это снимает «слепоту к масштабу чисел».
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from ..core.layers import XValNumberEmbedding


class FinancialTextEncoder(nn.Module):
    def __init__(
        self,
        d_value: int,
        vocab_size: int = 250001,
        d_embed: int = 256,
        max_len: int = 2048,
        n_heads: int = 8,
        n_layers: int = 4,
        pad_id: int = 1,
        num_id: int = 5,
    ):
        super().__init__()
        self.d_value = d_value
        self.pad_id = pad_id
        self.num_id = num_id

        self.d_embed = d_embed
        self.tok_emb = nn.Embedding(vocab_size, d_embed, padding_idx=pad_id)
        self.embed_proj = nn.Sequential(
            nn.Linear(d_embed, d_value, bias=False),
            nn.SiLU(),
            nn.Linear(d_value, d_value, bias=False),
        )

        # xVal: один обучаемый числовой вектор, масштабируемый значением числа
        self.num_xval = XValNumberEmbedding(d_value)

        head_dim = max(1, d_value // n_heads)
        if d_value % n_heads != 0:
            # гарантируем делимость для небольших d_value
            n_heads = max(1, d_value // head_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=d_value, nhead=n_heads,
            dim_feedforward=d_value * 2,
            batch_first=True, activation="gelu", norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

        pe = torch.zeros(max_len, d_value)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_value, 2).float()
                        * (-math.log(10000.0) / d_value))
        pe[:, 0::2] = torch.sin(pos * div[:pe[:, 0::2].shape[1]])
        pe[:, 1::2] = torch.cos(pos * div[:pe[:, 1::2].shape[1]])
        self.register_buffer("pos_emb", pe)

    def forward(
        self,
        tokens: torch.Tensor,
        number_values: torch.Tensor | None = None,
        number_mask: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        return_sequence: bool = False,
    ) -> torch.Tensor:
        """
        tokens:        (B, T) long  — id предобученного XLM-R токенизатора
        number_values: (B, T) float — извлечённые числа (0 где не число)
        number_mask:   (B, T) bool  — позиции <num>
        attention_mask:(B, T) bool  — валидные (не-pad) позиции
        return_sequence: если True, вернуть (B, T, d_value), иначе
                        маскированный mean-pool (B, d_value).
        """
        B, T = tokens.shape
        h = self.embed_proj(self.tok_emb(tokens))

        if number_values is not None and number_mask is not None:
            num_emb = self.num_xval(number_values)  # xVal: x * u
            h = torch.where(number_mask.unsqueeze(-1), num_emb, h)

        h = h + self.pos_emb[:T].unsqueeze(0)
        if attention_mask is None:
            attention_mask = tokens.ne(self.pad_id)
        # TransformerEncoder: key_padding_mask = True для pad-позиций
        h = self.encoder(h, src_key_padding_mask=~attention_mask.bool())
        if return_sequence:
            return h
        h = h * attention_mask.unsqueeze(-1)
        return h.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True).clamp(min=1)
