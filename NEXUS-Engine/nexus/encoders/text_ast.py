"""AST-BPE Embedder — дискретные модальности (текст, код, математика, AST)."""
from __future__ import annotations

from typing import List, Optional, Sequence

import torch
import torch.nn as nn

from ..bus import LatentPacket
from ..data.tokenizer import DEFAULT_TOKENIZER, ASTBPETokenizer
from .base import ModalityEncoder


class TextASTEncoder(ModalityEncoder):
    modality = "text"

    def __init__(self, d_latent: int, tokenizer: Optional[ASTBPETokenizer] = None,
                 vocab_size: Optional[int] = None):
        super().__init__(d_latent)
        self.tokenizer = tokenizer or DEFAULT_TOKENIZER
        self.vocab_size = vocab_size or self.tokenizer.vocab_size
        self.embed = nn.Embedding(self.vocab_size, d_latent)
        self.depth_embed = nn.Embedding(32, d_latent)   # глубина AST-узла

    def encode_text(self, texts: Sequence[str], device=None, max_len: Optional[int] = None):
        ids: List[List[int]] = [self.tokenizer.encode(t, bos=True, eos=True) for t in texts]
        if max_len:
            ids = [x[:max_len] for x in ids]
        width = max(len(x) for x in ids)
        pad = self.tokenizer.pad_id
        batch = [x + [pad] * (width - len(x)) for x in ids]
        return torch.tensor(batch, dtype=torch.long, device=device)

    def forward(self, tokens: torch.Tensor, depths: Optional[torch.Tensor] = None,
                modality: Optional[str] = None) -> LatentPacket:
        tokens = tokens.clamp(0, self.vocab_size - 1)
        x = self.embed(tokens)
        if depths is not None:
            x = x + self.depth_embed(depths.clamp(0, 31))
        b, t = tokens.shape
        # дискретные токены получают виртуальное время (равномерный такт)
        time = torch.arange(t, device=tokens.device, dtype=x.dtype).expand(b, t) * 1e-3
        return LatentPacket(features=x, modality=modality or self.modality, time=time)
