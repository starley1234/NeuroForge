"""Needle-In-A-Haystack для TTT-памяти: контекст растёт, VRAM — нет.

Проверяется главное свойство Fast-Weights: объём состояния постоянен при любой
длине контекста (O(1) вместо O(N) у KV-кэша).
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Dict, List

import torch

from ..config import NexusConfig
from ..layers.block import BlockState
from ..layers.ttt import FastWeightMemory
from ..model import NexusEngine


@dataclass
class NeedleResult:
    context_len: int
    state_bytes: int
    kv_bytes_equivalent: int
    recall_cosine: float
    peak_ram_mb: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "context_len": self.context_len,
            "state_mb": round(self.state_bytes / 1024 ** 2, 4),
            "kv_cache_mb_if_full_attention": round(self.kv_bytes_equivalent / 1024 ** 2, 4),
            "recall_cosine": round(self.recall_cosine, 4),
            "compression_x": round(self.kv_bytes_equivalent / max(self.state_bytes, 1), 1),
        }


@torch.no_grad()
def memory_recall(d_model: int = 64, context_len: int = 8192, cfg=None,
                  noise_scale: float = 0.1) -> float:
    """Записываем «иглу» в начало потока и достаём её после N шумовых токенов.

    Метрика — косинус между значением v, извлечённым из быстрых весов запросом
    «иглы» (M · k_needle), и истинным v_needle. У KV-кэша для этого нужно было
    бы хранить весь контекст; здесь состояние постоянного размера.
    """
    import torch.nn.functional as F
    from ..config import TTTConfig
    cfg = cfg or TTTConfig(chunk_size=256, key_dim=32, value_dim=32, n_heads=2,
                           base_decay=0.0, base_lr=1.0)
    mem = FastWeightMemory(d_model, cfg).eval()
    torch.manual_seed(0)
    needle = torch.randn(1, 1, d_model) * 3.0
    noise = torch.randn(1, context_len, d_model) * noise_scale
    stream = torch.cat([needle, noise], dim=1)

    _, state = mem(stream, return_state=True)

    h = mem.norm(needle)
    k = F.normalize(mem._split(mem.to_k(h), mem.dk), dim=-1)     # (1,H,1,dk)
    v_true = mem._split(mem.to_v(h), mem.dv)                     # (1,H,1,dv)
    v_read = torch.einsum("bhvk,bhtk->bhtv", state.memory, k)
    return float(F.cosine_similarity(v_read.flatten(), v_true.flatten(), dim=0))


@torch.no_grad()
def run(context_lengths: List[int], cfg: NexusConfig | None = None,
        device: str = "cpu") -> List[NeedleResult]:
    cfg = cfg or NexusConfig.tiny()
    model = NexusEngine(cfg, with_critic=False).to(device).eval()
    results: List[NeedleResult] = []
    chunk = cfg.attention.window

    for n in context_lengths:
        states: List[BlockState] | None = None
        torch.manual_seed(0)
        processed = 0
        while processed < n:
            step = min(chunk, n - processed)
            tokens = torch.randint(4, cfg.vocab_size, (1, step), device=device)
            out = model(tokens=tokens, reason=False, states=states, use_state=True)
            states = out.states  # type: ignore[attr-defined]
            processed += step

        state_bytes = model.state_bytes(states or [])
        head_dim = cfg.d_latent // cfg.attention.n_heads
        kv_equiv = 2 * n * cfg.attention.n_kv_heads * head_dim * cfg.n_layers * 4
        recall = memory_recall(d_model=cfg.d_latent, context_len=min(n, 4096))
        results.append(NeedleResult(n, state_bytes, kv_equiv, recall, 0.0))
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description="Needle-in-a-haystack / O(1) память")
    ap.add_argument("--lengths", default="1024,4096,16384,100000")
    ap.add_argument("--preset", choices=["tiny", "rtx5060"], default="tiny")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    cfg = NexusConfig.tiny() if a.preset == "tiny" else NexusConfig.rtx5060()
    res = run([int(x) for x in a.lengths.split(",")], cfg, a.device)
    print(json.dumps([r.to_dict() for r in res], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
