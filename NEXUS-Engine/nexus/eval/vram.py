"""Калькулятор бюджета VRAM (раздел 5 спецификации, цель — RTX 5060 16 ГБ)."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Dict

from ..config import NexusConfig

GB = 1024 ** 3


@dataclass
class VramBudget:
    core_gb: float
    moe_gb: float
    fno_gb: float
    optimizer_gb: float
    activations_gb: float
    ttt_state_gb: float

    @property
    def total_gb(self) -> float:
        return round(self.core_gb + self.moe_gb + self.fno_gb + self.optimizer_gb
                     + self.activations_gb + self.ttt_state_gb, 2)

    def to_dict(self) -> Dict[str, float]:
        d = {k: round(v, 3) for k, v in self.__dict__.items()}
        d["total_gb"] = self.total_gb
        d["headroom_gb_on_16gb"] = round(16.0 - self.total_gb, 2)
        d["fits_16gb"] = self.total_gb <= 16.0
        return d


def estimate(cfg: NexusConfig | None = None, batch_size: int = 2, chunk: int = 128,
             bytes_per_param: int = 2, moe_offload: float = 0.5,
             eight_bit_optimizer: bool = True, active_params: float | None = None,
             total_moe_params: float | None = None) -> VramBudget:
    cfg = cfg or NexusConfig.rtx5060()
    d, L = cfg.d_latent, cfg.n_layers

    # аналитическая оценка числа параметров ядра (без MoE)
    attn = 2 * d * d + 2 * d * (d * cfg.attention.n_kv_heads / cfg.attention.n_heads)
    ttt = d * (cfg.ttt.n_heads * (2 * cfg.ttt.key_dim + cfg.ttt.value_dim)) + \
        cfg.ttt.n_heads * cfg.ttt.value_dim * d
    expert = 3 * d * int(d * cfg.moe.expert_hidden_mult)
    dense_per_layer = attn + ttt + expert * (cfg.moe.n_active + cfg.moe.n_shared)
    active = active_params or (L * dense_per_layer + cfg.vocab_size * d)
    moe_all = total_moe_params or (L * expert * cfg.moe.n_experts)

    core_gb = active * bytes_per_param / GB
    moe_gb = moe_all * bytes_per_param * (1 - moe_offload) / GB
    fno_params = cfg.fno.width ** 2 * cfg.fno.modes ** 3 * 2 * cfg.fno.depth
    fno_gb = fno_params * bytes_per_param / GB
    opt_bytes = 1 if eight_bit_optimizer else 8
    optimizer_gb = active * opt_bytes * 2 / GB
    act_gb = batch_size * chunk * d * L * 18 * bytes_per_param / GB + 3.0 * 0
    ttt_state_gb = (batch_size * L * cfg.ttt.n_heads * cfg.ttt.key_dim
                    * cfg.ttt.value_dim * bytes_per_param) / GB
    return VramBudget(core_gb, moe_gb, fno_gb, optimizer_gb, act_gb, ttt_state_gb)


def main() -> None:
    ap = argparse.ArgumentParser(description="Бюджет VRAM NEXUS-Engine")
    ap.add_argument("--preset", choices=["tiny", "rtx5060"], default="rtx5060")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--chunk", type=int, default=128)
    ap.add_argument("--fp32", action="store_true")
    a = ap.parse_args()
    cfg = NexusConfig.tiny() if a.preset == "tiny" else NexusConfig.rtx5060()
    b = estimate(cfg, a.batch_size, a.chunk, 4 if a.fp32 else 2)
    print(json.dumps(b.to_dict(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
