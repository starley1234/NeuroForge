"""Фаза 4: Physics-RL через GRPO (Group Relative Policy Optimization).

Критик-сеть не нужна: baseline — среднее по группе сэмплов на один промпт.
Награда чисто физическая (nexus.training.rewards): компиляция SCAD, manifold,
запас прочности по FEM, штрафы за массу и тонкие стенки.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F

from ..config import NexusConfig
from ..data.tokenizer import DEFAULT_TOKENIZER
from ..model import NexusEngine
from .rewards import RewardBreakdown, score_scad


@dataclass
class GRPOConfig:
    group_size: int = 4
    max_new_tokens: int = 96
    temperature: float = 1.0
    top_k: int = 40
    kl_coef: float = 0.02
    lr: float = 1e-5
    clip: float = 0.2
    required_sf: float = 2.0


def _sequence_logprobs(model: NexusEngine, tokens: torch.Tensor,
                       prompt_len: int) -> torch.Tensor:
    logits = model(tokens=tokens, reason=False).logits[:, :-1]
    logp = torch.log_softmax(logits.float(), dim=-1)
    tgt = tokens[:, 1:]
    picked = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    mask = torch.zeros_like(picked)
    mask[:, prompt_len - 1:] = 1.0
    return (picked * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)


def rollout(model: NexusEngine, prompt: str, cfg: GRPOConfig, device: str = "cpu"):
    tok = DEFAULT_TOKENIZER
    ids = tok.encode(prompt + "<scad>", bos=True)
    base = torch.tensor([ids], dtype=torch.long, device=device).repeat(cfg.group_size, 1)
    out = model.generate(base, cfg.max_new_tokens, cfg.temperature, cfg.top_k)
    codes = [tok.decode(seq[len(ids):].tolist()) for seq in out]
    return out, len(ids), codes


def grpo_step(model: NexusEngine, reference: Optional[NexusEngine], prompt: str,
              cfg: GRPOConfig, opt: torch.optim.Optimizer,
              reward_kwargs: Optional[Dict] = None, device: str = "cpu") -> Dict[str, float]:
    seqs, prompt_len, codes = rollout(model, prompt, cfg, device)
    breakdowns: List[RewardBreakdown] = [
        score_scad(c, required_sf=cfg.required_sf, **(reward_kwargs or {})) for c in codes
    ]
    rewards = torch.tensor([b.total for b in breakdowns], device=device, dtype=torch.float32)
    adv = (rewards - rewards.mean()) / rewards.std().clamp(min=1e-4)

    logp = _sequence_logprobs(model, seqs, prompt_len)
    loss = -(adv.detach() * logp).mean()
    if reference is not None and cfg.kl_coef > 0:
        with torch.no_grad():
            ref_logp = _sequence_logprobs(reference, seqs, prompt_len)
        loss = loss + cfg.kl_coef * (logp - ref_logp).pow(2).mean()

    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    return {
        "loss": float(loss.detach()),
        "reward_mean": float(rewards.mean()),
        "reward_max": float(rewards.max()),
        "compile_rate": float(sum(b.compile > 0 for b in breakdowns)) / len(breakdowns),
        "manifold_rate": float(sum(b.manifold >= 1.5 for b in breakdowns)) / len(breakdowns),
    }


def train(prompts: Sequence[str], checkpoint: Optional[str] = None,
          out: str = "artifacts/checkpoints/core_rl.pt", steps: int = 10,
          cfg: Optional[GRPOConfig] = None, device: str = "cpu",
          preset: str = "tiny") -> List[Dict[str, float]]:
    cfg = cfg or GRPOConfig()
    if checkpoint and os.path.exists(checkpoint):
        blob = torch.load(checkpoint, map_location=device, weights_only=False)
        model_cfg = NexusConfig.from_dict(blob["config"])
        model = NexusEngine(model_cfg).to(device)
        model.load_state_dict(blob["model"])
    else:
        model_cfg = NexusConfig.tiny() if preset == "tiny" else NexusConfig.rtx5060()
        model = NexusEngine(model_cfg).to(device)

    reference = NexusEngine(model_cfg).to(device)
    reference.load_state_dict(model.state_dict())
    reference.eval().requires_grad_(False)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    log: List[Dict[str, float]] = []
    for step in range(steps):
        stats = grpo_step(model, reference, prompts[step % len(prompts)], cfg, opt, device=device)
        stats["step"] = step
        log.append(stats)
        print(f"grpo {step:3d} R={stats['reward_mean']:+.3f} "
              f"compile={stats['compile_rate']:.2f} manifold={stats['manifold_rate']:.2f}",
              flush=True)

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    torch.save({"model": model.state_dict(), "config": model_cfg.to_dict()}, out)
    with open(out + ".log.json", "w", encoding="utf-8") as fh:
        json.dump(log, fh, indent=2, ensure_ascii=False)
    return log


DEFAULT_PROMPTS = [
    "<task>Кронштейн L-образный, алюминий 6061, консольная нагрузка 350 Н вниз, запас 2.0.",
    "<task>Фланец с 6 болтами, сталь 304, осевая тяга 900 Н, минимальная стенка 2 мм.",
    "<task>Монтажная плита PETG с окном облегчения, распределённая нагрузка 120 Н.",
]


def main() -> None:
    ap = argparse.ArgumentParser(description="Physics-RL (GRPO) для NEXUS")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--out", default="artifacts/checkpoints/core_rl.pt")
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--group-size", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    train(DEFAULT_PROMPTS, a.checkpoint, a.out, a.steps,
          GRPOConfig(group_size=a.group_size, max_new_tokens=a.max_new_tokens),
          device=a.device)


if __name__ == "__main__":
    main()
