"""Дистилляция знаний из существующей LLM в ядро NEXUS.

Три режима:

``logit``     онлайн-дистилляция по логитам: KL(student ‖ teacher) при
              температуре T плюс CE по реальным токенам. Требует общего
              токенизатора — студент автоматически переводится на словарь
              учителя (`HFTokenizerAdapter`).
``cached``    то же, но логиты учителя считаются один раз и кладутся на диск
              (top-k), после чего учитель не нужен в памяти — удобно, когда
              учитель не влезает в VRAM рядом со студентом.
``sequence``  чёрный ящик: учитель генерирует ответы на промпты, студент
              учится на полученном корпусе. Работает с любым учителем,
              включая закрытые API-модели.

Учителя: `HFTeacher` (transformers) и `NexusTeacher` (другой чекпойнт NEXUS —
используется в тестах и офлайне).
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from ..config import NexusConfig
from ..data.corpora import PackedLMDataset, split_dataset
from ..data.tokenizer import DEFAULT_TOKENIZER
from ..model import NexusEngine
from ..registry import ModelRegistry
from .trainer import TrainConfig, Trainer, resume_or_new


# --------------------------------------------------------------- интерфейсы
class Teacher(Protocol):
    vocab_size: int

    def logits(self, tokens: torch.Tensor) -> torch.Tensor: ...
    def generate_text(self, prompt: str, max_new_tokens: int = 128) -> str: ...


class HFTokenizerAdapter:
    """Приводит токенизатор HuggingFace к интерфейсу ASTBPETokenizer."""

    def __init__(self, hf_tokenizer):
        self.hf = hf_tokenizer
        self.vocab_size = int(getattr(hf_tokenizer, "vocab_size", len(hf_tokenizer)))
        self.pad_id = hf_tokenizer.pad_token_id if hf_tokenizer.pad_token_id is not None else 0
        self.bos_id = hf_tokenizer.bos_token_id if hf_tokenizer.bos_token_id is not None else self.pad_id
        self.eos_id = hf_tokenizer.eos_token_id if hf_tokenizer.eos_token_id is not None else self.pad_id

    def encode(self, text: str, bos: bool = False, eos: bool = False) -> List[int]:
        ids = self.hf.encode(text, add_special_tokens=False)
        if bos:
            ids = [self.bos_id] + ids
        if eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        return self.hf.decode(list(ids), skip_special_tokens=True)

    def special(self, name: str) -> int:
        return self.pad_id


class HFTeacher:
    """Учитель на базе transformers (`AutoModelForCausalLM`)."""

    def __init__(self, model_id: str, device: str = "cpu", dtype: str = "auto",
                 trust_remote_code: bool = False):
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("нужен пакет transformers: pip install transformers") from exc
        torch_dtype = {"auto": None, "fp16": torch.float16,
                       "bf16": torch.bfloat16, "fp32": torch.float32}[dtype]
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch_dtype, trust_remote_code=trust_remote_code).to(device).eval()
        self.device = device
        self.adapter = HFTokenizerAdapter(self.tokenizer)
        self.vocab_size = int(self.model.get_output_embeddings().weight.shape[0])

    @torch.no_grad()
    def logits(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids=tokens.to(self.device)).logits.float()

    @torch.no_grad()
    def generate_text(self, prompt: str, max_new_tokens: int = 128) -> str:
        ids = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        out = self.model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=True,
                                  temperature=0.8, top_p=0.95)
        return self.tokenizer.decode(out[0], skip_special_tokens=True)


class NexusTeacher:
    """Учитель — другой чекпойнт NEXUS (офлайн, для тестов и самодистилляции)."""

    def __init__(self, model: NexusEngine, tokenizer=None, device: str = "cpu"):
        self.model = model.to(device).eval()
        self.device = device
        self.adapter = tokenizer or DEFAULT_TOKENIZER
        self.vocab_size = model.cfg.vocab_size

    @classmethod
    def from_registry(cls, name: str, ref: str = "production",
                      registry_root: str = "artifacts/registry", device: str = "cpu"):
        from .trainer import load_model
        model, _ = load_model(ModelRegistry(registry_root), name, ref, device)
        return cls(model, device=device)

    @torch.no_grad()
    def logits(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.model(tokens=tokens.to(self.device), reason=False).logits.float()

    @torch.no_grad()
    def generate_text(self, prompt: str, max_new_tokens: int = 64) -> str:
        ids = torch.tensor([self.adapter.encode(prompt, bos=True)], device=self.device)
        out = self.model.generate(ids, max_new_tokens=max_new_tokens)
        return self.adapter.decode(out[0].tolist())


# ------------------------------------------------------------------- потери
@dataclass
class DistillConfig:
    temperature: float = 2.0
    alpha: float = 0.7           # вес KD относительно CE
    top_k: int = 0               # 0 — полный словарь, иначе top-k логитов учителя
    mode: str = "logit"          # logit | cached | sequence


def kd_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor,
            targets: torch.Tensor, cfg: DistillConfig,
            pad_id: int = 0) -> Dict[str, torch.Tensor]:
    v = min(student_logits.shape[-1], teacher_logits.shape[-1])
    s = student_logits[..., :v]
    t = teacher_logits[..., :v]
    T = cfg.temperature
    mask = (targets != pad_id).reshape(-1)

    log_p = F.log_softmax(s.reshape(-1, v)[mask] / T, dim=-1)
    with torch.no_grad():
        q = F.softmax(t.reshape(-1, v)[mask] / T, dim=-1)
    kd = F.kl_div(log_p, q, reduction="batchmean") * (T * T)
    ce = F.cross_entropy(s.reshape(-1, v), targets.reshape(-1), ignore_index=pad_id)
    total = cfg.alpha * kd + (1 - cfg.alpha) * ce
    return {"loss": total, "kd": kd.detach(), "ce": ce.detach()}


# ------------------------------------------------- кэш логитов учителя (top-k)
class CachedLogitsDataset(Dataset):
    def __init__(self, path: str):
        blob = np.load(path)
        self.tokens = torch.from_numpy(blob["tokens"]).long()
        self.targets = torch.from_numpy(blob["targets"]).long()
        self.values = torch.from_numpy(blob["values"]).float()
        self.indices = torch.from_numpy(blob["indices"]).long()
        self.vocab_size = int(blob["vocab_size"])

    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, i: int):
        return {"tokens": self.tokens[i], "targets": self.targets[i],
                "topk_values": self.values[i], "topk_indices": self.indices[i]}


@torch.no_grad()
def cache_teacher_logits(teacher: Teacher, dataset: Dataset, path: str,
                         top_k: int = 64, batch_size: int = 2,
                         max_batches: Optional[int] = None) -> str:
    """Посчитать и сохранить top-k логитов учителя (учитель после этого не нужен)."""
    from torch.utils.data import DataLoader
    dl = DataLoader(dataset, batch_size=batch_size)
    toks, tgts, vals, idxs = [], [], [], []
    for i, batch in enumerate(dl):
        if max_batches and i >= max_batches:
            break
        logits = teacher.logits(batch["tokens"])
        v, ix = torch.topk(logits, min(top_k, logits.shape[-1]), dim=-1)
        toks.append(batch["tokens"].cpu().numpy())
        tgts.append(batch["targets"].cpu().numpy())
        vals.append(v.cpu().numpy().astype(np.float16))
        idxs.append(ix.cpu().numpy().astype(np.int32))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez_compressed(path, tokens=np.concatenate(toks), targets=np.concatenate(tgts),
                        values=np.concatenate(vals), indices=np.concatenate(idxs),
                        vocab_size=np.array(teacher.vocab_size))
    return path


def cached_kd_loss(model: NexusEngine, batch: Dict[str, torch.Tensor],
                   cfg: DistillConfig, pad_id: int = 0) -> Dict[str, torch.Tensor]:
    logits = model(tokens=batch["tokens"], reason=False).logits
    v = logits.shape[-1]
    idx = batch["topk_indices"].clamp(max=v - 1)
    teacher_sparse = torch.full_like(logits, float("-inf"))
    teacher_sparse.scatter_(-1, idx, batch["topk_values"].to(logits.dtype))
    return kd_loss(logits, teacher_sparse, batch["targets"], cfg, pad_id)


# ------------------------------------------- sequence-level (чёрный ящик)
def build_sequence_corpus(teacher: Teacher, prompts: Sequence[str], out_path: str,
                          max_new_tokens: int = 128, samples_per_prompt: int = 1) -> str:
    """Учитель генерирует ответы → JSONL-корпус для обычного LM-обучения."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        for prompt in prompts:
            for _ in range(samples_per_prompt):
                text = teacher.generate_text(prompt, max_new_tokens=max_new_tokens)
                fh.write(json.dumps({"text": text, "prompt": prompt}, ensure_ascii=False) + "\n")
    return out_path


# ------------------------------------------------------------------- обучение
def distill(
    teacher: Optional[Teacher] = None,
    source: str = "builtin:engineering",
    model_name: str = "core-distill",
    preset: str = "tiny",
    dcfg: Optional[DistillConfig] = None,
    resume: Optional[str] = None,
    seq_len: int = 256,
    limit: Optional[int] = None,
    cache_path: Optional[str] = None,
    registry_root: str = "artifacts/registry",
    tokenizer=None,
    promote: bool = False,
    **train_kwargs,
) -> Dict[str, object]:
    dcfg = dcfg or DistillConfig()
    device = train_kwargs.get("device", "cpu")
    cfg = NexusConfig.tiny() if preset == "tiny" else (
        NexusConfig.rtx5060_compact() if preset == "rtx5060-compact" else NexusConfig.rtx5060())

    if dcfg.mode == "cached" and cache_path and os.path.exists(cache_path):
        ds: Dataset = CachedLogitsDataset(cache_path)
        cfg.vocab_size = max(cfg.vocab_size, ds.vocab_size)  # type: ignore[attr-defined]
        pad_id = 0
    else:
        tok = tokenizer or getattr(teacher, "adapter", None) or DEFAULT_TOKENIZER
        ds = PackedLMDataset(source, tokenizer=tok, seq_len=seq_len, limit=limit, min_blocks=8)
        cfg.vocab_size = max(cfg.vocab_size, tok.vocab_size,
                             getattr(teacher, "vocab_size", 0) if teacher else 0)
        pad_id = getattr(tok, "pad_id", 0)
        if dcfg.mode == "cached":
            assert teacher is not None and cache_path, "для режима cached нужен teacher и cache_path"
            cache_teacher_logits(teacher, ds, cache_path, top_k=dcfg.top_k or 64)
            ds = CachedLogitsDataset(cache_path)

    train_ds, val_ds = split_dataset(ds, 0.1)
    registry = ModelRegistry(registry_root)
    model, parent = resume_or_new(registry, model_name, cfg, resume, device)

    if dcfg.mode == "cached":
        def loss_fn(m, b):
            return cached_kd_loss(m, b, dcfg, pad_id)
    elif dcfg.mode == "logit":
        assert teacher is not None, "для режима logit нужен учитель"

        def loss_fn(m, b):
            out = m(tokens=b["tokens"], reason=False)
            t_logits = teacher.logits(b["tokens"])
            return kd_loss(out.logits, t_logits, b["targets"], dcfg, pad_id)
    else:  # sequence — обычный LM-лосс на корпусе, порождённом учителем
        from .trainer import lm_loss

        def loss_fn(m, b):
            return lm_loss(m, b, reason=False, pad_id=pad_id)

    tcfg = TrainConfig(model_name=model_name, stage=f"distill:{dcfg.mode}",
                       dataset=source, registry_root=registry_root,
                       notes=f"T={dcfg.temperature}, alpha={dcfg.alpha}", **train_kwargs)
    trainer = Trainer(model, tcfg, train_ds, val_ds, loss_fn=loss_fn)
    metrics = trainer.fit()
    metrics["distill_mode"] = dcfg.mode  # type: ignore[assignment]
    version = trainer.save({k: v for k, v in metrics.items() if isinstance(v, (int, float))},
                           parent=parent)
    if promote:
        registry.promote(model_name, version.version, "production", reason="distill --promote")
    return {"metrics": metrics, "version": version.to_dict()}


def main() -> None:
    ap = argparse.ArgumentParser(description="Дистилляция LLM → NEXUS")
    ap.add_argument("--teacher", default=None, help="HF-модель (например Qwen/Qwen2.5-0.5B)")
    ap.add_argument("--teacher-nexus", default=None, help="имя модели NEXUS из реестра")
    ap.add_argument("--teacher-ref", default="production")
    ap.add_argument("--mode", choices=["logit", "cached", "sequence"], default="logit")
    ap.add_argument("--source", default="builtin:engineering")
    ap.add_argument("--model-name", default="core-distill")
    ap.add_argument("--preset", choices=["tiny", "rtx5060", "rtx5060-compact"], default="tiny")
    ap.add_argument("--temperature", type=float, default=2.0)
    ap.add_argument("--alpha", type=float, default=0.7)
    ap.add_argument("--top-k", type=int, default=64)
    ap.add_argument("--cache", default="artifacts/cache/teacher_topk.npz")
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--registry", default="artifacts/registry")
    ap.add_argument("--promote", action="store_true")
    a = ap.parse_args()

    teacher: Optional[Teacher] = None
    tokenizer = None
    if a.teacher:
        t = HFTeacher(a.teacher, device=a.device)
        teacher, tokenizer = t, t.adapter
    elif a.teacher_nexus:
        teacher = NexusTeacher.from_registry(a.teacher_nexus, a.teacher_ref, a.registry, a.device)

    out = distill(teacher, a.source, a.model_name, a.preset,
                  DistillConfig(a.temperature, a.alpha, a.top_k, a.mode),
                  seq_len=a.seq_len, cache_path=a.cache, registry_root=a.registry,
                  tokenizer=tokenizer, promote=a.promote,
                  epochs=a.epochs, batch_size=a.batch_size, grad_accum=a.grad_accum,
                  lr=a.lr, max_steps=a.max_steps, device=a.device)
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
