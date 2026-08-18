"""Универсальный тренер: общий цикл для pretrain / LM / дистилляции / дообучения.

Возможности: grad-accum, косинусный LR с прогревом, AMP на CUDA, клиппинг,
валидация, ранняя остановка, автосохранение **новой версии** в реестр моделей
(старые версии никогда не перезаписываются), возобновление из реестра.
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from ..config import NexusConfig
from ..model import NexusEngine
from ..registry import ModelRegistry, ModelVersion

LossFn = Callable[[NexusEngine, Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]


@dataclass
class TrainConfig:
    epochs: int = 1
    batch_size: int = 2
    grad_accum: int = 8
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup: int = 20
    max_steps: Optional[int] = None
    clip: float = 1.0
    device: str = "auto"
    amp: bool = False
    eight_bit: bool = False
    log_every: int = 10
    eval_every: int = 0                  # 0 — только в конце эпохи
    patience: int = 0                    # 0 — без ранней остановки
    seed: int = 0
    num_workers: int = 0
    reason_every: int = 4                # как часто включать латентный контур
    model_name: str = "core"
    stage: str = "train"
    dataset: str = ""
    notes: str = ""
    tokenizer_path: Optional[str] = None
    registry_root: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def build_optimizer(model: torch.nn.Module, lr: float, wd: float, eight_bit: bool = False):
    decay, no_decay = [], []
    for _, p in model.named_parameters():
        if p.requires_grad:
            (no_decay if p.ndim < 2 else decay).append(p)
    groups = [{"params": decay, "weight_decay": wd},
              {"params": no_decay, "weight_decay": 0.0}]
    if eight_bit:
        try:
            import bitsandbytes as bnb  # type: ignore
            return bnb.optim.AdamW8bit(groups, lr=lr, betas=(0.9, 0.95))
        except Exception:
            print("[warn] bitsandbytes недоступен → обычный AdamW")
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95))


def cosine_lr(step: int, total: int, base: float, warmup: int) -> float:
    if warmup and step < warmup:
        return base * (step + 1) / warmup
    t = (step - warmup) / max(total - warmup, 1)
    return base * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(max(t, 0.0), 1.0))))


def lm_loss(model: NexusEngine, batch: Dict[str, torch.Tensor],
            reason: bool = False, pad_id: int = 0) -> Dict[str, torch.Tensor]:
    """Стандартная LM-задача: батч уже сдвинут (tokens → targets)."""
    out = model(tokens=batch["tokens"], reason=reason)
    ce = F.cross_entropy(out.logits.reshape(-1, out.logits.shape[-1]),
                         batch["targets"].reshape(-1), ignore_index=pad_id)
    total = ce + (out.aux_loss if out.aux_loss is not None else 0.0)
    if out.reasoning is not None:
        total = total + out.reasoning.ponder_cost
    return {"loss": total, "ce": ce.detach()}


class Trainer:
    def __init__(self, model: NexusEngine, cfg: TrainConfig,
                 train_ds: Dataset, val_ds: Optional[Dataset] = None,
                 loss_fn: Optional[LossFn] = None,
                 registry: Optional[ModelRegistry] = None):
        from ..runtime import pick_device
        torch.manual_seed(cfg.seed)
        cfg.device = pick_device(cfg.device)
        self.cfg = cfg
        self.model = model.to(cfg.device)
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.loss_fn = loss_fn or (lambda m, b: lm_loss(m, b, reason=bool(b.get("_reason", False))))
        self.registry = registry or ModelRegistry(cfg.registry_root or "artifacts/registry")
        self.history: list[Dict[str, float]] = []

    # ------------------------------------------------------------------ цикл
    def fit(self) -> Dict[str, float]:
        cfg = self.cfg
        dl = DataLoader(self.train_ds, batch_size=cfg.batch_size, shuffle=True,
                        num_workers=cfg.num_workers, drop_last=False)
        total = cfg.max_steps or (len(dl) * cfg.epochs)
        opt = build_optimizer(self.model, cfg.lr, cfg.weight_decay, cfg.eight_bit)
        scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and cfg.device.startswith("cuda"))

        step, best, bad, t0 = 0, float("inf"), 0, time.time()
        stop = False
        self.model.train()
        for epoch in range(cfg.epochs):
            for batch in dl:
                for g in opt.param_groups:
                    g["lr"] = cosine_lr(step, total, cfg.lr, cfg.warmup)
                batch = {k: v.to(cfg.device) for k, v in batch.items()}
                reason = bool(cfg.reason_every) and (step % cfg.reason_every == 0)
                with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                    stats = self.loss_fn(self.model, {**batch, "_reason": reason})  # type: ignore[arg-type]
                loss = stats["loss"] / cfg.grad_accum
                scaler.scale(loss).backward() if scaler.is_enabled() else loss.backward()

                if (step + 1) % cfg.grad_accum == 0:
                    if scaler.is_enabled():
                        scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.clip)
                    if scaler.is_enabled():
                        scaler.step(opt)
                        scaler.update()
                    else:
                        opt.step()
                    opt.zero_grad(set_to_none=True)

                if step % cfg.log_every == 0:
                    msg = " ".join(f"{k}={float(v.detach()):.4f}" for k, v in stats.items()
                                   if isinstance(v, torch.Tensor))
                    print(f"[{cfg.stage}] step {step:5d}/{total} {msg} "
                          f"lr={opt.param_groups[0]['lr']:.2e} ({time.time() - t0:.1f}s)",
                          flush=True)
                self.history.append({"step": step,
                                     **{k: float(v.detach()) for k, v in stats.items()
                                        if isinstance(v, torch.Tensor)}})
                step += 1

                if cfg.eval_every and step % cfg.eval_every == 0 and self.val_ds is not None:
                    val = self.evaluate()
                    print(f"[{cfg.stage}] val {val}", flush=True)
                    if val["val_loss"] < best - 1e-4:
                        best, bad = val["val_loss"], 0
                    else:
                        bad += 1
                        if cfg.patience and bad >= cfg.patience:
                            print(f"[{cfg.stage}] ранняя остановка на шаге {step}")
                            stop = True
                    self.model.train()
                if (cfg.max_steps and step >= cfg.max_steps) or stop:
                    stop = True
                    break
            if stop:
                break

        metrics = {"steps": float(step), "train_loss": self.history[-1]["loss"] if self.history else 0.0,
                   "seconds": round(time.time() - t0, 2)}
        if self.val_ds is not None:
            metrics.update(self.evaluate())
        return metrics

    @torch.no_grad()
    def evaluate(self, dataset: Optional[Dataset] = None) -> Dict[str, float]:
        ds = dataset or self.val_ds
        if ds is None:
            return {}
        self.model.eval()
        dl = DataLoader(ds, batch_size=self.cfg.batch_size)
        tot, n = 0.0, 0
        for batch in dl:
            batch = {k: v.to(self.cfg.device) for k, v in batch.items()}
            stats = self.loss_fn(self.model, {**batch, "_reason": False})  # type: ignore[arg-type]
            bs = batch["tokens"].shape[0]
            tot += float(stats.get("ce", stats["loss"])) * bs
            n += bs
        self.model.train()
        loss = tot / max(n, 1)
        return {"val_loss": round(loss, 5), "val_ppl": round(math.exp(min(loss, 20)), 3)}

    # ------------------------------------------------------------- сохранение
    def save(self, metrics: Optional[Dict[str, float]] = None,
             parent: Optional[int] = None) -> ModelVersion:
        mv = self.registry.save(
            self.cfg.model_name, self.model.state_dict(), self.model.cfg.to_dict(),
            kind="core", metrics=metrics or {}, parent=parent,
            dataset=self.cfg.dataset, stage=self.cfg.stage, notes=self.cfg.notes,
            tokenizer_path=self.cfg.tokenizer_path,
            extra={"train_config": self.cfg.to_dict()},
        )
        print(f"[registry] сохранено {mv.tag} → {mv.path}")
        return mv


# ------------------------------------------------------------------ загрузка
def load_model(registry: ModelRegistry, name: str, ref: str = "latest",
               device: str = "cpu") -> tuple[NexusEngine, ModelVersion]:
    state, cfg_dict, mv = registry.load(name, ref, map_location=device)
    model = NexusEngine(NexusConfig.from_dict(cfg_dict)).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, mv


def resume_or_new(registry: ModelRegistry, name: str, cfg: NexusConfig,
                  ref: Optional[str] = None, device: str = "cpu"):
    """Продолжить дообучение с версии из реестра либо начать с нуля."""
    if ref:
        model, mv = load_model(registry, name, ref, device)
        return model, mv.version
    return NexusEngine(cfg).to(device), None
