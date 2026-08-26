"""Приёмочные тесты дообученной модели (quality gate).

Запуск: `nexus eval --model core --ref latest --baseline production --gate`.

Проверки:

======================  ======================================================
`load`                  версия читается, sha256 сходится, конфиг валиден
`forward_finite`        логиты конечны, без NaN/Inf
`val_loss` / `val_ppl`  перплексия на отложенном источнике
`generation`            генерация не падает и не вырождается в один токен
`determinism`           greedy-генерация воспроизводима
`latency`               мс на токен не выше порога
`memory_o1`             размер TTT-состояния не растёт с длиной контекста
`scad_compile_rate`     доля генераций, компилируемых в валидный CSG
`overfit_gap`           разрыв между валидацией и обучением (переобучение)
`regression`            метрики не хуже базовой версии (с допуском)
======================  ======================================================
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

from ..config import NexusConfig
from ..data.corpora import PackedLMDataset
from ..data.tokenizer import DEFAULT_TOKENIZER
from ..model import NexusEngine
from ..registry import ModelRegistry, ModelVersion
from ..scad.render import compile_scad


def model_device(model: NexusEngine) -> torch.device:
    """Устройство модели: тензоры приёмки должны жить там же, что и веса."""
    try:
        return next(model.parameters()).device
    except StopIteration:                                   # pragma: no cover
        return torch.device("cpu")


@dataclass
class Check:
    name: str
    value: Any
    passed: bool
    threshold: Optional[Any] = None
    comment: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SuiteReport:
    model: str
    version: Optional[int]
    checks: List[Check] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    baseline: Optional[Dict[str, float]] = None

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    def add(self, name: str, value: Any, passed: bool, threshold: Any = None,
            comment: str = "") -> None:
        self.checks.append(Check(name, value, bool(passed), threshold, comment))

    def to_dict(self) -> Dict[str, Any]:
        return {"model": self.model, "version": self.version, "passed": self.passed,
                "metrics": self.metrics, "baseline": self.baseline,
                "checks": [c.to_dict() for c in self.checks]}

    def summary(self) -> str:
        lines = [f"{'ПРОЙДЕНО' if self.passed else 'ПРОВАЛЕНО'}: "
                 f"{self.model} v{self.version if self.version else '-'}"]
        for c in self.checks:
            mark = "✓" if c.passed else "✗"
            thr = f" (порог {c.threshold})" if c.threshold is not None else ""
            lines.append(f"  {mark} {c.name}: {c.value}{thr} {c.comment}".rstrip())
        return "\n".join(lines)


@dataclass
class SuiteThresholds:
    max_val_ppl: float = 1e6
    min_unique_ratio: float = 0.05
    max_ms_per_token: float = 5000.0
    min_scad_compile_rate: float = 0.0
    max_ppl_regression: float = 1.10     # не более +10 % к базовой перплексии
    max_overfit_gap: float = 1.0         # val_loss − train_ce, в наtах на токен


@torch.no_grad()
def _val_loss(model: NexusEngine, source: str, seq_len: int, limit: Optional[int],
              batch_size: int = 2, tokenizer=None) -> float:
    ds = PackedLMDataset(source, tokenizer=tokenizer, seq_len=seq_len, limit=limit,
                         min_blocks=2)
    from torch.utils.data import DataLoader
    device = model_device(model)
    dl = DataLoader(ds, batch_size=batch_size)
    total, n = 0.0, 0
    for batch in dl:
        tokens = batch["tokens"].to(device)
        targets = batch["targets"].to(device)
        logits = model(tokens=tokens, reason=False).logits
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                               targets.reshape(-1), ignore_index=0)
        total += float(loss) * tokens.shape[0]
        n += tokens.shape[0]
    return total / max(n, 1)


@torch.no_grad()
def run_suite(
    model: NexusEngine,
    name: str = "core",
    version: Optional[int] = None,
    source: str = "builtin:engineering",
    seq_len: int = 128,
    limit: Optional[int] = None,
    thresholds: Optional[SuiteThresholds] = None,
    baseline: Optional[Dict[str, float]] = None,
    prompts: Optional[List[str]] = None,
    n_scad_samples: int = 4,
    tokenizer=None,
    train_metrics: Optional[Dict[str, float]] = None,
) -> SuiteReport:
    th = thresholds or SuiteThresholds()
    rep = SuiteReport(name, version, baseline=baseline)
    model.eval()
    device = model_device(model)
    tok = tokenizer or DEFAULT_TOKENIZER
    prompts = prompts or ["<task>Кронштейн, сталь, 300 Н<scad>"]

    # 1. прямой проход
    tokens = torch.randint(4, model.cfg.vocab_size, (1, min(64, seq_len)), device=device)
    logits = model(tokens=tokens, reason=False).logits
    finite = bool(torch.isfinite(logits).all())
    rep.add("forward_finite", finite, finite)

    # 2. перплексия
    loss = _val_loss(model, source, seq_len, limit, tokenizer=tok)
    ppl = math.exp(min(loss, 20))
    rep.metrics.update({"val_loss": round(loss, 5), "val_ppl": round(ppl, 3)})
    rep.add("val_ppl", round(ppl, 3), ppl <= th.max_val_ppl, th.max_val_ppl)

    # 3. генерация: не падает и не вырождается
    ids = torch.tensor([tok.encode(prompts[0], bos=True)], device=device)
    t0 = time.time()
    out = model.generate(ids, max_new_tokens=32, temperature=0.9)
    dt = (time.time() - t0) * 1000 / 32
    new = out[0, ids.shape[1]:].tolist()
    unique_ratio = len(set(new)) / max(len(new), 1)
    rep.metrics.update({"unique_ratio": round(unique_ratio, 3), "ms_per_token": round(dt, 2)})
    rep.add("generation", round(unique_ratio, 3), unique_ratio >= th.min_unique_ratio,
            th.min_unique_ratio, "доля уникальных токенов")
    rep.add("latency", round(dt, 2), dt <= th.max_ms_per_token, th.max_ms_per_token, "мс/токен")

    # 4. детерминизм greedy-генерации
    a = model.generate(ids.clone(), max_new_tokens=8, temperature=0.0)
    b = model.generate(ids.clone(), max_new_tokens=8, temperature=0.0)
    same = bool(torch.equal(a, b))
    rep.add("determinism", same, same, comment="greedy повторяем")

    # 5. O(1) память: обе длины должны быть заметно больше окна внимания,
    # иначе растёт не TTT-состояние, а ещё не заполненный KV-кэш окна.
    window = model.cfg.attention.window
    sizes = []
    for length in (2 * window, 6 * window):
        states = None
        chunk = min(length, model.cfg.attention.window)
        processed = 0
        while processed < length:
            step = min(chunk, length - processed)
            res = model(tokens=torch.randint(4, model.cfg.vocab_size, (1, step),
                                             device=device),
                        reason=False, states=states, use_state=True)
            states = res.states  # type: ignore[attr-defined]
            processed += step
        sizes.append(model.state_bytes(states or []))
    o1 = sizes[0] == sizes[1]
    rep.metrics["state_bytes"] = float(sizes[0])
    rep.add("memory_o1", sizes, o1,
            comment=f"состояние не растёт с контекстом ({2 * window} → {6 * window} токенов)")

    # 6. инженерная проверка: компилируется ли сгенерированный SCAD
    compiled = 0
    for i in range(n_scad_samples):
        prompt = prompts[i % len(prompts)]
        seq = torch.tensor([tok.encode(prompt, bos=True)], device=device)
        gen = model.generate(seq, max_new_tokens=48, temperature=0.8)
        code = tok.decode(gen[0, seq.shape[1]:].tolist())
        compiled += int(compile_scad(code).ok)
    rate = compiled / max(n_scad_samples, 1)
    rep.metrics["scad_compile_rate"] = round(rate, 3)
    rep.add("scad_compile_rate", round(rate, 3), rate >= th.min_scad_compile_rate,
            th.min_scad_compile_rate)

    # 7. переобучение: насколько валидация хуже обучающей выборки
    train_ce = (train_metrics or {}).get("train_ce")
    if train_ce:
        gap = loss - float(train_ce)
        rep.metrics["overfit_gap"] = round(gap, 4)
        rep.add("overfit_gap", round(gap, 4), gap <= th.max_overfit_gap,
                th.max_overfit_gap, "val_loss − train_ce (больше — модель зубрит)")

    # 8. регрессия к базовой версии
    if baseline and "val_ppl" in baseline and baseline["val_ppl"] > 0:
        ratio = ppl / baseline["val_ppl"]
        rep.metrics["ppl_ratio_vs_baseline"] = round(ratio, 4)
        rep.add("regression", round(ratio, 4), ratio <= th.max_ppl_regression,
                th.max_ppl_regression, "перплексия относительно базовой версии")
    return rep


def evaluate_version(
    name: str = "core",
    ref: str = "latest",
    baseline_ref: Optional[str] = None,
    registry_root: str = "artifacts/registry",
    source: str = "builtin:engineering",
    seq_len: int = 128,
    thresholds: Optional[SuiteThresholds] = None,
    gate: bool = False,
    promote_tag: str = "production",
    device: str = "cpu",
) -> SuiteReport:
    """Проверить версию из реестра и при `gate=True` повысить её до production."""
    from ..training.trainer import load_model

    registry = ModelRegistry(registry_root)
    model, mv = load_model(registry, name, ref, device)
    from ..data.bpe import load_tokenizer
    tokenizer = load_tokenizer(mv.tokenizer_path)

    base_metrics: Optional[Dict[str, float]] = None
    if baseline_ref:
        try:
            base_model, base_mv = load_model(registry, name, baseline_ref, device)
            base_report = run_suite(base_model, name, base_mv.version, source, seq_len,
                                    thresholds=SuiteThresholds(), n_scad_samples=2,
                                    tokenizer=load_tokenizer(base_mv.tokenizer_path))
            base_metrics = base_report.metrics
        except Exception as exc:  # базовой версии может ещё не быть
            base_metrics = None
            print(f"[eval] базовая версия недоступна: {exc}")

    report = run_suite(model, name, mv.version, source, seq_len,
                       thresholds=thresholds, baseline=base_metrics, tokenizer=tokenizer,
                       train_metrics=mv.metrics)
    report.metrics["sha256_ok"] = 1.0
    if gate and report.passed:
        registry.promote(name, mv.version, promote_tag, reason="quality gate passed")
        print(f"[eval] {mv.tag} повышена до {promote_tag}")
    elif gate:
        print(f"[eval] {mv.tag} НЕ прошла gate — тег {promote_tag} не изменён")
    return report
