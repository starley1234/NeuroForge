import json
import os

import pytest
import torch

from nexus.config import NexusConfig
from nexus.data.corpora import PackedLMDataset, iter_texts, split_dataset
from nexus.model import NexusEngine
from nexus.registry import ModelRegistry


# ------------------------------------------------------------------ корпуса
def test_builtin_corpus_packs_into_blocks():
    ds = PackedLMDataset("builtin:engineering", seq_len=64, min_blocks=4)
    assert len(ds) >= 4
    item = ds[0]
    assert item["tokens"].shape == (64,) and item["targets"].shape == (64,)
    assert torch.equal(item["tokens"][1:], item["targets"][:-1])   # корректный сдвиг


def test_dir_and_jsonl_sources(tmp_path):
    (tmp_path / "a.txt").write_text("Расчёт балки на изгиб. σ = M·y/I", encoding="utf-8")
    (tmp_path / "b.scad").write_text("cube([10,10,2], center=true);", encoding="utf-8")
    texts = list(iter_texts(f"dir:{tmp_path}"))
    assert len(texts) == 2

    jl = tmp_path / "c.jsonl"
    jl.write_text(json.dumps({"text": "hello"}) + "\n" + json.dumps({"text": "world"}) + "\n",
                  encoding="utf-8")
    assert list(iter_texts(f"jsonl:{jl}#text")) == ["hello", "world"]


def test_flywheel_source(tmp_path):
    from nexus.data.flywheel import run
    out = str(tmp_path / "fly")
    run(3, out, seed=1, grid=10, fem_grid=8, verbose=False)
    texts = list(iter_texts(f"flywheel:{out}"))
    assert len(texts) == 3 and "<scad>" in texts[0]


def test_unknown_source_raises():
    with pytest.raises(ValueError):
        list(iter_texts("magic:whatever"))


# ------------------------------------------------------------------ тренер
def test_train_lm_creates_new_versions(tmp_path):
    from nexus.training.train_lm import train
    root = str(tmp_path / "registry")
    first = train("builtin:engineering", "core", "tiny", seq_len=64, registry_root=root,
                  max_steps=2, batch_size=2, grad_accum=1, log_every=100)
    second = train("builtin:engineering", "core", "tiny", resume="latest", seq_len=64,
                   registry_root=root, max_steps=2, batch_size=2, grad_accum=1, log_every=100)
    reg = ModelRegistry(root)
    assert reg.versions("core") == [1, 2]
    assert second["version"]["parent"] == 1
    assert first["metrics"]["val_loss"] > 0
    assert reg.get("core", "latest").dataset == "builtin:engineering"


def test_trainer_early_stopping_and_eval(tmp_path):
    from nexus.data.corpora import PackedLMDataset
    from nexus.training.trainer import TrainConfig, Trainer
    ds = PackedLMDataset("builtin:engineering", seq_len=32, min_blocks=8)
    train_ds, val_ds = split_dataset(ds, 0.25)
    cfg = TrainConfig(max_steps=4, batch_size=2, grad_accum=1, eval_every=2, patience=1,
                      log_every=100, registry_root=str(tmp_path))
    tr = Trainer(NexusEngine(NexusConfig.tiny()), cfg, train_ds, val_ds)
    metrics = tr.fit()
    assert "val_loss" in metrics and metrics["steps"] <= 4
    mv = tr.save(metrics)
    assert mv.stage == "train" and mv.metrics["val_loss"] == metrics["val_loss"]


# --------------------------------------------------------------- дистилляция
def _teacher(tmp_path):
    from nexus.training.distill import NexusTeacher
    cfg = NexusConfig.tiny()
    return NexusTeacher(NexusEngine(cfg)), cfg


def test_distill_logit_mode(tmp_path):
    from nexus.training.distill import DistillConfig, distill
    teacher, _ = _teacher(tmp_path)
    out = distill(teacher, "builtin:engineering", "student", "tiny",
                  DistillConfig(temperature=2.0, alpha=0.7, mode="logit"),
                  seq_len=64, registry_root=str(tmp_path / "reg"),
                  max_steps=2, batch_size=2, grad_accum=1, log_every=100)
    assert out["version"]["stage"] == "distill:logit"
    assert out["metrics"]["val_loss"] > 0


def test_distill_cached_mode_needs_no_teacher_after_caching(tmp_path):
    from nexus.data.corpora import PackedLMDataset
    from nexus.training.distill import (CachedLogitsDataset, DistillConfig,
                                        cache_teacher_logits, distill)
    teacher, _ = _teacher(tmp_path)
    ds = PackedLMDataset("builtin:engineering", seq_len=32, min_blocks=4)
    cache = str(tmp_path / "cache.npz")
    cache_teacher_logits(teacher, ds, cache, top_k=8, batch_size=2)
    cached = CachedLogitsDataset(cache)
    assert len(cached) == len(ds)
    assert cached[0]["topk_values"].shape[-1] == 8

    out = distill(None, "builtin:engineering", "student-cached", "tiny",
                  DistillConfig(mode="cached", top_k=8), seq_len=32,
                  cache_path=cache, registry_root=str(tmp_path / "reg"),
                  max_steps=2, batch_size=2, grad_accum=1, log_every=100)
    assert out["version"]["stage"] == "distill:cached"


def test_distill_sequence_mode_from_teacher_corpus(tmp_path):
    from nexus.training.distill import DistillConfig, build_sequence_corpus, distill
    teacher, _ = _teacher(tmp_path)
    corpus = build_sequence_corpus(teacher, ["<task>кронштейн", "<task>фланец"],
                                   str(tmp_path / "seq.jsonl"), max_new_tokens=8)
    assert os.path.exists(corpus)
    out = distill(teacher, f"jsonl:{corpus}#text", "student-seq", "tiny",
                  DistillConfig(mode="sequence"), seq_len=32,
                  registry_root=str(tmp_path / "reg"),
                  max_steps=1, batch_size=1, grad_accum=1, log_every=100)
    assert out["version"]["stage"] == "distill:sequence"


def test_kd_loss_zero_when_student_equals_teacher():
    from nexus.training.distill import DistillConfig, kd_loss
    torch.manual_seed(0)
    logits = torch.randn(2, 6, 32)
    targets = torch.randint(1, 32, (2, 6))
    stats = kd_loss(logits, logits.clone(), targets, DistillConfig(alpha=1.0))
    assert float(stats["kd"]) < 1e-5


# ------------------------------------------------------- приёмочные тесты
def test_suite_detects_broken_model(tmp_path):
    from nexus.eval.suite import SuiteThresholds, run_suite
    model = NexusEngine(NexusConfig.tiny())
    good = run_suite(model, "core", 1, seq_len=32, n_scad_samples=1,
                     thresholds=SuiteThresholds(max_val_ppl=1e9, min_unique_ratio=0.0))
    assert good.passed and "val_ppl" in good.metrics

    strict = run_suite(model, "core", 1, seq_len=32, n_scad_samples=1,
                       thresholds=SuiteThresholds(max_val_ppl=1.0))
    assert not strict.passed
    assert any(c.name == "val_ppl" and not c.passed for c in strict.checks)


def test_gate_promotes_only_when_passing(tmp_path):
    from nexus.eval.suite import SuiteThresholds, evaluate_version
    from nexus.training.train_lm import train
    root = str(tmp_path / "registry")
    train("builtin:engineering", "core", "tiny", seq_len=64, registry_root=root,
          max_steps=2, batch_size=2, grad_accum=1, log_every=100)
    reg = ModelRegistry(root)

    bad = evaluate_version("core", "latest", registry_root=root, seq_len=32,
                           thresholds=SuiteThresholds(max_val_ppl=1.0), gate=True)
    assert not bad.passed and "production" not in reg.tags("core")

    ok = evaluate_version("core", "latest", registry_root=root, seq_len=32,
                          thresholds=SuiteThresholds(max_val_ppl=1e9, min_unique_ratio=0.0),
                          gate=True)
    assert ok.passed and reg.tags("core")["production"] == 1


def test_regression_check_against_baseline(tmp_path):
    from nexus.eval.suite import SuiteThresholds, run_suite
    model = NexusEngine(NexusConfig.tiny())
    baseline = {"val_ppl": 1.0}
    rep = run_suite(model, "core", 2, seq_len=32, n_scad_samples=1, baseline=baseline,
                    thresholds=SuiteThresholds(max_val_ppl=1e9, min_unique_ratio=0.0))
    reg_check = [c for c in rep.checks if c.name == "regression"][0]
    assert not reg_check.passed          # необученная модель хуже базовой


# ────────────────────────────────────── защита от переобучения и сравнение
def test_trainer_restores_best_weights_and_reports_gap(tmp_path):
    import torch

    from nexus.config import NexusConfig
    from nexus.data.corpora import PackedLMDataset, split_dataset
    from nexus.model import NexusEngine
    from nexus.training.trainer import TrainConfig, Trainer

    ds = PackedLMDataset("builtin:engineering", seq_len=32, min_blocks=8)
    train_ds, val_ds = split_dataset(ds, 0.25)
    cfg = TrainConfig(max_steps=6, batch_size=2, grad_accum=1, eval_every=2,
                      log_every=100, keep_best=True, registry_root=str(tmp_path))
    tr = Trainer(NexusEngine(NexusConfig.tiny()), cfg, train_ds, val_ds)
    metrics = tr.fit()
    assert tr.best_step >= 0 and tr._best_state is not None
    assert "overfit_gap" in metrics and "train_ce" in metrics
    assert metrics["val_loss"] <= tr.best_val + 1e-3      # веса не хуже лучших


def test_train_lm_accepts_separate_validation_source(tmp_path):
    from nexus.data.mathgen import write_jsonl
    from nexus.training.train_lm import train
    holdout = str(tmp_path / "holdout.jsonl")
    write_jsonl(20, holdout, seed=99)
    out = train("builtin:engineering", "core", "tiny", seq_len=64,
                val_source=f"jsonl:{holdout}#text", registry_root=str(tmp_path / "reg"),
                max_steps=2, batch_size=2, grad_accum=1, eval_every=2, log_every=100)
    assert out["metrics"]["val_loss"] > 0


def test_suite_flags_overfitting(tmp_path):
    from nexus.config import NexusConfig
    from nexus.eval.suite import SuiteThresholds, run_suite
    from nexus.model import NexusEngine
    model = NexusEngine(NexusConfig.tiny())
    th = SuiteThresholds(max_val_ppl=1e9, min_unique_ratio=0.0, max_overfit_gap=0.1)
    # искусственно «идеальная» обучающая выборка → большой разрыв
    rep = run_suite(model, "core", 1, seq_len=32, n_scad_samples=1, thresholds=th,
                    train_metrics={"train_ce": 0.01})
    check = [c for c in rep.checks if c.name == "overfit_gap"][0]
    assert not check.passed and rep.metrics["overfit_gap"] > 0.1


def test_max_steps_wins_over_epochs(tmp_path):
    """Бюджет шагов должен выбираться полностью, даже если корпус меньше."""
    from nexus.training.train_lm import train
    out = train("mathgen:40", "core", "tiny", seq_len=64,
                registry_root=str(tmp_path / "reg"), max_steps=120, batch_size=4,
                grad_accum=1, log_every=1000, eval_every=0)
    assert out["metrics"]["steps"] == 120       # раньше обрывалось на первой эпохе


def test_data_budget_report_warns_on_small_corpus(tmp_path, capsys):
    from nexus.config import NexusConfig
    from nexus.data.corpora import PackedLMDataset
    from nexus.model import NexusEngine
    from nexus.training.trainer import TrainConfig, Trainer
    ds = PackedLMDataset("builtin:engineering", seq_len=64, min_blocks=4)
    cfg = TrainConfig(max_steps=4, batch_size=2, grad_accum=1, log_every=100,
                      registry_root=str(tmp_path))
    Trainer(NexusEngine(NexusConfig.tiny()), cfg, ds).fit()
    printed = capsys.readouterr().out
    assert "токенов на параметр" in printed
    assert "ВНИМАНИЕ" in printed                # корпус меньше числа параметров
