#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════
  NEXUS-Capital — ВЕРИФИКАЦИЯ СИСТЕМЫ
═══════════════════════════════════════════════════════════════════════════

Одна команда проверяет и показывает, что всё работает:
    python verify.py            # полная проверка
    python verify.py --quick    # быстрая (без обучения, ~10 c)
    python verify.py --train    # + короткое обучение (~30 c)

Печатает читаемый отчёт по этапам и пишет лог в verify_report.log.
Присылайте этот лог — по нему сразу видно, где всё хорошо/плохо.
"""
from __future__ import annotations

import argparse
import io
import os
import platform
import sys
import time
import traceback
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime
from pathlib import Path

# Запускаемся из любой директории
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("NEXUS_OFFLINE_TOKENIZER", "1")  # надёжно в CI/песочнице

LOG_PATH = ROOT / "verify_report.log"
_log_buf = io.StringIO()


class Tee:
    """Дублирует вывод и в консоль, и в буфер лога."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            try:
                s.write(data)
                s.flush()
            except Exception:
                pass

    def flush(self):
        for s in self.streams:
            try:
                s.flush()
            except Exception:
                pass


def section(title: str) -> None:
    line = "─" * 70
    print(f"\n{line}\n  {title}\n{line}")


def ok(msg: str) -> None:
    print(f"  ✅ {msg}")


def warn(msg: str) -> None:
    print(f"  ⚠️  {msg}")


def fail(msg: str) -> None:
    print(f"  ❌ {msg}")


def info(msg: str) -> None:
    print(f"  ℹ️  {msg}")


def timed(fn):
    def wrapper(*a, **kw):
        t0 = time.time()
        result = fn(*a, **kw)
        return result, (time.time() - t0)
    return wrapper


# ─── Этапы ─────────────────────────────────────────────────────────────
def step_environment():
    section("0/6  Окружение и зависимости")
    info(f"Python   : {platform.python_version()}  ({sys.executable})")
    info(f"Platform : {platform.platform()}")
    info(f"CPU cores: {os.cpu_count()}")
    info(f"Time     : {datetime.now().isoformat(timespec='seconds')}")

    deps = {}
    for mod in ("torch", "numpy", "pandas", "pyarrow", "tokenizers",
                "huggingface_hub", "yaml", "fastapi", "uvicorn"):
        try:
            m = __import__(mod)
            deps[mod] = getattr(m, "__version__", "?")
            ok(f"{mod:18s} {deps[mod]}")
        except Exception as e:
            fail(f"{mod:18s} НЕ установлен: {e}")
            deps[mod] = None

    import torch
    if torch.cuda.is_available():
        ok(f"CUDA доступна: {torch.cuda.get_device_name(0)}")
    else:
        info("CUDA недоступна — работаем на CPU")
    return deps


def step_tokenizer():
    section("1/6  Токенизатор (RU+EN)")
    from nexus_capital import NexusTokenizer
    t0 = time.time()
    tok = NexusTokenizer.default(prefer_offline=True)
    ok(f"загружен '{tok.name}', vocab={tok.vocab_size} "
       f"за {time.time()-t0:.2f}c")

    samples = [
        "Revenue grew 20% in Q1 to $1,234.5M.",
        "Выручка выросла на 20 процентов, чистая прибыль 1,2 млрд руб.",
        "Net income $(120.4) million, gross profit 4,521.3.",
    ]
    for s in samples:
        e = tok.encode(s, max_length=64)
        nums = [round(v, 2) for v in e.number_values if v != 0.0]
        ok(f"  [{len(e.ids):2d} токенов] числа={nums}  «{s[:48]}…»")
    return True


def step_model_build():
    section("2/6  Сборка модели и форвард-проход")
    from nexus_capital import small_config, build_model
    cfg = small_config()
    cfg.mc_paths = 64
    cfg.mc_horizon = 8
    cfg.orderbook_levels = 16
    cfg.tabular_features = 16
    model = build_model(cfg)
    n = model.count_parameters()
    ok(f"модель собрана: {n/1e6:.2f}M параметров, d_value={cfg.d_value}")

    import torch
    import numpy as np
    from nexus_capital.data.orderbook_stream import collate_book_ticks
    from nexus_capital.data.edgar_parser import (
        synthetic_filing, fields_to_tensor, DEFAULT_UNIT_FIELDS)
    rng = np.random.default_rng(0)
    bt = collate_book_ticks(2, n_levels=16, T=16, n_features=8)
    fs, ms = [], []
    for _ in range(2):
        f, m = fields_to_tensor(synthetic_filing(rng), DEFAULT_UNIT_FIELDS[:16])
        fs.append(f); ms.append(m)
    batch = dict(
        book=bt["book"], ticks=bt["ticks"],
        fields=torch.stack(fs), field_mask=torch.stack(ms),
        tokens=torch.randint(0, cfg.vocab_size, (2, 16)),
    )
    model.eval()
    with torch.no_grad():
        out = model(**batch, workspace_mode="risk")

    r = out["workspace"]["risk"]
    ok(f"risk: E[P&L]={float(r['expected_pnl'].mean()):+.4f}  "
       f"σ={float(r['std_pnl'].mean()):.4f}  "
       f"VaR5={float(r['var'].mean()):+.4f}  "
       f"calibrated={r['calibrated']}")
    eco = out["economic"]
    ok(f"portfolio_weights sum={float(eco['portfolio_weights'].sum()):.4f}, "
       f"PD mean={float(eco['default_prob'].mean()):.4f}")

    # Проверка режимов pricing/negotiation
    with torch.no_grad():
        cp = torch.tensor([99.0, 149.0])
        op = model(**batch, workspace_mode="pricing", competitor_price=cp)
        price = op["workspace"]["optimal_price"]
        ok(f"pricing: конкуренты={cp.tolist()} → opt_price="
           f"{[round(float(x),2) for x in price]}")
        ng = model(**batch, workspace_mode="negotiation")
        sh = ng["workspace"]["negotiation"]
        ok(f"negotiation: share_a="
           f"{[round(float(x),2) for x in sh['share_a']]}")
    return model, cfg, batch


def step_backward(model, batch):
    section("3/6  Обратный проход и градиенты")
    import torch
    model.train()
    batch = dict(batch)
    batch["target_tokens"] = torch.randint(0, model.cfg.vocab_size, (2, 1))
    out = model(**batch)
    loss = out["loss"]
    loss.backward()
    n_grads = sum(1 for p in model.parameters() if p.grad is not None)
    gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1e9)
    ok(f"loss={float(loss):.4f}, тензоров с градиентом={n_grads}, "
       f"‖grad‖={float(gnorm):.3f}")
    assert torch.isfinite(loss), "loss не конечный!"
    assert n_grads > 0, "градиенты не текут!"
    return True


def step_calibration():
    section("4/6  Калибровка Neural SDE и бейзлайны")
    import numpy as np
    import torch
    from nexus_capital.workspace.neural_sde import NeuralSDE
    from nexus_capital.training.baselines import all_baselines

    # Синтетические возвраты с волатильностью 1.5% и толстыми хвостами
    rng = np.random.default_rng(7)
    rets = rng.standard_t(df=5, size=5000).astype(np.float32) * 0.015
    sde = NeuralSDE(8, d_hidden=32)
    cal = sde.calibrate_to_returns(torch.from_numpy(rets).unsqueeze(-1))
    ok(f"калибровка: mean={cal['emp_mean']:+.6f}  "
       f"std={cal['emp_std']:.5f}  df={cal['df']:.1f}  "
       f"calibrated={cal['calibrated']}")

    bl = all_baselines(rets)
    for b in bl:
        print("  baseline {:5s}  MSE={:.3e}  sign_acc={:.3f}  Sharpe={:.2f}"
              .format(b['name'], b['mse'], b['sign_acc'], b['sharpe']))
    return True


def step_tests():
    section("5/6  Модульные тесты")
    import subprocess
    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-q", "-W", "ignore",
         "--tb=short"],
        cwd=ROOT, capture_output=True, text=True,
    )
    dt = time.time() - t0
    print(proc.stdout[-2000:])
    if proc.returncode == 0:
        ok(f"все тесты прошли за {dt:.1f}c")
    else:
        fail(f"тесты упали (код {proc.returncode}) за {dt:.1f}c")
        print(proc.stderr[-1500:])
    return proc.returncode == 0


def step_train_smoke():
    section("6/6  Короткое обучение (рыночный этап)")
    import subprocess
    cmd = [
        sys.executable, "scripts/train_real_data.py",
        "--steps", "5", "--text-steps", "3",
        "--batch-size", "2", "--text-batch-size", "4",
        "--mc-paths", "16", "--seq-len", "16",
        "--max-samples", "200",
        "--start-month", "2024-01", "--end-month", "2024-01",
        "--offline-tokenizer",
    ]
    info(" ".join(cmd))
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    dt = time.time() - t0
    out = proc.stdout + proc.stderr
    # Покажем ключевые строки
    for line in out.splitlines():
        key = any(k in line for k in (
            "Бейзлайн", "zero", "mean", "ar1", "SDE откалиброван",
            "корпус", "text step", "step ", "mse", "var_breach",
            "Чекпоинт", "Итог", "perplexity", "train=", "val="))
        if key:
            print("    " + line)
    if proc.returncode == 0:
        ok(f"обучение завершилось за {dt:.1f}c")
    else:
        fail(f"обучение упало (код {proc.returncode}) за {dt:.1f}c")
        print(out[-2000:])
    return proc.returncode == 0


# ─── Главный запуск ────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Верификация NEXUS-Capital")
    ap.add_argument("--quick", action="store_true",
                    help="только окружение/модель/тесты, без обучения")
    ap.add_argument("--train", action="store_true",
                    help="включить короткое обучение")
    args = ap.parse_args()

    if args.quick:
        do_train = False
    elif args.train:
        do_train = True
    else:
        do_train = True  # по умолчанию полная проверка с коротким обучением

    sys.stdout = Tee(sys.__stdout__, _log_buf)
    sys.stderr = Tee(sys.__stderr__, _log_buf)

    print("╔" + "═" * 68 + "╗")
    print("║   NEXUS-Capital  —  verify report" + " " * 33 + "║")
    print("╚" + "═" * 68 + "╝")

    results = {}
    try:
        step_environment()
        results["tokenizer"] = _safe(step_tokenizer)
        model, cfg, batch = (None, None, None)
        try:
            model, cfg, batch = step_model_build()
            results["model_build"] = True
        except Exception:
            results["model_build"] = False
            traceback.print_exc()
        if model is not None:
            results["backward"] = _safe(step_backward, model, batch)
        results["calibration"] = _safe(step_calibration)
        results["tests"] = _safe(step_tests)
        if do_train:
            results["train_smoke"] = _safe(step_train_smoke)
    except Exception:
        traceback.print_exc()

    section("ИТОГ")
    for k, v in results.items():
        print(f"  {'✅' if v else '❌'}  {k}")
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    print(f"\n  Пройдено {passed}/{total} этапов")

    LOG_PATH.write_text(_log_buf.getvalue(), encoding="utf-8")
    print(f"\n  Лог сохранён: {LOG_PATH}")
    return 0 if passed == total else 1


def _safe(fn, *args):
    try:
        return bool(fn(*args))
    except Exception:
        traceback.print_exc()
        return False


if __name__ == "__main__":
    sys.exit(main())
