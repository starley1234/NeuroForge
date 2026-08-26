"""
Простые бейзлайны и честная валидация прогноза доходности.

Без сравнения с наивными моделями нельзя утверждать, что NEXUS-Capital что-то
выигрывает. Здесь:
  • zero / always-mean baseline;
  • AR(1) на лог-доходностях;
  • volatility tracking (для сравнения VaR с SDE).

И валидация VaR по частоте пробитий (Basel traffic-light): доля исходов
ниже прогнозного VaR должна быть около α.
"""
from __future__ import annotations

import numpy as np


def zero_baseline(returns: np.ndarray) -> dict:
    """Прогноз 0 на каждый следующий бар."""
    pred = np.zeros_like(returns)
    return _regression_metrics(returns, pred, name="zero")


def mean_baseline(returns: np.ndarray, window: int = 50) -> dict:
    """Скользящее среднее последних `window` баров."""
    pred = np.zeros_like(returns)
    for t in range(window, len(returns)):
        pred[t] = returns[t - window:t].mean()
    return _regression_metrics(returns[window:], pred[window:], name="mean")


def ar1_baseline(returns: np.ndarray, window: int = 100) -> dict:
    """Подгонка AR(1) скользящим окном: r_t = c + phi * r_{t-1} + eps."""
    pred = np.zeros_like(returns)
    for t in range(window, len(returns)):
        hist = returns[t - window:t]
        x = hist[:-1]
        y = hist[1:]
        x_mean, y_mean = x.mean(), y.mean()
        var = ((x - x_mean) ** 2).mean()
        if var > 1e-12:
            phi = ((x - x_mean) * (y - y_mean)).mean() / var
            c = y_mean - phi * x_mean
            pred[t] = c + phi * returns[t - 1]
        else:
            pred[t] = y_mean
    return _regression_metrics(returns[window:], pred[window:], name="ar1")


def _regression_metrics(y: np.ndarray, pred: np.ndarray, name: str) -> dict:
    y = np.asarray(y, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    mse = float(np.mean((y - pred) ** 2))
    mae = float(np.mean(np.abs(y - pred)))
    # Направление (sign accuracy)
    sign_acc = float(np.mean(np.sign(y) == np.sign(pred)))
    # Шарп предсказания (среднее/ст предсказания — proxy)
    sharpe = float(pred.mean() / (pred.std() + 1e-9))
    return {"name": name, "mse": mse, "mae": mae,
            "sign_acc": sign_acc, "sharpe": sharpe}


def var_coverage(realized_pnl: np.ndarray, var_forecast: float,
                 alpha: float = 0.05) -> dict:
    """
    Проверка калибровки VaR: доля реализаций ниже -VaR должна быть ≈ α.
    realized_pnl: (T,) реализованный P&L
    var_forecast: положительный VaR (убыток)
    """
    breaches = (realized_pnl < -var_forecast).mean()
    return {
        "expected_alpha": alpha,
        "observed_breach_rate": float(breaches),
        "calibrated": bool(abs(breaches - alpha) < 0.02),
    }


def all_baselines(returns: np.ndarray) -> list[dict]:
    out = [zero_baseline(returns), mean_baseline(returns),
           ar1_baseline(returns)]
    return out
