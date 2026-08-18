"""Ограничения и валидация входных данных для публичного API.

Смысл: сервис принимает произвольный код и числа от неизвестных клиентов.
Без ограничений один запрос с `grid: 512` съедает всю память, а скрипт на
мегабайт вешает воксeлизатор. Здесь собраны все пределы в одном месте — их
видно, их легко поменять, и они одинаково применяются в API, MCP и CLI-сервисе.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


class ValidationError(ValueError):
    """Некорректный вход: наверх уходит как HTTP 400 с понятным текстом."""


@dataclass(frozen=True)
class Limits:
    max_body_bytes: int = 2 * 1024 * 1024      # 2 МБ на запрос
    max_code_chars: int = 200_000              # ~200 КБ исходника OpenSCAD
    max_prompt_chars: int = 8_000
    max_new_tokens: int = 2_048
    max_batch_prompts: int = 32
    max_grid: int = 64                         # 64³ вокселей = предел разумного
    min_grid: int = 6
    max_force_n: float = 1e7                   # 10 МН — заведомо больше любой детали
    max_required_sf: float = 100.0
    max_concurrency: int = 4                   # тяжёлых операций одновременно
    request_timeout_s: float = 120.0

    @classmethod
    def from_env(cls, env: Optional[Dict[str, str]] = None) -> "Limits":
        import os
        env = env if env is not None else dict(os.environ)
        values: Dict[str, Any] = {}
        for field_name, field_type in (("max_body_bytes", int), ("max_code_chars", int),
                                       ("max_prompt_chars", int), ("max_new_tokens", int),
                                       ("max_batch_prompts", int), ("max_grid", int),
                                       ("max_force_n", float), ("max_concurrency", int),
                                       ("request_timeout_s", float)):
            raw = env.get("NEXUS_" + field_name.upper())
            if raw:
                try:
                    values[field_name] = field_type(raw)
                except ValueError:
                    continue
        return cls(**values)


DEFAULT_LIMITS = Limits()


# ------------------------------------------------------------- проверки
def check_code(code: Any, limits: Limits = DEFAULT_LIMITS) -> str:
    if not isinstance(code, str) or not code.strip():
        raise ValidationError("поле code должно быть непустой строкой с кодом OpenSCAD")
    if len(code) > limits.max_code_chars:
        raise ValidationError(
            f"код слишком длинный: {len(code)} символов, максимум {limits.max_code_chars}")
    return code


def check_prompt(prompt: Any, limits: Limits = DEFAULT_LIMITS) -> str:
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValidationError("поле prompt должно быть непустой строкой")
    if len(prompt) > limits.max_prompt_chars:
        raise ValidationError(
            f"запрос слишком длинный: {len(prompt)} символов, "
            f"максимум {limits.max_prompt_chars}")
    return prompt


def check_tokens(value: Any, limits: Limits = DEFAULT_LIMITS, default: int = 64) -> int:
    if value is None:
        return default
    try:
        n = int(value)
    except (TypeError, ValueError):
        raise ValidationError("max_new_tokens должно быть целым числом")
    if n < 1:
        raise ValidationError("max_new_tokens должно быть положительным")
    return min(n, limits.max_new_tokens)


def check_grid(value: Any, limits: Limits = DEFAULT_LIMITS, default: int = 24) -> int:
    if value is None:
        return default
    try:
        grid = int(value)
    except (TypeError, ValueError):
        raise ValidationError("grid должно быть целым числом")
    if grid < limits.min_grid or grid > limits.max_grid:
        raise ValidationError(
            f"grid вне диапазона {limits.min_grid}..{limits.max_grid} "
            f"(получено {grid}); большие сетки съедают память")
    return grid


def check_force(value: Any, limits: Limits = DEFAULT_LIMITS,
                default: Tuple[float, float, float] = (0.0, 0.0, -200.0)
                ) -> Tuple[float, float, float]:
    if value is None:
        return default
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValidationError("force должно быть массивом из трёх чисел [Fx, Fy, Fz]")
    out: List[float] = []
    for component in value:
        try:
            f = float(component)
        except (TypeError, ValueError):
            raise ValidationError("компоненты force должны быть числами")
        if not (-limits.max_force_n <= f <= limits.max_force_n) or f != f:
            raise ValidationError(
                f"компонента силы вне разумных пределов (±{limits.max_force_n:.0f} Н)")
        out.append(f)
    return (out[0], out[1], out[2])


def check_choice(value: Any, allowed: Sequence[str], name: str, default: str) -> str:
    if value is None:
        return default
    if value not in allowed:
        raise ValidationError(f"{name}: допустимо {', '.join(allowed)} (получено {value!r})")
    return str(value)


def check_number(value: Any, name: str, default: float, low: float, high: float) -> float:
    if value is None:
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValidationError(f"{name} должно быть числом")
    if number != number or not (low <= number <= high):
        raise ValidationError(f"{name} вне диапазона {low}..{high}")
    return number


def check_prompts(value: Any, limits: Limits = DEFAULT_LIMITS) -> List[str]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValidationError("поле prompts должно быть непустым массивом строк")
    if len(value) > limits.max_batch_prompts:
        raise ValidationError(
            f"в батче {len(value)} запросов, максимум {limits.max_batch_prompts}")
    return [check_prompt(p, limits) for p in value]


def unknown_fields(payload: Dict[str, Any], allowed: Iterable[str]) -> List[str]:
    """Лишние поля — почти всегда опечатка клиента, лучше сказать сразу."""
    return sorted(set(payload) - set(allowed))
