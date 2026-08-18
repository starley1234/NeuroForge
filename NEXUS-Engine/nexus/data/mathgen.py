"""Генератор верифицированной инженерной математики.

Зачем не брать готовые математические датасеты целиком: модели нужны не
олимпиадные задачи, а счёт, который она будет делать в проектировании —
изгиб балки, момент затяжки, тепловое расширение, размерные цепи, масса и
инерция. Ответ здесь вычисляется формулой (а не другой моделью), поэтому
метки верны по построению и датасет можно масштабировать бесплатно.

Формат сэмпла — пошаговое решение с явным ответом:

    <task>Балка ... Найти σ_max.
    Дано: F = 320 Н, L = 85 мм ...
    Решение:
      1. M = F·L = 27.2 Н·м
      ...
    Ответ: σ_max = 45.9 МПа
"""
from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

MATERIALS = {
    "PLA": dict(density=1240.0, yield_mpa=50.0, young_gpa=3.5, alpha=68e-6),
    "PETG": dict(density=1270.0, yield_mpa=53.0, young_gpa=2.1, alpha=60e-6),
    "ABS": dict(density=1040.0, yield_mpa=40.0, young_gpa=2.3, alpha=90e-6),
    "алюминий 6061": dict(density=2700.0, yield_mpa=276.0, young_gpa=68.9, alpha=23.6e-6),
    "сталь 304": dict(density=8000.0, yield_mpa=215.0, young_gpa=193.0, alpha=17.3e-6),
    "титан Ti-6Al-4V": dict(density=4430.0, yield_mpa=880.0, young_gpa=113.8, alpha=8.6e-6),
}

BOLTS = {  # M, шаг, площадь сечения по напряжениям (мм²), класс 8.8 (МПа)
    "M3": (0.5, 5.03), "M4": (0.7, 8.78), "M5": (0.8, 14.2),
    "M6": (1.0, 20.1), "M8": (1.25, 36.6), "M10": (1.5, 58.0), "M12": (1.75, 84.3),
}


@dataclass
class MathSample:
    kind: str
    question: str
    solution: str
    answer: float
    unit: str
    tolerance: float = 0.02

    @property
    def text(self) -> str:
        return (f"<task>{self.question}\nРешение:\n{self.solution}\n"
                f"Ответ: {self.answer:.4g} {self.unit}")

    def to_dict(self) -> Dict[str, object]:
        return {"kind": self.kind, "question": self.question, "solution": self.solution,
                "answer": self.answer, "unit": self.unit, "text": self.text}

    def check(self, value: float) -> bool:
        return abs(value - self.answer) <= self.tolerance * max(abs(self.answer), 1e-9)


def _r(rng: random.Random, a: float, b: float, digits: int = 1) -> float:
    return round(rng.uniform(a, b), digits)


# ────────────────────────────────────────────────────────────── генераторы
def cantilever_bending(rng: random.Random) -> MathSample:
    """Консольная балка прямоугольного сечения: σ = 6FL/(bh²)."""
    f = _r(rng, 50, 900)
    length = _r(rng, 30, 200)
    b = _r(rng, 8, 40)
    h = _r(rng, 3, 25)
    mat = rng.choice(list(MATERIALS))
    sigma = 6 * f * (length * 1e-3) / (b * 1e-3 * (h * 1e-3) ** 2) / 1e6
    sf = MATERIALS[mat]["yield_mpa"] / sigma
    return MathSample(
        "cantilever_bending",
        f"Консольная балка из материала «{mat}» прямоугольного сечения b×h = "
        f"{b}×{h} мм, вылет L = {length} мм, на конце сила F = {f} Н. "
        f"Найти максимальное нормальное напряжение σ_max.",
        f"  1. Изгибающий момент в заделке: M = F·L = {f}·{length}e-3 = {f * length * 1e-3:.4g} Н·м\n"
        f"  2. Момент сопротивления: W = b·h²/6 = {b}e-3·({h}e-3)²/6 = {b * 1e-3 * (h * 1e-3) ** 2 / 6:.4g} м³\n"
        f"  3. σ_max = M/W = {sigma:.4g} МПа\n"
        f"  4. Запас по пределу текучести ({MATERIALS[mat]['yield_mpa']} МПа): n = {sf:.3g}",
        sigma, "МПа")


def cantilever_deflection(rng: random.Random) -> MathSample:
    """Прогиб консоли: δ = FL³/(3EI)."""
    f = _r(rng, 20, 500)
    length = _r(rng, 40, 250)
    b = _r(rng, 10, 40)
    h = _r(rng, 4, 20)
    mat = rng.choice(list(MATERIALS))
    e = MATERIALS[mat]["young_gpa"] * 1e9
    inertia = b * 1e-3 * (h * 1e-3) ** 3 / 12
    delta = f * (length * 1e-3) ** 3 / (3 * e * inertia) * 1000
    return MathSample(
        "cantilever_deflection",
        f"Консоль из «{mat}» сечением {b}×{h} мм и вылетом {length} мм нагружена "
        f"силой {f} Н на конце. Найти прогиб δ.",
        f"  1. Момент инерции: I = b·h³/12 = {inertia:.4g} м⁴\n"
        f"  2. E = {MATERIALS[mat]['young_gpa']} ГПа\n"
        f"  3. δ = F·L³/(3·E·I) = {delta:.4g} мм",
        delta, "мм")


def bolt_torque(rng: random.Random) -> MathSample:
    """Момент затяжки: T = K·F·d."""
    size = rng.choice(list(BOLTS))
    d = float(size[1:])
    k = rng.choice([0.14, 0.16, 0.18, 0.20])
    preload = _r(rng, 2000, 20000, 0)
    torque = k * preload * d * 1e-3
    return MathSample(
        "bolt_torque",
        f"Болт {size} затягивают с усилием предварительной затяжки F = {preload:.0f} Н "
        f"при коэффициенте трения K = {k}. Найти момент затяжки T.",
        f"  1. T = K·F·d\n"
        f"  2. T = {k}·{preload:.0f}·{d}e-3\n"
        f"  3. T = {torque:.4g} Н·м",
        torque, "Н·м")


def bolt_stress(rng: random.Random) -> MathSample:
    """Напряжение в болте: σ = F/A_s."""
    size = rng.choice(list(BOLTS))
    area = BOLTS[size][1]
    force = _r(rng, 500, 30000, 0)
    sigma = force / area
    return MathSample(
        "bolt_stress",
        f"На болт {size} (площадь сечения по напряжениям A_s = {area} мм²) действует "
        f"растягивающая сила {force:.0f} Н. Найти напряжение растяжения σ.",
        f"  1. σ = F/A_s = {force:.0f}/{area}\n"
        f"  2. σ = {sigma:.4g} МПа\n"
        f"  3. Для класса 8.8 (σ_т = 640 МПа) запас n = {640 / sigma:.3g}",
        sigma, "МПа")


def thermal_expansion(rng: random.Random) -> MathSample:
    mat = rng.choice(list(MATERIALS))
    length = _r(rng, 20, 500)
    dt = _r(rng, 10, 150)
    alpha = MATERIALS[mat]["alpha"]
    dl = alpha * length * dt
    return MathSample(
        "thermal_expansion",
        f"Деталь из «{mat}» длиной {length} мм нагревается на ΔT = {dt} °C. "
        f"Коэффициент линейного расширения α = {alpha:.3g} 1/К. Найти удлинение ΔL.",
        f"  1. ΔL = α·L·ΔT = {alpha:.3g}·{length}·{dt}\n"
        f"  2. ΔL = {dl:.4g} мм",
        dl, "мм")


def mass_of_part(rng: random.Random) -> MathSample:
    mat = rng.choice(list(MATERIALS))
    a, b, c = _r(rng, 10, 120), _r(rng, 10, 90), _r(rng, 2, 40)
    hole_d = _r(rng, 3, min(a, b) / 3)
    density = MATERIALS[mat]["density"]
    volume = a * b * c - math.pi * (hole_d / 2) ** 2 * c
    mass = volume * 1e-9 * density * 1000
    return MathSample(
        "mass_of_part",
        f"Плита из «{mat}» {a}×{b}×{c} мм со сквозным отверстием Ø{hole_d} мм. "
        f"Плотность {density} кг/м³. Найти массу детали.",
        f"  1. V_плиты = {a}·{b}·{c} = {a * b * c:.6g} мм³\n"
        f"  2. V_отв = π·(Ø/2)²·h = {math.pi * (hole_d / 2) ** 2 * c:.6g} мм³\n"
        f"  3. V = {volume:.6g} мм³ = {volume * 1e-9:.4g} м³\n"
        f"  4. m = ρ·V = {mass:.4g} г",
        mass, "г")


def second_moment(rng: random.Random) -> MathSample:
    b, h = _r(rng, 5, 60), _r(rng, 5, 60)
    inertia = b * h ** 3 / 12
    return MathSample(
        "second_moment",
        f"Прямоугольное сечение b×h = {b}×{h} мм. Найти осевой момент инерции "
        f"относительно центральной оси, параллельной стороне b.",
        f"  1. I = b·h³/12\n  2. I = {b}·{h}³/12 = {inertia:.6g} мм⁴",
        inertia, "мм⁴")


def tolerance_stack(rng: random.Random) -> MathSample:
    n = rng.randint(3, 6)
    nominals = [_r(rng, 5, 60) for _ in range(n)]
    tols = [_r(rng, 0.02, 0.2, 2) for _ in range(n)]
    worst = sum(tols)
    rss = math.sqrt(sum(t * t for t in tols))
    return MathSample(
        "tolerance_stack",
        "Размерная цепь из звеньев: " +
        ", ".join(f"{v}±{t}" for v, t in zip(nominals, tols)) +
        " мм. Найти суммарный допуск замыкающего звена по методу RSS "
        "(квадратичное суммирование).",
        f"  1. Метод максимума-минимума: T = Σ|tᵢ| = {worst:.4g} мм\n"
        f"  2. Метод RSS: T = √(Σtᵢ²) = {rss:.4g} мм\n"
        f"  3. Номинал замыкающего: {sum(nominals):.4g} мм",
        rss, "мм")


def print_time_estimate(rng: random.Random) -> MathSample:
    volume_cm3 = _r(rng, 5, 300)
    flow = _r(rng, 4, 18)          # мм³/с
    infill = rng.choice([0.15, 0.2, 0.25, 0.4])
    solid = volume_cm3 * 1000 * (0.35 + 0.65 * infill)
    hours = solid / flow / 3600
    return MathSample(
        "print_time",
        f"Деталь объёмом {volume_cm3} см³ печатается с заполнением {infill * 100:.0f}% "
        f"при производительности экструдера {flow} мм³/с. Оценить время печати "
        f"(доля материала: 35% периметры + заполнение по объёму).",
        f"  1. V_материала = {volume_cm3}·1000·(0.35 + 0.65·{infill}) = {solid:.6g} мм³\n"
        f"  2. t = V/Q = {solid / flow:.6g} с\n"
        f"  3. t = {hours:.4g} ч",
        hours, "ч")


def press_fit(rng: random.Random) -> MathSample:
    d = _r(rng, 10, 60)
    interference = _r(rng, 0.005, 0.06, 3)
    mat = rng.choice(["алюминий 6061", "сталь 304"])
    e = MATERIALS[mat]["young_gpa"] * 1e9
    pressure = e * interference / d / 1e6 / 2
    return MathSample(
        "press_fit",
        f"Посадка с натягом: вал Ø{d} мм, натяг δ = {interference} мм, "
        f"втулка из «{mat}» (E = {MATERIALS[mat]['young_gpa']} ГПа). "
        f"Оценить контактное давление p по упрощённой формуле p ≈ E·δ/(2·d).",
        f"  1. p = E·δ/(2·d) = {e:.3g}·{interference}e-3/(2·{d}e-3)\n"
        f"  2. p = {pressure:.4g} МПа",
        pressure, "МПа")


def gear_ratio(rng: random.Random) -> MathSample:
    z1, z2 = rng.randint(12, 40), rng.randint(20, 90)
    module = rng.choice([0.5, 0.8, 1.0, 1.25, 1.5, 2.0])
    torque_in = _r(rng, 0.2, 20)
    ratio = z2 / z1
    torque_out = torque_in * ratio
    center = module * (z1 + z2) / 2
    return MathSample(
        "gear_ratio",
        f"Зубчатая пара: z₁ = {z1}, z₂ = {z2}, модуль m = {module} мм, "
        f"момент на входе {torque_in} Н·м. Найти момент на выходе.",
        f"  1. Передаточное число: i = z₂/z₁ = {ratio:.4g}\n"
        f"  2. Межосевое расстояние: a = m·(z₁+z₂)/2 = {center:.4g} мм\n"
        f"  3. T₂ = T₁·i = {torque_out:.4g} Н·м",
        torque_out, "Н·м")


def buckling(rng: random.Random) -> MathSample:
    d = _r(rng, 4, 25)
    length = _r(rng, 50, 400)
    mat = rng.choice(["алюминий 6061", "сталь 304", "PLA"])
    e = MATERIALS[mat]["young_gpa"] * 1e9
    inertia = math.pi * (d * 1e-3) ** 4 / 64
    f_cr = math.pi ** 2 * e * inertia / (length * 1e-3) ** 2
    return MathSample(
        "buckling",
        f"Круглая стойка Ø{d} мм длиной {length} мм из «{mat}» шарнирно оперта с двух "
        f"концов. Найти критическую силу продольного изгиба по Эйлеру.",
        f"  1. I = π·d⁴/64 = {inertia:.4g} м⁴\n"
        f"  2. F_кр = π²·E·I/L² = π²·{e:.3g}·{inertia:.4g}/({length}e-3)²\n"
        f"  3. F_кр = {f_cr:.4g} Н",
        f_cr, "Н")


def shear_pin(rng: random.Random) -> MathSample:
    d = _r(rng, 3, 20)
    force = _r(rng, 200, 20000, 0)
    area = math.pi * (d / 2) ** 2
    tau = force / area
    return MathSample(
        "shear_pin",
        f"Штифт Ø{d} мм работает на срез по одной плоскости под нагрузкой {force:.0f} Н. "
        f"Найти касательное напряжение τ.",
        f"  1. A = π·d²/4 = {area:.4g} мм²\n"
        f"  2. τ = F/A = {tau:.4g} МПа",
        tau, "МПа")


GENERATORS: Dict[str, Callable[[random.Random], MathSample]] = {
    "cantilever_bending": cantilever_bending,
    "cantilever_deflection": cantilever_deflection,
    "bolt_torque": bolt_torque,
    "bolt_stress": bolt_stress,
    "thermal_expansion": thermal_expansion,
    "mass_of_part": mass_of_part,
    "second_moment": second_moment,
    "tolerance_stack": tolerance_stack,
    "print_time": print_time_estimate,
    "press_fit": press_fit,
    "gear_ratio": gear_ratio,
    "buckling": buckling,
    "shear_pin": shear_pin,
}


def generate(n: int, seed: int = 0, kinds: Optional[List[str]] = None) -> List[MathSample]:
    rng = random.Random(seed)
    names = kinds or list(GENERATORS)
    return [GENERATORS[rng.choice(names)](rng) for _ in range(n)]


def write_jsonl(n: int, path: str, seed: int = 0,
                kinds: Optional[List[str]] = None) -> Dict[str, object]:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    counts: Dict[str, int] = {}
    with open(path, "w", encoding="utf-8") as fh:
        for sample in generate(n, seed, kinds):
            counts[sample.kind] = counts.get(sample.kind, 0) + 1
            fh.write(json.dumps(sample.to_dict(), ensure_ascii=False) + "\n")
    return {"path": path, "count": n, "kinds": counts}
