"""A/B-сравнение генераторов на реальных запросах с физической проверкой.

Главный вопрос продукта: «можно ли заменить большую LLM своей моделью?»
Ответ должен быть измеримым, а не интуитивным. Этот модуль прогоняет один и тот
же набор реальных промптов через несколько генераторов, проверяет каждый
результат движком и печатает сравнительную таблицу.

    nexus bench --prompts prompts.txt \\
                --model nexus:core:production \\
                --model "command:claude -p" \\
                --out artifacts/bench

Метрики на каждый генератор:

* `compile_rate`    — доля ответов, которые собираются в валидный CSG;
* `manifold_rate`   — доля деталей, проходящих геометрический аудит;
* `strength_rate`   — доля деталей с запасом прочности ≥ требуемого;
* `mean_reward`     — сводная физическая оценка;
* `sec_per_item`    — время на запрос;
* `mean_mass_g`     — средняя масса (косвенно — не «залил ли материалом»).
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from ..data.collect import Designer, make_designer
from ..mcp.server import NexusTools

DEFAULT_PROMPTS = [
    "Держатель кабеля: клипса на скотч под провод диаметром 6 мм",
    "Ручка для мебели: межосевое расстояние 96 мм, цилиндрическая форма",
    "Заглушка круглая в отверстие 20 мм с буртиком",
    "Кронштейн L-образный под полку, нагрузка 15 кг, крепление 4 винта М4",
    "Переходник с трубы 40 мм на 32 мм, длина 60 мм",
    "Корпус для платы 50×70 мм с крышкой на защёлках",
    "Шкив ременной передачи GT2 на 20 зубьев, вал 5 мм",
    "Фланец Ø110 мм под трубу 32 мм, 4 болта М8",
]


@dataclass
class BenchResult:
    name: str
    items: int = 0
    compiled: int = 0
    manifold: int = 0
    strong: int = 0
    rewards: List[float] = field(default_factory=list)
    masses: List[float] = field(default_factory=list)
    seconds: float = 0.0
    errors: Dict[str, int] = field(default_factory=dict)
    samples: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        n = max(self.items, 1)
        return {
            "name": self.name,
            "items": self.items,
            "compile_rate": round(self.compiled / n, 3),
            "manifold_rate": round(self.manifold / n, 3),
            "strength_rate": round(self.strong / n, 3),
            "mean_reward": round(sum(self.rewards) / len(self.rewards), 3) if self.rewards else 0.0,
            "mean_mass_g": round(sum(self.masses) / len(self.masses), 1) if self.masses else 0.0,
            "sec_per_item": round(self.seconds / n, 2),
            "errors": self.errors,
        }


def parse_designer(spec: str, registry: str = "artifacts/registry",
                   device: str = "auto") -> tuple[str, Designer]:
    """`nexus:core:production` | `command:claude -p` | `hf:Qwen/...` | `template`."""
    kind, _, rest = spec.partition(":")
    if kind == "nexus":
        name, _, ref = rest.partition(":")
        return (f"nexus:{name or 'core'}:{ref or 'production'}",
                make_designer("nexus", name=name or "core", ref=ref or "production",
                              registry_root=registry, device=device))
    if kind == "command":
        return f"command:{rest}", make_designer("command", command=rest.split())
    if kind == "hf":
        return f"hf:{rest}", make_designer("hf", model_id=rest, device=device)
    if kind == "template":
        return "template", make_designer("template", seed=0)
    raise ValueError(f"неизвестный генератор: {spec!r}")


def run_bench(
    designers: Sequence[str],
    prompts: Optional[Sequence[str]] = None,
    out_dir: str = "artifacts/bench",
    material: str = "pla",
    force_n: Sequence[float] = (0.0, 0.0, -150.0),
    required_sf: float = 2.0,
    grid: int = 18,
    registry: str = "artifacts/registry",
    device: str = "auto",
    max_new_tokens: int = 256,
    verbose: bool = True,
) -> List[BenchResult]:
    prompts = list(prompts or DEFAULT_PROMPTS)
    os.makedirs(out_dir, exist_ok=True)
    tools = NexusTools(os.path.join(out_dir, "_work"))
    results: List[BenchResult] = []

    for spec in designers:
        name, designer = parse_designer(spec, registry, device)
        res = BenchResult(name=name)
        if verbose:
            print(f"\n[bench] {name}: {len(prompts)} запросов", flush=True)
        t0 = time.time()
        for prompt in prompts:
            res.items += 1
            try:
                code = designer.propose(f"<task>{prompt}")
            except Exception as exc:
                res.errors[type(exc).__name__] = res.errors.get(type(exc).__name__, 0) + 1
                continue
            report = tools.score_design(code, list(force_n), "base", material,
                                        required_sf=required_sf, grid=grid)
            info = report.get("info") or {}
            res.rewards.append(float(report.get("total", -1)))
            if report.get("compile", 0) > 0:
                res.compiled += 1
                res.masses.append(float(info.get("mass_g", 0.0)))
            if report.get("manifold", 0) >= 1.5:
                res.manifold += 1
            if float(info.get("safety_factor", 0.0)) >= required_sf:
                res.strong += 1
            res.samples.append({"prompt": prompt, "code": code, "report": report})
        res.seconds = time.time() - t0
        results.append(res)
        if verbose:
            print(f"[bench] {name}: {json.dumps(res.to_dict(), ensure_ascii=False)}",
                  flush=True)

    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as fh:
        json.dump([r.to_dict() for r in results], fh, indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, "samples.jsonl"), "w", encoding="utf-8") as fh:
        for r in results:
            for sample in r.samples:
                fh.write(json.dumps({"designer": r.name, **sample},
                                    ensure_ascii=False) + "\n")
    return results


def format_table(results: Sequence[BenchResult]) -> str:
    header = (f"{'генератор':28s} {'компилируется':>13s} {'manifold':>9s} "
              f"{'прочность':>10s} {'награда':>8s} {'масса, г':>9s} {'с/запрос':>9s}")
    lines = [header, "-" * len(header)]
    for r in results:
        d = r.to_dict()
        lines.append(f"{d['name'][:28]:28s} {d['compile_rate']:13.2f} "
                     f"{d['manifold_rate']:9.2f} {d['strength_rate']:10.2f} "
                     f"{d['mean_reward']:8.2f} {d['mean_mass_g']:9.1f} "
                     f"{d['sec_per_item']:9.2f}")
    lines.append("")
    lines.append("Решение о замене большой LLM принимается по compile_rate и manifold_rate "
                 "на реальных запросах пользователей, а не по перплексии.")
    return "\n".join(lines)


def load_prompts(path: str, limit: Optional[int] = None) -> List[str]:
    """Промпты из текстового файла (по строке) или JSONL с полем spec/prompt."""
    prompts: List[str] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith("{"):
                rec = json.loads(line)
                value = rec.get("prompt") or rec.get("spec") or ""
            else:
                value = line
            if value:
                prompts.append(value)
            if limit and len(prompts) >= limit:
                break
    return prompts
