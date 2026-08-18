"""Сбор обучающих данных дистилляцией **с верификацией**.

Схема (та же, что дала Zero-to-CAD 1M и CAD-Recode, только с физикой):

    ТЗ → учитель пишет OpenSCAD → наш движок проверяет (CSG, manifold, стенки,
    МКЭ, масса) → неудачные попытки возвращаются учителю как обратная связь →
    прошедшие порог записываются в корпус вместе с траекторией исправлений.

Метки здесь не «мнение большой модели», а результат расчёта, поэтому корпус
самоочищается: плохое просто не попадает внутрь.

Учителя (`Designer`):
  * `CommandDesigner` — любой CLI (`claude -p`, `llm`, `ollama run …`): самый
    дешёвый способ подключить сильную модель, ключи остаются у вас;
  * `HFDesigner` — локальная модель через transformers;
  * `NexusDesigner` — наша же модель (самоулучшение после первого обучения);
  * `TemplateDesigner` — офлайн-заглушка на параметрических шаблонах.
"""
from __future__ import annotations

import json
import os
import random
import re
import subprocess
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence

from ..mcp.server import NexusTools

CODE_BLOCK = re.compile(r"```(?:openscad|scad|c\+\+|c)?\s*(.+?)```", re.DOTALL)

SYSTEM_PROMPT = (
    "Ты инженер-конструктор. Пиши только код OpenSCAD (без пояснений), "
    "который решает задачу. Требования: деталь должна быть единым замкнутым телом "
    "(manifold), минимальная толщина стенки 1.2 мм, отверстия сквозные, "
    "выдерживать указанную нагрузку с заданным запасом прочности."
)


class Designer(Protocol):
    def propose(self, spec: str, feedback: Optional[str] = None) -> str: ...


# ────────────────────────────────────────────────────────────── учителя
class TemplateDesigner:
    """Офлайн-заглушка: параметрические шаблоны со случайными параметрами."""

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def propose(self, spec: str, feedback: Optional[str] = None) -> str:
        from ..scad.generator import TEMPLATES, sample
        for name in TEMPLATES:
            if name.split("_")[0] in spec.lower():
                return sample(name, self.rng).code
        return sample(self.rng.choice(list(TEMPLATES)), self.rng).code


class CommandDesigner:
    """Внешний CLI как учитель: промпт на stdin, код на stdout.

        CommandDesigner(["claude", "-p"])          # Claude Code
        CommandDesigner(["ollama", "run", "qwen2.5-coder:7b"])
    """

    def __init__(self, command: Sequence[str], timeout: int = 180,
                 system: str = SYSTEM_PROMPT):
        self.command = list(command)
        self.timeout = timeout
        self.system = system

    def propose(self, spec: str, feedback: Optional[str] = None) -> str:
        prompt = f"{self.system}\n\nЗадача:\n{spec}\n"
        if feedback:
            prompt += f"\nПредыдущая попытка не прошла проверку:\n{feedback}\nИсправь код.\n"
        proc = subprocess.run(self.command, input=prompt, capture_output=True,
                              text=True, timeout=self.timeout)
        return extract_code(proc.stdout)


class HFDesigner:
    """Локальная модель через transformers."""

    def __init__(self, model_id: str, device: str = "cpu", max_new_tokens: int = 512,
                 system: str = SYSTEM_PROMPT):
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id).to(device).eval()
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.system = system

    def propose(self, spec: str, feedback: Optional[str] = None) -> str:
        import torch
        text = f"{self.system}\n\nЗадача: {spec}\n"
        if feedback:
            text += f"Замечания к прошлой попытке: {feedback}\n"
        text += "Код OpenSCAD:\n```\n"
        ids = self.tok(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(**ids, max_new_tokens=self.max_new_tokens,
                                      do_sample=True, temperature=0.7, top_p=0.95)
        return extract_code(self.tok.decode(out[0], skip_special_tokens=True))


class NexusDesigner:
    """Наша модель из реестра — самоулучшение (self-distillation)."""

    def __init__(self, name: str = "core", ref: str = "production",
                 registry_root: str = "artifacts/registry", device: str = "cpu",
                 max_new_tokens: int = 256):
        import torch
        from ..data.bpe import load_tokenizer
        from ..registry import ModelRegistry
        from ..training.trainer import load_model
        registry = ModelRegistry(registry_root)
        self.model, self.version = load_model(registry, name, ref, device)
        self.tok = load_tokenizer(self.version.tokenizer_path)
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.torch = torch

    def propose(self, spec: str, feedback: Optional[str] = None) -> str:
        prompt = spec + ("\n" + feedback if feedback else "") + "<scad>"
        ids = self.torch.tensor([self.tok.encode(prompt, bos=True)], device=self.device)
        out = self.model.generate(ids, max_new_tokens=self.max_new_tokens, temperature=0.9)
        return extract_code(self.tok.decode(out[0, ids.shape[1]:].tolist()))


def extract_code(text: str) -> str:
    """Достать код из ответа модели: блок ``` или текст как есть."""
    match = CODE_BLOCK.search(text)
    code = match.group(1) if match else text
    return code.strip()


# ────────────────────────────────────────────────────────────── сбор
@dataclass
class Task:
    spec: str
    force_n: List[float] = field(default_factory=lambda: [0.0, 0.0, -200.0])
    fixture: str = "base"
    material: str = "pla"
    required_sf: float = 2.0
    mass_budget_g: float = 200.0


@dataclass
class CollectStats:
    tasks: int = 0
    accepted: int = 0
    attempts: int = 0
    rewards: List[float] = field(default_factory=list)
    failures: Dict[str, int] = field(default_factory=dict)
    seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        acc = self.accepted / max(self.tasks, 1)
        mean = sum(self.rewards) / len(self.rewards) if self.rewards else 0.0
        return {"tasks": self.tasks, "accepted": self.accepted,
                "acceptance_rate": round(acc, 3),
                "attempts_per_task": round(self.attempts / max(self.tasks, 1), 2),
                "mean_reward": round(mean, 3), "failures": self.failures,
                "seconds": round(self.seconds, 1)}


def default_tasks(n: int, seed: int = 0) -> List[Task]:
    """ТЗ берём из нашего же генератора: он даёт реалистичные нагрузки."""
    from ..scad.generator import generate
    return [Task(s.spec, list(s.load.force_n), s.load.fixture, s.material,
                 s.load.safety_factor)
            for s in generate(n, seed=seed)]


def _feedback(report: Dict[str, Any]) -> str:
    info = report.get("info") or {}
    lines = []
    if report.get("compile", 0) <= 0:
        lines.append(f"код не компилируется: {info.get('error', 'ошибка синтаксиса CSG')}")
    if report.get("manifold", 0) < 1.5:
        lines.append("деталь не единое замкнутое тело (manifold) либо не технологична")
    if report.get("wall_penalty", 0) < 0:
        lines.append(f"слишком тонкие стенки: {info.get('min_wall_mm')} мм (нужно ≥ 1.2 мм)")
    if report.get("strength", 0) < 2.0:
        lines.append(f"недостаточный запас прочности: {info.get('safety_factor')} — "
                     f"усильте сечения или добавьте рёбра")
    if report.get("mass_penalty", 0) < 0:
        lines.append(f"перевес: {info.get('mass_g')} г — облегчите деталь")
    return "; ".join(lines) or "оценка ниже порога"


def _failure_key(report: Dict[str, Any]) -> str:
    if report.get("compile", 0) <= 0:
        return "compile"
    if report.get("manifold", 0) < 1.5:
        return "manifold"
    if report.get("strength", 0) < 1.0:
        return "strength"
    if report.get("wall_penalty", 0) < 0:
        return "thin_wall"
    return "below_threshold"


def collect(
    designer: Designer,
    tasks: Optional[Sequence[Task]] = None,
    n_tasks: int = 16,
    attempts: int = 3,
    threshold: float = 3.5,
    out_path: str = "artifacts/collected/dataset.jsonl",
    workdir: str = "artifacts/mcp",
    seed: int = 0,
    grid: int = 20,
    verbose: bool = True,
) -> CollectStats:
    """Прогнать учителя по задачам, проверить физикой, записать удачное."""
    tools = NexusTools(workdir)
    tasks = list(tasks or default_tasks(n_tasks, seed))
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    stats = CollectStats()
    t0 = time.time()

    with open(out_path, "w", encoding="utf-8") as fh:
        for i, task in enumerate(tasks):
            stats.tasks += 1
            feedback: Optional[str] = None
            trajectory: List[Dict[str, Any]] = []
            best: Optional[Dict[str, Any]] = None

            for attempt in range(attempts):
                stats.attempts += 1
                try:
                    code = designer.propose(task.spec, feedback)
                except Exception as exc:                      # учитель может упасть
                    trajectory.append({"attempt": attempt, "error": str(exc)})
                    break
                report = tools.score_design(code, task.force_n, task.fixture,
                                            task.material, task.required_sf,
                                            task.mass_budget_g, grid)
                total = float(report.get("total", -1))
                trajectory.append({"attempt": attempt, "reward": total,
                                   "feedback": feedback})
                if best is None or total > best["reward"]:
                    best = {"code": code, "reward": total, "report": report}
                if total >= threshold:
                    break
                feedback = _feedback(report)

            if best is None:
                stats.failures["no_output"] = stats.failures.get("no_output", 0) + 1
                continue
            stats.rewards.append(best["reward"])
            if best["reward"] < threshold:
                key = _failure_key(best["report"])
                stats.failures[key] = stats.failures.get(key, 0) + 1
                continue

            stats.accepted += 1
            record = {
                "spec": task.spec,
                "code": best["code"],
                "reward": best["reward"],
                "report": best["report"],
                "material": task.material,
                "load": {"force_n": task.force_n, "fixture": task.fixture},
                "attempts": len(trajectory),
                "trajectory": trajectory,
                # готовая обучающая строка для nexus.data.corpora (jsonl:...#text)
                "text": (task.spec + "<scad>" + best["code"] + "<fem>" +
                         json.dumps(best["report"].get("info", {}), ensure_ascii=False)),
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            if verbose and (i + 1) % 5 == 0:
                print(f"  [collect] {i + 1}/{len(tasks)} принято {stats.accepted}",
                      flush=True)

    stats.seconds = time.time() - t0
    with open(os.path.splitext(out_path)[0] + "_stats.json", "w", encoding="utf-8") as fh:
        json.dump(stats.to_dict(), fh, indent=2, ensure_ascii=False)
    return stats


def make_designer(kind: str, **kwargs) -> Designer:
    if kind == "template":
        return TemplateDesigner(int(kwargs.get("seed", 0)))
    if kind == "command":
        return CommandDesigner(kwargs["command"], int(kwargs.get("timeout", 180)))
    if kind == "hf":
        return HFDesigner(kwargs["model_id"], kwargs.get("device", "cpu"))
    if kind == "nexus":
        return NexusDesigner(kwargs.get("name", "core"), kwargs.get("ref", "production"),
                             kwargs.get("registry_root", "artifacts/registry"),
                             kwargs.get("device", "cpu"))
    raise ValueError(f"неизвестный тип учителя: {kind}")
