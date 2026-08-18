"""Экспорт наших данных в формат для дообучения готовых LLM (SFT).

Наши корпуса лежат в собственном формате (`spec`, `code`, `physics`), а
инструменты дообучения — Unsloth, TRL, LLaMA-Factory, Axolotl — ждут диалоги
в стиле chat: `{"messages": [{"role": "system"...}, {"role": "user"...},
{"role": "assistant"...}]}`.

Эта функция переводит одно в другое и попутно добавляет то, чего нет ни в одном
чужом датасете: **физические факты о детали** (масса, запас прочности, стенка).
Модель учится не только писать код, но и заявлять его свойства.

    nexus export-sft --source jsonl:artifacts/ingest/dataset.jsonl \\
                     --out artifacts/sft --with-physics
"""
from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence

SYSTEM_PROMPT = (
    "Ты инженер-конструктор. По техническому заданию пишешь параметрический код "
    "OpenSCAD. Требования: единое замкнутое тело (manifold), минимальная стенка "
    "1.2 мм, параметры вынесены в начало файла с комментариями, отверстия сквозные. "
    "Отвечай только кодом в блоке ```openscad."
)

REPAIR_SYSTEM_PROMPT = (
    "Ты инженер-конструктор. Тебе дают код детали и замечания расчёта прочности "
    "и технологичности. Исправь код так, чтобы замечания были сняты, не меняя "
    "присоединительных размеров. Отвечай только кодом в блоке ```openscad."
)


@dataclass
class SFTStats:
    total: int = 0
    exported: int = 0
    repair_pairs: int = 0
    skipped: Dict[str, int] = field(default_factory=dict)
    train: int = 0
    val: int = 0

    def skip(self, reason: str) -> None:
        self.skipped[reason] = self.skipped.get(reason, 0) + 1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _physics_line(record: Dict[str, Any]) -> str:
    """Короткая сводка расчёта — модель учится её предсказывать вместе с кодом."""
    physics = record.get("physics") or record.get("report", {}).get("info") or {}
    if not physics:
        return ""
    parts = []
    if physics.get("mass_g"):
        parts.append(f"масса {float(physics['mass_g']):.1f} г")
    if physics.get("min_wall_mm"):
        parts.append(f"минимальная стенка {float(physics['min_wall_mm']):.2f} мм")
    if physics.get("safety_factor"):
        parts.append(f"запас прочности {float(physics['safety_factor']):.2f}")
    return "// расчёт: " + ", ".join(parts) if parts else ""


def _load(source: str, limit: Optional[int] = None) -> Iterator[Dict[str, Any]]:
    path = source.split(":", 1)[1] if source.startswith("jsonl:") else source
    path = path.split("#", 1)[0]
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if limit and i >= limit:
                return
            if line.strip():
                yield json.loads(line)


def to_messages(record: Dict[str, Any], with_physics: bool = True,
                system: str = SYSTEM_PROMPT) -> Optional[Dict[str, Any]]:
    """Одна запись корпуса → диалог из трёх реплик."""
    spec = (record.get("spec") or record.get("prompt") or "").strip()
    code = (record.get("code") or "").strip()
    if not code:
        return None
    if not spec:
        spec = "Спроектируй деталь по этому описанию."

    answer = f"```openscad\n{code}\n```"
    if with_physics:
        line = _physics_line(record)
        if line:
            answer += f"\n{line}"
    return {"messages": [
        {"role": "system", "content": system},
        {"role": "user", "content": spec},
        {"role": "assistant", "content": answer},
    ]}


def repair_messages(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Из траектории «сломано → замечание → исправлено» делаем пример ремонта.

    Это самый ценный тип данных: чужие датасеты содержат только финальный код.
    """
    trajectory = record.get("trajectory") or []
    code = (record.get("code") or "").strip()
    if len(trajectory) < 2 or not code:
        return None
    feedback = next((t.get("feedback") for t in reversed(trajectory) if t.get("feedback")),
                    None)
    if not feedback:
        return None
    spec = (record.get("spec") or "").strip()
    return {"messages": [
        {"role": "system", "content": REPAIR_SYSTEM_PROMPT},
        {"role": "user", "content": f"Задача: {spec}\n\nЗамечания расчёта: {feedback}\n\n"
                                    f"Исправь деталь."},
        {"role": "assistant", "content": f"```openscad\n{code}\n```"},
    ]}


def export_sft(
    sources: Sequence[str],
    out_dir: str = "artifacts/sft",
    with_physics: bool = True,
    with_repair: bool = True,
    val_fraction: float = 0.05,
    limit: Optional[int] = None,
    min_code_chars: int = 40,
    seed: int = 0,
    verbose: bool = True,
) -> SFTStats:
    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(seed)
    stats = SFTStats()
    train_path = os.path.join(out_dir, "train.jsonl")
    val_path = os.path.join(out_dir, "val.jsonl")

    with open(train_path, "w", encoding="utf-8") as train_fh, \
            open(val_path, "w", encoding="utf-8") as val_fh:
        for source in sources:
            for record in _load(source, limit):
                stats.total += 1
                code = (record.get("code") or "").strip()
                if len(code) < min_code_chars:
                    stats.skip("too_short")
                    continue
                if record.get("valid") is False:
                    stats.skip("invalid")
                    continue

                samples = [to_messages(record, with_physics)]
                if with_repair:
                    repair = repair_messages(record)
                    if repair:
                        samples.append(repair)
                        stats.repair_pairs += 1

                for sample in samples:
                    if sample is None:
                        continue
                    target = val_fh if rng.random() < val_fraction else train_fh
                    target.write(json.dumps(sample, ensure_ascii=False) + "\n")
                    stats.exported += 1
                    if target is val_fh:
                        stats.val += 1
                    else:
                        stats.train += 1

    with open(os.path.join(out_dir, "stats.json"), "w", encoding="utf-8") as fh:
        json.dump(stats.to_dict(), fh, indent=2, ensure_ascii=False)
    if verbose:
        print(json.dumps(stats.to_dict(), indent=2, ensure_ascii=False))
    return stats
