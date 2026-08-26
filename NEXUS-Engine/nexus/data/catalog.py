"""Каталог открытых датасетов, пригодных для обучения NEXUS.

Каждая запись — готовая строка-источник для `nexus.data.corpora`, размер,
лицензия и честный комментарий: что реально даёт, чего стоит опасаться.
Проверить доступность: `nexus datasets check`.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional


@dataclass
class DatasetEntry:
    key: str
    name: str
    task: str                 # cad-code | cad-brep | sim | math | code | text
    size: str
    license: str
    commercial_ok: Optional[bool]
    source: str               # спецификация для nexus.data.corpora
    url: str
    notes: str
    priority: int = 3         # 1 — брать первым

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


CATALOG: List[DatasetEntry] = [
    # ───────────────────────────── CAD как код (главный приоритет) ─────────
    DatasetEntry(
        "thingiverse-openscad", "redcathode/thingiverse-openscad", "cad-code",
        "7.4k пар (26 МБ)", "CC-BY-NC-SA-4.0", False,
        "hf:redcathode/thingiverse-openscad:train#scad",
        "https://huggingface.co/datasets/redcathode/thingiverse-openscad",
        "Единственный публичный корпус именно OpenSCAD с синтетическими промптами. "
        "Мал, шумный, NC-лицензия — годится для исследований и как стилевой якорь, "
        "не для коммерческого релиза весов.", 1),
    DatasetEntry(
        "cad-coder", "gudo7208/CAD-Coder", "cad-code",
        "250k сэмплов, из них 8.2k «high» и 1.5k CoT", "Apache-2.0", True,
        "hf:gudo7208/CAD-Coder:train#messages",
        "https://huggingface.co/datasets/gudo7208/CAD-Coder",
        "Text→CadQuery с цепочками рассуждений и коммерчески пригодной лицензией. "
        "CadQuery ≠ OpenSCAD, но операции те же: транслируется скриптом.", 1),
    DatasetEntry(
        "zero-to-cad-1m", "ADSKAILab/Zero-To-CAD-1m", "cad-code",
        "1M исполняемых программ", "см. карточку датасета", None,
        "hf:ADSKAILab/Zero-To-CAD-1m:train#cadquery_file",
        "https://huggingface.co/datasets/ADSKAILab/Zero-To-CAD-1m",
        "Самый крупный корпус читаемых CAD-программ: булевы, фаски, скругления, "
        "лофты, свипы, оболочки. Сгенерирован LLM в среде с обратной связью — "
        "ровно та схема, что мы строим у себя.", 1),
    DatasetEntry(
        "benchcad", "BenchCAD/BenchCAD", "cad-code",
        "17.9k программ + QA + edit-bench (963 МБ)", "CC-BY-4.0", True,
        "hf:BenchCAD/BenchCAD:code_gen#code",
        "https://huggingface.co/datasets/BenchCAD/BenchCAD",
        "106 семейств промышленных деталей (шестерни, пружины, свёрла) с "
        "исполнением-верификацией. Отличный held-out бенчмарк, не только обучение.", 2),
    DatasetEntry(
        "gencad-code", "CADCODER/GenCAD-Code", "cad-code",
        "163k пар изображение→код", "см. карточку", None,
        "hf:CADCODER/GenCAD-Code:train#code",
        "https://huggingface.co/datasets/CADCODER/GenCAD-Code",
        "Для мультимодального этапа (картинка→код), когда подключим визуальный энкодер.", 3),
    DatasetEntry(
        "text2cad", "SadilKhan/Text2CAD", "cad-code",
        "~170k аннотаций (полный набор 605 ГБ)", "CC-BY-NC-SA-4.0", False,
        "hf:SadilKhan/Text2CAD:train#description",
        "https://huggingface.co/datasets/SadilKhan/Text2CAD",
        "Классика text-to-CAD, но NC-лицензия и огромный объём вложений.", 3),

    # ───────────────────────────── Геометрия и B-Rep ───────────────────────
    DatasetEntry(
        "fusion360-gallery", "Fusion 360 Gallery Reconstruction", "cad-brep",
        "8.6k последовательностей (2 ГБ)", "только некоммерческие исследования", False,
        "dir:./external/fusion360",
        "https://github.com/AutodeskAILab/Fusion360GalleryDataset",
        "Человеческие истории построения (sketch+extrude) — эталон «как проектирует "
        "человек». Мало, зато качественно; лицензия в стиле ImageNet.", 2),
    DatasetEntry(
        "abc-dataset", "ABC Dataset", "cad-brep",
        "1M+ CAD-моделей (терабайты)", "MIT (сборник), см. отдельные модели", True,
        "dir:./external/abc",
        "https://deep-geometry.github.io/abc-dataset/",
        "Геометрия без историй построения. Нам полезен как источник B-Rep для "
        "уровня 1 (GNO-энкодер) и для генерации FEM-меток нашим решателем.", 2),

    # ───────────────────────────── Симуляции / физика ──────────────────────
    DatasetEntry(
        "simjeb", "SimJEB — Simulated Jet Engine Bracket", "sim",
        "381 деталь × 4 нагрузочных случая", "ODC-BY / CC0 на данные, CAD — NC", None,
        "dir:./external/simjeb",
        "https://simjeb.github.io",
        "Реальные кронштейны с честным FEM (σ фон Мизеса, перемещения). "
        "Идеален как валидация нашего hex-решателя и FNO-суррогата.", 1),
    DatasetEntry(
        "deepjeb", "DeepJEB — синтетические кронштейны", "sim",
        "2138 деталей с полями напряжений", "ODC-BY", True,
        "dir:./external/deepjeb",
        "https://www.narnia.ai/dataset",
        "В 5.6 раза больше SimJEB, есть знаковые напряжения и собственные частоты — "
        "готовые метки для суррогатного критика.", 2),

    # ───────────────────────────── Математика и рассуждения ────────────────
    DatasetEntry(
        "openmathreasoning", "nvidia/OpenMathReasoning", "math",
        "540k задач, 3.2M CoT, 1.7M TIR", "CC-BY-4.0", True,
        "hf:nvidia/OpenMathReasoning:cot#problem",
        "https://huggingface.co/datasets/nvidia/OpenMathReasoning",
        "Лучшее соотношение «качество/лицензия» для математики. Есть решения с "
        "инструментами (TIR) — прямой аналог нашего latent-цикла с FNO-критиком.", 1),
    DatasetEntry(
        "openr1-math-220k", "open-r1/OpenR1-Math-220k", "math",
        "225k задач с проверенными решениями (4.2 ГБ)", "Apache-2.0", True,
        "hf:open-r1/OpenR1-Math-220k:train#problem",
        "https://huggingface.co/datasets/open-r1/OpenR1-Math-220k",
        "Решения от DeepSeek-R1 с автоматической верификацией ответа "
        "(`correctness_math_verify`) — можно фильтровать только доказанно верные.", 1),
    DatasetEntry(
        "nexus-engineering-math", "Встроенный генератор инженерной математики", "math",
        "сколько нужно (генерируется)", "MIT (наш код)", True,
        "mathgen:20000",
        "nexus/data/mathgen.py",
        "Балка на изгиб, момент затяжки, тепловое расширение, посадки, "
        "размерные цепи, масса и инерция. Ответ считается формулой, а не моделью, "
        "поэтому ошибок в метках нет по построению.", 1),

    # ───────────────────────────── Код и текст ─────────────────────────────
    DatasetEntry(
        "the-stack-v2-python", "bigcode/the-stack-v2 (python/scad)", "code",
        "терабайты; берём подвыборку", "разрешительные лицензии исходников", True,
        "hf:bigcode/the-stack-v2-dedup:train#content",
        "https://huggingface.co/datasets/bigcode/the-stack-v2",
        "Общий код нужен, чтобы модель понимала синтаксис и структуру. "
        "Берём Python/C/OpenSCAD подвыборку на 2–5 ГБ, не больше.", 2),
    DatasetEntry(
        "openscad-docs", "Документация и библиотеки OpenSCAD/BOSL2", "text",
        "~50–100 МБ", "GPL/CC (см. репозитории)", None,
        "dir:./external/openscad-docs",
        "https://github.com/BelfrySCAD/BOSL2",
        "Cheatsheet, руководство, BOSL2/MCAD — язык, идиомы и стандартные модули. "
        "Маленький, но очень плотный по полезности корпус.", 1),
]


def by_key(key: str) -> DatasetEntry:
    for entry in CATALOG:
        if entry.key == key:
            return entry
    raise KeyError(f"нет датасета {key!r} в каталоге")


def filter_catalog(task: Optional[str] = None, commercial_only: bool = False,
                   max_priority: int = 3) -> List[DatasetEntry]:
    out = [e for e in CATALOG if e.priority <= max_priority]
    if task:
        out = [e for e in out if e.task == task]
    if commercial_only:
        out = [e for e in out if e.commercial_ok]
    return sorted(out, key=lambda e: (e.priority, e.task))


# Рекомендованные миксы: доля токенов в обучающем корпусе.
MIXES: Dict[str, Dict[str, float]] = {
    "bootstrap": {          # первый прогон на своей машине, только своё и свободное
        "mathgen:20000": 0.25,
        "flywheel:artifacts/flywheel": 0.45,
        "builtin:engineering": 0.05,
        "hf:gudo7208/CAD-Coder:train#messages": 0.25,
    },
    "engineering": {        # основной профиль: CAD-код + физика + математика
        "flywheel:artifacts/flywheel": 0.30,
        "hf:ADSKAILab/Zero-To-CAD-1m:train#cadquery_file": 0.25,
        "hf:gudo7208/CAD-Coder:train#messages": 0.15,
        "mathgen:50000": 0.15,
        "hf:open-r1/OpenR1-Math-220k:train#problem": 0.10,
        "dir:./external/openscad-docs": 0.05,
    },
    "reasoning": {          # добавка рассуждений перед RL-фазой
        "hf:nvidia/OpenMathReasoning:cot#problem": 0.40,
        "hf:open-r1/OpenR1-Math-220k:train#problem": 0.30,
        "mathgen:50000": 0.30,
    },
}


def mix_source(name: str) -> str:
    """Строка-источник вида `mix:src1:0.3,src2:0.7` для nexus.data.corpora."""
    mix = MIXES[name]
    return "mix:" + ",".join(f"{src}={weight}" for src, weight in mix.items())


def summary() -> Dict[str, object]:
    return {
        "datasets": [e.to_dict() for e in CATALOG],
        "mixes": {k: v for k, v in MIXES.items()},
        "totals": {
            "entries": len(CATALOG),
            "commercially_safe": sum(1 for e in CATALOG if e.commercial_ok),
            "priority_1": sum(1 for e in CATALOG if e.priority == 1),
        },
    }
