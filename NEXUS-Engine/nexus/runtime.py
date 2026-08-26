"""Выбор устройства и диагностика GPU.

Главная ловушка: `pip install torch` с PyPI ставит **CPU-сборку**, и на машине
с RTX всё молча считается на процессоре. Для карт 50-й серии (Blackwell,
sm_120) нужны колёса cu128 и PyTorch ≥ 2.7:

    pip install torch --index-url https://download.pytorch.org/whl/cu128

Именно `--index-url`, а не `--extra-index-url`: иначе pip снова возьмёт
CPU-колесо с PyPI.
"""
from __future__ import annotations

import shutil
import subprocess
from typing import Any, Dict, List, Optional

# Минимальная сборка CUDA для архитектуры GPU
ARCH_REQUIREMENTS = [
    (12, 0, "cu128", "Blackwell (RTX 50xx)"),
    (9, 0, "cu121", "Hopper (H100)"),
    (8, 9, "cu121", "Ada (RTX 40xx)"),
    (8, 6, "cu118", "Ampere (RTX 30xx)"),
]


def nvidia_smi() -> Optional[Dict[str, str]]:
    """Данные о первой видеокарте через nvidia-smi (работает без torch)."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run(
            [exe, "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0 or not out.stdout.strip():
        return None
    name, memory, driver = (x.strip() for x in out.stdout.strip().splitlines()[0].split(","))
    return {"name": name, "vram_mb": memory, "driver": driver}


def pick_device(preference: str = "auto") -> str:
    """`auto` → cuda, если доступна; иначе то, что попросили."""
    if preference and preference != "auto":
        return preference
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def gpu_report() -> Dict[str, Any]:
    """Полная картина: что видит драйвер, что умеет установленный torch."""
    report: Dict[str, Any] = {"nvidia_smi": nvidia_smi()}
    try:
        import torch
        report["torch"] = torch.__version__
        report["torch_cuda_build"] = torch.version.cuda          # None у CPU-сборки
        report["cuda_available"] = torch.cuda.is_available()
        arch_list: List[str] = []
        try:
            arch_list = torch.cuda.get_arch_list()
        except Exception:
            pass
        report["arch_list"] = arch_list
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            report["gpu"] = props.name
            report["capability"] = f"sm_{props.major}{props.minor}"
            report["vram_gb"] = round(props.total_memory / 1024 ** 3, 1)
    except Exception as exc:                                     # pragma: no cover
        report["error"] = str(exc)

    report["problem"], report["fix"] = _diagnose(report)
    report["device"] = pick_device("auto")
    return report


def _diagnose(report: Dict[str, Any]):
    smi = report.get("nvidia_smi")
    build = report.get("torch_cuda_build")
    available = report.get("cuda_available")
    if not smi:
        return None, None
    if not build:
        return ("Видеокарта есть, но установлена CPU-сборка torch — "
                "обучение идёт на процессоре."), _fix_command(smi)
    if not available:
        return ("torch собран с CUDA, но карта не видна: обычно это старый драйвер "
                "или несовместимая версия колеса."), _fix_command(smi)
    arch = report.get("capability")
    arch_list = report.get("arch_list") or []
    if arch and arch_list and arch not in arch_list and not any(
            a.startswith(arch) for a in arch_list):
        return (f"Архитектура {arch} не поддерживается этой сборкой "
                f"({', '.join(arch_list)})."), _fix_command(smi)
    return None, None


def _fix_command(smi: Dict[str, str]) -> str:
    name = (smi or {}).get("name", "").upper()
    wheel = "cu128"
    for token, tag in (("RTX 50", "cu128"), ("RTX 40", "cu126"), ("RTX 30", "cu126")):
        if token in name:
            wheel = tag
            break
    return ("pip uninstall -y torch && "
            f"pip install torch --index-url https://download.pytorch.org/whl/{wheel}")


def recommended_wheel() -> str:
    smi = nvidia_smi()
    return _fix_command(smi).rsplit("/", 1)[-1] if smi else "cpu"


def describe(device: Optional[str] = None) -> str:
    device = device or pick_device("auto")
    if device.startswith("cuda"):
        try:
            import torch
            props = torch.cuda.get_device_properties(0)
            return (f"{props.name}, {props.total_memory / 1024 ** 3:.1f} ГБ VRAM, "
                    f"sm_{props.major}{props.minor}")
        except Exception:
            return device
    return device
