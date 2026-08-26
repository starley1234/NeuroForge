"""INT8 / ONNX helpers and MCU memory report."""

from __future__ import annotations

from pathlib import Path

import torch

from aura_micro.config import AuraConfig
from aura_micro.model import AuraMicro


def memory_report(model: AuraMicro) -> dict[str, float]:
    n = model.count_parameters()
    return {
        "params": float(n),
        "int8_flash_kb": n / 1024.0,
        "fp32_flash_kb": n * 4 / 1024.0,
        "state_ram_bytes_int8": float(model.cfg.hidden),
        "state_ram_bytes_fp32": float(model.cfg.hidden * 4),
        "within_500kb_flash": float(n / 1024.0 <= 500.0),
        "within_128kb_ram_state": float(model.cfg.hidden * 4 <= 128 * 1024),
    }


def dynamic_int8(model: AuraMicro) -> torch.nn.Module:
    model.eval()
    return torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)


def export_onnx(model: AuraMicro, path: str | Path, cfg: AuraConfig | None = None) -> Path:
    cfg = cfg or model.cfg
    model.eval()
    dummy = torch.randn(1, cfg.n_frames, cfg.feat_dim)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        str(path),
        input_names=["features"],
        output_names=["logits"],
        opset_version=17,
        dynamo=False,
    )
    return path
