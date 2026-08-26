"""Сквозная демонстрация NEXUS-Engine: все три уровня на одном прогоне.

    уровень 1: пять модальностей → LatentPacket
    уровень 2: Unified Latent Bus → TTT + Sliding Attention + MoE → Latent Reasoning
    уровень 3: дискретный выход (SCAD/текст) + непрерывный (поля, траектории)

Плюс инженерный контур: генерация детали → аудит → FEM → физическая награда.
"""
from __future__ import annotations

import json
import os
from typing import Dict

import numpy as np
import torch

from .config import NexusConfig
from .data.flywheel import run as flywheel_run
from .encoders.audio_visual import AudioSSMEncoder
from .encoders.scad_brep import BRepGNOEncoder, FieldEncoder
from .encoders.spatial import BiophysicsEncoder, EventODEEncoder, PointSSMEncoder
from .eval.needle import run as needle_run
from .eval.vram import estimate
from .fem.solver import solve
from .model import NexusEngine
from .scad.generator import generate
from .scad.render import render
from .training.rewards import score_scad


def run_demo(out_dir: str = "artifacts/demo", n_samples: int = 12) -> Dict[str, object]:
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(0)
    cfg = NexusConfig.tiny()
    report: Dict[str, object] = {}

    print("── 1. Инженерный контур: OpenSCAD → CSG → аудит → FEM ──")
    sample = generate(1, seed=7)[0]
    res = render(sample.code, resolution=24, material=sample.material,
                 stl_path=os.path.join(out_dir, "part.stl"))
    fem = solve(res.voxels, sample.load.force_n, sample.load.fixture, sample.material,
                iterations=150)
    reward = score_scad(sample.code, sample.load.force_n, sample.load.fixture,
                        sample.material, resolution=18, fem_iterations=80)
    report["engineering"] = {
        "template": sample.template,
        "mass": res.mass.to_dict(),
        "audit": res.audit_report.to_dict(),
        "fem": fem.to_dict(),
        "reward": reward.to_dict(),
    }
    print(f"   деталь: {sample.template}, масса {res.mass.mass_g:.1f} г, "
          f"σmax {fem.max_stress_pa/1e6:.1f} МПа, запас {fem.safety_factor:.2f}, "
          f"награда {reward.total:+.2f}")

    print("── 2. Мультимодальный вход: 5 классов сигналов на одну шину ──")
    model = NexusEngine(cfg)
    d = cfg.d_latent
    brep_enc = BRepGNOEncoder(d, width=64, layers=2)
    audio_enc = AudioSSMEncoder(d, width=32)
    point_enc = PointSSMEncoder(d, width=32)
    event_enc = EventODEEncoder(d, width=32)
    bio_enc = BiophysicsEncoder(d, channels=8, width=32)
    field_enc = FieldEncoder(d, width=32)
    for name, enc in [("brep", brep_enc), ("audio", audio_enc), ("gaussian", point_enc),
                      ("event", event_enc), ("ecg", bio_enc), ("stress", field_enc)]:
        model.register_encoder(name, enc)

    nodes, adj, mask = brep_enc.encode_graphs([res.graph])
    packets = [
        brep_enc(nodes, adj, mask),
        audio_enc(torch.randn(1, 3200)),
        point_enc(torch.randn(1, 24, 10)),
        event_enc(torch.rand(1, 32, 3), torch.cumsum(torch.rand(1, 32) * 1e-3, dim=1)),
        bio_enc(torch.randn(1, 16, 8), torch.cumsum(torch.rand(1, 16) * 4e-3, dim=1)),
        field_enc(torch.from_numpy(
            np.ascontiguousarray(fem.von_mises[None, None, :8, :8, :8] / max(fem.max_stress_pa, 1))
        ).float()),
    ]
    report["modalities"] = {p.modality: list(p.shape) for p in packets}
    print("   пакеты:", {p.modality: p.shape[1] for p in packets})

    print("── 3. Ядро + латентное рассуждение + двухрежимный вывод ──")
    tokens = model.text_encoder.encode_text([sample.spec[:400]])
    occ = torch.from_numpy(res.voxels.occupancy.astype(np.float32))[None, None]
    occ = torch.nn.functional.interpolate(occ, size=(cfg.fno.grid,) * 3, mode="trilinear")
    load = torch.tensor([sample.load.force_n], dtype=torch.float32) / 1000.0
    out = model(tokens=tokens, packets=packets, occupancy=occ, load=load,
                reason=True, continuous=True)
    report["core"] = {
        "logits": list(out.logits.shape),
        "actions": list(out.actions.shape),
        "field": list(out.field.shape),
        "physics": [round(float(v), 4) for v in out.physics[0].detach()],
        "reasoning": out.reasoning.to_dict(),
        "parameters": model.parameter_report(),
    }
    print(f"   латентных шагов рассуждения: {out.reasoning.steps}, "
          f"выход: logits {tuple(out.logits.shape)}, поле {tuple(out.field.shape)}")

    print("── 4. O(1) память (Needle-in-a-Haystack) ──")
    needle = [r.to_dict() for r in needle_run([512, 2048, 8192], cfg)]
    report["needle"] = needle
    print(f"   контекст 8192: состояние {needle[-1]['state_mb']} МБ против "
          f"{needle[-1]['kv_cache_mb_if_full_attention']} МБ у KV-кэша "
          f"(×{needle[-1]['compression_x']})")

    print("── 5. Маховик данных ──")
    stats = flywheel_run(n_samples, os.path.join(out_dir, "flywheel"), seed=3,
                         grid=18, fem_grid=8, verbose=False)
    report["flywheel"] = stats.to_dict()
    print("   ", stats.to_dict())

    print("── 6. Бюджет VRAM для RTX 5060 (16 ГБ) ──")
    report["vram_rtx5060"] = estimate(NexusConfig.rtx5060()).to_dict()
    print("   ", report["vram_rtx5060"])

    path = os.path.join(out_dir, "report.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    print(f"\nОтчёт сохранён: {path}")
    return report


if __name__ == "__main__":
    run_demo()
