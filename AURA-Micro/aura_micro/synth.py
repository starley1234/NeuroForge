"""On-device-free spatial synthesis: dry tones + 3-mic delays, Doppler, air absorption."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from aura_micro.config import CLASS_NAMES, AuraConfig
from aura_micro.physics import SPEED_OF_SOUND, air_absorption_db, azimuth_elevation, mic_positions


@dataclass
class SceneLabel:
    class_id: int
    theta: float
    phi: float
    range_m: float
    vr: float
    modifiers: np.ndarray  # [n_mod] 0/1


def _class_signature(class_id: int, n: int, sr: int, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(n) / sr
    name = CLASS_NAMES[class_id]
    if name == "silence":
        return rng.normal(0, 1e-4, n).astype(np.float32)
    if name in ("footsteps", "running"):
        rate = 2.0 if name == "footsteps" else 3.5
        env = (np.sin(2 * math.pi * rate * t) > 0.7).astype(np.float32)
        return (rng.normal(0, 0.4, n) * env).astype(np.float32)
    if name in ("car", "motorcycle", "train"):
        f0 = 80 + class_id * 7
        sig = 0.5 * np.sin(2 * math.pi * f0 * t) + 0.2 * np.sin(2 * math.pi * 2 * f0 * t)
        return (sig + 0.05 * rng.normal(0, 1, n)).astype(np.float32)
    if name == "drone":
        return (0.4 * np.sin(2 * math.pi * 180 * t) + 0.3 * np.sin(2 * math.pi * 420 * t)).astype(np.float32)
    if name in ("speech", "scream", "crowd"):
        f0 = 140 if name == "speech" else 380 if name == "scream" else 220
        form = np.sin(2 * math.pi * f0 * t) * (0.5 + 0.5 * np.sin(2 * math.pi * 4 * t))
        return (form + 0.08 * rng.normal(0, 1, n)).astype(np.float32)
    if name in ("glass_break", "gunshot", "explosion", "door_slam"):
        burst = np.exp(-t * (40 if name != "gunshot" else 80))
        return (rng.normal(0, 1, n) * burst).astype(np.float32)
    if name in ("siren", "alarm", "horn", "phone_ring"):
        f = 600 + 200 * np.sin(2 * math.pi * 2 * t)
        return np.sin(2 * math.pi * f * t).astype(np.float32)
    # broadband industrial / weather
    return rng.normal(0, 0.25, n).astype(np.float32)


def spatialize(
    dry: np.ndarray,
    cfg: AuraConfig,
    *,
    class_id: int,
    rng: np.random.Generator,
    theta: float | None = None,
    phi: float | None = None,
    r: float | None = None,
    vr: float | None = None,
) -> tuple[np.ndarray, SceneLabel]:
    n = int(dry.shape[0])
    theta = float(rng.uniform(-math.pi, math.pi) if theta is None else theta)
    phi = float(rng.uniform(-0.35, 0.55) if phi is None else phi)
    r = float(rng.uniform(0.8, cfg.max_range_m * 0.6) if r is None else r)
    vr = float(rng.uniform(-8.0, 8.0) if vr is None else vr)
    indoor = float(rng.random() < 0.5)
    occl = float(rng.random() < 0.25)
    corner = float(occl and rng.random() < 0.5)
    wall = float(occl and not corner)
    load = float(rng.random() < 0.15)
    stress = float(rng.random() < 0.12)
    mods = np.array(
        [indoor, 1.0 - indoor, occl, corner, wall, load, stress],
        dtype=np.float32,
    )
    dry = dry.astype(np.float32, copy=True)
    # Doppler: resample via time warp
    t = np.arange(n) / cfg.sample_rate
    stretch = 1.0 - vr / SPEED_OF_SOUND
    t_src = np.clip(t * stretch, 0, t[-1])
    dry = np.interp(t_src, t, dry).astype(np.float32)

    # HF air absorption via 1-pole tilt
    att = 10 ** (-air_absorption_db(r, 8000.0) / 20.0)
    dry = dry * (0.4 + 0.6 * att)

    if indoor:
        # cheap late reverb: decaying noise tail mix
        tail = rng.normal(0, 1, n).astype(np.float32)
        env = np.exp(-t * (2.5 if indoor else 8.0)).astype(np.float32)
        dry = 0.85 * dry + 0.15 * np.convolve(dry, env[: min(800, n)], mode="same")[:n]
        _ = tail

    if occl:
        # diffraction: low-pass
        k = np.array([0.15, 0.7, 0.15], dtype=np.float32)
        dry = np.convolve(dry, k, mode="same").astype(np.float32)

    mics = mic_positions(cfg)
    src_xy = np.array([r * math.cos(theta), r * math.sin(theta)], dtype=np.float64)
    src_xyz = np.array([src_xy[0], src_xy[1], r * math.sin(phi)], dtype=np.float64)
    # keep labels consistent with actual geometry
    theta, phi = azimuth_elevation(src_xyz, cfg)

    wav = np.zeros((3, n), dtype=np.float32)
    for i in range(3):
        dist = float(np.linalg.norm(mics[i] - src_xy))
        delay = int(round(dist / SPEED_OF_SOUND * cfg.sample_rate))
        gain = 1.0 / max(dist, 0.2)
        if delay < n:
            wav[i, delay:] = dry[: n - delay] * gain
        wav[i] += 0.01 * rng.normal(0, 1, n).astype(np.float32)

    peak = np.max(np.abs(wav)) + 1e-6
    wav = np.clip(wav / peak * 0.9, -1.0, 1.0)
    label = SceneLabel(class_id, theta, phi, r, vr, mods)
    return wav, label


def render_scene(
    cfg: AuraConfig,
    duration_s: float,
    rng: np.random.Generator | None = None,
    class_id: int | None = None,
    dry: np.ndarray | None = None,
) -> tuple[np.ndarray, SceneLabel]:
    rng = rng or np.random.default_rng()
    n = int(cfg.sample_rate * duration_s)
    if class_id is None:
        class_id = int(rng.integers(0, cfg.n_classes))
    if dry is None:
        dry = _class_signature(class_id, n, cfg.sample_rate, rng)
    elif dry.size != n:
        from aura_micro.audio_io import crop_or_pad

        dry = crop_or_pad(dry, n, rng)
    return spatialize(dry, cfg, class_id=class_id, rng=rng)


def label_to_tensors(wav: np.ndarray, lab: SceneLabel) -> dict[str, torch.Tensor]:
    return {
        "wav": torch.from_numpy(np.ascontiguousarray(wav)),
        "class_id": torch.tensor(lab.class_id, dtype=torch.long),
        "sin_theta": torch.tensor(math.sin(lab.theta), dtype=torch.float32),
        "cos_theta": torch.tensor(math.cos(lab.theta), dtype=torch.float32),
        "sin_phi": torch.tensor(math.sin(lab.phi), dtype=torch.float32),
        "cos_phi": torch.tensor(math.cos(lab.phi), dtype=torch.float32),
        "range_m": torch.tensor(lab.range_m, dtype=torch.float32),
        "vr": torch.tensor(lab.vr, dtype=torch.float32),
        "modifiers": torch.from_numpy(lab.modifiers),
    }


class SyntheticAuraDataset(Dataset):
    def __init__(self, cfg: AuraConfig, size: int = 256, duration_s: float = 0.4, seed: int = 0):
        self.cfg = cfg
        self.size = size
        self.duration_s = duration_s
        self.seed = seed

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        rng = np.random.default_rng(self.seed + idx * 9973)
        wav, lab = render_scene(self.cfg, self.duration_s, rng)
        return label_to_tensors(wav, lab)


def collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    keys = batch[0].keys()
    return {k: torch.stack([b[k] for b in batch], dim=0) for k in keys}
