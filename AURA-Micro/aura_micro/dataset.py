from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from aura_micro.audio_io import load_mono_wav
from aura_micro.config import AuraConfig
from aura_micro.download import list_esc50_items
from aura_micro.synth import SyntheticAuraDataset, label_to_tensors, render_scene


class MixedAuraDataset(Dataset):
    """ESC-50 dry clips (if present) spatialized + procedural fallback."""

    def __init__(
        self,
        cfg: AuraConfig,
        size: int,
        duration_s: float = 0.4,
        seed: int = 0,
        source: str = "auto",
        esc50_root: Path | None = None,
    ):
        self.cfg = cfg
        self.size = size
        self.duration_s = duration_s
        self.seed = seed
        self.items = list_esc50_items(esc50_root) if source != "synth" else []
        if source == "esc50" and not self.items:
            raise FileNotFoundError("ESC-50 not found. Run: python -m aura_micro download")
        self.fallback = SyntheticAuraDataset(cfg, size=size, duration_s=duration_s, seed=seed)

    def __len__(self) -> int:
        return self.size

    @property
    def using_esc50(self) -> bool:
        return bool(self.items)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        rng = np.random.default_rng(self.seed + idx * 9973)
        if self.items and rng.random() < 0.75:
            path, cid, _ = self.items[int(rng.integers(0, len(self.items)))]
            dry = load_mono_wav(path, self.cfg.sample_rate)
            wav, lab = render_scene(self.cfg, self.duration_s, rng, class_id=cid, dry=dry)
            return label_to_tensors(wav, lab)
        return self.fallback[idx]
