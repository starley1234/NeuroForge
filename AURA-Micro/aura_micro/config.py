from __future__ import annotations

from dataclasses import dataclass, field


CLASS_NAMES: tuple[str, ...] = (
    "silence",
    "footsteps",
    "running",
    "car",
    "drone",
    "speech",
    "scream",
    "glass_break",
    "gunshot",
    "industrial_grind",
    "brake_squeal",
    "siren",
    "dog_bark",
    "door_slam",
    "alarm",
    "motorcycle",
    "bicycle",
    "train",
    "helicopter",
    "crowd",
    "wind_noise",
    "rain",
    "thunder",
    "water_flow",
    "keyboard",
    "phone_ring",
    "horn",
    "explosion",
    "chainsaw",
    "unknown",
)

MODIFIER_NAMES: tuple[str, ...] = (
    "indoor",
    "outdoor",
    "occlusion",
    "around_corner",
    "through_wall",
    "high_load",
    "anomaly_stress",
)


@dataclass
class AuraConfig:
    sample_rate: int = 16_000
    n_mics: int = 3
    mic_baseline_m: float = 0.04  # 4 cm equilateral triangle
    n_mels: int = 32
    gcc_bins: int = 32
    win_ms: float = 20.0
    hop_ms: float = 10.0
    n_frames: int = 32  # ~320 ms context window
    hidden: int = 64
    stem_out: int = 32
    dds_channels: tuple[int, ...] = (48, 64, 64)
    dilations: tuple[int, ...] = (1, 2, 4)
    n_classes: int = field(default_factory=lambda: len(CLASS_NAMES))
    n_modifiers: int = field(default_factory=lambda: len(MODIFIER_NAMES))
    max_range_m: float = 50.0
    int8_budget_kb: float = 400.0

    @property
    def win_samples(self) -> int:
        return int(self.sample_rate * self.win_ms / 1000.0)

    @property
    def hop_samples(self) -> int:
        return int(self.sample_rate * self.hop_ms / 1000.0)

    @property
    def feat_dim(self) -> int:
        return self.n_mels * self.n_mics + self.gcc_bins * 3

    @property
    def spatial_dim(self) -> int:
        return self.gcc_bins * 3
