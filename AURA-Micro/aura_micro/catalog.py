"""Map public corpus labels (ESC-50) onto AURA-Micro heads."""

from __future__ import annotations

from aura_micro.config import CLASS_NAMES

# Official ESC-50 categories → AURA class (unmapped stay unused / unknown)
ESC50_TO_AURA: dict[str, str] = {
    "dog": "dog_bark",
    "rain": "rain",
    "sea_waves": "water_flow",
    "crying_baby": "scream",
    "door_wood_knock": "door_slam",
    "door_wood_creaks": "door_slam",
    "helicopter": "helicopter",
    "chainsaw": "chainsaw",
    "siren": "siren",
    "car_horn": "horn",
    "engine": "car",
    "train": "train",
    "church_bells": "alarm",
    "clock_alarm": "alarm",
    "airplane": "drone",
    "fireworks": "explosion",
    "thunderstorm": "thunder",
    "footsteps": "footsteps",
    "clapping": "crowd",
    "laughing": "speech",
    "coughing": "speech",
    "breathing": "speech",
    "sneezing": "speech",
    "glass_breaking": "glass_break",
    "hand_saw": "industrial_grind",
    "vacuum_cleaner": "industrial_grind",
    "washing_machine": "industrial_grind",
    "keyboard_typing": "keyboard",
    "mouse_click": "keyboard",
    "wind": "wind_noise",
    "pouring_water": "water_flow",
    "water_drops": "water_flow",
    "crackling_fire": "unknown",
}


def aura_class_id(name: str) -> int:
    return CLASS_NAMES.index(name)


def map_esc50(category: str) -> int | None:
    target = ESC50_TO_AURA.get(category)
    if target is None:
        return None
    return aura_class_id(target)
