"""Generate leakage-resistant, reproducible prompt → CSG-IR JSONL datasets."""
from __future__ import annotations
import argparse, hashlib, json, random
from collections import Counter
from pathlib import Path
from typing import Any
from neuroscad.ir import Program, validate
from neuroscad.templates import bushing, camera_pipe_bracket, mounting_plate

PROMPTS = {
    "split_pipe_clamp": {
        "ru": ("Разрезной хомут на трубу {tube:g} мм, крепёж М{screw:g}", "Кронштейн для камеры на трубу диаметром {tube:g} мм с винтом М{screw:g}"),
        "en": ("Split clamp for a {tube:g} mm tube with an M{screw:g} fastener", "Camera bracket on a {tube:g} mm pipe, secured by an M{screw:g} screw"),
    },
    "mounting_plate": {
        "ru": ("Монтажная пластина {length:g}x{width:g}x{thickness:g} мм с четырьмя отверстиями М{hole:g}", "Сделай пластину {length:g}×{width:g}×{thickness:g}, 4 отверстия под М{hole:g}"),
        "en": ("Mounting plate {length:g}x{width:g}x{thickness:g} mm with four M{hole:g} holes", "Make a {length:g}x{width:g}x{thickness:g} plate, 4 holes for M{hole:g}"),
    },
    "bushing": {
        "ru": ("Втулка {outer:g}/{inner:g}, высота {height:g} мм", "Цилиндрическая втулка {outer:g}/{inner:g} длиной {height:g} мм"),
        "en": ("Bushing {outer:g}/{inner:g}, height {height:g} mm", "Cylindrical bushing {outer:g}/{inner:g}, length {height:g} mm"),
    },
}


def _split(group: str) -> str:
    bucket = int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % 100
    return "train" if bucket < 90 else ("validation" if bucket < 95 else "test")


def _set_defaults(payload: dict[str, Any], values: dict[str, float]) -> None:
    for parameter in payload["parameters"]:
        if parameter["name"] in values: parameter["default"] = values[parameter["name"]]


def _sample(rng: random.Random) -> tuple[str, dict[str, float], dict[str, float]]:
    family = rng.choice(tuple(PROMPTS))
    if family == "split_pipe_clamp":
        prompt_values = {"tube": rng.randrange(16, 161) / 2, "screw": float(rng.randrange(3, 9))}
        defaults = {"wall": rng.randrange(4, 17) / 2, "clamp_width": float(rng.randrange(10, 41))}
    elif family == "mounting_plate":
        prompt_values = {"length": float(rng.randrange(40, 201)), "width": float(rng.randrange(30, 151)), "thickness": rng.randrange(4, 25) / 2, "hole": float(rng.randrange(3, 9))}
        defaults = {"edge_offset": float(rng.randrange(7, 13))}
    else:
        inner, wall = rng.randrange(4, 61) / 2, rng.randrange(3, 17) / 2
        prompt_values = {"outer": inner + 2 * wall, "inner": inner, "height": rng.randrange(4, 101) / 2}
        defaults = {}
    return family, prompt_values, defaults


def prepare(output: Path, samples: int, seed: int) -> dict[str, int]:
    rng = random.Random(seed); output.mkdir(parents=True, exist_ok=True)
    handles = {name: (output / f"{name}.jsonl").open("w", encoding="utf-8") for name in ("train", "validation", "test")}
    counts: Counter[str] = Counter(); families: Counter[str] = Counter(); seen: set[str] = set()
    generators = {"split_pipe_clamp": camera_pipe_bracket, "mounting_plate": mounting_plate, "bushing": bushing}
    try:
        attempts = 0
        while sum(counts.values()) < samples:
            attempts += 1
            if attempts > samples * 50: raise RuntimeError("requested more unique samples than generator can provide")
            family, prompt_values, defaults = _sample(rng)
            language = rng.choice(("ru", "en")); template_id = rng.randrange(len(PROMPTS[family][language]))
            geometry = {**prompt_values, **defaults}
            group = family + ":" + ":".join(f"{k}={geometry[k]}" for k in sorted(geometry))
            key = f"{group}:{language}:{template_id}"
            if key in seen: continue
            seen.add(key)
            prompt = PROMPTS[family][language][template_id].format(**prompt_values)
            payload = generators[family](prompt).to_dict(); _set_defaults(payload, defaults)
            validate(Program.from_dict(payload))
            split = _split(group)
            row = {"id": hashlib.sha256(key.encode()).hexdigest()[:20], "prompt": prompt, "target": payload,
                   "language": language, "family": family, "group": group,
                   "source": "neuroscad-synthetic-v2", "license": "Apache-2.0"}
            handles[split].write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            counts[split] += 1; families[family] += 1
    finally:
        for handle in handles.values(): handle.close()
    manifest = {"version": 2, "seed": seed, "requested": samples, "counts": dict(counts),
                "families": dict(families), "split_strategy": "sha256 geometry-family group 90/5/5"}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return dict(counts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/processed")); parser.add_argument("--samples", type=int, default=10_000); parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 10 <= args.samples <= 1_000_000: parser.error("--samples must be between 10 and 1,000,000")
    print(json.dumps(prepare(args.output, args.samples, args.seed), indent=2))

if __name__ == "__main__": main()
