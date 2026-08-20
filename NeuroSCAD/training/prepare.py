"""Generate leakage-resistant, reproducible prompt → CSG-IR JSONL datasets."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from neuroscad.templates import camera_pipe_bracket

RU = (
    "Разрезной хомут на трубу {tube:g} мм, крепёж М{screw:g}",
    "Кронштейн для камеры на трубу диаметром {tube:g} мм с винтом М{screw:g}",
    "Сделай зажим для трубы {tube:g} мм, стяжной болт М{screw:g}",
)
EN = (
    "Split clamp for a {tube:g} mm tube with an M{screw:g} fastener",
    "Camera bracket on a {tube:g} mm pipe, secured by an M{screw:g} screw",
    "Make a pipe clamp diameter {tube:g} mm with M{screw:g} bolt",
)


def _split(group: str) -> str:
    """Keep every paraphrase of one dimension tuple in exactly one split."""
    bucket = int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % 100
    return "train" if bucket < 90 else ("validation" if bucket < 95 else "test")


def prepare(output: Path, samples: int, seed: int) -> dict[str, int]:
    rng = random.Random(seed)
    output.mkdir(parents=True, exist_ok=True)
    handles = {name: (output / f"{name}.jsonl").open("w", encoding="utf-8") for name in ("train", "validation", "test")}
    counts: Counter[str] = Counter()
    seen: set[str] = set()
    try:
        attempts = 0
        while sum(counts.values()) < samples:
            attempts += 1
            if attempts > samples * 30: raise RuntimeError("requested more unique samples than generator can provide")
            tube = rng.randrange(16, 161) / 2  # 8.0 .. 80.0
            screw = float(rng.randrange(3, 11))
            language = rng.choice(("ru", "en")); template_id = rng.randrange(3)
            wall = rng.randrange(4, 17) / 2
            width = float(rng.randrange(10, 41))
            key = f"{tube}:{screw}:{language}:{template_id}:{wall}:{width}"
            if key in seen: continue
            seen.add(key)
            templates = RU if language == "ru" else EN
            prompt = templates[template_id].format(tube=tube, screw=screw)
            program = camera_pipe_bracket(prompt)
            payload = program.to_dict()
            # Add design variation while preserving graph topology.
            for parameter in payload["parameters"]:
                if parameter["name"] == "wall": parameter["default"] = wall
                if parameter["name"] == "clamp_width": parameter["default"] = width
            group = f"tube={tube}:screw={screw}:wall={wall}:width={width}"
            split = _split(group)
            row = {
                "id": hashlib.sha256(key.encode()).hexdigest()[:20],
                "prompt": prompt, "target": payload, "language": language,
                "family": "split_pipe_clamp", "group": group,
                "source": "neuroscad-synthetic-v1", "license": "Apache-2.0",
            }
            handles[split].write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            counts[split] += 1
    finally:
        for handle in handles.values(): handle.close()
    manifest = {"seed": seed, "requested": samples, "counts": dict(counts), "split_strategy": "sha256 dimension-group 90/5/5"}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return dict(counts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/processed"))
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 10 <= args.samples <= 1_000_000: parser.error("--samples must be between 10 and 1,000,000")
    print(json.dumps(prepare(args.output, args.samples, args.seed), indent=2))

if __name__ == "__main__": main()
