"""Profile operation coverage and complexity of an ingested OpenSCAD corpus."""
from __future__ import annotations
import argparse, json, re, statistics
from collections import Counter
from pathlib import Path
from typing import Any

FEATURES = {
    "cube": r"\bcube\s*\(", "cylinder": r"\bcylinder\s*\(", "sphere": r"\bsphere\s*\(",
    "difference": r"\bdifference\s*\(", "union": r"\bunion\s*\(", "intersection": r"\bintersection\s*\(",
    "hull": r"\bhull\s*\(", "minkowski": r"\bminkowski\s*\(", "linear_extrude": r"\blinear_extrude\s*\(",
    "rotate_extrude": r"\brotate_extrude\s*\(", "polygon": r"\bpolygon\s*\(", "offset": r"\boffset\s*\(",
    "module": r"\bmodule\s+[A-Za-z_]", "for_loop": r"\bfor\s*\(", "if": r"\bif\s*\(",
    "scale": r"\bscale\s*\(", "mirror": r"\bmirror\s*\(", "resize": r"\bresize\s*\(",
}


def summary(values: list[int]) -> dict[str, float | int]:
    ordered = sorted(values)
    if not ordered: return {"min": 0, "median": 0, "p90": 0, "max": 0}
    return {"min": ordered[0], "median": statistics.median(ordered),
            "p90": ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))], "max": ordered[-1]}


def profile(corpus: Path) -> dict[str, Any]:
    records = [json.loads(line) for line in (corpus / "records.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    feature_models: Counter[str] = Counter(); feature_calls: Counter[str] = Counter(); lines: list[int] = []; parameters: list[int] = []
    for record in records:
        code = record["source_code"]; lines.append(len(code.splitlines())); parameters.append(len(record.get("parameters", [])))
        for name, pattern in FEATURES.items():
            count = len(re.findall(pattern, code, re.I))
            if count: feature_models[name] += 1; feature_calls[name] += count
    total = len(records)
    return {"models": total, "with_original_prompt": sum(bool(row.get("prompt")) for row in records),
            "splits": dict(Counter(row.get("split") for row in records)),
            "code_lines": summary(lines), "parameters": summary(parameters),
            "feature_model_coverage": {name: {"models": feature_models[name], "rate": round(feature_models[name] / total, 4) if total else 0,
                                                       "calls": feature_calls[name]} for name in FEATURES},
            "warning": "Operation regexes estimate coverage; they are not a full OpenSCAD parser."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path); args = parser.parse_args(); result = profile(args.corpus)
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output: args.output.write_text(text, encoding="utf-8")
    print(text, end="")

if __name__ == "__main__": main()
