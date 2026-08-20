"""Join validated OpenSCAD models and teacher annotations into SFT JSONL."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


def target_text(code: str, plan: list[Any]) -> str:
    return "<design_plan>\n" + json.dumps(plan, ensure_ascii=False) + "\n</design_plan>\n<openscad>\n" + code.rstrip() + "\n</openscad>"


def replace_default(code: str, name: str, value: float) -> str:
    pattern = re.compile(rf"(?m)^(\s*{re.escape(name)}\s*=\s*)-?\d+(?:\.\d+)?(\s*;)")
    updated, count = pattern.subn(rf"\g<1>{value:g}\g<2>", code, count=1)
    if count != 1: raise ValueError(f"cannot replace parameter {name}")
    return updated


def build(corpus: Path, annotations: Path, output: Path, edits_per_model: int = 2) -> dict[str, int]:
    records = {row["id"]: row for row in (json.loads(line) for line in (corpus / "records.jsonl").read_text(encoding="utf-8").splitlines() if line.strip())}
    labels = {}
    for line in annotations.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if "annotation" in row: labels[row["id"]] = row["annotation"]
    output.mkdir(parents=True, exist_ok=True)
    handles = {split: (output / f"{split}.jsonl").open("w", encoding="utf-8") for split in ("train", "validation", "test")}
    counts: Counter[str] = Counter(); seen_prompts: set[tuple[str, str]] = set()
    try:
        for source_id, annotation in labels.items():
            if source_id not in records: continue
            record = records[source_id]; split = record["split"]; code = record["source_code"]
            prompts = [record.get("prompt"), annotation.get("description"), *annotation.get("paraphrases", [])]
            for prompt in prompts:
                if not isinstance(prompt, str) or len(prompt.strip()) < 3: continue
                key = (source_id, " ".join(prompt.lower().split()))
                if key in seen_prompts: continue
                seen_prompts.add(key)
                row = {"id": f"{source_id}:generate:{counts['generation']}", "prompt": prompt.strip(),
                       "target": target_text(code, annotation.get("construction_plan", [])), "task": "generate",
                       "source_id": source_id, "family": annotation.get("family"), "split": split,
                       "license": record.get("license"), "teacher_requirements": annotation.get("requirements", {})}
                handles[split].write(json.dumps(row, ensure_ascii=False) + "\n"); counts["generation"] += 1; counts[split] += 1
            edits = 0
            for parameter in record.get("parameters", []):
                if edits >= edits_per_model: break
                values = [parameter.get("minimum"), parameter.get("maximum")]
                for value in values:
                    if value is None or edits >= edits_per_model: continue
                    try: modified = replace_default(code, parameter["name"], float(value))
                    except ValueError: continue
                    instruction = f"Измени параметр {parameter['name']} на {float(value):g}. Сохрани конструкцию и параметричность.\n<existing_scad>\n{code}\n</existing_scad>"
                    row = {"id": f"{source_id}:edit:{edits}", "prompt": instruction,
                           "target": target_text(modified, [f"Change {parameter['name']} to {float(value):g} without altering topology"]),
                           "task": "edit", "source_id": source_id, "family": annotation.get("family"),
                           "split": split, "license": record.get("license")}
                    handles[split].write(json.dumps(row, ensure_ascii=False) + "\n"); counts["edit"] += 1; counts[split] += 1; edits += 1
    finally:
        for handle in handles.values(): handle.close()
    manifest = {"format": "neuroscad-distillation-v1", "counts": dict(counts),
                "leakage_control": "all augmentations inherit source-level split",
                "target_format": "concise design_plan + OpenSCAD, no private chain-of-thought"}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return dict(counts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True); parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True); parser.add_argument("--edits-per-model", type=int, default=2)
    args = parser.parse_args(); print(json.dumps(build(args.corpus, args.annotations, args.output, args.edits_per_model), indent=2))

if __name__ == "__main__": main()
