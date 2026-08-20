"""Evaluate model JSONL predictions before checkpoint promotion."""
from __future__ import annotations
import argparse, json
from collections import Counter
from pathlib import Path
from neuroscad.ir import Program, validate


def operation_counts(program: Program) -> Counter[str]:
    return Counter(node.op.value for node in program.nodes)


def score(predictions: Path) -> dict[str, float | int]:
    total = valid_json = valid_ir = exact = 0
    op_overlap = op_total = 0
    for line in predictions.open(encoding="utf-8"):
        row = json.loads(line); total += 1
        prediction = row.get("prediction")
        try:
            data = json.loads(prediction) if isinstance(prediction, str) else prediction
            valid_json += 1
            program = Program.from_dict(data); validate(program); valid_ir += 1
            if "target" in row and data == row["target"]: exact += 1
            if "target" in row:
                expected = operation_counts(Program.from_dict(row["target"])); actual = operation_counts(program)
                op_overlap += sum((expected & actual).values()); op_total += sum(expected.values())
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
    if not total: raise ValueError("prediction file is empty")
    return {
        "samples": total, "json_valid_rate": valid_json / total,
        "ir_valid_rate": valid_ir / total, "exact_match_rate": exact / total,
        "operation_recall": op_overlap / op_total if op_total else 0.0,
        "promotion_gate_pass": valid_ir / total >= 0.99,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("predictions", type=Path)
    args = parser.parse_args(); print(json.dumps(score(args.predictions), indent=2))

if __name__ == "__main__": main()
