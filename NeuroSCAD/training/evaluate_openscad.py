"""Execute and score OpenSCAD predictions; never use on the public API host."""
from __future__ import annotations
import argparse, json, re, tempfile
from collections import Counter
from pathlib import Path
from neuroscad.openscad_runner import audit_source, compile_file, mesh_metrics

_CODE = re.compile(r"<openscad>\s*(.*?)\s*</openscad>", re.S | re.I)


def extract_code(prediction: str) -> str:
    match = _CODE.search(prediction)
    if match: return match.group(1)
    fenced = re.search(r"```(?:openscad|scad)?\s*(.*?)```", prediction, re.S | re.I)
    return fenced.group(1) if fenced else prediction.strip()


def evaluate(path: Path, output: Path | None = None, timeout: int = 60) -> dict[str, float | int]:
    counts: Counter[str] = Counter(); details = output.open("w", encoding="utf-8") if output else None
    try:
        for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
            if not line.strip(): continue
            row = json.loads(line); counts["total"] += 1; code = extract_code(str(row["prediction"])); issues = audit_source(code, strict=True)
            result = {"id": row.get("id", index), "audit_issues": issues, "compiled": False}
            if issues: counts["audit_failed"] += 1
            else:
                with tempfile.TemporaryDirectory(prefix="neuroscad-eval-") as tmp:
                    source, stl = Path(tmp) / "model.scad", Path(tmp) / "model.stl"; source.write_text(code, encoding="utf-8")
                    try:
                        compile_file(source, stl, timeout); counts["compiled"] += 1; result["compiled"] = True
                        metrics = mesh_metrics(stl); result["mesh"] = metrics
                        if metrics.get("watertight"): counts["watertight"] += 1
                        if metrics.get("components") == 1: counts["single_component"] += 1
                        if float(metrics.get("volume_mm3", 0)) > 0: counts["positive_volume"] += 1
                    except Exception as exc:
                        counts["compile_failed"] += 1; result["error"] = str(exc)[-4000:]
            if details: details.write(json.dumps(result, ensure_ascii=False) + "\n")
    finally:
        if details: details.close()
    total = counts["total"]
    if not total: raise ValueError("prediction file is empty")
    return {**dict(counts), "compile_rate": counts["compiled"] / total,
            "watertight_rate": counts["watertight"] / total,
            "single_component_rate": counts["single_component"] / total}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("predictions", type=Path)
    parser.add_argument("--details", type=Path); parser.add_argument("--timeout", type=int, default=60)
    args = parser.parse_args(); print(json.dumps(evaluate(args.predictions, args.details, args.timeout), indent=2))

if __name__ == "__main__": main()
