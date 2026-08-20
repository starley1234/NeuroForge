"""Audit, compile, measure and render an owner-supplied OpenSCAD corpus."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from neuroscad.openscad_runner import (
    audit_source, compile_file, extract_customizer_parameters, mesh_metrics,
    openscad_version, render_views,
)


def split_for(source_id: str) -> str:
    bucket = int(source_id[:8], 16) % 100
    return "train" if bucket < 80 else ("validation" if bucket < 90 else "test")


def sidecar_for(source: Path) -> dict[str, Any]:
    path = source.with_suffix(".json")
    if not path.exists(): return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict): raise ValueError("sidecar must be a JSON object")
    return data


def ingest(input_dir: Path, output_dir: Path, license_name: str, strict: bool = True,
           dry_run: bool = False, views: int = 8, limit: int | None = None) -> dict[str, Any]:
    sources = sorted(input_dir.rglob("*.scad"))
    if limit is not None: sources = sources[:limit]
    output_dir.mkdir(parents=True, exist_ok=True)
    source_dir, mesh_dir, render_dir = output_dir / "sources", output_dir / "meshes", output_dir / "renders"
    source_dir.mkdir(exist_ok=True); mesh_dir.mkdir(exist_ok=True); render_dir.mkdir(exist_ok=True)
    counts: Counter[str] = Counter(); seen: set[str] = set()
    selected_views = ("iso_front", "iso_back", "front", "right", "back", "left", "top", "bottom")[:views]

    with (output_dir / "records.jsonl").open("w", encoding="utf-8") as records, (output_dir / "quarantine.jsonl").open("w", encoding="utf-8") as quarantine:
        for source in sources:
            try:
                code = source.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                code = source.read_text(encoding="utf-8", errors="replace")
            source_id = hashlib.sha256(code.encode("utf-8")).hexdigest()
            if source_id in seen:
                counts["duplicate_source"] += 1; continue
            seen.add(source_id)
            issues = audit_source(code, strict=strict)
            base = {"id": source_id, "original_path": str(source.relative_to(input_dir)), "issues": issues}
            if issues:
                quarantine.write(json.dumps({**base, "status": "rejected_by_audit"}, ensure_ascii=False) + "\n")
                counts["rejected"] += 1; continue
            copied = source_dir / f"{source_id}.scad"; copied.write_text(code, encoding="utf-8")
            metadata = sidecar_for(source)
            group_id = str(metadata.get("group_id", source_id))
            split_key = source_id if group_id == source_id else hashlib.sha256(group_id.encode()).hexdigest()
            record: dict[str, Any] = {
                **base, "status": "audited" if dry_run else "valid", "split": split_for(split_key),
                "group_id": group_id, "source_path": str(copied.relative_to(output_dir)), "source_code": code,
                "parameters": extract_customizer_parameters(code), "license": metadata.get("license", license_name),
                "prompt": metadata.get("prompt"), "tags": metadata.get("tags", []),
            }
            if not dry_run:
                stl = mesh_dir / f"{source_id}.stl"
                try:
                    record["compile"] = compile_file(source, stl)
                    record["mesh"] = mesh_metrics(stl)
                    record["geometry_sha256"] = hashlib.sha256(stl.read_bytes()).hexdigest()
                    generated = render_views(source, render_dir / source_id, views=selected_views)
                    record["renders"] = [str(Path(path).relative_to(output_dir)) for path in generated]
                except Exception as exc:
                    quarantine.write(json.dumps({**base, "status": "execution_failed", "error": str(exc)[-8000:]}, ensure_ascii=False) + "\n")
                    counts["execution_failed"] += 1; continue
            records.write(json.dumps(record, ensure_ascii=False) + "\n")
            counts[record["split"]] += 1
    manifest = {
        "format": "neuroscad-openscad-corpus-v1", "input": str(input_dir), "strict": strict,
        "dry_run": dry_run, "views": list(selected_views), "openscad_version": openscad_version(),
        "counts": dict(counts), "unique_sources": len(seen),
        "split_policy": "sidecar group_id (or source SHA-256) 80/10/10 before augmentation",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--license", default="owner-supplied"); parser.add_argument("--trusted", action="store_true", help="allow include/use/import; run only in an isolated container")
    parser.add_argument("--dry-run", action="store_true"); parser.add_argument("--views", type=int, choices=range(1, 9), default=8); parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if not args.input.is_dir(): parser.error("--input must be a directory")
    print(json.dumps(ingest(args.input, args.output, args.license, not args.trusted, args.dry_run, args.views, args.limit), ensure_ascii=False, indent=2))

if __name__ == "__main__": main()
