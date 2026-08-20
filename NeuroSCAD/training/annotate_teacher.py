"""Create descriptions and concise construction plans with an OpenAI-compatible teacher."""
from __future__ import annotations

import argparse
import base64
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

REQUIRED = {"description", "family", "requirements", "construction_plan", "paraphrases", "quality_notes"}
SYSTEM = """You annotate trusted OpenSCAD training examples. Describe only geometry supported by code and rendered views. Return strict JSON with: description (clear user request with dimensions when observable), family, requirements (object), construction_plan (short list of modeling steps, not hidden reasoning), paraphrases (array with concise, expert, and colloquial requests in Russian and English), quality_notes (array). Never invent loads, materials, standards, or dimensions not present in code."""


def image_url(path: Path) -> str:
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"


def build_content(record: dict[str, Any], root: Path, with_images: bool = True) -> list[dict[str, Any]]:
    text = f"Existing owner prompt: {record.get('prompt') or 'none'}\nCustomizer parameters: {json.dumps(record.get('parameters', []), ensure_ascii=False)}\nOpenSCAD code:\n```openscad\n{record['source_code'][:50000]}\n```"
    content: list[dict[str, Any]] = [{"type": "text", "text": text}]
    if with_images:
        for relative in record.get("renders", [])[:4]:
            path = root / relative
            if path.exists(): content.append({"type": "image_url", "image_url": {"url": image_url(path), "detail": "low"}})
    return content


def parse_annotation(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0]
        if cleaned.lstrip().startswith("json"): cleaned = cleaned.lstrip()[4:].lstrip()
    data = json.loads(cleaned)
    if not isinstance(data, dict) or not REQUIRED <= data.keys():
        raise ValueError(f"teacher response lacks fields: {sorted(REQUIRED - set(data if isinstance(data, dict) else {}))}")
    if not isinstance(data["paraphrases"], list) or len(data["paraphrases"]) < 2:
        raise ValueError("teacher must provide at least two paraphrases")
    return data


def call_teacher(endpoint: str, api_key: str, model: str, content: list[dict[str, Any]], timeout: int) -> dict[str, Any]:
    payload = {"model": model, "temperature": 0.2, "response_format": {"type": "json_object"},
               "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}]}
    request = urllib.request.Request(endpoint, data=json.dumps(payload).encode(), headers={"authorization": f"Bearer {api_key}", "content-type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.loads(response.read())
    return parse_annotation(result["choices"][0]["message"]["content"])


def annotate(corpus: Path, output: Path, endpoint: str, model: str, api_key: str,
             limit: int | None = None, with_images: bool = True, retries: int = 2, timeout: int = 120) -> dict[str, int]:
    done: set[str] = set()
    if output.exists():
        for line in output.read_text(encoding="utf-8").splitlines():
            try: done.add(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError): pass
    records_path = corpus / "records.jsonl"; counts = {"annotated": 0, "skipped": 0, "failed": 0}
    records = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if limit is not None: records = records[:limit]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as target:
        for record in records:
            if record["id"] in done: counts["skipped"] += 1; continue
            error = None
            for attempt in range(retries + 1):
                try:
                    annotation = call_teacher(endpoint, api_key, model, build_content(record, corpus, with_images), timeout)
                    target.write(json.dumps({"id": record["id"], "model": model, "annotation": annotation}, ensure_ascii=False) + "\n"); target.flush()
                    counts["annotated"] += 1; error = None; break
                except (ValueError, KeyError, urllib.error.URLError, TimeoutError) as exc:
                    error = str(exc); time.sleep(min(2 ** attempt, 8))
            if error is not None:
                target.write(json.dumps({"id": record["id"], "model": model, "error": error}, ensure_ascii=False) + "\n"); target.flush(); counts["failed"] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoint", required=True, help="exact OpenAI-compatible chat-completions URL")
    parser.add_argument("--model", required=True); parser.add_argument("--api-key-env", default="TEACHER_API_KEY")
    parser.add_argument("--limit", type=int); parser.add_argument("--no-images", action="store_true"); parser.add_argument("--retries", type=int, default=2)
    args = parser.parse_args(); api_key = os.getenv(args.api_key_env)
    if not api_key: parser.error(f"environment variable {args.api_key_env} is not set")
    print(json.dumps(annotate(args.corpus, args.output, args.endpoint, args.model, api_key, args.limit, not args.no_images, args.retries), indent=2))

if __name__ == "__main__": main()
