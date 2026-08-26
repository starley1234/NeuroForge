"""Dependency-free TF-IDF retrieval over the validated owner OpenSCAD corpus."""
from __future__ import annotations
import argparse, json, math, re
from collections import Counter
from pathlib import Path
from typing import Any

_TOKEN = re.compile(r"[A-Za-zА-Яа-яЁё_][A-Za-zА-Яа-яЁё_0-9.-]*")


def tokens(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN.findall(text)]


def retrieve(corpus: Path, annotations: Path | None, query: str, top_k: int = 4) -> list[dict[str, Any]]:
    labels: dict[str, dict[str, Any]] = {}
    if annotations and annotations.exists():
        for line in annotations.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if "annotation" in row: labels[row["id"]] = row["annotation"]
    records = [json.loads(line) for line in (corpus / "records.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    documents: list[Counter[str]] = []; frequencies: Counter[str] = Counter()
    for record in records:
        annotation = labels.get(record["id"], {})
        text = " ".join(str(value) for value in [record.get("prompt", ""), record.get("tags", ""),
                        annotation.get("description", ""), annotation.get("family", ""),
                        annotation.get("requirements", ""), " ".join(p["name"] for p in record.get("parameters", []))])
        vector = Counter(tokens(text)); documents.append(vector); frequencies.update(vector.keys())
    query_vector = Counter(tokens(query)); count = len(documents)
    def weight(term: str, frequency: int) -> float:
        return (1 + math.log(frequency)) * (math.log((count + 1) / (frequencies[term] + 1)) + 1)
    weighted_query = {term: weight(term, value) for term, value in query_vector.items()}
    query_norm = math.sqrt(sum(value * value for value in weighted_query.values())) or 1
    scored = []
    for record, document in zip(records, documents):
        weighted = {term: weight(term, value) for term, value in document.items()}
        norm = math.sqrt(sum(value * value for value in weighted.values())) or 1
        score = sum(weighted_query.get(term, 0) * value for term, value in weighted.items()) / (query_norm * norm)
        scored.append((score, record))
    return [{"score": round(score, 6), "id": record["id"], "prompt": record.get("prompt"),
             "source_code": record["source_code"], "parameters": record.get("parameters", [])}
            for score, record in sorted(scored, key=lambda item: item[0], reverse=True)[:top_k]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--annotations", type=Path); parser.add_argument("--prompt", required=True); parser.add_argument("--top-k", type=int, default=4)
    args = parser.parse_args(); print(json.dumps(retrieve(args.corpus, args.annotations, args.prompt, args.top_k), ensure_ascii=False, indent=2))

if __name__ == "__main__": main()
