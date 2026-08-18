"""
Обучение byte-level BPE токенизатора на корпусах проекта.

Корпуса:
  * data/russian_words.txt — словарь русских словоформ (~1.5M строк, cp1251);
  * data/data_benchmark.txt — TinyShakespeare (UTF-8).

Результат: registry/tokenizers/<name>/tokenizer.json + tokenizer_meta.json
(источники, размер словаря, степень сжатия на тестовых строках).

Запуск: python scripts/train_tokenizer.py [--vocab-size 8192]
"""
import json
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse

from fractal_holonet.tokenizer import BpeTokenizer


def prepare_russian_words(src: Path, dst: Path, max_lines: int = 400_000):
    """Конвертирует cp1251-словарь словоформ в UTF-8 (сэмпл)."""
    with open(src, "rb") as fin, open(dst, "w", encoding="utf-8") as fout:
        for i, line in enumerate(fin):
            if i >= max_lines:
                break
            fout.write(line.decode("cp1251", errors="replace").strip() + "\n")
    return dst


def compression_stats(tok: BpeTokenizer, samples):
    rows = []
    for name, text in samples:
        n_bytes = len(text.encode("utf-8"))
        n_tokens = len(tok.encode(text, add_bos=False))
        rows.append({"sample": name, "utf8_bytes": n_bytes, "bpe_tokens": n_tokens})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab-size", type=int, default=8192)
    ap.add_argument("--name", default=None, help="каталог токенизатора в registry/tokenizers/")
    ap.add_argument("--min-frequency", type=int, default=2)
    ap.add_argument("--max-word-lines", type=int, default=400_000)
    args = ap.parse_args()

    name = args.name or f"bpe-{args.vocab_size}"
    out_dir = ROOT / "registry" / "tokenizers" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "tokenizer.json"

    russian_words = ROOT / "data" / "russian_words.txt"
    data_benchmark = ROOT / "data" / "data_benchmark.txt"

    print("=" * 72)
    print(f"  ОБУЧЕНИЕ BPE-ТОКЕНИЗАТОРА (vocab={args.vocab_size})")
    print("=" * 72)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_words = Path(tmp) / "russian_words.utf8.txt"
        print(f"Конвертация {russian_words.name} (cp1251 -> UTF-8, до {args.max_word_lines:,} строк)...")
        t0 = time.time()
        prepare_russian_words(russian_words, tmp_words, max_lines=args.max_word_lines)
        print(f"  готово за {time.time()-t0:.1f}s, размер {tmp_words.stat().st_size:,} байт")

        files = [str(tmp_words), str(data_benchmark)]
        print("Обучение byte-level BPE...")
        t0 = time.time()
        tok = BpeTokenizer.train_from_files(
            files, vocab_size=args.vocab_size, path=str(out_path), encoding="utf-8",
            min_frequency=args.min_frequency,
        )
        print(f"  обучено за {time.time()-t0:.1f}s")

    print(f"\nСловарь: {tok.vocab_size} токенов | pad={tok.pad_token_id} bos={tok.bos_token_id} "
          f"eos={tok.eos_token_id} unk={tok.unk_token_id}")

    samples = [
        ("русский текст", "Искусственный интеллект — это комплекс технологических и программных решений. Голографическая память хранит ассоциации."),
        ("английский текст", "Fractal-HoloNet achieves linear O(N) context complexity and constant O(1) inference state."),
        ("пушкин", "Мороз и солнце; день чудесный! Еще ты дремлешь, друг прелестный."),
    ]
    stats = compression_stats(tok, samples)
    print("\nСтепень сжатия (UTF-8 байт -> BPE токены):")
    for r in stats:
        ratio = r["utf8_bytes"] / max(1, r["bpe_tokens"])
        print(f"  {r['sample']:<22} {r['utf8_bytes']:>4} байт -> {r['bpe_tokens']:>4} токенов (x{ratio:.2f})")

    # демонстрация сегментации
    demo = "Голографическая память нейросети сохраняет ассоциации между словами в комплексном пространстве чисел."
    ids = tok.encode(demo)
    pieces = [tok.decode([i]) for i in ids]
    print(f"\nСегментация: {' | '.join(pieces[:20])} ...")
    assert tok.decode(ids) == demo, "BPE roundtrip failed!"

    meta = {
        "name": name,
        "type": "byte-level-bpe",
        "vocab_size": tok.vocab_size,
        "pad_token_id": tok.pad_token_id,
        "bos_token_id": tok.bos_token_id,
        "eos_token_id": tok.eos_token_id,
        "unk_token_id": tok.unk_token_id,
        "sources": {
            "russian_words.txt": {"encoding": "cp1251", "max_lines": args.max_word_lines},
            "data_benchmark.txt": {"encoding": "utf-8", "full": True},
        },
        "min_frequency": args.min_frequency,
        "compression": stats,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    with open(out_dir / "tokenizer_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"\nТокенизатор сохранён: {out_path}")
    print(f"Метаданные: {out_dir / 'tokenizer_meta.json'}")


if __name__ == "__main__":
    main()
