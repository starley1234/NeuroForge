"""
MQAR benchmark: v1 (diagonal phase recurrence) vs ELAST-HOLO v2.

Multi-Query Associative Recall (Arora et al., 2024) probes whether an
architecture can retrieve values written earlier in the context under
distraction. Diagonal recurrent models fail this class; the v2 delta write
(M3) + iterative read (M5) are designed to solve it.

Run from the repo root:
    python research/benchmarks/mqar_benchmark.py [--quick]

Reports:
  * parameter counts (v2 is kept at or below v1 params),
  * val loss and query accuracy at train length,
  * length extrapolation (train 64, test 128/256).
"""
import os
import sys
import json
import time
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch
import torch.nn as nn

from fractal_holonet.core import ProductionFractalHoloNet, FractalHoloNetConfig
from research.arch_v2_core import ElasticHoloNet, ElasticHoloConfig


def make_mqar(num_samples, seq_len, num_pairs, n_keys=16, n_vals=16, vocab_size=64, seed=0):
    """
    Tokens: keys 0..n_keys-1, values n_keys..n_keys+n_vals-1,
    noise n_keys+n_vals..vocab-2, query symbol vocab-1.
    Layout: [noise, k1, v1, ..., kM, vM, noise ..., QUERY, k_q] -> target v_q.
    Returns x (N, L), y (N, L).
    """
    g = torch.Generator().manual_seed(seed)
    noise_lo, noise_hi = n_keys + n_vals, vocab_size - 1
    x = torch.randint(noise_lo, noise_hi, (num_samples, seq_len), generator=g)
    y = torch.roll(x, -1, dims=1)
    y[:, -1] = noise_lo
    for i in range(num_samples):
        keys = torch.randperm(n_keys, generator=g)[:num_pairs]
        vals = torch.randint(n_keys, n_keys + n_vals, (num_pairs,), generator=g)
        locs = (torch.randperm(seq_len // 2 - 1, generator=g)[:num_pairs] * 2).sort().values
        for j in range(num_pairs):
            x[i, locs[j]] = keys[j]
            x[i, locs[j] + 1] = vals[j]
        q = torch.randint(num_pairs, (1,), generator=g).item()
        x[i, -2] = vocab_size - 1  # QUERY
        x[i, -1] = keys[q]
        y[i, -1] = vals[q]
    return x, y


def query_accuracy(model, x, y, batch_size=64):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for i in range(0, x.size(0), batch_size):
            xb, yb = x[i : i + batch_size], y[i : i + batch_size]
            logits, _ = model(xb)
            pred = logits[:, -1, :].argmax(dim=-1)
            correct += (pred == yb[:, -1]).sum().item()
            total += xb.size(0)
    return correct / max(1, total)


def train_model(model, train_x, train_y, epochs, batch_size, lr=3e-3, seed=0, query_weight=8.0):
    """Standard MQAR training with query-position emphasis (identical for both
    models): the last position (the associative answer) is weighted x8 so the
    recall signal is not diluted 1/L across the sequence."""
    torch.manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(reduction="none")
    n = train_x.size(0)
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        total_loss, batches = 0.0, 0
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            xb, yb = train_x[idx], train_y[idx]
            optimizer.zero_grad()
            logits, _ = model(xb)
            l = criterion(logits.view(-1, logits.size(-1)), yb.view(-1)).view(xb.size(0), -1)
            w = torch.ones_like(l)
            w[:, -1] = query_weight
            loss = (l * w).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            batches += 1
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            print(f"    epoch {epoch+1:02d}/{epochs} | loss {total_loss/max(1,batches):.4f}")
    return time.time() - t0


def build_v1(vocab_size=64, d_model=48, n_layers=2, d_ff=256):
    cfg = FractalHoloNetConfig(
        vocab_size=vocab_size, d_model=d_model, n_layers=n_layers, d_ff=d_ff,
        dropout=0.0, tie_word_embeddings=True,
    )
    return ProductionFractalHoloNet(cfg), cfg


def build_v2(vocab_size=64, d_model=48, n_layers=2, d_ff=96, n_read_iters=3, **kw):
    cfg = ElasticHoloConfig(
        vocab_size=vocab_size, d_model=d_model, n_layers=n_layers, d_ff=d_ff,
        n_read_iters=n_read_iters, dropout=0.0, **kw,
    )
    return ElasticHoloNet(cfg), cfg


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="tiny budget for CI smoke")
    args = ap.parse_args()

    if args.quick:
        n_train, n_val, seq_len, epochs, num_pairs = 300, 100, 24, 20, 2
    else:
        n_train, n_val, seq_len, epochs, num_pairs = 600, 200, 24, 60, 3

    print("=" * 78)
    print("  MQAR: Fractal-HoloNet v1 (diagonal) vs ELAST-HOLO v2 (delta write + iter read)")
    print("=" * 78)

    train_x, train_y = make_mqar(n_train, seq_len, num_pairs, seed=0)
    val_x, val_y = make_mqar(n_val, seq_len, num_pairs, seed=1234)
    ext48_x, ext48_y = make_mqar(200, 48, num_pairs, seed=42)
    ext96_x, ext96_y = make_mqar(150, 96, num_pairs, seed=43)

    results = {}
    for name, (model, _cfg) in {
        "v1-diagonal": build_v1(),
        "v2-elast-holo": build_v2(),
    }.items():
        p = count_params(model)
        print(f"\n--- {name} | params: {p:,} ---")
        secs = train_model(model, train_x, train_y, epochs=epochs, batch_size=32, lr=3e-3, seed=0)
        acc = query_accuracy(model, val_x, val_y)
        acc48 = query_accuracy(model, ext48_x, ext48_y)
        acc96 = query_accuracy(model, ext96_x, ext96_y)
        results[name] = {
            "params": p,
            "train_sec": round(secs, 1),
            f"query_acc_L{seq_len}": round(acc, 4),
            "query_acc_L48_extrap": round(acc48, 4),
            "query_acc_L96_extrap": round(acc96, 4),
        }
        print(
            f"  query acc (L={seq_len}): {acc:.3f} | "
            f"extrap L=48: {acc48:.3f} | L=96: {acc96:.3f} | train {secs:.1f}s"
        )

    print("\n" + "=" * 78)
    print(f"  {'model':<18} {'params':>9} {f'acc L{seq_len}':>9} {'acc L48':>9} {'acc L96':>9}")
    for name, r in results.items():
        print(
            f"  {name:<18} {r['params']:>9,} {r[f'query_acc_L{seq_len}']:>9.3f} "
            f"{r['query_acc_L48_extrap']:>9.3f} {r['query_acc_L96_extrap']:>9.3f}"
        )

    out_path = os.path.join(os.path.dirname(__file__), "mqar_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
