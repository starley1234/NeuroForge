"""
Irregular-time benchmark: does the elastic clock (M1) actually help?

Task: next-step prediction of an irregularly sampled continuous signal
(inter-observation gaps dt vary). Two identical ELAST-HOLO signal nets:
  * dt-aware  : dt fed into the elastic clock (native irregular time);
  * dt-blind  : dt=None (the model sees only values, like a regular RNN).
If M1 works, dt-aware should win at equal parameters.

Run from the repo root:
    python research/benchmarks/irregular_time_bench.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch
import torch.nn as nn

from research.arch_v2_core import ElasticHoloConfig, ElasticSignalNet


def make_irregular_sine(num_samples, seq_len, seed=0):
    """x_t = sin(z_t), z_{t+1} = z_t + dt, dt ~ 0.3..1.8 (irregular grid)."""
    g = torch.Generator().manual_seed(seed)
    z = torch.zeros(num_samples, seq_len)
    z[:, 0] = torch.rand(num_samples, generator=g) * 6.0
    dt = 0.3 + torch.rand(num_samples, seq_len, generator=g) * 1.5
    for t in range(1, seq_len):
        z[:, t] = z[:, t - 1] + dt[:, t - 1]
    x = torch.sin(z).unsqueeze(-1) + 0.02 * torch.randn(num_samples, seq_len, 1, generator=g)
    return x, dt  # x: (N, L, 1), dt: (N, L)


def train_net(model, x, dt, epochs, batch_size, dt_aware, lr=3e-3, seed=0):
    torch.manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    mse = nn.MSELoss()
    n = x.size(0)
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        total, batches = 0.0, 0
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            xb, tb = x[idx], (dt[idx] if dt_aware else None)
            optimizer.zero_grad()
            pred, _, _ = model.forward_continuous(xb, dt=tb)
            loss = mse(pred[:, :-1, :], xb[:, 1:, :])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()
            batches += 1
    return total / max(1, batches), time.time() - t0


def eval_mse(model, x, dt, dt_aware):
    model.eval()
    with torch.no_grad():
        pred, _, _ = model.forward_continuous(x, dt=(dt if dt_aware else None))
        mse = ((pred[:, :-1, :] - x[:, 1:, :]) ** 2).mean().item()
    return mse


def main():
    print("=" * 78)
    print("  Irregular-time benchmark: elastic clock (M1) dt-aware vs dt-blind")
    print("=" * 78)

    n_train, n_val, seq_len, epochs = 600, 200, 64, 10
    train_x, train_dt = make_irregular_sine(n_train, seq_len, seed=0)
    val_x, val_dt = make_irregular_sine(n_val, seq_len, seed=99)

    cfg = ElasticHoloConfig(
        vocab_size=0, d_model=32, n_layers=1, d_ff=64, n_read_iters=1,
        dt_min=0.2, dt_max=2.0, use_slow_memory=True,
    )

    results = {}
    for aware in (False, True):
        tag = "dt-aware (M1)" if aware else "dt-blind"
        model = ElasticSignalNet(cfg, input_signal_dim=1, output_signal_dim=1)
        n_params = sum(p.numel() for p in model.parameters())
        loss, secs = train_net(model, train_x, train_dt, epochs=epochs, batch_size=32, dt_aware=aware)
        val_mse = eval_mse(model, val_x, val_dt, dt_aware=aware)
        results[tag] = {"params": n_params, "train_loss": round(loss, 5), "val_mse": round(val_mse, 5)}
        print(f"\n  {tag:<16} params {n_params:>7,} | train MSE {loss:.5f} | val MSE {val_mse:.5f} | {secs:.1f}s")

    aware_mse = results["dt-aware (M1)"]["val_mse"]
    blind_mse = results["dt-blind"]["val_mse"]
    win = (blind_mse - aware_mse) / max(blind_mse, 1e-12) * 100.0
    print("\n" + "=" * 78)
    print(f"  Relative gain of the elastic clock: {win:+.1f}%")
    print("=" * 78)


if __name__ == "__main__":
    main()
