"""Фаза 2: обучение FNO-суррогата FEM (критик прочности)."""
from __future__ import annotations

import argparse
import os
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from ..config import FNOConfig
from ..data.dataset import FieldDataset
from ..surrogates.fno import FEMCritic


def train(data: str, out: str, cfg: Optional[FNOConfig] = None, epochs: int = 5,
          batch_size: int = 4, lr: float = 1e-3, device: str = "cpu",
          val_split: float = 0.15, log_every: int = 5) -> Dict[str, float]:
    ds = FieldDataset(data)
    if len(ds) < 2:
        raise SystemExit("нужно ≥2 сэмпла — запустите flywheel")
    grid = ds.occ.shape[-1]
    cfg = cfg or FNOConfig(grid=grid, modes=min(8, grid // 2), width=24, depth=3)
    n_val = max(1, int(len(ds) * val_split))
    train_ds, val_ds = random_split(ds, [len(ds) - n_val, n_val],
                                    generator=torch.Generator().manual_seed(0))
    dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    vl = DataLoader(val_ds, batch_size=batch_size)

    model = FEMCritic(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    history: Dict[str, float] = {}

    for epoch in range(epochs):
        model.train()
        tot = 0.0
        for occ, load, sigma, scal in dl:
            occ, load, sigma, scal = (t.to(device) for t in (occ, load, sigma, scal))
            field, scalars = model(occ, load)
            loss = F.mse_loss(field, sigma) + 0.1 * F.mse_loss(scalars, scal)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss) * occ.shape[0]

        model.eval()
        vtot, rel = 0.0, 0.0
        with torch.no_grad():
            for occ, load, sigma, scal in vl:
                occ, load, sigma, scal = (t.to(device) for t in (occ, load, sigma, scal))
                field, scalars = model(occ, load)
                vtot += float(F.mse_loss(field, sigma)) * occ.shape[0]
                rel += float((field - sigma).norm() / sigma.norm().clamp(min=1e-6)) * occ.shape[0]
        history = {"epoch": epoch, "train_mse": tot / len(train_ds),
                   "val_mse": vtot / len(val_ds), "val_rel_l2": rel / len(val_ds)}
        if epoch % log_every == 0 or epoch == epochs - 1:
            print(f"epoch {epoch:3d} train={history['train_mse']:.5f} "
                  f"val={history['val_mse']:.5f} relL2={history['val_rel_l2']:.4f}", flush=True)

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    torch.save({"model": model.state_dict(), "config": cfg.__dict__}, out)
    print(f"FNO сохранён: {out}")
    return history


def main() -> None:
    ap = argparse.ArgumentParser(description="Обучение FNO-суррогата FEM")
    ap.add_argument("--data", default="artifacts/flywheel")
    ap.add_argument("--out", default="artifacts/checkpoints/fno.pt")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    train(a.data, a.out, epochs=a.epochs, batch_size=a.batch_size, lr=a.lr, device=a.device)


if __name__ == "__main__":
    main()
