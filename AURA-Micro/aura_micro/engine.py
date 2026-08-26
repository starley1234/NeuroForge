from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from aura_micro.config import AuraConfig
from aura_micro.dataset import MixedAuraDataset
from aura_micro.export import memory_report
from aura_micro.frontend import HardwareDSPFrontEnd
from aura_micro.losses import multitask_loss
from aura_micro.model import AuraMicro
from aura_micro.paths import CKPT_PATH
from aura_micro.pipeline import AuraPipeline
from aura_micro.synth import collate


def train(
    *,
    steps: int = 60,
    batch: int = 8,
    lr: float = 2e-3,
    source: str = "auto",
    out: Path | None = None,
    seed: int = 0,
) -> Path:
    cfg = AuraConfig()
    ds = MixedAuraDataset(cfg, size=max(steps * batch, 32), duration_s=0.35, seed=seed, source=source)
    print(f"dataset: {'ESC-50 + synth' if ds.using_esc50 else 'procedural synth'}  n={len(ds)}")
    loader = DataLoader(ds, batch_size=batch, shuffle=True, collate_fn=collate, num_workers=0)
    dsp = HardwareDSPFrontEnd(cfg).eval()
    net = AuraMicro(cfg)
    print("memory", memory_report(net))
    opt = torch.optim.AdamW(net.parameters(), lr=lr)
    net.train()
    step = 0
    while step < steps:
        for batch_data in loader:
            with torch.no_grad():
                feat = dsp(batch_data["wav"])
            pred = net(feat)
            losses = multitask_loss(pred, batch_data)
            step += 1
            if not torch.isfinite(losses["total"]):
                print(f"step {step}: non-finite loss, skip")
                continue
            opt.zero_grad(set_to_none=True)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            if step == 1 or step % 10 == 0 or step >= steps:
                print(
                    f"step {step}/{steps}  "
                    f"loss={losses['total'].item():.3f}  "
                    f"cls={losses['cls'].item():.3f}  doa={losses['doa'].item():.3f}"
                )
            if step >= steps:
                break
    out = Path(out or CKPT_PATH)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": net.state_dict(), "cfg": cfg.__dict__, "steps": steps}, out)
    print("saved", out)
    return out


def load_pipeline(ckpt: Path | None = None) -> AuraPipeline:
    cfg = AuraConfig()
    pipe = AuraPipeline(cfg)
    path = Path(ckpt or CKPT_PATH)
    if path.is_file():
        blob = torch.load(path, map_location="cpu", weights_only=False)
        pipe.net.load_state_dict(blob["state_dict"])
        print(f"loaded {path}")
    else:
        print("no checkpoint — random weights (train first)")
    pipe.eval()
    return pipe
