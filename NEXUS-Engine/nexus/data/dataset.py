"""Датасеты поверх артефактов маховика."""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .tokenizer import DEFAULT_TOKENIZER, ASTBPETokenizer


class ScadCorpus(Dataset):
    """Пары «ТЗ → OpenSCAD код» для языкового предобучения ядра."""

    def __init__(self, path: str, tokenizer: Optional[ASTBPETokenizer] = None,
                 seq_len: int = 512, only_valid: bool = True):
        self.tok = tokenizer or DEFAULT_TOKENIZER
        self.seq_len = seq_len
        self.records: List[Dict] = []
        with open(os.path.join(path, "dataset.jsonl"), encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                if only_valid and not rec.get("valid"):
                    continue
                self.records.append(rec)

    def __len__(self) -> int:
        return len(self.records)

    def _sequence(self, rec: Dict) -> List[int]:
        text = (
            rec["spec"]
            + "<scad>" + rec["code"]
            + "<fem>" + json.dumps({
                "mass_g": round(rec.get("mass", {}).get("mass_g", 0.0), 3),
                "sigma_max_mpa": round(rec.get("fem", {}).get("max_stress_pa", 0.0) / 1e6, 3),
                "safety_factor": round(rec.get("fem", {}).get("safety_factor", 0.0), 3),
            }, ensure_ascii=False)
        )
        return self.tok.encode(text, bos=True, eos=True)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        ids = self._sequence(self.records[i])[: self.seq_len]
        pad = self.tok.pad_id
        ids = ids + [pad] * (self.seq_len - len(ids))
        x = torch.tensor(ids, dtype=torch.long)
        return {"tokens": x, "targets": x.clone()}


class FieldDataset(Dataset):
    """(occupancy, вектор нагрузки) → поле фон Мизеса, для обучения FNO."""

    def __init__(self, path: str, normalize: bool = True):
        blob = np.load(os.path.join(path, "fields.npz"))
        self.occ = blob["occupancy"].astype(np.float32)
        self.sigma = blob["stress"].astype(np.float32)
        self.loads = np.zeros((len(self.occ), 3), dtype=np.float32)
        self.scalars = np.zeros((len(self.occ), 2), dtype=np.float32)
        with open(os.path.join(path, "dataset.jsonl"), encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                fid = rec.get("field_id")
                if fid is None or fid >= len(self.occ):
                    continue
                self.loads[fid] = np.asarray(rec["load"]["force_n"], dtype=np.float32) / 1000.0
                fem = rec.get("fem", {})
                self.scalars[fid] = [
                    np.log1p(fem.get("max_stress_pa", 0.0) / 1e6),
                    np.log1p(fem.get("mean_stress_pa", 0.0) / 1e6),
                ]
        if normalize:
            scale = np.maximum(self.sigma.reshape(len(self.sigma), -1).max(axis=1), 1e-6)
            self.sigma = self.sigma / scale[:, None, None, None]

    def __len__(self) -> int:
        return len(self.occ)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.occ[i]).unsqueeze(0),
            torch.from_numpy(self.loads[i]),
            torch.from_numpy(self.sigma[i]).unsqueeze(0),
            torch.from_numpy(self.scalars[i]),
        )
