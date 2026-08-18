from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
ESC50_DIR = DATA_DIR / "esc50"
CKPT_PATH = ROOT / "artifacts" / "aura_micro.pt"
WEB_DIR = ROOT / "web"
