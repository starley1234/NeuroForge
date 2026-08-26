from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly


def load_mono_wav(path: str | Path, target_sr: int) -> np.ndarray:
    path = Path(path)
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        n = wf.getnframes()
        ch = wf.getnchannels()
        sw = wf.getsampwidth()
        raw = wf.readframes(n)
    if sw == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sw == 4:
        data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        data = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        data = (data - 128.0) / 128.0
    if ch > 1:
        data = data.reshape(-1, ch).mean(axis=1)
    if sr != target_sr:
        g = np.gcd(sr, target_sr)
        data = resample_poly(data, target_sr // g, sr // g).astype(np.float32)
    return np.ascontiguousarray(data, dtype=np.float32)


def crop_or_pad(x: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    if x.size >= n:
        start = int(rng.integers(0, x.size - n + 1))
        return x[start : start + n].copy()
    out = np.zeros(n, dtype=np.float32)
    if x.size:
        start = int(rng.integers(0, n - x.size + 1))
        out[start : start + x.size] = x
    return out
