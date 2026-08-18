"""Equilateral 3-mic geometry and physically motivated spatial cues."""

from __future__ import annotations

import math

import numpy as np

from aura_micro.config import AuraConfig

SPEED_OF_SOUND = 343.0  # m/s
AIR_ABSORPTION_DB_PER_M_AT_8KHZ = 0.054  # approx. at 20C, 50% RH


def mic_positions(cfg: AuraConfig) -> np.ndarray:
    """Return (3, 2) XY positions of M1, M2, M3 (equilateral triangle)."""
    d = cfg.mic_baseline_m
    return np.array(
        [
            [0.0, 0.0],
            [d, 0.0],
            [d / 2.0, math.sqrt(3.0) / 2.0 * d],
        ],
        dtype=np.float64,
    )


def array_centroid(cfg: AuraConfig) -> np.ndarray:
    return mic_positions(cfg).mean(axis=0)


def itd_seconds(src_xy: np.ndarray, cfg: AuraConfig) -> np.ndarray:
    """ITD for pairs (M1-M2, M2-M3, M3-M1) in seconds."""
    mics = mic_positions(cfg)
    dist = np.linalg.norm(mics - src_xy[None, :], axis=1)
    tau = dist / SPEED_OF_SOUND
    return np.array([tau[0] - tau[1], tau[1] - tau[2], tau[2] - tau[0]])


def ild_db(src_xy: np.ndarray, cfg: AuraConfig, alpha: float = 1.0) -> np.ndarray:
    """Simple 1/r amplitude ILD (dB) for the three pairs."""
    mics = mic_positions(cfg)
    r = np.maximum(np.linalg.norm(mics - src_xy[None, :], axis=1), 1e-3)
    amp = 1.0 / (r**alpha)
    pairs = [(0, 1), (1, 2), (2, 0)]
    return np.array([20.0 * np.log10(amp[i] / amp[j] + 1e-12) for i, j in pairs])


def azimuth_elevation(src_xyz: np.ndarray, cfg: AuraConfig) -> tuple[float, float]:
    """Azimuth theta (xy) and elevation phi from array centroid."""
    c = np.append(array_centroid(cfg), 0.0)
    v = src_xyz - c
    rxy = math.hypot(v[0], v[1])
    theta = math.atan2(v[1], v[0])
    phi = math.atan2(v[2], rxy + 1e-12)
    return theta, phi


def doppler_shift_hz(f0: float, vr: float) -> float:
    """Delta-f for radial velocity vr (positive = approaching)."""
    return f0 * vr / SPEED_OF_SOUND


def looming_energy_gradient(energy: np.ndarray, dt: float) -> np.ndarray:
    """dE/dt ~ vr / r^2."""
    return np.gradient(energy, dt)


def air_absorption_db(distance_m: float, f_hz: float) -> float:
    """Linearized high-frequency air absorption vs distance."""
    scale = (f_hz / 8000.0) ** 2
    return AIR_ABSORPTION_DB_PER_M_AT_8KHZ * scale * distance_m
