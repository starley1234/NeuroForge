"""Hardware-style DSP front-end: log-Mel + GCC-PHAT (3 pairs)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from aura_micro.config import AuraConfig


def _mel_filterbank(n_fft: int, n_mels: int, sr: int) -> np.ndarray:
    def hz_to_mel(hz: np.ndarray | float) -> np.ndarray | float:
        return 2595.0 * np.log10(1.0 + np.asarray(hz) / 700.0)

    def mel_to_hz(mel: np.ndarray) -> np.ndarray:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    fft_bins = n_fft // 2 + 1
    mels = np.linspace(hz_to_mel(0.0), hz_to_mel(sr / 2.0), n_mels + 2)
    hz = mel_to_hz(mels)
    bins = np.floor((n_fft + 1) * hz / sr).astype(int)
    fb = np.zeros((n_mels, fft_bins), dtype=np.float32)
    for m in range(n_mels):
        left, center, right = bins[m], bins[m + 1], bins[m + 2]
        if center <= left:
            center = left + 1
        if right <= center:
            right = center + 1
        right = min(right, fft_bins - 1)
        for k in range(left, center):
            fb[m, k] = (k - left) / max(center - left, 1)
        for k in range(center, right):
            fb[m, k] = (right - k) / max(right - center, 1)
    return fb


class HardwareDSPFrontEnd(nn.Module):
    """Produces [B, T, feat_dim] = 3x log-Mel + 3x GCC-PHAT (32 bins each)."""

    def __init__(self, cfg: AuraConfig | None = None):
        super().__init__()
        self.cfg = cfg or AuraConfig()
        n_fft = self.cfg.win_samples
        fb = _mel_filterbank(n_fft, self.cfg.n_mels, self.cfg.sample_rate)
        self.register_buffer("mel_fb", torch.from_numpy(fb))
        window = torch.hann_window(n_fft)
        self.register_buffer("window", window)
        self.n_fft = n_fft
        self.hop = self.cfg.hop_samples

    def _stft(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, samples]
        return torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop,
            win_length=self.n_fft,
            window=self.window.to(x.dtype),
            center=True,
            return_complex=True,
        )  # [B, F, T]

    def _log_mel(self, spec: torch.Tensor) -> torch.Tensor:
        power = spec.abs().pow(2)
        mel = torch.matmul(self.mel_fb.to(power.dtype), power)  # [B, n_mels, T]
        return torch.log(mel + 1e-6)

    def _gcc_phat(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """GCC-PHAT cropped to gcc_bins around lag 0. a,b: [B, F, T] complex."""
        cross = a * b.conj()
        mag = cross.abs().clamp_min(1e-8)
        phat = cross / mag
        cc = torch.fft.irfft(phat, n=self.n_fft, dim=1)  # [B, n_fft, T]
        # circular shift so lag 0 is center
        cc = torch.fft.fftshift(cc, dim=1)
        center = self.n_fft // 2
        half = self.cfg.gcc_bins // 2
        cropped = cc[:, center - half : center - half + self.cfg.gcc_bins, :]
        return cropped

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """
        wav: [B, 3, samples]
        returns features [B, T, feat_dim]
        """
        if wav.dim() != 3 or wav.size(1) != 3:
            raise ValueError(f"expected [B, 3, S], got {tuple(wav.shape)}")
        specs = [self._stft(wav[:, i]) for i in range(3)]
        mels = [self._log_mel(s) for s in specs]  # each [B, n_mels, T]
        gccs = [
            self._gcc_phat(specs[0], specs[1]),
            self._gcc_phat(specs[1], specs[2]),
            self._gcc_phat(specs[2], specs[0]),
        ]
        feat = torch.cat(mels + gccs, dim=1)  # [B, feat, T]
        return feat.transpose(1, 2).contiguous()

    def extract_numpy(self, wav: np.ndarray) -> np.ndarray:
        t = torch.from_numpy(wav.astype(np.float32))
        if t.dim() == 2:
            t = t.unsqueeze(0)
        with torch.no_grad():
            return self.forward(t).squeeze(0).cpu().numpy()
