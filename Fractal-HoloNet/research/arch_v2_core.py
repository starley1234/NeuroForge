"""
ELAST-HOLO: Elastic-Time Holographic Associative Machine (architecture v2).

Design doc: research/ARCHITECTURE_V2.md

Implemented mechanisms:
  M1  Elastic clock: the network controls the pace of its own internal phase
      clock (dtheta per token / per external dt) -> context-dependent time
      scales and native support for irregularly sampled (event) streams.
  M2  Circulant channel mixing (optional): circulant transition on the state
      matrix diagonalized by 2-D FFT, O(D^2 log D) -> O(D log D) per axis.
  M3  Complex delta write with phase-corrected erase on a MATRIX state
      (diagonal-plus-rank-1 style, cf. Gated DeltaNet / RWKV-7):
        S = rot*rot^T o S + beta o ( k v^T - k_del (k_del^H S) )
      Unit-norm keys make the erase a non-expansive projector (stable),
      and the rank-1 update removes cross-key interference -> associative
      recall (MQAR) becomes learnable, unlike plain diagonal recurrences.
  M4  Dual fast/slow holographic memory with a surprise-gated consolidation
      into the slow state; both memories are phase-coherent (same elastic
      timer).
  M5  Iterative associative read: K fixed-point refinement steps over
      (q^H S) complex energies (modern-Hopfield style) with a constant budget.
  M7  Complex signal frontend: learnable analytic (Hilbert-pair) filter bank
      producing a complex embedding that matches the complex core.

Everything supports both O(N) sequence mode and O(1)-in-T streaming `step`
mode, and the two modes are numerically identical (verified by tests).

State per layer: (S_fast, S_slow, theta) -- complex (B, D, D), complex
(B, D, D), real (B, D) accumulated per-channel phase. Step cost is O(D^2)
(the known price of the delta rule; fused/low-rank kernels are future work).
"""
import math
import os
import json
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def complex_exp_i(phi: torch.Tensor) -> torch.Tensor:
    """exp(i*phi) for a real tensor phi."""
    return torch.complex(torch.cos(phi), torch.sin(phi))


class RMSNorm(nn.Module):
    """ONNX-friendly RMSNorm (same primitive form as v1 production RMSNorm)."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return self.weight * (x * torch.rsqrt(variance + self.eps))


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
class ElasticHoloConfig:
    def __init__(
        self,
        vocab_size: int = 300,
        d_model: int = 96,
        n_layers: int = 4,
        d_ff: int = 288,
        min_freq: float = 0.001,
        max_freq: float = math.pi,
        # M1 elastic clock bounds
        dt_min: float = 0.2,
        dt_max: float = 3.0,
        # M3 write
        gamma_bias: float = 1.5,   # retention prior
        beta_bias: float = 0.0,    # in-context learning rate prior
        use_phase_erase: bool = True,
        # M4 slow memory
        use_slow_memory: bool = True,
        slow_gamma: float = 0.99,
        slow_omega_ratio: float = 0.1,
        # M5 read
        n_read_iters: int = 2,
        # M2 circulant mixing
        use_circulant: bool = False,
        dropout: float = 0.0,
        tie_word_embeddings: bool = True,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        pad_token_id: int = 0,
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.gamma_bias = gamma_bias
        self.beta_bias = beta_bias
        self.use_phase_erase = use_phase_erase
        self.use_slow_memory = use_slow_memory
        self.slow_gamma = slow_gamma
        self.slow_omega_ratio = slow_omega_ratio
        self.n_read_iters = n_read_iters
        self.use_circulant = use_circulant
        self.dropout = dropout
        self.tie_word_embeddings = tie_word_embeddings
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ElasticHoloConfig":
        return cls(**d)

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ElasticHoloConfig":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


# ---------------------------------------------------------------------------
# Core: M1 + M2 + M3 + M4 + M5
# ---------------------------------------------------------------------------
class ElasticHoloCore(nn.Module):
    """Elastic-time holographic associative core with delta write and
    fast/slow dual memory. Sequence mode O(N), step mode O(1) in T."""

    def __init__(self, config: ElasticHoloConfig):
        super().__init__()
        self.config = config
        D = config.d_model

        # Log-spaced multi-scale base frequencies (per-channel time scales)
        freq_bands = torch.exp(
            torch.linspace(math.log(config.min_freq), math.log(config.max_freq), D)
        )
        self.register_buffer("base_omega", freq_bands)

        # Fused projection: q, k, v, d_gamma, d_beta, d_dt, d_phi
        self.in_proj = nn.Linear(D, 7 * D, bias=False)

        # M4: surprise -> consolidation gate (per channel)
        self.eta_proj = nn.Linear(D, D, bias=True)
        nn.init.zeros_(self.eta_proj.bias)  # consolidation is rare at init

        # M4: fast/slow read router (per token)
        self.alpha_proj = nn.Linear(D, 1, bias=True)

        # M5: iterative read machinery (operates on real/imag pairs, 2D)
        self.W_z = nn.Linear(2 * D, 2 * D, bias=False)
        self.W_qx = nn.Linear(D, 2 * D, bias=False)
        self.b_z = nn.Parameter(torch.zeros(2 * D))
        self.W_o = nn.Linear(2 * D, D, bias=False)
        self.norm_out = RMSNorm(D)
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

        # M2: circulant transition spectrum (optional)
        if config.use_circulant:
            self.circulant = nn.Parameter(torch.randn(D, dtype=torch.complex64) / math.sqrt(D))
        else:
            self.circulant = None

    # -- single-token primitive shared by BOTH modes (guarantees equality) ----
    def _recurrence(
        self,
        x_t: torch.Tensor,
        q: torch.Tensor,
        v: torch.Tensor,
        gamma: torch.Tensor,
        beta: torch.Tensor,
        omega: torch.Tensor,
        k_del: torch.Tensor,
        S_fast: torch.Tensor,
        S_slow: torch.Tensor,
        theta: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        One token of state evolution given precomputed projections.
        x_t: (B, D); q,v,gamma,beta,omega: (B, D); k_del: complex (B, D);
        S_fast/S_slow: complex (B, D, D); theta: real (B, D).
        """
        cfg = self.config
        rot = gamma * complex_exp_i(omega)                      # (B, D) complex decay
        rot_mat = rot.unsqueeze(-1) * rot.unsqueeze(-2)          # (B, D, D)

        # ---- M3: complex delta write on the matrix state --------------------
        v_c = v.to(dtype=torch.complex64)
        align = torch.bmm(k_del.conj().unsqueeze(1), S_fast).squeeze(1)   # k_del^H S (B, D)
        erase = k_del.unsqueeze(-1) * align.unsqueeze(-2)                 # rank-1 erase
        w = beta.unsqueeze(-1) * (k_del.unsqueeze(-1) * v_c.unsqueeze(-2) - erase)
        S_fast_next = rot_mat * S_fast + w

        # ---- M2: circulant channel mixing (2-D FFT) --------------------------
        if self.circulant is not None:
            spec = self.circulant[:, None] * self.circulant[None, :]  # (D, D)
            S_fast_next = torch.fft.ifft2(
                torch.fft.fft2(S_fast_next, dim=(-2, -1)) * spec, dim=(-2, -1)
            )

        # ---- M4: surprise-gated slow memory ---------------------------------
        if cfg.use_slow_memory:
            QS = torch.bmm(
                torch.stack([q.to(dtype=torch.complex64), k_del.conj()], dim=1), S_fast_next
            )  # (B, 2, D): q^T S, k_del^H S
            r_fast_pre, align_fast = QS[:, 0], QS[:, 1]
            eta = torch.sigmoid(self.eta_proj(align_fast.abs()))     # (B, D)
            rot_slow = cfg.slow_gamma * complex_exp_i(omega * cfg.slow_omega_ratio)
            rot_slow_mat = rot_slow.unsqueeze(-1) * rot_slow.unsqueeze(-2)
            decay_slow = 1.0 - eta * (1.0 - cfg.slow_gamma)
            S_slow_next = (decay_slow.unsqueeze(-1) * rot_slow_mat) * S_slow + eta.unsqueeze(-1) * w
            r_fast = r_fast_pre
        else:
            r_fast = torch.bmm(q.to(dtype=torch.complex64).unsqueeze(1), S_fast_next).squeeze(1)
            S_slow_next = S_slow

        # ---- M5: iterative associative read ---------------------------------
        alpha = torch.sigmoid(self.alpha_proj(x_t))  # (B, 1)
        mix = r_fast
        if cfg.use_slow_memory:
            r_slow = torch.bmm(q.to(dtype=torch.complex64).unsqueeze(1), S_slow_next).squeeze(1)
            mix = mix + alpha * r_slow
        z = torch.cat([mix.real, mix.imag], dim=-1)  # (B, 2D)
        for _ in range(cfg.n_read_iters):
            z = z + F.silu(self.W_z(z) + self.W_qx(x_t) + self.b_z)
        out = self.norm_out(self.dropout(self.W_o(z)))
        return out, S_fast_next, S_slow_next, theta

    def _project(self, x_t: torch.Tensor, dt: Optional[torch.Tensor] = None):
        """Compute q,k,v,gamma,beta,omega,k_del for a single (B, D) token."""
        cfg = self.config
        proj = self.in_proj(x_t)
        q, k, v, d_gamma, d_beta, d_dt, d_phi = torch.chunk(proj, 7, dim=-1)
        if dt is not None:
            dtheta = torch.clamp(dt, cfg.dt_min, cfg.dt_max)
        else:
            dtheta = cfg.dt_min + (cfg.dt_max - cfg.dt_min) * torch.sigmoid(d_dt)
        omega = self.base_omega * dtheta  # (B, D)
        gamma = torch.sigmoid(d_gamma + cfg.gamma_bias)
        beta = torch.sigmoid(d_beta + cfg.beta_bias)
        k_n = k / (k.norm(dim=-1, keepdim=True) + 1e-5)
        if cfg.use_phase_erase:
            k_del = k_n * complex_exp_i(math.pi * torch.tanh(d_phi))
        else:
            k_del = k_n.to(dtype=torch.complex64)
        return q, k, v, gamma, beta, omega, k_del, dtheta

    def _step_core(
        self,
        x_t: torch.Tensor,
        S_fast: torch.Tensor,
        S_slow: torch.Tensor,
        theta: torch.Tensor,
        dt: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        q, _k, v, gamma, beta, omega, k_del, dtheta = self._project(x_t, dt)
        theta = theta + dtheta
        return self._recurrence(x_t, q, v, gamma, beta, omega, k_del, S_fast, S_slow, theta)

    def forward_sequence(
        self, x: torch.Tensor, dt: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """O(N) sequence mode with bulk (time-vectorized) projections and a
        per-token recurrence over the state. Numerically identical to `step`."""
        cfg = self.config
        B, T, D = x.shape
        proj = self.in_proj(x)  # (B, T, 7D) -- one big op instead of T small ones
        q, k, v, d_gamma, d_beta, d_dt, d_phi = torch.chunk(proj, 7, dim=-1)
        if dt is not None:
            dtheta = torch.clamp(dt, cfg.dt_min, cfg.dt_max)
            if dtheta.dim() == 2:
                dtheta = dtheta.unsqueeze(-1)  # (B, T) -> (B, T, 1), broadcast to channels
        else:
            dtheta = cfg.dt_min + (cfg.dt_max - cfg.dt_min) * torch.sigmoid(d_dt)
        theta = dtheta.sum(dim=1)  # (B, D) accumulated phase over the sequence
        if theta.size(-1) == 1 and D > 1:
            theta = theta.expand(B, D)  # scalar dt -> per-channel state
        omega = self.base_omega * dtheta                    # (B, T, D)
        gamma = torch.sigmoid(d_gamma + cfg.gamma_bias)     # (B, T, D)
        beta = torch.sigmoid(d_beta + cfg.beta_bias)
        k_n = k / (k.norm(dim=-1, keepdim=True) + 1e-5)
        if cfg.use_phase_erase:
            k_del = k_n * complex_exp_i(math.pi * torch.tanh(d_phi))  # (B, T, D)
        else:
            k_del = k_n.to(dtype=torch.complex64)

        S_fast = torch.zeros(B, D, D, dtype=torch.complex64, device=x.device)
        S_slow = torch.zeros_like(S_fast)
        outs = []
        for t in range(T):
            out, S_fast, S_slow, _ = self._recurrence(
                x[:, t, :], q[:, t, :], v[:, t, :],
                gamma[:, t, :], beta[:, t, :], omega[:, t, :], k_del[:, t, :],
                S_fast, S_slow, theta,
            )
            outs.append(out)
        return torch.stack(outs, dim=1), (S_fast, S_slow, theta)

    def step(
        self,
        x_t: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
        dt: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        if x_t.dim() == 3:
            x_t = x_t.squeeze(1)
        B, D = x_t.shape
        if state is None:
            S_fast = torch.zeros(B, D, D, dtype=torch.complex64, device=x_t.device)
            S_slow = torch.zeros_like(S_fast)
            theta = torch.zeros(B, D, device=x_t.device, dtype=x_t.dtype)
        else:
            S_fast, S_slow, theta = state
        out, S_fast, S_slow, theta = self._step_core(x_t, S_fast, S_slow, theta, dt)
        return out.unsqueeze(1), (S_fast, S_slow, theta)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
        use_step: bool = False,
        dt: Optional[torch.Tensor] = None,
    ):
        if use_step or x.size(1) == 1:
            return self.step(x, state, dt)
        return self.forward_sequence(x, dt)


class ElasticHoloBlock(nn.Module):
    def __init__(self, config: ElasticHoloConfig):
        super().__init__()
        self.ln1 = RMSNorm(config.d_model)
        self.core = ElasticHoloCore(config)
        self.ln2 = RMSNorm(config.d_model)
        self.w12 = nn.Linear(config.d_model, 2 * config.d_ff, bias=False)
        self.w3 = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

    def forward(self, x, state=None, use_step=False, dt=None):
        core_out, next_state = self.core(self.ln1(x), state=state, use_step=use_step, dt=dt)
        x = x + core_out
        w1, w2 = torch.chunk(self.w12(self.ln2(x)), 2, dim=-1)
        ffn = self.w3(self.dropout(F.silu(w1) * w2))
        return x + ffn, next_state


# ---------------------------------------------------------------------------
# Language model
# ---------------------------------------------------------------------------
class ElasticHoloNet(nn.Module):
    """ELAST-HOLO language model. Same interface as ProductionFractalHoloNet:
    forward(input_ids, states, use_step) -> (logits, states); generate(...)."""

    def __init__(self, config: ElasticHoloConfig):
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList([ElasticHoloBlock(config) for _ in range(config.n_layers)])
        self.ln_f = RMSNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.head.weight = self.token_emb.weight
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.normal_(p, mean=0.0, std=0.02)

    def forward(self, input_ids, states=None, use_step=False, dt=None):
        x = self.token_emb(input_ids)
        next_states = []
        for i, block in enumerate(self.blocks):
            layer_state = states[i] if states is not None else None
            x, ns = block(x, state=layer_state, use_step=use_step, dt=dt)
            next_states.append(ns)
        x = self.ln_f(x)
        return self.head(x), next_states

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """Autoregressive generation.

        The state is warmed up token-by-token through the streaming `step`
        path, so the generated sequence is exactly what the O(1) streaming
        decoder produces (no seq/step floating-point divergence)."""
        self.eval()
        # warm-up через step-режим (точная стриминговая семантика)
        states = None
        logits = None
        for t in range(prompt_ids.size(1)):
            logits, states = self.forward(prompt_ids[:, t : t + 1], states=states, use_step=True)
        curr_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = [prompt_ids, curr_token]
        eos_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        for _ in range(max_new_tokens - 1):
            logits_step, states = self.forward(curr_token, states=states, use_step=True)
            step_logits = logits_step[:, -1, :] / max(temperature, 1e-5)
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(step_logits, min(top_k, step_logits.size(-1)))
                step_logits[step_logits < v[:, [-1]]] = -float("Inf")
            if top_p is not None and 0.0 < top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(step_logits, descending=True)
                cumulative = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                remove = cumulative > top_p
                remove[..., 1:] = remove[..., :-1].clone()
                remove[..., 0] = 0
                step_logits[remove.scatter(1, sorted_indices, remove)] = -float("Inf")
            probs = F.softmax(step_logits, dim=-1)
            curr_token = torch.multinomial(probs, num_samples=1)
            if eos_id is not None and (curr_token == eos_id).all():
                break
            generated.append(curr_token)
        return torch.cat(generated, dim=1)

    def save_pretrained(self, save_directory: str):
        os.makedirs(save_directory, exist_ok=True)
        self.config.save(os.path.join(save_directory, "config.json"))
        torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.pt"))

    @classmethod
    def from_pretrained(cls, save_directory: str, map_location: str = "cpu") -> "ElasticHoloNet":
        config = ElasticHoloConfig.load(os.path.join(save_directory, "config.json"))
        model = cls(config)
        state_dict = torch.load(
            os.path.join(save_directory, "pytorch_model.pt"), map_location=map_location
        )
        model.load_state_dict(state_dict)
        return model


# ---------------------------------------------------------------------------
# M7: continuous-signal model (elastic time + analytic frontend)
# ---------------------------------------------------------------------------
class ComplexSignalFrontend(nn.Module):
    """M7-lite: learnable analytic (Hilbert-pair) filter bank.

    A pair of 1-D convolutions forms a per-channel analytic filter; the real
    and imaginary parts are concatenated into a real (B, T, d_model) tensor
    consumed by the standard core."""

    def __init__(self, in_channels: int, d_model: int, kernel_size: int = 7):
        super().__init__()
        half = d_model // 2
        self.conv_re = nn.Conv1d(in_channels, half, kernel_size, padding=kernel_size // 2, bias=False)
        self.conv_im = nn.Conv1d(in_channels, half, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        if signal.dim() == 2:
            signal = signal.unsqueeze(-1)
        s = signal.permute(0, 2, 1)          # (B, C, T)
        re = self.conv_re(s)                 # (B, d/2, T)
        im = self.conv_im(s)
        z = torch.complex(re, im)            # (B, d/2, T)
        return torch.cat([z.real, z.imag], dim=1).permute(0, 2, 1)  # (B, T, D)


class ElasticSignalNet(nn.Module):
    """Continuous-signal ELAST-HOLO: forecasting + anomaly scoring with
    native support for irregular inter-observation intervals `dt`."""

    def __init__(
        self,
        config: ElasticHoloConfig,
        input_signal_dim: int = 1,
        output_signal_dim: int = 1,
        frontend_kernel: int = 7,
    ):
        super().__init__()
        self.config = config
        self.frontend = ComplexSignalFrontend(input_signal_dim, config.d_model, frontend_kernel)
        self.blocks = nn.ModuleList([ElasticHoloBlock(config) for _ in range(config.n_layers)])
        self.ln_f = RMSNorm(config.d_model)
        self.signal_head = nn.Linear(config.d_model, output_signal_dim, bias=False)
        self.anomaly_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.SiLU(),
            nn.Linear(config.d_model // 2, 1),
        )

    def forward_continuous(self, signal, dt=None, states=None, use_step=False):
        x = self.frontend(signal)
        next_states = []
        for i, block in enumerate(self.blocks):
            layer_state = states[i] if states is not None else None
            x, ns = block(x, state=layer_state, use_step=use_step, dt=dt)
            next_states.append(ns)
        x_norm = self.ln_f(x)
        pred_signal = self.signal_head(x_norm)
        anomaly = torch.sigmoid(self.anomaly_head(x_norm))
        return pred_signal, anomaly, next_states

    @torch.no_grad()
    def forecast_stream(self, signal_history, forecast_steps=50, dt=None):
        """Autoregressive O(1)-state forecast. Future dt is assumed equal to
        the last observed dt (or the learned clock if dt is None)."""
        self.eval()
        pred, _, states = self.forward_continuous(signal_history, dt=dt)
        curr = pred[:, -1:, :]
        last_dt = dt[:, -1:] if dt is not None else None
        forecasts = [curr]
        for _ in range(forecast_steps - 1):
            p, _, states = self.forward_continuous(curr, states=states, use_step=True, dt=last_dt)
            curr = p[:, -1:, :]
            forecasts.append(curr)
        return torch.cat(forecasts, dim=1)
