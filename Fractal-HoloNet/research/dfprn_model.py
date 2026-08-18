import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================================================================
# Modernized Architecture: Dynamic Fractal Phase-Resonance Network (DFPRN)
# =====================================================================
# Key Innovations:
# 1. Multi-Scale Phase Resonance (hierarchical frequency bands: short, medium, long-range)
# 2. Input-Dependent Dynamic Decay & Frequency (Contextual Phase Modulation)
# 3. Associative Holographic Memory Matrix (O(1) memory per step, O(T) compute)
# 4. Gated Fractal Routing (SwiGLU-style modulation + residual channel gating)
# =====================================================================

class MultiScaleHolographicPhaseCell(nn.Module):
    def __init__(self, d_model, num_heads=4):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # Base log-frequencies initialized across fractal octaves
        freq_init = torch.linspace(0.01, math.pi, d_model)
        self.base_omega = nn.Parameter(freq_init)
        
        # Linear projections
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        
        # Dynamic phase & decay modulators
        self.mod_proj = nn.Linear(d_model, 2 * d_model, bias=True)
        self.gate_proj = nn.Linear(d_model, d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        
        # Linear projections
        q = self.q_proj(x) # (B, T, D)
        k = self.k_proj(x) # (B, T, D)
        v = self.v_proj(x) # (B, T, D)
        
        # Dynamic phase frequency shift and retention gate
        mod = self.mod_proj(x)
        d_omega, d_gamma = torch.chunk(mod, 2, dim=-1)
        
        # Dynamic frequencies and decay factors
        omega = self.base_omega.unsqueeze(0).unsqueeze(0) + 0.1 * torch.tanh(d_omega)
        gamma = torch.sigmoid(d_gamma) # Dynamic forgetting factor in (0, 1)
        
        cos_w = torch.cos(omega)
        sin_w = torch.sin(omega)
        
        # Sequential associative state accumulation
        # State: complex key-value associative tensor (B, D)
        s_real = torch.zeros(B, D, device=x.device)
        s_imag = torch.zeros(B, D, device=x.device)
        
        # Holographic binding: input energy u_t = k * v
        # and retrieval = q * state
        outputs = []
        for t in range(T):
            u_t = k[:, t, :] * v[:, t, :]
            g_t = gamma[:, t, :]
            c_t = cos_w[:, t, :]
            s_t = sin_w[:, t, :]
            
            # Complex phase rotation + memory decay + new associative energy
            s_real = g_t * (s_real * c_t - s_imag * s_t) + u_t
            s_imag = g_t * (s_real * s_t + s_imag * c_t)
            
            # Query retrieval from holographic resonance state
            retrieved = q[:, t, :] * s_real
            outputs.append(retrieved)
            
        out = torch.stack(outputs, dim=1) # (B, T, D)
        gates = F.silu(self.gate_proj(x))
        return self.out_proj(out * gates)


class DFPRNBlock(nn.Module):
    def __init__(self, d_model, d_ff, num_heads=4):
        super().__init__()
        self.ln1 = nn.RMSNorm(d_model) if hasattr(nn, 'RMSNorm') else nn.LayerNorm(d_model)
        self.cell = MultiScaleHolographicPhaseCell(d_model, num_heads)
        self.ln2 = nn.RMSNorm(d_model) if hasattr(nn, 'RMSNorm') else nn.LayerNorm(d_model)
        
        # SwiGLU FFN
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        x = x + self.cell(self.ln1(x))
        # SwiGLU: (w1(x) * silu(w2(x))) -> w3
        norm_x = self.ln2(x)
        ffn = self.w3(F.silu(self.w1(norm_x)) * self.w2(norm_x))
        x = x + ffn
        return x


class DFPRNModel(nn.Module):
    """
    Dynamic Fractal Phase-Resonance Network (DFPRN).
    - O(N) context scaling complexity
    - Continuous multi-scale phase dynamics
    - Holographic associative query-key-value binding
    """
    def __init__(self, vocab_size, d_model=128, n_layers=4, d_ff=384, num_heads=4):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([DFPRNBlock(d_model, d_ff, num_heads) for _ in range(n_layers)])
        self.ln_f = nn.RMSNorm(d_model) if hasattr(nn, 'RMSNorm') else nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Weight tying
        self.head.weight = self.token_emb.weight

    def forward(self, idx):
        x = self.token_emb(idx)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x)
