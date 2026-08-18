import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================================================================
# V4: Fractal-HoloNet (Fractal Gated Holographic Resonance Network)
# =====================================================================
# Architecture Highlights:
# 1. Complex-Resonance Associative Core (CRAC):
#    Maintains persistent contextual memory in complex phase space with O(1) state memory.
# 2. Continuous Multi-Scale Fractal Frequencies:
#    Exponential geometric octave spacing allows simultaneous capture of 
#    local n-gram syntax (high frequencies) and ultra-long range dependencies (sub-harmonic frequencies).
# 3. Dynamic Associative Holographic Binding:
#    Computes value projection scaled by key phase shift, allowing selective retrieval via query interference.
# 4. SwiGLU Gated Feed-Forward Modulation:
#    Maximum expressivity and non-linear feature transformation.
# =====================================================================

class ComplexResonanceAssociativeCore(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        
        # Log-spaced multi-scale frequency bands spanning 2^0 to 2^8 octaves
        freq_bands = torch.exp(torch.linspace(math.log(0.005), math.log(math.pi), d_model))
        self.register_buffer("base_omega", freq_bands)
        
        # Fused linear projection: [Query, Key, Value, D_Omega, D_Gamma]
        self.in_proj = nn.Linear(d_model, 5 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.norm = nn.RMSNorm(d_model) if hasattr(nn, 'RMSNorm') else nn.LayerNorm(d_model)

    def forward(self, x):
        B, T, D = x.shape
        proj = self.in_proj(x)
        q, k, v, d_omega, d_gamma = torch.chunk(proj, 5, dim=-1)
        
        # Contextual phase shift and decay modulation
        omega = self.base_omega + 0.1 * torch.tanh(d_omega)
        gamma = torch.sigmoid(d_gamma + 1.5) # biased towards high retention (0.8 - 0.99)
        
        # Phase rotation parameters
        decay_real = gamma * torch.cos(omega)
        decay_imag = gamma * torch.sin(omega)
        
        # Associative energy
        kv = k * v
        
        # Optimized associative memory state recurrence
        s_real = torch.zeros(B, D, device=x.device)
        s_imag = torch.zeros(B, D, device=x.device)
        out = []
        
        for t in range(T):
            u_t = kv[:, t, :]
            dr = decay_real[:, t, :]
            di = decay_imag[:, t, :]
            
            # Complex phase rotation + memory decay + new associative energy
            new_sr = (s_real * dr - s_imag * di) + u_t
            new_si = (s_real * di + s_imag * dr)
            
            s_real = new_sr
            s_imag = new_si
            
            # Holographic query resonance
            retrieved = q[:, t, :] * s_real
            out.append(retrieved)
            
        out_tensor = torch.stack(out, dim=1)
        return self.out_proj(self.norm(out_tensor))


class FractalHoloNetBlock(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.ln1 = nn.RMSNorm(d_model) if hasattr(nn, 'RMSNorm') else nn.LayerNorm(d_model)
        self.core = ComplexResonanceAssociativeCore(d_model)
        self.ln2 = nn.RMSNorm(d_model) if hasattr(nn, 'RMSNorm') else nn.LayerNorm(d_model)
        
        # Fused SwiGLU FFN
        self.w12 = nn.Linear(d_model, 2 * d_ff, bias=False)
        self.w3 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        x = x + self.core(self.ln1(x))
        norm_x = self.ln2(x)
        w12_out = self.w12(norm_x)
        w1, w2 = torch.chunk(w12_out, 2, dim=-1)
        ffn = self.w3(F.silu(w1) * w2)
        x = x + ffn
        return x


class FractalHoloNet(nn.Module):
    """
    Fractal-HoloNet: Novel AI Architecture combining Fractal Phase Dynamics and Holographic Associative Memory.
    - True O(N) Context Complexity
    - O(1) State Memory for Continuous Token Streaming
    - Multi-Octave Fractal Frequency Grid
    """
    def __init__(self, vocab_size, d_model=128, n_layers=4, d_ff=384):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([FractalHoloNetBlock(d_model, d_ff) for _ in range(n_layers)])
        self.ln_f = nn.RMSNorm(d_model) if hasattr(nn, 'RMSNorm') else nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.token_emb.weight

    def forward(self, idx):
        x = self.token_emb(idx)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x)
