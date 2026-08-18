import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================================================================
# 1. Baseline Transformer (Standard Causal Multi-Head Self Attention + MLP)
# =====================================================================
class BaselineCausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # O(T^2) Attention
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.tril(torch.ones(T, T, device=x.device)).bool()
        scores = scores.masked_fill(~mask, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.out_proj(out)

class BaselineBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = BaselineCausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class StandardTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=4, d_ff=512, max_len=4096):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([BaselineBlock(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
        x = self.token_emb(idx) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x)


# =====================================================================
# 2. V1: Fractal Phase-State Recurrent Network (FPSRN-v1)
# =====================================================================
class FractalPhaseCellV1(nn.Module):
    """
    O(T) State Space / Phase Recurrence with multi-frequency rotation.
    Phase theta controls memory retention/rotation in complex-like phase space.
    """
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        # Phase parameters: frequency omega and damping gamma
        self.omega = nn.Parameter(torch.randn(d_model) * 0.1)
        self.gamma = nn.Parameter(torch.ones(d_model) * -0.5) # Log-decay
        self.w_in = nn.Linear(d_model, d_model, bias=False)
        self.w_gate = nn.Linear(d_model, d_model, bias=True)
        self.w_out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        # x: (B, T, D)
        B, T, D = x.shape
        decay = torch.sigmoid(self.gamma) # (D,) in (0, 1)
        freq = self.omega                 # (D,)
        
        # Parallel associative scan or sequential recurrent scan
        # Sequential implementation for clarity & exact state tracking
        h_real = torch.zeros(B, D, device=x.device)
        h_imag = torch.zeros(B, D, device=x.device)
        
        inp = self.w_in(x)
        gates = torch.sigmoid(self.w_gate(x))
        
        cos_w = torch.cos(freq)
        sin_w = torch.sin(freq)
        
        outputs = []
        for t in range(T):
            u = inp[:, t, :] * gates[:, t, :]
            # Complex rotation + decay: (h_real + i*h_imag) * decay * e^(i*omega) + u
            # e^(i*w) = cos(w) + i*sin(w)
            # new_real = decay * (h_real * cos_w - h_imag * sin_w) + u
            # new_imag = decay * (h_real * sin_w + h_imag * cos_w)
            h_real = decay * (h_real * cos_w - h_imag * sin_w) + u
            h_imag = decay * (h_real * sin_w + h_imag * cos_w)
            outputs.append(h_real)
            
        out = torch.stack(outputs, dim=1)
        return self.w_out(out)

class FPSRNBlockV1(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.phase_cell = FractalPhaseCellV1(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        x = x + self.phase_cell(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class FPSRNModelV1(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_layers=4, d_ff=512):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([FPSRNBlockV1(d_model, d_ff) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx):
        x = self.token_emb(idx)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x)
