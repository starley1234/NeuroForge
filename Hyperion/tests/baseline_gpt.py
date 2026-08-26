"""Минимальный GPT-2 baseline для сверки (не входит в пакет hyperion)."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class CausalSelfAttention(nn.Module):
    def __init__(self, dim, n_heads, max_seq_len=512, dropout=0.0):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.c_attn = nn.Linear(dim, 3 * dim, bias=False)
        self.c_proj = nn.Linear(dim, dim, bias=False)
        self.dropout = dropout
        self.register_buffer("bias", torch.tril(torch.ones(max_seq_len, max_seq_len))
                             .view(1, 1, max_seq_len, max_seq_len), persistent=False)

    def forward(self, x):
        B, S, D = x.shape
        qkv = self.c_attn(x).view(B, S, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(self.bias[:, :, :S, :S] == 0, float("-inf"))
        p = F.softmax(scores, dim=-1)
        out = p @ v
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        return self.c_proj(out)


class MLP(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * mult, bias=False)
        self.fc2 = nn.Linear(dim * mult, dim, bias=False)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, dim, n_heads, max_seq_len):
        super().__init__()
        self.ln1 = RMSNorm(dim)
        self.attn = CausalSelfAttention(dim, n_heads, max_seq_len)
        self.ln2 = RMSNorm(dim)
        self.mlp = MLP(dim)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPTBaseline(nn.Module):
    def __init__(self, vocab_size, dim=256, n_layers=4, n_heads=4, max_seq_len=512):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([Block(dim, n_heads, max_seq_len) for _ in range(n_layers)])
        self.ln_f = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            m.weight.data.normal_(mean=0.0, std=0.02)
            if getattr(m, "bias", None) is not None:
                m.bias.data.zero_()

    def forward(self, ids, labels=None):
        h = self.embed(ids)
        for b in self.blocks:
            h = b(h)
        h = self.ln_f(h)
        logits = self.lm_head(h)
        out = {"logits": logits}
        if labels is not None:
            out["loss"] = self.loss_fn(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
        return out


if __name__ == "__main__":
    torch.manual_seed(42)
    V, S, LAG = 512, 96, 8
    data = torch.randint(0, V, (8, S + LAG))
    data[:, LAG:] = data[:, :-LAG].clone()

    model = GPTBaseline(V)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
    for step in range(1, 201):
        opt.zero_grad()
        out = model(data[:, :S], labels=data[:, 1:S + 1].clone())
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    model.eval()
    gl = 0.0
    for t in range(3):
        prompt = torch.randint(0, V, (1, 24))
        prompt[0, LAG:] = prompt[0, :-LAG].clone()
        with torch.no_grad():
            out = model(prompt[:, :23], labels=prompt[:, 1:24].clone())
        gl += out["loss"].item()
    print(f"GPT-baseline: train={out['loss'].item():.3f} general={gl/3:.3f} (rand=6.24)")
