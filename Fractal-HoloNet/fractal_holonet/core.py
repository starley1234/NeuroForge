import math
from typing import Optional, Tuple, List, Dict, Any
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

class ProductionRMSNorm(nn.Module):
    """
    ONNX-friendly RMSNorm implementation using basic primitive math operations.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        x_norm = x * torch.rsqrt(variance + self.eps)
        return self.weight * x_norm


class FractalHoloNetConfig:
    def __init__(
        self,
        vocab_size: int = 50257,
        d_model: int = 256,
        n_layers: int = 6,
        d_ff: int = 768,
        min_freq: float = 0.001,
        max_freq: float = math.pi,
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
        self.dropout = dropout
        self.tie_word_embeddings = tie_word_embeddings
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id

    def to_dict(self) -> Dict[str, Any]:
        return {
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "d_ff": self.d_ff,
            "min_freq": self.min_freq,
            "max_freq": self.max_freq,
            "dropout": self.dropout,
            "tie_word_embeddings": self.tie_word_embeddings,
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id,
            "pad_token_id": self.pad_token_id,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FractalHoloNetConfig":
        return cls(**d)

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "FractalHoloNetConfig":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


class ComplexResonanceAssociativeCore(nn.Module):
    """
    Complex-Resonance Associative Core (CRAC).
    Supports parallel sequence processing O(N) and constant time state streaming step-by-step O(1).
    """
    def __init__(self, config: FractalHoloNetConfig):
        super().__init__()
        self.d_model = config.d_model
        
        # Log-spaced multi-scale frequency bands spanning geometric octaves
        freq_bands = torch.exp(
            torch.linspace(math.log(config.min_freq), math.log(config.max_freq), config.d_model)
        )
        self.register_buffer("base_omega", freq_bands)
        
        # Fused linear projection: [Query, Key, Value, D_Omega, D_Gamma]
        self.in_proj = nn.Linear(config.d_model, 5 * config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.norm = ProductionRMSNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

    def forward_sequence(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Processes full sequence (B, T, D) in linear time.
        """
        B, T, D = x.shape
        proj = self.in_proj(x)
        q, k, v, d_omega, d_gamma = torch.chunk(proj, 5, dim=-1)
        
        omega = self.base_omega + 0.1 * torch.tanh(d_omega)
        gamma = torch.sigmoid(d_gamma + 1.5) # High retention prior
        
        decay_real = gamma * torch.cos(omega)
        decay_imag = gamma * torch.sin(omega)
        
        kv = k * v
        
        s_real = torch.zeros(B, D, device=x.device, dtype=x.dtype)
        s_imag = torch.zeros(B, D, device=x.device, dtype=x.dtype)
        out = []
        
        for t in range(T):
            u_t = kv[:, t, :]
            dr = decay_real[:, t, :]
            di = decay_imag[:, t, :]
            
            new_sr = (s_real * dr - s_imag * di) + u_t
            new_si = (s_real * di + s_imag * dr)
            
            s_real = new_sr
            s_imag = new_si
            
            retrieved = q[:, t, :] * s_real
            out.append(retrieved)
            
        out_tensor = torch.stack(out, dim=1)
        res = self.out_proj(self.dropout(self.norm(out_tensor)))
        return res, (s_real, s_imag)

    def step(
        self,
        x_step: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Constant time O(1) single token streaming inference step.
        x_step: (B, 1, D) or (B, D)
        """
        if x_step.dim() == 3:
            x_step = x_step.squeeze(1)
        B, D = x_step.shape
        
        if state is None:
            s_real = torch.zeros(B, D, device=x_step.device, dtype=x_step.dtype)
            s_imag = torch.zeros(B, D, device=x_step.device, dtype=x_step.dtype)
        else:
            s_real, s_imag = state
            
        proj = self.in_proj(x_step)
        q, k, v, d_omega, d_gamma = torch.chunk(proj, 5, dim=-1)
        
        omega = self.base_omega + 0.1 * torch.tanh(d_omega)
        gamma = torch.sigmoid(d_gamma + 1.5)
        
        dr = gamma * torch.cos(omega)
        di = gamma * torch.sin(omega)
        u_t = k * v
        
        new_sr = (s_real * dr - s_imag * di) + u_t
        new_si = (s_real * di + s_imag * dr)
        
        retrieved = q * new_sr
        out = self.out_proj(self.norm(retrieved))
        if out.dim() == 2:
            out = out.unsqueeze(1)
        return out, (new_sr, new_si)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_step: bool = False
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if use_step or x.size(1) == 1:
            return self.step(x, state)
        return self.forward_sequence(x)


class FractalHoloNetBlock(nn.Module):
    def __init__(self, config: FractalHoloNetConfig):
        super().__init__()
        self.ln1 = ProductionRMSNorm(config.d_model)
        self.core = ComplexResonanceAssociativeCore(config)
        self.ln2 = ProductionRMSNorm(config.d_model)
        
        # Fused SwiGLU Feed-Forward Network
        self.w12 = nn.Linear(config.d_model, 2 * config.d_ff, bias=False)
        self.w3 = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_step: bool = False
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        core_out, next_state = self.core(self.ln1(x), state=state, use_step=use_step)
        x = x + core_out
        
        norm_x = self.ln2(x)
        w12_out = self.w12(norm_x)
        w1, w2 = torch.chunk(w12_out, 2, dim=-1)
        ffn = self.w3(self.dropout(F.silu(w1) * w2))
        x = x + ffn
        return x, next_state


class ProductionFractalHoloNet(nn.Module):
    """
    Production Ready Fractal-HoloNet Architecture.
    - Full causal sequence processing
    - Streaming token-by-token generation with O(1) state memory
    - Safe checkpoint loading & saving
    - ONNX export compatible
    """
    def __init__(self, config: FractalHoloNetConfig):
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList([FractalHoloNetBlock(config) for _ in range(config.n_layers)])
        self.ln_f = ProductionRMSNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        if config.tie_word_embeddings:
            self.head.weight = self.token_emb.weight
            
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.normal_(p, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        states: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_step: bool = False
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        input_ids: (B, T)
        states: list of (s_real, s_imag) for each layer
        """
        x = self.token_emb(input_ids)
        next_states = []
        
        for i, block in enumerate(self.blocks):
            layer_state = states[i] if states is not None else None
            x, next_layer_state = block(x, state=layer_state, use_step=use_step)
            next_states.append(next_layer_state)
            
        x = self.ln_f(x)
        logits = self.head(x)
        return logits, next_states

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        eos_token_id: Optional[int] = None
    ) -> torch.Tensor:
        """
        Autoregressive generation with O(1) step latency using persistent recurrent state.
        """
        self.eval()
        B = prompt_ids.size(0)
        device = prompt_ids.device
        
        # 1. Warm up state over the prompt sequence
        logits, states = self.forward(prompt_ids, states=None, use_step=False)
        curr_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        
        generated = [prompt_ids, curr_token]
        eos_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        
        for _ in range(max_new_tokens - 1):
            logits_step, states = self.forward(curr_token, states=states, use_step=True)
            step_logits = logits_step[:, -1, :] / max(temperature, 1e-5)
            
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(step_logits, min(top_k, step_logits.size(-1)))
                step_logits[step_logits < v[:, [-1]]] = -float('Inf')
                
            if top_p is not None and 0.0 < top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(step_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                step_logits[indices_to_remove] = -float('Inf')
                
            probs = F.softmax(step_logits, dim=-1)
            curr_token = torch.multinomial(probs, num_samples=1)
            
            if eos_id is not None and (curr_token == eos_id).all():
                break
            generated.append(curr_token)
                
        return torch.cat(generated, dim=1)

    def save_pretrained(self, save_directory: str):
        import os
        os.makedirs(save_directory, exist_ok=True)
        self.config.save(os.path.join(save_directory, "config.json"))
        torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.pt"))

    @classmethod
    def from_pretrained(cls, save_directory: str, map_location: str = "cpu") -> "ProductionFractalHoloNet":
        import os
        config = FractalHoloNetConfig.load(os.path.join(save_directory, "config.json"))
        model = cls(config)
        state_dict = torch.load(os.path.join(save_directory, "pytorch_model.pt"), map_location=map_location)
        model.load_state_dict(state_dict)
        return model
