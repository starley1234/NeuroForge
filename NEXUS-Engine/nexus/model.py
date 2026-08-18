"""NEXUS-Engine — сборка трёх уровней UniPhysical-Latent Framework."""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .bus import LatentPacket, UnifiedLatentBus
from .config import NexusConfig
from .encoders.text_ast import TextASTEncoder
from .heads import ContinuousHead, DiscreteHead, NexusOutput
from .layers.block import BlockState, NexusBlock, total_state_bytes
from .layers.common import RMSNorm, count_parameters
from .reasoning.workspace import LatentReasoner
from .surrogates.fno import FEMCritic


class NexusEngine(nn.Module):
    def __init__(self, cfg: Optional[NexusConfig] = None, with_critic: bool = True):
        super().__init__()
        self.cfg = cfg or NexusConfig()
        # --- уровень 1 (дискретный энкодер всегда в ядре) ---
        self.text_encoder = TextASTEncoder(self.cfg.d_latent, vocab_size=self.cfg.vocab_size)
        self.encoders: Dict[str, nn.Module] = {}
        # --- уровень 2 ---
        self.bus = UnifiedLatentBus(self.cfg.d_latent, self.cfg.max_time)
        self.blocks = nn.ModuleList(NexusBlock(self.cfg) for _ in range(self.cfg.n_layers))
        self.final_norm = RMSNorm(self.cfg.d_latent)
        self.reasoner = LatentReasoner(self.cfg.d_latent, self.cfg.reasoning)
        self.critic = FEMCritic(self.cfg.fno) if with_critic else None
        self.thought_mix = nn.Linear(self.cfg.d_latent * 2, self.cfg.d_latent)
        # --- уровень 3 ---
        self.lm_head = DiscreteHead(self.cfg, self.text_encoder.embed)
        self.continuous_head = ContinuousHead(self.cfg)
        self.apply(self._init_weights)
        self._scale_residual_projections()

    # ------------------------------------------------------- инициализация
    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """GPT-2-подобная инициализация: стартовый loss ≈ ln(V), а не десятки."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _scale_residual_projections(self) -> None:
        """Выходы остаточных веток масштабируются на 1/sqrt(2·L) (как в GPT-2)."""
        scale = (2 * max(self.cfg.n_layers, 1)) ** -0.5
        for block in self.blocks:
            projections = [m.out for m in (block.memory, block.attention) if m is not None]
            experts = (list(block.moe.experts) + list(block.moe.shared)
                       if block.moe is not None else [block.ffn])
            with torch.no_grad():
                for proj in projections:
                    proj.weight.mul_(scale)
                for expert in experts:
                    expert.w_down.weight.mul_(scale)

    # ------------------------------------------------------- регистрация мод.
    def register_encoder(self, name: str, encoder: nn.Module) -> None:
        """Подключение новой модальности: ядро замораживается, учится энкодер."""
        self.encoders[name] = encoder
        self.add_module(f"enc_{name}", encoder)

    def freeze_core(self) -> "NexusEngine":
        for module in (self.bus, self.blocks, self.final_norm, self.reasoner,
                       self.lm_head, self.continuous_head, self.text_encoder):
            for p in module.parameters():
                p.requires_grad_(False)
        return self

    # ------------------------------------------------------------------ ядро
    def encode(self, packets: Sequence[LatentPacket]) -> torch.Tensor:
        return self.bus(packets)

    def backbone(
        self,
        x: torch.Tensor,
        states: Optional[List[BlockState]] = None,
        use_state: bool = False,
    ):
        aux = x.new_zeros(())
        new_states: List[BlockState] = []
        for i, block in enumerate(self.blocks):
            st = states[i] if states else None
            x, ns, stats = block(x, st, use_state=use_state)
            aux = aux + stats.aux_loss
            if use_state:
                new_states.append(ns)
        return self.final_norm(x), new_states, aux

    # --------------------------------------------------------------- forward
    def forward(
        self,
        tokens: Optional[torch.Tensor] = None,
        packets: Optional[Sequence[LatentPacket]] = None,
        occupancy: Optional[torch.Tensor] = None,
        load: Optional[torch.Tensor] = None,
        reason: bool = True,
        states: Optional[List[BlockState]] = None,
        use_state: bool = False,
        continuous: bool = False,
        pos_offset: int = 0,
    ) -> NexusOutput:
        pack: List[LatentPacket] = list(packets or [])
        if tokens is not None:
            pack.insert(0, self.text_encoder(tokens, pos_offset=pos_offset))
        assert pack, "нужны либо tokens, либо packets"

        x = self.encode(pack)
        x, new_states, aux = self.backbone(x, states, use_state)

        trace = None
        if reason:
            probe = self._make_probe(occupancy, load)
            pooled = x.mean(dim=1)
            thought, trace = self.reasoner(pooled, probe)
            x = x + self.thought_mix(
                torch.cat([x, thought.unsqueeze(1).expand_as(x)], dim=-1)
            )

        logits = self.lm_head(x)
        actions = field = physics = None
        if continuous:
            actions, field, physics = self.continuous_head(x)

        out = NexusOutput(logits, actions, field, physics, aux, trace)
        out.states = new_states if use_state else None  # type: ignore[attr-defined]
        return out

    def _make_probe(self, occupancy, load) -> Optional[Callable[[torch.Tensor], torch.Tensor]]:
        if self.critic is None or occupancy is None:
            return None
        force = load if load is not None else occupancy.new_zeros(occupancy.shape[0], 3)

        def probe(_state: torch.Tensor) -> torch.Tensor:
            _, scalars = self.critic(occupancy, force)
            return torch.tanh(scalars[:, :1])

        return probe

    # ------------------------------------------------------------- удобства
    def loss(self, tokens: torch.Tensor, targets: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        out = self.forward(tokens=tokens, **kwargs)
        lm = F.cross_entropy(
            out.logits[:, :-1].reshape(-1, out.logits.shape[-1]),
            targets[:, 1:].reshape(-1),
            ignore_index=0,
        )
        ponder = out.reasoning.ponder_cost if out.reasoning else lm.new_zeros(())
        total = lm + (out.aux_loss or 0.0) + ponder
        return {"loss": total, "lm": lm.detach(),
                "aux": (out.aux_loss or lm.new_zeros(())).detach(),
                "ponder": ponder.detach()}

    @staticmethod
    def _sample(logits: torch.Tensor, temperature: float, top_k: int,
                top_p: float = 1.0) -> torch.Tensor:
        if temperature <= 0:
            return logits.argmax(-1, keepdim=True)
        logits = logits / temperature
        k = min(top_k, logits.shape[-1]) if top_k > 0 else logits.shape[-1]
        v, idx = torch.topk(logits, k, dim=-1)
        probs = torch.softmax(v, dim=-1)
        if 0 < top_p < 1.0:                       # nucleus: отсечь длинный хвост
            sorted_p, order = torch.sort(probs, dim=-1, descending=True)
            keep = (torch.cumsum(sorted_p, dim=-1) - sorted_p) < top_p
            keep[..., 0] = True
            mask = torch.zeros_like(probs, dtype=torch.bool).scatter(-1, order, keep)
            probs = torch.where(mask, probs, torch.zeros_like(probs))
            probs = probs / probs.sum(-1, keepdim=True).clamp(min=1e-9)
        return idx.gather(-1, torch.multinomial(probs, 1))

    @torch.no_grad()
    def generate(self, tokens: torch.Tensor, max_new_tokens: int = 64,
                 temperature: float = 0.8, top_k: int = 40, top_p: float = 1.0,
                 eos_id: Optional[int] = None, use_cache: bool = True) -> torch.Tensor:
        """Инкрементальная генерация: префикс считается один раз, дальше по токену.

        Состояние (TTT + окно KV) переносится между шагами, поэтому стоимость
        одного токена постоянна, а не растёт с длиной контекста.
        """
        self.eval()
        if not use_cache:
            for _ in range(max_new_tokens):
                window = tokens[:, -self.cfg.attention.window:]
                logits = self.forward(tokens=window, reason=False).logits[:, -1]
                tokens = torch.cat([tokens, self._sample(logits, temperature, top_k, top_p)],
                                   dim=1)
            return tokens

        offset = tokens.shape[1]
        out = self.forward(tokens=tokens, reason=False, use_state=True)
        states = out.states  # type: ignore[attr-defined]
        logits = out.logits[:, -1]
        finished = torch.zeros(tokens.shape[0], dtype=torch.bool, device=tokens.device)

        for _ in range(max_new_tokens):
            nxt = self._sample(logits, temperature, top_k, top_p)
            tokens = torch.cat([tokens, nxt], dim=1)
            if eos_id is not None:
                finished |= nxt.squeeze(-1) == eos_id
                if bool(finished.all()):
                    break
            step = self.forward(tokens=nxt, reason=False, states=states,
                                use_state=True, pos_offset=offset)
            states = step.states  # type: ignore[attr-defined]
            logits = step.logits[:, -1]
            offset += 1
        return tokens

    @torch.no_grad()
    def stream(self, tokens: torch.Tensor, max_new_tokens: int = 64,
               temperature: float = 0.8, top_k: int = 40, top_p: float = 1.0,
               eos_id: Optional[int] = None):
        """Итератор токенов: префикс считается один раз, дальше по токену."""
        self.eval()
        offset = tokens.shape[1]
        out = self.forward(tokens=tokens, reason=False, use_state=True)
        states = out.states  # type: ignore[attr-defined]
        logits = out.logits[:, -1]
        for _ in range(max_new_tokens):
            nxt = self._sample(logits, temperature, top_k, top_p)
            yield nxt
            if eos_id is not None and bool((nxt.squeeze(-1) == eos_id).all()):
                return
            step = self.forward(tokens=nxt, reason=False, states=states,
                                use_state=True, pos_offset=offset)
            states = step.states  # type: ignore[attr-defined]
            logits = step.logits[:, -1]
            offset += 1

    # ------------------------------------------------------------ статистика
    def parameter_report(self) -> Dict[str, float]:
        moe_total = sum(count_parameters(b.moe) for b in self.blocks if b.moe is not None)
        first_moe = next((b.moe for b in self.blocks if b.moe is not None), None)
        frac = first_moe.active_param_fraction if first_moe is not None else 1.0
        total = count_parameters(self)
        active = total - moe_total + moe_total * frac
        return {
            "total_params": float(total),
            "active_params": float(active),
            "moe_params": float(moe_total),
            "critic_params": float(count_parameters(self.critic)) if self.critic else 0.0,
            "active_fraction": float(active / max(total, 1)),
        }

    @staticmethod
    def state_bytes(states: List[BlockState]) -> int:
        return total_state_bytes(states)
