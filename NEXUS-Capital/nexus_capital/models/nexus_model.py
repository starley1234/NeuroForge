"""
Полная сборка модели NEXUS-Capital из 4 уровней.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from ..core.config import NexusConfig
from ..core.value_bus import UnifiedValueBus
from ..core.tokenizer import NexusTokenizer
from ..encoders import (
    OrderBookEncoder, StructuredEncoder, GraphRiskOperator,
    AudioStressEncoder, FinancialTextEncoder, MultimodalFusion,
)
from ..workspace.workspace import EconomicWorkspace
from ..outputs.engine import DualOutputEngine
from ..utility.losses import NexusLoss
from ..utility.balance import FieldReconstructionHead


class NexusCapital(nn.Module):
    """
    NEXUS-Capital (VALUEX) — экономически заземлённая мультимодальная модель.

    Поток данных:
        modality inputs -> encoders (Уровень 1)
            -> Unified Value Bus + TTT/Attn/MoE (Уровень 2)
            -> Economic Workspace (Neural SDE, MC, Game Theory) (Уровень 3)
            -> Dual Output Engine (Уровень 4)
    """

    def __init__(self, cfg: NexusConfig, tokenizer: NexusTokenizer | None = None):
        super().__init__()
        self.cfg = cfg

        # Токенизатор (предобученный XLM-R) — не обучается.
        # Если не передан, использует оффлайн BPE и синхронизирует vocab_size.
        if tokenizer is None:
            tokenizer = NexusTokenizer.default(prefer_offline=True)
        self.tokenizer = tokenizer
        self._sync_vocab_to_tokenizer(tokenizer)

        # Уровень 1
        self.orderbook_enc = OrderBookEncoder(
            cfg.d_value, levels=cfg.orderbook_levels,
            tick_features=cfg.tick_features)
        self.structured_enc = StructuredEncoder(
            cfg.d_value, n_fields=cfg.tabular_features)
        self.graph_enc = GraphRiskOperator(
            cfg.d_value, node_dim=cfg.graph_node_dim)
        self.audio_enc = AudioStressEncoder(
            cfg.d_value, n_mels=cfg.audio_n_mels)
        self.text_enc = FinancialTextEncoder(
            cfg.d_value,
            vocab_size=cfg.text_vocab_size,
            d_embed=cfg.text_embed_dim,
            max_len=cfg.text_n_pos,
            pad_id=tokenizer.pad_id,
            num_id=tokenizer.num_id,
        )
        self.fusion = MultimodalFusion(cfg.d_value)
        # Голова реконструкции полей отчётности (для активного L_balance)
        self.field_head = FieldReconstructionHead(
            cfg.d_value, cfg.tabular_features)

        # Уровень 2
        self.bus = UnifiedValueBus(cfg)

        # Уровень 3
        self.workspace = EconomicWorkspace(
            cfg.d_value,
            sde_hidden=cfg.sde_hidden,
            mc_paths=cfg.mc_paths,
            mc_horizon=cfg.mc_horizon,
            alpha=cfg.risk_alpha,
        )

        # Уровень 4
        self.output = DualOutputEngine(
            cfg.d_value, vocab_size=cfg.vocab_size, gamma=cfg.gamma_risk)

        self.loss_fn = NexusLoss(
            lambda_utility=cfg.lambda_utility,
            lambda_var=cfg.lambda_var,
            gamma_risk=cfg.gamma_risk,
            alpha=cfg.risk_alpha,
        )

    def _sync_vocab_to_tokenizer(self, tokenizer: NexusTokenizer) -> None:
        """Приводит размеры словаря конфига в соответствие токенизатору."""
        vs = tokenizer.vocab_size
        self.cfg.text_vocab_size = vs
        self.cfg.vocab_size = vs

    # ── кодирование модальностей ─────────────────────────────────
    def encode_modalities(
        self,
        book: torch.Tensor | None = None,
        ticks: torch.Tensor | None = None,
        fields: torch.Tensor | None = None,
        field_mask: torch.Tensor | None = None,
        graph_nodes: torch.Tensor | None = None,
        edge_index: torch.Tensor | None = None,
        edge_attr: torch.Tensor | None = None,
        graph_target: torch.Tensor | None = None,
        mels: torch.Tensor | None = None,
        tokens: torch.Tensor | None = None,
        number_values: torch.Tensor | None = None,
        number_mask: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        embs: dict[str, torch.Tensor] = {}
        if book is not None and ticks is not None:
            embs["orderbook"] = self.orderbook_enc(book, ticks)
        if fields is not None:
            embs["structured"] = self.structured_enc(fields, field_mask)
        if graph_nodes is not None and edge_index is not None:
            g = self.graph_enc(graph_nodes, edge_index, edge_attr,
                               target_node=graph_target)
            embs["graph"] = g["embedding"]
        if mels is not None:
            embs["audio"] = self.audio_enc(mels)["embedding"]
        if tokens is not None:
            embs["text"] = self.text_enc(
                tokens, number_values=number_values,
                number_mask=number_mask, attention_mask=attention_mask)

        if len(embs) == 0:
            raise ValueError("At least one modality must be provided")
        return self.fusion(embs)

    def forward(
        self,
        *,
        book: torch.Tensor | None = None,
        ticks: torch.Tensor | None = None,
        fields: torch.Tensor | None = None,
        field_mask: torch.Tensor | None = None,
        graph_nodes: torch.Tensor | None = None,
        edge_index: torch.Tensor | None = None,
        edge_attr: torch.Tensor | None = None,
        graph_target: torch.Tensor | None = None,
        mels: torch.Tensor | None = None,
        tokens: torch.Tensor | None = None,
        number_values: torch.Tensor | None = None,
        number_mask: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        workspace_mode: str = "risk",
        competitor_price: torch.Tensor | None = None,
        market_size: torch.Tensor | None = None,
        target_tokens: torch.Tensor | None = None,
    ) -> dict:
        # Уровни 1–2
        fused = self.encode_modalities(
            book=book, ticks=ticks, fields=fields, field_mask=field_mask,
            graph_nodes=graph_nodes, edge_index=edge_index,
            edge_attr=edge_attr, graph_target=graph_target, mels=mels,
            tokens=tokens, number_values=number_values,
            number_mask=number_mask, attention_mask=attention_mask,
        ).unsqueeze(1)  # (B,1,D) — последовательность из одного «события»

        h, bus_info = self.bus(fused)

        # Уровень 3
        ws = self.workspace(
            h.squeeze(1), mode=workspace_mode,
            competitor_price=competitor_price, market_size=market_size,
            discount=self.cfg.discount_rate,
        )

        # Уровень 4
        out = self.output(ws["latent"].unsqueeze(1))
        out["bus"] = bus_info
        out["workspace"] = ws
        out["invariants"] = bus_info["invariants"]

        # Активный L_balance: реконструкция полей из латентности и проверка
        # балансовых тождеств по ПРЕДСКАЗАННЫМ значениям.
        balance_term = h.new_zeros(())
        if fields is not None:
            balance_term = (
                0.5 * self.field_head.reconstruction_loss(
                    h.squeeze(1), fields, field_mask)
                + 1.0 * self.field_head.balance_loss(h.squeeze(1))
            )
            out["predicted_fields"] = self.field_head.predict_fields(
                h.squeeze(1))
            out["balance_loss"] = balance_term.detach()

        # Потери
        if target_tokens is not None or fields is not None:
            if target_tokens is not None:
                pred_loss = self.loss_fn.predictive_loss(
                    out["logits"], target_tokens)
            else:
                # Нет языковой цели — обучаем на реконструкции полей
                pred_loss = balance_term.detach() * 0.0 + balance_term
            returns = ws["risk"]["pnl"]
            cost = out["economic"]["utility"]["cost"]
            total, logs = self.loss_fn(
                pred_loss, returns=returns, cost=cost,
                balance=balance_term,
                aux_loss=bus_info["aux_loss"])
            out["loss"] = total
            out["loss_logs"] = logs

        return out

    @torch.no_grad()
    def stream_ticks(
        self, book: torch.Tensor, ticks: torch.Tensor,
        window: int = 1000,
    ) -> list[dict]:
        """
        Стриминговый режим: подаёт тики порциями и поддерживает TTT-состояние.
        Память не растёт с числом событий (O(1) на тик).
        """
        results = []
        T = ticks.shape[1]
        for start in range(0, T, window):
            chunk = ticks[:, start:start + window]
            fused = self.orderbook_enc(book, chunk).unsqueeze(1)
            h, info = self.bus(fused, use_pos=False)
            ws = self.workspace(h.squeeze(1), mode="risk")
            results.append({
                "step": start,
                "risk": {k: v.cpu() for k, v in ws["risk"].items()
                         if torch.is_tensor(v)},
                "ttt_state_norm": info["ttt_states"][-1].norm().item(),
            })
        return results

    def reset_streaming(self) -> None:
        self.bus.reset_ttt_state()

    # ── Текстовое (RU+EN) обучение ───────────────────────────────
    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        number_values: torch.Tensor | None = None,
        number_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Кодирует текст послойно: embedding финансового энкодера -> проекция
        в d_value -> ядро Value Bus. Возвращает (B, T, d_value).
        """
        return self.text_enc(
            input_ids, number_values=number_values,
            number_mask=number_mask, attention_mask=attention_mask,
            return_sequence=True,
        )

    def text_forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        number_values: torch.Tensor | None = None,
        number_mask: torch.Tensor | None = None,
        add_economic_loss: bool = True,
    ) -> dict[str, torch.Tensor]:
        """
        Специализированный форвард для двуязычного LM-обучения.

        input_ids: (B, T)
        labels:    (B, T) с -100 на не-целевых позициях
        """
        h_seq = self.encode_text(
            input_ids, attention_mask=attention_mask,
            number_values=number_values, number_mask=number_mask)
        # Пропускаем последовательность через Value Bus (без TTT-таргета)
        h, bus_info = self.bus(h_seq, use_pos=True)
        logits = self.output.discrete(h)  # (B, T, vocab)

        result: dict[str, Any] = {
            "logits": logits,
            "hidden_states": h,
            "bus": bus_info,
        }

        if labels is not None:
            V = logits.shape[-1]
            lm_loss = nn.functional.cross_entropy(
                logits.reshape(-1, V), labels.reshape(-1),
                ignore_index=-100)
            total = lm_loss
            logs: dict[str, torch.Tensor] = {"lm_loss": lm_loss.detach()}

            # Экономическое плечо: полезность/VaR по пулированному представлению
            if add_economic_loss:
                if attention_mask is None:
                    attention_mask = input_ids.ne(self.tokenizer.pad_id)
                pooled = (h * attention_mask.unsqueeze(-1)).sum(1) / \
                    attention_mask.sum(1, keepdim=True).clamp(min=1)
                ws = self.workspace(
                    pooled, mode="risk",
                    discount=self.cfg.discount_rate)
                returns = ws["risk"]["pnl"]
                cost = self.output.continuous.utility(pooled)["cost"]
                total, elogs = self.loss_fn(
                    lm_loss, returns=returns, cost=cost,
                    aux_loss=bus_info["aux_loss"])
                logs.update({f"econ_{k}": v for k, v in elogs.items()})
                result["workspace"] = ws

            result["loss"] = total
            result["loss_logs"] = logs

        return result

    def count_parameters(self, trainable_only: bool = True) -> int:
        return sum(p.numel() for p in self.parameters()
                   if p.requires_grad or not trainable_only)
