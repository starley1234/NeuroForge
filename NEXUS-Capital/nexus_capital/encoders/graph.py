"""
Relational Graph Neural Operator для сетей цепочек поставок, графов
контрагентов и B2B-цепочек.

Моделирует риск дефолта контрагентов как диффузию напряжения по графу:
каждые соседние узлы передают сигнал риска, а центральный узел аккумулирует
экспозицию. Используется простой, но устойчивый GraphSAGE-подобный блок
с нелинейной агрегацией и edge-фичами (логистические задержки, CDS-спреды).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core.layers import RMSNorm


class GraphRiskOperator(nn.Module):
    def __init__(self, d_value: int, node_dim: int = 64,
                 edge_dim: int = 8, n_layers: int = 3):
        super().__init__()
        self.node_proj = nn.Linear(node_dim, d_value)
        self.edge_proj = nn.Linear(edge_dim, d_value)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "self_lin": nn.Linear(d_value, d_value),
                "agg_lin": nn.Linear(d_value, d_value),
                "edge_lin": nn.Linear(d_value, d_value),
                "norm": RMSNorm(d_value),
            }) for _ in range(n_layers)
        ])
        self.pool = nn.Sequential(
            nn.Linear(d_value, d_value),
            nn.SiLU(),
            nn.Linear(d_value, d_value),
        )
        # Голова риска дефолта (вероятность)
        self.default_head = nn.Sequential(
            nn.Linear(d_value, d_value // 2),
            nn.SiLU(),
            nn.Linear(d_value // 2, 1),
        )

    def forward(
        self,
        nodes: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        target_node: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        nodes:      (N, node_dim) признаки узлов
        edge_index: (2, E) рёбра (source, target)
        edge_attr:  (E, edge_dim) атрибуты рёбер (задержка, спред, объём)
        target_node:(B,) индексы центральных узлов для пулинга; если None —
                    возвращается среднее по всем узлам.
        """
        h = self.node_proj(nodes)
        e = self.edge_proj(edge_attr)

        src, dst = edge_index[0], edge_index[1]
        for layer in self.layers:
            agg = torch.zeros_like(h)
            msg = layer["edge_lin"](e) * h[src]            # (E, D)
            agg.index_add_(0, dst, msg)
            norm = torch.zeros(h.shape[0], 1, device=h.device)
            ones = torch.ones(msg.shape[0], 1, device=h.device)
            norm.index_add_(0, dst, ones).clamp_(min=1.0)
            agg = agg / norm
            h = layer["norm"](F.silu(layer["self_lin"](h))
                              + layer["agg_lin"](agg))

        if target_node is not None:
            h_target = h[target_node]
        else:
            h_target = h.mean(dim=0, keepdim=True)

        pooled = self.pool(h_target)
        pd = torch.sigmoid(self.default_head(pooled))
        return {
            "embedding": pooled,
            "node_states": h,
            "default_prob": pd.squeeze(-1),
        }
