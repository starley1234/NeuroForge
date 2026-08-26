"""Загрузка NexusConfig из YAML-файлов конфигурации."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .config import NexusConfig, small_config, rtx5060_config


def _flatten(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, prefix=key + "_"))
        else:
            out[key] = v
    return out


def load_config(path: str | Path) -> NexusConfig:
    """
    Читает YAML вида:
        model: {d_value: 1536, ...}
        encoders: {...}
        ...
    и преобразует в плоский NexusConfig.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    flat = _flatten(raw)

    # Сопоставление имён секций с полями дата-класса
    mapping = {
        "model_d_value": "d_value",
        "model_n_layers": "n_layers",
        "model_n_heads": "n_heads",
        "model_d_ff": "d_ff",
        "model_dropout": "dropout",
        "encoders_orderbook_levels": "orderbook_levels",
        "encoders_tick_features": "tick_features",
        "encoders_tabular_features": "tabular_features",
        "encoders_graph_node_dim": "graph_node_dim",
        "encoders_audio_n_mels": "audio_n_mels",
        "encoders_text_vocab_size": "text_vocab_size",
        "encoders_text_seq_len": "text_seq_len",
        "ttt_rank": "ttt_rank",
        "ttt_alpha": "ttt_alpha",
        "ttt_eta": "ttt_eta",
        "ttt_mem_dim": "ttt_mem_dim",
        "moe_n_experts": "n_experts",
        "moe_experts_per_token": "experts_per_token",
        "moe_expert_ff": "expert_ff",
        "moe_expert_names": "expert_names",
        "sde_hidden": "sde_hidden",
        "sde_steps": "sde_steps",
        "sde_mc_paths": "mc_paths",
        "sde_mc_horizon": "mc_horizon",
        "utility_gamma_risk": "gamma_risk",
        "utility_risk_alpha": "risk_alpha",
        "utility_lambda_utility": "lambda_utility",
        "utility_lambda_var": "lambda_var",
        "utility_discount_rate": "discount_rate",
    }
    kwargs: dict[str, Any] = {}
    for yaml_key, cfg_field in mapping.items():
        if yaml_key in flat and cfg_field in NexusConfig.__dataclass_fields__:
            kwargs[cfg_field] = flat[yaml_key]

    # Производные поля для текстового позиционного кодирования
    if "text_seq_len" in kwargs:
        kwargs["text_n_pos"] = kwargs["text_seq_len"]
        kwargs["max_seq_len"] = kwargs["text_seq_len"]
    if "text_vocab_size" in kwargs:
        kwargs["vocab_size"] = kwargs["text_vocab_size"]

    return NexusConfig(**kwargs)


def named_config(name: str) -> NexusConfig:
    if name in ("small", "cpu", "tiny"):
        return small_config()
    if name in ("rtx5060", "gpu", "mvp"):
        return rtx5060_config()
    raise ValueError(f"Unknown preset: {name}")
