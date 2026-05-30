"""Model profile — auto-detect model metadata from HuggingFace config.json.

Detects architecture, MoE, MLA, quantization, and other properties
needed for profiling and kernel selection.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Detected model metadata."""

    name: str = ""
    architecture: str = ""
    hidden_size: int = 0
    num_layers: int = 0
    num_heads: int = 0
    num_kv_heads: int = 0
    vocab_size: int = 0
    intermediate_size: int = 0
    is_moe: bool = False
    num_experts: int = 0
    num_experts_per_tok: int = 0
    is_mla: bool = False
    quantization: str = ""
    dtype: str = ""
    max_position_embeddings: int = 0


def detect_model_info(model_path: str) -> ModelInfo:
    """Auto-detect model info from config.json at model_path."""
    config_path = Path(model_path) / "config.json"
    if not config_path.exists():
        log.warning("No config.json found at %s", model_path)
        return ModelInfo(name=Path(model_path).name)

    try:
        data = json.loads(config_path.read_text())
    except json.JSONDecodeError:
        log.error("Failed to parse %s", config_path)
        return ModelInfo(name=Path(model_path).name)

    if "model" in data and isinstance(data["model"], dict):
        data = data["model"]

    arch = ""
    architectures = data.get("architectures", [])
    if architectures:
        arch = architectures[0]
    elif data.get("model_type"):
        arch = data["model_type"]

    is_moe = bool(data.get("num_local_experts") or data.get("num_experts"))
    is_mla = "mla" in arch.lower() or data.get("q_lora_rank", 0) > 0

    return ModelInfo(
        name=Path(model_path).name,
        architecture=arch,
        hidden_size=data.get("hidden_size", 0),
        num_layers=data.get("num_hidden_layers", 0),
        num_heads=data.get("num_attention_heads", 0),
        num_kv_heads=data.get("num_key_value_heads", 0),
        vocab_size=data.get("vocab_size", 0),
        intermediate_size=data.get("intermediate_size", 0),
        is_moe=is_moe,
        num_experts=data.get("num_local_experts", data.get("num_experts", 0)),
        num_experts_per_tok=data.get("num_experts_per_tok", 0),
        is_mla=is_mla,
        quantization=data.get("quantization_config", {}).get("quant_method", ""),
        dtype=data.get("torch_dtype", ""),
        max_position_embeddings=data.get("max_position_embeddings", 0),
    )
