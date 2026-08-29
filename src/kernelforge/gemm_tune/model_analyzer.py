# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Model analysis: read config.json and extract GEMM-relevant parameters."""

from __future__ import annotations

import contextlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class ModelProfile:
    """Extracted model characteristics relevant to GEMM tuning."""

    model_path: str
    # Architecture
    architecture: str = ""
    # MoE detection
    is_moe: bool = False
    num_experts: int = 0
    num_experts_per_tok: int = 0  # topk
    # Dimensions
    hidden_size: int = 0
    intermediate_size: int = 0  # dense MLP
    moe_intermediate_size: int = 0  # MoE MLP (may differ)
    num_hidden_layers: int = 0
    # Attention heads
    num_attention_heads: int = 0
    num_key_value_heads: int = 0
    # Per-head dims (0 = derive head_dim from hidden_size // num_attention_heads).
    # v_head_dim may differ from the qk head_dim (e.g. MiMo, DeepSeek MLA).
    head_dim: int = 0
    v_head_dim: int = 0
    # MLA (deepseek_v3) low-rank attention dims (0 when the model is not MLA).
    q_lora_rank: int = 0
    kv_lora_rank: int = 0
    qk_nope_head_dim: int = 0
    qk_rope_head_dim: int = 0
    o_lora_rank: int = 0
    o_groups: int = 0
    # Activation
    hidden_act: str = "silu"
    # Quantization (from config or CLI override)
    quant_method: str = ""  # "", "fp8", "awq", "gptq", "compressed-tensors"
    quant_bits: int = 0
    quant_group_size: int = 0
    # Gate/up fusion: most MoE models fuse gate+up (use_g1u1=1)
    use_g1u1: bool = True
    # Model dtype from config.json (torch_dtype field)
    model_dtype: str = "bfloat16"
    # Raw config for advanced consumers
    raw_config: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def effective_moe_intermediate(self) -> int:
        """The intermediate size relevant for MoE GEMM shapes."""
        return self.moe_intermediate_size or self.intermediate_size

    @property
    def unquantized_linear_modules(self) -> list[str]:
        """Linear modules the checkpoint deliberately left at the model dtype.

        Quantization is decided per module, not per model. A checkpoint labelled
        ``mxfp4`` or ``fp8`` normally keeps ``lm_head`` and the attention
        projections in bf16 -- they are the numerically sensitive ones, and on a
        MoE model they are a rounding error of the weight bytes anyway (experts
        carry ~99% of the parameters). Quark writes that list as ``exclude``;
        AWQ/GPTQ call it ``modules_to_not_convert``.

        Norm layers are dropped: they are in the same list but are not GEMMs.

        Returns:
            Module names, empty when the config records no exclusions.
        """
        qconfig = self.raw_config.get("quantization_config")
        if not isinstance(qconfig, dict):
            return []
        for key in ("exclude", "modules_to_not_convert", "exclude_layers"):
            entries = qconfig.get(key)
            if isinstance(entries, list) and entries:
                return [str(name) for name in entries if "norm" not in str(name).lower()]
        return []

    @property
    def keeps_dense_layers_at_model_dtype(self) -> bool:
        """Whether substantial dense GEMMs stay at model dtype after quantization.

        The router asks this because ``precision`` cannot answer it: that field
        describes the weight format of the quantized majority, while the
        untouched minority is what the dense GEMM path actually dispatches.
        ``lm_head`` alone is excluded: one output projection per forward does
        not justify competing with the quantized dense tuner for a shared time
        budget. Runtime evidence can still request bf16 tuning explicitly.
        """
        return any(name.rsplit(".", 1)[-1].lower() != "lm_head" for name in self.unquantized_linear_modules)

    @property
    def activation_type_str(self) -> str:
        """Map hidden_act to aiter ActivationType enum string."""
        mapping = {
            "silu": "ActivationType.Silu",
            "swiglu": "ActivationType.Silu",
            "gelu": "ActivationType.Gelu",
            "gelu_new": "ActivationType.Gelu",
            "gelu_fast": "ActivationType.Gelu",
            "relu": "ActivationType.Relu",
        }
        return mapping.get(self.hidden_act.lower(), "ActivationType.Silu")


def _resolve_llm_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return the sub-dict that holds LLM params (MoE, dimensions, etc.).

    VL / multi-modal models (Qwen3VLMoe, Qwen3_5MoeForConditionalGeneration,
    etc.) nest the language model config under keys like ``text_config``,
    ``language_config``, or ``llm_config``.  Pure LLMs keep everything at the
    top level.  We return a merged view: nested values override top-level ones
    so that callers always find the right fields.
    """
    for key in ("text_config", "language_config", "llm_config"):
        nested = config.get(key)
        if isinstance(nested, dict) and nested:
            merged = dict(config)
            merged.update(nested)
            return merged
    return config


def _extract_quant_info(config: dict[str, Any]) -> tuple[str, int, int]:
    """Extract quantization method, bits, and group_size from config."""
    qconfig = config.get("quantization_config", {})
    if not isinstance(qconfig, dict):
        return "", 0, 0

    method = str(qconfig.get("quant_method", "")).strip()
    bits = 0
    group_size = 0

    with contextlib.suppress(TypeError, ValueError):
        # AWQ / GPTQ style
        if "bits" in qconfig and qconfig["bits"] is not None:
            bits = int(qconfig["bits"])
        if "group_size" in qconfig and qconfig["group_size"] is not None:
            group_size = int(qconfig["group_size"])

        # compressed-tensors style
        if method == "compressed-tensors":
            groups = qconfig.get("config_groups", {})
            if isinstance(groups, dict):
                for _, grp in groups.items():
                    weights = grp.get("weights", {})
                    if isinstance(weights, dict):
                        nb = weights.get("num_bits")
                        gs = weights.get("group_size")
                        if nb is not None:
                            bits = int(nb)
                        if gs is not None:
                            group_size = int(gs)
                        break

    return method, bits, group_size


def analyze_model(model_path: str) -> ModelProfile:
    """Read config.json from model_path and build a ModelProfile.

    Args:
        model_path: Path to model directory (must contain config.json).

    Returns:
        ModelProfile with all extracted fields.

    Raises:
        FileNotFoundError: If config.json does not exist.
        json.JSONDecodeError: If config.json is malformed.
    """
    config_file = Path(model_path) / "config.json"
    if not config_file.is_file():
        raise FileNotFoundError(f"config.json not found at {config_file}")

    config = json.loads(config_file.read_text(encoding="utf-8"))

    # Architecture
    architectures = config.get("architectures", [])
    arch = architectures[0] if architectures else config.get("model_type", "")

    # For VL / multi-modal models, the LLM params live inside a nested config.
    # Merge the nested dict so downstream lookups find MoE / dimension fields.
    llm_cfg = _resolve_llm_config(config)

    # MoE detection — check multiple field names across model families
    num_experts = int(
        llm_cfg.get("num_local_experts", 0) or llm_cfg.get("num_experts", 0) or llm_cfg.get("n_routed_experts", 0)
    )
    topk = int(
        llm_cfg.get("num_experts_per_tok", 0) or llm_cfg.get("num_selected_experts", 0) or llm_cfg.get("top_k", 0)
    )
    is_moe = num_experts > 1

    # Dimensions
    hidden_size = int(llm_cfg.get("hidden_size", 0))
    intermediate_size = int(llm_cfg.get("intermediate_size", 0))
    moe_intermediate_size = int(llm_cfg.get("moe_intermediate_size", 0))
    num_hidden_layers = int(llm_cfg.get("num_hidden_layers", 0))

    # Activation
    hidden_act = str(llm_cfg.get("hidden_act", "silu")).lower()

    # Model dtype
    model_dtype = str(llm_cfg.get("torch_dtype", config.get("torch_dtype", "bfloat16"))).replace("torch.", "")

    # Attention heads
    num_attention_heads = int(llm_cfg.get("num_attention_heads", 0))
    num_key_value_heads = int(llm_cfg.get("num_key_value_heads", num_attention_heads))

    # Per-head dims. ``head_dim`` is the qk head dim (config-explicit or derived);
    # ``v_head_dim`` defaults to it when not separately specified.
    head_dim = int(llm_cfg.get("head_dim", 0))
    v_head_dim = int(llm_cfg.get("v_head_dim", 0))
    # MLA low-rank dims (deepseek_v3 family); 0 when absent.
    q_lora_rank = int(llm_cfg.get("q_lora_rank", 0) or 0)
    kv_lora_rank = int(llm_cfg.get("kv_lora_rank", 0) or 0)
    qk_nope_head_dim = int(llm_cfg.get("qk_nope_head_dim", 0) or 0)
    qk_rope_head_dim = int(llm_cfg.get("qk_rope_head_dim", 0) or 0)
    o_lora_rank = int(llm_cfg.get("o_lora_rank", 0) or 0)
    o_groups = int(llm_cfg.get("o_groups", 0) or 0)

    # Quantization
    quant_method, quant_bits, quant_group_size = _extract_quant_info(config)

    # Gate/up fusion heuristic: almost all modern MoE models use fused gate+up
    # Exception: some very old models or custom architectures
    use_g1u1 = True

    profile = ModelProfile(
        model_path=model_path,
        architecture=arch,
        is_moe=is_moe,
        num_experts=num_experts,
        num_experts_per_tok=topk,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        moe_intermediate_size=moe_intermediate_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        v_head_dim=v_head_dim,
        q_lora_rank=q_lora_rank,
        kv_lora_rank=kv_lora_rank,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        o_lora_rank=o_lora_rank,
        o_groups=o_groups,
        hidden_act=hidden_act,
        quant_method=quant_method,
        quant_bits=quant_bits,
        quant_group_size=quant_group_size,
        use_g1u1=use_g1u1,
        model_dtype=model_dtype,
        raw_config=config,
    )

    log.info(
        "Model analysis: arch=%s, is_moe=%s, experts=%d, topk=%d, "
        "hidden=%d, inter=%d, moe_inter=%d, heads=%d, kv_heads=%d, "
        "quant=%s/%d-bit, dtype=%s",
        arch,
        is_moe,
        num_experts,
        topk,
        hidden_size,
        intermediate_size,
        moe_intermediate_size,
        num_attention_heads,
        num_key_value_heads,
        quant_method or "none",
        quant_bits,
        model_dtype,
    )
    return profile
