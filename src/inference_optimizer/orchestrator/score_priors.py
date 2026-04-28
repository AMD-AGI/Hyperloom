"""Per model-class action priors — DESIGN §9.2 (STUB v0.5).

Initial scores baked from sprint+marathon runs. Scheduler multiplies these
priors with pressure / mode_gate / depth_gate / diminishing.

STATUS:
    Skeleton only. The scoring matrix below is illustrative; final values
    must be filled in per IMPLEMENTATION-CHECKLIST §3.31‒3.38.

References:
    - DESIGN §9.2 Initial priors
    - DESIGN §9.3 Score update rules
"""
from __future__ import annotations

from enum import Enum


class ModelClass(str, Enum):
    DENSE = "dense"
    MOE_MLA = "moe_mla"
    MOE_SWA = "moe_swa"
    MOE_MLA_NSA = "moe_mla_nsa"
    UNKNOWN = "unknown"


# Each row: action_name → initial prior (higher = more likely picked first).
# Values from DESIGN §9.2 table.
INITIAL_PRIORS: dict[ModelClass, dict[str, float]] = {
    ModelClass.DENSE: {
        "backends":     3.0,
        "params":       5.0,
        "kernel-opt":   8.0,
        "torch.compile": 7.0,
        "sweep":        1.0,
    },
    ModelClass.MOE_MLA: {
        "backends":     9.0,
        "params":       6.0,
        "kernel-opt":   2.0,
        "torch.compile": 0.0,
        "sweep":        1.0,
    },
    ModelClass.MOE_SWA: {
        "backends":     8.0,
        "params":       7.0,
        "kernel-opt":   2.0,
        "torch.compile": 0.0,
        "sweep":        1.0,
    },
    ModelClass.MOE_MLA_NSA: {
        "backends":    10.0,
        "params":       5.0,
        "kernel-opt":   2.0,
        "torch.compile": 0.0,
        "sweep":        1.0,
    },
    # default for unknown classifications
    ModelClass.UNKNOWN: {
        "backends":     5.0,
        "params":       5.0,
        "kernel-opt":   5.0,
        "torch.compile": 3.0,
        "sweep":        1.0,
    },
}


_DEFAULT_PRIOR: float = 1.0


def _coerce_class(model_class: ModelClass | str) -> ModelClass:
    if isinstance(model_class, ModelClass):
        return model_class
    try:
        return ModelClass(str(model_class))
    except ValueError:
        return ModelClass.UNKNOWN


def prior_for(
    model_class: ModelClass | str,
    action_name: str,
    *,
    default: float = _DEFAULT_PRIOR,
) -> float:
    """Return the initial prior for ``action_name`` under ``model_class``.

    Treats hyphen / underscore variants of action names as equivalent so
    DESIGN tables (``kernel-opt``) and YAML filenames (``kernel_opt``)
    line up.
    """
    cls = _coerce_class(model_class)
    table = INITIAL_PRIORS.get(cls, INITIAL_PRIORS[ModelClass.UNKNOWN])
    if action_name in table:
        return table[action_name]
    # try canonical forms in both directions
    for variant in (action_name.replace("_", "-"), action_name.replace("-", "_")):
        if variant in table:
            return table[variant]
    return default


# ---------------------------------------------------------------------------
# Heuristic classifier
# ---------------------------------------------------------------------------
_DENSE_PATTERNS: tuple[str, ...] = (
    "gpt-oss",
    "gpt_oss",
    "llama",
    "qwen-dense",
    "qwen_dense",
    "qwen2-dense",
    "-dense",         # catches "Qwen2-7B-Dense" after lowercasing
    "_dense",
    "phi-3",
    "mistral-7b",
)
# DeepSeek-V2 / V3 with MLA + MoE layers
_MOE_MLA_PATTERNS: tuple[str, ...] = (
    "deepseek-v2",
    "deepseek-v3",
    "deepseek_v2",
    "deepseek_v3",
    "deepseek-coder-v",
)
# Mixtral / SWA-style MoE
_MOE_SWA_PATTERNS: tuple[str, ...] = (
    "mixtral",
    "qwen-moe",
    "qwen_moe",
    "qwen2-moe",
)
# Kimi / NSA-bearing MoE+MLA
_MOE_MLA_NSA_PATTERNS: tuple[str, ...] = (
    "kimi",
    "nsa",
)


def classify_model(model_path: str) -> ModelClass:
    """Pattern-match a model path to a :class:`ModelClass`.

    The matcher is case-insensitive and uses substring containment so it
    works for both HF hub paths (``deepseek-ai/DeepSeek-V3-0324``) and
    local paths (``/srv/models/llama-3-8b-instruct``).
    """
    if not model_path:
        return ModelClass.UNKNOWN
    needle = str(model_path).lower()
    # Order matters: NSA first (most specific), then MLA, then SWA, then DENSE.
    for pat in _MOE_MLA_NSA_PATTERNS:
        if pat in needle:
            return ModelClass.MOE_MLA_NSA
    for pat in _MOE_MLA_PATTERNS:
        if pat in needle:
            return ModelClass.MOE_MLA
    for pat in _MOE_SWA_PATTERNS:
        if pat in needle:
            return ModelClass.MOE_SWA
    for pat in _DENSE_PATTERNS:
        if pat in needle:
            return ModelClass.DENSE
    return ModelClass.UNKNOWN


__all__ = [
    "ModelClass",
    "INITIAL_PRIORS",
    "prior_for",
    "classify_model",
]
