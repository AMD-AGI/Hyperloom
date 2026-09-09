"""Structured quantization config -> natural-language ``--quantize`` prompt."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence


# Sentinel for "do not quantize" (the dropdown default).
NO_QUANTIZATION = "none"

# The full set of serving-validated global schemes.
SUPPORTED_SCHEMES: tuple[str, ...] = ("fp8", "ptpc_fp8", "mxfp4", "mxfp4_fp8")

# Schemes that require MI355X-class hardware; offered on no other GPU type.
MI355X_ONLY: frozenset[str] = frozenset({"mxfp4", "mxfp4_fp8"})

# argparse ``choices=`` for the structured CLI flag (``none`` = no quantization).
QUANT_SCHEME_CHOICES: list[str] = [NO_QUANTIZATION, *SUPPORTED_SCHEMES]


class SchemeNotSupportedError(ValueError):
    """A scheme was requested on a GPU type that does not support it."""


def supported_schemes(gpu_type: str | None) -> list[str]:
    """Return the schemes selectable for ``gpu_type``."""
    if (gpu_type or "").strip().lower() == "mi355x":
        return list(SUPPORTED_SCHEMES)
    return [s for s in SUPPORTED_SCHEMES if s not in MI355X_ONLY]


def validate_scheme(scheme: str | None, gpu_type: str | None) -> None:
    """Raise if ``scheme`` is unknown or unsupported on ``gpu_type``."""
    if not scheme or scheme == NO_QUANTIZATION:
        return
    if scheme not in SUPPORTED_SCHEMES:
        raise ValueError(f"unknown quantization scheme {scheme!r}; choose one of {list(SUPPORTED_SCHEMES)}")
    gpu = (gpu_type or "").strip().lower()
    if scheme in MI355X_ONLY and gpu and gpu != "mi355x":
        raise SchemeNotSupportedError(
            f"quantization scheme {scheme!r} requires an MI355X target, "
            f"but the GPU type is {gpu!r}; supported on {gpu!r}: "
            f"{supported_schemes(gpu)}"
        )


@dataclass(frozen=True)
class QuantizationConfig:
    """Structured quantization request."""

    global_scheme: str
    output_dir: str | None = None
    # Per-layer / per-group overrides, e.g. {"self_attn": "fp8", "moe/mlp": "ptpc_fp8"}.
    layer_overrides: Mapping[str, str] = field(default_factory=dict)
    kv_cache: str | None = None  # only "fp8" supported today; None = off.
    exclude_layers: Sequence[str] = field(default_factory=tuple)
    calib_dataset: str | None = None
    num_calib_data: int | None = None
    seq_len: int | None = None
    acceptable_eval_gap: float | None = None  # relative, e.g. 0.03 = 3%.


def _join_clauses(items: Sequence[str]) -> str:
    """Join clauses as ``a``, ``a and b``, or ``a, b and c``."""
    items = list(items)
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return f"{', '.join(items[:-1])} and {items[-1]}"


def _strategy_paragraph(cfg: QuantizationConfig) -> str:
    """Render the quantization-strategy paragraph for a prompt."""
    sentences = [f"Apply {cfg.global_scheme} as the global quantization scheme."]
    if cfg.layer_overrides:
        clauses = [f"the {layer} layers with {scheme}" for layer, scheme in cfg.layer_overrides.items()]
        sentences.append(f"Override {_join_clauses(clauses)}.")
    if cfg.kv_cache:
        sentences.append(f"Quantize the kv_cache with {cfg.kv_cache}.")
    if cfg.exclude_layers:
        sentences.append(f"Additionally exclude {_join_clauses(list(cfg.exclude_layers))} from quantization.")
    if cfg.output_dir:
        sentences.append(f"Write the quantized model to {cfg.output_dir}.")
    return "Quantization strategy:\n" + " ".join(sentences)


def _calibration_paragraph(cfg: QuantizationConfig) -> str | None:
    """Render the calibration paragraph, composing only set fields."""
    if cfg.calib_dataset is None and cfg.num_calib_data is None and cfg.seq_len is None:
        return None
    # Compose only the parts that were set.
    head = "Calibrate"
    if cfg.calib_dataset is not None:
        head += f" with the {cfg.calib_dataset} dataset"
    if cfg.num_calib_data is not None:
        head += f" using {cfg.num_calib_data} samples"
    if cfg.seq_len is not None:
        head += f" at a sequence length of {cfg.seq_len}"
    return "Calibration:\n" + head + "."


def _evaluation_paragraph(cfg: QuantizationConfig) -> str | None:
    """Render the evaluation paragraph describing the accuracy budget."""
    if cfg.acceptable_eval_gap is None:
        return None
    pct = f"{cfg.acceptable_eval_gap * 100:g}"
    return f"Evaluation:\nKeep the quantized model's accuracy within {pct}% of the bf16 baseline."


def build_quantization_prompt(
    cfg: QuantizationConfig,
    *,
    model_path: str | None = None,
    gpu_type: str | None = None,
    skill_path: str | None = None,
) -> str:
    """Render ``cfg`` into the natural-language quantization prompt."""
    paragraphs: list[str] = []

    if model_path:
        target = f" on an {gpu_type.upper()} target" if gpu_type else ""
        if skill_path:
            paragraphs.append(f"Use the skill at {skill_path} to quantize {model_path}{target}.")
        else:
            paragraphs.append(f"Quantize {model_path}{target}.")

    paragraphs.append(_strategy_paragraph(cfg))
    for para in (_calibration_paragraph(cfg), _evaluation_paragraph(cfg)):
        if para:
            paragraphs.append(para)

    return "\n\n".join(paragraphs)


def resolve_scheme_prompt(scheme: str | None) -> str | None:
    """Map a global scheme enum to its ``--quantize`` prompt."""
    if not scheme or scheme == NO_QUANTIZATION:
        return None
    if scheme not in SUPPORTED_SCHEMES:
        raise ValueError(f"unknown quantization scheme {scheme!r}; choose one of {list(SUPPORTED_SCHEMES)}")
    return build_quantization_prompt(QuantizationConfig(global_scheme=scheme))


__all__ = [
    "NO_QUANTIZATION",
    "SUPPORTED_SCHEMES",
    "MI355X_ONLY",
    "QUANT_SCHEME_CHOICES",
    "SchemeNotSupportedError",
    "QuantizationConfig",
    "supported_schemes",
    "validate_scheme",
    "build_quantization_prompt",
    "resolve_scheme_prompt",
]
