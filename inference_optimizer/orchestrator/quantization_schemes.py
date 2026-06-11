"""Structured quantization config -> natural-language ``--quantize`` prompt.

The frontend / backend builds a structured quantization request (global scheme
+ optional per-layer overrides, kv_cache, calibration, eval gap); this module
turns it into the natural-language prompt the quantization-agent expects.
Free-text ``--quantize`` stays available for power users — this is the
structured path, so the config -> prompt translation lives in exactly one
place and is reused by the UI's editable "Preview" prompt.

Two invariants, both from issue #453:

* **No hard-coded defaults.** :func:`build_quantization_prompt` only emits a
  sentence for a field the caller set explicitly. Anything left unset is
  *omitted* so Quark's intake + plan skill fills it from its own defaults
  (``exclude_layers`` / ``pileval`` / ``num_calib_data`` / ``seq_len`` / ...).
  The UI is responsible for surfacing Quark's defaults if it wants them
  visible in the preview — it passes them in via the config.
* **GPU-constrained schemes.** ``fp8`` / ``ptpc_fp8`` work on any DCGPU;
  ``mxfp4`` / ``mxfp4_fp8`` are MI355X-only. :func:`validate_scheme` enforces
  this so an mxfp4 request on an mi300x target fails loudly instead of
  producing an unservable artifact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence


# Sentinel for "do not quantize" (the dropdown default).
NO_QUANTIZATION = "none"

# The full set of serving-validated global schemes (issue #453 §4).
SUPPORTED_SCHEMES: tuple[str, ...] = ("fp8", "ptpc_fp8", "mxfp4", "mxfp4_fp8")

# Schemes that require MI355X-class hardware; offered on no other GPU type.
MI355X_ONLY: frozenset[str] = frozenset({"mxfp4", "mxfp4_fp8"})

# argparse ``choices=`` for the structured CLI flag (``none`` = no quantization).
QUANT_SCHEME_CHOICES: list[str] = [NO_QUANTIZATION, *SUPPORTED_SCHEMES]


class SchemeNotSupportedError(ValueError):
    """A scheme was requested on a GPU type that does not support it."""


def supported_schemes(gpu_type: str | None) -> list[str]:
    """Return the schemes selectable for ``gpu_type``.

    MI355X gets the full set; every other (DCGPU) target drops the
    ``mxfp4`` / ``mxfp4_fp8`` MI355X-only schemes.
    """
    if (gpu_type or "").strip().lower() == "mi355x":
        return list(SUPPORTED_SCHEMES)
    return [s for s in SUPPORTED_SCHEMES if s not in MI355X_ONLY]


def validate_scheme(scheme: str | None, gpu_type: str | None) -> None:
    """Raise if ``scheme`` is unknown or unsupported on ``gpu_type``.

    No-op for the ``none`` sentinel / empty scheme. Raises ``ValueError`` for
    an unknown scheme and :class:`SchemeNotSupportedError` for an MI355X-only
    scheme on a *known* non-MI355X target. When ``gpu_type`` is empty/unknown
    (the real GPU is resolved later via the rocm-smi probe) the hardware
    constraint is not enforced here — the constraint check needs a concrete
    target to act on.
    """
    if not scheme or scheme == NO_QUANTIZATION:
        return
    if scheme not in SUPPORTED_SCHEMES:
        raise ValueError(
            f"unknown quantization scheme {scheme!r}; "
            f"choose one of {list(SUPPORTED_SCHEMES)}"
        )
    gpu = (gpu_type or "").strip().lower()
    if scheme in MI355X_ONLY and gpu and gpu != "mi355x":
        raise SchemeNotSupportedError(
            f"quantization scheme {scheme!r} requires an MI355X target, "
            f"but the GPU type is {gpu!r}; supported on {gpu!r}: "
            f"{supported_schemes(gpu)}"
        )


@dataclass(frozen=True)
class QuantizationConfig:
    """Structured quantization request (issue #453 §4).

    Only ``global_scheme`` is required. Every other field is optional and,
    when left at its ``None`` / empty default, is omitted from the generated
    prompt so Quark's intake + plan skill supplies the default. ``output_dir``
    is optional here because the inference_optimizer prelude injects the export
    directory itself; standalone callers may set it to fold the destination
    into the prompt.
    """

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
    """Render the quantization-strategy paragraph for a prompt.

    Args:
        cfg: Quantization configuration to describe.

    Returns:
        A prose paragraph covering the global scheme, layer overrides,
        kv-cache handling, exclusions, and output directory.
    """
    sentences = [f"Apply {cfg.global_scheme} as the global quantization scheme."]
    if cfg.layer_overrides:
        clauses = [
            f"the {layer} layers with {scheme}"
            for layer, scheme in cfg.layer_overrides.items()
        ]
        sentences.append(f"Override {_join_clauses(clauses)}.")
    if cfg.kv_cache:
        sentences.append(f"Quantize the kv_cache with {cfg.kv_cache}.")
    if cfg.exclude_layers:
        sentences.append(
            f"Additionally exclude {_join_clauses(list(cfg.exclude_layers))} "
            f"from quantization."
        )
    if cfg.output_dir:
        sentences.append(f"Write the quantized model to {cfg.output_dir}.")
    return "Quantization strategy:\n" + " ".join(sentences)


def _calibration_paragraph(cfg: QuantizationConfig) -> str | None:
    """Render the calibration paragraph, composing only set fields.

    Args:
        cfg: Quantization configuration to describe.

    Returns:
        A calibration paragraph, or ``None`` when no calibration fields
        are configured.
    """
    if cfg.calib_dataset is None and cfg.num_calib_data is None and cfg.seq_len is None:
        return None
    # Compose only the parts that were set, e.g. "Calibrate with the pileval
    # dataset using 512 samples at a sequence length of 2048."
    head = "Calibrate"
    if cfg.calib_dataset is not None:
        head += f" with the {cfg.calib_dataset} dataset"
    if cfg.num_calib_data is not None:
        head += f" using {cfg.num_calib_data} samples"
    if cfg.seq_len is not None:
        head += f" at a sequence length of {cfg.seq_len}"
    return "Calibration:\n" + head + "."


def _evaluation_paragraph(cfg: QuantizationConfig) -> str | None:
    """Render the evaluation paragraph describing the accuracy budget.

    Args:
        cfg: Quantization configuration to describe.

    Returns:
        An evaluation paragraph, or ``None`` when no acceptable eval gap
        is configured.
    """
    if cfg.acceptable_eval_gap is None:
        return None
    pct = f"{cfg.acceptable_eval_gap * 100:g}"
    return (
        "Evaluation:\n"
        f"Keep the quantized model's accuracy within {pct}% of the bf16 baseline."
    )


def build_quantization_prompt(
    cfg: QuantizationConfig,
    *,
    model_path: str | None = None,
    gpu_type: str | None = None,
    skill_path: str | None = None,
) -> str:
    """Render ``cfg`` into the natural-language quantization prompt.

    The output mirrors issue #453 §5.6: an optional intro line followed by
    three light-headed groups — **Quantization strategy**, **Calibration**,
    **Evaluation**. Groups with no explicitly-set fields are dropped entirely
    (no hard-coded defaults). ``model_path`` / ``skill_path`` / ``gpu_type``
    populate the intro line; the inference_optimizer prelude leaves them unset
    because its adapter folds the source model + export dir into the prompt.
    """
    paragraphs: list[str] = []

    if model_path:
        target = f" on an {gpu_type.upper()} target" if gpu_type else ""
        if skill_path:
            paragraphs.append(
                f"Use the skill at {skill_path} to quantize {model_path}{target}."
            )
        else:
            paragraphs.append(f"Quantize {model_path}{target}.")

    paragraphs.append(_strategy_paragraph(cfg))
    for para in (_calibration_paragraph(cfg), _evaluation_paragraph(cfg)):
        if para:
            paragraphs.append(para)

    return "\n\n".join(paragraphs)


def resolve_scheme_prompt(scheme: str | None) -> str | None:
    """Map a global scheme enum to its ``--quantize`` prompt.

    Returns ``None`` when ``scheme`` is falsy or ``"none"`` (= no
    quantization). Raises ``ValueError`` for an unknown scheme so a typo'd enum
    fails loudly instead of silently skipping quantization. The prompt carries
    the global scheme only — no hard-coded kv_cache / exclude_layers defaults;
    Quark's intake + plan skill supplies those. Per-layer overrides and the
    other §4 knobs travel through :func:`build_quantization_prompt` with a fully
    populated :class:`QuantizationConfig` (the structured UI path).
    """
    if not scheme or scheme == NO_QUANTIZATION:
        return None
    if scheme not in SUPPORTED_SCHEMES:
        raise ValueError(
            f"unknown quantization scheme {scheme!r}; "
            f"choose one of {list(SUPPORTED_SCHEMES)}"
        )
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
