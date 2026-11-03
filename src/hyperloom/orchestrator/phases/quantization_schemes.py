"""Quantization scheme enum -> natural-language ``--quantize`` prompt."""

from __future__ import annotations


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


def resolve_scheme_prompt(scheme: str | None) -> str | None:
    """Map a global scheme, already accepted by :func:`validate_scheme`, to its ``--quantize`` prompt."""
    if not scheme or scheme == NO_QUANTIZATION:
        return None
    return f"Quantization strategy:\nApply {scheme} as the global quantization scheme."


__all__ = [
    "NO_QUANTIZATION",
    "SUPPORTED_SCHEMES",
    "MI355X_ONLY",
    "QUANT_SCHEME_CHOICES",
    "SchemeNotSupportedError",
    "supported_schemes",
    "validate_scheme",
    "resolve_scheme_prompt",
]
