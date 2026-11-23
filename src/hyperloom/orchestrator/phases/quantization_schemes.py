"""Quantization scheme enum -> natural-language ``--quantize`` prompt."""

from __future__ import annotations

from hyperloom_common.gpu_card import peak_rate

# Sentinel for "do not quantize" (the dropdown default).
NO_QUANTIZATION = "none"

# The full set of serving-validated global schemes.
SUPPORTED_SCHEMES: tuple[str, ...] = ("fp8", "ptpc_fp8", "mxfp4", "mxfp4_fp8")

# Schemes whose weights are fp4: offered only on an arch with an fp4 matrix rate.
FP4_SCHEMES: frozenset[str] = frozenset({"mxfp4", "mxfp4_fp8"})

# argparse ``choices=`` for the structured CLI flag (``none`` = no quantization).
QUANT_SCHEME_CHOICES: list[str] = [NO_QUANTIZATION, *SUPPORTED_SCHEMES]


class SchemeNotSupportedError(ValueError):
    """A scheme was requested on a GPU arch that does not support it."""


def supported_schemes(gfx: str | None) -> list[str]:
    """Return the schemes selectable on arch ``gfx``; every scheme when the arch is unknown."""
    if not gfx or peak_rate(gfx, "fp4") is not None:
        return list(SUPPORTED_SCHEMES)
    return [s for s in SUPPORTED_SCHEMES if s not in FP4_SCHEMES]


def validate_scheme(scheme: str | None, gfx: str | None) -> None:
    """Raise if ``scheme`` is unknown or unsupported on arch ``gfx``; an unknown arch constrains nothing."""
    if not scheme or scheme == NO_QUANTIZATION:
        return
    if scheme not in SUPPORTED_SCHEMES:
        raise ValueError(f"unknown quantization scheme {scheme!r}; choose one of {list(SUPPORTED_SCHEMES)}")
    if scheme not in supported_schemes(gfx):
        raise SchemeNotSupportedError(
            f"quantization scheme {scheme!r} needs fp4 matrix support, which {gfx} does not have; "
            f"supported on {gfx}: {supported_schemes(gfx)}"
        )


def resolve_scheme_prompt(scheme: str | None) -> str | None:
    """Map a global scheme, already accepted by :func:`validate_scheme`, to its ``--quantize`` prompt."""
    if not scheme or scheme == NO_QUANTIZATION:
        return None
    return f"Quantization strategy:\nApply {scheme} as the global quantization scheme."


__all__ = [
    "NO_QUANTIZATION",
    "SUPPORTED_SCHEMES",
    "FP4_SCHEMES",
    "QUANT_SCHEME_CHOICES",
    "SchemeNotSupportedError",
    "supported_schemes",
    "validate_scheme",
    "resolve_scheme_prompt",
]
