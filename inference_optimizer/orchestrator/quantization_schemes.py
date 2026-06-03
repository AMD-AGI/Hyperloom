"""Curated quantization scheme -> ``--quantize`` prompt mapping.

The frontend exposes a small "quantization precision" dropdown; the backend
sends the chosen scheme *name* (an enum) here and gets the natural-language
prompt the quantization-agent expects. Free-text ``--quantize`` stays
available for power users — this is the structured path for UI/backend-driven
requests, so the enum -> prompt translation lives in exactly one place.

Keep the curated set small and serving-validated. The full Quark scheme list
(``LLMTemplate.get_supported_schemes()``) is much larger (~23 entries); only
add more here after validating load + accuracy on the target serving stack.
"""

from __future__ import annotations


# Sentinel for "do not quantize" (the dropdown default).
NO_QUANTIZATION = "none"

# scheme enum -> quantize prompt. ``None`` means no quantization.
# The prompt carries the scheme only; the source model path + export dir are
# folded in by quantization_request_handlers, so callers never repeat them.
QUANT_SCHEME_PROMPTS: dict[str, str | None] = {
    NO_QUANTIZATION: None,
    "fp8":          "fp8 global scheme, fp8 kv_cache, exclude lm_head",
    "int8":         "int8 weight and activation quantization, exclude lm_head",
    "int4_wo_128":  "int4 weight-only quantization with group_size 128, exclude lm_head",
    "mxfp4":        "mxfp4 global scheme, exclude lm_head",
    "mxfp4_fp8":    "mxfp4 weights with fp8 activation, exclude lm_head",
}

# argparse ``choices=`` for the structured CLI flag.
QUANT_SCHEME_CHOICES: list[str] = list(QUANT_SCHEME_PROMPTS.keys())


def resolve_scheme_prompt(scheme: str | None) -> str | None:
    """Map a curated scheme enum to its ``--quantize`` prompt.

    Returns ``None`` when ``scheme`` is falsy or ``"none"`` (= no
    quantization). Raises ``ValueError`` for an unknown scheme so a typo'd
    enum fails loudly instead of silently skipping quantization.
    """
    if not scheme or scheme == NO_QUANTIZATION:
        return None
    try:
        return QUANT_SCHEME_PROMPTS[scheme]
    except KeyError:
        raise ValueError(
            f"unknown quantization scheme {scheme!r}; "
            f"choose one of {QUANT_SCHEME_CHOICES}"
        ) from None


__all__ = [
    "NO_QUANTIZATION",
    "QUANT_SCHEME_PROMPTS",
    "QUANT_SCHEME_CHOICES",
    "resolve_scheme_prompt",
]
