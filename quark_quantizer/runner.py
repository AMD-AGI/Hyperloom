# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""The shell around the original ``quantization_agent`` implementation.

This module adds exactly one thing on top of the existing (Quark-driving)
``quantization_agent``: a **parameter-based trigger**. The caller (a backend,
driven by a frontend toggle) passes ``enabled`` to decide whether quantization
runs at all — the outermost orchestration LLM has no natural-language path into
this module, so it cannot accidentally activate quantization.

The shell does NOT reimplement prompt->CLI->execute. When enabled it forwards
the natural-language prompt to ``quantization_agent.quantize_via_prompt``, whose
in-agent LLM turns the prompt into the Quark CLI and runs it. The shell only
gates the call and normalizes the result.

Conceptually two inputs go in: the ``enabled`` switch and the NL ``prompt``.
``workspace`` (and a few optional knobs) are plumbing the wrapped agent needs.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class QuarkRunResult:
    """Normalized outcome of a (possibly skipped) quantization request.

    Attributes:
        status: ``"skipped"`` (disabled), or the wrapped agent's status
            (``"success"`` / ``"partial"`` / ``"failed"``).
        output_dir: Exported quantized model dir, or ``None``.
        eval_gap: Relative eval gap from the final attempt, if any.
        final: The wrapped agent's final outcome id, if any.
        error: Non-empty on shell-level failures.
    """

    status: str
    output_dir: str | None = None
    eval_gap: float | None = None
    final: str | None = None
    error: str = ""


async def quantize(
    prompt: str,
    *,
    enabled: bool = False,
    workspace: str | Path,
    quark_root: str | Path | None = None,
    interactive: bool | None = False,
    acceptable_eval_gap: float | None = None,
    max_requantize_attempts: int = 1,
    model: str | None = None,
    quantize_via_prompt: Callable[..., Any] | None = None,
    log: Callable[[str], None] | None = None,
) -> QuarkRunResult:
    """Quantize from a natural-language prompt, gated on ``enabled``.

    Args:
        prompt: Natural-language quantization request. Forwarded verbatim to the
            wrapped ``quantization_agent``, whose LLM turns it into the Quark CLI.
        enabled: Master switch (default ``False`` — the module is a no-op unless
            the caller opts in). When ``False`` the wrapped agent is never run.
        workspace: Directory the wrapped agent writes artifacts into.
        quark_root: Quark checkout root; falls back to ``$QUARK_ROOT``.
        interactive: Forwarded interactivity flag (default ``False`` = batch).
        acceptable_eval_gap: Max tolerated relative accuracy gap.
        max_requantize_attempts: Cap on requantize retries.
        model: Optional model id passed to the wrapped agent.
        quantize_via_prompt: Override for the wrapped entry (testing seam).
        log: Optional line-logging callback.

    Returns:
        The normalized :class:`QuarkRunResult`.
    """
    if not enabled:
        if log:
            log("[quark_quantizer] disabled (enabled=False); skipping quantization")
        return QuarkRunResult(status="skipped")

    qvp = quantize_via_prompt
    if qvp is None:
        # quantization_agent is a top-level package (sibling of inference_optimizer);
        # imported lazily so the shell loads even where its deps are absent.
        from quantization_agent import quantize_via_prompt as qvp_real

        qvp = qvp_real

    result = await qvp(
        prompt,
        workspace=workspace,
        quark_root=quark_root,
        interactive=interactive,
        acceptable_eval_gap=acceptable_eval_gap,
        max_requantize_attempts=max_requantize_attempts,
        model=model,
        log=log,
    )

    assessment = getattr(result, "assessment", None)
    final = getattr(assessment, "final", None)
    eval_gap = getattr(assessment, "eval_gap", None)
    qdir = getattr(result, "quantized_model_dir", None)
    return QuarkRunResult(
        status=result.status,
        output_dir=str(qdir) if qdir else None,
        eval_gap=eval_gap,
        final=str(final) if final is not None else None,
    )


def quantize_sync(prompt: str, *, enabled: bool = False, **kwargs: Any) -> QuarkRunResult:
    """Synchronous wrapper around :func:`quantize` for non-async callers.

    Args:
        prompt: Natural-language quantization request.
        enabled: Master switch (see :func:`quantize`).
        **kwargs: Forwarded to :func:`quantize`.

    Returns:
        The :class:`QuarkRunResult`.
    """
    return asyncio.run(quantize(prompt, enabled=enabled, **kwargs))


__all__ = [
    "QuarkRunResult",
    "quantize",
    "quantize_sync",
]
