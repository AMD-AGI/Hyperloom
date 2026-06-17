"""Adapter: drive the quantization-agent from inference_optimizer.

Thin shim between ``cli._run_quantization_prelude`` and the standalone
``quantization_agent`` package. It builds an effective prompt (source model
path + export dir + the user's ``--quantize`` text), runs
``quantize_via_prompt`` once, and maps its ``QuantSkillRunResult.status`` to a
concrete decision:

  * ``success``                    -> return ``quantized_model_dir``
  * ``partial`` (model usable)     -> warn, then return ``quantized_model_dir``
  * ``partial`` (no usable model)  -> ``SystemExit(3)``
  * ``failed``                     -> ``SystemExit(3)``

When the user explicitly asked for quantization we must never silently fall
through and optimize the un-quantized source model — a quantization failure
is a hard stop for the whole run.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path


async def run_quantization_prelude_async(
    *,
    prompt: str,
    source_model: str,
    workspace: Path,
) -> str:
    """Quantize ``source_model`` per ``prompt``; return the exported dir.

    Awaits the async ``quantize_via_prompt`` directly (the caller already
    runs inside ``asyncio.run``). Raises ``SystemExit(3)`` when no usable
    quantized model was produced.

    Args:
        prompt: User-provided quantization instructions (e.g. scheme text).
        source_model: Path to the model to quantize.
        workspace: Working directory; the quantized model is exported under it.

    Returns:
        The path to the exported quantized model directory.

    Raises:
        SystemExit: If quantization failed or produced no usable model.
    """
    # quantization_agent is a top-level package (sibling of inference_optimizer);
    # imported lazily so this module loads even where its deps are absent.
    from quantization_agent import quantize_via_prompt

    workspace = Path(workspace)
    export_dir = workspace / "quantized"

    # The agent is prompt-driven: fold the source model + export dir into the
    # prompt so the user's --quantize text can be just the scheme
    # (e.g. "fp8 with fp8 kv_cache, exclude lm_head").
    effective_prompt = (
        f"Quantize the model at {source_model}. "
        f"Export the HuggingFace-format quantized model to {export_dir}. "
        f"Run the COMPLETE PTQ phase chain end-to-end in this one session "
        f"(intake -> plan -> manifest -> exec -> export -> validate -> eval); "
        f"do not stop early or hand back to a parent agent. "
        f"interactive=off: accept all CRITICAL STOP defaults. "
        f"{prompt}"
    )

    result = await quantize_via_prompt(
        effective_prompt,
        workspace=workspace,
        interactive=False,
    )

    final = result.assessment.final
    qdir = result.quantized_model_dir

    if result.status == "success":
        print(
            f"Quantization: success (final={final}, "
            f"eval_gap={result.assessment.eval_gap}) -> {qdir}"
        )
        return str(qdir)

    if result.status == "partial" and qdir is not None:
        print(
            f"Quantization: PARTIAL (final={final}); quantized model is loadable "
            f"so continuing, but audit/eval was incomplete. Review {workspace}.",
            file=sys.stderr,
        )
        return str(qdir)

    print(
        f"ERROR: quantization {result.status} (final={final}). Refusing to "
        f"optimize the un-quantized source model. See {workspace} for details.",
        file=sys.stderr,
    )
    raise SystemExit(3)


def run_quantization_prelude(
    *,
    prompt: str,
    source_model: str,
    workspace: Path,
) -> str:
    """Sync wrapper for non-async callers / tests.

    DO NOT call from within a running event loop — the cli prelude awaits
    :func:`run_quantization_prelude_async` directly because ``_run_optimize``
    already runs under ``asyncio.run``.

    Args:
        prompt: User-provided quantization instructions (e.g. scheme text).
        source_model: Path to the model to quantize.
        workspace: Working directory; the quantized model is exported under it.

    Returns:
        The path to the exported quantized model directory.
    """
    return asyncio.run(
        run_quantization_prelude_async(
            prompt=prompt, source_model=source_model, workspace=workspace
        )
    )


__all__ = ["run_quantization_prelude", "run_quantization_prelude_async"]
