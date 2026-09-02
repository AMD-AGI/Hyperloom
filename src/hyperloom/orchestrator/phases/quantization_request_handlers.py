"""Adapter: drive the quantization-agent from hyperloom.inference_optimizer.

Thin shim between ``cli._run_quantization_prelude`` and the
``hyperloom.agents.quantization`` package. It builds an effective prompt
(source model path + export dir + the user's ``--quantize`` text), runs
``quantize_via_prompt`` once, and maps its ``QuantSkillRunResult.status`` to a
concrete decision:

  * ``success``                    -> return ``quantized_model_dir``
  * ``partial`` (model usable)     -> warn, then return ``quantized_model_dir``
  * ``partial`` (no usable model)  -> ``SystemExit(3)``
  * ``failed``                     -> ``SystemExit(3)``

A quantization failure is a hard stop for the whole run: this adapter never
silently falls through to optimize the un-quantized source model.
"""

from __future__ import annotations

import sys
from pathlib import Path


async def run_quantization_prelude_async(
    *,
    prompt: str,
    source_model: str,
    workspace: Path,
    provider: str = "claude",
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
    # Import lazily so this module loads without the quantization runtime deps.
    from hyperloom.agents.quantization import quantize_via_prompt

    workspace = Path(workspace)
    export_dir = workspace / "quantized"

    # Fold the source model + export dir into the prompt so the user's
    # --quantize text can be just the scheme.
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
        provider=provider,
    )

    final = result.assessment.final
    qdir = result.quantized_model_dir

    if result.status == "success":
        print(f"Quantization: success (final={final}, eval_gap={result.assessment.eval_gap}) -> {qdir}")
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


__all__ = ["run_quantization_prelude_async"]
