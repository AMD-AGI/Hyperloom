"""Adapter: drive quantization from inference_optimizer via the quark_quantizer shell.

Thin shim between ``cli._run_quantization_prelude`` and the ``quark_quantizer``
shell (which itself wraps the original ``quantization_agent``). The prelude
position and order are unchanged (it still runs ONCE before the optimization
loop, then the caller rewrites ``--model`` to the exported model). The only
behavioral change vs. calling ``quantization_agent`` directly is that the trigger
is now a parameter (``enabled``) rather than an implicit code path — here we are
past the ``--quantize`` / ``--quantize-scheme`` gate, so we enable it.

It builds an effective prompt (source model path + export dir + the user's
``--quantize`` text), runs the shell once, and maps its status to a decision:

  * ``success``                    -> return ``output_dir``
  * ``partial`` (model usable)     -> warn, then return ``output_dir``
  * ``partial`` (no usable model)  -> ``SystemExit(3)``
  * ``failed`` / other             -> ``SystemExit(3)``

When the user explicitly asked for quantization we must never silently fall
through and optimize the un-quantized source model — a quantization failure is
a hard stop for the whole run.
"""

from __future__ import annotations

import sys
from pathlib import Path


async def run_quantization_prelude_async(
    *,
    prompt: str,
    source_model: str,
    workspace: Path,
) -> str:
    """Quantize ``source_model`` per ``prompt``; return the exported dir.

    Awaits the async ``quark_quantizer.quantize`` directly (the caller already
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
    # quark_quantizer is a top-level package (sibling of inference_optimizer);
    # imported lazily so this module loads even where its deps are absent.
    from quark_quantizer import quantize

    workspace = Path(workspace)
    export_dir = workspace / "quantized"

    # Fold the source model + export dir into the prompt so the user's
    # --quantize text can be just the scheme (e.g. "fp8 with fp8 kv_cache,
    # exclude lm_head"). The wrapped agent's LLM turns this into the Quark CLI.
    effective_prompt = (
        f"Quantize the model at {source_model}. "
        f"Export the HuggingFace-format quantized model to {export_dir}. "
        f"Run the COMPLETE PTQ phase chain end-to-end in this one session "
        f"(intake -> plan -> manifest -> exec -> export -> validate -> eval); "
        f"do not stop early or hand back to a parent agent. "
        f"interactive=off: accept all CRITICAL STOP defaults. "
        f"{prompt}"
    )

    # Reaching the prelude already means quantization was explicitly requested
    # (the --quantize / --quantize-scheme gate in cli), so enable it here.
    result = await quantize(effective_prompt, enabled=True, workspace=workspace)

    if result.status == "success":
        print(f"Quantization: success (final={result.final}, eval_gap={result.eval_gap}) -> {result.output_dir}")
        return str(result.output_dir)

    if result.status == "partial" and result.output_dir is not None:
        print(
            f"Quantization: PARTIAL (final={result.final}); quantized model is loadable "
            f"so continuing, but audit/eval was incomplete. Review {workspace}.",
            file=sys.stderr,
        )
        return str(result.output_dir)

    print(
        f"ERROR: quantization {result.status} (final={result.final}). Refusing to "
        f"optimize the un-quantized source model. See {workspace} for details.",
        file=sys.stderr,
    )
    raise SystemExit(3)


__all__ = ["run_quantization_prelude_async"]
