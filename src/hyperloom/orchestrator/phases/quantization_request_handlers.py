"""Adapter: drive the quantization-agent from hyperloom.inference_optimizer."""

from __future__ import annotations

import sys
from pathlib import Path


async def run_quantization_prelude_async(
    *,
    prompt: str,
    source_model: str,
    workspace: Path,
) -> str:
    """Quantize ``source_model`` per ``prompt``; return the exported dir."""
    # Import lazily so this module loads without the quantization runtime deps.
    from hyperloom.agents.quantization import quantize_via_prompt

    workspace = Path(workspace)
    export_dir = workspace / "quantized"

    # Fold the source model + export dir into the prompt so the user's --quantize text can be just the scheme.
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
