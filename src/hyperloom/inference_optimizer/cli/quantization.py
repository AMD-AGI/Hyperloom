# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CLI entry — ``optimize`` subcommand wiring Claude+Codex backends, executors, objective, and Coordinator.run().

Env vars consumed: MODEL_PATH, OPENAI_BASE_URL / ANTHROPIC_BASE_URL +
OPENAI_API_KEY / ANTHROPIC_API_KEY, ROCR_VISIBLE_DEVICES,
CLAUDE_MODEL, CODEX_MODEL, USER_DATA_PATH.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

log = logging.getLogger(__name__)


def _quantization_enabled_via_env() -> bool:
    """Return ``True`` iff the deterministic quantization master switch is on.

    Quantization is gated on ``$HYPERLOOM_QUANTIZE_ENABLED`` (truthy = ``1`` /
    ``true`` / ``yes`` / ``on``, case-insensitive). This makes the on/off
    decision a deterministic, frontend-settable env flag rather than something
    an LLM agent infers from natural language. Anything else — including unset —
    disables quantization.

    Returns:
        ``True`` when the env var is set to a recognized truthy value.
    """
    return os.environ.get("HYPERLOOM_QUANTIZE_ENABLED", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


async def _run_quantization_prelude(args: argparse.Namespace) -> None:
    """Run the quantization-agent once before the optimization loop.

    No-op unless ``--quantize "<prompt>"`` was passed. When set, this drives
    AMD Quark PTQ from the prompt via the ``quantization_request_handlers``
    adapter, then rewrites ``args.model`` (+ ``$MODEL_PATH``) to the exported
    quantized model so every downstream phase (baseline / profile / sweep /
    kernel) optimizes the quantized model instead of the source.

    Contract:
      * Reached only on the fresh-launch path; a resumed session takes its
        model from the manifest and never re-quantizes.
      * On a failed/blocked quantization the process exits with code 3 —
        we must not silently fall through and optimize the un-quantized
        source model when the user explicitly asked for quantization.
      * On a scheme/GPU mismatch (e.g. an MI355X-only scheme on an mi300x
        target), the structured ``--quantize-scheme`` path reports the error
        and *skips* quantization, then continues optimizing the un-quantized
        model. The mismatch is a config error caught before any Quark work
        runs, not a mid-run failure, so the run proceeds rather than aborting.
        The skip is made **detectable** so a launcher / UI never mistakes the
        run for quantized: a ``QUANTIZATION_SKIPPED:`` marker line on stdout
        plus the ``$HYPERLOOM_QUANTIZATION_SKIPPED`` env var (set to the reason).

    Args:
        args: Parsed CLI arguments; reads ``quantize`` / ``quantize_scheme`` /
            ``gpu_type`` and rewrites ``args.model`` in place to the exported
            quantized model path on success.
    """
    # Free-text --quantize wins; otherwise resolve the structured
    # --quantize-scheme enum (the UI/backend path) to a prompt.
    prompt = getattr(args, "quantize", None)
    if not prompt:
        from hyperloom.orchestrator.phases.quantization_schemes import (
            SchemeNotSupportedError,
            resolve_scheme_prompt,
            validate_scheme,
        )

        scheme = getattr(args, "quantize_scheme", None)
        # Constrain the scheme by the target GPU via the --gpu-type / $GPU_TYPE
        # hint (empty => no enforcement).
        gpu_hint = (getattr(args, "gpu_type", None) or os.environ.get("GPU_TYPE", "")).strip().lower()
        try:
            validate_scheme(scheme, gpu_hint)
        except SchemeNotSupportedError as exc:
            # Pre-flight config error: skip quantization and continue on the
            # un-quantized model, made machine-detectable via a stdout marker + env var.
            reason = str(exc)
            os.environ["HYPERLOOM_QUANTIZATION_SKIPPED"] = reason
            print(
                f"QUANTIZATION_SKIPPED: {reason}; continuing optimization on the "
                "un-quantized model. Pick a scheme supported by this GPU TYPE "
                "(or change GPU_TYPE) to actually quantize."
            )
            print(f"ERROR: quantization skipped — {reason}", file=sys.stderr)
            return
        prompt = resolve_scheme_prompt(scheme)
    if not prompt:
        return

    # Deterministic master switch: quantization runs ONLY when
    # $HYPERLOOM_QUANTIZE_ENABLED is truthy, regardless of the flags. Absent /
    # false => skip and continue on the un-quantized model (detectable via the
    # QUANTIZATION_SKIPPED marker + $HYPERLOOM_QUANTIZATION_SKIPPED).
    if not _quantization_enabled_via_env():
        reason = "HYPERLOOM_QUANTIZE_ENABLED is not set to a truthy value"
        os.environ["HYPERLOOM_QUANTIZATION_SKIPPED"] = reason
        print(
            f"QUANTIZATION_SKIPPED: {reason}; continuing optimization on the "
            "un-quantized model. Set HYPERLOOM_QUANTIZE_ENABLED=1 to quantize."
        )
        return

    from ..session.paths import workspace_root

    source_model = str(args.model)
    workspace = workspace_root() / "quantization" / Path(source_model).name
    workspace.mkdir(parents=True, exist_ok=True)

    # Adapter lives in the orchestrator package; lazy-import so the CLI imports
    # cleanly without the quantization deps. Await the async form directly since
    # _run_optimize already runs under asyncio.run.
    from hyperloom.orchestrator.phases.quantization_request_handlers import (
        run_quantization_prelude_async,
    )

    quantized_model_dir = await run_quantization_prelude_async(
        prompt=prompt,
        source_model=source_model,
        workspace=workspace,
        provider=(
            str(getattr(args, "quantize_provider", None) or "").strip().lower()
            or os.environ.get("HYPERLOOM_QUANT_PROVIDER", "").strip().lower()
            or "claude"
        ),
    )

    args.model = Path(quantized_model_dir)
    os.environ["MODEL_PATH"] = str(quantized_model_dir)
    # Preserve the SOURCE model identity for session naming / display: the export
    # dir basename is always "quantized", so pin "<source>-quantized" to keep
    # the real model name in the session dir, SharedState, and manifest.
    args.model_display_name = f"{Path(source_model).name}-quantized"
    print(f"Quantization prelude: model -> {quantized_model_dir}")
