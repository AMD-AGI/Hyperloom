# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CLI entry — ``optimize`` subcommand wiring Claude+Codex backends, executors, objective, and Coordinator.run()."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

log = logging.getLogger(__name__)


def _quantization_enabled_via_env() -> bool:
    """Return ``True`` iff the deterministic quantization master switch is on."""
    return os.environ.get("HYPERLOOM_QUANTIZE_ENABLED", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


async def _run_quantization_prelude(args: argparse.Namespace) -> None:
    """Run the quantization-agent once before the optimization loop."""
    # Free-text --quantize wins; otherwise resolve the structured --quantize-scheme enum (the UI/backend path) to a
    # prompt.
    prompt = getattr(args, "quantize", None)
    if not prompt:
        from hyperloom.orchestrator.phases.quantization_schemes import (
            SchemeNotSupportedError,
            resolve_scheme_prompt,
            validate_scheme,
        )

        scheme = getattr(args, "quantize_scheme", None)
        # Constrain the scheme by the target GPU via the --gpu-type / $GPU_TYPE hint (empty => no enforcement).
        gpu_hint = (getattr(args, "gpu_type", None) or os.environ.get("GPU_TYPE", "")).strip().lower()
        try:
            validate_scheme(scheme, gpu_hint)
        except SchemeNotSupportedError as exc:
            # Pre-flight config error: skip quantization and continue on the un-quantized model, made
            # machine-detectable via a stdout marker + env var.
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

    # Deterministic master switch: quantization runs ONLY when $HYPERLOOM_QUANTIZE_ENABLED is truthy, regardless of
    # the flags.
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

    # Adapter lives in the orchestrator package; lazy-import so the CLI imports cleanly without the quantization deps.
    from hyperloom.orchestrator.phases.quantization_request_handlers import (
        run_quantization_prelude_async,
    )

    quantized_model_dir = await run_quantization_prelude_async(
        prompt=prompt,
        source_model=source_model,
        workspace=workspace,
    )

    args.model = Path(quantized_model_dir)
    os.environ["MODEL_PATH"] = str(quantized_model_dir)
    # Preserve the SOURCE model identity for session naming / display: the export dir basename is always "quantized",
    # so pin "<source>-quantized" to keep the real model name in the session dir, SharedState, and manifest.
    args.model_display_name = f"{Path(source_model).name}-quantized"
    print(f"Quantization prelude: model -> {quantized_model_dir}")
