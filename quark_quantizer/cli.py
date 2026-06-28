# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Standalone CLI for quark_quantizer.

The trigger is an explicit parameter (``--enabled``), mirroring the
frontend->backend toggle: nothing runs unless the caller opts in. The natural
language goes in ``--prompt`` and is forwarded to the wrapped
``quantization_agent`` (which turns it into the Quark CLI and executes it).

Example::

    python -m quark_quantizer.cli \\
        --enabled \\
        --prompt "fp8 global scheme, fp8 kv_cache, exclude lm_head" \\
        --workspace /scratch/run-1/wks

Exit codes:
    0   success / partial / skipped (disabled)
    1   failed (no usable quantized model)
    2   argparse / input validation error
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Any

from .runner import quantize


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Build and parse the CLI arguments.

    Args:
        argv: Argument list; defaults to ``sys.argv`` when ``None``.

    Returns:
        The parsed namespace.
    """
    p = argparse.ArgumentParser(
        prog="quark_quantizer",
        description="Parameter-gated shell over quantization_agent (Quark PTQ).",
    )
    # The trigger is a parameter, not an agent decision.
    group = p.add_mutually_exclusive_group()
    group.add_argument(
        "--enabled",
        dest="enabled",
        action="store_true",
        help="Run quantization. Without this flag the module is a no-op (skipped).",
    )
    group.add_argument(
        "--disabled",
        dest="enabled",
        action="store_false",
        help="Explicitly disable (default).",
    )
    p.set_defaults(enabled=False)

    p.add_argument("--prompt", default="", help="Natural-language quantization request.")
    p.add_argument("--workspace", required=True, help="Scratch dir for the wrapped agent's artifacts.")
    p.add_argument("--quark-root", default=None, help="Quark checkout root (else $QUARK_ROOT).")
    p.add_argument("--acceptable-eval-gap", type=float, default=None, help="Max relative eval gap.")
    p.add_argument("--model-id", default=None, help="Override the wrapped agent's model id.")
    p.add_argument("--verbose", action="store_true", help="Stream logs to stderr.")
    return p.parse_args(argv)


async def _run(args: argparse.Namespace) -> int:
    """Run one request and print a JSON summary.

    Args:
        args: Parsed CLI arguments.

    Returns:
        The process exit code.
    """

    def log(line: str) -> None:
        if args.verbose:
            print(line, file=sys.stderr, flush=True)

    result = await quantize(
        args.prompt,
        enabled=args.enabled,
        workspace=args.workspace,
        quark_root=args.quark_root,
        acceptable_eval_gap=args.acceptable_eval_gap,
        model=args.model_id,
        log=log,
    )

    summary: dict[str, Any] = {
        "status": result.status,
        "output_dir": result.output_dir,
        "eval_gap": result.eval_gap,
        "final": result.final,
        "error": result.error,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))

    return 1 if result.status == "failed" else 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Argument list; defaults to ``sys.argv`` when ``None``.

    Returns:
        The process exit code.
    """
    args = _parse_args(argv)
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
