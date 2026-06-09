"""Standalone CLI for the quantization-agent.

Lets you drive the Quark PTQ skill chain from a natural-language prompt
without going through ``inference_optimizer``. The prompt is fed to the
Claude Agent SDK, which loads ``quantization_agent/SKILL.md`` as the
runtime contract and invokes the Quark skills end-to-end.

Example::

    python -m quantization_agent.cli \\
        --prompt "把 Qwen/Qwen3-0.5B 量化成 fp8 (kv_cache 也 fp8, 排除 lm_head)" \\
        --workspace /scratch/qwen3-0.5b-ws \\
        --quark-root /scratch/kewang/workspace/Quark \\
        --interactive off \\
        --acceptable-eval-gap 0.03 \\
        --max-requantize-attempts 1

Exit codes:
    0   success or partial (model usable; partial means audit/eval gap)
    1   failed (model unusable or MUST-validate violation)
    2   argparse / input validation error
    3   operator-rejected checkpoint (interactive 'n' on retry / eval gap)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Any

from .driver.retry import quantize_via_prompt
from .driver.runner import DEFAULT_MODEL


def _interactive_value(raw: str) -> bool | None:
    raw = raw.strip().lower()
    if raw in ("auto", "", "default"):
        return None
    if raw in ("on", "true", "yes", "1"):
        return True
    if raw in ("off", "false", "no", "0"):
        return False
    raise argparse.ArgumentTypeError(
        f"--interactive expects auto / on / off (got {raw!r})"
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="quantization_agent",
        description="Drive the AMD Quark PTQ skill chain from a natural-language prompt.",
    )
    p.add_argument(
        "--prompt",
        required=True,
        help="Natural-language description of what to quantize.",
    )
    p.add_argument(
        "--workspace",
        required=True,
        help="Per-run scratch dir for session_context.json, manifest, reports, eval_report.json.",
    )
    p.add_argument(
        "--quark-root",
        default=None,
        help="Quark repo root (defaults to $QUARK_ROOT).",
    )
    p.add_argument(
        "--interactive",
        type=_interactive_value,
        default=None,
        metavar="auto|on|off",
        help="Checkpoint relay mode. 'auto' uses tty detection (default).",
    )
    p.add_argument(
        "--acceptable-eval-gap",
        type=float,
        default=None,
        metavar="FLOAT",
        help="Max relative quality gap (e.g. 0.03 = 3%%). "
        "Falls back to <workspace>/eval_gap_threshold.txt or 0.03 if unset.",
    )
    p.add_argument(
        "--max-requantize-attempts",
        type=int,
        default=1,
        metavar="N",
        help="Upper bound on Python-driven retries for Ask-class outcomes "
        "(#3/#6/#16/#26) and #30. Default 1.",
    )
    p.add_argument(
        "--model-id",
        default=None,
        help=f"Override the Claude model id used by the SDK (default {DEFAULT_MODEL}).",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Stream SDK output lines to stderr.",
    )
    return p.parse_args(argv)


async def _run(args: argparse.Namespace) -> int:
    def log(line: str) -> None:
        if args.verbose:
            print(line, file=sys.stderr, flush=True)

    result = await quantize_via_prompt(
        args.prompt,
        workspace=args.workspace,
        quark_root=args.quark_root,
        interactive=args.interactive,
        acceptable_eval_gap=args.acceptable_eval_gap,
        max_requantize_attempts=args.max_requantize_attempts,
        model=args.model_id,
        log=log,
    )

    summary: dict[str, Any] = {
        "status": result.status,
        "quantized_model_dir": (
            str(result.quantized_model_dir) if result.quantized_model_dir else None
        ),
        "assessment": result.assessment.to_dict(),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))

    if result.status == "failed":
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
