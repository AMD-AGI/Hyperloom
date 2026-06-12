#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Render a markdown session report from a ``session_breakdown.json``.

Usage::

    # Deterministic only (no LLM):
    python -m inference_optimizer.scripts.dump_session_report \\
        --input  /wekafs/.../session_breakdown.json \\
        --output /wekafs/.../session_report.md

    # With LLM-polished prose (OpenAI-compatible endpoint):
    HYPERLOOM_REPORT_LLM_BACKEND=openai \\
    OPENAI_BASE_URL=https://core42.primus-safe.amd.com/api/v1/llm-proxy/v1 \\
    OPENAI_API_KEY=... \\
    python -m inference_optimizer.scripts.dump_session_report \\
        --input  /wekafs/.../session_breakdown.json \\
        --output /wekafs/.../session_report.md

When --output is omitted the report is written to
``<session_dir>/session_report.md`` next to the input file. The LLM
user prompt and raw response (when used) are persisted alongside as
``session_report_prompt.json`` / ``session_report_llm_raw.txt`` so
hallucinations can be audited after the fact.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from inference_optimizer.breakdown.reporters import render_session_report
from inference_optimizer.breakdown.reporters.llm_client import build_client_from_env

log = logging.getLogger("dump_session_report")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the session-report CLI.

    Args:
        argv (list[str] | None): Argument vector to parse; defaults to
            ``sys.argv`` when ``None``.

    Returns:
        argparse.Namespace: Parsed arguments with ``input``, ``output``,
        ``no_llm``, and ``debug_dump`` attributes.
    """
    p = argparse.ArgumentParser(
        description="Render a Hyperloom session_breakdown.json to markdown.",
    )
    p.add_argument("--input", "-i", required=True, type=Path,
                   help="Path to session_breakdown.json")
    p.add_argument("--output", "-o", type=Path, default=None,
                   help="Output markdown path (defaults to <session_dir>/session_report.md)")
    p.add_argument("--no-llm", action="store_true",
                   help="Skip LLM narrative pass even if env vars are configured")
    p.add_argument("--debug-dump", action="store_true",
                   help="Also write LLM prompt + raw response next to the report")
    return p.parse_args(argv)


def _resolve_output(input_path: Path, requested: Path | None) -> Path:
    """Resolve the markdown output path for the report.

    Args:
        input_path (Path): Path to the input ``session_breakdown.json``.
        requested (Path | None): Explicitly requested output path, if any.

    Returns:
        Path: ``requested`` when provided, otherwise ``session_report.md`` next
        to the input file.
    """
    if requested is not None:
        return requested
    return input_path.parent / "session_report.md"


def main(argv: list[str] | None = None) -> int:
    """Render a session breakdown to a markdown report.

    Loads the breakdown JSON, optionally builds an LLM client from the
    environment, renders the report, and writes it (plus optional debug dumps).

    Args:
        argv (list[str] | None): Argument vector to parse; defaults to
            ``sys.argv`` when ``None``.

    Returns:
        int: ``0`` on success, or ``2`` when the input is missing/unparseable.
    """
    logging.basicConfig(
        level=os.environ.get("HYPERLOOM_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _parse_args(argv)

    if not args.input.exists():
        log.error("input not found: %s", args.input)
        return 2
    try:
        breakdown = json.loads(args.input.read_text())
    except Exception as exc:  # noqa: BLE001
        log.error("failed to parse %s: %s", args.input, exc)
        return 2

    llm = None if args.no_llm else build_client_from_env()
    if llm is not None:
        log.info("LLM backend active: %s", type(llm).__name__)
    else:
        log.info("LLM backend disabled — deterministic-only report")

    result = render_session_report(breakdown, llm_client=llm)
    out_path = _resolve_output(args.input, args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(result.markdown)
    log.info("wrote %s (%d bytes, used_llm=%s)",
             out_path, len(result.markdown), result.used_llm)

    if args.debug_dump:
        (out_path.parent / "session_report_prompt.json").write_text(result.llm_user_prompt)
        (out_path.parent / "session_report_llm_raw.txt").write_text(result.llm_raw_response or "")
        log.info("wrote debug dumps next to %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
