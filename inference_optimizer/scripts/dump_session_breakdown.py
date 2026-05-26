#!/usr/bin/env python3
"""Dump ``session_breakdown.json`` for one hyperloom session directory.

This is the offline / historical / debugging entrypoint. The same
builder is used by:

* Coordinator action ``session_breakdown`` (live, agent-driven)
* ``cli.py`` finally block (live, end-of-session safety net)
* This script (offline / batch / WekaFS sessions)

Examples
--------

::

    # Live session in this sandbox (USER_DATA_PATH or /workspace/hyperloom)
    python -m inference_optimizer.scripts.dump_session_breakdown

    # Historical session on WekaFS
    python -m inference_optimizer.scripts.dump_session_breakdown \\
        --session-dir /wekafs/users/zgong/inference_optimizer-sessions/<sid>

    # Override output path (don't touch session_dir)
    python -m inference_optimizer.scripts.dump_session_breakdown \\
        --session-dir <SD> --output /tmp/breakdown-<sid>.json

    # Bulk historical
    for d in /wekafs/users/*/inference_optimizer-sessions/*; do
        [ -d "$d" ] || continue
        python -m inference_optimizer.scripts.dump_session_breakdown \\
            --session-dir "$d" > /dev/null
    done
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from ..breakdown import BREAKDOWN_FILENAME, build, write_breakdown_json
from ..paths import session_dir as default_session_dir


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dump_session_breakdown",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--session-dir",
        type=Path,
        default=None,
        help=(
            "Hyperloom session directory. Defaults to "
            "$USER_DATA_PATH (set by production launchers) or "
            "/workspace/hyperloom."
        ),
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help=(
            f"Override output file path. Defaults to "
            f"<session_dir>/{BREAKDOWN_FILENAME}."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build the breakdown dict and print summary stats; do not write.",
    )
    parser.add_argument(
        "--print",
        dest="print_json",
        action="store_true",
        help="Also print the full JSON to stdout (useful for piping).",
    )
    parser.add_argument(
        "--include-transcripts",
        action="store_true",
        help=(
            "Inline specialist transcript bodies under "
            "specialist_runs[i].transcripts[j].body. Mirrors "
            "INFERENCE_OPTIMIZER_BREAKDOWN_INCLUDE_TRANSCRIPTS=1."
        ),
    )
    parser.add_argument(
        "--verbose", "-v", action="count", default=0,
        help="-v INFO, -vv DEBUG.",
    )
    return parser


def _setup_logging(verbose: int) -> None:
    level = logging.WARNING
    if verbose == 1:
        level = logging.INFO
    elif verbose >= 2:
        level = logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _summary_line(breakdown: dict) -> str:
    sess = breakdown.get("session") or {}
    final = breakdown.get("final") or {}
    cap = breakdown.get("capability_summary") or {}
    geak_n = len(breakdown.get("geak_invocations") or [])
    oob_n = len(breakdown.get("oob_invocations") or [])
    lifecycle = breakdown.get("kernel_lifecycle") or {}
    sweep = breakdown.get("sweep") or {}
    warnings = breakdown.get("warnings") or []
    dj_n = len(breakdown.get("decision_journal") or [])
    kp_n = len(breakdown.get("kernel_profiling") or [])
    return (
        f"session_id={sess.get('session_id', '?')}  "
        f"claw_session_id={sess.get('claw_session_id') or '(none)'}  "
        f"stop_reason={sess.get('stop_reason') or '?'}  "
        f"gain_validated={final.get('cumulative_gain_pct_validated', 0.0):.2f}%  "
        f"geak={geak_n}  oob={oob_n}  "
        f"detected={len(lifecycle.get('detected') or [])}  "
        f"adopted={len(lifecycle.get('adopted') or [])}  "
        f"sweep={len(sweep.get('all_variants') or [])}  "
        f"decision_journal={dj_n}  kernel_profiling={kp_n}  "
        f"warnings={len(warnings)}"
    )


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    _setup_logging(args.verbose)
    log = logging.getLogger("dump_session_breakdown")

    sd = args.session_dir if args.session_dir else default_session_dir()
    sd = Path(sd).resolve()
    if not sd.exists():
        print(f"ERROR: session-dir does not exist: {sd}", file=sys.stderr)
        return 2

    if args.dry_run:
        breakdown = build(sd, include_transcripts=args.include_transcripts)
        print(_summary_line(breakdown))
        if args.print_json:
            print(json.dumps(breakdown, indent=2, sort_keys=True))
        return 0

    try:
        out_path = write_breakdown_json(
            sd,
            output_path=args.output,
            include_transcripts=args.include_transcripts,
        )
    except Exception as exc:  # noqa: BLE001
        log.exception("write_breakdown_json failed")
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    breakdown = build(sd, include_transcripts=args.include_transcripts)
    print(f"Wrote {out_path}")
    print(_summary_line(breakdown))
    if args.print_json:
        print(json.dumps(breakdown, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
