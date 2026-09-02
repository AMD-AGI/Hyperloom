#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Check what a real session would send the predictor, before spending GPU time.

Usage:
    python src/hyperloom/inference_optimizer/tools/predictor_probe.py SESSION_DIR
    python .../predictor_probe.py SESSION_DIR --out request.json
    python .../predictor_probe.py SESSION_DIR --endpoint http://predictor:8973

Why this exists: both known mismatches between what Hyperloom reports and what
a consumer was trained on are silent. A phase name it has never seen still
renders; an evidence key it does not recognise just drops the sentence. Nothing
raises, nothing logs, and the only symptom is a KEEP rate that looks like a
weak model. This turns that into a number you can read before the first
benchmark cycle.

It reads a persisted ``state.json``, so it costs no GPU time and needs no
running loop. With ``--endpoint`` it also asks the service what it made of the
request; without one it prints the request and the coverage report.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from hyperloom.common.jsonio import read_json
from hyperloom.inference_optimizer.session.session_paths import manifest_path, state_path
from hyperloom.orchestrator.predictor import config as pconfig
from hyperloom.orchestrator.predictor.client import predict
from hyperloom.orchestrator.predictor.payload import build_request

#: Evidence sub-blocks, in the order the consumer renders them.
_BLOCKS = ("roofline", "window", "operators", "hot_kernels")


def _state_view(state: dict[str, Any]) -> SimpleNamespace:
    """Wrap a persisted ``state.json`` so the payload builder can read it.

    ``build_request`` reads attributes because it normally gets the live
    ``SharedState``. A namespace over the parsed JSON gives the same surface
    without importing the dataclass or reconstructing invariants the builder
    does not touch.
    """
    return SimpleNamespace(**state)


def _leaf_coverage(value: Any, prefix: str = "") -> tuple[int, list[str]]:
    """Count leaves and collect the dotted paths whose value is ``None``.

    Nested one level is enough for this body; a present-but-null leaf is the
    interesting case because it renders as an omitted clause.
    """
    total = 0
    nulls: list[str] = []
    if isinstance(value, dict):
        for key, sub in value.items():
            sub_total, sub_nulls = _leaf_coverage(sub, f"{prefix}.{key}" if prefix else str(key))
            total += sub_total
            nulls.extend(sub_nulls)
        return total, nulls
    if isinstance(value, list):
        # A list is one decision: present or not. Its rows are reported by count.
        return (1, [] if value else [prefix])
    total = 1
    if value is None:
        nulls.append(prefix)
    return total, nulls


def _report(request: dict[str, Any]) -> list[str]:
    """Human-readable coverage of the request body."""
    lines: list[str] = []
    ident = request.get("identification") or {}
    workload = request.get("workload") or {}
    phase = request.get("phase") or {}
    perf = request.get("performance") or {}
    evidence = request.get("evidence") or {}

    lines.append(f"model      : {ident.get('model_name')} ({ident.get('model_class')})")
    lines.append(
        f"stack      : {ident.get('framework')} {ident.get('framework_version')} "
        f"on {ident.get('gpu_type')}, tp={ident.get('tp')} ep={ident.get('ep')} "
        f"precision={ident.get('precision')}"
    )
    lines.append(
        f"workload   : isl={workload.get('isl')} osl={workload.get('osl')} "
        f"conc={workload.get('conc')} max_model_len={workload.get('max_model_len')}"
    )
    lines.append(
        f"phase      : {phase.get('phase')} (reason={phase.get('phase_reason')!r}, "
        f"cycle={phase.get('macro_cycle')}, {phase.get('phase_elapsed_seconds')}s in)"
    )
    stack = perf.get("optimization_stack") or []
    lines.append(
        f"performance: baseline={perf.get('baseline_tput')} best={perf.get('current_best_tput')} "
        f"gain={perf.get('cumulative_gain_validated')}% "
        f"keep_threshold={perf.get('keep_threshold_pct')}% stack_depth={len(stack)}"
    )

    lines.append("")
    if not evidence.get("profile_available"):
        lines.append("evidence   : NONE (no profile attached to this decision point)")
        lines.append("             The consumer renders its architecture-only paragraph.")
    else:
        present = [b for b in _BLOCKS if evidence.get(b) is not None]
        absent = [b for b in _BLOCKS if evidence.get(b) is None]
        lines.append(f"evidence   : {len(present)}/4 blocks -> {', '.join(present) or 'none'}")
        if absent:
            lines.append(f"             absent: {', '.join(absent)}")
        kernels = evidence.get("hot_kernels") or []
        if kernels:
            framed = sum(1 for k in kernels if k.get("source_line"))
            named = sum(1 for k in kernels if k.get("name"))
            with_args = sum(1 for k in kernels if k.get("args"))
            lines.append(
                f"kernels    : {len(kernels)} rows, {named} named, {with_args} with args, "
                f"{framed} with a source line"
            )
            if not framed:
                lines.append(
                    "             No source frames: a patch answer can name a file but not a "
                    "location. Check kernel_source_resolution.json in the analysis run dir."
                )
        window = evidence.get("window") or {}
        if window and not window.get("total_gpu_time_ms"):
            lines.append(
                "             WARNING: window block without a duration. The consumer would "
                "render a sentence its corpus never contains."
            )

    total, nulls = _leaf_coverage(request)
    lines.append("")
    lines.append(f"fields     : {total - len(nulls)}/{total} populated")
    if nulls:
        lines.append(f"             null: {', '.join(sorted(nulls))}")
    return lines


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("session_dir", help="Session directory holding state.json")
    parser.add_argument("--out", help="Write the request body here (feed it to service.py --dry-run)")
    parser.add_argument("--endpoint", help="Also POST the request and report what came back")
    parser.add_argument(
        "--phase-label",
        default=pconfig.DEFAULT_PHASE_LABEL,
        help=f"Value sent as phase.phase (default {pconfig.DEFAULT_PHASE_LABEL})",
    )
    parser.add_argument("--timeout-sec", type=float, default=pconfig.DEFAULT_TIMEOUT_SEC)
    parser.add_argument("--json", action="store_true", help="Print the request body instead of the report")
    args = parser.parse_args(argv)

    session_dir = Path(args.session_dir)
    state_file = state_path(session_dir)
    if not state_file.is_file():
        print(f"state.json not found at {state_file}", file=sys.stderr)
        return 2
    state = read_json(state_file, require_dict=True, strict=True)

    session_id = ""
    manifest_file = manifest_path(session_dir)
    if manifest_file.is_file():
        manifest = read_json(manifest_file, require_dict=True, default={}) or {}
        session_id = str(manifest.get("session_id") or "")

    request = build_request(_state_view(state), session_id=session_id, phase_label=args.phase_label)

    if args.json:
        print(json.dumps(request, indent=2, sort_keys=True))
    else:
        print(f"session    : {session_id or session_dir.name}")
        for line in _report(request):
            print(line)

    if args.out:
        Path(args.out).write_text(json.dumps(request, indent=2, sort_keys=True), encoding="utf-8")
        print(f"\nrequest written to {args.out}")

    if args.endpoint:
        print(f"\n--- POST {args.endpoint} ---")
        prediction = predict(request, endpoint=args.endpoint, timeout_sec=args.timeout_sec)
        print(f"parsed        : {prediction.parsed}")
        if prediction.error:
            print(f"error         : {prediction.error}")
        print(f"server_args   : {prediction.server_args}")
        print(f"envs          : {prediction.envs}")
        print(f"source_change : {prediction.source_change or '(none)'}")
        for key in ("model", "phase_rendered", "prompt_chars", "finish_reason", "dropped_flags"):
            if key in prediction.meta:
                print(f"{key:14}: {prediction.meta[key]}")
        if not prediction.parsed:
            # The pump treats this as "chain stops here", which is a normal
            # outcome; it is only a problem if it is the usual outcome.
            print("\nNo action. Fine occasionally; a pattern means the prompt is off-distribution.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
