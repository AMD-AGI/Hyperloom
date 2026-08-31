# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Extract Forge results and render the path-gated E2E PR report."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_RESULT_MARKER = "__FORGE_RESULT__"


def _strings(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _strings(child)


def extract_forge_result(payload: Any) -> dict[str, Any] | None:
    """Return the last valid ``__FORGE_RESULT__`` object in a log payload."""

    if isinstance(payload, dict) and isinstance(payload.get("mean_case_speedup"), (int, float)):
        return payload
    decoder = json.JSONDecoder()
    found: dict[str, Any] | None = None
    for text in _strings(payload):
        offset = 0
        while (index := text.find(_RESULT_MARKER, offset)) >= 0:
            candidate = text[index + len(_RESULT_MARKER) :].lstrip()
            try:
                decoded, consumed = decoder.raw_decode(candidate)
            except json.JSONDecodeError:
                offset = index + len(_RESULT_MARKER)
                continue
            if isinstance(decoded, dict):
                found = decoded
            offset = index + len(_RESULT_MARKER) + consumed
    return found


def _timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _phase_time(detail: dict[str, Any], phase: str) -> datetime | None:
    conditions = detail.get("orchestration", {}).get("conditions", [])
    matches = [
        parsed
        for condition in conditions
        if isinstance(condition, dict) and condition.get("phase") == phase
        if (parsed := _timestamp(condition.get("time"))) is not None
    ]
    return matches[-1] if matches else None


def _duration(start: datetime | None, end: datetime | None) -> str:
    if start is None or end is None:
        return "–"
    seconds = max(0, round((end - start).total_seconds()))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    parts = []
    if hours:
        parts.append(f"{hours}h")
    if minutes or hours:
        parts.append(f"{minutes}m")
    parts.append(f"{seconds}s")
    return " ".join(parts)


def _cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\r", "").replace("\n", "<br>")


def _number(value: Any, digits: int) -> str | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
        return None
    return f"{float(value):.{digits}f}"


def render_report(
    *,
    result_label: str,
    detail: dict[str, Any],
    forge_result: dict[str, Any] | None,
    max_hours: str,
    max_iters: str,
    gpus: str,
    workspace: str,
    head_ref: str,
    head_sha: str,
    session_id: str,
    details_url: str = "",
    error: str = "",
    now: datetime | None = None,
) -> str:
    """Render the stable PR-comment body for one Forge E2E workload."""

    queued = _phase_time(detail, "Queued")
    dispatched = _phase_time(detail, "Dispatched")
    terminal = _phase_time(detail, "Succeeded") or _phase_time(detail, "Failed")
    terminal = terminal or now or datetime.now(timezone.utc)
    rows = [
        ("result", result_label),
        ("example", f"`triton-softmax-forge-loop` (max_hours={max_hours}, max_iters={max_iters})"),
        ("resources", f"{gpus}× GPU"),
        ("workspace", f"`{workspace}`"),
        ("PR branch", f"`{head_ref}`"),
        ("commit", f"`{head_sha}`"),
        ("session_id", f"`{session_id}`"),
        ("queue → dispatch", _duration(queued, dispatched)),
        ("run time", _duration(dispatched, terminal)),
        ("total", _duration(queued, terminal)),
    ]

    if forge_result:
        checkpoint = forge_result.get("checkpoint")
        if not isinstance(checkpoint, dict):
            checkpoint = {}
        baseline = _number(forge_result.get("pristine_baseline_ms", forge_result.get("baseline_ms")), 4)
        best = _number(forge_result.get("best_ms"), 4)
        speedup = _number(forge_result.get("mean_case_speedup", forge_result.get("total_speedup")), 6)
        if baseline is not None and best is not None:
            rows.append(("baseline → best", f"{baseline} ms → {best} ms"))
        if speedup is not None:
            percent = (float(speedup) - 1.0) * 100.0
            rows.append(("speedup", f"**{speedup}x** ({percent:+.2f}%)"))

        passed = forge_result.get("validation_passed", checkpoint.get("validation_passed"))
        snr = _number(forge_result.get("snr_db", checkpoint.get("snr_db")), 1)
        if isinstance(passed, bool) or snr is not None:
            verdict = "PASS" if passed is True else "FAIL" if passed is False else "unknown"
            rows.append(("validation", f"{verdict}{f' (SNR {snr} dB)' if snr is not None else ''}"))

        iterations = forge_result.get("iteration_count")
        best_iteration = forge_result.get("best_iteration")
        if isinstance(iterations, int):
            suffix = f" (best at iteration {best_iteration})" if isinstance(best_iteration, int) else ""
            rows.append(("iterations", f"{iterations}{suffix}"))
        best_commit = forge_result.get("best_commit")
        if isinstance(best_commit, str) and best_commit:
            rows.append(("best commit", f"`{best_commit}`"))

    if error:
        rows.append(("reason", error))

    table = "\n".join(f"| {_cell(key)} | {_cell(value)} |" for key, value in rows)
    details = f"\n\n[Actions run]({details_url})" if details_url else ""
    return (
        "<!-- hyperloom-forge-ci-e2e-report -->\n"
        f"## Hyperloom Forge E2E — {result_label}\n\n"
        "| item | value |\n"
        "|---|---|\n"
        f"{table}{details}"
    )


def _read_json(path: str) -> Any:
    if not path:
        return {}
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("extract")
    render = subparsers.add_parser("render")
    render.add_argument("--result-label", required=True)
    render.add_argument("--detail-file", required=True)
    render.add_argument("--forge-result-file", default="")
    render.add_argument("--max-hours", required=True)
    render.add_argument("--max-iters", required=True)
    render.add_argument("--gpus", required=True)
    render.add_argument("--workspace", required=True)
    render.add_argument("--head-ref", required=True)
    render.add_argument("--head-sha", required=True)
    render.add_argument("--session-id", required=True)
    render.add_argument("--details-url", default="")
    render.add_argument("--error", default="")
    args = parser.parse_args()

    if args.command == "extract":
        try:
            payload = json.load(sys.stdin)
        except json.JSONDecodeError:
            return 1
        result = extract_forge_result(payload)
        if result is None:
            return 1
        json.dump(result, sys.stdout, separators=(",", ":"))
        return 0

    detail = _read_json(args.detail_file)
    forge_result = _read_json(args.forge_result_file) or None
    print(
        render_report(
            result_label=args.result_label,
            detail=detail if isinstance(detail, dict) else {},
            forge_result=forge_result if isinstance(forge_result, dict) else None,
            max_hours=args.max_hours,
            max_iters=args.max_iters,
            gpus=args.gpus,
            workspace=args.workspace,
            head_ref=args.head_ref,
            head_sha=args.head_sha,
            session_id=args.session_id,
            details_url=args.details_url,
            error=args.error,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
