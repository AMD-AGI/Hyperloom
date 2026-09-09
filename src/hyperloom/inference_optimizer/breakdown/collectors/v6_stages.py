# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Measurement timeline projections for SBD V6."""

from __future__ import annotations

from typing import Any

from ._common import (
    _dict_rows,
    _first,
    _mapping,
    _optional_bool,
    _parse_iso_unix,
    _to_float,
    _to_int,
)


# Statuses a producer writes for work that ran and did not succeed.
_FAILED_STATUSES = frozenset({"failed", "error", "failure", "timeout", "aborted"})
_OK_STATUSES = frozenset({"ok", "succeeded", "success", "complete", "completed", "done"})
_PARTIAL_STATUSES = frozenset({"partial", "partial_success", "degraded"})
_SKIPPED_STATUSES = frozenset({"skipped", "skip", "not_run", "noop", "no_op"})
_STOP_REASONS_SWEEP = frozenset({"sweep_failed", "sweep_unusable", "sweep_timeout"})


def _text(value: Any) -> str | None:
    """Return ``value`` as a non-empty stripped string, or ``None``."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _lower(value: Any) -> str:
    return str(value or "").strip().lower()


def _lane_status(raw: Any, *, where: str, warnings: list[str], allow_partial: bool = False) -> str:
    """Map a producer's status spelling onto the V6 enum for the field."""
    status = _lower(raw)
    if not status or status in _SKIPPED_STATUSES:
        return "skipped"
    if status in _OK_STATUSES:
        return "succeeded"
    if allow_partial and status in _PARTIAL_STATUSES:
        return "partial"
    if status not in _FAILED_STATUSES:
        warnings.append(f"v6.timeline.kernel: unrecognized {where} status {status!r}; reported as failed")
    return "failed"


def _action_rows(phase_timeline: Any, actions: frozenset[str]) -> list[dict[str, Any]]:
    """Return the ``phase_timeline`` rows for ``actions``, oldest first."""
    rows = [row for row in _dict_rows(phase_timeline) if _lower(row.get("action")) in actions]
    rows.sort(key=lambda row: (_parse_iso_unix(row.get("ts")) is None, _parse_iso_unix(row.get("ts")) or 0.0))
    return rows


def _time_window(*row_groups: list[dict[str, Any]]) -> tuple[str, str]:
    """Return ``(start_time, end_time)`` spanning every timestamped row given."""
    stamps: list[tuple[float, str]] = []
    for rows in row_groups:
        for row in rows:
            raw = _first(row.get("ts"), row.get("ended_at"), row.get("started_at"), row.get("timestamp"))
            parsed = _parse_iso_unix(raw)
            if parsed is not None:
                stamps.append((parsed, str(raw)))
    if not stamps:
        return "", ""
    stamps.sort(key=lambda item: item[0])
    return stamps[0][1], stamps[-1][1]


def _sequence(value: Any) -> list[Any]:
    """Read a recorded field that should be a sequence, whatever it turned out to be."""
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _int_list(value: Any) -> list[int]:
    """Coerce a recorded grid to the ints it can supply, dropping the rest."""
    return [number for number in (_to_int(item) for item in _sequence(value)) if number is not None]


# conc_sweep
def _conc_point(point: dict[str, Any], warnings: list[str]) -> dict[str, Any]:
    return {
        "conc": _to_int(point.get("conc")),
        "status": _lane_status(point.get("status"), where="conc_sweep.points", warnings=warnings),
        "output_throughput": _to_float(point.get("output_throughput")),
        "ttft_mean_ms": _to_float(point.get("ttft_mean_ms")),
        "e2el_mean_ms": _to_float(point.get("e2el_mean_ms")),
        "error_class": _text(point.get("error_class")),
        "error": _text(point.get("error")),
        "report_path": _text(point.get("report_path")),
    }


def _conc_arm(arm: Any, warnings: list[str]) -> dict[str, Any]:
    mapping = _mapping(arm)
    return {
        # Non-nullable: the baseline arm's defining property is that it adds no server args, and ``""`` says that
        # where ``None`` would not.
        "extra_server_args": str(mapping.get("extra_server_args") or ""),
        "points": [_conc_point(point, warnings) for point in _dict_rows(mapping.get("points"))],
    }


def _conc_pair_error(row: dict[str, Any], points_by_arm: dict[str, dict[int | None, dict[str, Any]]]) -> str | None:
    """Explain why a concurrency pair produced no speedup."""
    if _to_float(row.get("speedup")) is not None:
        return None
    conc = _to_int(row.get("conc"))
    reasons = []
    for arm in ("baseline", "optimized"):
        status = _lower(row.get(f"{arm}_status"))
        if status in _OK_STATUSES:
            continue
        point = _mapping(points_by_arm.get(arm, {}).get(conc))
        detail = _text(_first(point.get("error"), point.get("error_class"), status)) or "no point recorded"
        reasons.append(f"{arm}: {detail}")
    return "; ".join(reasons) or None


def project_conc_sweep_event(
    conc_sweep_summary: Any,
    state: Any,
    phase_timeline: Any,
    warnings: list[str],
) -> dict[str, Any] | None:
    """Project the baseline-vs-optimized concurrency curve into a V6 event."""
    summary = _mapping(conc_sweep_summary)
    state = _mapping(state)
    last = _mapping(state.get("last_conc_sweep"))
    rows = _action_rows(phase_timeline, frozenset({"conc_sweep"}))
    if not summary and not last:
        return None

    reported = _lower(_first(summary.get("status"), last.get("status")))
    budget_exhausted = _optional_bool(_first(summary.get("budget_exhausted"), last.get("budget_exhausted")))
    # One normalization feeds both the event status and ``result.status``, so a producer's spelling cannot make the
    # two disagree.
    result_status = _lane_status(reported, where="conc_sweep", warnings=warnings)
    if result_status == "succeeded":
        # A curve cut short by the time budget still produced usable pairs, but not the ladder that was asked for.
        status = "degraded" if budget_exhausted else "succeeded"
    else:
        status = result_status

    points_by_arm = {
        arm: {_to_int(point.get("conc")): point for point in _dict_rows(_mapping(summary.get(arm)).get("points"))}
        for arm in ("baseline", "optimized")
    }
    comparison = [
        {
            "conc": _to_int(row.get("conc")),
            "baseline_throughput": _to_float(row.get("baseline_tput")),
            "optimized_throughput": _to_float(row.get("optimized_tput")),
            "speedup": _to_float(row.get("speedup")),
            "error": _conc_pair_error(row, points_by_arm),
        }
        for row in _dict_rows(summary.get("comparison"))
    ]
    result_summary = _mapping(summary.get("summary"))
    stop_reason = _lower(state.get("stop_reason"))
    # ``last_conc_sweep.ts`` is stamped when the sweep finishes, and nothing records when it started.
    start_time, end_time = _time_window(rows)
    end_time = _first(str(last.get("ts") or ""), end_time) or ""
    return {
        "type": "conc_sweep",
        "kind": "conc_sweep",
        "status": status,
        "start_time": start_time,
        "end_time": end_time,
        "ext": {
            "trigger": {
                # The conc sweep is dispatched as an action, not as a phase, so nothing records which path reached it.
                "kind": None,
                "source_task_id": None,
            },
            "input_anchor": {
                # The optimized arm is defined by its server args, which the arm itself carries; no variant or task id
                # is stamped on it.
                "base_variant_id": None,
                "base_task_id": None,
                "input_throughput_tok_s_per_gpu": None,
            },
            "plan": {
                "grid_source": None,
                "concs_requested": _int_list(
                    summary.get("concs_requested") or state.get("conc_sweep_concs"),
                ),
                "budget_sec": _to_int(
                    _first(summary.get("total_budget_sec"), state.get("conc_sweep_total_budget_sec"))
                ),
            },
            "arms": {
                "baseline": _conc_arm(summary.get("baseline"), warnings),
                "optimized": _conc_arm(summary.get("optimized"), warnings),
            },
            "comparison": comparison,
            "result": {
                "status": result_status,
                # The axis the speedups were taken on; it differs by workload.
                "metric": _text(result_summary.get("metric")) or "output_throughput",
                "best_conc": _to_int(result_summary.get("best_conc")),
                "best_speedup": _to_float(result_summary.get("best_speedup")),
                "skip_reason": _text(_first(summary.get("skip_reason"), last.get("skip_reason"))),
                "budget_exhausted": budget_exhausted,
            },
            "runtime": {
                "workspace": _text(_first(summary.get("workspace"), last.get("workspace"))),
                "elapsed_sec": _to_float(summary.get("elapsed_sec")),
                "budget_remaining_sec": _to_float(summary.get("budget_remaining_sec")),
            },
            "artifacts": {
                "report_json_path": _text(summary.get("report_json_path")),
                "report_csv_path": _text(summary.get("report_csv_path")),
                "report_path": _text(summary.get("report_path")),
            },
            "failure": {
                "stop_reason": stop_reason if stop_reason in _STOP_REASONS_SWEEP else None,
                "failed_task_id": None,
                "message": _text(_first(summary.get("budget_skip_reason"), last.get("skip_reason"))),
            },
        },
    }
