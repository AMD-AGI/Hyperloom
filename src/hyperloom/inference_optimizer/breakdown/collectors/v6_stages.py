# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Measurement timeline projections for SBD V6.

``install`` and ``model_gate`` need the durable on-disk event stream because
they run before ``session_dir`` exists or before the state machine starts. So
do ``kernel``, ``roofline`` and ``baseline``: each records its own event
through the recorder as it runs, which is why nothing here projects them.
``baseline`` was the last to move, and why is worth keeping: V5 stamps its
action row and its attempt summary when the measurement *completes* and
nothing recorded when it began, so the projected window collapsed onto its own
end and the event sorted onto the timeline at the moment it finished.
Everything projected here happens inside the Coordinator, so its evidence is
already in ``state.json``, the recorder fragments and ``reports/``: these are
pure projections over V5 sections the exporter has already built, and add no
writer call sites anywhere in ``orchestrator/``.

Each projector follows the collector contract in :mod:`._common` — pure over
its inputs, never mutating them, never raising, recording problems in
``warnings`` and returning a best-effort partial. A stage the session has no
evidence for returns ``None`` so it stays out of the timeline entirely; an
empty shell with ``status: skipped`` would claim the stage was considered and
declined, which is a different fact.

Fields the V6 design names but that nothing in V5 persists are emitted as
``None`` rather than back-filled from a plausible neighbour. Each such case
carries a comment saying what is missing and why the nearest value would be
wrong.
"""

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


# Statuses a producer writes for work that ran and did not succeed. Checked
# case-insensitively; ``skipped`` is deliberately absent because a skip is not
# a failure.
_FAILED_STATUSES = frozenset({"failed", "error", "failure", "timeout", "aborted"})
_OK_STATUSES = frozenset({"ok", "succeeded", "success", "complete", "completed", "done"})
_PARTIAL_STATUSES = frozenset({"partial", "partial_success", "degraded"})
_SKIPPED_STATUSES = frozenset({"skipped", "skip", "not_run", "noop", "no_op"})
_STOP_REASONS_SWEEP = frozenset({"sweep_failed", "sweep_unusable", "sweep_timeout"})


def _text(value: Any) -> str | None:
    """Return ``value`` as a non-empty stripped string, or ``None``.

    V6 distinguishes "not recorded" from "recorded as empty", so a blank
    string never survives into the payload as ``""``.
    """
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _lower(value: Any) -> str:
    return str(value or "").strip().lower()


def _lane_status(raw: Any, *, where: str, warnings: list[str], allow_partial: bool = False) -> str:
    """Map a producer's status spelling onto the V6 enum for the field.

    Producers write their own vocabulary — a backend attempt succeeds as
    ``completed``, forge-fusion as ``ok`` — while these V6 fields are closed
    enums. Passing the raw value through hands a strict consumer a payload it
    has to reject over an ordinary success, which is the opposite of what the
    field is for.

    A spelling nothing here recognizes is read as ``failed``, matching how the
    other lanes already treat an unknown status, and warned about so an
    unlisted vocabulary surfaces rather than being quietly filed as a defeat.
    """
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
    """Return ``(start_time, end_time)`` spanning every timestamped row given.

    Rows arrive from several producers that stamp their time under different
    keys, so all the conventional ones are consulted. Both ends are ``""``
    when nothing in the groups carries a parseable timestamp; the V6 sorter
    treats that as "place last" rather than "happened at epoch".
    """
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
    """Read a recorded field that should be a sequence, whatever it turned out to be.

    A corrupt or hand-edited ``state.json`` that stored a bare scalar where a
    list belongs would otherwise raise out of the projector, and a raise here
    costs the whole timeline rather than one field. A string is wrapped rather
    than iterated: its characters are never the intended elements.
    """
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _int_list(value: Any) -> list[int]:
    """Coerce a recorded grid to the ints it can supply, dropping the rest."""
    return [number for number in (_to_int(item) for item in _sequence(value)) if number is not None]


# ---------------------------------------------------------------------------
# conc_sweep
# ---------------------------------------------------------------------------
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
        # Non-nullable: the baseline arm's defining property is that it adds
        # no server args, and ``""`` says that where ``None`` would not.
        "extra_server_args": str(mapping.get("extra_server_args") or ""),
        "points": [_conc_point(point, warnings) for point in _dict_rows(mapping.get("points"))],
    }


def _conc_pair_error(row: dict[str, Any], points_by_arm: dict[str, dict[int | None, dict[str, Any]]]) -> str | None:
    """Explain why a concurrency pair produced no speedup.

    The pairing is an outer join, so a pair fails when an arm errored, or when
    an arm has no point at that concurrency at all. Only the arm that did not
    succeed can say why — reporting the first status of the two hands back
    ``"succeeded"`` as the error whenever it is the optimized arm that broke.
    """
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
    """Project the baseline-vs-optimized concurrency curve into a V6 event.

    ``conc_sweep_summary`` mirrors ``reports/conc_sweep_summary.json`` and is
    already all but field-aligned with the V6 ``ext``; the work here is
    renaming the paired-comparison columns and splitting the flat summary into
    ``result`` / ``runtime`` / ``artifacts``.

    Args:
        conc_sweep_summary (Any): The V5 ``conc_sweep_summary`` section.
        state (Any): The V5 ``state.json`` mapping.
        phase_timeline (Any): The V5 ``phase_timeline`` rows.
        warnings (list[str]): V6 warning sink (mutated in place).

    Returns:
        dict[str, Any] | None: The timeline event, or ``None`` when the
        session never ran a concurrency sweep.
    """
    summary = _mapping(conc_sweep_summary)
    state = _mapping(state)
    last = _mapping(state.get("last_conc_sweep"))
    rows = _action_rows(phase_timeline, frozenset({"conc_sweep"}))
    if not summary and not last:
        return None

    reported = _lower(_first(summary.get("status"), last.get("status")))
    budget_exhausted = _optional_bool(_first(summary.get("budget_exhausted"), last.get("budget_exhausted")))
    # One normalization feeds both the event status and ``result.status``, so a
    # producer's spelling cannot make the two disagree. An unrecognized word
    # lands on ``failed`` with a warning, as it does in every other lane,
    # rather than on a silent ``degraded`` that reads like a real measurement.
    result_status = _lane_status(reported, where="conc_sweep", warnings=warnings)
    if result_status == "succeeded":
        # A curve cut short by the time budget still produced usable pairs,
        # but not the ladder that was asked for.
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
    # ``last_conc_sweep.ts`` is stamped when the sweep finishes, and nothing
    # records when it started. It closes the window rather than collapsing it
    # to a point; ``elapsed_sec`` carries the duration for a reader who wants
    # the span, and back-dating the start from it would invent a timestamp.
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
                # The conc sweep is dispatched as an action, not as a phase,
                # so nothing records which path reached it.
                "kind": None,
                "source_task_id": None,
            },
            "input_anchor": {
                # The optimized arm is defined by its server args, which the
                # arm itself carries; no variant or task id is stamped on it.
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
