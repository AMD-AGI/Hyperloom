# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Measurement timeline projections for SBD V6.

``install`` and ``model_gate`` need the durable on-disk event stream because
they run before ``session_dir`` exists or before the state machine starts. So
does ``kernel``: KERNEL records its own event through the recorder as it runs,
which is why nothing here projects one.
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
# The sweep lane spells success ``ok`` where the kernel lanes spell it
# ``succeeded``; the two vocabularies are not interchangeable.
_SWEEP_POINT_OK = "ok"
_STOP_REASONS_SWEEP = frozenset({"sweep_failed", "sweep_unusable", "sweep_timeout"})
_STOP_REASONS_CONC_SWEEP = frozenset({"conc_sweep_failed", "conc_sweep_unusable", "conc_sweep_timeout"})


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


def _sweep_point_status(raw: Any, *, warnings: list[str]) -> str:
    """Map a grid point's status onto the sweep enum ``ok | skipped | failed``.

    Kept separate from :func:`_lane_status` because the sweep lane spells
    success ``ok``. That is not cosmetic: :func:`project_sweep_event` counts
    usable points by that exact word, so a producer switching to ``succeeded``
    would not merely look odd — the stage would report ``skipped`` while
    carrying a full grid of measured points.
    """
    status = _lower(raw)
    if not status or status in _SKIPPED_STATUSES:
        return "skipped"
    if status in _OK_STATUSES:
        return _SWEEP_POINT_OK
    if status in _FAILED_STATUSES:
        return "failed"
    warnings.append(f"v6.timeline.sweep: unrecognized point status {status!r}; reported as failed")
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


def _failure(error_class: Any = None, error: Any = None) -> dict[str, Any]:
    return {"error_class": _text(error_class), "error": _text(error)}


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


def _sorted_grid(rows: list[dict[str, Any]], key: str) -> list[int]:
    """Return the distinct integer values of ``key`` across ``rows``, ascending."""
    values = {parsed for row in rows if (parsed := _to_int(row.get(key))) is not None}
    return sorted(values)


def _unique_paths(values: list[Any]) -> list[str]:
    seen: list[str] = []
    for value in values:
        text = _text(value)
        if text and text not in seen:
            seen.append(text)
    return seen


# ---------------------------------------------------------------------------
# baseline
# ---------------------------------------------------------------------------
def project_baseline_event(
    baseline: Any,
    phase_timeline: Any,
    warnings: list[str],
) -> dict[str, Any] | None:
    """Project the pre-optimization reference measurement into a V6 event.

    ``baseline`` is close to 1:1 with the V6 ``ext``; the phase timeline
    supplies what the section drops — the task id the measurement ran under
    and the wall-clock window it occupied.

    ``degraded`` is never emitted. Baseline either produced a usable
    throughput or it did not; a run that only succeeded after retries is
    reported as ``succeeded`` with its retries visible in
    ``baseline.attempts_history``, which is where a reader can weigh them.

    Args:
        baseline (Any): The V5 ``baseline`` section.
        phase_timeline (Any): The V5 ``phase_timeline`` rows.
        warnings (list[str]): V6 warning sink (mutated in place).

    Returns:
        dict[str, Any] | None: The timeline event, or ``None`` when the
        session holds no baseline evidence at all.
    """
    section = _mapping(baseline)
    rows = _action_rows(phase_timeline, frozenset({"baseline"}))
    attempts = _dict_rows(section.get("attempts_history"))
    throughput = _to_float(section.get("throughput_tok_s_per_gpu"))
    # ``collect_baseline`` returns 0.0, not ``None``, for a session that never
    # measured, so a positive number is the only proof a baseline exists.
    measured = throughput is not None and throughput > 0
    if not rows and not attempts and not measured:
        return None

    failed_rows = [row for row in rows + attempts if _lower(row.get("status")) in _FAILED_STATUSES]
    if measured:
        status = "succeeded"
    elif failed_rows:
        status = "failed"
    else:
        # Attempts exist but none failed and none produced a number: the
        # measurement was cut short rather than attempted and lost.
        status = "skipped"

    last_failure = failed_rows[-1] if failed_rows else {}
    # Only ``BaselineAttemptSummary`` carries failure prose; ``PhaseEvent``
    # stops at the class, so the text is looked up on the attempt side.
    last_failed_attempt = next(
        (row for row in reversed(attempts) if _lower(row.get("status")) in _FAILED_STATUSES),
        {},
    )
    start_time, end_time = _time_window(rows, attempts)
    if not start_time:
        warnings.append("v6.timeline.baseline: no timestamped baseline evidence; the event carries no time window")
    return {
        "type": "baseline",
        "kind": "baseline",
        "status": status,
        "start_time": start_time,
        "end_time": end_time,
        "ext": {
            "task_id": _text(
                _first(
                    *(row.get("task_id") for row in reversed(rows)),
                    *(row.get("task_id") for row in reversed(attempts)),
                )
            ),
            "throughput_tok_s_per_gpu": throughput if measured else None,
            "ttft_mean_ms": _to_float(section.get("ttft_mean_ms")),
            "e2el_mean_ms": _to_float(section.get("e2el_mean_ms")),
            "benchmark_report_path": _text(section.get("benchmark_report_path")),
            "failure": _failure(
                _first(last_failure.get("error_class"), last_failed_attempt.get("error_class")),
                _first(last_failed_attempt.get("error_excerpt"), last_failed_attempt.get("stderr_tail")),
            ),
        },
    }


# ---------------------------------------------------------------------------
# sweep
# ---------------------------------------------------------------------------
def _sweep_variant(point: dict[str, Any], warnings: list[str]) -> dict[str, Any]:
    return {
        # V5 names sweep points, it does not id them; the name is the identity
        # the rest of the session joins on.
        "variant_id": _text(point.get("variant_name")),
        # Sweep points are measured by the grid runner, which does not stamp
        # an orchestrator task id onto the row it writes.
        "task_id": _text(point.get("task_id")),
        "conc": _to_int(point.get("conc")),
        "isl": _to_int(point.get("isl")),
        "osl": _to_int(point.get("osl")),
        "status": _sweep_point_status(point.get("status"), warnings=warnings),
        "output_throughput_tok_s": _to_float(point.get("output_throughput_tok_s")),
        "ttft_mean_ms": _to_float(point.get("ttft_mean_ms")),
        "e2el_mean_ms": _to_float(point.get("e2el_mean_ms")),
        "benchmark_report_path": _text(point.get("benchmark_report_path")),
        # The grid runner records failure prose but not a class for it.
        "error_class": _text(point.get("error_class")),
        "error": _text(point.get("error")),
    }


def project_sweep_event(
    sweep: Any,
    state: Any,
    baseline: Any,
    phase_timeline: Any,
    warnings: list[str],
) -> dict[str, Any] | None:
    """Project the multi-dimensional load grid sweep into a V6 event.

    ``collect_sweep`` has already merged ``state.last_sweep`` with its disk
    scan, so ``all_variants`` is the authoritative point list and the grid in
    ``ext.plan`` is read back off it rather than off the request that asked
    for it — a sweep that lost points to a budget cut reports the grid it
    actually measured.

    ``ext.input_anchor.input_throughput_tok_s_per_gpu`` stays ``None``.
    Nothing snapshots the entry throughput when SWEEP opens, and
    ``state.current_best.tput`` is the end-of-session figure, which would
    misreport the sweep's own starting point whenever anything was adopted
    after it.

    Args:
        sweep (Any): The V5 ``sweep`` section.
        state (Any): The V5 ``state.json`` mapping.
        baseline (Any): The V5 ``baseline`` section, for the anchor task id.
        phase_timeline (Any): The V5 ``phase_timeline`` rows.
        warnings (list[str]): V6 warning sink (mutated in place).

    Returns:
        dict[str, Any] | None: The timeline event, or ``None`` when the
        session never swept.
    """
    section = _mapping(sweep)
    state = _mapping(state)
    last_sweep = _mapping(state.get("last_sweep"))
    attempts = _dict_rows(state.get("sweep_attempts"))
    rows = _action_rows(phase_timeline, frozenset({"sweep"}))
    points = _dict_rows(section.get("all_variants"))
    if not points and not last_sweep and not attempts and not rows:
        return None

    variants = [_sweep_variant(point, warnings) for point in points]
    ok_count = sum(1 for variant in variants if variant["status"] == _SWEEP_POINT_OK)
    failed_count = sum(1 for variant in variants if variant["status"] == "failed")

    stop_reason = _lower(state.get("stop_reason"))
    failed_variant = next((variant for variant in reversed(variants) if variant["status"] == "failed"), {})
    failed_attempt = next(
        (row for row in reversed(attempts) if _lower(row.get("status")) in _FAILED_STATUSES),
        {},
    )
    # A sweep can fail before it lays down its first grid point, in which case
    # the only record of it is a failed attempt or a sweep stop reason. Reading
    # status off the point list alone reports that run as ``skipped`` while the
    # same event carries ``stop_reason: sweep_failed``.
    failed_outright = bool(failed_attempt) or stop_reason in _STOP_REASONS_SWEEP
    if ok_count and (failed_count or failed_outright):
        status = "degraded"
    elif ok_count:
        status = "succeeded"
    elif failed_count or failed_outright:
        status = "failed"
    else:
        status = "skipped"
    start_time, end_time = _time_window(rows, attempts, [last_sweep] if last_sweep else [])
    return {
        "type": "sweep",
        "kind": "sweep",
        "status": status,
        "start_time": start_time,
        "end_time": end_time,
        "ext": {
            "trigger": {
                # A phase_history transition into SWEEP is the only trigger
                # evidence that survives; which business path asked for it
                # (post-KEEP re-sweep vs. operator forced) is not recorded.
                "kind": "phase_entry" if rows or last_sweep else None,
                "source_task_id": None,
            },
            "input_anchor": {
                "baseline_task_id": _text(
                    _first(
                        *(
                            row.get("task_id")
                            for row in reversed(_dict_rows(_mapping(baseline).get("attempts_history")))
                        )
                    )
                ),
                # ``current_best`` carries no task id in any recorded shape.
                "current_best_task_id": None,
                "input_throughput_tok_s_per_gpu": None,
            },
            "plan": {
                # ``state.last_sweep`` records the grid's contents, never
                # where the request for it came from.
                "grid_source": _text(last_sweep.get("grid_source")),
                "conc_grid": _sorted_grid(variants, "conc"),
                "isl_grid": _sorted_grid(variants, "isl"),
                "osl_grid": _sorted_grid(variants, "osl"),
            },
            "sweep": {
                "best_overall": _mapping(_first(section.get("best_overall"), last_sweep.get("best_overall"))) or None,
                "pareto_front": _dict_rows(_first(section.get("pareto_front"), last_sweep.get("pareto_front"))) or None,
                "all_variants": variants,
            },
            "artifacts": {
                "sweep_report_paths": _unique_paths([variant["benchmark_report_path"] for variant in variants]),
                "sweep_dir": _text(last_sweep.get("workspace")),
            },
            "failure": {
                "stop_reason": stop_reason if stop_reason in _STOP_REASONS_SWEEP else None,
                "failed_task_id": _text(failed_attempt.get("task_id")),
                "message": _text(_first(failed_variant.get("error"), failed_attempt.get("error"))),
            },
        },
    }


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
            "baseline_output_throughput": _to_float(row.get("baseline_tput")),
            "optimized_output_throughput": _to_float(row.get("optimized_tput")),
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
                "stop_reason": stop_reason if stop_reason in _STOP_REASONS_CONC_SWEEP else None,
                "failed_task_id": None,
                "message": _text(_first(summary.get("budget_skip_reason"), last.get("skip_reason"))),
            },
        },
    }
