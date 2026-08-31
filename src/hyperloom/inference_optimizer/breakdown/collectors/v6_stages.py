# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Measurement and Kernel timeline projections for SBD V6.

``install`` and ``model_gate`` need the durable on-disk event stream because
they run before ``session_dir`` exists or before the state machine starts.
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
    _operation_task_id,
    _optional_bool,
    _parse_iso_unix,
    _safe_get,
    _string_list,
    _to_float,
    _to_int,
)


# Statuses a producer writes for work that ran and did not succeed. Checked
# case-insensitively; ``skipped`` is deliberately absent because a skip is not
# a failure.
_FAILED_STATUSES = frozenset({"failed", "error", "failure", "timeout", "aborted"})
_OK_STATUSES = frozenset({"ok", "succeeded", "success", "complete", "completed", "done"})
# ``optimizations.attempts[].kind`` values the Kernel Agent owns. Framework and
# Explore work shares the same attempt model and must not be pulled in here.
_KERNEL_ATTEMPT_KINDS = frozenset(
    {
        "kernel_optimization",
        "kernel_collective",
        "gemm_tuning",
        "integrate_patch",
    }
)
# ``optimizations.attempts[].kind`` -> V6 ``attempts[].source_kind``. A
# ``kernel_optimization`` row is refined further by its producer, because both
# a per-kernel rewrite and a whole-pipeline GEAK run record under that kind.
_SOURCE_KIND_BY_ATTEMPT_KIND = {
    "kernel_optimization": "kernel_rewrite",
    "kernel_collective": "collective",
    "gemm_tuning": "gemm_tuning",
    "integrate_patch": "kernel_rewrite",
}
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


def _upper(value: Any) -> str:
    return str(value or "").strip().upper()


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
def _sweep_variant(point: dict[str, Any]) -> dict[str, Any]:
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
        "status": _lower(point.get("status")) or "skipped",
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

    variants = [_sweep_variant(point) for point in points]
    ok_count = sum(1 for variant in variants if variant["status"] == "ok")
    failed_count = sum(1 for variant in variants if variant["status"] == "failed")
    if ok_count and failed_count:
        status = "degraded"
    elif ok_count:
        status = "succeeded"
    elif failed_count:
        status = "failed"
    else:
        status = "skipped"

    stop_reason = _lower(state.get("stop_reason"))
    failed_variant = next((variant for variant in reversed(variants) if variant["status"] == "failed"), {})
    failed_attempt = next(
        (row for row in reversed(attempts) if _lower(row.get("status")) in _FAILED_STATUSES),
        {},
    )
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
                    _first(*(row.get("task_id") for row in reversed(_dict_rows(_mapping(baseline).get("attempts_history")))))
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
def _conc_point(point: dict[str, Any]) -> dict[str, Any]:
    return {
        "conc": _to_int(point.get("conc")),
        "status": _lower(point.get("status")) or "skipped",
        "output_throughput": _to_float(point.get("output_throughput")),
        "ttft_mean_ms": _to_float(point.get("ttft_mean_ms")),
        "e2el_mean_ms": _to_float(point.get("e2el_mean_ms")),
        "error_class": _text(point.get("error_class")),
        "error": _text(point.get("error")),
        "report_path": _text(point.get("report_path")),
    }


def _conc_arm(arm: Any) -> dict[str, Any]:
    mapping = _mapping(arm)
    return {
        # Non-nullable: the baseline arm's defining property is that it adds
        # no server args, and ``""`` says that where ``None`` would not.
        "extra_server_args": str(mapping.get("extra_server_args") or ""),
        "points": [_conc_point(point) for point in _dict_rows(mapping.get("points"))],
    }


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
    if reported == "skipped":
        status = "skipped"
    elif reported in _FAILED_STATUSES:
        status = "failed"
    elif reported in _OK_STATUSES:
        # A curve cut short by the time budget still produced usable pairs,
        # but not the ladder that was asked for.
        status = "degraded" if budget_exhausted else "succeeded"
    else:
        status = "degraded"

    comparison = [
        {
            "conc": _to_int(row.get("conc")),
            "baseline_output_throughput": _to_float(row.get("baseline_tput")),
            "optimized_output_throughput": _to_float(row.get("optimized_tput")),
            "speedup": _to_float(row.get("speedup")),
            # A pair only fails to form when one arm has no throughput; the
            # arm's own point carries why, and this row does not duplicate it.
            "error": None
            if _to_float(row.get("speedup")) is not None
            else _text(_first(row.get("baseline_status"), row.get("optimized_status"))),
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
                "concs_requested": [
                    value
                    for value in (
                        _to_int(item)
                        for item in (summary.get("concs_requested") or state.get("conc_sweep_concs") or [])
                    )
                    if value is not None
                ],
                "budget_sec": _to_int(
                    _first(summary.get("total_budget_sec"), state.get("conc_sweep_total_budget_sec"))
                ),
            },
            "arms": {
                "baseline": _conc_arm(summary.get("baseline")),
                "optimized": _conc_arm(summary.get("optimized")),
            },
            "comparison": comparison,
            "result": {
                "status": reported or "skipped",
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


# ---------------------------------------------------------------------------
# kernel
# ---------------------------------------------------------------------------
def _window_index(ts: Any, windows: list[dict[str, Any]]) -> int:
    """Return the index of the window ``ts`` belongs to.

    Windows are pre-sorted by ``(cycle, start_time)``. A row lands in the last
    window that opened at or before it, which puts evidence recorded just
    after a window closed — a lane that finished writing during the phase
    transition — back on the cycle that produced it rather than on the next
    one. Rows with no usable timestamp land in the final window, where the
    unfinished work of a session sits.
    """
    if not windows:
        return -1
    stamp = _parse_iso_unix(ts)
    if stamp is None:
        return len(windows) - 1
    index = 0
    for position, window in enumerate(windows):
        start = _parse_iso_unix(window.get("start_time"))
        if start is not None and start <= stamp:
            index = position
    return index


def _kernel_rewrite(entry: dict[str, Any], attempt: dict[str, Any]) -> dict[str, Any]:
    """Project one ``kernel_journey`` entry plus its best backend attempt."""
    e2e = _mapping(entry.get("e2e"))
    dispatch = _mapping(entry.get("dispatch"))
    decision = _upper(attempt.get("decision"))
    optimized_files = _string_list(attempt.get("optimized_files"))
    return {
        "rewrite_id": str(_first(attempt.get("attempt_id"), entry.get("kernel_id")) or ""),
        # Backend attempts are keyed by run, not by orchestrator task.
        "task_id": None,
        "parent_run_id": _text(attempt.get("run_id")),
        "kernel_id": str(entry.get("kernel_id") or ""),
        "kernel_name": _text(entry.get("name")),
        "task_group_key": _text(dispatch.get("task_group")),
        "source_file": _text(_first(entry.get("source_file"), e2e.get("target_file"))),
        "backend": _lower(attempt.get("backend")) or None,
        "execution_status": _lower(attempt.get("status")) or "skipped",
        "micro_decision": decision or "SKIPPED",
        "verification": {
            "compile_passed": _optional_bool(attempt.get("compile_passed")),
            "correctness_passed": _optional_bool(attempt.get("correctness_passed")),
            # Which reference the correctness check ran against is decided
            # inside the backend and never surfaces on the attempt record.
            "correctness_source": None,
            "micro_speedup": _to_float(_first(attempt.get("micro_speedup"), entry.get("micro_speedup"))),
        },
        "artifact": {
            "artifact_path": _first(
                optimized_files[0] if optimized_files else None,
                e2e.get("patch_path"),
            ),
            # Snapshots are materialized by integrate, which records them on
            # its own patch manifest rather than back onto the rewrite.
            "snapshot_dir": None,
            "target_file": _text(e2e.get("target_file")),
        },
        # Filled by the caller once the window's attempts are known.
        "final_rebench_attempt_ids": [],
        "outcome": _upper(_first(e2e.get("decision"), entry.get("outcome"), decision)) or "SKIPPED",
        "reason": _text(e2e.get("rejection_reason")),
        "failure": _failure(attempt.get("error_class"), attempt.get("error")),
    }


def _fusion_run(result: dict[str, Any], integrate: dict[str, Any]) -> dict[str, Any]:
    """Project ``state.last_fusion`` (+ its integrate verdict) into one row.

    Only the last fusion survives in state — the field is overwritten per run,
    so a session that fused more than once reports its final attempt. There is
    no per-run fusion ledger to read instead.
    """
    kept = _optional_bool(result.get("kept"))
    compile_pass_flag = _text(result.get("compile_pass_flag"))
    serving_speedup = _to_float(result.get("serving_speedup"))
    kernel_speedup = _to_float(result.get("kernel_speedup"))
    integrate_decision = _upper(integrate.get("decision"))
    return {
        "run_id": str(_first(result.get("run_id"), result.get("experiment_id"), "forge_fusion") or "forge_fusion"),
        "task_id": _text(_first(result.get("task_id"), "kernel_entry_fusion")),
        "status": _lower(result.get("status")) or "skipped",
        "candidate_kind": "compile_pass" if compile_pass_flag else ("authored_fusion" if kept else None),
        "source_file": _text(_first(result.get("source_file"), result.get("target_file"))),
        "best_pattern": _text(result.get("best_pattern")),
        "env_flags": _mapping(result.get("env_flags")),
        "candidate_speedup": _first(serving_speedup, kernel_speedup),
        "candidate_speedup_basis": (
            "serving_ab" if serving_speedup is not None else ("kernel_microbenchmark" if kernel_speedup is not None else None)
        ),
        "patch_path": _text(_first(result.get("patch"), integrate.get("patch_path"))),
        # forge-fusion emits the V6 vocabulary verbatim.
        "micro_decision": _lower(result.get("micro_decision")) or ("failed" if result.get("error") else "skipped"),
        "final_rebench_attempt_ids": [],
        "outcome": integrate_decision or _upper(result.get("decision")) or "SKIPPED",
        "reason": _text(_first(integrate.get("reason"), result.get("verdict"))),
        "failure": _failure(
            _first(result.get("error_class"), integrate.get("error_class")),
            _first(result.get("error"), integrate.get("error")),
        ),
        "workspace": _text(result.get("workspace")),
    }


def _gemm_tuner_attempts(run: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "tuner": str(_first(candidate.get("tuner"), candidate.get("libtype"), run.get("engine"), "") or ""),
            "status": _lower(candidate.get("status")) or "skipped",
            "best_micro_speedup": _to_float(
                _first(candidate.get("best_micro_speedup"), candidate.get("best_speedup"))
            ),
            "tuned_file": _text(candidate.get("tuned_file")),
            "reason": _text(_first(candidate.get("reason"), candidate.get("error"))),
        }
        for candidate in _dict_rows(run.get("candidates"))
    ]


def _gemm_tuning_run(run: dict[str, Any]) -> dict[str, Any]:
    """Project one ``optimizations.gemm_tuning_runs`` row."""
    parameters = _mapping(run.get("parameters"))
    summary = _mapping(run.get("summary"))
    speedup = _to_float(run.get("best_speedup"))
    status = _lower(run.get("status"))
    return {
        # A tuning run's artifact is its dispatch CSV, which is also the
        # identity ``GemmTuningRun`` is keyed by.
        "run_id": str(_first(run.get("tuned_file"), run.get("workspace"), run.get("engine"), "") or ""),
        "task_id": None,
        "backend": _lower(run.get("engine")) or None,
        "status": "succeeded" if status in _OK_STATUSES else ("skipped" if status == "skipped" else "failed"),
        "precision": _text(run.get("precision")),
        "shape_source": _text(parameters.get("shape_source")),
        "shape_artifact_path": _text(_first(parameters.get("shape_artifact_path"), parameters.get("shapes_path"))),
        "tuner_attempts": _gemm_tuner_attempts(run),
        "recommended_env": _mapping(_first(summary.get("recommended_env"), parameters.get("recommended_env"))),
        "micro_decision": (
            "candidate"
            if speedup is not None and speedup > 1.0
            else ("no_improvement" if speedup is not None else ("failed" if run.get("error") else "skipped"))
        ),
        "final_rebench_attempt_ids": [],
        "outcome": _upper(run.get("decision")) or ("KEEP" if run.get("adopted") else "SKIPPED"),
        "reason": _text(run.get("error")),
        "failure": _failure(run.get("error_class"), run.get("error")),
        "workspace": _text(run.get("workspace")),
    }


def _collective_run(attempt: dict[str, Any]) -> dict[str, Any]:
    """Project one ``collective.attempts[]`` campaign."""
    status = _lower(attempt.get("status"))
    kept = _optional_bool(attempt.get("kept"))
    return {
        "run_id": str(_first(attempt.get("collective_attempt_id"), attempt.get("experiment_id"), "") or ""),
        "task_id": None,
        "kernel_id": _text(attempt.get("kernel_id")),
        "kernel_name": _text(attempt.get("kernel_name")),
        "source_file": _text(attempt.get("source_file")),
        "collective_op": _text(attempt.get("collective_op")),
        "world_size": _to_int(attempt.get("world_size")),
        "status": "succeeded" if status in _OK_STATUSES else ("skipped" if status == "skipped" else "failed"),
        "kernel_speedup": _to_float(attempt.get("kernel_speedup")),
        "patch_path": _text(attempt.get("patch_path")),
        "micro_decision": (
            "candidate" if kept else ("failed" if attempt.get("error") else "no_improvement" if status else "skipped")
        ),
        "final_rebench_attempt_ids": [],
        # The E2E gate is the verdict that decides adoption; the
        # microbenchmark ``decision`` only decides whether it runs.
        "outcome": _upper(_first(attempt.get("integration_decision"), attempt.get("decision"))) or "SKIPPED",
        "reason": _text(_first(attempt.get("integration_error"), attempt.get("error"))),
        "failure": _failure(
            _first(attempt.get("error_class"), attempt.get("integration_error_class")),
            _first(attempt.get("error"), attempt.get("integration_error")),
        ),
        "workspace": _text(_first(attempt.get("workspace"), attempt.get("integration_workspace"))),
    }


def _geak_run(geak: dict[str, Any]) -> dict[str, Any]:
    """Project the session's GEAK whole-pipeline run.

    ``collect_geak`` folds the run into one session-scoped record, so this is
    at most one row regardless of how many macro cycles entered KERNEL.
    """
    handoff = _mapping(geak.get("handoff"))
    accepted = _mapping(geak.get("accepted_config"))
    status = _lower(geak.get("status"))
    return {
        "run_id": str(_first(geak.get("exp_root"), geak.get("report_path"), "geak") or "geak"),
        "task_id": _text(handoff.get("task_id")),
        "status": "succeeded" if status in _OK_STATUSES else ("skipped" if status == "skipped" else "failed"),
        # Whether GEAK's own baseline matched Hyperloom's is reasoned about at
        # rebench time and not written back onto the run record.
        "baseline_alignment_status": _text(geak.get("metric_basis")),
        "claim_gain": {
            "output_throughput": _to_float(geak.get("final_throughput_tok_s")),
            "gain_pct": _to_float(geak.get("gain_pct")),
            "basis": _lower(geak.get("metric_basis")) if _lower(geak.get("metric_basis")) in {"hot", "cold"} else None,
        },
        "accepted_config": {
            "extra_server_args": str(accepted.get("extra_server_args") or ""),
            "extra_envs": _mapping(accepted.get("extra_envs")),
            "overlay_path": _text(_first(accepted.get("overlay_path"), geak.get("final_patch"))),
        },
        "kernel_rewrite_ids": _string_list(
            [
                _first(kernel.get("kernel_id"), kernel.get("name"), kernel) if isinstance(kernel, dict) else kernel
                for kernel in geak.get("accepted_kernels") or []
            ]
        ),
        "final_rebench_attempt_ids": [],
        "outcome": "KEEP" if status in _OK_STATUSES else ("FAILED" if status else "SKIPPED"),
        "reason": _text(geak.get("likely_cause")),
        "failure": _failure(geak.get("error_class"), geak.get("error")),
        "artifacts": {
            "handoff_path": _text(handoff.get("path")),
            "result_path": _text(geak.get("report_path")),
            "final_launch_script": _text(geak.get("final_launch_script")),
            "workspace": _text(geak.get("exp_root")),
        },
    }


def _attempt_source_kind(attempt: dict[str, Any]) -> str:
    """Classify which lane a final-rebench attempt was validating."""
    kind = _lower(attempt.get("kind"))
    producer = _lower(attempt.get("producer"))
    backend = _lower(attempt.get("backend"))
    name = _lower(attempt.get("name"))
    if "fusion" in producer or "fusion" in name:
        return "fusion"
    if kind == "kernel_optimization" and "geak" in f"{producer} {backend}":
        return "geak_e2e"
    return _SOURCE_KIND_BY_ATTEMPT_KIND.get(kind, "kernel_rewrite")


def _attempt_validation_source(attempt: dict[str, Any], source_kind: str) -> str:
    """Name the harness the final throughput was measured on."""
    basis = _lower(attempt.get("validation_basis"))
    if "geak" in basis:
        return "geak_same_harness"
    if _lower(attempt.get("kind")) == "integrate_patch" or "integrate" in basis:
        return "integrate_e2e"
    if source_kind == "geak_e2e" and not basis:
        return "geak_same_harness"
    return "orchestrator_same_harness"


def _attempt_accuracy(attempt: dict[str, Any]) -> dict[str, Any]:
    """Read the accuracy gate's ruling off the attempt's recorded gates.

    A gate row records only ``kind`` / ``name`` / ``status`` / ``decision`` /
    ``reason``, so ``reference`` and ``value`` come back ``None`` unless a
    producer starts carrying the numbers the gate compared. What the gate
    *ruled* survives, which is the part a reader acts on.
    """
    gate = next(
        (row for row in _dict_rows(attempt.get("gates")) if "accuracy" in _lower(row.get("kind"))),
        {},
    )
    if not gate:
        return {"required": False, "reference": None, "value": None, "passed": None}
    status = _lower(gate.get("status"))
    return {
        "required": True,
        "reference": _to_float(gate.get("reference")),
        "value": _to_float(_first(gate.get("value"), gate.get("observed"))),
        "passed": True if status in _OK_STATUSES or _upper(gate.get("decision")) == "PASS" else (
            False if status in _FAILED_STATUSES else None
        ),
    }


def _final_rebench_attempt(
    attempt: dict[str, Any],
    task_id_by_operation: dict[str, str],
    throughput_unit: str,
) -> dict[str, Any]:
    """Project one ``optimizations.attempts[]`` row into a V6 rebench row."""
    source_kind = _attempt_source_kind(attempt)
    attempt_id = str(attempt.get("attempt_id") or "")
    status = _lower(attempt.get("status"))
    decision = _upper(attempt.get("decision"))
    artifacts = _dict_rows(attempt.get("artifacts"))
    return {
        "attempt_id": attempt_id,
        # The projected attempt row drops the task id; it is joined back off
        # the raw operation the row was built from.
        "task_id": _text(task_id_by_operation.get(attempt_id)),
        "backend": _lower(attempt.get("backend")) or None,
        "source_kind": source_kind,
        "source_id": str(_first(attempt.get("kernel_id"), attempt.get("name"), attempt_id) or ""),
        "validation_source": _attempt_validation_source(attempt, source_kind),
        "status": "succeeded" if status in _OK_STATUSES else ("skipped" if status == "skipped" else "failed"),
        "base_tput": _to_float(attempt.get("throughput_before")),
        "output_throughput": _to_float(attempt.get("throughput_after")),
        # Deliberately the attempt's own local gain: the session-cumulative
        # figure is a different quantity and lives in ``optimizations``.
        "gain_pct": _to_float(attempt.get("local_gain_pct")),
        "throughput_unit": throughput_unit,
        "keep_threshold_pct": _to_float(attempt.get("keep_threshold_pct")),
        "accuracy": _attempt_accuracy(attempt),
        "engagement": {
            # Whether the candidate really took effect in the measured process
            # is not verified by any producer today; asserting it from the
            # decision alone would defeat the point of the check.
            "config_matched": None,
            "artifact_applied": _optional_bool(attempt.get("integrated")),
            "overlay_loaded": None,
            "evidence_path": _text(next((row.get("path") for row in artifacts if row.get("path")), None)),
            "reason": None,
        },
        "decision": decision or "FAILED",
        # A REVERT the measurement itself broke is not the same claim as a
        # candidate that measured fairly and did not earn its threshold.
        "is_fault": status in _FAILED_STATUSES and decision != "REVERT",
        "reason": _text(attempt.get("decision_reason")),
        "failure": _failure(
            None if status not in _FAILED_STATUSES else _lower(attempt.get("decision_source")) or None,
            attempt.get("decision_reason") if status in _FAILED_STATUSES else None,
        ),
        "artifacts": {
            "workspace": _text(next((row.get("path") for row in artifacts if _lower(row.get("kind")) == "workspace"), None)),
            "benchmark_report_path": _text(
                next((row.get("path") for row in artifacts if "report" in _lower(row.get("kind"))), None)
            ),
            "patch_manifest_path": _text(
                next((row.get("path") for row in artifacts if "manifest" in _lower(row.get("kind"))), None)
            ),
        },
    }


def _kernel_route(state: dict[str, Any], collective: dict[str, Any], lanes: dict[str, list[Any]]) -> str | None:
    """Name the execution path this Kernel visit actually took."""
    if _optional_bool(collective.get("only_mode")) is True:
        return "collective_only"
    if lanes["collective_runs"] and not any(
        lanes[name] for name in ("kernel_rewrites", "fusion_runs", "gemm_tuning_runs", "geak_runs")
    ):
        return "collective_only"
    optimizer = _lower(state.get("kernel_optimizer"))
    return optimizer if optimizer in {"geak", "forge"} else None


def project_kernel_events(
    state: Any,
    windows: list[dict[str, Any]],
    warnings: list[str],
    *,
    optimizations: Any = None,
    kernel_journey: Any = None,
    collective: Any = None,
    geak: Any = None,
    baseline: Any = None,
    recorded_operations: list[dict[str, Any]] | None = None,
    kernel_optimization_summary: Any = None,
) -> list[dict[str, Any]]:
    """Project each Kernel Agent visit into its own V6 timeline event.

    KERNEL is re-entered by the macro loop, so V6 disambiguates the visits by
    ``ext.macro_cycle`` and this returns one event per window handed in by
    ``_phase_windows``. Five candidate lanes and one unified final-rebench
    ledger are bucketed into those windows: lane rows by their own timestamp,
    rebench attempts by the ``macro_cycle`` the recorder stamped on them,
    which is authoritative and needs no time reasoning.

    ``geak_runs`` is the exception. ``collect_geak`` folds the whole pipeline
    into one session-scoped record with no cycle on it, so it is attached to
    the first window and a warning names the ambiguity when there is more than
    one.

    Args:
        state (Any): The V5 ``state.json`` mapping.
        windows (list[dict[str, Any]]): ``_phase_windows(state, _KERNEL_PHASES)``.
        warnings (list[str]): V6 warning sink (mutated in place).
        optimizations (Any): The V5 ``optimizations`` section.
        kernel_journey (Any): The V5 ``kernel_journey`` section.
        collective (Any): The V5 ``collective`` section.
        geak (Any): The ``collect_geak`` record, or ``{}``.
        baseline (Any): The V5 ``baseline`` section, for the throughput unit.
        recorded_operations (list[dict[str, Any]] | None): Raw recorder
            operations, joined on ``operation_id`` for the attempt task ids.
        kernel_optimization_summary (Any): The V5 summary section, used only
            to cross-check the attempt count.

    Returns:
        list[dict[str, Any]]: One event per Kernel visit, oldest first. Empty
        when the session never entered KERNEL and recorded no kernel work.
    """
    state = _mapping(state)
    optimizations = _mapping(optimizations)
    collective = _mapping(collective)
    geak = _mapping(geak)
    windows = [window for window in windows or [] if isinstance(window, dict)]

    journey_entries = _dict_rows(_mapping(kernel_journey).get("kernels"))
    fusion_result = _mapping(state.get("last_fusion"))
    fusion_integrate = _mapping(state.get("last_fusion_integrate"))
    gemm_runs = _dict_rows(optimizations.get("gemm_tuning_runs"))
    collective_attempts = _dict_rows(collective.get("attempts"))
    kernel_attempts = [
        row for row in _dict_rows(optimizations.get("attempts")) if _lower(row.get("kind")) in _KERNEL_ATTEMPT_KINDS
    ]
    geak_engaged = _optional_bool(geak.get("engaged")) is True or bool(geak.get("status"))

    has_evidence = bool(
        journey_entries or fusion_result or gemm_runs or collective_attempts or kernel_attempts or geak_engaged
    )
    if not windows:
        if not has_evidence:
            return []
        # Kernel work without a KERNEL_AGENT transition means phase_history
        # was lost or truncated. Keep the evidence rather than drop it, on one
        # synthetic window carrying the session's own cycle.
        warnings.append(
            "v6.timeline.kernel: kernel evidence exists with no KERNEL_AGENT phase history; "
            "reported as a single window"
        )
        windows = [
            {
                "cycle": _to_int(state.get("macro_cycle")) or 0,
                "start_time": "",
                "end_time": "",
                "rows": [],
                "exit_row": {},
            }
        ]

    task_id_by_operation = {
        str(operation.get("operation_id") or ""): _operation_task_id(operation)
        for operation in recorded_operations or []
        if isinstance(operation, dict) and operation.get("operation_id")
    }
    throughput_unit = str(_mapping(baseline).get("throughput_unit") or "tok/s")

    lanes_by_window: list[dict[str, list[Any]]] = [
        {
            "kernel_rewrites": [],
            "fusion_runs": [],
            "gemm_tuning_runs": [],
            "collective_runs": [],
            "geak_runs": [],
            "attempts": [],
        }
        for _ in windows
    ]

    for entry in journey_entries:
        attempts = _dict_rows(entry.get("backend_attempts"))
        if not attempts:
            # Discovered or dispatched but never attempted: the journey row is
            # still the only record that the kernel was considered.
            attempts = [{}]
        for attempt in attempts:
            index = _window_index(_first(attempt.get("ts"), _safe_get(entry, "e2e", "ts")), windows)
            lanes_by_window[index]["kernel_rewrites"].append(_kernel_rewrite(entry, attempt))

    if fusion_result:
        index = _window_index(_first(fusion_integrate.get("ts"), fusion_result.get("ts")), windows)
        lanes_by_window[index]["fusion_runs"].append(_fusion_run(fusion_result, fusion_integrate))

    for run in gemm_runs:
        lanes_by_window[_window_index(run.get("ts"), windows)]["gemm_tuning_runs"].append(_gemm_tuning_run(run))

    for attempt in collective_attempts:
        index = _window_index(_first(attempt.get("integration_ts"), attempt.get("ts")), windows)
        lanes_by_window[index]["collective_runs"].append(_collective_run(attempt))

    if geak_engaged:
        if len(windows) > 1:
            warnings.append(
                "v6.timeline.kernel: the GEAK record is session-scoped across "
                f"{len(windows)} kernel visits; attached to the first"
            )
        lanes_by_window[0]["geak_runs"].append(_geak_run(geak))

    cycle_to_index = {int(window.get("cycle") or 0): position for position, window in enumerate(windows)}
    for attempt in kernel_attempts:
        cycle = _to_int(attempt.get("macro_cycle"))
        index = cycle_to_index.get(cycle) if cycle is not None else None
        if index is None:
            index = _window_index(_first(attempt.get("ended_at"), attempt.get("started_at")), windows)
        lanes_by_window[index]["attempts"].append(
            _final_rebench_attempt(attempt, task_id_by_operation, throughput_unit)
        )

    expected_attempts = _to_int(_safe_get(_mapping(kernel_optimization_summary), "totals", "attempted"))
    if expected_attempts is not None and expected_attempts != len(kernel_attempts):
        warnings.append(
            "v6.timeline.kernel: kernel_optimization_summary counts "
            f"{expected_attempts} attempts but the recorder ledger holds {len(kernel_attempts)}"
        )

    events: list[dict[str, Any]] = []
    for window, lanes in zip(windows, lanes_by_window):
        events.append(_kernel_event(state, collective, window, lanes))
    return events


def _kernel_event(
    state: dict[str, Any],
    collective: dict[str, Any],
    window: dict[str, Any],
    lanes: dict[str, list[Any]],
) -> dict[str, Any]:
    """Assemble one Kernel visit from its window and its bucketed lanes."""
    attempts = lanes["attempts"]
    # Every lane row links back to the rebench attempts that validated it. The
    # link is made here rather than in the lane projectors because only now is
    # the window's attempt set known.
    by_source: dict[str, list[str]] = {}
    for attempt in attempts:
        by_source.setdefault(attempt["source_kind"], []).append(attempt["attempt_id"])
    for lane_name, source_kind in (
        ("kernel_rewrites", "kernel_rewrite"),
        ("fusion_runs", "fusion"),
        ("gemm_tuning_runs", "gemm_tuning"),
        ("collective_runs", "collective"),
        ("geak_runs", "geak_e2e"),
    ):
        ids = by_source.get(source_kind, [])
        for row in lanes[lane_name]:
            row["final_rebench_attempt_ids"] = list(ids)

    exit_row = _mapping(window.get("exit_row"))
    evidence = _mapping(exit_row.get("evidence"))
    entry_row = next(iter(window.get("rows") or []), {})
    stage_failed = bool(evidence.get("error") or evidence.get("error_class"))
    did_work = any(lanes[name] for name in lanes)
    kept = any(_upper(attempt.get("decision")) == "KEEP" for attempt in attempts)
    if stage_failed:
        status = "failed"
    elif not window.get("end_time") and did_work:
        # The session ended inside KERNEL; the visit has real work on it but
        # never reached its own exit.
        status = "degraded"
    elif kept or attempts:
        status = "succeeded"
    elif did_work:
        # Candidates were produced but none reached a final rebench.
        status = "degraded"
    else:
        status = "skipped"

    return {
        "type": "kernel",
        "kind": "agent",
        "status": status,
        "start_time": str(window.get("start_time") or ""),
        "end_time": str(window.get("end_time") or ""),
        "ext": {
            "macro_cycle": int(window.get("cycle") or 0),
            "entry": {
                "from_phase": _text(_mapping(entry_row).get("from_phase")),
                "route": _kernel_route(state, collective, lanes),
                # No task id is stamped on the phase transition into KERNEL.
                "input_task_id": None,
                # The first attempt's own comparison base is the throughput
                # this visit started from, measured rather than inferred.
                "input_throughput": next(
                    (attempt["base_tput"] for attempt in attempts if attempt["base_tput"] is not None),
                    None,
                ),
            },
            "kernel_rewrites": lanes["kernel_rewrites"],
            "fusion_runs": lanes["fusion_runs"],
            "gemm_tuning_runs": lanes["gemm_tuning_runs"],
            "collective_runs": lanes["collective_runs"],
            "geak_runs": lanes["geak_runs"],
            "attempts": attempts,
            "exit_reason": _text(exit_row.get("reason")),
            "failure": {
                "failed_task_id": _text(_first(evidence.get("failed_task_id"), evidence.get("task_id"))),
                "error_class": _text(evidence.get("error_class")),
                "error": _text(evidence.get("error")),
            },
        },
    }
