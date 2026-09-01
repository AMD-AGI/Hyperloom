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
# ``optimizations.attempts[].kind`` values that can belong to the Kernel Agent.
# This is a candidate filter, never a verdict: ``integrate_patch`` lands every
# patch source, so its kind says nothing about which lever produced it. See
# ``_owns_kernel_attempt`` for how ownership is actually decided.
_KERNEL_ATTEMPT_KINDS = frozenset(
    {
        "kernel_optimization",
        "kernel_collective",
        "gemm_tuning",
        "integrate_patch",
    }
)
# Kinds the Kernel Agent is the sole producer of. A row carrying one of these
# is the Kernel Agent's even with no ownership metadata on it.
_KERNEL_ONLY_ATTEMPT_KINDS = frozenset({"kernel_optimization", "kernel_collective", "gemm_tuning"})
_KERNEL_AGENT = "kernel_agent"
_KERNEL_PHASE = "KERNEL_AGENT"
# ``optimizations.attempts[].kind`` -> V6 ``attempts[].source_kind``. A
# ``kernel_optimization`` row is refined further by its producer, because both
# a per-kernel rewrite and a whole-pipeline GEAK run record under that kind.
_SOURCE_KIND_BY_ATTEMPT_KIND = {
    "kernel_optimization": "kernel_rewrite",
    "kernel_collective": "collective",
    "gemm_tuning": "gemm_tuning",
    "integrate_patch": "kernel_rewrite",
}
_PARTIAL_STATUSES = frozenset({"partial", "partial_success", "degraded"})
_SKIPPED_STATUSES = frozenset({"skipped", "skip", "not_run", "noop", "no_op"})
# ``kernel_journey`` rolls a kernel's lifecycle up into its own vocabulary,
# which is not the ``outcome`` enum. Uppercasing it emits ``ADOPTED`` into a
# field whose closed enum has no such member.
_ROLLUP_TO_OUTCOME = {
    "adopted": "KEEP",
    "reverted": "REVERT",
    "failed": "FAILED",
    # Considered but never carried to an end-to-end verdict. The candidate's
    # own ``micro_decision`` is where that part of the story is told.
    "attempted": "SKIPPED",
    "dispatched": "SKIPPED",
    "discovered": "SKIPPED",
    "skipped": "SKIPPED",
}
_OUTCOME_VALUES = frozenset({"KEEP", "REVERT", "FAILED", "NEEDS_REVIEW", "SKIPPED"})
_MICRO_DECISIONS = frozenset({"KEEP", "REVERT", "PARTIAL", "FAILED", "SKIPPED"})
# A final rebench either adopted, rejected, could not decide, or never reached
# a verdict. ``PARTIAL`` / ``SKIPPED`` are candidate-level words and say
# nothing about a measurement that ran.
_ATTEMPT_DECISIONS = frozenset({"KEEP", "REVERT", "NEEDS_REVIEW", "FAILED"})
# The sweep lane spells success ``ok`` where the kernel lanes spell it
# ``succeeded``; the two vocabularies are not interchangeable.
_SWEEP_POINT_OK = "ok"
# ``make_proposal`` emits a fifth verdict for a rewrite that cleared the gates
# it could measure and left the rest unproven. ``micro_decision`` has no such
# member; ``PARTIAL`` is the one that means the same thing here, and reading it
# as an unknown spelling would file a measured near-miss as a skip.
_MICRO_DECISION_ALIASES = {"NEEDS_REVIEW": "PARTIAL", "REVIEW": "PARTIAL"}
# ``revalidation_status`` is stamped on ``geak_result`` only by the verdicts
# that close a candidate out. A promotion writes a ``geak_e2e`` stack entry and
# a rebench attempt instead, so the absence of this field is not a KEEP.
_GEAK_REVALIDATION_OUTCOMES = {
    "no_promote": "REVERT",
    "no_material": "REVERT",
    "failed": "FAILED",
    "fallback_failed": "FAILED",
}
# ``collect_geak``'s own words for a run whose result it could not read back.
# Both carry ``error_class: no_result``; recognizing them here keeps the shared
# status normalizer from reporting an ordinary GEAK miss as vocabulary drift.
_GEAK_STATUS_ALIASES = {"missing": "failed", "no_result_recovered_from_disk": "failed"}
# ``missing`` is the one GEAK status that is not a record of work: it is what
# ``collect_geak`` returns when ``kernel_optimizer=geak`` was selected, nothing
# was recorded, and nothing was on disk either — the launch config echoed back.
# GEAK is the default backend, so reading it as evidence would conjure a Kernel
# visit onto every session that ended before KERNEL.
_GEAK_CONFIG_ECHO_STATUSES = frozenset({"missing"})
# The run-level lanes spell the same field in their own lowercase vocabulary.
_CANDIDATE_DECISIONS = frozenset({"candidate", "no_improvement", "failed", "skipped"})
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


def _lane_outcome(*values: Any, where: str, warnings: list[str]) -> str:
    """Resolve a candidate's business outcome onto the closed ``outcome`` enum.

    The first value that *says something recognizable* wins; ``kernel_journey``'s
    coarse rollup vocabulary is translated rather than uppercased, and
    ``PARTIAL`` — legal for a ``micro_decision`` but not for an ``outcome`` —
    becomes ``NEEDS_REVIEW``, which is the enum's word for "measured, but not a
    verdict".

    An unrecognized spelling is warned about and then *skipped*, not returned
    on. The values are passed in priority order precisely because the later
    ones are weaker restatements of the same fact, so a new spelling of the
    strongest one must not also discard the fallbacks that still parse —
    that would disable the drift tolerance in the case it exists for.
    """
    drifted = False
    for value in values:
        raw = _lower(value)
        if not raw:
            continue
        mapped = _ROLLUP_TO_OUTCOME.get(raw, raw.upper())
        if mapped in _OUTCOME_VALUES:
            return mapped
        if mapped == "PARTIAL":
            return "NEEDS_REVIEW"
        warnings.append(f"v6.timeline.kernel: unrecognized {where} outcome {raw!r}; ignored")
        drifted = True
    if drifted:
        warnings.append(f"v6.timeline.kernel: no recognizable {where} outcome; reported as SKIPPED")
    return "SKIPPED"


def _attempt_decision(raw: Any, *, warnings: list[str]) -> str:
    """Resolve a final rebench's verdict onto the ``attempts[].decision`` enum.

    Distinct from :func:`_micro_decision`, whose enum is the candidate-level
    one. A rebench that produced no verdict at all reports ``FAILED``; a
    verdict spelled in a vocabulary this does not know reports
    ``NEEDS_REVIEW``, because something was decided and filing it as a failure
    would claim more than the record supports.
    """
    decision = _upper(raw)
    if not decision:
        return "FAILED"
    if decision in _ATTEMPT_DECISIONS:
        return decision
    if decision in _MICRO_DECISION_ALIASES or decision == "PARTIAL":
        return "NEEDS_REVIEW"
    warnings.append(f"v6.timeline.kernel: unrecognized attempt decision {decision!r}; reported as NEEDS_REVIEW")
    return "NEEDS_REVIEW"


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


def _candidate_decision(raw: Any, *, where: str, warnings: list[str], default: str) -> str:
    """Resolve a run-level ``micro_decision`` onto the lowercase lane enum.

    ``kernel_rewrites`` spells this field in the ``KEEP``/``REVERT`` vocabulary;
    the fusion, GEMM and collective lanes use ``candidate | no_improvement |
    failed | skipped`` instead. Those three are the only place a producer's word
    reaches the field unmediated, so drift is warned about rather than shipped.
    """
    decision = _lower(raw)
    if not decision:
        return default
    if decision in _CANDIDATE_DECISIONS:
        return decision
    warnings.append(f"v6.timeline.kernel: unrecognized {where} decision {decision!r}; reported as {default}")
    return default


def _micro_decision(raw: Any, *, where: str, warnings: list[str], default: str = "SKIPPED") -> str:
    """Resolve a candidate-level decision onto the ``micro_decision`` enum."""
    decision = _MICRO_DECISION_ALIASES.get(_upper(raw), _upper(raw))
    if not decision:
        return default
    if decision in _MICRO_DECISIONS:
        return decision
    warnings.append(f"v6.timeline.kernel: unrecognized {where} decision {decision!r}; reported as {default}")
    return default


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


def _adopted_attempt(attempts: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Pick the backend attempt the kernel's end-to-end verdict belongs to.

    A kernel is rewritten by several backends and at most one of those attempts
    is carried to integrate, so the kernel-level ``e2e`` block describes exactly
    one of them. V5 already resolves this: ``collectors/kernels.py`` stamps the
    kernel decision and its verification onto the adopted attempt alone, using
    ``verification.best_attempt_id`` then ``best_backend`` then the highest
    micro speedup. The hints are consumed there and do not survive onto the
    journey entry, so the same order is re-derived from what does survive —
    ``best_artifact_path``, which the recorder writes onto the adopted row and
    no other, then the speedup ordering V5 falls back to. A session recorded
    before that stamp existed has neither, and lands on the speedup ordering.

    Returns:
        The adopted attempt, or ``None`` when there is nothing to adopt.
    """
    rows = [row for row in attempts if row]
    if not rows:
        return None
    stamped = [row for row in rows if row.get("best_artifact_path")]

    def _rank(row: dict[str, Any]) -> tuple[float, str]:
        speedup = _to_float(_first(row.get("micro_speedup"), row.get("speedup")))
        return (speedup if speedup is not None else float("-inf"), str(row.get("ts") or ""))

    return max(stamped or rows, key=_rank)


def _rewrite_id(entry: dict[str, Any], attempt: dict[str, Any], sequence: int) -> str:
    """Identify one backend attempt on one kernel, uniquely within the session.

    ``attempt_id`` is the producer's own key and wins whenever it survived.
    A recording that lost it used to fall back to ``kernel_id``, which *every*
    backend attempt on that kernel shares — so two rows took one id, the
    adopted-set test in :func:`_link_rebench_attempts` matched both, and the
    losing backend claimed the winner's final rebench. The fallback now carries
    the backend and the attempt's position, which is what tells the rows apart.
    """
    attempt_id = _text(attempt.get("attempt_id"))
    if attempt_id:
        return attempt_id
    kernel_id = _text(entry.get("kernel_id")) or "unknown"
    backend = _lower(attempt.get("backend")) or "unknown"
    return f"{kernel_id}:{backend}:{sequence}"


def _kernel_rewrite(
    entry: dict[str, Any],
    attempt: dict[str, Any],
    *,
    sequence: int,
    adopted: bool,
    warnings: list[str],
) -> dict[str, Any]:
    """Project one ``kernel_journey`` entry's backend attempt into a rewrite row.

    ``adopted`` says whether this is the attempt the kernel's ``e2e`` verdict
    was reached on. Only that one may carry kernel-level facts: the E2E
    ``outcome``, the kernel's best micro speedup, the integrated patch, and —
    via :func:`_lane_identity_keys` — the kernel's identity when final
    rebenches are linked. A losing attempt that inherited them would read as
    having been integrated and validated, when the run rejected it.

    The verdict and the verification gates are read off the attempt row, where
    the recorder folds the kernel-level ``proposal`` / ``verification`` onto
    the adopted attempt. A backend writes none of them itself, so a row that
    carries them carries them because it won.
    """
    e2e = _mapping(entry.get("e2e"))
    dispatch = _mapping(entry.get("dispatch"))
    where = "kernel_rewrites"
    decision = _micro_decision(attempt.get("decision"), where=where, warnings=warnings)
    execution_status = _lane_status(attempt.get("status"), where=where, warnings=warnings, allow_partial=True)
    optimized_files = _string_list(attempt.get("optimized_files"))
    if adopted:
        outcome = _lane_outcome(e2e.get("decision"), entry.get("outcome"), decision, where=where, warnings=warnings)
        reason = _text(e2e.get("rejection_reason"))
        # The kernel-level best speedup describes whichever attempt achieved
        # it, so it is only a fallback for the attempt that did.
        micro_speedup = _to_float(_first(attempt.get("micro_speedup"), entry.get("micro_speedup")))
        # The integrated patch is likewise the adopted attempt's; it is what
        # the kernel was carried to integrate with.
        artifact_path = _first(
            _text(attempt.get("best_artifact_path")),
            e2e.get("patch_path"),
            optimized_files[0] if optimized_files else None,
        )
    else:
        # This attempt never reached a final rebench, so it has no end-to-end
        # result of its own. Its own decision is still reported as
        # ``micro_decision``; ``outcome`` says only how far it got.
        outcome = "FAILED" if execution_status == "failed" or decision == "FAILED" else "SKIPPED"
        reason = "not the adopted backend attempt for this kernel"
        micro_speedup = _to_float(attempt.get("micro_speedup"))
        # Its own output only. The kernel's integrated patch belongs to the
        # attempt that produced it.
        artifact_path = optimized_files[0] if optimized_files else None
    return {
        "rewrite_id": _rewrite_id(entry, attempt, sequence),
        # Backend attempts are keyed by run, not by orchestrator task.
        "task_id": None,
        "parent_run_id": _text(attempt.get("run_id")),
        "kernel_id": str(entry.get("kernel_id") or ""),
        "kernel_name": _text(entry.get("name")),
        "task_group_key": _text(dispatch.get("task_group")),
        "source_file": _text(_first(entry.get("source_file"), e2e.get("target_file"))),
        "backend": _lower(attempt.get("backend")) or None,
        "execution_status": execution_status,
        "micro_decision": decision,
        "verification": {
            "compile_passed": _optional_bool(attempt.get("compile_passed")),
            "correctness_passed": _optional_bool(attempt.get("correctness_passed")),
            "correctness_source": _text(attempt.get("correctness_source")),
            "micro_speedup": micro_speedup,
        },
        "artifact": {
            # The rewritten source, where the run resolved one. An attempt's
            # ``optimized_files`` is its own output path, which for a real
            # backend run is the stdout log rather than the rewrite, so it is
            # the last resort and not the first.
            "artifact_path": artifact_path,
            # Snapshots are materialized by integrate, which records them on
            # its own patch manifest rather than back onto the rewrite.
            "snapshot_dir": None,
            "target_file": _text(e2e.get("target_file")),
        },
        # Filled by the caller once the window's attempts are known.
        "final_rebench_attempt_ids": [],
        "outcome": outcome,
        "reason": reason,
        "failure": _failure(attempt.get("error_class"), attempt.get("error")),
    }


def _fusion_paired(result: dict[str, Any], integrate: dict[str, Any]) -> bool:
    """Report whether an integrate verdict adjudicates *this* fusion run.

    The two live in separate last-write-wins state fields. ``last_fusion`` is
    rewritten by every fusion run; ``last_fusion_integrate`` only by a run that
    reached integration — which a fusion that was not kept never does, and one
    missing its patch or target file returns before. So the verdict sitting in
    state may belong to an earlier round, and reading it as this run's would
    report a new candidate as adopted on the strength of an old measurement.

    ``fusion_run_id`` is stamped on both by the phase handler and is the
    authoritative pairing. Sessions recorded before it existed are paired on
    the patch the verdict names, which is the artifact integrate was handed.
    """
    if not result or not integrate:
        return False
    run_id = _lower(result.get("fusion_run_id"))
    verdict_run_id = _lower(integrate.get("fusion_run_id"))
    if run_id or verdict_run_id:
        return bool(run_id) and run_id == verdict_run_id
    patch = _lower(result.get("patch"))
    return bool(patch) and patch == _lower(integrate.get("patch_path"))


def _fusion_run(result: dict[str, Any], integrate: dict[str, Any], warnings: list[str]) -> dict[str, Any]:
    """Project ``state.last_fusion`` (+ its integrate verdict) into one row.

    Only the last fusion survives in state — the field is overwritten per run,
    so a session that fused more than once reports its final attempt. There is
    no per-run fusion ledger to read instead.

    The integrate verdict is used only when :func:`_fusion_paired` confirms it
    belongs to this run. A fusion is a *micro* KEEP the moment KernelForge
    keeps it, and that is not adoption: the e2e re-baseline decides. So an
    unpaired or absent verdict can never produce ``outcome: KEEP`` — a kept
    candidate that was never adjudicated reports ``NEEDS_REVIEW``.
    """
    paired = _fusion_paired(result, integrate)
    if integrate and not paired:
        warnings.append(
            "v6.timeline.kernel: last_fusion_integrate does not name this fusion run; "
            "its verdict is not reported against it"
        )
        integrate = {}
    kept = _optional_bool(result.get("kept"))
    compile_pass_flag = _text(result.get("compile_pass_flag"))
    serving_speedup = _to_float(result.get("serving_speedup"))
    kernel_speedup = _to_float(result.get("kernel_speedup"))
    micro_decision = _candidate_decision(
        result.get("micro_decision"),
        where="fusion_runs",
        warnings=warnings,
        default="failed" if result.get("error") else "skipped",
    )
    if integrate:
        outcome = _lane_outcome(integrate.get("decision"), where="fusion_runs", warnings=warnings)
    elif micro_decision == "candidate":
        # Kept by the microbenchmark and never adjudicated end to end. The run
        # ended before integrate, or integrate had nothing to apply.
        outcome = "NEEDS_REVIEW"
    elif micro_decision == "failed":
        outcome = "FAILED"
    elif micro_decision == "no_improvement":
        # Nothing was proposed for adoption, which is a settled negative.
        outcome = "REVERT"
    else:
        outcome = "SKIPPED"
    return {
        "run_id": str(
            _first(result.get("fusion_run_id"), result.get("run_id"), result.get("experiment_id"), "forge_fusion")
            or "forge_fusion"
        ),
        "task_id": _text(_first(result.get("task_id"), "kernel_entry_fusion")),
        # forge-fusion reports success as ``ok``; the field's enum does not
        # have that word.
        "status": _lane_status(result.get("status"), where="fusion_runs", warnings=warnings),
        "candidate_kind": "compile_pass" if compile_pass_flag else ("authored_fusion" if kept else None),
        "source_file": _text(_first(result.get("source_file"), result.get("target_file"))),
        "best_pattern": _text(result.get("best_pattern")),
        "env_flags": _mapping(result.get("env_flags")),
        "candidate_speedup": _first(serving_speedup, kernel_speedup),
        "candidate_speedup_basis": (
            "serving_ab"
            if serving_speedup is not None
            else ("kernel_microbenchmark" if kernel_speedup is not None else None)
        ),
        "patch_path": _text(_first(result.get("patch"), integrate.get("patch_path"))),
        # forge-fusion emits the V6 vocabulary verbatim; anything else is drift.
        "micro_decision": micro_decision,
        "final_rebench_attempt_ids": [],
        "outcome": outcome,
        "reason": _text(_first(integrate.get("reason"), result.get("verdict"))),
        "failure": _failure(
            _first(result.get("error_class"), integrate.get("error_class")),
            _first(result.get("error"), integrate.get("error")),
        ),
        "workspace": _text(result.get("workspace")),
    }


def _gemm_tuner_attempts(run: dict[str, Any], warnings: list[str]) -> list[dict[str, Any]]:
    return [
        {
            "tuner": str(_first(candidate.get("tuner"), candidate.get("libtype"), run.get("engine"), "") or ""),
            # This enum, alone among the lanes, has a fourth member for a tuner
            # that ran cleanly and produced nothing.
            "status": (
                "empty"
                if _lower(candidate.get("status")) == "empty"
                else _lane_status(candidate.get("status"), where="tuner_attempts", warnings=warnings)
            ),
            "best_micro_speedup": _to_float(_first(candidate.get("best_micro_speedup"), candidate.get("best_speedup"))),
            "tuned_file": _text(candidate.get("tuned_file")),
            "reason": _text(_first(candidate.get("reason"), candidate.get("error"))),
        }
        for candidate in _dict_rows(run.get("candidates"))
    ]


def _gemm_tuning_run(run: dict[str, Any], warnings: list[str]) -> dict[str, Any]:
    """Project one ``optimizations.gemm_tuning_runs`` row.

    Like the fusion and collective lanes, a tuning run that beat its baseline
    and that nothing adjudicated reports ``NEEDS_REVIEW`` rather than
    ``SKIPPED``. The tuner does not adopt — only the final rebench does — and
    ``SKIPPED`` is the one word :func:`_settle_pending_outcomes` will not
    revisit, so a tuned table the rebench went on to KEEP used to stay on the
    timeline as work that never happened.
    """
    parameters = _mapping(run.get("parameters"))
    summary = _mapping(run.get("summary"))
    speedup = _to_float(run.get("best_speedup"))
    micro_decision = (
        "candidate"
        if speedup is not None and speedup > 1.0
        else ("no_improvement" if speedup is not None else ("failed" if run.get("error") else "skipped"))
    )
    if _text(run.get("decision")) or run.get("adopted"):
        outcome = _lane_outcome(
            run.get("decision"),
            "adopted" if run.get("adopted") else None,
            where="gemm_tuning_runs",
            warnings=warnings,
        )
    elif micro_decision == "candidate":
        outcome = "NEEDS_REVIEW"
    elif micro_decision == "failed":
        outcome = "FAILED"
    elif micro_decision == "no_improvement":
        # The tuner measured no improvement, so it offered nothing to
        # integrate; there is no verdict outstanding.
        outcome = "REVERT"
    else:
        outcome = "SKIPPED"
    return {
        # A tuning run's artifact is its dispatch CSV, which is also the
        # identity ``GemmTuningRun`` is keyed by.
        "run_id": str(_first(run.get("tuned_file"), run.get("workspace"), run.get("engine"), "") or ""),
        "task_id": None,
        "backend": _lower(run.get("engine")) or None,
        "status": _lane_status(run.get("status"), where="gemm_tuning_runs", warnings=warnings),
        "precision": _text(run.get("precision")),
        "shape_source": _text(parameters.get("shape_source")),
        "shape_artifact_path": _text(_first(parameters.get("shape_artifact_path"), parameters.get("shapes_path"))),
        "tuner_attempts": _gemm_tuner_attempts(run, warnings),
        "recommended_env": _mapping(_first(summary.get("recommended_env"), parameters.get("recommended_env"))),
        "micro_decision": micro_decision,
        "final_rebench_attempt_ids": [],
        "outcome": outcome,
        "reason": _text(run.get("error")),
        "failure": _failure(run.get("error_class"), run.get("error")),
        "workspace": _text(run.get("workspace")),
    }


def _collective_run(attempt: dict[str, Any], warnings: list[str]) -> dict[str, Any]:
    """Project one ``collective.attempts[]`` campaign.

    The campaign row is persisted when the microbenchmark verdict lands, and
    the integration fields are merged onto it later — so a campaign that was
    kept but never integrated (the run ended, or integration crashed between
    the two writes) sits in state carrying ``decision: KEEP`` and no
    ``integration_decision``. Only the e2e gate adopts, so that row reports
    ``NEEDS_REVIEW``: a candidate awaiting a verdict, not an adopted one.
    """
    status = _lower(attempt.get("status"))
    kept = _optional_bool(attempt.get("kept"))
    micro_decision = (
        "candidate" if kept else ("failed" if attempt.get("error") else "no_improvement" if status else "skipped")
    )
    if _text(attempt.get("integration_decision")):
        outcome = _lane_outcome(attempt.get("integration_decision"), where="collective_runs", warnings=warnings)
    elif micro_decision == "candidate":
        outcome = "NEEDS_REVIEW"
    elif micro_decision == "failed":
        outcome = "FAILED"
    elif micro_decision == "no_improvement":
        # The campaign proposed nothing for integration; nothing is pending.
        outcome = "REVERT"
    else:
        outcome = "SKIPPED"
    return {
        "run_id": str(_first(attempt.get("collective_attempt_id"), attempt.get("experiment_id"), "") or ""),
        "task_id": None,
        "kernel_id": _text(attempt.get("kernel_id")),
        "kernel_name": _text(attempt.get("kernel_name")),
        "source_file": _text(attempt.get("source_file")),
        "collective_op": _text(attempt.get("collective_op")),
        "world_size": _to_int(attempt.get("world_size")),
        "status": _lane_status(attempt.get("status"), where="collective_runs", warnings=warnings),
        "kernel_speedup": _to_float(attempt.get("kernel_speedup")),
        "patch_path": _text(attempt.get("patch_path")),
        "micro_decision": micro_decision,
        "final_rebench_attempt_ids": [],
        # The E2E gate is the verdict that decides adoption; the
        # microbenchmark ``decision`` only decides whether it runs.
        "outcome": outcome,
        "reason": _text(_first(attempt.get("integration_error"), attempt.get("error"))),
        "failure": _failure(
            _first(attempt.get("error_class"), attempt.get("integration_error_class")),
            _first(attempt.get("error"), attempt.get("integration_error")),
        ),
        "workspace": _text(_first(attempt.get("workspace"), attempt.get("integration_workspace"))),
    }


def _geak_run(geak: dict[str, Any], revalidation_status: str, warnings: list[str]) -> dict[str, Any]:
    """Project the session's GEAK whole-pipeline run.

    ``collect_geak`` folds the run into one session-scoped record, so this is
    at most one row regardless of how many macro cycles entered KERNEL.

    ``status`` says only that the GEAK runner finished; it is the claim, not
    the verdict. What settles a GEAK candidate is the orchestrator's own
    rebench, which either promotes it — leaving a linked final-rebench attempt
    — or closes it out by stamping ``revalidation_status`` on ``geak_result``
    with ``no_promote`` / ``no_material`` / a failure. So ``outcome`` starts
    from that stamp, and :func:`_settle_pending_outcomes` settles it once the
    window's rebench attempts are linked. A finished runner nobody adjudicated
    is ``NEEDS_REVIEW``, never ``KEEP``.

    ``status`` and ``outcome`` are derived from one normalized value. They used
    to keep separate vocabularies — ``outcome`` consulted the whole
    ``_SKIPPED_STATUSES`` set while ``status`` compared against the literal
    ``"skipped"`` — so a producer writing ``not_run`` emitted a row that
    reported ``status: failed`` beside ``outcome: SKIPPED``.
    """
    handoff = _mapping(geak.get("handoff"))
    accepted = _mapping(geak.get("accepted_config"))
    status = _lower(geak.get("status"))
    run_status = _GEAK_STATUS_ALIASES.get(status) or _lane_status(
        status,
        where="geak_runs",
        warnings=warnings,
    )
    revalidated = _GEAK_REVALIDATION_OUTCOMES.get(_lower(revalidation_status))
    if revalidated:
        outcome = revalidated
    elif run_status == "succeeded":
        outcome = "NEEDS_REVIEW"
    elif run_status == "skipped":
        outcome = "SKIPPED"
    else:
        outcome = "FAILED"
    return {
        "run_id": str(_first(geak.get("exp_root"), geak.get("report_path"), "geak") or "geak"),
        "task_id": _text(handoff.get("task_id")),
        # The design's enum for this field has no ``partial``; a salvaged run
        # is reported by its ``error_class`` and ``reason``, not by a status
        # the contract does not define.
        "status": "failed" if run_status == "partial" else run_status,
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
                for kernel in _sequence(geak.get("accepted_kernels"))
            ]
        ),
        "final_rebench_attempt_ids": [],
        "outcome": outcome,
        "reason": _text(_first(geak.get("likely_cause"), _text(revalidation_status))),
        "failure": _failure(geak.get("error_class"), geak.get("error")),
        "artifacts": {
            "handoff_path": _text(handoff.get("path")),
            "result_path": _text(geak.get("report_path")),
            "final_launch_script": _text(geak.get("final_launch_script")),
            "workspace": _text(geak.get("exp_root")),
        },
    }


def _owns_kernel_attempt(attempt: dict[str, Any]) -> bool:
    """Report whether a final-rebench attempt belongs to the Kernel Agent.

    ``kind`` alone cannot answer this. ``integrate_patch`` is the action that
    lands *every* patch source — Framework Agent enablement and Explore config
    patches included — which is why ``attribution.py`` resolves its family from
    ownership metadata rather than from the action name. Filtering on kind here
    would file a Framework Agent patch under the Kernel timeline, and on a
    session with no kernel work at all it would conjure a Kernel visit out of
    one.

    ``attempts[].agent`` is the right field: ``_attempt_agent`` has already run
    the producer's recorded value, the kind table, and ``patch_author`` over the
    row. ``phase`` backs it up. Only when a row carries neither — an old
    recording from before producers stamped ownership — does the kind decide,
    and then only for the kinds no other agent produces.
    """
    agent = _lower(attempt.get("agent"))
    phase = _upper(attempt.get("phase"))
    if agent or phase:
        return agent == _KERNEL_AGENT or phase == _KERNEL_PHASE
    return _lower(attempt.get("kind")) in _KERNEL_ONLY_ATTEMPT_KINDS


def _attempt_source_kind(attempt: dict[str, Any]) -> str:
    """Classify which lane a final-rebench attempt was validating."""
    kind = _lower(attempt.get("kind"))
    producer = _lower(attempt.get("producer"))
    backend = _lower(attempt.get("backend"))
    name = _lower(attempt.get("name"))
    if "fusion" in producer or "fusion" in name:
        return "fusion"
    if name == "geak_e2e" or (kind == "kernel_optimization" and "geak" in f"{producer} {backend}"):
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
        "passed": True
        if status in _OK_STATUSES or _upper(gate.get("decision")) == "PASS"
        else (False if status in _FAILED_STATUSES else None),
    }


def _final_rebench_attempt(
    attempt: dict[str, Any],
    task_id_by_operation: dict[str, str],
    throughput_unit: str,
    attributed_gain_pct: float | None,
    warnings: list[str],
) -> dict[str, Any]:
    """Project one ``optimizations.attempts[]`` row into a V6 rebench row."""
    source_kind = _attempt_source_kind(attempt)
    attempt_id = str(attempt.get("attempt_id") or "")
    status = _lower(attempt.get("status"))
    decision = _attempt_decision(attempt.get("decision"), warnings=warnings)
    artifacts = _dict_rows(attempt.get("artifacts"))
    output_throughput = _to_float(attempt.get("throughput_after"))
    # Recorder operations use their terminal business word as ``status``. A
    # rebench that ran cleanly and rejected the candidate is therefore written
    # as ``reverted`` rather than ``succeeded`` even though it produced the
    # measurement that made the decision. V6's status answers a different
    # question -- whether a usable measurement formed -- so a measured REVERT
    # is a success, while an apply/launch failure that merely defaults its
    # decision to REVERT remains a failure.
    measured_revert = (
        decision == "REVERT"
        and output_throughput is not None
        and output_throughput > 0
        and status not in _SKIPPED_STATUSES
    )
    projected_status = (
        "succeeded"
        if status in _OK_STATUSES or measured_revert
        else ("skipped" if status in _SKIPPED_STATUSES else "failed")
    )
    # Any non-skipped attempt that failed to form a measurement is a rebench
    # fault, including legacy rows whose producer wrote the business terminal
    # word ``reverted`` rather than one of the canonical failure statuses.
    is_fault = projected_status == "failed"
    return {
        "attempt_id": attempt_id,
        # The projected attempt row drops the task id; it is joined back off
        # the raw operation the row was built from.
        "task_id": _text(task_id_by_operation.get(attempt_id)),
        "backend": _lower(attempt.get("backend")) or None,
        "source_kind": source_kind,
        "source_id": str(_first(attempt.get("kernel_id"), attempt.get("name"), attempt_id) or ""),
        "validation_source": _attempt_validation_source(attempt, source_kind),
        "status": projected_status,
        "base_tput": _to_float(attempt.get("throughput_before")),
        "output_throughput": output_throughput,
        # These are deliberately two fields. ``local_gain_pct`` is measured
        # against this attempt's dynamic input and cannot be summed. The
        # attributed figure comes from ``optimizations.entries`` and uses the
        # one session baseline, so it is the additive contribution consumed by
        # ``outcome.validation.attribution``.
        "local_gain_pct": _to_float(attempt.get("local_gain_pct")),
        "attributed_gain_pct": attributed_gain_pct,
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
        "decision": decision,
        # A REVERT the measurement itself broke is not the same claim as a
        # candidate that measured fairly and did not earn its threshold.
        "is_fault": is_fault,
        "reason": _text(attempt.get("decision_reason")),
        "failure": _failure(
            None if not is_fault else _lower(attempt.get("decision_source")) or None,
            attempt.get("decision_reason") if is_fault else None,
        ),
        "artifacts": {
            "workspace": _text(
                next((row.get("path") for row in artifacts if _lower(row.get("kind")) == "workspace"), None)
            ),
            "benchmark_report_path": _text(
                next((row.get("path") for row in artifacts if "report" in _lower(row.get("kind"))), None)
            ),
            "patch_manifest_path": _text(
                next((row.get("path") for row in artifacts if "manifest" in _lower(row.get("kind"))), None)
            ),
        },
    }


def _recorded_geak_final_attempts(
    recorded_operations: list[dict[str, Any]] | None,
    geak: dict[str, Any],
) -> list[dict[str, Any]]:
    """Recover GEAK's real final rebench from its recorder operation.

    Older sessions did not write GEAK's final validation as an
    ``optimizations.attempts[]`` row. The producer instead upserted one
    ``kernel_optimizer_run`` operation across runner, candidate and
    final-validation stages; the V5 optimization ledger deliberately excludes
    that route-level kind. Reading only the ledger thus leaves an old,
    successfully promoted GEAK candidate with no final rebench and an
    ``outcome`` of ``NEEDS_REVIEW``.

    The final-validation substep and gate are author-time evidence and contain
    the authoritative validation tier plus the measured throughput. Shape that
    operation like a canonical attempt for V6 only, keeping the V5 projection
    untouched.
    """
    attempts: list[dict[str, Any]] = []
    for operation in recorded_operations or []:
        if not isinstance(operation, dict):
            continue
        if _lower(operation.get("kind")) != "kernel_optimizer_run":
            continue
        if "geak" not in {_lower(operation.get("name")), _lower(operation.get("strategy"))}:
            continue

        final_steps = [
            row
            for row in _dict_rows(operation.get("substeps"))
            if _lower(row.get("kind")) in {"final_validation", "final_validation_failed"}
        ]
        final_gates = [
            row for row in _dict_rows(operation.get("gates")) if _lower(row.get("kind")) == "final_validation"
        ]
        if not final_steps and not final_gates:
            # Runner/candidate evidence is not a final rebench verdict.
            continue

        step = final_steps[-1] if final_steps else {}
        gate = final_gates[-1] if final_gates else {}
        evidence = _mapping(gate.get("evidence"))
        details = _mapping(evidence.get("details"))
        measured_tput = _to_float(
            _first(
                evidence.get("measured_tput"),
                details.get("measured_tput"),
                details.get("output_throughput"),
            )
        )
        step_kind = _lower(step.get("kind"))
        if step:
            # A terminal substep is newer and more specific than a route gate.
            # In particular, never let a stale passed gate turn an explicit
            # ``final_validation_failed`` step into KEEP.
            validation_passed = step_kind == "final_validation" and (
                _optional_bool(_safe_get(step, "metadata", "final_validation")) is True
                or _lower(step.get("status")) in _OK_STATUSES
            )
        else:
            # Older recorder payloads may have only the final-validation gate.
            validation_passed = _lower(gate.get("status")) in ({"passed"} | _OK_STATUSES)
        if validation_passed:
            decision = "KEEP"
            status = "succeeded"
        elif measured_tput is not None and measured_tput > 0:
            # The candidate was measured and rejected against current_best.
            decision = "REVERT"
            status = "succeeded"
        else:
            decision = "FAILED"
            status = "failed"

        validation_source = str(
            _first(
                evidence.get("source"),
                _safe_get(step, "metadata", "validation_source"),
                "geak_final_validation",
            )
            or "geak_final_validation"
        )
        report_path = _text(geak.get("report_path"))
        base_tput = _to_float(_first(details.get("current_best_tput"), details.get("baseline_tput")))
        local_gain_pct = (
            (measured_tput - base_tput) / base_tput * 100.0
            if measured_tput is not None and base_tput is not None and base_tput > 0
            else None
        )
        attempts.append(
            {
                "attempt_id": str(operation.get("operation_id") or ""),
                "agent": str(operation.get("agent") or _KERNEL_AGENT),
                "producer": "geak",
                # Reuse the canonical attempt vocabulary so the existing source
                # classifier and linker can consume the recovered row.
                "kind": "kernel_optimization",
                "name": "geak",
                "phase": str(operation.get("phase") or _KERNEL_PHASE),
                "macro_cycle": operation.get("macro_cycle"),
                "backend": "geak",
                "status": status,
                "decision": decision,
                "throughput_before": base_tput,
                "throughput_after": measured_tput,
                "local_gain_pct": local_gain_pct,
                "validation_basis": validation_source,
                "integrated": decision == "KEEP",
                "started_at": str(operation.get("started_at") or ""),
                "ended_at": str(_first(step.get("ended_at"), operation.get("ended_at")) or ""),
                "decision_source": "geak_final_validation",
                "decision_reason": _text(_first(details.get("reason"), gate.get("reason"), operation.get("error"))),
                "gates": _dict_rows(operation.get("gates")),
                "artifacts": ([{"kind": "benchmark_report", "path": report_path}] if report_path else []),
            }
        )
    return attempts


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
) -> list[dict[str, Any]]:
    """Project each Kernel Agent visit into its own V6 timeline event.

    KERNEL is re-entered by the macro loop, so V6 disambiguates the visits by
    ``ext.macro_cycle`` and this returns one event per window handed in by
    ``_phase_windows``. Five candidate lanes and one unified final-rebench
    ledger are bucketed into those windows: lane rows by their own timestamp,
    rebench attempts by the ``macro_cycle`` the recorder stamped on them,
    which is authoritative and needs no time reasoning.

    ``geak_runs`` is the exception. ``collect_geak`` folds the whole pipeline
    into one session-scoped record with no cycle on it. A recorded final
    validation can resolve that ambiguity with its route ``macro_cycle``;
    otherwise the record is attached to the first window and warned when
    there is more than one.

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
        row
        for row in _dict_rows(optimizations.get("attempts"))
        if _lower(row.get("kind")) in _KERNEL_ATTEMPT_KINDS and _owns_kernel_attempt(row)
    ]
    # Recover old sessions whose GEAK final rebench exists only on the
    # route-level ``kernel_optimizer_run`` operation. New sessions have the
    # canonical ``geak_e2e`` attempt introduced by PR #1340.
    recorded_geak_attempts = _recorded_geak_final_attempts(recorded_operations, geak)
    # Detect the canonical attempt the same way the lane classifier does. A
    # narrower test here (``name == "geak_e2e"``) lets a canonical row spelled
    # any other way slip past, and the recovery below then adds a second row
    # for the same final validation — the canonical and route operations have
    # different ids, so the de-dup cannot catch it either.
    canonical_geak_attempts = [row for row in kernel_attempts if _attempt_source_kind(row) == "geak_e2e"]
    known_attempt_ids = {str(row.get("attempt_id") or "") for row in kernel_attempts}
    if not canonical_geak_attempts:
        for attempt in recorded_geak_attempts:
            attempt_id = str(attempt.get("attempt_id") or "")
            if attempt_id and attempt_id not in known_attempt_ids:
                kernel_attempts.append(attempt)
                known_attempt_ids.add(attempt_id)
    geak_status = _lower(geak.get("status"))
    geak_engaged = _optional_bool(geak.get("engaged")) is True or bool(geak_status)
    # Engagement is not evidence. ``collect_geak`` reports ``missing`` when the
    # optimizer flag selected GEAK and neither a result nor an on-disk working
    # tree was found, which is the launch config read back rather than a record
    # of work. GEAK is the default backend, so counting it would synthesize a
    # Kernel visit for every session that ended in PRELUDE.
    geak_evidence = geak_engaged and geak_status not in _GEAK_CONFIG_ECHO_STATUSES

    has_evidence = bool(
        journey_entries or fusion_result or gemm_runs or collective_attempts or kernel_attempts or geak_evidence
    )
    if not windows:
        if not has_evidence:
            return []
        # Kernel work without a KERNEL_AGENT transition means phase_history
        # was lost or truncated. Keep the evidence rather than drop it, on one
        # synthetic window carrying the session's own cycle.
        warnings.append(
            "v6.timeline.kernel: kernel evidence exists with no KERNEL_AGENT phase history; reported as a single window"
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

    # Rewrite rows that may speak for their whole kernel. Only the adopted
    # backend attempt gets the kernel's identity when rebenches are linked;
    # see ``_adopted_attempt``.
    adopted_by_window: list[set[str]] = [set() for _ in windows]

    for entry in journey_entries:
        attempts = _dict_rows(entry.get("backend_attempts"))
        if not attempts:
            # Discovered or dispatched but never attempted: the journey row is
            # still the only record that the kernel was considered.
            attempts = [{}]
        adopted = _adopted_attempt(attempts)
        for sequence, attempt in enumerate(attempts):
            index = _window_index(_first(attempt.get("ts"), _safe_get(entry, "e2e", "ts")), windows)
            is_adopted = attempt is adopted or adopted is None
            row = _kernel_rewrite(entry, attempt, sequence=sequence, adopted=is_adopted, warnings=warnings)
            lanes_by_window[index]["kernel_rewrites"].append(row)
            if is_adopted:
                adopted_by_window[index].add(row["rewrite_id"])

    if fusion_result:
        index = _window_index(_first(fusion_integrate.get("ts"), fusion_result.get("ts")), windows)
        lanes_by_window[index]["fusion_runs"].append(_fusion_run(fusion_result, fusion_integrate, warnings))

    for run in gemm_runs:
        lanes_by_window[_window_index(run.get("ts"), windows)]["gemm_tuning_runs"].append(
            _gemm_tuning_run(run, warnings)
        )

    for attempt in collective_attempts:
        index = _window_index(_first(attempt.get("integration_ts"), attempt.get("ts")), windows)
        lanes_by_window[index]["collective_runs"].append(_collective_run(attempt, warnings))

    # A cycle resolves to a window only when it owns exactly one. The phase
    # machine advances monotonically inside a macro-cycle, so KERNEL is entered
    # once per cycle and this is normally total — but a truncated or unstamped
    # ``phase_history`` can still yield two windows on one cycle, and a
    # single-valued dict would silently file every attempt of that cycle onto
    # whichever window happened to be built last. Ambiguous cycles fall through
    # to the timestamp placement instead of picking one arbitrarily.
    windows_by_cycle: dict[int, list[int]] = {}
    for position, window in enumerate(windows):
        windows_by_cycle.setdefault(int(window.get("cycle") or 0), []).append(position)
    cycle_to_index = {cycle: found[0] for cycle, found in windows_by_cycle.items() if len(found) == 1}
    ambiguous_cycles = sorted(cycle for cycle, found in windows_by_cycle.items() if len(found) > 1)
    if ambiguous_cycles:
        warnings.append(
            "v6.timeline.kernel: macro cycle(s) "
            f"{', '.join(str(cycle) for cycle in ambiguous_cycles)} hold more than one KERNEL window; "
            "their attempts are placed by timestamp instead"
        )

    if geak_engaged:
        geak_attempt_indexes: list[int] = []
        geak_cycle_evidence = canonical_geak_attempts or recorded_geak_attempts
        for attempt in geak_cycle_evidence:
            cycle = _to_int(attempt.get("macro_cycle"))
            index = cycle_to_index.get(cycle) if cycle is not None else None
            if index is None:
                timestamp = _first(attempt.get("ended_at"), attempt.get("started_at"))
                if timestamp:
                    index = _window_index(timestamp, windows)
            if index is not None:
                geak_attempt_indexes.append(index)
        geak_window_resolved = (
            bool(geak_cycle_evidence)
            and len(geak_attempt_indexes) == len(geak_cycle_evidence)
            and len(set(geak_attempt_indexes)) == 1
        )
        geak_window_index = geak_attempt_indexes[0] if geak_window_resolved else 0
        if len(windows) > 1 and not geak_window_resolved:
            warnings.append(
                "v6.timeline.kernel: the GEAK record is session-scoped across "
                f"{len(windows)} kernel visits; attached to the first"
            )
        # The verdict on a GEAK candidate is written back onto ``geak_result``
        # rather than onto the run record ``collect_geak`` projects.
        revalidation_status = str(_mapping(state.get("geak_result")).get("revalidation_status") or "")
        lanes_by_window[geak_window_index]["geak_runs"].append(_geak_run(geak, revalidation_status, warnings))

    # The kernel each rebench says it was validating, where the producer
    # recorded one. ``source_id`` on the projected row falls back to the
    # operation name and then to the attempt's own id, so it cannot be read
    # back as "a kernel was named".
    subject_ids = {
        str(attempt.get("attempt_id") or ""): _lower(attempt.get("kernel_id"))
        for attempt in kernel_attempts
        if attempt.get("attempt_id") and _lower(attempt.get("kernel_id"))
    }

    attributed_gain_by_attempt = {
        str(entry.get("adopted_attempt_id") or ""): _to_float(entry.get("gain_pct"))
        for entry in _dict_rows(optimizations.get("entries"))
        if entry.get("adopted_attempt_id")
    }

    for attempt in kernel_attempts:
        cycle = _to_int(attempt.get("macro_cycle"))
        index = cycle_to_index.get(cycle) if cycle is not None else None
        if index is None:
            index = _window_index(_first(attempt.get("ended_at"), attempt.get("started_at")), windows)
        lanes_by_window[index]["attempts"].append(
            _final_rebench_attempt(
                attempt,
                task_id_by_operation,
                throughput_unit,
                attributed_gain_by_attempt.get(str(attempt.get("attempt_id") or "")),
                warnings,
            )
        )

    # Do not compare this list's length with
    # ``kernel_optimization_summary.totals.attempted``. The summary counts one
    # latest micro-level record per kernel plus collective attempts, whereas
    # this lane contains final rebenches and also admits fusion, GEMM and the
    # route-level ``geak_e2e`` attempt. A difference is expected and is not
    # evidence that either recorder stream lost data.

    events: list[dict[str, Any]] = []
    for window, lanes, adopted_ids in zip(windows, lanes_by_window, adopted_by_window):
        events.append(_kernel_event(state, collective, window, lanes, adopted_ids, subject_ids, warnings))
    return events


_LANE_SOURCE_KINDS = (
    ("kernel_rewrites", "kernel_rewrite"),
    ("fusion_runs", "fusion"),
    ("gemm_tuning_runs", "gemm_tuning"),
    ("collective_runs", "collective"),
    ("geak_runs", "geak_e2e"),
)

# Per lane, the row fields whose values a rebench attempt's ``source_id`` may
# be spelled as. ``source_id`` is the attempt's ``kernel_id`` where it has one
# and its operation ``name`` otherwise, so both the subject and the run key
# have to be offered.
_LANE_IDENTITY_FIELDS = {
    "kernel_rewrites": ("kernel_id", "kernel_name", "rewrite_id"),
    "fusion_runs": ("run_id", "task_id", "source_file"),
    "gemm_tuning_runs": ("run_id",),
    "collective_runs": ("kernel_id", "kernel_name", "run_id"),
    "geak_runs": ("run_id", "task_id"),
}


def _lane_identity_keys(lane_name: str, row: dict[str, Any], *, adopted: bool = True) -> set[str]:
    """Return the identifiers a rebench attempt could name this row by.

    A rewrite row that is not its kernel's adopted attempt answers only to its
    own ``rewrite_id``. Every backend attempt on a kernel carries the same
    ``kernel_id``, so offering that as a key would let each of them claim the
    one rebench the adopted attempt actually triggered.
    """
    fields = _LANE_IDENTITY_FIELDS.get(lane_name, ())
    if lane_name == "kernel_rewrites" and not adopted:
        fields = ("rewrite_id",)
    keys = set()
    for field in fields:
        value = _lower(row.get(field))
        if value:
            keys.add(value)
    return keys


def _link_rebench_attempts(
    lanes: dict[str, list[Any]],
    adopted_rewrites: set[str],
    subject_ids: dict[str, str],
    warnings: list[str],
) -> None:
    """Point every candidate at the final rebench attempts that validated it.

    Matching is by identity — the attempt's ``source_id`` against the names the
    candidate is known by — because the design asks for the rebenches *this
    candidate* triggered, and grouping by ``source_kind`` alone would give two
    rewrites in one visit the same two attempts each.

    An edge is made only where exactly one candidate answers to the attempt's
    ``source_id``. Two candidates sharing an identity are as unresolvable as
    none, so a contested attempt stays unlinked and is warned about — the rule
    applies to every ambiguity, not only to the attempts that matched nothing.

    An attempt that names no subject can still be placed: where its lane holds
    exactly one row, the kind grouping is unambiguous and that row takes it.
    An attempt that *does* name a subject its lane speaks the same language as
    — a kernel id, against a lane keyed on kernel identity — and matches none
    of them is never absorbed, however few candidates are on offer. The record
    says it validated some other kernel, and one available row is not evidence
    against that. All three cases are warned about, because a wrong edge here
    reads as a candidate having been validated when it was not.

    The comparison is deliberately namespace-aware. A fusion rebench carrying
    the kernel it patched is not in conflict with the visit's only fusion run,
    which is identified by run id; treating those two vocabularies as rivals
    would withhold a link the record fully supports.
    """
    attempts = lanes["attempts"]
    by_source: dict[str, list[dict[str, Any]]] = {}
    for attempt in attempts:
        by_source.setdefault(attempt["source_kind"], []).append(attempt)

    for lane_name, source_kind in _LANE_SOURCE_KINDS:
        rows = lanes[lane_name]
        candidates = by_source.get(source_kind, [])
        if not rows or not candidates:
            continue
        keys_by_row: list[set[str]] = []
        for row in rows:
            adopted = lane_name != "kernel_rewrites" or row["rewrite_id"] in adopted_rewrites
            keys_by_row.append(_lane_identity_keys(lane_name, row, adopted=adopted))
            row["final_rebench_attempt_ids"] = []

        claimed_by: dict[str, list[int]] = {}
        for attempt in candidates:
            source_id = _lower(attempt.get("source_id"))
            if not source_id:
                continue
            owners = [position for position, keys in enumerate(keys_by_row) if source_id in keys]
            if owners:
                claimed_by[attempt["attempt_id"]] = owners

        contested: list[str] = []
        for attempt in candidates:
            owners = claimed_by.get(attempt["attempt_id"])
            if owners is None:
                continue
            if len(owners) == 1:
                rows[owners[0]]["final_rebench_attempt_ids"].append(attempt["attempt_id"])
            else:
                contested.append(attempt["attempt_id"])
        if contested:
            warnings.append(
                f"v6.timeline.kernel: {len(contested)} {source_kind} rebench attempt(s) answer to more than "
                f"one recorded candidate; left unlinked"
            )

        # Contested attempts named a subject, so they are neither anonymous nor
        # unmatched; excluding them here keeps the fallback below from
        # absorbing an attempt whose owner is genuinely undecidable.
        leftover = [attempt for attempt in candidates if attempt["attempt_id"] not in claimed_by]
        if not leftover:
            continue
        order = [attempt["attempt_id"] for attempt in candidates]
        # Only a lane keyed on kernel identity can be contradicted by an
        # attempt that names a kernel.
        keyed_on_kernel = "kernel_id" in _LANE_IDENTITY_FIELDS.get(lane_name, ())
        conflicting = [attempt for attempt in leftover if keyed_on_kernel and subject_ids.get(attempt["attempt_id"])]
        conflicting_ids = {attempt["attempt_id"] for attempt in conflicting}
        anonymous = [attempt["attempt_id"] for attempt in leftover if attempt["attempt_id"] not in conflicting_ids]
        if anonymous:
            if len(rows) == 1:
                rows[0]["final_rebench_attempt_ids"] = sorted(
                    set(rows[0]["final_rebench_attempt_ids"]) | set(anonymous),
                    key=order.index,
                )
            else:
                warnings.append(
                    f"v6.timeline.kernel: {len(anonymous)} {source_kind} rebench attempt(s) name no source and "
                    f"{len(rows)} candidates could own them; left unlinked"
                )
        if conflicting:
            named = ", ".join(sorted({subject_ids[attempt["attempt_id"]] for attempt in conflicting}))
            warnings.append(
                f"v6.timeline.kernel: {len(conflicting)} {source_kind} rebench attempt(s) name kernels "
                f"({named}) that match none of the {len(rows)} recorded candidates; left unlinked"
            )


def _settle_pending_outcomes(lanes: dict[str, list[Any]], warnings: list[str]) -> None:
    """Let a linked final rebench settle a candidate still awaiting a verdict.

    A candidate reports ``NEEDS_REVIEW`` when it was produced but nothing
    adjudicated it — a GEAK run the runner finished, a fusion kept by its
    microbenchmark, a collective campaign whose integration never wrote back.
    Where the window holds a final rebench that names such a candidate, that
    measurement *is* the adjudication and outranks the pending state.

    Only pending rows are touched. A candidate that already carries an
    explicit e2e verdict keeps it: two verdicts disagreeing is a fact worth
    seeing, not one to paper over by preferring whichever ran last.
    """
    by_id = {attempt["attempt_id"]: attempt for attempt in lanes["attempts"] if attempt.get("attempt_id")}
    for lane_name, _ in _LANE_SOURCE_KINDS:
        for row in lanes[lane_name]:
            if row.get("outcome") != "NEEDS_REVIEW":
                continue
            linked = [by_id[aid] for aid in row.get("final_rebench_attempt_ids") or [] if aid in by_id]
            # A skipped or faulted rebench measured nothing, so it decided
            # nothing; the candidate stays pending.
            measured = [attempt for attempt in linked if attempt["status"] == "succeeded"]
            if not measured:
                continue
            decisions = {decision for attempt in measured if (decision := _upper(attempt.get("decision")))}
            if decisions == {"KEEP"}:
                row["outcome"] = "KEEP"
            elif decisions == {"REVERT"}:
                row["outcome"] = "REVERT"
            elif decisions:
                identity = str(
                    _first(
                        row.get("rewrite_id"),
                        row.get("run_id"),
                        row.get("kernel_id"),
                        row.get("task_id"),
                        "unknown",
                    )
                    or "unknown"
                )
                warnings.append(
                    "v6.timeline.kernel: linked final rebenches disagree for "
                    f"{lane_name} {identity} ({', '.join(sorted(decisions))}); "
                    "outcome left as NEEDS_REVIEW"
                )


def _kernel_event(
    state: dict[str, Any],
    collective: dict[str, Any],
    window: dict[str, Any],
    lanes: dict[str, list[Any]],
    adopted_rewrites: set[str],
    subject_ids: dict[str, str],
    warnings: list[str],
) -> dict[str, Any]:
    """Assemble one Kernel visit from its window and its bucketed lanes."""
    attempts = lanes["attempts"]
    # Every lane row links back to the rebench attempts that validated it. The
    # link is made here rather than in the lane projectors because only now is
    # the window's attempt set known.
    _link_rebench_attempts(lanes, adopted_rewrites, subject_ids, warnings)
    # A candidate left pending by its own producer is settled by the rebench
    # that measured it, now that the links exist.
    _settle_pending_outcomes(lanes, warnings)

    exit_row = _mapping(window.get("exit_row"))
    evidence = _mapping(exit_row.get("evidence"))
    entry_row = next(iter(window.get("rows") or []), {})
    stage_failed = bool(evidence.get("error") or evidence.get("error_class"))
    candidates = any(lanes[name] for name, _ in _LANE_SOURCE_KINDS)
    did_work = candidates or bool(attempts)
    kept = any(_upper(attempt.get("decision")) == "KEEP" for attempt in attempts)
    # A rebench that produced a measurement, even one that decided REVERT. A
    # skipped attempt measured nothing, so the count of attempt rows alone must
    # not be read as the stage having concluded anything.
    measured = [attempt for attempt in attempts if attempt["status"] == "succeeded"]
    faulted = [attempt for attempt in attempts if attempt["status"] == "failed"]
    if stage_failed and not (kept or measured):
        status = "failed"
    elif not window.get("end_time") and did_work:
        # The session ended inside KERNEL; the visit has real work on it but
        # never reached its own exit.
        status = "degraded"
    elif stage_failed:
        # The visit adopted or measured something and *then* hit a phase-level
        # error. ``failed`` would deny work the session's own ledger kept; the
        # error is still reported, in ``ext.failure``.
        status = "degraded"
    elif kept or measured:
        status = "succeeded"
    elif faulted:
        # Every rebench that ran faulted. Whether the visit found anything is
        # unknown, not negative — the measurements never landed.
        status = "failed"
    elif candidates:
        # Candidates were produced and none of them earned a measurement,
        # either because no rebench ran or because every one was skipped.
        status = "degraded"
    else:
        # Nothing was built and nothing was measured. A visit holding only
        # skipped rebench rows lands here too: a skip is a record of work
        # declined, not of work done.
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
