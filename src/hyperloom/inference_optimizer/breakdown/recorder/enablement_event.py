# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The SBD V6 ``enablement`` event: the repair lane, recorded as it runs.

A combo that cannot boot -- or boots and fails its accuracy eval -- opens a
lane that dispatches an authoring specialist, applies its patch, optionally
compiles a component, benches the result, and either lands or rearms against
the next gap. Each round is one row, keyed by the specialist task that authored
it, so the sequence stays legible: counters cannot say which round landed the
fix, and the lane's own ``launch_log`` is replaced on every advance.

The event covers the whole session rather than a phase, because the lane does:
a combo that cannot boot never leaves PRELUDE, and the round repairing it is
judged in FRAMEWORK_AGENT. Nothing here holds a recorder object for the same
reason -- the facts are produced in six modules on different ticks, so every
entry point below is a module-level function that opens the event idempotently
and records its own row.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

from .event_fields import (
    as_dict as _as_dict,
    as_list as _as_list,
    clip as _clip,
    float_or_none as _float_or_none,
    now_iso_seconds as _now,
    text_or_none as _text_or_none,
)
from .event_ids import event_id
from .event_rows import rows_for_event, sort_rows, wire_rows
from .event_sink import EventSink, make_sink
from .event_timeline import finish_event, open_event

log = logging.getLogger(__name__)

EVENT_TYPE = "enablement"
EVENT_KIND = "enablement"

EVENT_COMPONENT = "enablement"

#: A literal, not a read of ``state.phase``: a lane's rounds are ruled in a
#: different phase from its trigger, so a phase-scoped id would halve it.
EVENT_PHASE = "enablement"

#: Also a literal: a lane opened in cycle 0 can still be rearming in cycle 4.
EVENT_CYCLE = 0

PRODUCER = "orchestrator"

SECTION_EVENT = "enablement_event"

#: One row per authoring round, keyed by its specialist task. The dispatch
#: writes the gap and the rearm merges in the outcome, so a round killed
#: between the two survives as a dispatch with no outcome.
SECTION_ATTEMPT = "enablement_attempt"

SECTION_BUILD = "enablement_build"

#: One row per revalidation window, keyed by its generation. Eval-origin only:
#: a KEEP there is provisional until a genuine baseline re-measures accuracy.
SECTION_REVALIDATION = "enablement_revalidation"

#: One row per unclassifiable launch failure, keyed by the log's digest --
#: such a log dispatches nothing, so it would leave no other trace.
SECTION_HUMAN_REVIEW = "enablement_human_review"

#: The lane was opened by a baseline that could not launch at all.
ORIGIN_BOOT = "boot"

#: The lane was opened by a baseline that launched and failed its accuracy eval.
ORIGIN_EVAL = "eval"

# Round statuses, in the integrate gate's own words. ``advanced``: the patch
# did not make the combo runnable but the boot now stops deeper, and that is
# scored as progress.
ROUND_KEPT = "kept"
ROUND_ADVANCED = "advanced"
ROUND_REVERTED = "reverted"

# ``stalled`` is the terminal that stops the run; ``pending`` is a lane still
# working when the session ended, which is not a lane that gave up.
OUTCOME_SUCCEEDED = "succeeded"
OUTCOME_STALLED = "stalled"
OUTCOME_PENDING = "pending"

# A lane that ran rounds without landing one is degraded rather than failed:
# the run continued and those rounds may still have moved the boot forward.
STATUS_SUCCEEDED = "succeeded"
STATUS_FAILED = "failed"
STATUS_DEGRADED = "degraded"
STATUS_SKIPPED = "skipped"

#: Logs are clipped from the front: the tail is the part that names the gap.
MAX_LOG_EXCERPT_CHARS = 2000

#: Attempt runtimes are capped at five in state; the same bound applies here.
MAX_RUNTIME_RECORDS = 5


def enablement_event_id() -> str:
    """Build the enablement lane's event id, ``enablement:0:enablement``."""
    return event_id(EVENT_PHASE, EVENT_CYCLE, EVENT_COMPONENT)


def _sink() -> EventSink | None:
    """The sink rows are written through; ``None`` when no session is bound."""
    try:
        from ...session.session_binding import bound_session_or_none

        if bound_session_or_none() is None:
            return None
        return make_sink(enablement_event_id(), producer=PRODUCER)
    except Exception:  # noqa: BLE001 — the lane outranks its own record
        log.debug("enablement event: cannot resolve a sink", exc_info=True)
        return None


def _open(*, mode: str = "", origin: str = "", start_time: str = "") -> int | None:
    """Put the lane on the timeline, once, however many callers ask.

    :func:`open_event` returns an earlier open's sequence rather than writing a
    second shell, so no entry point below owns the lane's lifetime. ``None``
    means the shell write failed; the caller still records, because finalize
    recovers the event from the fragments.
    """
    shell: dict[str, Any] = {}
    if mode:
        shell["mode"] = str(mode)
    if origin:
        shell["origin"] = str(origin)
    if shell:
        # Onto the fragment too: assembly rebuilds ``ext`` from fragments,
        # so a shell-only field dies at the first close.
        sink = _sink()
        if sink is not None:
            sink.record(SECTION_EVENT, dict(shell))
    return open_event(
        event_type=EVENT_TYPE,
        event=enablement_event_id(),
        event_section=SECTION_EVENT,
        producer=PRODUCER,
        kind=EVENT_KIND,
        start_time=start_time or _now(),
        ext=shell,
    )


def record_trigger(
    *,
    origin: str,
    mode: str,
    kind: str = "",
    evidence: Any = "",
    observed_accuracy: Any = None,
    accuracy_floor: Any = None,
    observed_task: Any = None,
    observed_metric: Any = None,
    eval_contract_fingerprint: Any = None,
    probe_config_path: Any = None,
) -> None:
    """Open the lane and record what opened it. Never raises.

    The first call wins the ``trigger`` block: a lane reopened by a second
    failure is the same lane on the same gap, and letting the newest failure
    overwrite the oldest is how an eval-less re-baseline downgrades a measured
    ``accuracy_below_floor`` to an empty ``accuracy_unavailable``. The accuracy
    fields are eval-origin only.
    """
    try:
        sink = _sink()
        if sink is None:
            return
        _open(mode=mode, origin=origin)
        if _recorded_trigger():
            # Keyed rows deep-merge, so a second trigger would overwrite the
            # first field by field; mode and origin are already on the shell.
            return
        trigger: dict[str, Any] = {
            "kind": str(kind or ""),
            "recorded_at": _now(),
            "evidence_excerpt": _tail(evidence),
            "observed_accuracy": _float_or_none(observed_accuracy),
            "accuracy_floor": _float_or_none(accuracy_floor),
            "observed_task": _text_or_none(observed_task),
            "observed_metric": _text_or_none(observed_metric),
            "eval_contract_fingerprint": _text_or_none(eval_contract_fingerprint),
            "probe_config_path": _text_or_none(probe_config_path),
        }
        sink.record(
            SECTION_EVENT,
            {
                "mode": str(mode or ""),
                "origin": str(origin or ""),
                # Under its own key so a later write cannot flatten it.
                "trigger": trigger,
            },
        )
    except Exception:  # noqa: BLE001 — the lane outranks its own record
        log.debug("enablement event: trigger record failed", exc_info=True)


def record_dispatch(
    *,
    task_id: str,
    attempt: int,
    failure_kind: str = "",
    launch_log: Any = "",
    candidate_refs: Any = None,
    mode: str = "",
    origin: str = "",
) -> None:
    """Record an authoring round at the moment it is dispatched. Never raises.

    Written before the specialist has done anything, because what the round was
    asked to repair is lost once the lane moves on: ``launch_log`` is replaced
    by every advance, and the classified kind only ever existed in the params
    this dispatch built.
    """
    try:
        sink = _sink()
        if sink is None:
            return
        _open(mode=mode, origin=origin)
        sink.record(
            SECTION_ATTEMPT,
            {
                "attempt": int(attempt or 0),
                "task_id": str(task_id or ""),
                "failure_kind": str(failure_kind or ""),
                "dispatched_at": _now(),
                "launch_log_excerpt": _tail(launch_log),
                "candidate_refs": [str(ref) for ref in _as_list(candidate_refs)],
            },
            row_type="attempt",
            natural_ids=_row_id(task_id, attempt),
        )
    except Exception:  # noqa: BLE001
        log.debug("enablement event: dispatch record failed", exc_info=True)


def record_round(
    *,
    task_id: str,
    attempt: int,
    result: Mapping[str, Any] | None,
    stall_streak: int,
    succeeded: bool,
    validation_pending: bool = False,
) -> None:
    """Record how an authoring round settled. Never raises.

    Merges onto the row :func:`record_dispatch` opened; a round nothing
    dispatched opens its own row here, being still a round the lane spent.
    ``validation_pending`` means an eval-origin KEEP opened a revalidation
    window instead of landing.
    """
    try:
        sink = _sink()
        if sink is None:
            return
        _open()
        res = _as_dict(result)
        status = str(res.get("status") or "")
        advanced = status == ROUND_ADVANCED or bool(res.get("advanced"))
        row: dict[str, Any] = {
            "attempt": int(attempt or 0),
            "task_id": str(task_id or ""),
            "settled_at": _now(),
            "status": status,
            "advanced": advanced,
            "reason": str(res.get("reason") or ""),
            "landed": bool(succeeded),
            "validation_pending": bool(validation_pending),
            "stall_streak_after": int(stall_streak or 0),
            "patches_applied": [str(p) for p in _as_list(res.get("patches_applied")) if str(p)],
            "artifacts_applied": [
                _artifact_row(a) for a in _as_list(res.get("artifacts_applied")) if isinstance(a, dict)
            ],
            "setup_commands_applied": [str(c) for c in _as_list(res.get("setup_commands_applied")) if str(c)],
            "patches_dropped_by_grounding": [str(d) for d in _as_list(res.get("patches_dropped_by_grounding"))[:8]],
            "patches_span_multiple_roots": bool(res.get("patches_span_multiple_roots")),
            "framework_root": _text_or_none(res.get("framework_root")),
            "accepted_config_path": _text_or_none(res.get("enablement_accepted_config_path")),
            "effective_config": _config_row(res.get("enablement_effective_config")),
            "stack_action": _stack_action_row(res.get("enablement_kept_stack_action")),
            "runtime": _runtime_row(res.get("enablement_active_runtime")),
            "localization_manifest": _as_dict(res.get("enablement_localization_manifest")) or None,
            # The next round's gap, on the round that revealed it.
            "next_launch_log_excerpt": _tail(res.get("enablement_launch_log")),
        }
        sink.record(SECTION_ATTEMPT, row, row_type="attempt", natural_ids=_row_id(task_id, attempt))
    except Exception:  # noqa: BLE001
        log.debug("enablement event: round record failed", exc_info=True)


def record_human_review(*, digest: str, failure_kind: str, reason: str = "", signature: Any = None) -> None:
    """Record a launch failure the lane could not act on, keyed by ``digest``.

    Never raises. Such a log dispatches nothing, so it leaves no round behind.
    """
    try:
        sink = _sink()
        if sink is None or not str(digest or "").strip():
            return
        _open()
        sink.record(
            SECTION_HUMAN_REVIEW,
            {
                "digest": str(digest),
                "failure_kind": str(failure_kind or ""),
                "reason": _clip(reason, 400),
                "signature": _as_dict(signature) or None,
                "recorded_at": _now(),
            },
            row_type="human_review",
            natural_ids=str(digest),
        )
    except Exception:  # noqa: BLE001
        log.debug("enablement event: human-review record failed", exc_info=True)


def record_build(*, task_id: str, entry: Mapping[str, Any] | None = None, novelty_key: str = "") -> None:
    """Record one ``BuildResult.to_state()`` entry, keyed by ``task_id``. Never raises."""
    try:
        sink = _sink()
        if sink is None or not str(task_id or "").strip():
            return
        _open()
        manifest = _as_dict(entry)
        action = _as_dict(manifest.get("action"))
        installed = _as_dict(manifest.get("installed_versions"))
        row: dict[str, Any] = {
            "task_id": str(task_id),
            "recorded_at": _now(),
            "component": str(action.get("component") or manifest.get("component") or ""),
            "ref": str(
                installed.get("aiter_ref")
                or installed.get("vllm_ref")
                or installed.get("sgl_kernel_ref")
                or action.get("ref")
                or ""
            ),
            "gpu_arch": str(installed.get("arch") or action.get("gpu_arch") or ""),
            "max_jobs": int(action.get("max_jobs") or 0),
            "installed_versions": {str(k): str(v) for k, v in installed.items()},
            "build_probes": [str(p) for p in _as_list(manifest.get("build_probes"))[:8]],
            "build_log_path": _text_or_none(manifest.get("build_log_path")),
            "attempt_root": _text_or_none(manifest.get("attempt_root")),
        }
        if novelty_key:
            row["novelty_key"] = str(novelty_key)
        # ``ok`` separates a build that ran from a verdict-less sentinel,
        # whose row still belongs on the timeline.
        if manifest.get("ok") is not None:
            row["ok"] = bool(manifest.get("ok"))
            row["failure_class"] = str(manifest.get("failure_class") or "ok")
            row["failure_summary"] = _clip(manifest.get("failure_summary"), 1000)
        sink.record(SECTION_BUILD, row, row_type="build", natural_ids=str(task_id))
    except Exception:  # noqa: BLE001
        log.debug("enablement event: build record failed", exc_info=True)


def record_revalidation(
    *,
    generation: int,
    task_id: str = "",
    config_path: str = "",
    reason: str = "",
) -> None:
    """Record a revalidation window opening, keyed by ``generation``. Never raises."""
    try:
        sink = _sink()
        if sink is None:
            return
        _open()
        row: dict[str, Any] = {"generation": int(generation or 0), "opened_at": _now()}
        if task_id:
            row["task_id"] = str(task_id)
        if config_path:
            row["config_path"] = str(config_path)
        if reason:
            row["reason"] = str(reason)
        sink.record(
            SECTION_REVALIDATION,
            row,
            row_type="revalidation",
            natural_ids=str(int(generation or 0)),
        )
    except Exception:  # noqa: BLE001
        log.debug("enablement event: revalidation record failed", exc_info=True)


def record_revalidation_outcome(
    *,
    generation: int,
    promoted: bool,
    task_id: str = "",
    accuracy: Any = None,
    accuracy_floor: Any = None,
    error_class: str = "",
    reason: str = "",
) -> None:
    """Record how a revalidation window closed. Never raises.

    ``error_class`` is set only when the baseline failed rather than measuring
    under the floor; a window the run stopped is not a window that failed.
    """
    try:
        sink = _sink()
        if sink is None:
            return
        _open()
        row: dict[str, Any] = {
            "generation": int(generation or 0),
            "closed_at": _now(),
            "promoted": bool(promoted),
            "accuracy": _float_or_none(accuracy),
            "accuracy_floor": _float_or_none(accuracy_floor),
        }
        if task_id:
            row["task_id"] = str(task_id)
        if error_class:
            row["error_class"] = str(error_class)
        if reason:
            row["reason"] = str(reason)
        sink.record(
            SECTION_REVALIDATION,
            row,
            row_type="revalidation",
            natural_ids=str(int(generation or 0)),
        )
    except Exception:  # noqa: BLE001
        log.debug("enablement event: revalidation outcome record failed", exc_info=True)


def finish(
    *,
    outcome: str,
    reason: str = "",
    kept_patches: Any = None,
    kept_artifacts: Any = None,
    setup_commands: Any = None,
    accepted_config: Any = None,
    accepted_config_path: str = "",
    setting_script: str = "",
    active_runtime: Any = None,
    attempt_runtimes: Any = None,
    framework_root: str = "",
    stall_streak: int = 0,
) -> None:
    """Close the lane on the terminal it reached. Never raises.

    Called where the terminal is *set*, not where it is later observed. A lane
    still working when the session ended is not closed here at all: finalize
    recovers it as ``interrupted``, because nothing judged it. ``outcome`` is
    :data:`OUTCOME_SUCCEEDED` or :data:`OUTCOME_STALLED`.
    """
    try:
        sink = _sink()
        if sink is None:
            return
        sequence = _open()
        end_time = _now()
        settled = str(outcome or "").strip().lower() or OUTCOME_PENDING
        active = _runtime_row(active_runtime)
        active_root = str(active.get("venv_root") or "") if active else ""
        result: dict[str, Any] = {
            "outcome": settled,
            "reason": str(reason or ""),
            "stall_streak": int(stall_streak or 0),
            "kept_patches": [str(p) for p in _as_list(kept_patches) if str(p)],
            "kept_artifacts": [_artifact_row(a) for a in _as_list(kept_artifacts) if isinstance(a, dict)],
            "setup_commands": [str(c) for c in _as_list(setup_commands) if str(c)],
            "accepted_config": _config_row(accepted_config),
            "accepted_config_path": _text_or_none(accepted_config_path),
            "setting_script": _text_or_none(setting_script),
            "framework_root": _text_or_none(framework_root),
            "active_runtime": active or None,
            "attempt_runtimes": [
                _runtime_row(runtime, promoted=str(_as_dict(runtime).get("venv_root") or "") == active_root)
                for runtime in _as_list(attempt_runtimes)[-MAX_RUNTIME_RECORDS:]
                if isinstance(runtime, Mapping)
            ],
        }
        sink.record(SECTION_EVENT, {"result": result, "end_time": end_time})

        from .assembler import event_parts

        ext, derived = assemble_enablement_ext(event_parts(ENABLEMENT_EVENT_SECTIONS), event=enablement_event_id())
        finish_event(
            event_type=EVENT_TYPE,
            event=enablement_event_id(),
            sequence=sequence,
            status=derived or _status_for(settled, attempts=0),
            ext=ext,
            kind=EVENT_KIND,
            start_time=_start_time(),
            end_time=end_time,
        )
    except Exception:  # noqa: BLE001
        log.debug("enablement event: finish failed", exc_info=True)


#: Every section the enablement event assembles from. Duplicated from the
#: assembler so :func:`finish` can read its own parts without an import cycle.
ENABLEMENT_EVENT_SECTIONS: tuple[str, ...] = (
    SECTION_EVENT,
    SECTION_ATTEMPT,
    SECTION_BUILD,
    SECTION_REVALIDATION,
    SECTION_HUMAN_REVIEW,
)


def assemble_enablement_ext(
    parts: Mapping[str, list[dict[str, Any]]],
    *,
    event: str,
) -> tuple[dict[str, Any], str]:
    """Assemble the enablement event's ``ext`` from its recorded rows.

    The returned status is empty when no write has closed the event, which
    leaves the caller's own reading standing.
    """
    header = _header(rows_for_event(parts.get(SECTION_EVENT) or [], event))
    attempts = wire_rows(
        sort_rows(rows_for_event(parts.get(SECTION_ATTEMPT) or [], event), keys=("attempt", "dispatched_at")),
        drop=("event_id",),
    )
    builds = wire_rows(
        sort_rows(rows_for_event(parts.get(SECTION_BUILD) or [], event), keys=("recorded_at", "task_id")),
        drop=("event_id",),
    )
    revalidations = wire_rows(
        sort_rows(rows_for_event(parts.get(SECTION_REVALIDATION) or [], event), keys=("generation",)),
        drop=("event_id",),
    )
    human_review = wire_rows(
        sort_rows(rows_for_event(parts.get(SECTION_HUMAN_REVIEW) or [], event), keys=("recorded_at", "digest")),
        drop=("event_id",),
    )
    result = _as_dict(header.get("result"))
    ext: dict[str, Any] = {
        "mode": str(header.get("mode") or ""),
        "origin": str(header.get("origin") or ""),
        # Recorded, not inferred: an event that exists at all was engaged.
        "engaged": True,
        "trigger": _as_dict(header.get("trigger")) or None,
        "attempts": {
            "count": len(attempts),
            # Rounds with a verdict; ``count`` includes one still in flight.
            "settled": sum(1 for row in attempts if row.get("status")),
            "landed": sum(1 for row in attempts if str(row.get("status") or "") == ROUND_KEPT),
            "advanced": sum(1 for row in attempts if row.get("advanced")),
            "rows": attempts,
        },
        "builds": {
            "count": len(builds),
            "failed": sum(1 for row in builds if row.get("ok") is False),
            "rows": builds,
        },
        "revalidations": {
            "count": len(revalidations),
            "promoted": sum(1 for row in revalidations if row.get("promoted")),
            "rows": revalidations,
        },
        "human_review": {"count": len(human_review), "rows": human_review},
        "result": result or None,
    }
    status = _status_for(str(result.get("outcome") or ""), attempts=len(attempts)) if result else ""
    return ext, status


def _status_for(outcome: str, *, attempts: int) -> str:
    """The status the event reports: degraded when the lane closed on neither
    terminal, skipped when it ran no round at all."""
    settled = str(outcome or "").strip().lower()
    if settled == OUTCOME_SUCCEEDED:
        return STATUS_SUCCEEDED
    if settled == OUTCOME_STALLED:
        return STATUS_FAILED
    return STATUS_DEGRADED if attempts else STATUS_SKIPPED


def _header(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Fold the event-level fragments into one header, rather than take
    ``rows[0]``, so a spool holding two does not drop the second."""
    header: dict[str, Any] = {}
    for row in rows:
        if isinstance(row, Mapping):
            header.update({key: value for key, value in row.items() if value not in (None, "")})
    return header


def _recorded_trigger() -> bool:
    """Whether the lane already recorded what opened it.

    Read from the spool, not memory: the trigger and the failure that would
    overwrite it come from different ticks and, on resume, different processes.
    """
    try:
        from .assembler import event_parts

        rows = rows_for_event(event_parts((SECTION_EVENT,)).get(SECTION_EVENT) or [], enablement_event_id())
        return any(_as_dict(row.get("trigger")) for row in rows)
    except Exception:  # noqa: BLE001 — a spool we cannot read is not a trigger we have
        log.debug("enablement event: cannot read back the trigger", exc_info=True)
        return False


def _start_time() -> str:
    """The start time the open write stored, read back for the close: nothing
    here holds the lane, so the fragment is its only durable identity."""
    try:
        from .assembler import event_parts

        rows = rows_for_event(event_parts((SECTION_EVENT,)).get(SECTION_EVENT) or [], enablement_event_id())
        for row in rows:
            recorded = str(row.get("start_time") or "")
            if recorded:
                return recorded
    except Exception:  # noqa: BLE001 — a missing start time is not worth failing the close
        log.debug("enablement event: cannot read back the start time", exc_info=True)
    return ""


def _row_id(task_id: Any, attempt: Any) -> str:
    """The natural id of an authoring round's row: the specialist task id the
    dispatch and rearm both name it by, or its ordinal for a synthesised round.
    """
    task = str(task_id or "").strip()
    if task:
        return task
    return f"round{int(attempt or 0)}"


def _tail(value: Any) -> str | None:
    """The tail of a log, which is the part that names the gap."""
    text = str(value or "")
    if not text.strip():
        return None
    return text[-MAX_LOG_EXCERPT_CHARS:]


def _artifact_row(artifact: Mapping[str, Any]) -> dict[str, Any]:
    """Project one installed artifact, dropping the ``backup`` / ``source`` /
    ``existed`` bookkeeping, which is not a fact about the repair."""
    row = _as_dict(artifact)
    return {
        "target": str(row.get("target") or ""),
        "rel_target": str(row.get("rel_target") or ""),
        "kind": str(row.get("kind") or ""),
    }


def _config_row(config: Any) -> dict[str, Any] | None:
    """Project the env/arg layers a bench ran with."""
    row = _as_dict(config)
    if not row:
        return None
    return {
        "extra_server_args": str(row.get("extra_server_args") or ""),
        "extra_envs": {str(k): str(v) for k, v in _as_dict(row.get("extra_envs")).items()},
    }


def _stack_action_row(action: Any) -> dict[str, Any] | None:
    """Project the stack action a round acquired its capability through."""
    row = _as_dict(action)
    if not row:
        return None
    return {
        "kind": str(row.get("kind") or ""),
        "framework": str(row.get("framework") or ""),
        "capability": str(row.get("capability") or ""),
        "acquisition_method": str(row.get("acquisition_method") or ""),
        "repo_url": str(row.get("repo_url") or ""),
        "ref": str(row.get("ref") or ""),
        "index_url": str(row.get("index_url") or ""),
        "reason": str(row.get("reason") or ""),
    }


def _runtime_row(runtime: Any, *, promoted: bool = True) -> dict[str, Any]:
    """Project one provisioned framework runtime."""
    row = _as_dict(runtime)
    if not row:
        return {}
    return {
        "venv_root": str(row.get("venv_root") or ""),
        "bin_path": str(row.get("bin_path") or ""),
        "python_path": str(row.get("python_path") or ""),
        "installed_versions": {str(k): str(v) for k, v in _as_dict(row.get("installed_versions")).items()},
        "promoted": bool(promoted),
    }


__all__ = [
    "ENABLEMENT_EVENT_SECTIONS",
    "EVENT_COMPONENT",
    "EVENT_CYCLE",
    "EVENT_KIND",
    "EVENT_PHASE",
    "EVENT_TYPE",
    "MAX_LOG_EXCERPT_CHARS",
    "MAX_RUNTIME_RECORDS",
    "ORIGIN_BOOT",
    "ORIGIN_EVAL",
    "OUTCOME_PENDING",
    "OUTCOME_STALLED",
    "OUTCOME_SUCCEEDED",
    "PRODUCER",
    "ROUND_ADVANCED",
    "ROUND_KEPT",
    "ROUND_REVERTED",
    "SECTION_ATTEMPT",
    "SECTION_BUILD",
    "SECTION_EVENT",
    "SECTION_HUMAN_REVIEW",
    "SECTION_REVALIDATION",
    "STATUS_DEGRADED",
    "STATUS_FAILED",
    "STATUS_SKIPPED",
    "STATUS_SUCCEEDED",
    "assemble_enablement_ext",
    "enablement_event_id",
    "finish",
    "record_build",
    "record_dispatch",
    "record_human_review",
    "record_revalidation",
    "record_revalidation_outcome",
    "record_round",
    "record_trigger",
]
