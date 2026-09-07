# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The SBD V6 ``enablement`` event: the repair lane, recorded as it runs.

Enablement is the one subsystem with a full action lifecycle and no record of
it. A (model, backend) combo that cannot boot -- or boots and fails its
accuracy eval -- opens a lane that dispatches an authoring specialist, applies
its patch, optionally compiles a component, benches the result, and either
lands or rearms against the next gap. That is a sequence of dispatched actions
with outcomes, which is what the timeline is for; until now it was published as
a flat projection of ``SharedState.enablement`` at export.

The projection lost the lane's history, which is the only part worth reading.

**Rounds were folded into counters.** ``attempts`` said how many rounds ran and
``kept_patches`` said which patches survived all of them, so a five-round lane
where round 3 landed the fix and rounds 4-5 chased a deeper gap read exactly
like a five-round lane that never landed anything but accumulated patches. Each
round's own failure kind, the log it was dispatched against, the status it
settled on and the products it contributed are now one row per round, keyed by
the specialist task that authored it, so the sequence is legible.

**The trigger was reconstructed, and reconstructed wrongly.** ``origin`` was
derived as ``"eval" if origin == "eval" or baseline_eval_kind else "boot"``,
because ``origin`` is cleared on success while ``baseline_eval_kind`` is not --
a disjunction over two fields with different lifetimes, standing in for a fact
nobody recorded. It is now recorded when the lane opens, and it stays what it
was.

**``launch_log`` describes only the last round.** The lane replaces it on every
advance, so the exported excerpt is the gap the *newest* round faced and the
export presented it as the reason the lane ran at all. The opening trigger and
each round's own log are now separate facts.

**``failure_kind`` was always empty.** The collector read
``state["enablement"]["failure_kind"]``; ``EnablementRound`` has no such field.
The classified kind lives in the specialist params the dispatch builds, which
is where it is now recorded from.

The event covers the whole session because the lane does. It is not scoped to a
phase: the pump runs on every coordinator tick precisely because a combo that
cannot boot never leaves PRELUDE, and the round that repairs it is judged in
FRAMEWORK_AGENT. Scoping the event by ``state.phase`` would split one lane into
a PRELUDE half holding the trigger and a FRAMEWORK_AGENT half holding the
outcome, neither of which is a lane.

For the same reason nothing here holds a recorder object. The facts are
produced in six modules on different ticks -- the writeback that stores the
trigger, the pump that dispatches, the integrate gate that rules, the build
executor, the revalidation enqueue, the promote -- and threading one object
through all of them would make the lane's record depend on the call graph that
happens to reach it. Every entry point below is a module-level function that
opens the event idempotently and records its own row, so a fact lands from
wherever it is produced.
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

#: The component segment of the enablement event id.
EVENT_COMPONENT = "enablement"

#: The phase segment. A literal, not a read of ``state.phase``: the lane is
#: driven by a phase-independent pump and its rounds are ruled in a different
#: phase from the one its trigger was recorded in, so a phase-scoped id would
#: cut one lane into halves that are each missing the other's half of the story.
EVENT_PHASE = "enablement"

#: The macro-cycle segment. Also a literal, and for the same reason: a lane
#: opened in cycle 0 can still be rearming in cycle 4, and there is one lane.
EVENT_CYCLE = 0

PRODUCER = "orchestrator"

#: The event-level section: the admitted mode, the trigger that opened the
#: lane, and the terminal reading it settled on.
SECTION_EVENT = "enablement_event"

#: One row per authoring round, keyed by the specialist task that authored it.
#: The dispatch writes what the round was asked to repair and the rearm merges
#: in what it settled on, so a round killed between the two is on the timeline
#: as a dispatched round with no outcome rather than as nothing at all.
SECTION_ATTEMPT = "enablement_attempt"

#: One row per targeted build, keyed by its task id.
SECTION_BUILD = "enablement_build"

#: One row per revalidation window, keyed by its generation. Eval-origin only:
#: a KEEP there is provisional until a genuine baseline re-measures accuracy.
SECTION_REVALIDATION = "enablement_revalidation"

#: One row per distinct launch failure the lane could not classify, keyed by
#: the log's digest. These are the rounds that never happened: a non-blank log
#: that matches no actionable signature dispatches nothing and is filed for a
#: human, and the projection published only how many there had been.
SECTION_HUMAN_REVIEW = "enablement_human_review"

#: The lane was opened by a baseline that could not launch at all.
ORIGIN_BOOT = "boot"

#: The lane was opened by a baseline that launched and failed its accuracy eval.
ORIGIN_EVAL = "eval"

# The statuses an authoring round settles on, in the integrate gate's own
# words. ``advanced`` is the one worth naming: the patch did not make the combo
# runnable, but it cleared the gap it targeted and the boot now stops somewhere
# deeper, which is progress and is scored as progress.
ROUND_KEPT = "kept"
ROUND_ADVANCED = "advanced"
ROUND_REVERTED = "reverted"

# The lane's own outcome. ``stalled`` is the terminal that stops the run;
# ``pending`` is a lane that was still working when the session ended, which is
# a different thing from one that gave up.
OUTCOME_SUCCEEDED = "succeeded"
OUTCOME_STALLED = "stalled"
OUTCOME_PENDING = "pending"

# Event statuses. A lane that stalled is a failure of the lane, not of the
# recording, and a lane that ran rounds without landing one is degraded rather
# than failed: the run continued and the rounds it did land may still have
# moved the boot forward.
STATUS_SUCCEEDED = "succeeded"
STATUS_FAILED = "failed"
STATUS_DEGRADED = "degraded"
STATUS_SKIPPED = "skipped"

#: Trigger and per-round logs are tracebacks and eval transcripts. The tail is
#: the part that names the gap, so they are clipped from the front and the
#: budget matches what the projection published.
MAX_LOG_EXCERPT_CHARS = 2000

#: Attempt runtimes are capped at five in state, so the same bound applies here
#: rather than letting the event hold a history state no longer has.
MAX_RUNTIME_RECORDS = 5


def enablement_event_id() -> str:
    """Build the enablement lane's event id.

    Returns:
        str: ``enablement:0:enablement``. Both leading segments are literals;
        see :data:`EVENT_PHASE`.
    """
    return event_id(EVENT_PHASE, EVENT_CYCLE, EVENT_COMPONENT)


def _sink() -> EventSink | None:
    """The sink every row here is written through, or ``None`` with no session.

    Returns:
        EventSink | None: The bound session's sink, or ``None`` when nothing is
        bound -- a unit test driving the lane directly, or a resume before the
        session scope is entered. Recording is best-effort either way.
    """
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

    :func:`open_event` returns the sequence an earlier open took rather than
    writing a second shell, which is what lets every entry point below call
    this without any of them owning the lane's lifetime.

    Args:
        mode (str): The admitted ``--enablement`` mode, when known.
        origin (str): :data:`ORIGIN_BOOT` or :data:`ORIGIN_EVAL`, when known.
        start_time (str): When the lane opened; defaults to now.

    Returns:
        int | None: The storage sequence to close with, or ``None`` when the
        shell write failed. A caller that gets ``None`` still records: the
        fragments land, and finalize recovers the event from them.
    """
    shell: dict[str, Any] = {}
    if mode:
        shell["mode"] = str(mode)
    if origin:
        shell["origin"] = str(origin)
    if shell:
        # Onto the fragment as well as the shell. Assembly rebuilds ``ext``
        # from the fragments, so a field that only ever rode on the shell is
        # dropped the moment anything closes the event -- and the shell write
        # is skipped entirely on every open after the first.
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

    The first call wins the ``trigger`` block. A lane reopened by a second
    failure of the same kind -- a re-baseline that fails its eval again, a
    revalidation that comes back under the floor -- is the same lane still
    working on the same gap, and its later evidence belongs to the round that
    faced it, not to the trigger. What the projection did instead was let the
    newest failure overwrite the oldest, which is how an eval-less re-baseline
    could downgrade a measured ``accuracy_below_floor`` to an empty
    ``accuracy_unavailable``.

    Args:
        origin (str): :data:`ORIGIN_BOOT` or :data:`ORIGIN_EVAL`.
        mode (str): The admitted ``--enablement`` mode.
        kind (str): The trigger's failure kind -- the eval kind for an
            eval-origin lane, the classified boot signature for a boot-origin
            one.
        evidence (Any): The launch log or eval transcript the trigger was read
            from; clipped to its tail.
        observed_accuracy (Any): The accuracy measured, eval-origin only.
        accuracy_floor (Any): The floor it was graded against.
        observed_task (Any): The eval task.
        observed_metric (Any): The eval metric.
        eval_contract_fingerprint (Any): The contract the revalidation must
            reproduce.
        probe_config_path (Any): The config the failing eval ran.
    """
    try:
        sink = _sink()
        if sink is None:
            return
        _open(mode=mode, origin=origin)
        if _recorded_trigger():
            # Repeated calls on one keyed row deep-merge, so a second trigger
            # would overwrite the first field by field rather than being
            # ignored -- which is the projection's failure mode reproduced
            # inside the recorder. The mode and origin are already on the
            # shell, so there is nothing left to write.
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
                # Merged under a key of its own so a later write that carries
                # ``mode`` or the outcome cannot flatten the trigger, and so
                # the first trigger is the one assembly reads.
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
    asked to repair is a fact of the dispatch and is lost once the lane moves
    on: ``launch_log`` is replaced by every advance, and the classified kind
    only ever existed in the params this dispatch built.

    Args:
        task_id (str): The specialist task id, which keys the round's row.
        attempt (int): The round's ordinal, 1-based.
        failure_kind (str): The signature the round was pointed at.
        launch_log (Any): The log it was dispatched against; clipped.
        candidate_refs (Any): The candidate refs the mandate carried.
        mode (str): The admitted mode, for a lane whose trigger went unrecorded.
        origin (str): The lane's origin, likewise.
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

    Merges onto the row :func:`record_dispatch` opened, so the round reads as
    one thing that was asked to repair a named gap and then either did or did
    not. A round nothing dispatched -- a build routed into the lane, a round
    the pump found finished without a rearm -- opens its own row here, because
    it is still a round the lane spent.

    Args:
        task_id (str): The specialist task id whose row this settles.
        attempt (int): The round's ordinal, for a row keyed without a task id.
        result (Mapping[str, Any] | None): The ``integrate_patch`` result.
        stall_streak (int): The streak after this round was scored.
        succeeded (bool): Whether the lane reached terminal success on it.
        validation_pending (bool): Whether an eval-origin KEEP opened a
            revalidation window instead of landing.
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
            # The gap the *next* round will face, recorded on the round that
            # revealed it. This is the fact the projection's single
            # ``launch_log`` overwrote on every advance.
            "next_launch_log_excerpt": _tail(res.get("enablement_launch_log")),
        }
        sink.record(SECTION_ATTEMPT, row, row_type="attempt", natural_ids=_row_id(task_id, attempt))
    except Exception:  # noqa: BLE001
        log.debug("enablement event: round record failed", exc_info=True)


def record_human_review(*, digest: str, failure_kind: str, reason: str = "", signature: Any = None) -> None:
    """Record a launch failure the lane could not act on. Never raises.

    A non-blank log that classifies to no actionable signature dispatches
    nothing, so it leaves no round behind -- and a lane that spent a whole
    session declining to dispatch reads, from counters alone, exactly like a
    lane that was never triggered. One row per distinct log, keyed by the same
    digest the lane dedupes on.

    Args:
        digest (str): The log's digest, which keys the row.
        failure_kind (str): The kind it classified to, ``UNKNOWN`` in practice.
        reason (str): Why it was filed for a human rather than dispatched.
        signature (Any): The classified signature, as the classifier stated it.
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
    """Record one targeted build the lane ran. Never raises.

    Args:
        task_id (str): The build task id, which keys the row.
        entry (Mapping[str, Any] | None): The ``BuildResult.to_state()`` entry.
        novelty_key (str): The novelty key the enqueue was idempotent on.
    """
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
        # ``ok`` distinguishes a build that ran from a routing sentinel, which
        # carries no verdict; the projection filtered sentinels out by the same
        # test and so published nothing about a build that was only enqueued.
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
    """Record a revalidation window opening. Never raises.

    Args:
        generation (int): The window's generation, which keys the row.
        task_id (str): The baseline task id enqueued to revalidate.
        config_path (str): The config it will run -- the accepted one from the
            KEEP'd bench, or the original probe config as a fallback.
        reason (str): Why the window opened, when it was not a KEEP.
    """
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

    Args:
        generation (int): The window's generation, whose row this settles.
        promoted (bool): Whether a genuine baseline promoted and cleared it.
        task_id (str): The baseline task that answered.
        accuracy (Any): The accuracy it measured.
        accuracy_floor (Any): The floor it was graded against.
        error_class (str): The failure class, when it failed rather than
            measuring under the floor.
        reason (str): Why it did not promote, in the caller's words -- a
            window the run stopped is not a window that failed.
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

    Called where the terminal is *set*, not where it is later observed: success
    on the KEEP that landed or the revalidation that promoted, ``stalled`` on
    the round that hit the cap. A lane that was still working when the session
    ended is not closed here at all -- finalize recovers it as ``interrupted``,
    which is the honest reading, because nothing judged it.

    Args:
        outcome (str): :data:`OUTCOME_SUCCEEDED` or :data:`OUTCOME_STALLED`.
        reason (str): The stop reason or the terminal's own words.
        kept_patches (Any): The patches the lane landed, in order.
        kept_artifacts (Any): The artifacts it installed.
        setup_commands (Any): The env-setup commands it replays.
        accepted_config (Any): The env/arg layers the landed bench ran with.
        accepted_config_path (str): The materialized config it accepted.
        setting_script (str): The reproduction script it wrote.
        active_runtime (Any): The promoted framework runtime.
        attempt_runtimes (Any): Every runtime it provisioned.
        framework_root (str): The source tree the patches apply against.
        stall_streak (int): The streak the lane ended on.
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


#: Every section the enablement event assembles from. Declared here as well as
#: in the assembler so :func:`finish` can read its own parts without importing
#: the assembler's tuple, which would close an import cycle.
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
    """Assemble the enablement event's ``ext`` out of its recorded rows.

    Args:
        parts (Mapping[str, list[dict[str, Any]]]): The enablement sections as
            read back from the spool, section name to row list.
        event (str): The event id to assemble.

    Returns:
        tuple[dict[str, Any], str]: The ``ext`` payload and the status the lane
            settled on. The status is empty when no write has closed the
            event, which leaves the caller's own reading standing.
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
        # Recorded, not inferred: the lane exists on the timeline because it
        # was triggered, so an event that is here at all was engaged. The
        # projection had to reconstruct this from four unrelated signals
        # because a section with no event behind it could equally mean
        # "armed and never needed".
        "engaged": True,
        "trigger": _as_dict(header.get("trigger")) or None,
        "attempts": {
            "count": len(attempts),
            # Rounds that got a verdict, which is the number the projection's
            # ``attempts`` counter was read as and is not: it counted
            # dispatches, including the one still in flight.
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
    """The status the event reports, from the terminal the lane reached.

    A lane that landed its repair succeeded. One that hit the stall cap failed,
    and it failed the run with it. A lane that closed on neither ran rounds
    that neither landed nor gave up, which is degraded -- and one that closed
    having run no round at all was admitted and never needed, which is skipped.
    """
    settled = str(outcome or "").strip().lower()
    if settled == OUTCOME_SUCCEEDED:
        return STATUS_SUCCEEDED
    if settled == OUTCOME_STALLED:
        return STATUS_FAILED
    return STATUS_DEGRADED if attempts else STATUS_SKIPPED


def _header(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Fold the event-level fragments into one header.

    There is one fragment per event, so this is normally a single row; folding
    rather than taking ``rows[0]`` keeps a spool that somehow holds two from
    dropping whichever one is second.
    """
    header: dict[str, Any] = {}
    for row in rows:
        if isinstance(row, Mapping):
            header.update({key: value for key, value in row.items() if value not in (None, "")})
    return header


def _recorded_trigger() -> bool:
    """Whether the lane already recorded what opened it.

    Read back from the spool rather than held in memory, because the trigger
    and the failure that would overwrite it are recorded from different ticks
    and, on a resume, from different processes.
    """
    try:
        from .assembler import event_parts

        rows = rows_for_event(event_parts((SECTION_EVENT,)).get(SECTION_EVENT) or [], enablement_event_id())
        return any(_as_dict(row.get("trigger")) for row in rows)
    except Exception:  # noqa: BLE001 — a spool we cannot read is not a trigger we have
        log.debug("enablement event: cannot read back the trigger", exc_info=True)
        return False


def _start_time() -> str:
    """The start time the open write stored, read back for the close.

    Nothing here holds the lane's start in memory, because nothing here holds
    the lane. The open write put it on the event-level fragment, which is the
    lane's only durable identity, so the close reads it from there.
    """
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
    """The natural id of an authoring round's row.

    The specialist task id when there is one, because that is what the dispatch
    and the rearm both name the round by. A round the lane synthesised carries
    no task id -- a build routed into the lane, a round found finished without
    a rearm -- and falls back to its ordinal, which is unique per lane and is
    what keeps a synthesised round from upserting onto a real one.
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
    """Project one installed artifact, dropping the backup bookkeeping.

    ``backup`` / ``source`` / ``existed`` describe how the install was made
    reversible, which is the executor's business and not a fact about the
    repair.
    """
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
