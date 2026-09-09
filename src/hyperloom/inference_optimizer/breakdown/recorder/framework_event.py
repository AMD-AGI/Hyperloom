# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The SBD V6 ``framework_agent`` event: both OPTIMIZE arms, recorded live.

FRAMEWORK_AGENT runs two arms in one phase: the configuration arm searches
server args and env vars, the source arm lands upstream patches. The phase
leaves only when both have run dry, because one arm going quiet is a reason to
switch levers inside the phase rather than to abandon the other.

The wire shape follows the progression a proposal moves along rather than the
arm it belongs to, since the arms differ in content and not in lifecycle:
``proposals`` is the main line, ``runs`` holds the dispatch facts those rows
reference, and ``attempts`` is one uniform row per thing measured, which the
adoption ledger walks. ``plateau`` is off that progression, and each evaluation
records the inputs and thresholds it used because re-deriving them later reads
a history that has kept growing. A proposal names its producer because the
orchestration agent, the seed grid and the switch manifest all yield variants
nobody proposed.

Gates and lifecycle steps are their own sections, composed into the rows that
own them at assembly: a row's list field cannot accumulate, because repeated
writes on one key deep-merge and a merge replaces a list wholesale.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable, Mapping
from typing import Any

from .event_fields import (
    as_dict as _as_dict,
    failure_row as _failure_row,
    float_or_none as _float_or_none,
    int_or_none as _int_or_none,
    now_iso_micros as _now_precise,
    now_iso_seconds as _now,
    text_or_none as _text_or_none,
    worst_status as _worst_status,
)
from .event_ids import event_id
from .event_rows import group_rows, rows_for_event, sort_rows, wire_rows
from .event_sink import RecordSink, make_sink
from .event_timeline import finish_event, open_event

log = logging.getLogger(__name__)

EVENT_TYPE = "framework_agent"
EVENT_KIND = "agent"

#: The phase and component segments of a framework event id. One event per
#: macro cycle: the phase can be re-entered, and each entry is a cycle.
EVENT_PHASE = "framework_agent"
EVENT_COMPONENT = "framework"

PRODUCER = "orchestrator"

SECTION_EVENT = "framework_event"
SECTION_PLATEAU = "framework_plateau"
SECTION_RUN = "framework_run"
SECTION_PROPOSAL = "framework_proposal"
SECTION_PROPOSAL_STEP = "framework_proposal_step"
SECTION_ATTEMPT = "framework_attempt"
SECTION_ATTEMPT_GATE = "framework_attempt_gate"

# Row types name a keyed row's kind within its section. Plateau readings and
# lifecycle steps have none: they are appended, so there is no key to build.
ROW_RUN = "run"
ROW_PROPOSAL = "proposal"
ROW_ATTEMPT = "attempt"
ROW_ATTEMPT_GATE = "attempt_gate"

#: Which arm a row belongs to. A field rather than a container, so one shape
#: serves both arms and a consumer can select on it.
ARM_CONFIG = "config"
ARM_SOURCE = "source"

#: A run's position on the chain, which is not derivable from its arm: the
#: source arm dispatches a specialist twice, to discover candidates and then to
#: author a patch from one.
ROLE_CONFIG = "config"
ROLE_DISCOVERY = "discovery"
ROLE_AUTHORING = "authoring"

#: Where a proposal came from. ``specialist`` carries the domain in
#: ``producer_ref``; the other two have no dispatch behind them at all.
PRODUCER_SPECIALIST = "specialist"
PRODUCER_ORCHESTRATION = "orchestration_agent"
PRODUCER_SEED_GRID = "seed_grid"

#: Which reader evaluated a plateau. The advisory asks whether to switch arms
#: and the exit whether the phase may leave, so a snapshot that did not say
#: which one is not interpretable.
PLATEAU_PATH_ADVISORY = "advisory"
PLATEAU_PATH_EXIT = "exit"

#: The steps a proposal can move through. Recorded as they happen rather than
#: derived from counters, so a candidate re-authored twice reads as two steps.
STEP_PROPOSED = "proposed"
STEP_REVIEWED = "reviewed"
STEP_AUDITED = "audited"
STEP_ROUTED = "routed"
STEP_AUTHORED = "authored"
STEP_REAUTHORED = "reauthored"
STEP_APPLY_RETRIED = "apply_retried"
STEP_ATTEMPTED = "attempted"
STEP_DROPPED = "dropped"

#: The dispositions a proposal settles on. ``dropped`` covers every way it
#: never reached a measurement; the row's ``reason`` says which.
DISPOSITION_ATTEMPTED = "attempted"
DISPOSITION_DROPPED = "dropped"
DISPOSITION_PENDING = "pending"

#: Who authored a review. A proposal blocked because the Critic had no manifest
#: to read must not be reported as one the Critic examined and refused.
REVIEWER_CRITIC = "critic"
REVIEWER_CRITIC_UNAVAILABLE = "critic_unavailable"

__all__ = [
    "ARM_CONFIG",
    "ARM_SOURCE",
    "DISPOSITION_ATTEMPTED",
    "DISPOSITION_DROPPED",
    "DISPOSITION_PENDING",
    "EVENT_COMPONENT",
    "EVENT_KIND",
    "EVENT_PHASE",
    "EVENT_TYPE",
    "PLATEAU_PATH_ADVISORY",
    "PLATEAU_PATH_EXIT",
    "PRODUCER",
    "PRODUCER_ORCHESTRATION",
    "PRODUCER_SEED_GRID",
    "PRODUCER_SPECIALIST",
    "REVIEWER_CRITIC",
    "REVIEWER_CRITIC_UNAVAILABLE",
    "ROLE_AUTHORING",
    "ROLE_CONFIG",
    "ROLE_DISCOVERY",
    "SECTION_ATTEMPT",
    "SECTION_ATTEMPT_GATE",
    "SECTION_EVENT",
    "SECTION_PLATEAU",
    "SECTION_PROPOSAL",
    "SECTION_PROPOSAL_STEP",
    "SECTION_RUN",
    "STEP_APPLY_RETRIED",
    "STEP_ATTEMPTED",
    "STEP_AUDITED",
    "STEP_AUTHORED",
    "STEP_DROPPED",
    "STEP_PROPOSED",
    "STEP_REAUTHORED",
    "STEP_REVIEWED",
    "STEP_ROUTED",
    "FrameworkEventRecorder",
    "assemble_framework_ext",
    "framework_event_id",
    "record_review_evidence",
    "make_framework_recorder",
    "producer_for_provenance",
]


def producer_for_provenance(provenance: Any) -> tuple[str, str]:
    """Map a config variant's provenance label onto ``(producer, producer_ref)``.

    The explore grid's labels are deliberately its own vocabulary. The ref names
    the specialist's domain and is empty otherwise; an unlabelled variant reads
    as the orchestration agent's, matching the grid parser's own default.
    """
    label = str(provenance or "").strip()
    if label.startswith("specialist:"):
        return PRODUCER_SPECIALIST, label.split(":", 1)[1].strip()
    if label == "default_grid":
        return PRODUCER_SEED_GRID, ""
    return PRODUCER_ORCHESTRATION, ""


def framework_event_id(macro_cycle: Any) -> str:
    """Build ``framework_agent:{macro_cycle}:framework``. Raises ``ValueError``
    if ``macro_cycle`` is not a non-negative integer."""
    return event_id(EVENT_PHASE, macro_cycle, EVENT_COMPONENT)


def _key(value: Any) -> str:
    """Escape a data-derived id for use as a fragment natural id: candidate ids
    here are PR urls and a fragment key joins its segments on ``:``. Injective
    -- ``%`` first, then the separator -- because a key two distinct ids can
    both produce merges their rows silently."""
    return str(value or "").replace("%", "%25").replace(":", "%3A")


def _stack(values: Mapping[str, Any]) -> dict[str, Any]:
    """Project the configuration stack an attempt was measured against. Both
    arms have one -- a source patch sits on top of whatever the session is
    serving, exactly as a config variant does -- so it is recorded uniformly."""
    return {
        "throughput": _float_or_none(values.get("throughput")),
        "accuracy": _float_or_none(values.get("accuracy")),
        "extra_server_args": str(values.get("extra_server_args") or ""),
        "extra_envs": dict(_as_dict(values.get("extra_envs"))),
        "remove_args": [str(arg) for arg in (values.get("remove_args") or []) if str(arg or "")],
        "unset_envs": [str(env) for env in (values.get("unset_envs") or []) if str(env or "")],
        "args_mode": _text_or_none(values.get("args_mode")),
    }


def _resume_gate_ordinals(event: str) -> dict[tuple[str, str], int]:
    """The ordinal held by each ``(attempt_id, gate)`` already on record.

    A resume re-entering the same macro cycle binds to the same event id, so a
    second recorder must not renumber a gate it re-rules or reuse a taken
    number; a re-ruled gate keeps its original ordinal and position.
    """
    try:
        from .assembler import framework_event_parts

        parts = framework_event_parts()
    except Exception:  # noqa: BLE001 — a fresh event has nothing to read
        return {}

    gates: dict[tuple[str, str], int] = {}
    for row in rows_for_event(parts.get(SECTION_ATTEMPT_GATE) or [], event):
        key = (str(row.get("attempt_id") or ""), str(row.get("gate") or ""))
        gates[key] = _int_or_none(row.get("ordinal")) or 0
    return gates


class FrameworkEventRecorder:
    """Records one FRAMEWORK_AGENT entry's facts, one fragment per row.

    Holds a sink and the ordinals its keyed gate rows are ordered by. Nothing
    else it writes is read back until :meth:`finish`, which assembles from the
    fragments and not from anything remembered, so an entry recorded across a
    resume assembles from both halves."""

    def __init__(self, sink: RecordSink, *, macro_cycle: int = 0):
        """Bind a recorder to the event of one phase entry. ``macro_cycle`` is
        recorded on the event as well as being a segment of its id, because a
        consumer reading the assembled event does not parse the id."""
        self._sink = sink
        self._t0 = time.monotonic()
        self._start_time = _now()
        self._sequence: int | None = None
        self._closed = False
        # Only gates are counted: appended rows carry their order in the write
        # itself, while a keyed gate needs a number no other leg has spent.
        self._gate_ordinals = _resume_gate_ordinals(self.event_id)
        self._sink.record(SECTION_EVENT, {"macro_cycle": int(macro_cycle or 0)})

    @property
    def event_id(self) -> str:
        """str: The event every row this recorder writes is tagged with."""
        return self._sink.event_id

    # ---- lifecycle -------------------------------------------------------

    def begin(self) -> None:
        """Put the event on the timeline. Opening is idempotent, so a phase
        whose entry hook runs twice in a cycle shares one timeline entry."""
        self._sequence = open_event(
            event_type=EVENT_TYPE,
            event=self.event_id,
            event_section=SECTION_EVENT,
            producer=PRODUCER,
            kind=EVENT_KIND,
            start_time=self._start_time,
        )

    def record_policy(self, **fields: Any) -> None:
        """Record the thresholds the entry runs under, as it resolves them, so
        the event says what the phase used rather than what survived."""
        config = _as_dict(fields.get("config"))
        source = _as_dict(fields.get("source"))
        self._sink.record(
            SECTION_EVENT,
            {
                "policy": {
                    "keep_threshold_pct": _float_or_none(fields.get("keep_threshold_pct")),
                    "variant_timeout_sec": _int_or_none(fields.get("variant_timeout_sec")),
                    "overtime_kill_ratio": _float_or_none(fields.get("overtime_kill_ratio")),
                    "force_exit_budget_pct": _float_or_none(fields.get("force_exit_budget_pct")),
                    ARM_CONFIG: {
                        "keep_gain_threshold_pct": _float_or_none(config.get("keep_gain_threshold_pct")),
                        "empty_streak_threshold": _int_or_none(config.get("empty_streak_threshold")),
                        "lookback": _int_or_none(config.get("lookback")),
                    },
                    ARM_SOURCE: {
                        "no_keep_streak_threshold": _int_or_none(source.get("no_keep_streak_threshold")),
                        "discovery_retry_limit": _int_or_none(source.get("discovery_retry_limit")),
                        "authoring_enabled": None
                        if source.get("authoring_enabled") is None
                        else bool(source.get("authoring_enabled")),
                    },
                },
            },
        )

    # ---- plateau ---------------------------------------------------------

    def record_plateau(
        self,
        *,
        arm: str,
        path: str,
        triggered: bool | None,
        inputs: Mapping[str, Any] | None = None,
        thresholds: Mapping[str, Any] | None = None,
    ) -> None:
        """Snapshot one plateau evaluation with the values it ruled on.

        The inputs and thresholds are recorded beside the verdict rather than
        referenced, because the history they count keeps growing: recomputing
        at export returns a number the phase never acted on. Every call
        appends, since two readings that agree are still two readings. A
        ``triggered`` of ``None`` could not rule, and the phase then behaves as
        though the arm were live.
        """
        self._sink.append(
            SECTION_PLATEAU,
            {
                "arm": str(arm or ""),
                "path": str(path or ""),
                "evaluated_at": _now_precise(),
                "triggered": None if triggered is None else bool(triggered),
                "inputs": dict(_as_dict(inputs)),
                "thresholds": dict(_as_dict(thresholds)),
            },
        )

    # ---- runs ------------------------------------------------------------

    def record_run(self, run_id: str, **fields: Any) -> None:
        """Record one specialist dispatch, or update the one already open.

        A dispatch and the result harvested from its worktree minutes later are
        two calls on one ``run_id`` -- what a proposal's ``run_ref`` points at
        -- and keying the fragment by it keeps them one row."""
        key = str(run_id or "")
        if not key:
            return
        row: dict[str, Any] = {"run_id": key}
        for name in (
            "role",
            "arm",
            "status",
            "domain",
            "scope",
            "gap_canonical_id",
            "reason",
            "dispatched_at",
            "completed_at",
            "worktree",
            "summary",
        ):
            if name in fields:
                row[name] = str(fields.get(name) or "")
        for name in ("tags", "transcripts", "new_findings", "residual_questions", "notes"):
            if name in fields:
                row[name] = [str(item) for item in (fields.get(name) or []) if str(item or "")]
        if "parallelism" in fields:
            row["parallelism"] = _int_or_none(fields.get("parallelism"))
        if "proposals_total" in fields:
            row["proposals_total"] = _int_or_none(fields.get("proposals_total"))
        if "empty" in fields:
            row["empty"] = bool(fields.get("empty"))
        for name in ("confidence_avg", "confidence"):
            if name in fields:
                row[name] = _float_or_none(fields.get(name))
        if fields.get("ensemble_scores"):
            row["ensemble_scores"] = dict(fields.get("ensemble_scores") or {})
        self._sink.record(SECTION_RUN, row, row_type=ROW_RUN, natural_ids=_key(key))

    # ---- proposals -------------------------------------------------------

    def record_proposal(self, proposal_id: str, **fields: Any) -> None:
        """Record one pursued thing, or update the one already open.

        Only the fields the caller passes are written, so ``run_ref`` is absent
        rather than empty on a proposal with no dispatch behind it -- the
        load-bearing fact for the producers with no parent run."""
        key = str(proposal_id or "")
        if not key:
            return
        row: dict[str, Any] = {"proposal_id": key}
        for name in (
            "arm",
            "producer",
            "producer_ref",
            "run_ref",
            "domain",
            "scope",
            "lever_kind",
            "gap_canonical_id",
            "source_ref",
            "repo",
            "title",
            "verdict",
            "route",
        ):
            if name in fields:
                row[name] = str(fields.get(name) or "")
        if "changed_files" in fields:
            row["changed_files"] = [str(path) for path in (fields.get("changed_files") or []) if str(path or "")]
        if "confidence" in fields:
            row["confidence"] = _float_or_none(fields.get("confidence"))
        self._sink.record(SECTION_PROPOSAL, row, row_type=ROW_PROPOSAL, natural_ids=_key(key))

    def record_proposal_review(
        self,
        proposal_id: str,
        *,
        verdict: str,
        effective_verdict: str = "",
        held_to_rule: str = "",
        reviewer: str = REVIEWER_CRITIC,
        iteration: Any = None,
        reason: str = "",
        confidence: Any = None,
        failure_reason_code: str = "",
        concerns: Any = None,
        advisory: Mapping[str, Any] | None = None,
        variants: Iterable[Mapping[str, Any]] | None = None,
    ) -> None:
        """Record the Critic's ruling on one proposal, inline on its row.

        The review lives on the proposal so that it follows the proposal
        wherever it was raised. Both verdicts are kept: a reject the loop held
        to a rule that only declared ``advise`` is two facts, and reporting
        either alone misreads the round.

        Args:
            advisory: Recorded as given, since the vocabulary is the Critic's
                own and a re-spelling here would drift from what it emitted.
            variants: Per-variant rulings for a grid reviewed by
                ``verdict_map``; a rejected variant reaches no bench, so no
                attempt row carries its ruling.
        """
        key = str(proposal_id or "")
        if not key:
            return
        authored = str(verdict or "")
        review: dict[str, Any] = {
            "verdict": authored,
            "effective_verdict": str(effective_verdict or "") or authored,
            "held_to_rule": str(held_to_rule or ""),
            "reviewer": str(reviewer or ""),
            "iteration": _int_or_none(iteration),
            "reason": str(reason or ""),
            "confidence": _float_or_none(confidence),
            "failure_reason_code": str(failure_reason_code or ""),
            "concerns": [str(item) for item in (concerns or []) if str(item or "")],
            "reviewed_at": _now(),
        }
        for field, value in _as_dict(advisory).items():
            name = str(field or "")
            if name and name not in review:
                review[name] = value
        rows = [
            {
                "variant_name": str(row.get("variant_name") or ""),
                "verdict": str(row.get("verdict") or ""),
                "effective_verdict": str(row.get("effective_verdict") or "") or str(row.get("verdict") or ""),
                "held_to_rule": str(row.get("held_to_rule") or ""),
                "reason": str(row.get("reason") or ""),
                "failure_reason_code": str(row.get("failure_reason_code") or ""),
            }
            for row in (variants or [])
            if str(row.get("variant_name") or "")
        ]
        if rows:
            review["variants"] = rows
        self._sink.record(
            SECTION_PROPOSAL,
            {"proposal_id": key, "critic_review": review},
            row_type=ROW_PROPOSAL,
            natural_ids=_key(key),
        )

    def record_proposal_review_outcome(
        self,
        proposal_id: str,
        *,
        materialized: Any = None,
        denied: Any = None,
        patch_verdict_key: str = "",
        reauthored: Any = None,
    ) -> None:
        """Record what the loop did with a ruling, onto the ruling itself.

        An ``advise`` that materialised and an ``advise`` held at the patch gate
        are the same ruling with opposite outcomes. ``patch_verdict_key`` is the
        subject the patch gate consults this ruling under, which connects a
        blocked ``integrate_patch`` back to the review that blocked it.
        """
        key = str(proposal_id or "")
        if not key:
            return
        outcome: dict[str, Any] = {}
        if materialized is not None:
            outcome["materialized"] = bool(materialized)
        if denied is not None:
            outcome["denied"] = bool(denied)
        if reauthored is not None:
            outcome["reauthored"] = bool(reauthored)
        if str(patch_verdict_key or ""):
            outcome["patch_verdict_key"] = str(patch_verdict_key)
        if not outcome:
            return
        self._sink.record(
            SECTION_PROPOSAL,
            {"proposal_id": key, "critic_review": {"outcome": outcome}},
            row_type=ROW_PROPOSAL,
            natural_ids=_key(key),
        )

    def record_proposal_step(
        self,
        proposal_id: str,
        *,
        step: str,
        run_ref: str = "",
        outcome: str = "",
        reason: str = "",
    ) -> None:
        """Record one step of a proposal's lifecycle as it happens.

        Steps are recorded rather than derived from counters: a candidate
        re-authored twice then retried once is three rows a reader can follow.
        ``step`` is a ``STEP_*`` value."""
        key = str(proposal_id or "")
        name = str(step or "")
        if not key or not name:
            return
        self._sink.append(
            SECTION_PROPOSAL_STEP,
            {
                "proposal_id": key,
                "step": name,
                "ts": _now_precise(),
                "run_ref": str(run_ref or ""),
                "outcome": str(outcome or ""),
                "reason": str(reason or ""),
            },
        )

    def settle_proposal(
        self,
        proposal_id: str,
        *,
        disposition: str,
        reason: str = "",
    ) -> None:
        """Record where a proposal ended up. ``disposition`` is a
        ``DISPOSITION_*`` value; ``pending`` is one the phase never resolved,
        the honest reading of a session killed mid-review."""
        key = str(proposal_id or "")
        if not key:
            return
        self._sink.record(
            SECTION_PROPOSAL,
            {
                "proposal_id": key,
                "terminal": {
                    "disposition": str(disposition or ""),
                    "reason": str(reason or ""),
                    "settled_at": _now(),
                },
            },
            row_type=ROW_PROPOSAL,
            natural_ids=_key(key),
        )

    # ---- attempts --------------------------------------------------------

    def record_attempt(self, attempt_id: str, **fields: Any) -> None:
        """Record one measured attempt, or update the one already open.

        This is the row the adoption ledger walks, so the throughput pair is
        recorded on it rather than cited: a later attempt on the same lever
        overwrites the measurements this one was judged on."""
        key = str(attempt_id or "")
        if not key:
            return
        row: dict[str, Any] = {"attempt_id": key}
        for name in (
            "arm",
            "round_id",
            "task_id",
            "proposal_ref",
            "provenance",
            "outcome",
            "reason",
            "stage",
            "decision",
            "validation_basis",
            "fingerprint",
            "variant_name",
            "candidate_id",
            "source_ref",
            "route",
            "patch_source",
            "patch_path",
        ):
            if name in fields:
                row[name] = str(fields.get(name) or "")
        for name in ("adopted", "attribution_eligible"):
            if name in fields:
                row[name] = None if fields.get(name) is None else bool(fields.get(name))
        for name in ("accepted_kernels", "target_files", "patches_applied"):
            if name in fields:
                row[name] = [str(item) for item in (fields.get(name) or []) if str(item or "")]
        if "ts" not in fields:
            row["ts"] = _now()
        else:
            row["ts"] = str(fields.get("ts") or "")
        if "measured_against" in fields:
            row["measured_against"] = _stack(_as_dict(fields.get("measured_against")))
        if "config_delta" in fields:
            delta = _as_dict(fields.get("config_delta"))
            row["config_delta"] = {
                "extra_server_args": str(delta.get("extra_server_args") or ""),
                "extra_envs": dict(_as_dict(delta.get("extra_envs"))),
                "remove_args": [str(arg) for arg in (delta.get("remove_args") or []) if str(arg or "")],
                "unset_envs": [str(env) for env in (delta.get("unset_envs") or []) if str(env or "")],
                "args_mode": _text_or_none(delta.get("args_mode")),
            }
        if "measurement" in fields:
            measured = _as_dict(fields.get("measurement"))
            row["measurement"] = {
                "before_tput": _float_or_none(measured.get("before_tput")),
                "after_tput": _float_or_none(measured.get("after_tput")),
                "gain_pct": _float_or_none(measured.get("gain_pct")),
                "runtime_sec": _float_or_none(measured.get("runtime_sec")),
                "estimated_output_throughput": _float_or_none(measured.get("estimated_output_throughput")),
            }
        if "accuracy" in fields:
            accuracy = _as_dict(fields.get("accuracy"))
            row["accuracy"] = {
                "required": None if accuracy.get("required") is None else bool(accuracy.get("required")),
                "reference": _float_or_none(accuracy.get("reference")),
                "value": _float_or_none(accuracy.get("value")),
                "passed": None if accuracy.get("passed") is None else bool(accuracy.get("passed")),
            }
        if "failure" in fields:
            failure = _as_dict(fields.get("failure"))
            row["failure"] = {
                "error_class": str(failure.get("error_class") or ""),
                "error_excerpt": str(failure.get("error_excerpt") or ""),
            }
        if "artifacts" in fields:
            artifacts = _as_dict(fields.get("artifacts"))
            row["artifacts"] = {
                "workspace": str(artifacts.get("workspace") or ""),
                "server_log_path": str(artifacts.get("server_log_path") or ""),
                "raw_result_path": str(artifacts.get("raw_result_path") or ""),
            }
        self._sink.record(SECTION_ATTEMPT, row, row_type=ROW_ATTEMPT, natural_ids=_key(key))

    def record_attempt_gate(
        self,
        attempt_id: str,
        gate: str,
        *,
        passed: bool | None,
        reason: str = "",
        observed: Any = None,
        threshold: Any = None,
    ) -> None:
        """Record one gate's verdict on one attempt, as it is evaluated.

        A gate that was never reached writes no row, which is how assembly
        tells "did not pass" apart from "did not apply". A whole gating
        sequence fits inside one clock tick, so order comes from an ordinal
        assigned on the first evaluation and reused. ``passed`` of ``None``
        is a gate that ran but could not rule.
        """
        key = str(attempt_id or "")
        name = str(gate or "")
        if not key or not name:
            return
        ordinal = self._gate_ordinals.get((key, name))
        if ordinal is None:
            ordinal = max(self._gate_ordinals.values(), default=0) + 1
            self._gate_ordinals[(key, name)] = ordinal
        self._sink.record(
            SECTION_ATTEMPT_GATE,
            {
                "attempt_id": key,
                "gate": name,
                "ordinal": ordinal,
                "passed": None if passed is None else bool(passed),
                "reason": str(reason or ""),
                "observed": _float_or_none(observed),
                "threshold": _float_or_none(threshold),
                # The latest ruling's time; position comes from the ordinal.
                "ts": _now_precise(),
            },
            row_type=ROW_ATTEMPT_GATE,
            natural_ids=(_key(key), name),
        )

    # ---- close -----------------------------------------------------------

    def finish(
        self,
        *,
        exit_reason: str = "",
        trigger: str = "",
        hint: str = "",
        switch_bottleneck: bool | None = None,
        failure: Mapping[str, Any] | None = None,
    ) -> None:
        """Close the event on the phase's own exit evidence. ``failure`` carries
        ``failed_task_id`` / ``error_class`` / ``error`` when the entry
        failed."""
        failed = _as_dict(failure)
        payload: dict[str, Any] = {
            "exit": {
                "reason": str(exit_reason or ""),
                "trigger": str(trigger or ""),
                "hint": str(hint or ""),
                "switch_bottleneck": None if switch_bottleneck is None else bool(switch_bottleneck),
            },
        }
        if failed:
            payload["failure"] = {
                "failed_task_id": str(failed.get("failed_task_id") or ""),
                "error_class": str(failed.get("error_class") or ""),
                "error": str(failed.get("error") or ""),
            }
        self._close(status="failed" if failed else "", payload=payload)

    def finish_crashed(self, exc: BaseException) -> None:
        """Close an event whose phase raised instead of exiting."""
        if self._closed:
            return
        self._close(
            status="failed",
            payload={
                "failure": _failure_row(
                    phase=EVENT_TYPE,
                    error_class=type(exc).__name__,
                    message=f"framework agent phase raised: {exc!r}",
                )
            },
        )

    def _close(self, *, status: str, payload: Mapping[str, Any]) -> None:
        """Record the terminal facts and close the event. ``status`` is used
        only when assembly derives nothing of its own."""
        if self._closed:
            return
        self._closed = True
        end_time = _now()
        self._sink.record(
            SECTION_EVENT,
            {
                **payload,
                "end_time": end_time,
                "duration_sec": round(time.monotonic() - self._t0, 3),
            },
        )
        from .assembler import framework_event_parts

        ext, derived = assemble_framework_ext(framework_event_parts(), event=self.event_id)
        finish_event(
            event_type=EVENT_TYPE,
            event=self.event_id,
            sequence=self._sequence,
            status=derived or status or "succeeded",
            ext=ext,
            kind=EVENT_KIND,
            start_time=self._start_time,
            end_time=end_time,
        )


def assemble_framework_ext(
    parts: Mapping[str, list[dict[str, Any]]],
    *,
    event: str,
) -> tuple[dict[str, Any], str]:
    """Assemble one framework event's ``ext``, and the status derived from its
    rows, out of the sections read back from the spool. Rows of every other
    event are ignored, and the status is empty when the event holds no work,
    which leaves the caller's own reading standing."""
    event_rows = rows_for_event(parts.get(SECTION_EVENT) or [], event)
    header = event_rows[0] if event_rows else {}

    plateau = wire_rows(
        sort_rows(
            rows_for_event(parts.get(SECTION_PLATEAU) or [], event),
            keys=("evaluated_at", "arm"),
        ),
        drop=("event_id",),
    )
    runs = wire_rows(
        sort_rows(
            rows_for_event(parts.get(SECTION_RUN) or [], event),
            keys=("dispatched_at", "run_id"),
        ),
        drop=("event_id",),
    )
    attempt_rows = sort_rows(
        rows_for_event(parts.get(SECTION_ATTEMPT) or [], event),
        keys=("ts", "attempt_id"),
    )
    gates_by_attempt = group_rows(
        sort_rows(
            rows_for_event(parts.get(SECTION_ATTEMPT_GATE) or [], event),
            keys=("ordinal", "ts", "gate"),
        ),
        "attempt_id",
    )
    attempts: list[dict[str, Any]] = []
    for row in wire_rows(attempt_rows, drop=("event_id",)):
        key = str(row.get("attempt_id") or "")
        row["gates"] = wire_rows(gates_by_attempt.get(key, []), drop=("event_id", "attempt_id", "ordinal"))
        row["blocked_by"] = _text_or_none(_blocking_gate(row["gates"]))
        attempts.append(row)

    steps_by_proposal = group_rows(
        sort_rows(
            rows_for_event(parts.get(SECTION_PROPOSAL_STEP) or [], event),
            keys=("ts", "step"),
        ),
        "proposal_id",
    )
    attempts_by_proposal: dict[str, list[str]] = {}
    for row in attempts:
        ref = str(row.get("proposal_ref") or "")
        if ref:
            attempts_by_proposal.setdefault(ref, []).append(str(row.get("attempt_id") or ""))
    proposals: list[dict[str, Any]] = []
    for row in wire_rows(
        sort_rows(rows_for_event(parts.get(SECTION_PROPOSAL) or [], event), keys=("proposal_id",)),
        drop=("event_id",),
    ):
        key = str(row.get("proposal_id") or "")
        row["lifecycle"] = wire_rows(
            steps_by_proposal.get(key, []),
            drop=("event_id", "proposal_id"),
        )
        # Derived at close: the attempt already names its proposal, and a
        # second copy of the link is a second thing that can disagree.
        row["attempt_refs"] = attempts_by_proposal.get(key, [])
        proposals.append(row)

    # A run's own row does not name what it produced -- the proposal does -- so
    # the back-reference is projected here for a reader following the chain.
    produced: dict[str, list[str]] = {}
    for row in proposals:
        ref = str(row.get("run_ref") or "")
        if ref:
            produced.setdefault(ref, []).append(str(row.get("proposal_id") or ""))
    for row in runs:
        row["produced_ids"] = produced.get(str(row.get("run_id") or ""), [])

    ext = {
        "macro_cycle": _int_or_none(header.get("macro_cycle")) or 0,
        "policy": _as_dict(header.get("policy")),
        "plateau": plateau,
        "runs": runs,
        "proposals": proposals,
        "attempts": attempts,
        "exit": _as_dict(header.get("exit")),
        "failure": _as_dict(header.get("failure")) or None,
        "duration_sec": header.get("duration_sec"),
    }
    return ext, _derived_status(header, runs=runs, proposals=proposals, attempts=attempts)


def _derived_status(
    header: Mapping[str, Any],
    *,
    runs: list[dict[str, Any]],
    proposals: list[dict[str, Any]],
    attempts: list[dict[str, Any]],
) -> str:
    """Decide the status the event closes on: ``failed`` when the entry
    recorded a failure, ``skipped`` when it did nothing at all, ``degraded``
    when it worked but never closed, and otherwise the worst status its runs
    reported -- an entry whose every run failed is not a success."""
    if _as_dict(header.get("failure")):
        return "failed"
    if not (runs or proposals or attempts):
        return "skipped"
    if not str(header.get("end_time") or ""):
        return "degraded"
    return _worst_status(row.get("status") for row in runs) or "succeeded"


def _blocking_gate(gates: list[dict[str, Any]]) -> str:
    """Name the first gate that did not pass, or ``""`` when all of them did. A
    gate that ruled ``None`` blocks only if nothing after it failed outright,
    so an attempt rejected on the keep threshold reports that threshold."""
    unresolved = ""
    for row in gates:
        passed = row.get("passed")
        if passed is False:
            return str(row.get("gate") or "")
        if passed is None and not unresolved:
            unresolved = str(row.get("gate") or "")
    return unresolved


def record_review_evidence(
    *,
    macro_cycle: Any,
    proposal_id: str,
    artifacts: Mapping[str, Any] | None = None,
    kb: Mapping[str, Any] | None = None,
) -> None:
    """Attach what a ruling was grounded in, onto the ruling.

    Called from the Critic's own turn rather than through the phase's recorder,
    because that is where these facts exist: the artifacts are written by the
    review runtime and the KB write result only comes back on its emit. Silent
    when there is no session or no such event, since the Critic also runs on
    ticks in phases that record no framework event.
    """
    key = str(proposal_id or "")
    evidence: dict[str, Any] = {}
    if artifacts:
        evidence["artifacts"] = dict(artifacts)
    if kb:
        evidence["kb"] = dict(kb)
    if not key or not evidence:
        return
    from ...session.session_binding import session_is_bound

    try:
        if not session_is_bound():
            return
        make_sink(framework_event_id(macro_cycle), producer=PRODUCER).record(
            SECTION_PROPOSAL,
            {"proposal_id": key, "critic_review": evidence},
            row_type=ROW_PROPOSAL,
            natural_ids=_key(key),
        )
    except Exception:  # noqa: BLE001 — observability cannot change the review
        log.debug("framework timeline: review evidence record failed for %s", key, exc_info=True)


def make_framework_recorder(*, macro_cycle: Any = 0) -> FrameworkEventRecorder | None:
    """Build a recorder already opened on the timeline, or ``None``. Phase
    behavior must not depend on the recorder existing, so construction failures
    degrade to "no event", and an unbound session declines rather than writing
    the timeline into an arbitrary directory."""
    from ...session.session_binding import session_is_bound

    try:
        if not session_is_bound():
            log.warning(
                "framework timeline: no session bound; this phase entry's whole event will be "
                "missing from the breakdown. The coordinator binds at startup, so this means "
                "either that never happened or the entry ran outside the session's context"
            )
            return None
        recorder = FrameworkEventRecorder(
            make_sink(framework_event_id(macro_cycle), producer=PRODUCER),
            macro_cycle=int(macro_cycle or 0),
        )
    except Exception:  # noqa: BLE001 — observability cannot change phase behavior
        log.warning(
            "framework timeline: recorder construction failed; this phase entry's whole event "
            "will be missing from the breakdown",
            exc_info=True,
        )
        return None
    recorder.begin()
    return recorder
