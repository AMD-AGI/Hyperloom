# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The configuration arm's rows on the FRAMEWORK_AGENT event.

Timeline only. The adoption ledger and the optimization journal are written
beside these rows and do not depend on a recorder being open, so an outcome is
recorded here verbatim rather than collapsed the way the journal collapses
``KEEP_UNSTABLE`` and ``KILLED_OVERTIME`` into a plain revert.
"""

from __future__ import annotations

from typing import Any, Collection, Iterable, Mapping, NamedTuple, Sequence

from hyperloom.common.timeutil import now_iso
from hyperloom.inference_optimizer.breakdown.recorder.framework_event import (
    ARM_CONFIG,
    DISPOSITION_ATTEMPTED,
    ROLE_CONFIG,
    STEP_ATTEMPTED,
    GateVerdict,
)

from ..state.failure_evidence import UNMEASURED_OUTCOMES, classify_failure_attribution

#: Explore outcomes that are not an attempt at anything: a variant skipped as a
#: duplicate was never measured, so no measurement would back its funnel row.
NON_ATTEMPT_OUTCOMES = frozenset({"SKIPPED_DEDUP"})

#: The gate whose verdict the shared accuracy block is projected from.
_ACCURACY_GATE = "accuracy"

#: How much of a run's error text a reason field carries.
_REASON_EXCERPT_CHARS = 200


class _ConfigAttemptContext(NamedTuple):
    """What every attempt row produced by one grid dispatch shares."""

    task_id: str
    round_id: str
    proposal_ref: str
    adopted_variants: Collection[str]


def record_config_run(coord: Any, *, task: Any, result_dict: Mapping[str, Any]) -> None:
    """Record the configuration arm's grid dispatch on the framework event.

    The arm's dispatch is one ``explore`` task, and it earns a run row for the
    same reason a specialist dispatch does: the event reduces its status over
    what it dispatched, so an arm whose grid came back with nothing must not
    read as an arm that was never tried. Recorded at completion, where the
    outcome is known, with the dispatch time taken off the task row; a grid
    that measured nothing still lands, which is the case
    :func:`record_config_attempts` never sees.
    """
    recorder = coord._coord.phase_framework.timeline()
    if recorder is None:
        return
    task_id = str(getattr(task, "task_id", "") or "")
    if not task_id:
        return
    variants = result_dict.get("per_variant_outcomes")
    measured = [row for row in variants if isinstance(row, dict)] if isinstance(variants, list) else []
    recorder.record_run(
        task_id,
        role=ROLE_CONFIG,
        arm=ARM_CONFIG,
        status=str(result_dict.get("status") or "") or "succeeded",
        dispatched_at=str(getattr(task, "created_at", "") or ""),
        completed_at=now_iso("seconds"),
        reason=str(result_dict.get("error") or "")[:_REASON_EXCERPT_CHARS],
        # A grid the run stopped before it measured anything is not a grid
        # whose variants were measured and lost.
        empty=not measured,
        workspace=str(result_dict.get("workspace") or ""),
    )


def _attempt_measurement(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """What one variant's benchmark measured.

    Both ends of the pair: the anchor advances on every KEEP, so a percentage
    without its denominator adds to nothing.
    """
    return {
        "before_tput": metrics.get("base_tput"),
        "after_tput": metrics.get("tput"),
        "gain_pct": metrics.get("gain_pct"),
        "runtime_sec": metrics.get("runtime_sec"),
        "estimated_output_throughput": metrics.get("estimated_output_throughput"),
    }


def _attempt_config_delta(variant: Mapping[str, Any]) -> dict[str, Any]:
    """The launch-configuration change one variant applied."""
    return {
        "extra_server_args": variant.get("extra_server_args"),
        "extra_envs": variant.get("extra_envs"),
        "remove_args": variant.get("remove_args"),
        "unset_envs": variant.get("unset_envs"),
        "args_mode": variant.get("args_mode"),
    }


def _attempt_accuracy(gates: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """The accuracy block, projected from the gate that ruled.

    This arm gates accuracy rather than reporting it. No gate row means nothing
    gated the variant, which is not a gate that refused it.
    """
    ruling = next((gate for gate in gates if str(gate.get("gate") or "") == _ACCURACY_GATE), {})
    return {
        "required": True if ruling else None,
        "value": ruling.get("observed"),
        "reference": ruling.get("threshold"),
        "passed": ruling.get("passed"),
    }


def _attempt_failure(row: Mapping[str, Any], outcome: str) -> dict[str, Any]:
    """What is known about a variant's failure, attributed only when it never measured."""
    error_class = str(row.get("error_class") or "")
    error_excerpt = str(row.get("error_excerpt") or "")
    attribution = ""
    if outcome in UNMEASURED_OUTCOMES:
        attribution = classify_failure_attribution(
            error_class=error_class,
            error_excerpt=error_excerpt,
            reason=row.get("reason"),
            explicit=row.get("failure_attribution"),
        )
    return {
        "error_class": error_class,
        "error_excerpt": error_excerpt,
        "attribution": attribution,
    }


def _attempt_artifacts(row: Mapping[str, Any]) -> dict[str, str]:
    """Where one variant's run left its workspace and its logs."""
    return {
        "workspace": str(row.get("workspace") or ""),
        "server_log_path": str(row.get("server_log_path") or ""),
        "raw_result_path": str(row.get("raw_result_path") or ""),
    }


def _attempt_fields(
    row: Mapping[str, Any],
    context: _ConfigAttemptContext,
    gates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """The framework-event fields describing one measured configuration variant."""
    variant = row.get("variant") if isinstance(row.get("variant"), dict) else {}
    metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
    outcome = str(row.get("outcome") or "")
    fingerprint = str(row.get("fingerprint") or "")
    adopted = fingerprint in context.adopted_variants
    return {
        "arm": ARM_CONFIG,
        "round_id": context.round_id,
        "task_id": context.task_id,
        "proposal_ref": context.proposal_ref,
        "provenance": str(row.get("provenance") or ""),
        "outcome": outcome,
        "reason": str(row.get("reason") or ""),
        "reasoning": str(variant.get("note") or ""),
        "reasoning_origin": str(variant.get("reasoning_origin") or ""),
        "experience_citations": variant.get("experience_citations") or [],
        "stage": str(row.get("stage") or ""),
        "fingerprint": fingerprint,
        # The fingerprint is the join key; the name is what a reader
        # recognises the variant by.
        "variant_name": str(row.get("variant_name") or ""),
        "measurement": _attempt_measurement(metrics),
        "config_delta": _attempt_config_delta(variant),
        "accuracy": _attempt_accuracy(gates),
        "failure": _attempt_failure(row, outcome),
        "artifacts": _attempt_artifacts(row),
        "decision": outcome,
        "adopted": adopted,
        # Every variant recorded here ran: the deduped one is skipped above.
        "benched": True,
        # Recorded rather than referenced: every KEEP advances the stack, so
        # the session's current config is not what this variant was measured
        # on top of.
        "measured_against": row.get("measured_against") or {},
        # What stood behind the verdict. Absent when nothing ruled.
        "validation_basis": str(row.get("validation_basis") or ""),
        # A pair is what makes a gain addable, so eligibility follows the pair
        # being present rather than the outcome being a KEEP.
        "attribution_eligible": (adopted and metrics.get("base_tput") is not None and metrics.get("tput") is not None),
    }


def _record_attempt_gates(recorder: Any, attempt_id: str, gates: Iterable[Mapping[str, Any]]) -> None:
    """Record each named gate's verdict against one attempt."""
    for gate in gates:
        name = str(gate.get("gate") or "")
        if not name:
            continue
        recorder.record_attempt_gate(
            attempt_id,
            GateVerdict(
                gate=name,
                passed=gate.get("passed"),
                reason=str(gate.get("reason") or ""),
                observed=gate.get("observed"),
                threshold=gate.get("threshold"),
            ),
        )


def _record_variant_attempt(recorder: Any, row: Any, context: _ConfigAttemptContext) -> bool:
    """Record one measured variant and its gates; false when the row is no attempt at anything."""
    if not isinstance(row, dict):
        return False
    outcome = str(row.get("outcome") or "")
    if outcome in NON_ATTEMPT_OUTCOMES:
        return False
    # The fingerprint identifies the variant within the round and the round
    # within the task, so the three together identify the attempt.
    fingerprint = str(row.get("fingerprint") or "")
    parts = (context.task_id, context.round_id, fingerprint)
    attempt_id = ":".join(part for part in parts if part) or context.task_id
    if not attempt_id:
        return False
    gates = [gate for gate in (row.get("gates") or []) if isinstance(gate, dict)]
    recorder.record_attempt(attempt_id, **_attempt_fields(row, context, gates))
    _record_attempt_gates(recorder, attempt_id, gates)
    return True


def _settle_proposal_as_attempted(recorder: Any, proposal_ref: str, recorded: int) -> None:
    """Settle the proposal behind a grid, now that its variants have been recorded.

    Settled here rather than at materialization: a grid whose task dies before
    any variant runs has no attempt to stand behind an ``attempted`` reading,
    and is more honestly left unsettled.
    """
    recorder.record_proposal_step(proposal_ref, step=STEP_ATTEMPTED, outcome=str(recorded))
    recorder.settle_proposal(proposal_ref, disposition=DISPOSITION_ATTEMPTED)


def record_config_attempts(
    coord: Any,
    *,
    task: Any,
    per_variant: list[dict[str, Any]],
    result_dict: Mapping[str, Any],
    adopted_variants: Collection[str],
) -> None:
    """Record the configuration arm's measured attempts on the framework event."""
    recorder = coord._coord.phase_framework.timeline()
    if recorder is None:
        return
    params = getattr(task, "params", None) or {}
    context = _ConfigAttemptContext(
        task_id=str(getattr(task, "task_id", "") or ""),
        round_id=str(result_dict.get("round_id") or ""),
        proposal_ref=str(params.get("proposal_msg_id") or ""),
        adopted_variants=adopted_variants,
    )
    recorded = 0
    for row in per_variant:
        if _record_variant_attempt(recorder, row, context):
            recorded += 1
    if recorded and context.proposal_ref:
        _settle_proposal_as_attempted(recorder, context.proposal_ref, recorded)
