# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The configuration arm's attempt rows on the FRAMEWORK_AGENT event."""

from __future__ import annotations

from typing import Any, Collection, Mapping, Sequence

from hyperloom.inference_optimizer.breakdown.recorder.framework_event import (
    ARM_CONFIG,
    DISPOSITION_ATTEMPTED,
    STEP_ATTEMPTED,
    GateVerdict,
)

#: Explore outcomes that are not an attempt at anything: a variant skipped as a
#: duplicate was never measured, so no measurement would back its funnel row.
_NON_ATTEMPT_OUTCOMES: frozenset[str] = frozenset({"SKIPPED_DEDUP"})


def _attempt_failure(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "error_class": str(row.get("error_class") or ""),
        "error_excerpt": str(row.get("error_excerpt") or ""),
        # Explore classified this variant's failure once; the verdict it stored is the one verdict.
        "attribution": str(row.get("failure_attribution") or ""),
    }


def _attempt_fields(
    row: Mapping[str, Any],
    gates: Sequence[Mapping[str, Any]],
    shared: Mapping[str, str],
    adopted: bool,
) -> dict[str, Any]:
    variant = row.get("variant") if isinstance(row.get("variant"), dict) else {}
    metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
    outcome = str(row.get("outcome") or "")
    # This arm gates accuracy rather than reporting it, so the block both arms
    # carry is projected from the gate that ruled. No gate row means nothing
    # gated the variant, which is not a gate that refused it.
    ruling = next((gate for gate in gates if str(gate.get("gate") or "") == "accuracy"), {})
    return {
        "arm": ARM_CONFIG,
        **shared,
        "provenance": row.get("provenance"),
        "outcome": outcome,
        "reason": row.get("reason"),
        "reasoning": variant.get("note"),
        "reasoning_origin": variant.get("reasoning_origin"),
        "experience_citations": variant.get("experience_citations"),
        "stage": row.get("stage"),
        "fingerprint": row.get("fingerprint"),
        # The fingerprint is the join key; the name is what a reader
        # recognises the variant by.
        "variant_name": row.get("variant_name"),
        "measurement": {
            # Both ends of the pair: the anchor advances on every KEEP, so a
            # percentage without its denominator adds to nothing.
            "before_tput": metrics.get("base_tput"),
            "after_tput": metrics.get("tput"),
            "gain_pct": metrics.get("gain_pct"),
            "runtime_sec": metrics.get("runtime_sec"),
            "estimated_output_throughput": metrics.get("estimated_output_throughput"),
        },
        "config_delta": {
            "extra_server_args": variant.get("extra_server_args"),
            "extra_envs": variant.get("extra_envs"),
            "remove_args": variant.get("remove_args"),
            "unset_envs": variant.get("unset_envs"),
            "args_mode": variant.get("args_mode"),
        },
        "accuracy": {
            "required": True if ruling else None,
            "value": ruling.get("observed"),
            "reference": ruling.get("threshold"),
            "passed": ruling.get("passed"),
        },
        "failure": _attempt_failure(row),
        "artifacts": {
            "workspace": row.get("workspace"),
            "server_log_path": row.get("server_log_path"),
            "raw_result_path": row.get("raw_result_path"),
        },
        "decision": outcome,
        "adopted": adopted,
        # Every variant recorded here ran: the deduped one is skipped above.
        "benched": True,
        # Recorded rather than referenced: every KEEP advances the stack, so
        # the session's current config is not what this variant was measured
        # on top of.
        "measured_against": row.get("measured_against"),
        # What stood behind the verdict. Absent when nothing ruled.
        "validation_basis": row.get("validation_basis"),
        # A pair is what makes a gain addable, so eligibility follows the pair
        # being present rather than the outcome being a KEEP.
        "attribution_eligible": (adopted and metrics.get("base_tput") is not None and metrics.get("tput") is not None),
    }


def _record_variant_attempt(
    recorder: Any,
    row: Any,
    shared: Mapping[str, str],
    adopted_variants: Collection[str],
) -> bool:
    if not isinstance(row, dict):
        return False
    if str(row.get("outcome") or "") in _NON_ATTEMPT_OUTCOMES:
        return False
    # The fingerprint identifies the variant within the round and the round
    # within the task, so the three together identify the attempt.
    fingerprint = str(row.get("fingerprint") or "")
    parts = (shared["task_id"], shared["round_id"], fingerprint)
    attempt_id = ":".join(part for part in parts if part) or shared["task_id"]
    if not attempt_id:
        return False
    gates = [gate for gate in (row.get("gates") or []) if isinstance(gate, dict)]
    recorder.record_attempt(attempt_id, **_attempt_fields(row, gates, shared, fingerprint in adopted_variants))
    for gate in gates:
        recorder.record_attempt_gate(
            attempt_id,
            GateVerdict(
                gate=str(gate.get("gate") or ""),
                passed=gate.get("passed"),
                reason=str(gate.get("reason") or ""),
                observed=gate.get("observed"),
                threshold=gate.get("threshold"),
            ),
        )
    return True


def _record_config_attempts(
    coord: Any,
    *,
    task: Any,
    per_variant: list[dict[str, Any]],
    result_dict: Mapping[str, Any],
    adopted_variants: Collection[str],
) -> None:
    """Record the configuration arm's measured attempts on the framework event.

    Timeline only: the ledger row is written by ``fact_write_hook``, which does
    not depend on a recorder being open. The outcome is recorded verbatim here,
    unlike the journal beside it, which collapses ``KEEP_UNSTABLE`` and
    ``KILLED_OVERTIME`` into a plain revert.
    """
    recorder = coord._coord.phase_framework.timeline()
    if recorder is None:
        return
    params = getattr(task, "params", None) or {}
    shared = {
        "task_id": str(getattr(task, "task_id", "") or ""),
        "round_id": str(result_dict.get("round_id") or ""),
        "proposal_ref": str(params.get("proposal_msg_id") or ""),
    }
    recorded = 0
    for row in per_variant:
        if _record_variant_attempt(recorder, row, shared, adopted_variants):
            recorded += 1
    if not (recorded and shared["proposal_ref"]):
        return
    # Settled here rather than at materialization: a grid whose task dies before
    # any variant runs has no attempt to stand behind an ``attempted`` reading,
    # and is more honestly left unsettled.
    recorder.record_proposal_step(shared["proposal_ref"], step=STEP_ATTEMPTED, outcome=str(recorded))
    recorder.settle_proposal(shared["proposal_ref"], disposition=DISPOSITION_ATTEMPTED)
