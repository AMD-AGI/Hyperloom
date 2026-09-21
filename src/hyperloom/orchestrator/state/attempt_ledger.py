# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The attempts ledger: one row shape, written from one place per lever.

``SharedState.attempts`` is control-plane state -- the per-lever dryness
judgment reads it -- so the write must not sit behind the breakdown timeline
recorder, whose lifetime is one FRAMEWORK_AGENT phase entry.
"""

from __future__ import annotations

from typing import Any, Mapping

from hyperloom.inference_optimizer.breakdown.agent_ownership import LEVER_CONFIG, patch_lever_kind

from ..actions.executors._grid_base import is_kept

#: Lanes that own their own ``apply_failed`` retry ladder and re-dispatch the candidate.
_RETRY_LANES = ("perf_framework", "perf_explore")


def _record(
    state: Any,
    *,
    lever_kind: str,
    outcome: str,
    gain_pct: float | None,
    before_tput: float | None,
    after_tput: float | None,
    task_id: str = "",
    round_id: str = "",
    fingerprint: str = "",
    variant_name: str = "",
    candidate_id: str = "",
    specialist_task_id: str = "",
    error_class: str = "",
    provenance: str = "",
) -> None:
    """Append one row, stamped with the macro-cycle by ``record_attempt``.

    ``lever_kind`` through ``after_tput`` are what the dryness judgment reads;
    the rest are the forensic record. Both levers write this one field set, so
    a reader walks them without knowing which arm produced a row.
    """
    state.record_attempt(
        {
            "lever_kind": lever_kind,
            "outcome": outcome,
            "adopted": is_kept(outcome),
            "gain_pct": gain_pct,
            "before_tput": before_tput,
            "after_tput": after_tput,
            "task_id": task_id,
            "round_id": round_id,
            "fingerprint": fingerprint,
            "variant_name": variant_name,
            "candidate_id": candidate_id,
            "specialist_task_id": specialist_task_id,
            "error_class": error_class,
            "provenance": provenance,
        }
    )


def record_config_attempt(
    state: Any,
    *,
    task_id: str,
    round_id: str,
    fingerprint: str,
    variant_name: str,
    outcome: str,
    gain_pct: float | None,
    before_tput: float | None,
    after_tput: float | None,
    error_class: str,
    provenance: str,
) -> None:
    """Record one benchmarked explore-grid variant."""
    _record(
        state,
        lever_kind=LEVER_CONFIG,
        outcome=outcome,
        gain_pct=gain_pct,
        before_tput=before_tput,
        after_tput=after_tput,
        task_id=task_id,
        round_id=round_id,
        fingerprint=fingerprint,
        variant_name=variant_name,
        error_class=error_class,
        provenance=provenance,
    )


def record_patch_attempt(
    state: Any,
    *,
    task_id: str,
    specialist_task_id: str,
    outcome: str,
    gain_pct: float | None,
    before_tput: float | None,
    after_tput: float | None,
    error_class: str,
    evidence: Mapping[str, Any],
) -> None:
    """Record one resolved ``integrate_patch`` candidate.

    The lever comes from :func:`patch_lever_kind`, which honours an explicit
    ``lever_kind`` in *evidence* and derives one for the local-exploration arm,
    which dispatches against a gap without naming a lever.

    A candidate the lane will re-dispatch, and one dropped before it reached the
    bench, are not resolved: counting either would let apply-conflict noise and
    guard filtering read as a search that has run out of things to try.
    """
    if outcome == "skipped" or (outcome == "apply_failed" and evidence.get("lane") in _RETRY_LANES):
        return
    _record(
        state,
        lever_kind=patch_lever_kind(evidence),
        outcome=outcome,
        gain_pct=gain_pct,
        before_tput=before_tput,
        after_tput=after_tput,
        task_id=task_id,
        candidate_id=str(evidence.get("framework_agent_candidate_id") or ""),
        specialist_task_id=specialist_task_id,
        error_class=error_class,
    )
