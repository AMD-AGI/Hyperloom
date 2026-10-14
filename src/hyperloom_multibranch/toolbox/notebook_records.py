# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The notebook's request and result rows from the toolbox's own types.

The notebook still stores its stand-ins (``contracts.notebook.peers``), which carry fewer fields than the measurement
contract. This module is the only place that knows the difference; when the notebook takes the contract types the
functions become the identity and go away.
"""

from __future__ import annotations

from collections.abc import Mapping

from hyperloom.contracts import measurement as m
from hyperloom_multibranch.contracts.notebook import peers

_VALIDITY = {
    m.Validity.SUCCEEDED: peers.Validity.SUCCESS,
    m.Validity.CANDIDATE_FAILED: peers.Validity.CANDIDATE_FAILURE,
}


def notebook_request(request: m.TestRequest, priority: int) -> peers.TestRequest:
    return peers.TestRequest(
        request_id=request.request_id or "",
        kind=peers.TestKind(request.kind.value),
        branch=request.branch_id,
        parent_commit=request.parent_commit,
        intent_fingerprint=request.intent_fingerprint or "",
        delta=request.delta,
        purpose=request.purpose.value,
        priority=str(priority),
        diff_ref=request.diff_ref,
        retest_of=request.retest_of,
        avoid_nodes=request.avoid_nodes,
        accuracy_rounds=request.accuracy,
    )


def _sandbox(facts: m.SandboxFacts | None) -> peers.SandboxFacts:
    if facts is None:
        return peers.SandboxFacts(node="")
    return peers.SandboxFacts(
        node=facts.node_name or "",
        gpu_ids=facts.gpu_ids,
        gpu_model=facts.gpu_model or "",
        power_cap_watts=facts.power_cap_w,
        partition_mode=facts.partition_mode or "",
        image_digest=facts.image_digest or "",
        backend=facts.backend,
    )


def _attempt(attempt: m.Attempt) -> peers.Attempt:
    return peers.Attempt(
        node=attempt.node or "",
        gpu_ids=attempt.sandbox.gpu_ids if attempt.sandbox else (),
        failure=peers.FailureClass(attempt.failure_class.value) if attempt.failure_class else None,
        started_at=attempt.started_at or 0.0,
        finished_at=attempt.finished_at or 0.0,
    )


def _rounds(rounds: tuple[m.RoundValue, ...]) -> tuple[peers.RoundValues, ...]:
    """One row per (kind, metric), values in round order."""

    grouped: dict[tuple[peers.RoundKind, str], list[float]] = {}
    for value in rounds:
        kind = peers.RoundKind.WARMUP if value.phase == "warmup" else peers.RoundKind.FORMAL
        for metric, number in value.metrics.items():
            grouped.setdefault((kind, metric), []).append(number)
    return tuple(peers.RoundValues(kind, metric, tuple(values)) for (kind, metric), values in grouped.items())


def _accuracy(values: tuple[m.AccuracyValue, ...]) -> Mapping[str, float]:
    return {f"{value.task}/{value.metric}": value.score for value in values}


def notebook_result(result: m.TestResult) -> peers.TestResult:
    """The row for a measured test; ``measurement_failed`` is withdrawn instead, never written as a result."""

    if result.validity not in _VALIDITY:
        raise ValueError(f"a {result.validity.value} test has no notebook result row")
    return peers.TestResult(
        request_id=result.request_id,
        validity=_VALIDITY[result.validity],
        sandbox=_sandbox(result.sandbox),
        manifest_sha256=result.manifest_sha256 or "",
        rounds=_rounds(result.rounds),
        accuracy=_accuracy(result.accuracy),
        failure=peers.FailureClass(result.failure_class.value) if result.failure_class else None,
        failure_reason=result.failure_reason or "",
        attempts=tuple(map(_attempt, result.attempts)),
        queued_at=result.queued_at,
        started_at=result.started_at,
        finished_at=result.finished_at,
        gpu_seconds=result.gpu_seconds,
        profile_location=result.profile_location,
    )
