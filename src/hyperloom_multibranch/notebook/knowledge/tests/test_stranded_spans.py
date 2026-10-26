# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""How long a branch's lead over integration stayed stranded, as CLOSE's breakdown records it."""

from __future__ import annotations

from pathlib import Path

import pytest

from hyperloom_multibranch.contracts.notebook import INTEGRATION_BRANCH, Role
from hyperloom_multibranch.contracts.notebook.peers import Purpose, Validity
from hyperloom_multibranch.contracts.notebook.records import LineageEvent, MergeAttempt, NotebookSnapshot
from hyperloom_multibranch.contracts.notebook.vocab import Conclusion, LineageKind, MergeOutcome
from hyperloom_multibranch.notebook.knowledge.breakdown import CloseOutcome, build_breakdown
from hyperloom_multibranch.notebook.knowledge.stranded_spans import stranded_spans
from hyperloom_multibranch.notebook.tests.factories import FakeClock, delta, intent, judgement, request, result
from hyperloom_multibranch.notebook.tests.sessions import METRIC, Writers, scripted_session

MINUTE = 60.0
FUSED = delta("--sampler fused")


async def _stranded_then_merged(root: Path, clock: FakeClock) -> tuple[NotebookSnapshot, float]:
    """The scripted session, then ``s3`` from integration's ``i1`` (1041): its ``c3`` (1124, +8% over it) commits, its
    merge is skipped for a conflict 10 minutes later, and a second merge takes it onto integration 30 minutes after
    that, as ``i2`` (1120); the snapshot is taken 20 minutes later. Also when the merge failed."""

    await scripted_session(root, clock)
    w = Writers(root, clock)
    scheduler, gate, integrator, toolbox = (
        w.as_role(role) for role in (Role.SCHEDULER, Role.GATE, Role.INTEGRATOR, Role.TOOLBOX)
    )
    await scheduler.record_lineage(LineageEvent(LineageKind.OPEN, "s3", from_commit="i1", reason="refill"))
    await w.measure("s3", "u1", intent("s3", "i1", "sampler", "fused top-k"), FUSED)
    await gate.write_judgement("u1", judgement(8.0, ("u1",), recorded={METRIC: 1124.0}), Conclusion.COMMITTED, "c3")
    clock.advance(10 * MINUTE)
    await integrator.record_merge_attempt(MergeAttempt(("c3",), "i1", skipped_reason="unresolved patch conflict"))
    failed_at = clock.now
    clock.advance(30 * MINUTE)
    merge = intent(INTEGRATION_BRANCH, "i1", "merge", "c3")
    await toolbox.write_request(request("m2", merge, FUSED, purpose=Purpose.INTEGRATION))
    await integrator.record_merge_attempt(MergeAttempt(("c3",), "i1", request_id="m2"))
    await toolbox.write_result(result("m2"))
    await integrator.write_judgement("m2", judgement(7.6, ("m2",), recorded={METRIC: 1120.0}), Conclusion.MERGED, "i2")
    clock.advance(20 * MINUTE)
    return await w.as_role(Role.OBSERVER).snapshot(), failed_at


async def test_a_lead_over_the_bar_holds_from_its_first_failed_merge_until_integration_catches_up(
    tmp_path: Path,
) -> None:
    snapshot, failed_at = await _stranded_then_merged(tmp_path / "notebook", FakeClock())

    (span,) = stranded_spans(snapshot, METRIC)

    assert (span.branch, span.metric, span.bar_pct, span.failed_commits) == ("s3", METRIC, 1.6, 1)
    assert span.first_at == span.peak_at == pytest.approx(failed_at, abs=1.0)
    assert span.minutes == pytest.approx(30.0, abs=0.1)
    assert span.peak_lead_pct == pytest.approx((1124.0 / 1041.0 - 1.0) * 100.0)
    assert stranded_spans(snapshot, "another_metric") == ()


async def test_a_commit_keeps_when_its_first_merge_failed_once_a_later_one_merges_it(tmp_path: Path) -> None:
    snapshot, failed_at = await _stranded_then_merged(tmp_path / "notebook", FakeClock())

    (c3,) = (commit for commit in snapshot.committed if commit.commit_id == "c3")

    assert (c3.merged_into, c3.merge_attempts) == ("i2", 2)
    assert c3.last_merge is not None and c3.last_merge.outcome is MergeOutcome.MERGED
    assert c3.first_failed_at == pytest.approx(failed_at, abs=1.0)
    assert all(commit.first_failed_at is None for commit in snapshot.committed if commit.commit_id != "c3")


async def test_a_merge_test_the_candidate_failed_fails_its_commit_and_a_failed_measurement_does_not(
    tmp_path: Path,
) -> None:
    root, clock = tmp_path / "notebook", FakeClock()
    await scripted_session(root, clock)
    w = Writers(root, clock)
    scheduler, gate, integrator, toolbox = (
        w.as_role(role) for role in (Role.SCHEDULER, Role.GATE, Role.INTEGRATOR, Role.TOOLBOX)
    )
    tested = (("s3", "c3", "m2", Validity.CANDIDATE_FAILED), ("s4", "c4", "m3", Validity.MEASUREMENT_FAILED))
    for branch, commit, _, _ in tested:
        change = delta(f"--{branch}-fused")
        await scheduler.record_lineage(LineageEvent(LineageKind.OPEN, branch, from_commit="i1", reason="refill"))
        await w.measure(branch, f"u-{branch}", intent(branch, "i1", "sampler", f"fused {branch}"), change)
        kept = judgement(8.0, (f"u-{branch}",), recorded={METRIC: 1124.0})
        await gate.write_judgement(f"u-{branch}", kept, Conclusion.COMMITTED, commit)
    clock.advance(10 * MINUTE)
    failed_at = clock.now
    for branch, commit, merge, validity in tested:
        merging = intent(INTEGRATION_BRANCH, "i1", "merge", commit)
        await toolbox.write_request(request(merge, merging, delta(f"--{branch}-fused"), purpose=Purpose.INTEGRATION))
        await integrator.record_merge_attempt(MergeAttempt((commit,), "i1", request_id=merge))
        await toolbox.write_result(result(merge, validity=validity, failure_reason="server exited during warmup"))
    clock.advance(30 * MINUTE)
    snapshot = await w.as_role(Role.OBSERVER).snapshot()

    failed = {commit.commit_id: commit.first_failed_at for commit in snapshot.committed if commit.branch != "s1"}
    merges = {commit.commit_id: commit.last_merge for commit in snapshot.committed if commit.branch != "s1"}

    assert failed["c3"] == pytest.approx(failed_at, abs=1.0)
    assert failed["c4"] is None
    assert merges["c3"] is not None and merges["c4"] is not None
    assert (merges["c3"].outcome, merges["c3"].reason) == (
        MergeOutcome.NOT_STACKABLE,
        "its merge test failed: server exited during warmup",
    )
    assert merges["c4"].outcome is MergeOutcome.PENDING
    (span,) = stranded_spans(snapshot, METRIC)
    assert (span.branch, span.minutes) == ("s3", pytest.approx(30.0, abs=0.1))


async def test_the_breakdown_holds_the_spans_when_the_scheduler_names_the_metric(tmp_path: Path) -> None:
    snapshot, _ = await _stranded_then_merged(tmp_path / "notebook", FakeClock())

    close = build_breakdown(snapshot, CloseOutcome(metric=METRIC))["close"]

    assert [(row["branch"], row["failed_commits"]) for row in close["stranded"]] == [("s3", 1)]
    assert "stranded" not in build_breakdown(snapshot)["close"]
