# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The startup brief's unfinished leads: what the session left worth finishing, and what drops out once taken up."""

from __future__ import annotations

from pathlib import Path

from hyperloom_multibranch.contracts.notebook import INTEGRATION_BRANCH, Role
from hyperloom_multibranch.contracts.notebook.peers import IntentLayer, Verdict
from hyperloom_multibranch.contracts.notebook.records import LineageEvent, MergeAttempt, Withdrawal
from hyperloom_multibranch.contracts.notebook.vocab import Conclusion, LineageKind, UnfinishedReason
from hyperloom_multibranch.notebook import LocalNotebook
from hyperloom_multibranch.notebook.local.unfinished import MAX_UNFINISHED
from hyperloom_multibranch.notebook.tests.factories import TTL, FakeClock, delta, intent, judgement, request, result
from hyperloom_multibranch.notebook.tests.sessions import METRIC, Writers, scripted_session
from hyperloom_multibranch.notebook.tools import UNFINISHED_TITLE, render_brief

LANDED = intent("s3", "i1", "sampler", "fused top-k")
NEAR = intent("s3", "i1", "attention", "fused rope")
RETRIED = intent("s3", "i1", "norm", "fused rms norm")
CLAIMED = intent("s3", "i1", "allreduce", "custom allreduce")
BLOCKED = intent("s3", "i1", "gemm", "fp8 gemm")
DROPPED = intent("s3", "i1", "scheduler", "chunked prefill 8k", IntentLayer.CONFIG)


async def _leftovers(root: Path, clock: FakeClock) -> LocalNotebook:
    """The scripted session, then ``s3`` from the integration head: a commit whose merge was not stackable, one on it
    not merge-tested yet, a near miss, a near miss tried again on ``s4``, a near miss ``s1`` claims, a blocked gain
    waiting for its diagnosis, and an explore withdrawn by the stop of ``s3``."""

    observer = await scripted_session(root, clock)
    w = Writers(root, clock)
    scheduler, gate, toolbox = w.as_role(Role.SCHEDULER), w.as_role(Role.GATE), w.as_role(Role.TOOLBOX)
    integrator = w.as_role(Role.INTEGRATOR)
    for branch, start in (("s3", "i1"), ("s4", "c0")):
        await scheduler.record_lineage(LineageEvent(LineageKind.OPEN, branch, from_commit=start, reason="refill"))
    await w.measure("s3", "u1", LANDED, delta("--sampler fused"))
    await gate.write_judgement("u1", judgement(3.0, ("u1",), recorded={METRIC: 1071.0}), Conclusion.COMMITTED, "c3")
    await toolbox.write_request(request("m2", intent(INTEGRATION_BRANCH, "i1", "merge", "c3")))
    await integrator.record_merge_attempt(MergeAttempt(("c3",), "i1", request_id="m2"))
    await toolbox.write_result(result("m2"))
    unstacked = judgement(-0.5, ("m2",), verdict=Verdict.REGRESSED)
    await integrator.write_judgement("m2", unstacked, Conclusion.NOT_STACKABLE)
    await w.measure("s3", "u2", intent("s3", "c3", "sampler", "fused min-p"), delta("--min-p fused"))
    await gate.write_judgement("u2", judgement(2.0, ("u2",), recorded={METRIC: 1090.0}), Conclusion.COMMITTED, "c4")
    for request_id, of in (("n1", NEAR), ("n2", RETRIED), ("n3", CLAIMED)):
        await w.measure("s3", request_id, of, delta(f"--{request_id}"))
        flat = judgement(0.8, (request_id,), verdict=Verdict.FLAT)
        await gate.write_judgement(request_id, flat, Conclusion.RECORDED_ONLY)
    await w.measure("s3", "d1", BLOCKED, delta("--fp8-gemm"))
    blocked = judgement(2.0, ("d1",), verdict=Verdict.BLOCKED)
    await gate.write_judgement("d1", blocked, Conclusion.DIAGNOSIS_PENDING)
    dropped = delta("--chunked-prefill-size 8192")
    claim = await w.as_role(Role.SPECIALIST_RUNTIME, "s3").acquire(DROPPED, dropped.fingerprint(), ttl_seconds=TTL)
    await toolbox.write_request(request("x1", DROPPED, dropped))
    await toolbox.attach_request(claim.key, "x1", ttl_seconds=TTL)
    await toolbox.write_withdrawal(Withdrawal("x1", "branch stopped"))
    await scheduler.record_lineage(LineageEvent(LineageKind.STOP, "s3", reason="no progress"))
    await scheduler.release_branch("s3")
    await w.measure("s4", "n2b", intent("s4", "c0", "norm", "fused rms norm"), delta("--n2b"))
    await w.as_role(Role.SPECIALIST_RUNTIME, "s1").acquire(
        intent("s1", "c1", "allreduce", "custom allreduce"), delta("--n3").fingerprint(), ttl_seconds=TTL
    )
    return observer


async def test_the_brief_lists_what_the_session_left_unfinished_in_order_and_drops_what_was_taken_up(
    tmp_path: Path,
) -> None:
    observer = await _leftovers(tmp_path / "notebook", FakeClock())

    brief = await observer.startup_brief("i1")

    assert [(item.request_id, item.reason) for item in brief.unfinished] == [
        ("u1", UnfinishedReason.NOT_STACKABLE),
        ("n1", UnfinishedReason.NEAR_BAR),
        ("d1", UnfinishedReason.REPAIR_UNFINISHED),
        ("x1", UnfinishedReason.WITHDRAWN),
    ]
    landed, near, _, withdrawn = brief.unfinished
    assert (landed.branch, landed.measured_on, landed.gain_pct) == ("s3", "i1", 3.0)
    assert landed.detail == "commit c3; merge m2 on i1 -0.50%, bar 1.00%"
    assert (near.target, near.mechanism, near.gain_pct, near.detail) == ("attention", "fused rope", 0.8, "bar 1.00%")
    assert (withdrawn.gain_pct, withdrawn.detail) == (None, "branch stopped")
    assert withdrawn.change == "add args --chunked-prefill-size 8192"
    assert len(brief.unfinished) <= MAX_UNFINISHED


async def test_a_running_branchs_commits_and_the_merged_ones_are_not_unfinished(tmp_path: Path) -> None:
    observer = await scripted_session(tmp_path / "notebook", FakeClock())

    brief = await observer.startup_brief("i1")

    assert brief.unfinished == ()


async def test_the_rendered_brief_names_each_lead_and_how_to_finish_it(tmp_path: Path) -> None:
    observer = await _leftovers(tmp_path / "notebook", FakeClock())

    text = render_brief(await observer.startup_brief("i1"))
    lines = text.splitlines()
    start = lines.index(UNFINISHED_TITLE)

    assert "notebook_replay" in UNFINISHED_TITLE and lines.index("Tried:") < start
    assert lines[start + 1] == (
        "- u1 on s3: sampler / fused top-k (add args --sampler fused) on i1 +3.00%: committed, merge not stackable "
        "(commit c3; merge m2 on i1 -0.50%, bar 1.00%)"
    )
    assert lines[start + 4] == (
        "- x1 on s3: scheduler / chunked prefill 8k (add args --chunked-prefill-size 8192) on i1: withdrawn before its "
        "result (branch stopped)"
    )
