# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""One whole session written through the local notebook, for the tests of everything that reads a finished notebook.

* ``c0`` is the root; the rule gate baselines it at 1000 output tokens/s.
* ``s1`` (kernel layer): ``r1`` tunes the MoE GEMM and commits ``c1`` (+4%); ``r4`` tunes it again on ``c1`` and is
  flat.
* ``s2`` (config layer): ``r2`` splits the KV cache and regresses (-2%), ``r3`` fails to launch, ``r5`` is faster but
  fails the accuracy check. The scheduler stops ``s2``.
* The integrator merges ``c1`` onto ``c0`` as ``i1`` (``m1``).
* ``s1`` confirms a hypothesis, ``s2`` falsifies one.
* LLM calls cost $1.25 for ``s1``'s specialist, $0.50 for ``s2``'s, and $0.125 each for the judge and the resolver.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from hyperloom.contracts.recipe import RecipeDelta
from hyperloom_multibranch.contracts.notebook import INTEGRATION_BRANCH, Role, Writer
from hyperloom_multibranch.contracts.notebook.peers import (
    BranchLogEntry,
    BranchLogType,
    HypothesisStatus,
    Intent,
    IntentLayer,
    TestResult,
    Validity,
    Verdict,
)
from hyperloom_multibranch.contracts.notebook.records import (
    BaselineRecord,
    LineageEvent,
    LlmUsage,
    ManifestFingerprint,
    MergeAttempt,
    NotebookSnapshot,
)
from hyperloom_multibranch.contracts.notebook.vocab import Conclusion, LineageKind, UsageSource
from hyperloom_multibranch.notebook import LocalNotebook
from hyperloom_multibranch.notebook.tests.factories import (
    TTL,
    FakeClock,
    delta,
    intent,
    judgement,
    profile,
    request,
    result,
)

METRIC = "output_tps"
BASELINE = {METRIC: 1000.0}
GEMM = intent("s1", "c0")
SPLIT_KV = intent("s2", "c0", "attention", "split kv", IntentLayer.CONFIG)
CHUNKED_PREFILL = intent("s2", "c0", "scheduler", "chunked prefill", IntentLayer.CONFIG)
FP8_KV = intent("s2", "c0", "kv cache", "fp8 kv cache", IntentLayer.CONFIG)
GEMM_AGAIN = intent("s1", "c1")


@dataclass(frozen=True)
class Writers:
    root: Path
    clock: FakeClock

    def as_role(self, role: Role, branch: str = "") -> LocalNotebook:
        return LocalNotebook(self.root, Writer(role, branch or role.value, branch), clock=self.clock)

    async def measure(
        self, branch: str, request_id: str, of: Intent, change: RecipeDelta, outcome: TestResult | None = None
    ) -> None:
        claim = await self.as_role(Role.SPECIALIST_RUNTIME, branch).acquire(of, change.fingerprint(), ttl_seconds=TTL)
        toolbox = self.as_role(Role.TOOLBOX)
        await toolbox.write_request(request(request_id, of, change))
        await toolbox.attach_request(claim.key, request_id, ttl_seconds=TTL)
        await toolbox.write_result(outcome or result(request_id))


async def scripted_session(root: Path, clock: FakeClock) -> LocalNotebook:
    """Write the session this module describes and return an observer of it."""

    w = Writers(root, clock)
    scheduler, gate, integrator = w.as_role(Role.SCHEDULER), w.as_role(Role.GATE), w.as_role(Role.INTEGRATOR)
    await scheduler.write_session_profile(profile())
    await scheduler.write_manifest_fingerprint(
        ManifestFingerprint("act-1", {"run_test": "local-1"}, {"specialist": "p1"}, {"scheduler": "s1"})
    )
    await scheduler.record_lineage(LineageEvent(LineageKind.ROOT, INTEGRATION_BRANCH, commit_id="c0"))
    await gate.write_baseline(BaselineRecord("c0", BASELINE, {METRIC: 0.8}, {METRIC: 1.6}, ("b1", "b2")))
    for branch in ("s1", "s2"):
        await scheduler.record_lineage(LineageEvent(LineageKind.OPEN, branch, from_commit="c0", reason="first batch"))

    await w.measure("s1", "r1", GEMM, delta("--max-num-seqs 256"))
    await gate.write_judgement("r1", judgement(4.0, ("r1",), recorded={METRIC: 1040.0}), Conclusion.COMMITTED, "c1")
    await w.measure("s2", "r2", SPLIT_KV, delta("--block-size 32"))
    regressed = judgement(-2.0, ("r2",), recorded={METRIC: 980.0}, verdict=Verdict.REGRESSED)
    await gate.write_judgement("r2", regressed, Conclusion.RECORDED_ONLY)
    launch_failure = result("r3", validity=Validity.CANDIDATE_FAILED, failure_reason="server exited during warmup")
    await w.measure("s2", "r3", CHUNKED_PREFILL, delta("--enable-chunked-prefill"), launch_failure)
    await w.measure("s2", "r5", FP8_KV, delta("--kv-cache-dtype fp8"))
    await gate.write_judgement("r5", judgement(6.0, ("r5",), recorded={METRIC: 1060.0}), Conclusion.ACCURACY_FAILED)
    await w.measure("s1", "r4", GEMM_AGAIN, delta("--max-num-seqs 512"))
    flat = judgement(0.3, ("r4",), recorded={METRIC: 1043.0}, verdict=Verdict.FLAT)
    await gate.write_judgement("r4", flat, Conclusion.RECORDED_ONLY)

    await w.as_role(Role.TOOLBOX).write_request(request("m1", intent(INTEGRATION_BRANCH, "c0", "merge", "c1")))
    await integrator.record_merge_attempt(MergeAttempt(("c1",), "c0", request_id="m1"))
    await w.as_role(Role.TOOLBOX).write_result(result("m1", node="node-2"))
    merged = judgement(4.1, ("m1",), recorded={METRIC: 1041.0})
    await integrator.write_judgement("m1", merged, Conclusion.MERGED, "i1")

    confirmed = BranchLogEntry(
        "s1",
        BranchLogType.HYPOTHESIS,
        "the MoE GEMM is bound by memory bandwidth",
        ("r1",),
        None,
        HypothesisStatus.CONFIRMED,
    )
    falsified = BranchLogEntry(
        "s2",
        BranchLogType.HYPOTHESIS,
        "splitting the KV cache helps at this concurrency",
        ("r2",),
        None,
        HypothesisStatus.FALSIFIED,
    )
    await w.as_role(Role.SPECIALIST_RUNTIME, "s1").append_branch_log(confirmed)
    await w.as_role(Role.SPECIALIST_RUNTIME, "s2").append_branch_log(falsified)
    for branch, cost in (("s1", 1.25), ("s2", 0.5)):
        await w.as_role(Role.SPECIALIST_RUNTIME, branch).record_llm_usage(
            LlmUsage(branch, UsageSource.SPECIALIST, cost)
        )
    for source in (UsageSource.JUDGE, UsageSource.RESOLVER):
        await scheduler.record_llm_usage(LlmUsage("", source, 0.125))
    await scheduler.record_lineage(LineageEvent(LineageKind.STOP, "s2", reason="no progress"))
    await scheduler.release_branch("s2")
    return w.as_role(Role.OBSERVER)


@pytest.fixture
async def snapshot(tmp_path: Path) -> NotebookSnapshot:
    """The scripted session's snapshot; a test package's conftest imports this fixture to use it."""

    return await (await scripted_session(tmp_path / "notebook", FakeClock())).snapshot()
