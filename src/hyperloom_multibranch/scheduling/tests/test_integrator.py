# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The integrator against the local notebook, on the peer stand-ins for the repository, toolbox and rule gate.

The notebook holds the root ``c0`` and two commits of ``s1``: ``c1`` (+4%) and ``c2`` (+2%), so ``c1`` is tried first.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from hyperloom.contracts.recipe import (
    ComponentChange,
    ComponentPatches,
    LaunchConfig,
    PatchRef,
    Recipe,
    RecipeDelta,
    ValueChange,
)
from hyperloom_multibranch.contracts.notebook import INTEGRATION_BRANCH, Role, Writer
from hyperloom_multibranch.contracts.notebook.peers import Judgement, TestKind, Verdict
from hyperloom_multibranch.contracts.notebook.records import LineageEvent
from hyperloom_multibranch.contracts.notebook.vocab import Conclusion, LineageKind, MergeOutcome
from hyperloom_multibranch.contracts.scheduling import PatchConflict, PriorityClass, RepoMerge
from hyperloom_multibranch.notebook import LocalNotebook
from hyperloom_multibranch.notebook.tests.factories import FakeClock, delta, intent, judgement, request, result
from hyperloom_multibranch.scheduling.clock import Clock
from hyperloom_multibranch.scheduling.decisions import DecisionKind, DecisionLog
from hyperloom_multibranch.scheduling.integrator import (
    MERGE_PURPOSE,
    RESOLVER_SYSTEM_PROMPT,
    SUPERSEDED,
    Integrator,
    LlmPatchResolver,
    MergeReport,
    merge_intent,
)
from hyperloom_multibranch.scheduling.llm import CompletionFailed
from hyperloom_multibranch.scheduling.mb_fakes import MB_FAKE_BASE_IMAGE_DIGEST
from hyperloom_multibranch.scheduling.tests.peer_fakes import MbFakeGate, MbFakeRepo, MbFakeToolbox, ScriptedCompletion

COMPONENT = "vllm"
UPSTREAM_REF = "a" * 40
HEAD_PATCH = "patch/vllm/0001-head.patch"
CANDIDATE_PATCH = "patch/vllm/0002-candidate.patch"
RESOLVED_TEXT = "--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-old\n+resolved\n"
GAINS = {"c1": 4.0, "c2": 2.0}
#: More merge tests than any round compares at once, so the gate stand-in never answers.
HELD_FOREVER = 99


def recipe(server_args: str = "", patches: tuple[str, ...] = ()) -> Recipe:
    refs = tuple(PatchRef(path, hashlib.sha256(path.encode()).hexdigest()) for path in patches)
    component = ComponentPatches(COMPONENT, "https://example.com/vllm.git", UPSTREAM_REF, refs)
    return Recipe(MB_FAKE_BASE_IMAGE_DIGEST, LaunchConfig(server_args=server_args), (component,))


def clean_merge(server_args: str) -> RepoMerge:
    return RepoMerge("", delta(server_args))


def improved() -> Judgement:
    return judgement(3.0, (), recorded={"output_throughput": 1030.0})


@dataclass
class Bench:
    root: Path
    clock: FakeClock
    repo: MbFakeRepo
    toolbox: MbFakeToolbox
    gate: MbFakeGate
    completion: ScriptedCompletion
    integrator: Integrator
    decisions: Path

    def handle(self, role: Role) -> LocalNotebook:
        return LocalNotebook(self.root, Writer(role, role.value), clock=self.clock)

    async def outcome(self, commit: str) -> MergeOutcome | None:
        found = next(item for item in await self.handle(Role.OBSERVER).committed() if item.commit_id == commit)
        return found.last_merge.outcome if found.last_merge is not None else None

    def logged(self) -> list[dict[str, object]]:
        return [json.loads(line) for line in self.decisions.read_text().splitlines()]


async def _one(bench: Bench) -> MergeReport | None:
    """A round one merge wide: the report of the one candidate tried, or ``None`` when none was."""

    reports = await bench.integrator.integrate_next(1)
    assert len(reports) <= 1
    return reports[0] if reports else None


async def _commit_specialist_work(bench: Bench) -> None:
    scheduler, toolbox, gate = (bench.handle(role) for role in (Role.SCHEDULER, Role.TOOLBOX, Role.GATE))
    await scheduler.record_lineage(LineageEvent(LineageKind.ROOT, INTEGRATION_BRANCH, commit_id="c0"))
    await scheduler.record_lineage(LineageEvent(LineageKind.OPEN, "s1", from_commit="c0", reason="first batch"))
    for parent, commit in (("c0", "c1"), ("c1", "c2")):
        test = request(f"r{commit}", intent("s1", parent, mechanism=commit), delta(f"--max-num-seqs {commit[1:]}"))
        await toolbox.write_request(test)
        await toolbox.write_result(result(test.request_id))
        made = judgement(GAINS[commit], (test.request_id,))
        await gate.write_judgement(test.request_id, made, Conclusion.COMMITTED, commit)


@pytest.fixture
async def bench(tmp_path: Path) -> Bench:
    clock = FakeClock()
    root = tmp_path / "notebook"
    repo = MbFakeRepo(
        recipes={"c0": recipe(patches=(HEAD_PATCH,)), "c1": recipe("--max-num-seqs 256", (CANDIDATE_PATCH,))},
        integration_head="c0",
        merges={"c1": clean_merge("--max-num-seqs 256"), "c2": clean_merge("--max-num-seqs 128")},
        patches={("c0", HEAD_PATCH): "head patch text", ("c1", CANDIDATE_PATCH): "candidate patch text"},
    )
    repo.recipes["c2"] = recipe("--max-num-seqs 128")
    toolbox = MbFakeToolbox(LocalNotebook(root, Writer(Role.TOOLBOX, "toolbox"), clock=clock))
    gate = MbFakeGate(LocalNotebook(root, Writer(Role.GATE, "gate"), clock=clock), toolbox, default=improved())
    completion = ScriptedCompletion()
    decisions = tmp_path / "MB-decisions.jsonl"
    integrator = Integrator(
        LocalNotebook(root, Writer(Role.INTEGRATOR, "integrator"), clock=clock),
        repo,
        toolbox,
        gate,
        LlmPatchResolver(completion, repo),
        DecisionLog(decisions, Clock(now=clock)),
    )
    made = Bench(root, clock, repo, toolbox, gate, completion, integrator, decisions)
    await _commit_specialist_work(made)
    return made


async def test_an_improving_merge_is_tested_from_the_head_then_written(bench: Bench) -> None:
    conflict = ValueChange("--max-num-seqs", "64", "256")
    bench.repo.merges = {**bench.repo.merges, "c1": replace(bench.repo.merges["c1"], config_conflicts=(conflict,))}

    report = await _one(bench)

    assert report is not None and report.outcome is MergeOutcome.MERGED
    assert (report.candidate, report.onto, report.merged_commit) == ("c1", "c0", "i1")
    [test] = bench.toolbox.submitted
    assert (test.kind, test.branch, test.parent_commit) == (TestKind.TEST, INTEGRATION_BRANCH, "c0")
    assert (test.intent_fingerprint, test.purpose) == (merge_intent("c1"), MERGE_PURPOSE)
    assert test.priority == PriorityClass.INTEGRATION_MERGE.value
    assert bench.repo.integrated == [("c0", delta("--max-num-seqs 256"), ("c1",))]
    record = await bench.handle(Role.OBSERVER).record(test.request_id)
    assert (record.conclusion, record.commit_id) == (Conclusion.MERGED, "i1")
    lineage = await bench.handle(Role.OBSERVER).lineage()
    assert lineage.commit("i1").recipe_digest == bench.repo.recipes["i1"].digest()
    [logged] = bench.logged()
    assert (logged["kind"], logged["branch"]) == (DecisionKind.MERGE.value, INTEGRATION_BRANCH)
    assert logged["detail"]["config_conflicts"] == [{"name": "--max-num-seqs", "before": "64", "after": "256"}]


@pytest.mark.parametrize(
    "outcome",
    [
        replace(improved(), verdict=Verdict.FLAT, gain_pct=0.2),
        replace(improved(), verdict=Verdict.REGRESSED, gain_pct=-3.0),
        replace(improved(), confirmed=False),
    ],
    ids=["flat", "regressed", "unconfirmed"],
)
async def test_a_merge_that_does_not_improve_is_not_stackable(bench: Bench, outcome: Judgement) -> None:
    bench.gate.verdicts = {merge_intent("c1"): outcome}

    report = await _one(bench)

    assert report is not None and report.outcome is MergeOutcome.NOT_STACKABLE and report.reason
    assert bench.repo.integrated == [] and bench.repo.integration_head == "c0"
    record = await bench.handle(Role.OBSERVER).record(str(report.request_id))
    assert (record.conclusion, record.commit_id) == (Conclusion.NOT_STACKABLE, None)
    assert await bench.outcome("c1") is MergeOutcome.NOT_STACKABLE


async def test_each_commit_is_tried_once_per_integration_head(bench: Bench) -> None:
    bench.gate.verdicts = {merge_intent("c1"): replace(improved(), verdict=Verdict.FLAT)}

    tried = [await _one(bench) for _ in range(4)]

    assert [(report.candidate, report.onto, report.outcome) for report in tried if report is not None] == [
        ("c1", "c0", MergeOutcome.NOT_STACKABLE),
        ("c2", "c0", MergeOutcome.MERGED),
        ("c1", "i1", MergeOutcome.NOT_STACKABLE),
    ]
    assert tried[3] is None


async def test_a_round_tests_k_candidates_at_once_and_merges_the_first_improvement(bench: Bench) -> None:
    bench.gate.held_until = 2

    first = await bench.integrator.integrate_next(2)
    second = await bench.integrator.integrate_next(2)

    assert [(r.candidate, r.onto, r.outcome, r.reason) for r in first] == [
        ("c1", "c0", MergeOutcome.MERGED, "improved +3.00% on the integration head"),
        ("c2", "c0", MergeOutcome.PENDING, SUPERSEDED),
    ]
    superseded = await bench.handle(Role.OBSERVER).record(str(first[1].request_id))
    assert superseded.conclusion is None
    assert [(r.candidate, r.onto, r.outcome) for r in second] == [("c2", "i1", MergeOutcome.MERGED)]
    assert [merged_from for _, _, merged_from in bench.repo.integrated] == [("c1",), ("c2",)]
    assert await bench.integrator.integrate_next(2) == ()


async def test_commits_of_a_stopped_branch_stay_eligible(bench: Bench) -> None:
    scheduler = bench.handle(Role.SCHEDULER)
    await scheduler.record_lineage(LineageEvent(LineageKind.STOP, "s1", reason="judge: no progress"))
    await scheduler.release_branch("s1")

    report = await _one(bench)

    assert report is not None and (report.candidate, report.outcome) == ("c1", MergeOutcome.MERGED)


async def test_a_resolved_patch_conflict_is_added_to_the_merged_change(bench: Bench) -> None:
    conflict = PatchConflict(COMPONENT, CANDIDATE_PATCH, "hunk 1 FAILED")
    bench.repo.merges = {**bench.repo.merges, "c1": replace(bench.repo.merges["c1"], patch_conflicts=(conflict,))}
    bench.completion.fallback = "```json\n" + json.dumps({"patch": RESOLVED_TEXT}) + "\n```"

    report = await _one(bench)

    assert report is not None and report.outcome is MergeOutcome.MERGED
    [(stored, text)] = bench.repo.stored
    assert text == RESOLVED_TEXT
    [test] = bench.toolbox.submitted
    assert test.delta == replace(
        delta("--max-num-seqs 256"), components=(ComponentChange(COMPONENT, add_patches=(stored,)),)
    )
    [(system, prompt)] = bench.completion.calls
    assert system == RESOLVER_SYSTEM_PROMPT
    assert all(text in prompt for text in ("candidate patch text", "head patch text", "hunk 1 FAILED"))


async def test_a_conflict_on_a_component_the_delta_changes_extends_that_change(bench: Bench) -> None:
    moved = ComponentChange(COMPONENT, upstream_ref="b" * 40)
    conflict = PatchConflict(COMPONENT, HEAD_PATCH, "does not apply on the new ref")
    bench.repo.merges = {"c1": RepoMerge("", RecipeDelta(components=(moved,)), patch_conflicts=(conflict,))}
    bench.completion.fallback = json.dumps({"patch": RESOLVED_TEXT})

    await _one(bench)

    [(stored, _)] = bench.repo.stored
    [test] = bench.toolbox.submitted
    assert test.delta == RecipeDelta(components=(replace(moved, add_patches=(stored,)),))
    assert "head patch text" in bench.completion.calls[0][1]


@pytest.mark.parametrize(
    "scripted",
    [
        ScriptedCompletion(fallback=json.dumps({"patch": None})),
        ScriptedCompletion(error=CompletionFailed("patch_resolve call failed: RuntimeError: status=529")),
    ],
    ids=["model-declines", "call-fails"],
)
async def test_an_unresolved_patch_conflict_is_skipped_and_recorded(bench: Bench, scripted: ScriptedCompletion) -> None:
    bench.completion.fallback, bench.completion.error = scripted.fallback, scripted.error
    conflict = PatchConflict(COMPONENT, CANDIDATE_PATCH, "hunk 1 FAILED")
    bench.repo.merges = {**bench.repo.merges, "c1": RepoMerge("", RecipeDelta(), patch_conflicts=(conflict,))}

    report = await _one(bench)

    assert report is not None and report.outcome is MergeOutcome.CONFLICT_SKIPPED
    assert CANDIDATE_PATCH in report.reason and report.request_id is None
    assert await bench.outcome("c1") is MergeOutcome.CONFLICT_SKIPPED
    assert [test.intent_fingerprint for test in bench.toolbox.submitted] == []
    following = await _one(bench)
    assert following is not None and following.candidate == "c2"


async def test_a_merge_the_recipe_contract_refuses_is_skipped_and_the_round_goes_on(bench: Bench) -> None:
    bench.repo.invalid_merges = frozenset({"c1"})

    reports = await bench.integrator.integrate_next(2)

    assert [(r.candidate, r.outcome) for r in reports] == [
        ("c1", MergeOutcome.CONFLICT_SKIPPED),
        ("c2", MergeOutcome.MERGED),
    ]
    assert "invalid" in reports[0].reason
    assert await bench.outcome("c1") is MergeOutcome.CONFLICT_SKIPPED


async def test_a_withdrawn_merge_test_writes_nothing_to_the_integration_branch(bench: Bench) -> None:
    bench.toolbox.withdrawing = frozenset({"c0"})

    report = await _one(bench)

    assert report is not None and report.outcome is MergeOutcome.PENDING and report.request_id
    assert bench.repo.integrated == []
    assert await bench.outcome("c1") is MergeOutcome.PENDING


async def test_nothing_to_integrate_once_every_commit_is_merged(bench: Bench) -> None:
    merged = [await _one(bench) for _ in range(2)]

    assert [report.outcome for report in merged if report is not None] == [MergeOutcome.MERGED] * 2
    assert await _one(bench) is None
    assert len(bench.logged()) == 2


async def _cancel(task: asyncio.Task[object]) -> None:
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_a_merge_cancelled_during_its_test_is_abandoned_and_its_test_withdrawn(bench: Bench) -> None:
    bench.gate.held_until = HELD_FOREVER
    merging = asyncio.ensure_future(bench.integrator.integrate_next(1))
    await bench.gate.comparing.wait()

    await _cancel(merging)
    await bench.integrator.abandon()

    assert bench.toolbox.withdrawn == [INTEGRATION_BRANCH]
    assert bench.repo.integrated == []


async def test_a_merge_cancelled_while_committing_is_still_recorded(bench: Bench) -> None:
    bench.repo.commit_released = asyncio.Event()
    merging = asyncio.ensure_future(bench.integrator.integrate_next(1))
    await bench.repo.committing.wait()

    await _cancel(merging)
    abandoning = asyncio.ensure_future(bench.integrator.abandon())
    bench.repo.commit_released.set()
    await abandoning

    [test] = bench.toolbox.submitted
    record = await bench.handle(Role.OBSERVER).record(test.request_id)
    assert (record.conclusion, record.commit_id) == (Conclusion.MERGED, bench.repo.integration_head)
    assert bench.toolbox.withdrawn == [INTEGRATION_BRANCH]
