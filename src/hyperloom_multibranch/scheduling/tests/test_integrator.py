# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The integrator against the local notebook, the git experiment repository, the toolbox's fake and the rule.

The root pins ``vllm`` with one patch and ``--max-num-seqs 64``; branch ``s1`` commits ``c1`` (+4%, a patch and
``--max-num-seqs 256``) and then ``c2`` (+2%, ``--max-num-seqs 128``), so ``c1`` is tried first. What a merge test
measures follows from the tags of the recipe it builds (:data:`CURVES`).
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pytest

from hyperloom.contracts.measurement import Purpose, TestKind, TestRequest
from hyperloom.contracts.recipe import (
    ComponentChange,
    ComponentPatches,
    ConfigDelta,
    LaunchConfig,
    PatchRef,
    Recipe,
    RecipeDelta,
    apply,
)
from hyperloom.measure.compile import LocalBlobStore
from hyperloom_multibranch.contracts.notebook import INTEGRATION_BRANCH, Role, Writer
from hyperloom_multibranch.contracts.notebook.records import LineageEvent
from hyperloom_multibranch.contracts.notebook.vocab import Conclusion, LineageKind, MergeOutcome
from hyperloom_multibranch.contracts.toolbox import RequestRejected
from hyperloom_multibranch.notebook import LocalNotebook
from hyperloom_multibranch.notebook.tests.factories import FakeClock, intent, judgement, request, result
from hyperloom_multibranch.repo.git_repo import GitExperimentRepo
from hyperloom_multibranch.scheduling.clock import Clock
from hyperloom_multibranch.scheduling.decisions import DecisionKind, DecisionLog
from hyperloom_multibranch.scheduling.integrator import (
    RESOLVER_SYSTEM_PROMPT,
    SUPERSEDED,
    Integrator,
    LlmPatchResolver,
    MergeReport,
)
from hyperloom_multibranch.scheduling.llm import CompletionFailed
from hyperloom_multibranch.scheduling.mb_fakes import MB_FAKE_BASE_IMAGE_DIGEST
from hyperloom_multibranch.scheduling.patches import PatchStore
from hyperloom_multibranch.scheduling.requests import SessionTests
from hyperloom_multibranch.scheduling.rule_judge import MbFakeRuleJudge
from hyperloom_multibranch.scheduling.tests.peer_fakes import (
    MAIN_POINT,
    TAG_FLAG,
    Curves,
    RecordingToolbox,
    ScriptedCompletion,
    recipe_effect,
)
from hyperloom_multibranch.toolbox.fake_model import Fate

COMPONENT = "vllm"
UPSTREAM_REF = "a" * 40
UPSTREAM_REPO = "https://example.com/vllm.git"
HEAD_TEXT = "--- a/head.py\n+++ b/head.py\n@@ -1 +1 @@\n-old\n+head\n"
CANDIDATE_TEXT = "--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-old\n+candidate\n"
RESOLVED_TEXT = "--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-old\n+resolved\n"
REFUSAL = "hunk 1 FAILED at x.py"
GAINS = {"c1": 4.0, "c2": 2.0}
ARGS = {"c1": "--max-num-seqs 256", "c2": "--max-num-seqs 128"}
#: ``c1`` merged onto the root gains 3%, ``c2`` 2%, and ``c2`` onto a head holding ``c1`` gains another 1.9%.
CURVES: Curves = {
    frozenset({"c1"}): {MAIN_POINT: 1030.0},
    frozenset({"c2"}): {MAIN_POINT: 1020.0},
    frozenset({"c1", "c2"}): {MAIN_POINT: 1050.0},
}


async def _instant(seconds: float) -> None:
    await asyncio.sleep(0)


class HeldRetests(RecordingToolbox):
    """Holds every attempt once a confirming retest is submitted; a test double, not a placeholder."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.confirming = asyncio.Event()

    async def submit_test(self, request: TestRequest) -> str:
        if request.purpose is Purpose.CONFIRM:
            self.executor.gate.clear()
            self.confirming.set()
        return await super().submit_test(request)


class HeldRepo(GitExperimentRepo):
    """Holds every merge until ``released`` is set; a test double, not a placeholder."""

    committing: asyncio.Event
    released: asyncio.Event

    async def commit(self, branch: str, parent: str, recipe: Recipe, message: str) -> str:
        if branch == INTEGRATION_BRANCH:
            self.committing.set()
            await self.released.wait()
        return await super().commit(branch, parent, recipe, message)


@dataclass
class Bench:
    root_dir: Path
    clock: FakeClock
    repo: GitExperimentRepo
    root: str
    commits: dict[str, str]
    deltas: dict[str, RecipeDelta]
    patches: PatchStore
    toolbox: RecordingToolbox
    completion: ScriptedCompletion
    integrator: Integrator
    decisions: Path

    def handle(self, role: Role) -> LocalNotebook:
        return LocalNotebook(self.root_dir, Writer(role, role.value), clock=self.clock)

    async def outcome(self, tag: str) -> MergeOutcome | None:
        committed = await self.handle(Role.OBSERVER).committed()
        found = next(item for item in committed if item.commit_id == self.commits[tag])
        return found.last_merge.outcome if found.last_merge is not None else None

    def merge_tests(self) -> dict[str, TestRequest]:
        return {rid: test for rid, test in self.toolbox.requests.items() if test.purpose is Purpose.INTEGRATION}

    def logged(self) -> list[dict[str, object]]:
        return [json.loads(line) for line in self.decisions.read_text().splitlines()]

    def refuse(self, tag: str, code: str = "delta_not_applicable") -> None:
        """The toolbox refuses the merge test of ``tag``'s change as it was committed."""

        refused = self.deltas[tag]
        self.toolbox.reject_if(lambda test: RequestRejected(REFUSAL, code) if test.delta == refused else None)


async def _one(bench: Bench) -> MergeReport | None:
    """A round one merge wide: the report of the one candidate tried, or ``None`` when none was."""

    reports = await bench.integrator.integrate_next(1)
    assert len(reports) <= 1
    return reports[0] if reports else None


def _root_recipe(head_patch: PatchRef) -> Recipe:
    component = ComponentPatches(COMPONENT, UPSTREAM_REPO, UPSTREAM_REF, (head_patch,))
    return Recipe(MB_FAKE_BASE_IMAGE_DIGEST, LaunchConfig(server_args="--max-num-seqs 64"), (component,))


def _change(tag: str, patch: PatchRef | None) -> RecipeDelta:
    config = ConfigDelta(server_args=f"{ARGS[tag]} {TAG_FLAG}{tag}")
    # Pinning the component in full lets the toolbox fake, whose parent recipes are empty, accept the change.
    pinned = ComponentChange(COMPONENT, UPSTREAM_REF, UPSTREAM_REPO, add_patches=(patch,)) if patch else None
    components = () if pinned is None else (pinned,)
    return RecipeDelta(config=config, components=components)


async def _commit_specialist_work(bench: Bench, invalid: bool) -> None:
    """``s1`` commits ``c1`` and ``c2``; with ``invalid``, ``c1``'s change fits no recipe of this component."""

    scheduler, toolbox, gate = (bench.handle(role) for role in (Role.SCHEDULER, Role.TOOLBOX, Role.GATE))
    await scheduler.record_lineage(LineageEvent(LineageKind.OPEN, "s1", from_commit=bench.root, reason="first batch"))
    await bench.repo.create_branch("s1", bench.root)
    parent = bench.root
    for tag in ("c1", "c2"):
        recipe = apply(await bench.repo.read_recipe(parent), bench.deltas[tag])
        if invalid and tag == "c1":
            moved = ComponentChange(COMPONENT, upstream_repo="https://example.com/fork.git")
            bench.deltas[tag] = replace(bench.deltas[tag], components=(moved,))
        commit = await bench.repo.commit("s1", parent, recipe, tag)
        test = request(f"r{tag}", intent("s1", parent, mechanism=tag), bench.deltas[tag])
        await toolbox.write_request(test)
        await toolbox.write_result(result(test.request_id))
        await gate.write_judgement(
            test.request_id, judgement(GAINS[tag], (test.request_id,)), Conclusion.COMMITTED, commit
        )
        bench.commits[tag], parent = commit, commit


async def _bench(
    tmp_path: Path,
    curves: Curves = CURVES,
    *,
    accuracy: dict[str, float] | None = None,
    invalid: bool = False,
    toolbox_type: type[RecordingToolbox] = RecordingToolbox,
    repo_type: type[GitExperimentRepo] = GitExperimentRepo,
) -> Bench:
    clock = FakeClock()
    notebook = tmp_path / "notebook"
    patches = PatchStore(LocalBlobStore(tmp_path / "blobs"))
    repo = await repo_type.create(tmp_path / "repo", _root_recipe(patches.store(COMPONENT, HEAD_TEXT)))
    root = await repo.head(INTEGRATION_BRANCH)
    await LocalNotebook(notebook, Writer(Role.SCHEDULER, "scheduler"), clock=clock).record_lineage(
        LineageEvent(LineageKind.ROOT, INTEGRATION_BRANCH, commit_id=root)
    )
    toolbox_notebook = LocalNotebook(notebook, Writer(Role.TOOLBOX, "toolbox"), clock=clock)
    toolbox = toolbox_type(toolbox_notebook, recipe_effect(tmp_path / "repo", curves), accuracy)
    merger = LocalNotebook(notebook, Writer(Role.INTEGRATOR, "integrator"), clock=clock)
    tests = SessionTests(toolbox, merger, Clock(now=clock, sleep=_instant))
    judge = MbFakeRuleJudge(LocalNotebook(notebook, Writer(Role.GATE, "gate"), clock=clock), tests)
    await judge.establish_baseline(root)
    completion = ScriptedCompletion()
    decisions = tmp_path / "MB-decisions.jsonl"
    resolver = LlmPatchResolver(completion, patches)
    integrator = Integrator(merger, repo, tests, judge, resolver, DecisionLog(decisions, Clock(now=clock)))
    deltas = {"c1": _change("c1", patches.store(COMPONENT, CANDIDATE_TEXT)), "c2": _change("c2", None)}
    made = Bench(notebook, clock, repo, root, {}, deltas, patches, toolbox, completion, integrator, decisions)
    await _commit_specialist_work(made, invalid)
    return made


@pytest.fixture
async def bench(tmp_path: Path) -> Bench:
    return await _bench(tmp_path)


async def test_an_improving_merge_is_tested_and_confirmed_from_the_head_then_written(bench: Bench) -> None:
    report = await _one(bench)

    head = await bench.repo.head(INTEGRATION_BRANCH)
    assert report is not None and report.outcome is MergeOutcome.MERGED
    assert (report.candidate, report.onto, report.merged_commit) == (bench.commits["c1"], bench.root, head)
    [test] = bench.merge_tests().values()
    assert (test.kind, test.branch_id, test.parent_commit, test.delta) == (
        TestKind.TEST,
        INTEGRATION_BRANCH,
        bench.root,
        bench.deltas["c1"],
    )
    assert (test.accuracy, test.intent_fingerprint, test.priority) == (True, None, None)
    [retest] = [test for test in bench.toolbox.requests.values() if test.purpose is Purpose.CONFIRM]
    assert retest.retest_of in bench.merge_tests() and retest.delta == bench.deltas["c1"]
    first = (await bench.handle(Role.OBSERVER).record(str(retest.retest_of))).result
    assert first is not None and first.sandbox is not None
    assert retest.avoid_nodes == (first.sandbox.node_name,)
    merged = await bench.repo.read_recipe(head)
    assert merged == apply(await bench.repo.read_recipe(bench.root), bench.deltas["c1"])
    record = await bench.handle(Role.OBSERVER).record(str(report.request_id))
    assert (record.conclusion, record.commit_id) == (Conclusion.MERGED, head)
    assert record.judgement is not None and record.judgement.gain_pct == pytest.approx(3.0)
    lineage = await bench.handle(Role.OBSERVER).lineage()
    assert lineage.commit(head).recipe_digest == merged.digest()
    [logged] = bench.logged()
    assert (logged["kind"], logged["branch"]) == (DecisionKind.MERGE.value, INTEGRATION_BRANCH)
    assert logged["detail"]["config_conflicts"] == [{"name": "--max-num-seqs", "before": "64", "after": "256"}]


@pytest.mark.parametrize(
    ("value", "inaccurate", "verdict"),
    [(1005.0, False, "flat"), (950.0, False, "regressed"), (1030.0, True, "blocked")],
    ids=["flat", "regressed", "inaccurate"],
)
async def test_a_merge_that_does_not_improve_is_not_stackable(
    tmp_path: Path, value: float, inaccurate: bool, verdict: str
) -> None:
    curves = {**CURVES, frozenset({"c1"}): {MAIN_POINT: value}}
    bench = await _bench(tmp_path, curves)
    if inaccurate:
        bench = await _bench(tmp_path / "again", curves, accuracy={bench.deltas["c1"].fingerprint(): 0.5})

    report = await _one(bench)

    assert report is not None and report.outcome is MergeOutcome.NOT_STACKABLE
    assert report.reason.startswith(verdict)
    assert await bench.repo.head(INTEGRATION_BRANCH) == bench.root
    record = await bench.handle(Role.OBSERVER).record(str(report.request_id))
    assert (record.conclusion, record.commit_id) == (Conclusion.NOT_STACKABLE, None)
    assert await bench.outcome("c1") is MergeOutcome.NOT_STACKABLE


async def test_each_commit_is_tried_once_per_integration_head(tmp_path: Path) -> None:
    curves = {frozenset({"c1"}): {MAIN_POINT: 1005.0}, frozenset({"c2"}): {MAIN_POINT: 1020.0}}
    bench = await _bench(tmp_path, {**curves, frozenset({"c1", "c2"}): {MAIN_POINT: 1025.0}})

    tried = [await _one(bench) for _ in range(4)]

    head = await bench.repo.head(INTEGRATION_BRANCH)
    c1, c2 = bench.commits["c1"], bench.commits["c2"]
    assert [(report.candidate, report.onto, report.outcome) for report in tried if report is not None] == [
        (c1, bench.root, MergeOutcome.NOT_STACKABLE),
        (c2, bench.root, MergeOutcome.MERGED),
        (c1, head, MergeOutcome.NOT_STACKABLE),
    ]
    assert tried[3] is None


async def test_a_round_tests_k_candidates_at_once_and_merges_the_first_improvement(bench: Bench) -> None:
    first = await bench.integrator.integrate_next(2)
    merged = await bench.repo.head(INTEGRATION_BRANCH)
    second = await bench.integrator.integrate_next(2)

    c1, c2 = bench.commits["c1"], bench.commits["c2"]
    assert [(r.candidate, r.onto, r.outcome, r.reason) for r in first] == [
        (c1, bench.root, MergeOutcome.MERGED, "improved +3.00% on the integration head"),
        (c2, bench.root, MergeOutcome.PENDING, SUPERSEDED),
    ]
    superseded = await bench.handle(Role.OBSERVER).record(str(first[1].request_id))
    assert superseded.conclusion is None
    assert [(r.candidate, r.onto, r.outcome) for r in second] == [(c2, merged, MergeOutcome.MERGED)]
    head = await bench.repo.read_recipe(await bench.repo.head(INTEGRATION_BRANCH))
    assert TAG_FLAG + "c1" in head.launch.server_args and TAG_FLAG + "c2" in head.launch.server_args
    assert await bench.integrator.integrate_next(2) == ()


async def test_commits_of_a_stopped_branch_stay_eligible(bench: Bench) -> None:
    scheduler = bench.handle(Role.SCHEDULER)
    await scheduler.record_lineage(LineageEvent(LineageKind.STOP, "s1", reason="judge: no progress"))
    await scheduler.release_branch("s1")

    report = await _one(bench)

    assert report is not None and (report.candidate, report.outcome) == (bench.commits["c1"], MergeOutcome.MERGED)


async def test_a_patch_the_toolbox_refuses_is_rewritten_and_the_merge_tested_again(bench: Bench) -> None:
    bench.refuse("c1")
    bench.completion.fallback = "```json\n" + json.dumps({"patch": RESOLVED_TEXT}) + "\n```"

    report = await _one(bench)

    assert report is not None and report.outcome is MergeOutcome.MERGED
    [(system, prompt)] = bench.completion.calls
    assert system == RESOLVER_SYSTEM_PROMPT
    assert all(text in prompt for text in (CANDIDATE_TEXT, HEAD_TEXT, REFUSAL))
    [test] = bench.merge_tests().values()
    [change] = test.delta.components
    [rewritten] = change.add_patches
    assert bench.patches.text(rewritten) == RESOLVED_TEXT
    head = await bench.repo.read_recipe(await bench.repo.head(INTEGRATION_BRANCH))
    assert rewritten in head.component(COMPONENT).patches


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
    bench.refuse("c1")

    report = await _one(bench)

    assert report is not None and report.outcome is MergeOutcome.CONFLICT_SKIPPED
    assert REFUSAL in report.reason and report.request_id is None
    assert await bench.outcome("c1") is MergeOutcome.CONFLICT_SKIPPED
    assert bench.merge_tests() == {}
    following = await _one(bench)
    assert following is not None and following.candidate == bench.commits["c2"]


async def test_a_refused_change_that_adds_no_patch_is_skipped_without_asking_the_model(bench: Bench) -> None:
    bench.refuse("c2")

    reports = await bench.integrator.integrate_next(2)

    assert [(r.candidate, r.outcome) for r in reports] == [
        (bench.commits["c1"], MergeOutcome.MERGED),
        (bench.commits["c2"], MergeOutcome.CONFLICT_SKIPPED),
    ]
    assert "unresolved patch conflict" in reports[1].reason and bench.completion.calls == []


async def test_a_merge_the_recipe_contract_refuses_is_skipped_and_the_round_goes_on(tmp_path: Path) -> None:
    bench = await _bench(tmp_path, invalid=True)

    reports = await bench.integrator.integrate_next(2)

    assert [(r.candidate, r.outcome) for r in reports] == [
        (bench.commits["c1"], MergeOutcome.CONFLICT_SKIPPED),
        (bench.commits["c2"], MergeOutcome.MERGED),
    ]
    assert "invalid" in reports[0].reason
    assert await bench.outcome("c1") is MergeOutcome.CONFLICT_SKIPPED


async def test_a_merge_test_the_toolbox_refuses_is_skipped_and_the_round_goes_on(bench: Bench) -> None:
    bench.refuse("c1", "base_image_mismatch")

    reports = await bench.integrator.integrate_next(2)

    assert [(r.candidate, r.outcome) for r in reports] == [
        (bench.commits["c1"], MergeOutcome.CONFLICT_SKIPPED),
        (bench.commits["c2"], MergeOutcome.MERGED),
    ]
    assert "base_image_mismatch" in reports[0].reason and reports[0].request_id is None
    assert bench.completion.calls == []
    assert await bench.outcome("c1") is MergeOutcome.CONFLICT_SKIPPED


async def test_a_withdrawn_merge_test_writes_nothing_to_the_integration_branch(bench: Bench) -> None:
    bench.toolbox.script(lambda test: test.purpose is Purpose.INTEGRATION, [Fate.MEASUREMENT])

    report = await _one(bench)

    assert report is not None and report.outcome is MergeOutcome.PENDING and report.request_id
    assert await bench.repo.head(INTEGRATION_BRANCH) == bench.root
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
    bench.toolbox.executor.gate.clear()
    merging = asyncio.ensure_future(bench.integrator.integrate_next(1))
    while not bench.merge_tests():
        await asyncio.sleep(0)

    await _cancel(merging)
    await bench.integrator.abandon()

    [request_id] = bench.merge_tests()
    assert bench.toolbox.withdrawn.keys() == {request_id}
    assert (await bench.handle(Role.OBSERVER).record(request_id)).withdrawn_at is not None
    assert await bench.repo.head(INTEGRATION_BRANCH) == bench.root


async def test_a_merge_cancelled_while_it_is_retested_is_abandoned_without_a_merge(tmp_path: Path) -> None:
    bench = await _bench(tmp_path, toolbox_type=HeldRetests)
    held = bench.toolbox
    assert isinstance(held, HeldRetests)
    merging = asyncio.ensure_future(bench.integrator.integrate_next(1))
    await held.confirming.wait()

    await _cancel(merging)
    await bench.integrator.abandon()

    assert await bench.repo.head(INTEGRATION_BRANCH) == bench.root
    [retest] = [rid for rid, test in held.requests.items() if test.purpose is Purpose.CONFIRM]
    assert held.withdrawn.keys() == {retest}


async def test_a_merge_cancelled_while_committing_is_still_recorded(tmp_path: Path) -> None:
    bench = await _bench(tmp_path, repo_type=HeldRepo)
    repo = bench.repo
    assert isinstance(repo, HeldRepo)
    repo.committing, repo.released = asyncio.Event(), asyncio.Event()
    merging = asyncio.ensure_future(bench.integrator.integrate_next(1))
    await repo.committing.wait()

    await _cancel(merging)
    abandoning = asyncio.ensure_future(bench.integrator.abandon())
    repo.released.set()
    await abandoning

    [request_id] = bench.merge_tests()
    record = await bench.handle(Role.OBSERVER).record(request_id)
    assert (record.conclusion, record.commit_id) == (Conclusion.MERGED, await repo.head(INTEGRATION_BRANCH))
    assert bench.toolbox.withdrawn == {}
