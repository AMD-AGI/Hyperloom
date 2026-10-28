# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The integrator against the local blackboard, the git experiment repository, the toolbox's fake and the rule gate.

The root runs ``--max-num-seqs 64``; branch ``s1`` commits ``c1`` (+4%, ``vllm`` pinned with a patch and
``--max-num-seqs 256``) and then ``c2`` (+2%, ``--max-num-seqs 128``), so ``c1`` is tried first. What a merge test
measures follows from the tags of the recipe it builds (:data:`CURVES`). The repository applies the merged patches to
a ``vllm`` package the bench lays out; a conflicting ``c1`` carries a second patch of the same line.

The toolbox's fake reads each parent's recipe from the bench's repository, as the real toolbox does, so a merge onto a
head that already pins ``vllm`` is accepted with a delta of patches alone; the root pins no component.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pytest

from hyperloom.common.perf_metric import (
    GRADED_DURATION,
    GRADED_ERROR_RATE,
    GRADED_INTVTY,
    GRADED_INTVTY_P50,
    GRADED_OUTPUT,
)
from hyperloom.contracts.measurement import Purpose, TestKind, TestRequest
from hyperloom.contracts.recipe import (
    ComponentChange,
    ConfigDelta,
    KernelArtifact,
    KernelKind,
    LaunchConfig,
    PatchRef,
    Recipe,
    RecipeDelta,
    apply,
)
from hyperloom.measure.compile import LocalBlobStore, PackageBase
from hyperloom_multibranch.contracts.judgement import AccuracyVerdict, Judgement, Verdict
from hyperloom_multibranch.contracts.blackboard import INTEGRATION_BRANCH, Role, Writer
from hyperloom_multibranch.contracts.blackboard.records import LineageEvent
from hyperloom_multibranch.contracts.blackboard.vocab import Conclusion, LineageKind, MergeOutcome
from hyperloom_multibranch.contracts.repo import RebasedPatch, ReplacedPatch, RepoMerge, ResolvedPatch, UnitedTable
from hyperloom_multibranch.contracts.toolbox import RequestRejected
from hyperloom_multibranch.blackboard import LocalBlackboard
from hyperloom_multibranch.blackboard.knowledge.warm_start import PreparedReplay
from hyperloom_multibranch.blackboard.rsi.stranded import StrandedLead
from hyperloom_multibranch.blackboard.tests.factories import FakeClock, intent, judgement, request, result
from hyperloom_multibranch.repo.git_repo import GitExperimentRepo
from hyperloom_multibranch.scheduling.candidates import MERGE_INFRA_RETRIES
from hyperloom_multibranch.scheduling.clock import Clock
from hyperloom_multibranch.scheduling.decisions import DecisionKind, DecisionLog
from hyperloom_multibranch.scheduling.explore import EXPLORE_TICK_SECONDS
from hyperloom_multibranch.scheduling.integrator import (
    ADOPTING_LEAD,
    ALREADY_ON_HEAD,
    CONFLICT_TRUNCATED,
    CONFLICT_UNRESOLVED,
    FOUNDATION_FIX_VERDICTS,
    NO_PATCH,
    PROMPT_VERSION,
    RESOLVER_ATTEMPTS,
    RESOLVER_SYSTEM_PROMPT,
    RESOLVER_UNREACHED,
    RESOLVER_UNREACHED_RETRIES,
    SUPERSEDED,
    Integrator,
    LlmPatchResolver,
    MergeRecords,
    MergeReport,
    not_stackable_reason,
)
from hyperloom_multibranch.contracts.llm import STOPPED_AT_MAX_TOKENS, CompletionFailed, CompletionReply
from hyperloom_multibranch.rulegate import agentx
from hyperloom_multibranch.rulegate.service import BASELINE_TESTS
from hyperloom_multibranch.scheduling.requests import SessionChecks, SessionTests, integration_test_request
from hyperloom_multibranch.scheduling.tests.peer_fakes import (
    BASE_IMAGE_DIGEST,
    MAIN_POINT,
    TAG_FLAG,
    Curves,
    FakeRegressionCheck,
    RecordingToolbox,
    ScriptedCompletion,
    measure_no_throughput,
    recipe_effect,
    session_gate,
)
from hyperloom_multibranch.toolbox.attempts import DEFAULT_MAX_INFRA_RETRIES
from hyperloom_multibranch.toolbox.fake_model import Fate

COMPONENT = "vllm"
UPSTREAM_REF = "a" * 40
UPSTREAM_REPO = "https://example.com/vllm.git"


def _patch_text(file: str, before: str, after: str) -> str:
    path = f"{COMPONENT}/{file}"
    return f"diff --git a/{path} b/{path}\n--- a/{path}\n+++ b/{path}\n@@ -1 +1 @@\n-{before}\n+{after}\n"


CANDIDATE_TEXT = _patch_text("x.py", "old", "candidate")
#: A second patch of ``c1`` that no longer applies once its first one is applied.
CONFLICTING_TEXT = _patch_text("x.py", "old", "second")
RESOLVED_TEXT = _patch_text("x.py", "candidate", "resolved")
REFUSAL = "the base image moved"
RESOLVER_MODEL = "resolver-model"
GAINS = {"c1": 4.0, "c2": 2.0}
ARGS = {"c1": "--max-num-seqs 256", "c2": "--max-num-seqs 128"}
#: ``c1`` merged onto the root gains 3%, ``c2`` 2%, and ``c2`` onto a head holding ``c1`` gains another 1.9%.
CURVES: Curves = {
    frozenset({"c1"}): {MAIN_POINT: 1030.0},
    frozenset({"c2"}): {MAIN_POINT: 1020.0},
    frozenset({"c1", "c2"}): {MAIN_POINT: 1050.0},
}


#: Real seconds between two looks at the toolbox while the merge it waits for is computed off the event loop.
POLL_SECONDS = 0.01
#: Real seconds a look waits for the round to reach the state it waits for before the test fails instead of hanging.
WAIT_SECONDS = 5.0


async def _instant(seconds: float) -> None:
    await asyncio.sleep(0)


async def _poll(seconds: float) -> None:
    """The round's wait before it looks for a newly committed commit again; the bench's merges run in real time."""

    await asyncio.sleep(POLL_SECONDS)


async def _until(ready: Callable[[], bool]) -> None:
    """Wait for ``ready`` while the round runs on the event loop beside this look."""

    async def looking() -> None:
        while not ready():
            await asyncio.sleep(POLL_SECONDS)

    await asyncio.wait_for(looking(), WAIT_SECONDS)


class HeldRetests(RecordingToolbox):
    """Holds every attempt once a confirming retest is submitted; a test double, not a placeholder."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.confirming = asyncio.Event()

    async def submit_test(self, request: TestRequest) -> str:
        if request.purpose is not Purpose.CONFIRM:
            return await super().submit_test(request)
        self.executor.gate.clear()
        accepted = await super().submit_test(request)
        self.confirming.set()
        return accepted


class HeldRepo(GitExperimentRepo):
    """Holds every merge until ``released`` is set; a test double, not a placeholder."""

    committing: asyncio.Event
    released: asyncio.Event

    async def commit_integration(self, onto: str, delta: RecipeDelta, merged_from: Sequence[str]) -> str:
        self.committing.set()
        await self.released.wait()
        return await super().commit_integration(onto, delta, merged_from)


def _package(root: Path) -> PackageBase:
    """The installed ``vllm`` the repository applies merged patches to: two files that read ``old``."""

    package = root / COMPONENT
    package.mkdir(parents=True)
    (package / "x.py").write_text("old\n")
    (package / "y.py").write_text("old\n")
    return PackageBase(COMPONENT, package, None, None)


@dataclass
class Bench:
    root_dir: Path
    clock: FakeClock
    repo: GitExperimentRepo
    root: str
    commits: dict[str, str]
    deltas: dict[str, RecipeDelta]
    toolbox: RecordingToolbox
    completion: ScriptedCompletion
    integrator: Integrator
    decisions: Path
    regression: FakeRegressionCheck

    def handle(self, role: Role) -> LocalBlackboard:
        return LocalBlackboard(self.root_dir, Writer(role, role.value), clock=self.clock)

    async def outcome(self, tag: str) -> MergeOutcome | None:
        committed = await self.handle(Role.OBSERVER).committed()
        found = next(item for item in committed if item.commit_id == self.commits[tag])
        return found.last_merge.outcome if found.last_merge is not None else None

    def merge_tests(self) -> dict[str, TestRequest]:
        return {rid: test for rid, test in self.toolbox.requests.items() if test.purpose is Purpose.INTEGRATION}

    def logged(self) -> list[dict[str, object]]:
        return [json.loads(line) for line in self.decisions.read_text().splitlines()]

    def refuse(self, tag: str) -> None:
        """The toolbox refuses every merge test that carries ``tag``."""

        def refused(test: TestRequest) -> RequestRejected | None:
            carries = test.purpose is Purpose.INTEGRATION and f"{TAG_FLAG}{tag}" in test.delta.config.server_args
            return RequestRejected(REFUSAL, "base_image_mismatch") if carries else None

        self.toolbox.reject_if(refused)


async def _one(bench: Bench) -> MergeReport | None:
    """A round one merge wide: the report of the one candidate tried, or ``None`` when none was."""

    reports = await bench.integrator.integrate_next(1, EXPLORE_TICK_SECONDS, ())
    assert len(reports) <= 1
    return reports[0] if reports else None


ROOT_RECIPE = Recipe(BASE_IMAGE_DIGEST, LaunchConfig(server_args="--max-num-seqs 64"))


def _change(tag: str, patches: tuple[PatchRef, ...] = ()) -> RecipeDelta:
    config = ConfigDelta(server_args=f"{ARGS[tag]} {TAG_FLAG}{tag}")
    pinned = ComponentChange(COMPONENT, UPSTREAM_REF, UPSTREAM_REPO, add_patches=patches) if patches else None
    components = () if pinned is None else (pinned,)
    return RecipeDelta(config=config, components=components)


async def _commit_specialist_work(bench: Bench, invalid: bool, foundation: str, tags: tuple[str, ...]) -> None:
    """``s1`` commits ``tags`` in turn; with ``invalid``, ``c1`` also moves the base image, which no merge carries,
    and the gate judged the commit of tag ``foundation`` a foundation fix."""

    scheduler, toolbox, gate = (bench.handle(role) for role in (Role.SCHEDULER, Role.TOOLBOX, Role.GATE))
    await scheduler.record_lineage(LineageEvent(LineageKind.OPEN, "s1", from_commit=bench.root, reason="first batch"))
    await bench.repo.create_branch("s1", bench.root)
    parent = bench.root
    for tag in tags:
        recipe = apply(await bench.repo.read_recipe(parent), bench.deltas[tag])
        if invalid and tag == "c1":
            recipe = replace(recipe, base_image_digest="sha256:" + "1" * 64)
        commit = await bench.repo.commit("s1", parent, recipe, tag)
        test = request(f"r{tag}", intent("s1", parent, mechanism=tag), bench.deltas[tag])
        await toolbox.write_request(test)
        await toolbox.write_result(result(test.request_id))
        await gate.write_judgement(
            test.request_id,
            judgement(GAINS[tag], (test.request_id,), foundation_fix=tag == foundation),
            Conclusion.COMMITTED,
            commit,
        )
        bench.commits[tag], parent = commit, commit


async def _bench(
    tmp_path: Path,
    curves: Curves = CURVES,
    *,
    accuracy: dict[str, float] | None = None,
    invalid: bool = False,
    foundation: str = "",
    conflicting: bool = False,
    tags: tuple[str, ...] = ("c1", "c2"),
    toolbox_type: type[RecordingToolbox] = RecordingToolbox,
    repo_type: type[GitExperimentRepo] = GitExperimentRepo,
    completion_type: type[ScriptedCompletion] = ScriptedCompletion,
) -> Bench:
    clock = FakeClock()
    blackboard = tmp_path / "blackboard"
    package = _package(tmp_path / "site")
    repo = repo_type(tmp_path / "repo", LocalBlobStore(tmp_path / "blobs"), lambda layout: package)
    root = await repo.init_root(ROOT_RECIPE)
    await LocalBlackboard(blackboard, Writer(Role.SCHEDULER, "scheduler"), clock=clock).record_lineage(
        LineageEvent(LineageKind.ROOT, INTEGRATION_BRANCH, commit_id=root)
    )
    toolbox_blackboard = LocalBlackboard(blackboard, Writer(Role.TOOLBOX, "toolbox"), clock=clock)
    toolbox = toolbox_type(toolbox_blackboard, recipe_effect(tmp_path / "repo", curves), accuracy, repo=repo)
    merger = LocalBlackboard(blackboard, Writer(Role.INTEGRATOR, "integrator"), clock=clock)
    tests = SessionTests(toolbox, Clock(now=clock, sleep=_instant))
    gate = session_gate(LocalBlackboard(blackboard, Writer(Role.GATE, "gate"), clock=clock), toolbox)
    baseline = integration_test_request(TestKind.TEST, Purpose.BASELINE, root)
    await gate.establish_baseline(root, [await tests.submit(baseline) for _ in range(BASELINE_TESTS)])
    completion = completion_type()
    decisions = tmp_path / "MB-decisions.jsonl"
    resolver = LlmPatchResolver(completion, repo, RESOLVER_MODEL)
    records = MergeRecords(merger, DecisionLog(decisions, Clock(now=clock)))
    regression = FakeRegressionCheck()
    integrator = Integrator(records, repo, SessionChecks(tests, gate, regression), resolver, Clock(clock, _poll))
    texts = (CANDIDATE_TEXT, CONFLICTING_TEXT) if conflicting else (CANDIDATE_TEXT,)
    patches = tuple([await repo.store_patch(COMPONENT, text) for text in texts])
    deltas = {"c1": _change("c1", patches), "c2": _change("c2")}
    made = Bench(blackboard, clock, repo, root, {}, deltas, toolbox, completion, integrator, decisions, regression)
    await _commit_specialist_work(made, invalid, foundation, tags)
    return made


@pytest.fixture
async def bench(tmp_path: Path) -> Bench:
    return await _bench(tmp_path)


#: The Recipe KB champion PRELUDE prepares for the bench's root: ``c1``'s launch config and no whole recipe, as a
#: lead the KB could not give the champion's files for gives. It gains the 3% ``c1`` gains on the root.
CHAMPION = PreparedReplay("inference:champion", None, LaunchConfig(server_args=f"{ARGS['c1']} {TAG_FLAG}c1"))


async def test_a_champion_that_improves_on_the_root_is_where_explore_opens(bench: Bench) -> None:
    opened = await bench.integrator.replay_champion(CHAMPION, bench.root)

    head = await bench.repo.head(INTEGRATION_BRANCH)
    assert opened == head != bench.root
    [request_id], [test] = bench.merge_tests(), bench.merge_tests().values()
    assert (test.kind, test.branch_id, test.parent_commit) == (TestKind.TEST, INTEGRATION_BRANCH, bench.root)
    replayed = await bench.repo.read_recipe(head)
    assert replayed == apply(await bench.repo.read_recipe(bench.root), test.delta)
    record = await bench.handle(Role.OBSERVER).record(request_id)
    assert (record.conclusion, record.commit_id) == (Conclusion.MERGED, head)
    assert record.judgement is not None and record.judgement.gain_pct == pytest.approx(3.0)
    lineage = await bench.handle(Role.OBSERVER).lineage()
    assert lineage.commit(head).recipe_digest == replayed.digest()
    [logged] = bench.logged()
    assert logged["detail"] == {
        "champion": CHAMPION.canonical_id,
        "onto": bench.root,
        "request_id": request_id,
        "merged_commit": head,
    }


async def test_a_champion_that_does_not_improve_leaves_explore_on_the_root(tmp_path: Path) -> None:
    bench = await _bench(tmp_path, {**CURVES, frozenset({"c1"}): {MAIN_POINT: 950.0}})

    opened = await bench.integrator.replay_champion(CHAMPION, bench.root)

    assert opened == bench.root == await bench.repo.head(INTEGRATION_BRANCH)
    [request_id] = bench.merge_tests()
    record = await bench.handle(Role.OBSERVER).record(request_id)
    assert (record.conclusion, record.commit_id) == (Conclusion.NOT_STACKABLE, None)
    [logged] = bench.logged()
    assert str(logged["reason"]).startswith("regressed")


async def test_a_champion_the_toolbox_refuses_leaves_explore_on_the_root(bench: Bench) -> None:
    """A refusal the PRELUDE preflight cannot foresee, such as backpressure, must not end the session before EXPLORE:
    the champion is no specialist's work and nothing of the session needs it."""

    bench.refuse("c1")

    opened = await bench.integrator.replay_champion(CHAMPION, bench.root)

    assert opened == bench.root == await bench.repo.head(INTEGRATION_BRANCH)
    [logged] = bench.logged()
    assert logged["reason"] == f"the toolbox refused the replay: {REFUSAL}"
    assert logged["detail"] == {
        "champion": CHAMPION.canonical_id,
        "onto": bench.root,
        "request_id": "",
        "merged_commit": "",
    }


async def test_a_champion_the_root_carries_already_is_never_tested(bench: Bench) -> None:
    opened = await bench.integrator.replay_champion(replace(CHAMPION, launch=ROOT_RECIPE.launch), bench.root)

    assert opened == bench.root and not bench.merge_tests()
    [logged] = bench.logged()
    assert logged["reason"] == ALREADY_ON_HEAD


async def test_an_improving_merge_is_tested_and_confirmed_from_the_head_then_written(bench: Bench) -> None:
    report = await _one(bench)

    head = await bench.repo.head(INTEGRATION_BRANCH)
    assert report is not None and report.outcome is MergeOutcome.MERGED
    assert (report.candidate, report.onto, report.merged_commit) == (bench.commits["c1"], bench.root, head)
    [test] = bench.merge_tests().values()
    assert (test.kind, test.branch_id, test.parent_commit) == (TestKind.TEST, INTEGRATION_BRANCH, bench.root)
    assert (test.accuracy, test.intent_fingerprint, test.priority) == (True, None, None)
    [retest] = [test for test in bench.toolbox.requests.values() if test.purpose is Purpose.CONFIRM]
    assert retest.retest_of in bench.merge_tests() and retest.delta == test.delta
    first = (await bench.handle(Role.OBSERVER).record(str(retest.retest_of))).result
    assert first is not None and first.sandbox is not None
    assert retest.avoid_nodes == (first.sandbox.node_name,)
    merged = await bench.repo.read_recipe(head)
    assert merged == apply(await bench.repo.read_recipe(bench.root), test.delta)
    assert merged == await bench.repo.read_recipe(bench.commits["c1"])
    record = await bench.handle(Role.OBSERVER).record(str(report.request_id))
    assert (record.conclusion, record.commit_id) == (Conclusion.MERGED, head)
    assert record.judgement is not None and record.judgement.gain_pct == pytest.approx(3.0)
    assert (report.gain_pct, report.threshold_pct) == (record.judgement.gain_pct, record.judgement.threshold_pct)
    lineage = await bench.handle(Role.OBSERVER).lineage()
    assert lineage.commit(head).recipe_digest == merged.digest()
    [logged] = bench.logged()
    assert (logged["kind"], logged["branch"]) == (DecisionKind.MERGE.value, INTEGRATION_BRANCH)
    assert logged["detail"]["config_conflicts"] == []


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
    assert (report.gain_pct, report.threshold_pct) == (None, None)
    assert await bench.repo.head(INTEGRATION_BRANCH) == bench.root
    record = await bench.handle(Role.OBSERVER).record(str(report.request_id))
    assert (record.conclusion, record.commit_id) == (Conclusion.NOT_STACKABLE, None)
    assert await bench.outcome("c1") is MergeOutcome.NOT_STACKABLE


async def test_an_improving_merge_the_regression_set_fails_stays_pending_and_is_tried_on_the_next_head(
    tmp_path: Path,
) -> None:
    """``c2``'s change is made on a branch of its own, so that the merge onto the next head does not carry ``c1``."""

    bench = await _bench(tmp_path, tags=("c1",))
    c2 = await _committed_elsewhere(bench, bench.deltas["c2"], GAINS["c2"])
    bench.regression.failures = {bench.root: "case kv-overflow reproduced"}

    report = await _one(bench)

    assert report is not None and report.outcome is MergeOutcome.PENDING
    assert report.reason == "the regression set failed: case kv-overflow reproduced"
    [test] = bench.merge_tests().values()
    assert bench.regression.calls == [(bench.root, test.delta, (bench.commits["c1"],))]
    assert await bench.repo.head(INTEGRATION_BRANCH) == bench.root
    record = await bench.handle(Role.OBSERVER).record(str(report.request_id))
    assert (record.conclusion, record.commit_id) == (None, None)
    assert await bench.outcome("c1") is MergeOutcome.PENDING
    assert [row["reason"] for row in bench.logged()] == [report.reason]

    bench.regression.failures = {}
    merged, retried = await _one(bench), await _one(bench)

    assert merged is not None and (merged.candidate, merged.outcome) == (c2, MergeOutcome.MERGED)
    assert retried is not None and (retried.candidate, retried.onto) == (bench.commits["c1"], merged.merged_commit)
    assert retried.outcome is MergeOutcome.MERGED


@pytest.mark.parametrize(("value", "verdict"), [(1005.0, "flat"), (950.0, "regressed")], ids=["flat", "regressed"])
async def test_a_foundation_fix_that_passes_the_accuracy_gate_and_the_regression_set_merges_without_a_gain(
    tmp_path: Path, value: float, verdict: str
) -> None:
    bench = await _bench(tmp_path, {**CURVES, frozenset({"c1"}): {MAIN_POINT: value}}, foundation="c1")

    report = await _one(bench)

    head = await bench.repo.head(INTEGRATION_BRANCH)
    assert report is not None and report.outcome is MergeOutcome.MERGED and report.foundation_fix
    assert (report.candidate, report.merged_commit) == (bench.commits["c1"], head)
    assert report.reason.startswith(f"a foundation fix, {verdict} ")
    [test] = bench.merge_tests().values()
    assert bench.regression.calls == [(bench.root, test.delta, (bench.commits["c1"],))]
    record = await bench.handle(Role.OBSERVER).record(str(report.request_id))
    assert (record.conclusion, record.commit_id) == (Conclusion.MERGED, head)
    assert record.judgement is not None and report.gain_pct == record.judgement.gain_pct
    [logged] = bench.logged()
    assert logged["detail"]["foundation_fix"] is True
    assert (logged["detail"]["gain_pct"], logged["detail"]["threshold_pct"]) == (report.gain_pct, report.threshold_pct)


async def test_a_foundation_fix_the_regression_set_fails_stays_pending(tmp_path: Path) -> None:
    bench = await _bench(tmp_path, {**CURVES, frozenset({"c1"}): {MAIN_POINT: 950.0}}, foundation="c1")
    bench.regression.failures = {bench.root: "case kv-overflow reproduced"}

    report = await _one(bench)

    assert report is not None and report.outcome is MergeOutcome.PENDING and report.foundation_fix
    assert report.reason == "the regression set failed: case kv-overflow reproduced"
    assert await bench.repo.head(INTEGRATION_BRANCH) == bench.root


async def test_a_foundation_fix_the_accuracy_gate_fails_is_not_stackable_and_runs_no_regression_set(
    tmp_path: Path,
) -> None:
    curves = {**CURVES, frozenset({"c1"}): {MAIN_POINT: 950.0}}
    bench = await _bench(tmp_path, curves)
    bench = await _bench(tmp_path / "again", curves, accuracy={bench.deltas["c1"].fingerprint(): 0.5}, foundation="c1")

    report = await _one(bench)

    assert report is not None and report.outcome is MergeOutcome.NOT_STACKABLE
    assert report.reason.startswith("regressed")
    assert bench.regression.calls == []
    assert await bench.repo.head(INTEGRATION_BRANCH) == bench.root


#: A foundation fix's change: a tag flag of its own and the environment variable it sets.
FIX = RecipeDelta(config=ConfigDelta(server_args=f"{TAG_FLAG}fix", envs={"HL_FIX": "1"}))


async def test_a_foundation_fix_whose_parent_is_merged_carries_only_itself_and_merges_without_a_gain(
    tmp_path: Path,
) -> None:
    bench = await _bench(tmp_path, {**CURVES, frozenset({"c1", "fix"}): {MAIN_POINT: 1030.0}}, tags=("c1",))
    parent = await _one(bench)
    assert parent is not None and parent.outcome is MergeOutcome.MERGED
    fix = await _committed_on(bench, "s1", bench.commits["c1"], FIX, 0.0, foundation=True, name="fix")

    report = await _one(bench)

    assert report is not None and (report.candidate, report.onto) == (fix, parent.merged_commit)
    assert report.outcome is MergeOutcome.MERGED and report.foundation_fix
    assert report.reason.startswith("a foundation fix, flat ")


async def test_a_foundation_fix_that_carries_a_parent_never_merged_is_judged_like_any_candidate(
    tmp_path: Path,
) -> None:
    """``c2``, a foundation fix cut from ``c1``, which regresses, would bring ``c1`` onto the integration head with it."""

    regressed: Curves = {frozenset({"c1"}): {MAIN_POINT: 950.0}, frozenset({"c1", "c2"}): {MAIN_POINT: 950.0}}
    bench = await _bench(tmp_path, regressed, foundation="c2")

    report = await _one(bench)

    assert report is not None and (report.candidate, report.onto) == (bench.commits["c2"], bench.root)
    assert report.outcome is MergeOutcome.NOT_STACKABLE and not report.foundation_fix
    assert report.reason.startswith("regressed ")
    assert bench.regression.calls == []
    assert await bench.repo.head(INTEGRATION_BRANCH) == bench.root
    assert await bench.outcome("c1") is MergeOutcome.NOT_STACKABLE


@pytest.mark.parametrize("verdict", [Verdict.NEEDS_CONFIRM, Verdict.BLOCKED, Verdict.INCONCLUSIVE])
def test_a_foundation_fix_on_an_unsettled_or_refused_verdict_is_not_stackable(verdict: Verdict) -> None:
    judged = Judgement(verdict, 5.0, 1.0, ("r1",), {}, accuracy=AccuracyVerdict.PASSED)

    assert not_stackable_reason(judged, foundation_fix=True) == f"{verdict.value} +5.00% on the integration head"


@pytest.mark.parametrize("verdict", [Verdict.FLAT, Verdict.REGRESSED])
def test_a_foundation_fix_without_a_gain_may_be_merged_only_once_it_passed_the_accuracy_gate(verdict: Verdict) -> None:
    judged = Judgement(verdict, -2.0, 1.0, ("r1",), {}, accuracy=AccuracyVerdict.PASSED)

    assert verdict in FOUNDATION_FIX_VERDICTS
    assert not_stackable_reason(judged, foundation_fix=True) == ""
    assert not_stackable_reason(judged, foundation_fix=False) == f"{verdict.value} -2.00% on the integration head"
    assert not_stackable_reason(replace(judged, accuracy=AccuracyVerdict.FAILED), foundation_fix=True)
    assert not_stackable_reason(replace(judged, accuracy=AccuracyVerdict.NOT_MEASURED), foundation_fix=True)


def test_an_improvement_judged_without_its_eval_is_not_stackable() -> None:
    judged = Judgement(Verdict.IMPROVED, 5.0, 1.0, ("r1",), {}, accuracy=AccuracyVerdict.NOT_MEASURED)

    reason = not_stackable_reason(judged, foundation_fix=False)

    assert reason == "improved on the integration head with its accuracy not_measured"
    assert not_stackable_reason(replace(judged, accuracy=AccuracyVerdict.PASSED), foundation_fix=False) == ""


AGENTX_PARENT = {
    GRADED_INTVTY_P50: 100.0,
    GRADED_INTVTY: 100.0,
    GRADED_OUTPUT: 1000.0,
    GRADED_ERROR_RATE: 0.0,
    GRADED_DURATION: 10.0,
}


def test_an_agentx_foundation_fix_whose_p50_falls_with_its_guards_intact_may_be_merged() -> None:
    judged = agentx.compare({**AGENTX_PARENT, GRADED_INTVTY_P50: 50.0}, AGENTX_PARENT, "r1")

    assert judged.verdict is Verdict.REGRESSED and judged.accuracy is AccuracyVerdict.PASSED
    assert not_stackable_reason(judged, foundation_fix=True) == ""


def test_an_agentx_foundation_fix_that_breaks_a_guard_is_not_stackable() -> None:
    judged = agentx.compare({**AGENTX_PARENT, GRADED_OUTPUT: 500.0}, AGENTX_PARENT, "r1")

    assert judged.verdict is Verdict.REGRESSED and judged.accuracy is AccuracyVerdict.PASSED
    reason = not_stackable_reason(judged, foundation_fix=True)
    assert reason == "a foundation fix, regressed on the integration head, refused by output_regressed"


async def test_a_merge_measuring_no_throughput_stays_pending_and_merging_goes_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bench = await _bench(tmp_path)
    measure_no_throughput(monkeypatch)

    report = await _one(bench)

    assert report is not None and report.outcome is MergeOutcome.PENDING
    assert report.reason.startswith("the merge test failed: ") and "positive" in report.reason
    assert await bench.repo.head(INTEGRATION_BRANCH) == bench.root


async def test_a_merge_that_could_not_be_measured_is_tried_again_on_the_next_head(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``c2``'s change is made on a branch of its own, so that the merge onto the next head does not carry ``c1``."""

    bench = await _bench(tmp_path, tags=("c1",))
    c2 = await _committed_elsewhere(bench, bench.deltas["c2"], GAINS["c2"])
    measure_no_throughput(monkeypatch)
    unmeasured = await _one(bench)
    monkeypatch.undo()

    merged = await _one(bench)
    retried = await _one(bench)

    c1 = bench.commits["c1"]
    assert unmeasured is not None and (unmeasured.candidate, unmeasured.onto) == (c1, bench.root)
    assert unmeasured.outcome is MergeOutcome.PENDING and unmeasured.reason.startswith("the merge test failed")
    assert (await bench.handle(Role.OBSERVER).record(str(unmeasured.request_id))).conclusion is None
    assert merged is not None and (merged.candidate, merged.outcome) == (c2, MergeOutcome.MERGED)
    assert retried is not None and (retried.candidate, retried.onto) == (c1, merged.merged_commit)
    assert retried.outcome is MergeOutcome.MERGED


async def test_a_not_stackable_commit_is_tried_again_once_a_merge_changed_what_its_change_names(
    tmp_path: Path,
) -> None:
    """``c2``'s change is made on a branch of its own, so that its merge does not carry ``c1``."""

    curves = {frozenset({"c1"}): {MAIN_POINT: 1005.0}, frozenset({"c2"}): {MAIN_POINT: 1020.0}}
    bench = await _bench(tmp_path, {**curves, frozenset({"c1", "c2"}): {MAIN_POINT: 1025.0}}, tags=("c1",))
    c2 = await _committed_elsewhere(bench, bench.deltas["c2"], GAINS["c2"])

    tried = [await _one(bench) for _ in range(4)]

    head = await bench.repo.head(INTEGRATION_BRANCH)
    c1 = bench.commits["c1"]
    assert [(report.candidate, report.onto, report.outcome) for report in tried if report is not None] == [
        (c1, bench.root, MergeOutcome.NOT_STACKABLE),
        (c2, bench.root, MergeOutcome.MERGED),
        (c1, head, MergeOutcome.NOT_STACKABLE),
    ]
    assert tried[3] is None
    assert [row["detail"]["config_conflicts"] for row in bench.logged()] == [
        [],
        [],
        [{"name": "--max-num-seqs", "before": "128", "after": "256"}],
    ]


async def test_a_merge_carries_the_commits_its_candidate_builds_on_that_were_never_merged(tmp_path: Path) -> None:
    """``c1`` alone is not stackable; ``c2`` built on it brings its change, and the merge lands both, so ``c1`` is
    merged rather than left behind its branch's lead."""

    bench = await _bench(tmp_path, {**CURVES, frozenset({"c1"}): {MAIN_POINT: 1005.0}})

    refused, carrying = await _one(bench), await _one(bench)

    c1, c2 = bench.commits["c1"], bench.commits["c2"]
    assert refused is not None and (refused.candidate, refused.outcome) == (c1, MergeOutcome.NOT_STACKABLE)
    assert carrying is not None and (carrying.candidate, carrying.onto) == (c2, bench.root)
    assert carrying.outcome is MergeOutcome.MERGED and carrying.gain_pct == pytest.approx(5.0)
    assert await bench.repo.read_recipe(str(carrying.merged_commit)) == await bench.repo.read_recipe(c2)
    observer = bench.handle(Role.OBSERVER)
    assert {item.commit_id: item.merged_into for item in await observer.committed()} == {
        c1: carrying.merged_commit,
        c2: carrying.merged_commit,
    }
    assert (await observer.lineage()).commit(str(carrying.merged_commit)).merged_from == (c1, c2)
    assert await _one(bench) is None


#: A change that names nothing ``c1`` or ``c2`` name: an environment variable and a tag flag of its own.
APART = RecipeDelta(config=ConfigDelta(server_args=f"{TAG_FLAG}apart", envs={"HL_APART": "1"}))


async def test_a_not_stackable_commit_whose_change_the_head_did_not_touch_is_not_tried_again(bench: Bench) -> None:
    apart = await _committed_elsewhere(bench, APART, 9.0)

    rounds = [await _one(bench) for _ in range(4)]

    c1, c2 = bench.commits["c1"], bench.commits["c2"]
    assert [(report.candidate, report.outcome) for report in rounds if report is not None] == [
        (apart, MergeOutcome.NOT_STACKABLE),
        (c1, MergeOutcome.MERGED),
        (c2, MergeOutcome.MERGED),
    ]
    assert rounds[3] is None
    tried = [test for test in bench.merge_tests().values() if f"{TAG_FLAG}apart" in test.delta.config.server_args]
    assert len(tried) == 1


async def test_a_merge_test_the_candidate_failed_is_not_tried_again_on_a_head_its_change_does_not_name(
    bench: Bench,
) -> None:
    apart = await _committed_elsewhere(bench, APART, 9.0)
    bench.toolbox.script(lambda test: f"{TAG_FLAG}apart" in test.delta.config.server_args, [Fate.CANDIDATE])

    rounds = [await _one(bench) for _ in range(4)]

    c1, c2 = bench.commits["c1"], bench.commits["c2"]
    assert [(report.candidate, report.outcome) for report in rounds if report is not None] == [
        (apart, MergeOutcome.PENDING),
        (c1, MergeOutcome.MERGED),
        (c2, MergeOutcome.MERGED),
    ]
    assert rounds[0] is not None and "candidate_failed" in rounds[0].reason and "serve_flag" in rounds[0].reason
    assert rounds[3] is None
    tried = [test for test in bench.merge_tests().values() if f"{TAG_FLAG}apart" in test.delta.config.server_args]
    assert len(tried) == 1


#: ``c1`` then ``c2``, then ``e1`` and ``e2``, each committed on a branch of its own, move the head four times; only
#: ``e2`` sets a variable ``apart`` sets.
LATER_CURVES: Curves = {
    **CURVES,
    frozenset({"c1", "c2", "e1"}): {MAIN_POINT: 1070.0},
    frozenset({"c1", "c2", "e1", "e2"}): {MAIN_POINT: 1090.0},
}
LATER = {
    "e1": RecipeDelta(config=ConfigDelta(server_args=f"{TAG_FLAG}e1", envs={"HL_E1": "1"})),
    "e2": RecipeDelta(config=ConfigDelta(server_args=f"{TAG_FLAG}e2", envs={"HL_APART": "2"})),
}


async def test_a_merge_test_ended_by_the_infrastructure_is_retried_on_the_next_heads_up_to_the_cap(
    tmp_path: Path,
) -> None:
    """Past the cap such a test holds as an answer: ``apart`` comes back only on the head ``e2``, which sets what it
    sets, moved to."""

    bench = await _bench(tmp_path, LATER_CURVES)
    apart = await _committed_elsewhere(bench, APART, 9.0)
    for gain, (name, delta) in zip((1.5, 1.0), LATER.items()):
        await bench.handle(Role.SCHEDULER).record_lineage(
            LineageEvent(LineageKind.OPEN, name, from_commit=bench.root, reason="refill")
        )
        await bench.repo.create_branch(name, bench.root)
        await _committed_on(bench, name, bench.root, delta, gain, name=name)
    infra_ending = [Fate.INFRA] * (DEFAULT_MAX_INFRA_RETRIES + 1)
    bench.toolbox.script(lambda test: f"{TAG_FLAG}apart" in test.delta.config.server_args, infra_ending)

    rounds = [await _one(bench) for _ in range(9)]

    tried = [report for report in rounds if report is not None and report.candidate == apart]
    heads = [report.merged_commit for report in rounds if report is not None and report.merged_commit is not None]
    assert len(heads) == 4 and rounds[8] is None
    assert [report.onto for report in tried] == [bench.root, *heads[:2], heads[3]]
    assert all(report.outcome is MergeOutcome.PENDING for report in tried)
    assert all(report.reason.startswith("the merge test failed") for report in tried)
    assert len(tried) == MERGE_INFRA_RETRIES + 2


#: ``parent`` and ``child``, one after the other on ``s1``, and ``across``, on a branch of its own, which sets the
#: variable ``parent`` sets and nothing ``child`` sets.
PARENT = RecipeDelta(config=ConfigDelta(server_args=f"{TAG_FLAG}parent", envs={"HL_PARENT": "1"}))
CHILD = RecipeDelta(config=ConfigDelta(server_args=f"{TAG_FLAG}child", envs={"HL_CHILD": "1"}))
ACROSS = RecipeDelta(config=ConfigDelta(server_args=f"{TAG_FLAG}across", envs={"HL_PARENT": "2"}))


async def test_a_not_stackable_commit_is_tried_again_once_a_merge_changed_what_a_parent_it_carries_names(
    tmp_path: Path,
) -> None:
    curves: Curves = {
        frozenset({"parent", "child"}): {MAIN_POINT: 1005.0},
        frozenset({"across"}): {MAIN_POINT: 1030.0},
        frozenset({"across", "parent", "child"}): {MAIN_POINT: 1060.0},
    }
    bench = await _bench(tmp_path, curves, tags=())
    parent = await _committed_on(bench, "s1", bench.root, PARENT, 1.0, name="parent")
    child = await _committed_on(bench, "s1", parent, CHILD, 5.0, name="child")
    across = await _committed_elsewhere(bench, ACROSS, 0.5)

    rounds = [await _one(bench) for _ in range(3)]

    merged = {item.commit_id: item.merged_into for item in await bench.handle(Role.OBSERVER).committed()}
    assert [(report.candidate, report.onto, report.outcome) for report in rounds if report is not None] == [
        (child, bench.root, MergeOutcome.NOT_STACKABLE),
        (across, bench.root, MergeOutcome.MERGED),
        (child, merged[across], MergeOutcome.MERGED),
    ]


async def test_a_not_stackable_commit_is_not_tried_again_on_a_head_that_patched_the_component_elsewhere(
    bench: Bench,
) -> None:
    aside = await bench.repo.store_patch(COMPONENT, _patch_text("y.py", "old", "aside"))
    pinned = ComponentChange(COMPONENT, UPSTREAM_REF, UPSTREAM_REPO, add_patches=(aside,))
    delta = RecipeDelta(config=ConfigDelta(server_args=f"{TAG_FLAG}aside"), components=(pinned,))
    elsewhere = await _committed_elsewhere(bench, delta, 9.0)

    rounds = [await _one(bench) for _ in range(4)]

    c1, c2 = bench.commits["c1"], bench.commits["c2"]
    assert [(report.candidate, report.outcome) for report in rounds if report is not None] == [
        (elsewhere, MergeOutcome.NOT_STACKABLE),
        (c1, MergeOutcome.MERGED),
        (c2, MergeOutcome.MERGED),
    ]
    assert rounds[3] is None
    tried = [test for test in bench.merge_tests().values() if f"{TAG_FLAG}aside" in test.delta.config.server_args]
    assert len(tried) == 1


async def test_a_round_tests_k_candidates_at_once_and_merges_the_first_improvement(bench: Bench) -> None:
    first = await bench.integrator.integrate_next(2, EXPLORE_TICK_SECONDS, ())
    merged = await bench.repo.head(INTEGRATION_BRANCH)
    second = await bench.integrator.integrate_next(2, EXPLORE_TICK_SECONDS, ())

    c1, c2 = bench.commits["c1"], bench.commits["c2"]
    assert [(r.candidate, r.onto, r.outcome, r.reason) for r in first] == [
        (c1, bench.root, MergeOutcome.MERGED, "improved +3.00% on the integration head"),
        (c2, bench.root, MergeOutcome.PENDING, SUPERSEDED),
    ]
    superseded = await bench.handle(Role.OBSERVER).record(str(first[1].request_id))
    assert superseded.conclusion is None
    assert [(base, sources) for base, _, sources in bench.regression.calls] == [(bench.root, (c1,)), (merged, (c2,))]
    assert [(r.candidate, r.onto, r.outcome) for r in second] == [(c2, merged, MergeOutcome.MERGED)]
    head = await bench.repo.read_recipe(await bench.repo.head(INTEGRATION_BRANCH))
    assert TAG_FLAG + "c1" in head.launch.server_args and TAG_FLAG + "c2" in head.launch.server_args
    assert await bench.integrator.integrate_next(2, EXPLORE_TICK_SECONDS, ()) == ()


async def test_a_commit_made_during_a_round_takes_the_slot_the_round_had_nothing_to_fill(tmp_path: Path) -> None:
    bench = await _bench(tmp_path, tags=("c1",))
    bench.toolbox.executor.gate.clear()
    merging = asyncio.ensure_future(bench.integrator.integrate_next(2, EXPLORE_TICK_SECONDS, ()))
    await _until(lambda: len(bench.merge_tests()) == 1)

    arrived = await _committed_elsewhere(bench, _change("c2"), 2.0)
    await _until(lambda: len(bench.merge_tests()) == 2)
    bench.toolbox.executor.gate.set()
    reports = await merging

    assert len(reports) == 2
    assert {(report.candidate, report.onto) for report in reports} == {
        (bench.commits["c1"], bench.root),
        (arrived, bench.root),
    }
    assert {report.outcome for report in reports} == {MergeOutcome.MERGED, MergeOutcome.PENDING}


async def _committed_elsewhere(bench: Bench, delta: RecipeDelta, gain: float, *, foundation: bool = False) -> str:
    """A commit of a new branch ``s2`` on the root, that the gate judged ``gain``, a foundation fix with
    ``foundation``, and committed."""

    scheduler = bench.handle(Role.SCHEDULER)
    await scheduler.record_lineage(LineageEvent(LineageKind.OPEN, "s2", from_commit=bench.root, reason="refill"))
    await bench.repo.create_branch("s2", bench.root)
    return await _committed_on(bench, "s2", bench.root, delta, gain, foundation=foundation)


async def _committed_on(
    bench: Bench,
    branch: str,
    parent: str,
    delta: RecipeDelta,
    gain: float,
    *,
    foundation: bool = False,
    name: str = "d1",
) -> str:
    """Commit ``name``, ``delta`` on ``parent`` of the open ``branch``, that the gate judged ``gain``, a foundation fix
    with ``foundation``, and committed."""

    toolbox, gate = bench.handle(Role.TOOLBOX), bench.handle(Role.GATE)
    commit = await bench.repo.commit(branch, parent, apply(await bench.repo.read_recipe(parent), delta), name)
    test = request(f"r{name}", intent(branch, parent, mechanism=name), delta)
    await toolbox.write_request(test)
    await toolbox.write_result(result(test.request_id))
    judged = judgement(gain, (test.request_id,), foundation_fix=foundation)
    await gate.write_judgement(test.request_id, judged, Conclusion.COMMITTED, commit)
    return commit


async def _committed_again(bench: Bench) -> str:
    """A commit of a new branch ``s2`` that makes ``c1``'s change again, committed on the root."""

    return await _committed_elsewhere(bench, RecipeDelta(config=bench.deltas["c1"].config), 5.0)


#: The shapes the head's rows won in the united slot, as ``union_rows`` names them by their key columns.
KEPT_SHAPES = ({"M": "1024", "N": "4096"},)
UNITED = UnitedTable(
    KernelArtifact(KernelKind.GEMM, "linear", "kernel/theirs.csv", "c" * 64),
    KernelArtifact(KernelKind.GEMM, "linear", "kernel/ours.csv", "d" * 64),
    KernelArtifact(KernelKind.GEMM, "linear", "kernel/united.csv", "e" * 64),
    KEPT_SHAPES,
)
UNITED_ROW = {
    "kind": KernelKind.GEMM.value,
    "target": "linear",
    "theirs": "c" * 64,
    "ours": "d" * 64,
    "table": "e" * 64,
    "kept_head_shapes": [dict(shape) for shape in KEPT_SHAPES],
}


class UnitingRepo(GitExperimentRepo):
    """Reports every merge as uniting one kernel slot; a test double for the union a merge makes where a table's
    shapes can be keyed, which needs an aiter install this bench has none of."""

    async def merge(self, candidate: str, onto: str, protected: frozenset[str] = frozenset()) -> RepoMerge:
        return replace(await super().merge(candidate, onto, protected), united_tables=(UNITED,))


async def test_a_merge_that_unites_two_kernel_tables_says_so_in_the_decision_log(tmp_path: Path) -> None:
    bench = await _bench(tmp_path, repo_type=UnitingRepo)

    merged = await _one(bench)
    again = await _committed_again(bench)
    reports = await bench.integrator.integrate_next(2, EXPLORE_TICK_SECONDS, ())

    assert merged is not None and merged.outcome is MergeOutcome.MERGED
    [skipped] = [report for report in reports if report.candidate == again]
    assert skipped.outcome is MergeOutcome.ALREADY_ON_HEAD
    rows = {entry["detail"]["candidate"]: entry["detail"]["united_tables"] for entry in bench.logged()}
    assert rows[merged.candidate] == [UNITED_ROW]
    assert rows[again] == [UNITED_ROW]


async def test_a_merge_that_leaves_the_head_s_recipe_unchanged_is_skipped_untested(bench: Bench) -> None:
    await _one(bench)
    head = await bench.repo.head(INTEGRATION_BRANCH)
    again = await _committed_again(bench)

    reports = await bench.integrator.integrate_next(2, EXPLORE_TICK_SECONDS, ())

    [skipped] = [report for report in reports if report.candidate == again]
    assert (skipped.onto, skipped.outcome, skipped.reason) == (head, MergeOutcome.ALREADY_ON_HEAD, ALREADY_ON_HEAD)
    assert skipped.request_id is None
    assert [test.parent_commit for test in bench.merge_tests().values()] == [bench.root, head]
    assert [report.candidate for report in reports] == [again, bench.commits["c2"]]
    [recorded] = [item for item in await bench.handle(Role.OBSERVER).committed() if item.commit_id == again]
    assert recorded.last_merge is not None and recorded.last_merge.outcome is MergeOutcome.ALREADY_ON_HEAD
    [logged] = [entry for entry in bench.logged() if entry["detail"]["candidate"] == again]
    assert (logged["reason"], logged["detail"]["outcome"]) == (ALREADY_ON_HEAD, MergeOutcome.ALREADY_ON_HEAD.value)


async def test_a_commit_already_on_the_head_is_not_tried_again_when_the_head_moves(bench: Bench) -> None:
    await _one(bench)
    again = await _committed_again(bench)
    [skipped] = await bench.integrator.integrate_next(1, EXPLORE_TICK_SECONDS, ())
    head = await bench.repo.head(INTEGRATION_BRANCH)

    moved = await bench.integrator.integrate_next(2, EXPLORE_TICK_SECONDS, ())

    assert (skipped.candidate, skipped.outcome) == (again, MergeOutcome.ALREADY_ON_HEAD)
    assert await bench.repo.head(INTEGRATION_BRANCH) != head
    assert again not in {report.candidate for report in moved}
    assert await bench.integrator.integrate_next(2, EXPLORE_TICK_SECONDS, ()) == ()


async def test_commits_of_a_stopped_branch_stay_eligible(bench: Bench) -> None:
    scheduler = bench.handle(Role.SCHEDULER)
    await scheduler.record_lineage(LineageEvent(LineageKind.STOP, "s1", reason="judge: no progress"))
    await scheduler.release_branch("s1")

    report = await _one(bench)

    assert report is not None and (report.candidate, report.outcome) == (bench.commits["c1"], MergeOutcome.MERGED)


async def test_a_patch_that_does_not_apply_on_the_head_is_rewritten_before_the_merge_is_tested(
    tmp_path: Path,
) -> None:
    bench = await _bench(tmp_path, conflicting=True)
    bench.completion.fallback = "```json\n" + json.dumps({"patch": RESOLVED_TEXT}) + "\n```"

    report = await _one(bench)

    assert report is not None and report.outcome is MergeOutcome.MERGED
    [(system, prompt)] = bench.completion.calls
    assert system == RESOLVER_SYSTEM_PROMPT
    assert prompt.index(CONFLICTING_TEXT) < prompt.index(CANDIDATE_TEXT)
    [test] = bench.merge_tests().values()
    [change] = test.delta.components
    first, rewritten = change.add_patches
    assert first == bench.deltas["c1"].components[0].add_patches[0]
    head = await bench.repo.head(INTEGRATION_BRANCH)
    assert await bench.repo.patch_text(head, rewritten.path) == RESOLVED_TEXT
    assert rewritten in (await bench.repo.read_recipe(head)).component(COMPONENT).patches
    conflicting = bench.deltas["c1"].components[0].add_patches[1]
    written = ResolvedPatch(COMPONENT, conflicting.path, rewritten, RESOLVER_MODEL, PROMPT_VERSION, 1)
    assert (await bench.handle(Role.OBSERVER).snapshot()).resolved_patches == (written,)
    assert report.resolved_patches == (written,)
    [logged] = bench.logged()
    assert logged["detail"]["resolved_patches"] == [
        {"component": COMPONENT, "replaces": conflicting.path, "patch": rewritten.path, "attempts": 1}
    ]


#: What the transport says of a call the gateway ended before the model answered.
GATEWAY_TIMEOUT = "patch_resolve call failed: RuntimeError: HTTP 504 Gateway Timeout"


@pytest.mark.parametrize(
    ("scripted", "named", "calls"),
    [
        (ScriptedCompletion(fallback=json.dumps({"patch": None})), CONFLICT_UNRESOLVED, 1),
        (
            ScriptedCompletion(error=CompletionFailed(GATEWAY_TIMEOUT)),
            RESOLVER_UNREACHED,
            1 + RESOLVER_UNREACHED_RETRIES,
        ),
        (
            ScriptedCompletion(fallback='{"patch": "diff --git a/vllm/x.py', stop_reason=STOPPED_AT_MAX_TOKENS),
            CONFLICT_TRUNCATED,
            1,
        ),
    ],
    ids=["model-declines", "call-fails", "answer-cut-off"],
)
async def test_an_unresolved_patch_conflict_is_skipped_and_recorded(
    tmp_path: Path, scripted: ScriptedCompletion, named: str, calls: int
) -> None:
    """A conflict the model declined, one it was never asked about and one whose answer the output cap cut off are
    all skipped, under reasons that tell them apart: a failed call or a cut-off answer says nothing about whether
    the two patches can be combined. Only a call that failed is made again, and only once."""

    bench = await _bench(tmp_path, conflicting=True)
    bench.completion.fallback, bench.completion.error = scripted.fallback, scripted.error
    bench.completion.stop_reason = scripted.stop_reason

    report = await _one(bench)

    assert report is not None and report.outcome is MergeOutcome.CONFLICT_SKIPPED
    assert report.reason.startswith(named) and report.request_id is None
    assert len(bench.completion.calls) == calls
    conflicting = bench.deltas["c1"].components[0].add_patches[1]
    assert conflicting.path in report.reason
    assert await bench.outcome("c1") is MergeOutcome.CONFLICT_SKIPPED
    assert bench.merge_tests() == {}
    following = await _one(bench)
    assert following is not None and following.candidate == bench.commits["c2"]


REBASED = RebasedPatch(COMPONENT, "patch/vllm/0001-x.patch", PatchRef("patch/vllm/0001-x.rebased.patch", "f" * 64))


class RebasingRepo(GitExperimentRepo):
    """Reports every merge as having rebased one patch three ways; a test double for the rebase, which needs a file of
    more lines than the bench's package holds."""

    async def merge(self, candidate: str, onto: str, protected: frozenset[str] = frozenset()) -> RepoMerge:
        return replace(await super().merge(candidate, onto, protected), rebased_patches=(REBASED,))


async def test_a_merge_lost_to_a_conflict_names_the_patches_the_rebase_settled_in_the_merge_log(
    tmp_path: Path,
) -> None:
    bench = await _bench(tmp_path, conflicting=True, repo_type=RebasingRepo)
    bench.completion.fallback = json.dumps({"patch": None})

    report = await _one(bench)

    assert report is not None and report.outcome is MergeOutcome.CONFLICT_SKIPPED
    assert report.rebased_patches == (REBASED,) and report.resolved_patches == ()
    [logged] = bench.logged()
    assert logged["detail"]["rebased_patches"] == [{"component": COMPONENT, "path": REBASED.path}]
    assert logged["detail"]["resolved_patches"] == []


#: A patch of the line ``c1``'s patch rewrites as well, so that ``c1``'s patch replaces it where it may.
HEAD_TEXT = _patch_text("x.py", "old", "head")


async def _committed_head_patch(bench: Bench, *, foundation: bool) -> str:
    """Branch ``s2``'s commit of :data:`HEAD_TEXT` on the root, judged +9% and so merged before ``c1``."""

    patch = await bench.repo.store_patch(COMPONENT, HEAD_TEXT)
    pinned = ComponentChange(COMPONENT, UPSTREAM_REF, UPSTREAM_REPO, add_patches=(patch,))
    delta = RecipeDelta(config=ConfigDelta(server_args=f"{TAG_FLAG}head"), components=(pinned,))
    return await _committed_elsewhere(bench, delta, 9.0, foundation=foundation)


async def test_a_patch_a_foundation_fix_brought_is_never_replaced_by_a_later_merge(tmp_path: Path) -> None:
    bench = await _bench(tmp_path, {frozenset({"head"}): {MAIN_POINT: 1040.0}}, tags=("c1",))
    fix = await _committed_head_patch(bench, foundation=True)

    merged, conflicted = await _one(bench), await _one(bench)

    assert merged is not None and (merged.candidate, merged.outcome) == (fix, MergeOutcome.MERGED)
    assert conflicted is not None and conflicted.candidate == bench.commits["c1"]
    assert conflicted.outcome is MergeOutcome.CONFLICT_SKIPPED and conflicted.reason.startswith(CONFLICT_UNRESOLVED)
    assert HEAD_TEXT in bench.completion.calls[0][1]
    head = await bench.repo.read_recipe(await bench.repo.head(INTEGRATION_BRANCH))
    [kept] = head.component(COMPONENT).patches
    assert await bench.repo.patch_text(merged.merged_commit, kept.path) == HEAD_TEXT


async def test_a_merge_that_takes_a_head_patch_out_says_so_in_its_reason_and_the_merge_log(tmp_path: Path) -> None:
    curves = {frozenset({"head"}): {MAIN_POINT: 1040.0}, frozenset({"head", "c1"}): {MAIN_POINT: 1070.0}}
    bench = await _bench(tmp_path, curves, tags=("c1",))
    await _committed_head_patch(bench, foundation=False)

    first, replacing = await _one(bench), await _one(bench)

    assert first is not None and replacing is not None and replacing.candidate == bench.commits["c1"]
    [taken] = (await bench.repo.read_recipe(str(first.merged_commit))).component(COMPONENT).patches
    [by] = bench.deltas["c1"].components[0].add_patches
    assert replacing.outcome is MergeOutcome.MERGED
    assert replacing.replaced_patches == (ReplacedPatch(COMPONENT, (taken.path,), by),)
    swap = f"took out the head's {COMPONENT} patch {taken.path} for the candidate's {by.path}"
    assert replacing.reason == f"improved +2.88% on the integration head; {swap}"
    assert (await bench.repo.read_recipe(str(replacing.merged_commit))).component(COMPONENT).patches == (by,)
    logged = bench.logged()[-1]
    assert logged["reason"] == replacing.reason
    assert logged["detail"]["replaced_patches"] == [{"component": COMPONENT, "removed": [taken.path], "by": by.path}]
    assert bench.logged()[0]["detail"]["replaced_patches"] == []


#: A rewrite of the conflicting patch against a line no stack of the bench holds, so it applies nowhere.
STRAY_TEXT = _patch_text("x.py", "stray", "resolved")


async def test_a_rewrite_that_does_not_apply_is_an_unresolved_conflict_and_spends_no_merge_test(tmp_path: Path) -> None:
    bench = await _bench(tmp_path, conflicting=True)
    bench.completion.fallback = json.dumps({"patch": STRAY_TEXT})

    report = await _one(bench)

    assert report is not None and report.outcome is MergeOutcome.CONFLICT_SKIPPED and report.request_id is None
    conflicting = bench.deltas["c1"].components[0].add_patches[1]
    assert report.reason.startswith(CONFLICT_UNRESOLVED)
    assert f"no rewrite of {conflicting.path} applied in {RESOLVER_ATTEMPTS} attempts, the last: " in report.reason
    assert "does not apply" in report.reason
    assert len(bench.completion.calls) == RESOLVER_ATTEMPTS
    assert bench.merge_tests() == {}
    assert await bench.outcome("c1") is MergeOutcome.CONFLICT_SKIPPED


@pytest.mark.parametrize(
    ("first", "quoted"),
    [(json.dumps({"patch": STRAY_TEXT}), "does not apply"), ("no rewrite today", NO_PATCH)],
    ids=["does-not-apply", "no-patch"],
)
async def test_a_rewrite_that_does_not_apply_is_asked_for_again_with_why(
    tmp_path: Path, first: str, quoted: str
) -> None:
    bench = await _bench(tmp_path, conflicting=True)
    bench.completion.replies = [first, json.dumps({"patch": RESOLVED_TEXT})]

    report = await _one(bench)

    assert report is not None and report.outcome is MergeOutcome.MERGED
    [written] = (await bench.handle(Role.OBSERVER).snapshot()).resolved_patches
    assert written.attempts == RESOLVER_ATTEMPTS
    (_, asked), (_, again) = bench.completion.calls
    assert again.startswith(asked)
    feedback = again.removeprefix(asked)
    assert feedback.startswith("\n\nYour last rewrite failed to apply: ") and quoted in feedback


@dataclass
class UnreachedFirst(ScriptedCompletion):
    """Fails the first ``unreached`` calls as the gateway timing out, then answers as scripted."""

    unreached: int = 0

    async def __call__(self, system: str, user: str) -> CompletionReply:
        if self.unreached:
            self.unreached -= 1
            self.calls.append((system, user))
            raise CompletionFailed(GATEWAY_TIMEOUT)
        return await super().__call__(system, user)


async def test_a_resolver_call_that_failed_is_made_again_and_counts_as_no_answer(tmp_path: Path) -> None:
    bench = await _bench(tmp_path, conflicting=True, completion_type=UnreachedFirst)
    assert isinstance(bench.completion, UnreachedFirst)
    bench.completion.unreached = RESOLVER_UNREACHED_RETRIES
    bench.completion.replies = [json.dumps({"patch": STRAY_TEXT}), json.dumps({"patch": RESOLVED_TEXT})]

    report = await _one(bench)

    assert report is not None and report.outcome is MergeOutcome.MERGED
    [written] = (await bench.handle(Role.OBSERVER).snapshot()).resolved_patches
    assert written.attempts == RESOLVER_ATTEMPTS
    asked = [user for _, user in bench.completion.calls]
    assert len(asked) == RESOLVER_UNREACHED_RETRIES + RESOLVER_ATTEMPTS
    assert len(set(asked[: RESOLVER_UNREACHED_RETRIES + 1])) == 1


async def test_a_merge_the_recipe_contract_refuses_is_skipped_and_the_round_goes_on(tmp_path: Path) -> None:
    """``c2`` carries ``c1``'s change of the base image and is refused with it; ``c2``'s change made again on a branch
    of its own is merged."""

    bench = await _bench(tmp_path, invalid=True)
    apart = await _committed_elsewhere(bench, bench.deltas["c2"], 1.0)

    reports = await bench.integrator.integrate_next(3, EXPLORE_TICK_SECONDS, ())

    assert [(r.candidate, r.outcome) for r in reports] == [
        (bench.commits["c1"], MergeOutcome.CONFLICT_SKIPPED),
        (bench.commits["c2"], MergeOutcome.CONFLICT_SKIPPED),
        (apart, MergeOutcome.MERGED),
    ]
    assert all("does not combine" in r.reason and "base image" in r.reason for r in reports[:2])
    assert await bench.outcome("c1") is MergeOutcome.CONFLICT_SKIPPED


async def _commit_moving_the_component(bench: Bench, ref: str) -> str:
    """Branch ``s2`` starts from the root and commits ``c3``, a server argument whose recipe labels ``vllm`` with the
    upstream commit ``ref`` instead of the one the root recorded."""

    scheduler, toolbox, gate = (bench.handle(role) for role in (Role.SCHEDULER, Role.TOOLBOX, Role.GATE))
    await scheduler.record_lineage(LineageEvent(LineageKind.OPEN, "s2", from_commit=bench.root, reason="second batch"))
    await bench.repo.create_branch("s2", bench.root)
    config = ConfigDelta(server_args=f"--max-num-batched-tokens 2048 {TAG_FLAG}c3")
    change = RecipeDelta(config, components=(ComponentChange(COMPONENT, ref, UPSTREAM_REPO),))
    commit = await bench.repo.commit("s2", bench.root, apply(await bench.repo.read_recipe(bench.root), change), "c3")
    test = request("rc3", intent("s2", bench.root, mechanism="c3"), change)
    await toolbox.write_request(test)
    await toolbox.write_result(result(test.request_id))
    await gate.write_judgement(test.request_id, judgement(1.0, (test.request_id,)), Conclusion.COMMITTED, commit)
    return commit


async def test_a_candidate_naming_another_upstream_commit_is_merged_and_the_head_keeps_its_pin(tmp_path: Path) -> None:
    bench = await _bench(tmp_path, {**CURVES, frozenset({"c1", "c2", "c3"}): {MAIN_POINT: 1080.0}})
    for _ in range(2):
        await _one(bench)
    head = await bench.repo.head(INTEGRATION_BRANCH)
    ref = "b" * 40
    moved = await _commit_moving_the_component(bench, ref)

    report = await _one(bench)

    assert report is not None and (report.candidate, report.onto, report.outcome) == (
        moved,
        head,
        MergeOutcome.MERGED,
    )
    assert report.request_id is not None and "does not combine" not in report.reason
    tested_refs = {change.upstream_ref for test in bench.merge_tests().values() for change in test.delta.components}
    assert ref not in tested_refs
    assert await bench.handle(Role.OBSERVER).search(f"both sides moved {COMPONENT}") == ()
    [logged] = [row for row in bench.logged() if row["detail"]["candidate"] == moved]
    assert (logged["kind"], logged["branch"]) == (DecisionKind.MERGE.value, INTEGRATION_BRANCH)
    assert logged["detail"]["outcome"] == MergeOutcome.MERGED.value
    pinned = (await bench.repo.read_recipe(await bench.repo.head(INTEGRATION_BRANCH))).component(COMPONENT)
    assert pinned is not None and pinned.upstream_ref == UPSTREAM_REF


async def test_a_merge_test_the_toolbox_refuses_is_skipped_and_the_round_goes_on(bench: Bench) -> None:
    """``c2``'s merge carries ``c1`` and is refused with it; ``c2``'s change made again on a branch of its own is
    merged."""

    bench.refuse("c1")
    apart = await _committed_elsewhere(bench, bench.deltas["c2"], 1.0)

    reports = await bench.integrator.integrate_next(3, EXPLORE_TICK_SECONDS, ())

    assert [(r.candidate, r.outcome) for r in reports] == [
        (bench.commits["c1"], MergeOutcome.CONFLICT_SKIPPED),
        (bench.commits["c2"], MergeOutcome.CONFLICT_SKIPPED),
        (apart, MergeOutcome.MERGED),
    ]
    assert all("base_image_mismatch" in r.reason and r.request_id is None for r in reports[:2])
    assert bench.completion.calls == []
    assert await bench.outcome("c1") is MergeOutcome.CONFLICT_SKIPPED


async def test_a_merge_test_that_measures_nothing_writes_nothing_to_the_integration_branch(bench: Bench) -> None:
    bench.toolbox.script(lambda test: test.purpose is Purpose.INTEGRATION, [Fate.MEASUREMENT])

    report = await _one(bench)

    assert report is not None and report.outcome is MergeOutcome.PENDING and report.request_id
    assert "measurement_failed" in report.reason
    assert await bench.repo.head(INTEGRATION_BRANCH) == bench.root
    assert await bench.outcome("c1") is MergeOutcome.PENDING


async def test_nothing_to_integrate_once_every_commit_is_merged(bench: Bench) -> None:
    merged = [await _one(bench) for _ in range(2)]

    assert [report.outcome for report in merged if report is not None] == [MergeOutcome.MERGED] * 2
    assert await _one(bench) is None
    assert len(bench.logged()) == 2


def _leading(branch: str, head: str) -> tuple[StrandedLead, ...]:
    """``branch`` stranded at ``head``, 10% ahead of integration over a 1% bar; the integrator reads only those two."""

    return (StrandedLead(branch, True, head, GRADED_OUTPUT, 1100.0, 1000.0, 10.0, 1.0, ()),)


async def test_a_stranded_lead_s_head_is_adopted_first_and_lands_what_its_branch_holds(tmp_path: Path) -> None:
    bench = await _bench(tmp_path, {**CURVES, frozenset({"c1"}): {MAIN_POINT: 1005.0}})
    refused = await _one(bench)
    c1, c2 = bench.commits["c1"], bench.commits["c2"]

    [adopted] = await bench.integrator.integrate_next(2, EXPLORE_TICK_SECONDS, _leading("s1", c2))

    assert refused is not None and (refused.candidate, refused.outcome) == (c1, MergeOutcome.NOT_STACKABLE)
    assert (adopted.candidate, adopted.onto, adopted.adopts) == (c2, bench.root, "s1")
    assert adopted.outcome is MergeOutcome.MERGED and not adopted.foundation_fix
    assert adopted.reason.startswith(f"{ADOPTING_LEAD.format('s1')}: improved ")
    assert {rid: test.consistency_check for rid, test in bench.merge_tests().items()} == {
        str(refused.request_id): None,
        str(adopted.request_id): True,
    }
    assert await bench.repo.read_recipe(str(adopted.merged_commit)) == await bench.repo.read_recipe(c2)
    merged = {item.commit_id: item.merged_into for item in await bench.handle(Role.OBSERVER).committed()}
    assert merged == {c1: adopted.merged_commit, c2: adopted.merged_commit}
    assert [(row["reason"], row["detail"]["adopts"]) for row in bench.logged()] == [
        (refused.reason, ""),
        (adopted.reason, "s1"),
    ]


async def test_a_lead_is_adopted_once_per_integration_head_and_a_skipped_adoption_says_it_is_one(bench: Bench) -> None:
    bench.refuse("c2")
    c1, c2 = bench.commits["c1"], bench.commits["c2"]

    [skipped] = await bench.integrator.integrate_next(1, EXPLORE_TICK_SECONDS, _leading("s1", c2))
    [following] = await bench.integrator.integrate_next(1, EXPLORE_TICK_SECONDS, _leading("s1", c2))

    adopting = f"{ADOPTING_LEAD.format('s1')}: "
    assert (skipped.candidate, skipped.outcome) == (c2, MergeOutcome.CONFLICT_SKIPPED)
    assert skipped.reason.startswith(f"{adopting}the toolbox refused the merge test: ")
    [recorded] = [item for item in await bench.handle(Role.OBSERVER).committed() if item.commit_id == c2]
    assert recorded.last_merge is not None and recorded.last_merge.reason == skipped.reason
    assert (following.candidate, following.adopts, following.outcome) == (c1, "", MergeOutcome.MERGED)


async def test_a_lead_whose_head_was_already_tried_on_this_head_is_not_adopted(tmp_path: Path) -> None:
    """Adopting it would test the same merge on the same head again, which differs only by noise."""

    bench = await _bench(tmp_path, {**CURVES, frozenset({"c1"}): {MAIN_POINT: 1005.0}})
    refused = await _one(bench)
    c1 = bench.commits["c1"]

    reports = await bench.integrator.integrate_next(1, EXPLORE_TICK_SECONDS, _leading("s1", c1))

    assert refused is not None and (refused.candidate, refused.outcome) == (c1, MergeOutcome.NOT_STACKABLE)
    assert all(not report.adopts and report.candidate != c1 for report in reports)
    assert len(bench.merge_tests()) == 1 + len(reports)


async def test_a_lead_adoption_is_never_judged_as_a_foundation_fix(tmp_path: Path) -> None:
    bench = await _bench(tmp_path, {frozenset({"c1"}): {MAIN_POINT: 950.0}}, foundation="c1", tags=("c1",))

    [adopted] = await bench.integrator.integrate_next(1, EXPLORE_TICK_SECONDS, _leading("s1", bench.commits["c1"]))

    assert adopted.outcome is MergeOutcome.NOT_STACKABLE and not adopted.foundation_fix
    assert adopted.reason.startswith(f"{ADOPTING_LEAD.format('s1')}: regressed ")
    assert await bench.repo.head(INTEGRATION_BRANCH) == bench.root


async def _cancel(task: asyncio.Task[object]) -> None:
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_a_merge_cancelled_during_its_test_is_abandoned_and_its_test_withdrawn(bench: Bench) -> None:
    bench.toolbox.executor.gate.clear()
    merging = asyncio.ensure_future(bench.integrator.integrate_next(1, EXPLORE_TICK_SECONDS, ()))
    await _until(lambda: bool(bench.merge_tests()))

    await _cancel(merging)
    await bench.integrator.abandon()

    [request_id] = bench.merge_tests()
    assert bench.toolbox.withdrawn.keys() == {request_id}
    assert (await bench.handle(Role.OBSERVER).record(request_id)).withdrawn_at is not None
    assert await bench.repo.head(INTEGRATION_BRANCH) == bench.root


async def test_a_merge_cancelled_while_it_is_retested_is_abandoned_and_its_retest_withdrawn(tmp_path: Path) -> None:
    bench = await _bench(tmp_path, toolbox_type=HeldRetests)
    held = bench.toolbox
    assert isinstance(held, HeldRetests)
    merging = asyncio.ensure_future(bench.integrator.integrate_next(1, EXPLORE_TICK_SECONDS, ()))
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
    merging = asyncio.ensure_future(bench.integrator.integrate_next(1, EXPLORE_TICK_SECONDS, ()))
    await repo.committing.wait()

    await _cancel(merging)
    abandoning = asyncio.ensure_future(bench.integrator.abandon())
    repo.released.set()
    await abandoning

    [request_id] = bench.merge_tests()
    record = await bench.handle(Role.OBSERVER).record(request_id)
    assert (record.conclusion, record.commit_id) == (Conclusion.MERGED, await repo.head(INTEGRATION_BRANCH))
    assert bench.toolbox.withdrawn == {}
