# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""PRELUDE's contract: what it roots the session on, the baseline, the root's profile and the champion's replay it
runs through the toolbox, how it fails, and which earlier PRELUDE a session may be checked against."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import pytest

from hyperloom.common.outside_facts.rule_pack import rule_pack_identity
from hyperloom.common.perf_metric import GRADED_OUTPUT
from hyperloom.contracts.measurement import Purpose, TestKind, TestResult
from hyperloom.contracts.recipe import PATCH_ROOT, LaunchConfig, PatchRef, RecipeDelta
from hyperloom.measure.compile import LocalBlobStore
from hyperloom_multibranch.contracts.blackboard import INTEGRATION_BRANCH, Role, Writer
from hyperloom_multibranch.contracts.blackboard.records import ToolResultPointer
from hyperloom_multibranch.contracts.toolbox import RequestRejected
from hyperloom_multibranch.blackboard import LocalBlackboard
from hyperloom_multibranch.blackboard.knowledge.warm_start import (
    LAUNCH_ONLY_REASON,
    PreparedReplay,
    RecipeDownloader,
    ReplayRefusal,
    ReplayTarget,
    WarmLead,
)
from hyperloom_multibranch.repo.git_repo import GitExperimentRepo
from hyperloom_multibranch.scheduling.clock import Clock
from hyperloom_multibranch.scheduling.context import SchedulerContext
from hyperloom_multibranch.scheduling.roster import Roster, RosterCaps
from hyperloom_multibranch.scheduling.decisions import DecisionLog
from hyperloom_multibranch.scheduling.events import PROFILE_UNAVAILABLE, PROFILED, IntegrationProfiler
from hyperloom_multibranch.scheduling.forks import ForkDesk
from hyperloom_multibranch.scheduling.layout import SessionLayout, create_session
from hyperloom_multibranch.rulegate.inferencex import FLOOR_PCT
from hyperloom_multibranch.rulegate.rules import AgentXRule, Rule
from hyperloom_multibranch.rulegate.service import BASELINE_TESTS, GateService
from hyperloom_multibranch.scheduling import prelude as prelude_module
from hyperloom_multibranch.scheduling.prelude import (
    BASELINE_ESTABLISHED,
    PREFLIGHT_OUT_OF_TIME,
    Prelude,
    PreludeFailed,
    PreludeResult,
    PreludeServices,
    ReplayChampion,
    WarmStartSource,
)
from hyperloom_multibranch.scheduling.prelude_reader import MB_FRAMEWORK_REPO_ENV, MB_FRAMEWORK_VERSION_ENV
from hyperloom_multibranch.scheduling.processes import ProcessRegistry
from hyperloom_multibranch.scheduling.requests import SessionTests, integration_test_request
from hyperloom_multibranch.scheduling.settings import InferenceXShape, KernelOptimizer, Workload
from hyperloom_multibranch.scheduling.tests.framework_checkout import framework_checkout, head_of
from hyperloom_multibranch.scheduling.tests.peer_fakes import (
    BASE_IMAGE_DIGEST,
    OneChampionKB,
    RecordedReservations,
    RecordingToolbox,
    measure_no_throughput,
    session_gate,
)
from hyperloom_multibranch.toolbox.fake_model import Fate, FakeToolboxConfig
from hyperloom_multibranch.toolbox.analysis import CODE_TOOL_UNAVAILABLE
from hyperloom_multibranch.toolbox.attempts import TestJob
from hyperloom_multibranch.toolbox.tools.base import JobStatus

NOW = 1_000_000.0
#: The image the fake toolbox measures on, which the root recipe must carry.
IMAGE = BASE_IMAGE_DIGEST
WORKLOAD = Workload(
    model="/models/Llama-3-8B",
    framework="sglang",
    tp=1,
    conc=16,
    precision="bf16",
    gpu_type="mi355x",
    kernel_optimizer=KernelOptimizer.GEAK,
    inferencex=InferenceXShape(isl=1024, osl=1024),
)
AGENTX = replace(WORKLOAD, inferencex=None)
#: The op shares :class:`AnalysingToolbox`'s TraceLens job finds in every profile.
ROOT_SHARES = {"gemm": 48.0, "attention": 31.0}


@pytest.fixture(autouse=True)
def framework(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The framework checkout the scheduler runs on, which the session roots on."""

    checkout = framework_checkout(tmp_path)
    monkeypatch.setenv(MB_FRAMEWORK_REPO_ENV, str(checkout))
    return checkout


class AnalysingToolbox(RecordingToolbox):
    """The recording toolbox with a TraceLens route whose job ends at once, leaving :data:`ROOT_SHARES` on the commit
    the profile test measured, or refused with ``analysis_refusal``; ``analysed`` holds each profile test analysed. A
    double for a backend that profiles, not a placeholder: the fake has no tool jobs of its own."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.analysed: list[str] = []
        self.analysis_refusal: RequestRejected | None = None

    async def submit_analysis(
        self, branch: str, tool: str, params: Mapping[str, Any], profile_request_id: str = ""
    ) -> str:
        if self.analysis_refusal is not None:
            raise self.analysis_refusal
        self.analysed.append(profile_request_id)
        job = f"tracelens-{len(self.analysed)}"
        commit = self.requests[profile_request_id].parent_commit
        pointer = ToolResultPointer(branch, job, "/analysis", "trace analysed", commit, profile_request_id, ROOT_SHARES)
        await self.blackboard.notify_tool_result(pointer)
        return job

    def tool_status(self, job_id: str) -> JobStatus:
        return JobStatus.SUCCEEDED


@dataclass
class PreludePeers:
    """The repository, toolbox and rule gate PRELUDE works with, on the session's blackboard; ``tests`` waits for the
    toolbox's outcomes without real sleeps. ``opened`` holds each commit the integration branch was opened on, and
    ``replays`` each champion replayed, with whether the baseline was established then."""

    repo: GitExperimentRepo
    toolbox: AnalysingToolbox
    tests: SessionTests
    gate: GateService
    opened: list[str] = field(default_factory=list)
    replays: list[tuple[str, bool]] = field(default_factory=list)


async def _instant(seconds: float) -> None:
    await asyncio.sleep(0)


def _peers(layout: SessionLayout, accuracy: float | None = None, rule: Rule | None = None) -> PreludePeers:
    """``accuracy`` is the eval score every baseline test measures, the fake's own when ``None``; ``rule`` is the
    gate's, InferenceX's when ``None``."""

    scores = {} if accuracy is None else {RecipeDelta().fingerprint(): accuracy}
    toolbox_blackboard = LocalBlackboard(layout.blackboard, Writer(Role.TOOLBOX, "toolbox"), clock=_clock().now)
    repo = GitExperimentRepo(layout.repo, LocalBlobStore(layout.blobs))
    toolbox = AnalysingToolbox(toolbox_blackboard, accuracy=scores, conc=WORKLOAD.conc, repo=repo)
    tests = SessionTests(toolbox, Clock(now=_clock().now, sleep=_instant))
    gate_blackboard = LocalBlackboard(layout.blackboard, Writer(Role.GATE, "gate"), clock=_clock().now)
    gate = session_gate(gate_blackboard, toolbox, rule)
    return PreludePeers(repo, toolbox, tests, gate)


def _clock() -> Clock:
    return Clock(now=lambda: NOW)


def _layout(tmp_path: Path) -> SessionLayout:
    return create_session(tmp_path / "workspace", WORKLOAD.model_basename, _clock())


def _blackboard(layout: SessionLayout) -> LocalBlackboard:
    return LocalBlackboard(layout.blackboard, Writer(Role.SCHEDULER, "scheduler"), clock=_clock().now)


def _logged(layout: SessionLayout) -> list[dict[str, Any]]:
    return [json.loads(line) for line in layout.decisions.read_text().splitlines()]


def _established(layout: SessionLayout) -> bool:
    return BASELINE_ESTABLISHED in [entry["reason"] for entry in _logged(layout)]


def _prelude(
    layout: SessionLayout,
    peers: PreludePeers,
    recipe_kb: RecipeDownloader | None = None,
    replay: ReplayChampion | None = None,
    processes: ProcessRegistry | None = None,
) -> Prelude:
    """``replay`` replays the champion in place of the integrator; by default it only notes the replay in
    ``peers.replays`` and waits for the baseline's anchoring."""

    clock = _clock()
    roster = Roster(RosterCaps(1, 1, 1), layout.roster, clock)
    decisions = DecisionLog(layout.decisions, clock)
    desk = ForkDesk(roster, decisions, clock)
    registry = processes or ProcessRegistry(layout.pids, clock)
    context = SchedulerContext(
        layout, roster, _blackboard(layout), decisions, registry, clock, desk, RecordedReservations()
    )
    warm_start = None if recipe_kb is None else WarmStartSource(recipe_kb, LocalBlobStore(layout.blobs))

    async def open_integration(root: str) -> None:
        peers.opened.append(root)

    async def replay_champion(champion: PreparedReplay, root: str, anchored: Awaitable[object]) -> str:
        peers.replays.append((champion.canonical_id, _established(layout)))
        await asyncio.shield(anchored)
        return root

    profiler = IntegrationProfiler(
        peers.toolbox, peers.tests, _blackboard(layout), Clock(now=clock.now, sleep=_instant)
    )
    services = PreludeServices(
        peers.repo,
        peers.gate,
        peers.tests,
        peers.toolbox,
        open_integration,
        replay or replay_champion,
        profiler.profile_root,
    )
    return Prelude(context, services, warm_start=warm_start)


def _warm_start_decision(layout: SessionLayout) -> dict[str, Any]:
    return next(entry["detail"] for entry in _logged(layout) if entry["reason"] == "warm start read")


async def _run(prelude: Prelude, workload: Workload = WORKLOAD) -> PreludeResult:
    return await prelude.root(workload, IMAGE)


def _by_purpose(toolbox: RecordingToolbox, purpose: Purpose) -> list[str]:
    return [request_id for request_id, request in toolbox.requests.items() if request.purpose is purpose]


async def test_prelude_starts_no_process_and_roots_on_the_checkout_with_root_tests_a_root_profile_and_the_champion(
    tmp_path: Path, framework: Path
) -> None:
    layout = _layout(tmp_path)
    processes, peers = ProcessRegistry(layout.pids, _clock()), _peers(layout)

    result = await _run(_prelude(layout, peers, recipe_kb=OneChampionKB(), processes=processes))

    assert not layout.pids.exists() and await processes.leftovers() == ()
    root = result.root_commit
    assert root == await peers.repo.head(INTEGRATION_BRANCH) and peers.opened == [root]
    recipe = await peers.repo.read_recipe(root)
    [component] = recipe.components
    assert (component.upstream_ref, component.patches) == (head_of(framework), ())
    assert (recipe.launch, recipe.base_image_digest) == (LaunchConfig(), IMAGE)
    assert json.loads(layout.root_recipe.read_text()) == recipe.to_document()
    lineage_root = (await _blackboard(layout).lineage()).commit(root)
    assert lineage_root is not None
    assert (lineage_root.branch, lineage_root.recipe_digest) == (INTEGRATION_BRANCH, recipe.digest())
    baseline_tests, [profile] = (
        _by_purpose(peers.toolbox, Purpose.BASELINE),
        _by_purpose(peers.toolbox, Purpose.PROFILE),
    )
    assert len(baseline_tests) == BASELINE_TESTS and len(peers.toolbox.requests) == BASELINE_TESTS + 1
    request = peers.toolbox.requests[profile]
    assert (request.kind, request.branch_id, request.parent_commit) == (TestKind.PROFILE, INTEGRATION_BRANCH, root)
    assert peers.toolbox.analysed == [profile]
    baseline = await _blackboard(layout).baseline()
    assert (baseline.root_commit, sorted(baseline.request_ids)) == (root, sorted(baseline_tests))
    assert result.longest_root_test == FakeToolboxConfig().attempt_seconds
    session_profile = await _blackboard(layout).session_profile()
    assert session_profile == result.profile
    assert (session_profile.session_id, session_profile.hardware) == (layout.session_id, "mi355x")
    assert (session_profile.isl, session_profile.osl, dict(session_profile.roofline_ceiling)) == (1024, 1024, {})
    assert session_profile.kernel_optimizer == "geak"
    assert (session_profile.model_type, session_profile.architectures) == ("", ())
    shares = await _blackboard(layout).op_shares(root)
    assert shares is not None and (dict(shares.shares), shares.profile_request_id) == (ROOT_SHARES, profile)
    assert result.root_profile_job == "tracelens-1"
    assert peers.replays == [("inference:champion", False)]
    rows = _logged(layout)
    [rooted] = [entry["detail"] for entry in rows if entry["reason"] == "root committed"]
    assert rooted == {"root_commit": root}
    [profiled] = [entry for entry in rows if entry["kind"] == "profile"]
    assert (profiled["reason"], profiled["detail"]["commit"], profiled["detail"]["request_id"]) == (
        PROFILED,
        root,
        profile,
    )


async def test_an_agentx_prelude_anchors_its_baseline_on_one_root_test_and_has_no_sequence_lengths(
    tmp_path: Path,
) -> None:
    layout = _layout(tmp_path)
    peers = _peers(layout, rule=AgentXRule())

    result = await _run(_prelude(layout, peers), AGENTX)

    [root_test], [profile] = _by_purpose(peers.toolbox, Purpose.BASELINE), _by_purpose(peers.toolbox, Purpose.PROFILE)
    baseline = await _blackboard(layout).baseline()
    assert (baseline.root_commit, baseline.request_ids, baseline.source) == (result.root_commit, (root_test,), "")
    assert result.longest_root_test == FakeToolboxConfig().attempt_seconds
    assert peers.toolbox.analysed == [profile]
    [established] = [entry for entry in _logged(layout) if entry["reason"] == "baseline established"]
    assert established["detail"]["request_ids"] == [root_test]
    profile_record = await _blackboard(layout).session_profile()
    assert profile_record == result.profile
    assert (profile_record.isl, profile_record.osl, dict(profile_record.roofline_ceiling)) == (None, None, {})


async def test_a_prelude_whose_backend_cannot_analyse_the_root_profile_records_it_and_goes_on(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    peers = _peers(layout)
    peers.toolbox.analysis_refusal = RequestRejected("no TraceLens on this backend", CODE_TOOL_UNAVAILABLE)

    result = await _run(_prelude(layout, peers))

    assert (await _blackboard(layout).baseline()).root_commit == result.root_commit
    assert await _blackboard(layout).op_shares(result.root_commit) is None
    assert result.root_profile_job == ""
    [unavailable] = [entry for entry in _logged(layout) if entry["kind"] == "profile"]
    assert (unavailable["reason"], unavailable["detail"]["code"]) == (PROFILE_UNAVAILABLE, CODE_TOOL_UNAVAILABLE)


async def test_the_recipe_kb_champions_for_the_profile_become_warm_start_leads_prepared_for_replay(
    tmp_path: Path,
) -> None:
    layout = _layout(tmp_path)
    kb = OneChampionKB()
    peers = _peers(layout)

    prelude = _prelude(layout, peers, recipe_kb=kb)
    result = await _run(prelude)

    [lead] = result.warm_leads
    assert (lead.canonical_id, lead.server_args, lead.gain_pct) == (
        "inference:champion",
        "--enable-expert-parallel",
        12.0,
    )
    assert kb.searched[0]["match"]["hardware"] == result.profile.hardware
    assert lead.replay is not None and lead.replay.recipe is None and lead.refusal is None
    assert lead.replay.launch.server_args == "--enable-expert-parallel"
    assert [canonical_id for canonical_id, _ in peers.replays] == [lead.canonical_id]
    assert _warm_start_decision(layout) == {"leads": [lead.describe()], "unread": ""}


async def test_a_replay_acceptance_refuses_on_the_root_stays_a_lead_named_not_replayable(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    peers = _peers(layout)
    tuning = RequestRejected("TunableOp tunes online", "online_tuning_unsupported")
    peers.toolbox.reject_if(lambda request: tuning if request.delta.config.server_args else None)

    prelude = _prelude(layout, peers, recipe_kb=OneChampionKB())
    result = await _run(prelude)

    [lead] = result.warm_leads
    version = rule_pack_identity()["version"]
    assert (lead.replay, lead.refusal) == (
        None,
        ReplayRefusal("online_tuning_unsupported", "TunableOp tunes online", version),
    )
    assert lead.describe().endswith(
        f"; not replayable here under rule pack {version}: online_tuning_unsupported (TunableOp tunes online)"
    )
    assert _warm_start_decision(layout) == {"leads": [lead.describe()], "unread": ""}


OTHER_IMAGE = "sha256:" + "b" * 64
GATE_PATCH = "0099-gate.patch"
#: What acceptance answers a replay whose patch stack does not apply on the root.
STACK_REFUSED = RequestRejected(f"{GATE_PATCH} does not apply", "delta_not_applicable")


def _patched_replays(source_image: str) -> Callable[..., tuple[WarmLead, ...]]:
    """``replay_leads`` preparing each lead's champion as the root plus a patch whose variable its launch sets, built
    on ``source_image`` (the session's own when empty)."""

    def prepared(kb: object, profile: object, found: Sequence[WarmLead], target: ReplayTarget) -> tuple[WarmLead, ...]:
        root = target.root
        gate = PatchRef(f"{PATCH_ROOT}/{root.components[0].component}/{GATE_PATCH}", "f" * 64)
        components = (replace(root.components[0], patches=(*root.components[0].patches, gate)), *root.components[1:])
        champion = replace(root, launch=LaunchConfig("--enable-expert-parallel", {"GATE": "1"}), components=components)
        launch = champion.launch
        return tuple(
            replace(lead, replay=PreparedReplay(lead.canonical_id, champion, launch, "", (), source_image, ("GATE",)))
            for lead in found
        )

    return prepared


async def _lead_with_its_patches_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source_image: str
) -> tuple[WarmLead, SessionLayout]:
    """The one warm lead of a PRELUDE whose champion, built on ``source_image``, adds a patch acceptance refuses."""

    layout = _layout(tmp_path)
    peers = _peers(layout)
    peers.toolbox.reject_if(lambda request: STACK_REFUSED if request.delta.components else None)
    monkeypatch.setattr(prelude_module, "replay_leads", _patched_replays(source_image))
    prelude = _prelude(layout, peers, recipe_kb=OneChampionKB())
    [lead] = (await _run(prelude)).warm_leads
    return lead, layout


async def test_a_champion_from_another_image_whose_patches_acceptance_refuses_keeps_its_launch_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lead, layout = await _lead_with_its_patches_refused(tmp_path, monkeypatch, OTHER_IMAGE)

    assert lead.refusal is None and lead.replay is not None and lead.replay.recipe is None
    assert lead.replay.reason == f"{LAUNCH_ONLY_REASON}: delta_not_applicable"
    assert (lead.replay.source_image, dict(lead.replay.launch.envs)) == (OTHER_IMAGE, {})
    assert ("patch", GATE_PATCH) in [(item.kind, item.ref.rsplit("/", 1)[-1]) for item in lead.replay.references]
    assert _warm_start_decision(layout) == {"leads": [lead.describe()], "unread": ""}


async def test_a_champion_from_this_image_whose_patches_acceptance_refuses_is_named_not_replayable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lead, _ = await _lead_with_its_patches_refused(tmp_path, monkeypatch, "")

    version = rule_pack_identity()["version"]
    assert (lead.replay, lead.refusal) == (None, ReplayRefusal("delta_not_applicable", STACK_REFUSED.reason, version))


async def test_a_replay_acceptance_has_not_answered_for_in_preludes_time_stays_offered_marked_not_checked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout = _layout(tmp_path)
    peers = _peers(layout)

    async def unanswered(request: object) -> None:
        await asyncio.sleep(60)

    monkeypatch.setattr(peers.toolbox, "check_test", unanswered)
    monkeypatch.setattr(prelude_module, "PREFLIGHT_SECONDS", 0.01)
    prelude = _prelude(layout, peers, recipe_kb=OneChampionKB())
    result = await _run(prelude)

    [lead] = result.warm_leads
    assert (lead.refusal, lead.unchecked) == (None, PREFLIGHT_OUT_OF_TIME) and lead.replay is not None
    assert lead.describe().endswith(f"; not checked against acceptance: {PREFLIGHT_OUT_OF_TIME}")
    assert _warm_start_decision(layout) == {"leads": [lead.describe()], "unread": ""}


@pytest.mark.parametrize(
    ("recipe_kb", "unread"),
    [(None, "no Recipe KB is configured"), (OneChampionKB(unreachable=True), "KBStoreError: connection refused")],
)
async def test_without_a_readable_recipe_kb_the_session_roots_with_no_leads(
    recipe_kb: RecipeDownloader | None, unread: str, tmp_path: Path
) -> None:
    layout = _layout(tmp_path)

    prelude = _prelude(layout, _peers(layout), recipe_kb=recipe_kb)
    result = await _run(prelude)

    assert result.warm_leads == () and result.start_commit == result.root_commit
    assert _warm_start_decision(layout) == {"leads": [], "unread": unread}


async def test_the_profile_names_the_model_type_and_architectures_of_the_models_config(tmp_path: Path) -> None:
    model = tmp_path / "models" / "Qwen3-8B"
    model.mkdir(parents=True)
    (model / "config.json").write_text(json.dumps({"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"]}))
    workload = replace(WORKLOAD, model=str(model))
    layout = _layout(tmp_path)
    peers = _peers(layout)

    prelude = _prelude(layout, peers)
    result = await _run(prelude, workload)

    assert (result.profile.model_type, result.profile.architectures) == ("qwen3", ("Qwen3ForCausalLM",))
    assert await _blackboard(layout).session_profile() == result.profile


async def test_baseline_is_measured_through_the_toolbox(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    peers = _peers(layout)

    result = await _run(_prelude(layout, peers))

    baseline_tests = _by_purpose(peers.toolbox, Purpose.BASELINE)
    assert len(baseline_tests) == BASELINE_TESTS
    baseline = await _blackboard(layout).baseline()
    assert baseline.root_commit == result.root_commit
    assert sorted(baseline.request_ids) == sorted(baseline_tests)
    assert baseline.values[GRADED_OUTPUT] == pytest.approx(1000.0)
    assert dict(baseline.threshold_pct) == {GRADED_OUTPUT: FLOOR_PCT}
    assert set(baseline_tests) <= set(peers.toolbox.results)
    for request_id in baseline_tests:
        request = peers.toolbox.requests[request_id]
        assert (request.branch_id, request.parent_commit) == (INTEGRATION_BRANCH, result.root_commit)
        assert (request.kind, request.delta) == (TestKind.TEST, RecipeDelta())
        assert (request.accuracy, request.intent_fingerprint, request.priority) == (True, None, None)


async def test_measuring_the_baseline_reports_its_longest_root_test(tmp_path: Path) -> None:
    layout = _layout(tmp_path)

    longest = (await _run(_prelude(layout, _peers(layout)))).longest_root_test

    assert longest == FakeToolboxConfig().attempt_seconds
    [established] = [entry for entry in _logged(layout) if entry["reason"] == "baseline established"]
    assert established["detail"]["longest_test_seconds"] == longest


async def test_baseline_tests_that_fail_to_measure_fail_prelude(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    peers = _peers(layout)
    peers.toolbox.script(lambda request: request.purpose is Purpose.BASELINE, [Fate.MEASUREMENT])

    with pytest.raises(PreludeFailed, match="the baseline anchors nothing: 0 of 3 root tests measured"):
        await _run(_prelude(layout, peers))

    assert not _established(layout)


async def test_a_baseline_below_the_accuracy_floor_fails_prelude(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    peers = _peers(layout, accuracy=0.0)
    with pytest.raises(PreludeFailed, match="the baseline anchors nothing: baseline accuracy"):
        await _run(_prelude(layout, peers))
    assert set(_by_purpose(peers.toolbox, Purpose.BASELINE)) <= set(peers.toolbox.results)


async def test_a_baseline_measuring_no_throughput_fails_prelude(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    measure_no_throughput(monkeypatch)
    layout = _layout(tmp_path)
    with pytest.raises(PreludeFailed, match="the baseline anchors nothing: .*: a throughput must be positive"):
        await _run(_prelude(layout, _peers(layout)))


async def test_refused_baseline_test_fails_prelude_and_takes_the_accepted_ones_back(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    peers = _peers(layout)
    peers.toolbox.executor.gate.clear()
    accepted = iter(range(BASELINE_TESTS - 1))
    refusal = RequestRejected("the base image moved", code="base_image_mismatch")
    peers.toolbox.reject_if(
        lambda request: None if request.purpose is not Purpose.BASELINE or next(accepted, None) is not None else refusal
    )

    with pytest.raises(PreludeFailed, match="refused the baseline test: base_image_mismatch"):
        await _run(_prelude(layout, peers))

    baseline_tests = _by_purpose(peers.toolbox, Purpose.BASELINE)
    assert len(baseline_tests) == BASELINE_TESTS - 1
    assert set(baseline_tests) <= set(peers.toolbox.withdrawn)
    assert not set(baseline_tests) & set(peers.toolbox.results)


async def test_a_cancelled_prelude_takes_its_root_tests_back(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    peers = _peers(layout)
    peers.toolbox.executor.gate.clear()
    task = asyncio.create_task(_run(_prelude(layout, peers)))
    while len(_by_purpose(peers.toolbox, Purpose.BASELINE)) < BASELINE_TESTS:
        await asyncio.sleep(0.01)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert set(peers.toolbox.requests) <= set(peers.toolbox.withdrawn)


def _measuring_replay(
    layout: SessionLayout, peers: PreludePeers, events: list[tuple[str, bool]], failure: Exception | None = None
) -> ReplayChampion:
    """A replay that measures as the integrator's does: its test is submitted at once and judged once the baseline is
    anchored, the test taken back when the replay is cancelled. ``events`` notes ``("measured", baseline established)``
    once the test has ended, then ``("judged", baseline established)`` or ``("cancelled", baseline established)``;
    ``failure`` is raised in place of the judgement."""

    async def replay(champion: PreparedReplay, root: str, anchored: Awaitable[object]) -> str:
        peers.replays.append((champion.canonical_id, _established(layout)))

        async def judge(request_id: str) -> str:
            await peers.toolbox.outcome(request_id)
            events.append(("measured", _established(layout)))
            if failure is not None:
                raise failure
            try:
                await asyncio.shield(anchored)
            except asyncio.CancelledError:
                events.append(("cancelled", _established(layout)))
                raise
            events.append(("judged", _established(layout)))
            return root

        return await peers.tests.judged(integration_test_request(TestKind.TEST, Purpose.INTEGRATION, root), judge)

    return replay


def _replaying(
    tmp_path: Path, rule: Rule | None = None, failure: Exception | None = None
) -> tuple[SessionLayout, PreludePeers, list[tuple[str, bool]], Prelude]:
    """PRELUDE with a Recipe KB champion, replayed by :func:`_measuring_replay`."""

    layout = _layout(tmp_path)
    peers = _peers(layout, rule=rule)
    events: list[tuple[str, bool]] = []
    replay = _measuring_replay(layout, peers, events, failure)
    return layout, peers, events, _prelude(layout, peers, recipe_kb=OneChampionKB(), replay=replay)


def _root_tests_held_until(monkeypatch: pytest.MonkeyPatch, peers: PreludePeers, released: Callable[[], bool]) -> None:
    """Every root test's attempt waits until ``released`` says so."""

    run = peers.toolbox.executor.run

    async def held(job: TestJob) -> TestResult:
        while job.request.purpose is Purpose.BASELINE and not released():
            await asyncio.sleep(0.01)
        return await run(job)

    monkeypatch.setattr(peers.toolbox.executor, "run", held)


async def test_the_champion_is_replayed_beside_the_root_tests_on_the_profile_and_judged_once_the_baseline_anchored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(MB_FRAMEWORK_VERSION_ENV, "0.5.2")
    layout, peers, events, prelude = _replaying(tmp_path)
    _root_tests_held_until(monkeypatch, peers, lambda: bool(_by_purpose(peers.toolbox, Purpose.INTEGRATION)))

    result = await _run(prelude)

    assert peers.replays == [("inference:champion", False)]
    assert events[-1] == ("judged", True)
    assert result.profile.framework_version == "0.5.2"
    assert (await _blackboard(layout).session_profile()).framework_version == "0.5.2"


async def test_an_agentx_champion_measured_before_its_root_test_ended_is_judged_once_that_test_anchored_the_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout, peers, events, prelude = _replaying(tmp_path, AgentXRule())
    _root_tests_held_until(monkeypatch, peers, lambda: bool(events))

    await _run(prelude, AGENTX)

    assert peers.replays == [("agentx:champion", False)]
    assert events == [("measured", False), ("judged", True)]
    assert len(_by_purpose(peers.toolbox, Purpose.BASELINE)) == 1


async def test_a_failing_agentx_replay_takes_back_the_root_test_running_beside_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    failure = RuntimeError("the integration branch cannot be written")
    layout, peers, events, prelude = _replaying(tmp_path, AgentXRule(), failure)
    _root_tests_held_until(monkeypatch, peers, lambda: False)

    with pytest.raises(RuntimeError, match="cannot be written"):
        await asyncio.wait_for(_run(prelude, AGENTX), 60)

    [root_test] = _by_purpose(peers.toolbox, Purpose.BASELINE)
    assert root_test in peers.toolbox.withdrawn and root_test not in peers.toolbox.results
    assert not _established(layout)


async def test_a_failing_agentx_root_test_ends_the_replay_running_beside_it(tmp_path: Path) -> None:
    layout, peers, events, prelude = _replaying(tmp_path, AgentXRule())
    peers.toolbox.script(lambda request: request.purpose is Purpose.BASELINE, [Fate.MEASUREMENT])

    with pytest.raises(PreludeFailed, match="the baseline anchors nothing"):
        await asyncio.wait_for(_run(prelude, AGENTX), 60)

    assert ("judged", True) not in events and not _established(layout)
