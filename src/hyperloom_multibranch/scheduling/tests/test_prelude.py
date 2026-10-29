# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""PRELUDE's contract: the child's command line, what a finished child roots the session on, and how it fails."""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import pytest

import hyperloom
from hyperloom.common.io import atomic_write_json
from hyperloom.common.outside_facts.rule_pack import rule_pack_identity
from hyperloom.common.perf_metric import GRADED_OUTPUT
from hyperloom.contracts.measurement import Purpose, TestKind, TestRequest
from hyperloom.contracts.recipe import PATCH_ROOT, LaunchConfig, PatchRef, Recipe, RecipeDelta
from hyperloom.measure.compile import LocalBlobStore
from hyperloom_multibranch.contracts.blackboard import INTEGRATION_BRANCH, Role, Writer
from hyperloom_multibranch.contracts.blackboard.records import ToolResultPointer
from hyperloom_multibranch.contracts.blackboard.vocab import LeaseKind
from hyperloom_multibranch.contracts.toolbox import RequestRejected
from hyperloom_multibranch.blackboard import LocalBlackboard
from hyperloom_multibranch.blackboard.codec import encode
from hyperloom_multibranch.blackboard.knowledge.warm_start import (
    LAUNCH_ONLY_REASON,
    PreparedReplay,
    RecipeDownloader,
    ReplayRefusal,
    ReplayTarget,
    WarmLead,
)
from hyperloom_multibranch.repo.git_repo import GitExperimentRepo
from hyperloom_multibranch.scheduling.budget import HardStop, SessionBudget
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
    CHILD_CARDS_ENV,
    CHILD_FIRST,
    CHILD_TRACE_ANALYSIS_ROUTE,
    PREFLIGHT_OUT_OF_TIME,
    Prelude,
    PreludeFailed,
    PreludeResult,
    PreludeServices,
    TRACE_ANALYSIS_ROUTE_ENV,
    WarmStartSource,
    prelude_argv,
    reused_prelude,
)
from hyperloom_multibranch.scheduling.prelude_reader import MB_FRAMEWORK_REPO_ENV, MB_FRAMEWORK_VERSION_ENV
from hyperloom_multibranch.scheduling.processes import ProcessRegistry, RegisteredProcess
from hyperloom_multibranch.scheduling.requests import SessionTests
from hyperloom_multibranch.scheduling.settings import InferenceXShape, KernelOptimizer, ReusedPrelude, Workload
from hyperloom_multibranch.scheduling.tests.peer_fakes import (
    BASE_IMAGE_DIGEST,
    OneChampionKB,
    LAUNCH_FLAG,
    LeasingToolbox,
    MbFakeChildRegistry,
    RecordedReservations,
    measure_no_throughput,
    session_gate,
)
from hyperloom_multibranch.toolbox.fake_model import Fate, FakeToolboxConfig
from hyperloom_multibranch.toolbox.analysis import CODE_TOOL_UNAVAILABLE
from hyperloom_multibranch.toolbox.tools.base import JobStatus
from hyperloom_multibranch.scheduling.tests.child_sessions import (
    AGENTX_BENCHMARK,
    AGENTX_RUN_ID,
    AGENTX_RUN_PERF,
    AGENTX_RUN_SECONDS,
    BASELINE_ERROR_CLASS,
    POINT_CONFIG,
    ROOFLINE_PEAK,
    ROOFLINE_RUN_ID,
    TRACE_FILE,
    child_with_patch,
    failed_child,
    framework_checkout,
    head_of,
    with_agentx_baseline_run,
    with_roofline_profile,
    write_child_session,
)

NOW = 1_000_000.0
#: The image the fake toolbox measures on, which the root recipe must carry.
IMAGE = BASE_IMAGE_DIGEST
STUBBORN_CHILD = ("sh", "-c", "trap '' TERM; sleep 60")
#: Real seconds a child that outlives its root tests runs; one that holds PRELUDE up runs for :data:`LONG_CHILD`.
SHORT_CHILD, LONG_CHILD = 2.0, 600.0
#: Clock seconds left before the hard stop when the stubborn child's PRELUDE is cancelled.
NEAR_HARD_STOP_SECONDS = 0.4
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


@pytest.fixture(autouse=True)
def child_framework(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The scheduler, and so the child, is launched on the framework checkout the child sessions are built on."""

    monkeypatch.setenv(MB_FRAMEWORK_REPO_ENV, str(tmp_path / "framework"))


class StubbornRegistry(ProcessRegistry):
    """Runs a main-flow child that ignores SIGTERM; a failure double, not a placeholder."""

    async def spawn(
        self, argv: Sequence[str], *, branch: str, log: Path, env: Mapping[str, str] | None = None
    ) -> RegisteredProcess:
        return await super().spawn(STUBBORN_CHILD, branch=branch, log=log, env=env)


@dataclass
class PreludePeers:
    """The repository, toolbox and rule gate PRELUDE works with, on the session's blackboard; ``tests`` waits for the
    toolbox's outcomes without real sleeps. ``opened`` holds each commit the integration branch was opened on, with
    whether the child had been started then, and ``replays`` each champion replayed, with whether the child had
    exited then; ``on_this_machine`` says whether the toolbox measures on this machine's cards, and so leases the
    child its cards, or in sandboxes elsewhere, where PRELUDE starts no child."""

    repo: GitExperimentRepo
    toolbox: LeasingToolbox
    tests: SessionTests
    gate: GateService
    opened: list[tuple[str, bool]] = field(default_factory=list)
    replays: list[tuple[str, bool]] = field(default_factory=list)
    on_this_machine: bool = True


async def _instant(seconds: float) -> None:
    await asyncio.sleep(0)


def _peers(
    layout: SessionLayout,
    accuracy: float | None = None,
    rule: Rule | None = None,
    toolbox_type: type[LeasingToolbox] = LeasingToolbox,
) -> PreludePeers:
    """``accuracy`` is the eval score every baseline test measures, the fake's own when ``None``; ``rule`` is the
    gate's, InferenceX's when ``None``."""

    scores = {} if accuracy is None else {RecipeDelta().fingerprint(): accuracy}
    toolbox_blackboard = LocalBlackboard(layout.blackboard, Writer(Role.TOOLBOX, "toolbox"), clock=_clock().now)
    repo = GitExperimentRepo(layout.repo, LocalBlobStore(layout.blobs))
    toolbox = toolbox_type(toolbox_blackboard, accuracy=scores, conc=WORKLOAD.conc, repo=repo)
    toolbox.clock = _clock().now
    tests = SessionTests(toolbox, Clock(now=_clock().now, sleep=_instant))
    gate_blackboard = LocalBlackboard(layout.blackboard, Writer(Role.GATE, "gate"), clock=_clock().now)
    gate = session_gate(gate_blackboard, toolbox, rule)
    return PreludePeers(repo, toolbox, tests, gate)


def _clock(times: list[float] | None = None) -> Clock:
    current = [NOW] if times is None else times
    return Clock(now=lambda: current[0])


def _layout(tmp_path: Path) -> SessionLayout:
    return create_session(tmp_path / "workspace", WORKLOAD.model_basename, _clock())


def _blackboard(layout: SessionLayout) -> LocalBlackboard:
    return LocalBlackboard(layout.blackboard, Writer(Role.SCHEDULER, "scheduler"), clock=_clock().now)


def _prelude(
    layout: SessionLayout,
    processes: ProcessRegistry,
    peers: PreludePeers,
    clock: Clock | None = None,
    recipe_kb: RecipeDownloader | None = None,
) -> Prelude:
    clock = clock or _clock()
    roster = Roster(RosterCaps(1, 1, 1), layout.roster, clock)
    decisions = DecisionLog(layout.decisions, clock)
    desk = ForkDesk(roster, decisions, clock)
    context = SchedulerContext(
        layout, roster, _blackboard(layout), decisions, processes, clock, desk, RecordedReservations()
    )
    warm_start = None if recipe_kb is None else WarmStartSource(recipe_kb, LocalBlobStore(layout.blobs))

    child = processes if isinstance(processes, MbFakeChildRegistry) else None

    async def open_integration(root: str) -> None:
        peers.opened.append((root, child is not None and child.child is not None))

    async def replay_champion(champion: PreparedReplay, root: str, anchored: Awaitable[object]) -> str:
        peers.replays.append((champion.canonical_id, child is not None and child.child_exited()))
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
        peers.on_this_machine,
        open_integration,
        replay_champion,
        profiler.profile_root,
    )
    return Prelude(context, services, warm_start=warm_start)


def _warm_start_decision(layout: SessionLayout) -> dict[str, Any]:
    logged = [json.loads(line) for line in layout.decisions.read_text().splitlines()]
    return next(entry["detail"] for entry in logged if entry["reason"] == "warm start read")


async def _run(prelude: Prelude, budget: SessionBudget, workload: Workload = WORKLOAD) -> PreludeResult:
    return await prelude.root(workload, budget, HardStop(budget, _clock()), IMAGE)


def _registry(
    layout: SessionLayout, session: Path | None, exit_code: int = 0, linger: float = 0.0
) -> MbFakeChildRegistry:
    child = "" if session is None else str(session)
    return MbFakeChildRegistry(layout.pids, _clock(), session=child, exit_code=exit_code, linger=linger)


def _optimize_options() -> set[str]:
    """Every option ``hyperloom optimize`` declares, read from the parser's source text without importing it."""

    source = (Path(hyperloom.__file__).parent / "inference_optimizer" / "cli" / "parser.py").read_text()
    start = source.index('sub.add_parser("optimize"')
    end = source.find("sub.add_parser(", start + 1)
    optimize = source[start : end if end != -1 else None]
    options: set[str] = set()
    for call in optimize.split("add_argument(")[1:]:
        head = call.split("help=")[0]
        names = re.findall(r'"(--[a-z0-9-]+)"', head)
        options.update(names)
        if "BooleanOptionalAction" in head:
            options.update(f"--no-{name[2:]}" for name in names)
    return options


def test_child_command_uses_only_options_the_main_flow_declares(tmp_path: Path) -> None:
    argv = prelude_argv(WORKLOAD, 1.5, tmp_path / "launch.json")
    assert argv[:4] == (sys.executable, "-m", "hyperloom", "optimize")
    assert (Path(hyperloom.__file__).parent / "__main__.py").is_file()
    assert {arg for arg in argv[4:] if arg.startswith("--")} <= _optimize_options()
    for flag in ("--no-framework-agent", "--no-kernel", "--no-enable-conc-sweep", "--no-warm-replay"):
        assert flag in argv
    for flag in ("--no-research-scout", "--no-static-recon", "--degraded-kb"):
        assert flag in argv
    assert argv[argv.index("--enablement") + 1] == "off"
    assert argv[argv.index("--max-hours") + 1] == "1.500000"
    assert argv[argv.index(LAUNCH_FLAG) + 1] == str(tmp_path / "launch.json")
    assert argv[argv.index("--gpu-type") + 1] == "mi355x"


def test_an_agentx_child_is_given_no_sequence_lengths(tmp_path: Path) -> None:
    inferencex = prelude_argv(WORKLOAD, 1.5, tmp_path / "launch.json")
    agentx = prelude_argv(AGENTX, 1.5, tmp_path / "launch.json")

    assert (inferencex[inferencex.index("--isl") + 1], inferencex[inferencex.index("--osl") + 1]) == ("1024", "1024")
    assert not {"--isl", "--osl"} & set(agentx)
    assert [arg for arg in inferencex if arg not in ("--isl", "--osl", "1024")] == list(agentx)


async def test_an_agentx_session_profile_has_no_sequence_lengths_and_no_throughput_ceiling(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    child = child_with_patch(tmp_path)
    with_agentx_baseline_run(child)
    processes, peers = _registry(layout, child), _peers(layout, rule=AgentXRule())

    result = await _run(_prelude(layout, processes, peers), SessionBudget.from_hours(2.0, NOW), AGENTX)

    profile = await _blackboard(layout).session_profile()
    assert profile == result.profile
    assert (profile.isl, profile.osl, dict(profile.roofline_ceiling)) == (None, None, {})
    assert not {"--isl", "--osl"} & set(processes.argv)


async def test_the_root_is_the_checkout_the_child_runs_on_committed_before_the_child_starts(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    child = child_with_patch(tmp_path)
    processes, peers = _registry(layout, child), _peers(layout)

    result = await _run(_prelude(layout, processes, peers), SessionBudget.from_hours(2.0, NOW))

    assert float(processes.argv[processes.argv.index("--max-hours") + 1]) == pytest.approx(2.0)
    assert result.root_commit == await peers.repo.head(INTEGRATION_BRANCH)
    recipe = await peers.repo.read_recipe(result.root_commit)
    [component] = recipe.components
    assert (component.upstream_ref, component.patches) == (head_of(tmp_path / "framework"), ())
    assert (recipe.launch, recipe.base_image_digest) == (LaunchConfig(), IMAGE)
    assert json.loads(layout.root_recipe.read_text()) == recipe.to_document()
    assert json.loads(layout.prelude_child.read_text())["session_dir"] == str(child)
    assert await processes.leftovers() == ()
    root = (await _blackboard(layout).lineage()).commit(result.root_commit)
    assert root is not None
    assert (root.branch, root.recipe_digest) == (INTEGRATION_BRANCH, recipe.digest())
    profile = await _blackboard(layout).session_profile()
    assert profile == result.profile
    assert (profile.session_id, profile.hardware) == (layout.session_id, "mi355x")
    assert dict(profile.roofline_ceiling) == {GRADED_OUTPUT: ROOFLINE_PEAK}
    assert (profile.isl, profile.osl) == (1024, 1024)
    assert profile.kernel_optimizer == "geak"
    assert (profile.model_type, profile.architectures) == ("", ())
    assert peers.opened == [(result.root_commit, False)]


async def test_the_root_profile_of_the_childs_trace_is_started_right_after_the_root_is_committed(
    tmp_path: Path,
) -> None:
    layout = _layout(tmp_path)
    child = child_with_patch(tmp_path)
    trace_dir = with_roofline_profile(child)
    processes, peers = _registry(layout, child), _peers(layout)

    result = await _run(_prelude(layout, processes, peers), SessionBudget.from_hours(2.0, NOW))

    [call] = peers.toolbox.root_profiles
    copy = layout.root_profile / ROOFLINE_RUN_ID
    assert (call.commit_id, call.source_id) == (result.root_commit, ROOFLINE_RUN_ID)
    assert (call.trace_dir, call.point_config) == (copy / "torch_trace", copy / "baseline_config.with_envs.yaml")
    assert call.trace_dir.is_relative_to(layout.toolbox) and call.point_config.read_text() == POINT_CONFIG
    assert (call.trace_dir / TRACE_FILE).read_bytes() == (trace_dir / TRACE_FILE).read_bytes()
    assert (call.trace_dir / "capture_traces" / "bs_8_rank0.json.gz").is_file()
    reasons = [entry["reason"] for entry in _logged(layout)]
    assert reasons.index("root committed") < reasons.index("root profile started")
    [started] = [entry for entry in _logged(layout) if entry["reason"] == "root profile started"]
    assert started["detail"] == {
        "source_id": ROOFLINE_RUN_ID,
        "trace_dir": str(trace_dir),
        "job_id": f"prelude:{ROOFLINE_RUN_ID}",
    }


async def test_a_refused_root_profile_is_recorded_and_prelude_goes_on(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    child = child_with_patch(tmp_path)
    with_roofline_profile(child)
    processes, peers = _registry(layout, child), _peers(layout)
    peers.toolbox.refusal = RequestRejected("no *.trace.json* under the trace dir", "profile_without_trace")

    result = await _run(_prelude(layout, processes, peers), SessionBudget.from_hours(2.0, NOW))

    assert (await _blackboard(layout).baseline()).root_commit == result.root_commit
    [refused] = [entry for entry in _logged(layout) if entry["reason"] == "root profile not started"]
    assert (refused["detail"]["code"], refused["detail"]["reason"]) == (
        "profile_without_trace",
        "no *.trace.json* under the trace dir",
    )
    assert peers.toolbox.root_profiles == []


async def test_a_child_without_a_roofline_profile_starts_no_root_profile(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    processes, peers = _registry(layout, child_with_patch(tmp_path)), _peers(layout)

    await _run(_prelude(layout, processes, peers), SessionBudget.from_hours(2.0, NOW))

    [skipped] = [entry for entry in _logged(layout) if entry["reason"] == "root profile not started"]
    assert skipped["detail"] == {"reason": "the child left no roofline profile"}
    assert peers.toolbox.root_profiles == [] and not layout.root_profile.exists()


async def test_a_root_profile_whose_point_config_is_gone_is_recorded_and_prelude_goes_on(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    child = child_with_patch(tmp_path)
    (with_roofline_profile(child).parents[1] / "baseline_config.with_envs.yaml").unlink()
    processes, peers = _registry(layout, child), _peers(layout)

    await _run(_prelude(layout, processes, peers), SessionBudget.from_hours(2.0, NOW))

    [skipped] = [entry for entry in _logged(layout) if entry["reason"] == "root profile not started"]
    assert skipped["detail"]["reason"].startswith("cannot copy the profile into the session")
    assert peers.toolbox.root_profiles == []


class MbFakeSlowChildRegistry(MbFakeChildRegistry):
    """The PRELUDE child stand-in whose run moves the clock in ``times`` on by ``seconds``; deleted together with
    :class:`MbFakeChildRegistry`."""

    def __init__(self, layout: SessionLayout, session: Path, times: list[float], seconds: float) -> None:
        super().__init__(layout.pids, _clock(times), session=str(session), exit_code=0, linger=0.0)
        self._times = times
        self._seconds = seconds

    async def spawn(
        self, argv: Sequence[str], *, branch: str, log: Path, env: Mapping[str, str] | None = None
    ) -> RegisteredProcess:
        spawned = await super().spawn(argv, branch=branch, log=log, env=env)
        if LAUNCH_FLAG in argv:
            self._times[0] += self._seconds
        return spawned


async def test_the_gpu_time_of_a_child_that_held_no_lease_is_bounded_by_its_wall_time_times_tp_and_recorded(
    tmp_path: Path,
) -> None:
    layout = _layout(tmp_path)
    times = [NOW]
    processes = MbFakeSlowChildRegistry(layout, child_with_patch(tmp_path), times, 1800.0)
    workload = replace(WORKLOAD, tp=4)
    peers = _peers(layout)
    peers.toolbox.lease_refusal = RequestRejected("no card would be left to measure on", "lease_leaves_no_measurement")

    budget = SessionBudget.from_hours(2.0, NOW)
    prelude = _prelude(layout, processes, peers, clock=_clock(times))
    result = await prelude.root(workload, budget, HardStop(budget, _clock(times)), IMAGE)

    assert result.child_gpu_seconds_upper_bound == 1800.0 * 4
    logged = [json.loads(line) for line in layout.decisions.read_text().splitlines()]
    [rooted] = [entry["detail"] for entry in logged if entry["reason"] == "root committed"]
    assert rooted["child_gpu_seconds_upper_bound"] == 1800.0 * 4


async def test_a_child_on_leased_cards_is_charged_once_by_its_lease_to_the_integration_branch(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    times = [NOW]
    processes = MbFakeSlowChildRegistry(layout, child_with_patch(tmp_path), times, 1800.0)
    peers = _peers(layout)
    peers.toolbox.clock = _clock(times).now

    budget = SessionBudget.from_hours(2.0, NOW)
    prelude = _prelude(layout, processes, peers, clock=_clock(times))
    result = await prelude.root(WORKLOAD, budget, HardStop(budget, _clock(times)), IMAGE)

    reader = LocalBlackboard(layout.blackboard, Writer(Role.SCHEDULER, "scheduler"), clock=_clock(times).now)
    [hold] = await reader.card_holds()
    assert (hold.start.kind, hold.start.holder, hold.start.branch_id) == (
        LeaseKind.PRELUDE_CHILD,
        "prelude",
        INTEGRATION_BRANCH,
    )
    [integration] = [row for row in await reader.progress() if row.branch == INTEGRATION_BRANCH]
    assert hold.closed and integration.held_card_seconds == 1800.0 * WORKLOAD.tp
    assert result.child_gpu_seconds_upper_bound == 0.0
    [rooted] = [entry["detail"] for entry in _logged(layout) if entry["reason"] == "root committed"]
    assert rooted["child_gpu_seconds_upper_bound"] == 0.0


async def test_the_recipe_kb_champions_for_the_profile_become_warm_start_leads_prepared_for_replay(
    tmp_path: Path,
) -> None:
    layout = _layout(tmp_path)
    kb = OneChampionKB()
    budget = SessionBudget.from_hours(2.0, NOW)
    peers = _peers(layout)

    prelude = _prelude(layout, _registry(layout, child_with_patch(tmp_path)), peers, recipe_kb=kb)
    result = await prelude.root(WORKLOAD, budget, HardStop(budget, _clock()), IMAGE)

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
    budget = SessionBudget.from_hours(2.0, NOW)
    peers = _peers(layout)
    tuning = RequestRejected("TunableOp tunes online", "online_tuning_unsupported")
    peers.toolbox.reject_if(lambda request: tuning if request.delta.config.server_args else None)

    prelude = _prelude(layout, _registry(layout, child_with_patch(tmp_path)), peers, recipe_kb=OneChampionKB())
    result = await prelude.root(WORKLOAD, budget, HardStop(budget, _clock()), IMAGE)

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
    budget = SessionBudget.from_hours(2.0, NOW)
    peers = _peers(layout)
    peers.toolbox.reject_if(lambda request: STACK_REFUSED if request.delta.components else None)
    monkeypatch.setattr(prelude_module, "replay_leads", _patched_replays(source_image))
    prelude = _prelude(layout, _registry(layout, child_with_patch(tmp_path)), peers, recipe_kb=OneChampionKB())
    [lead] = (await prelude.root(WORKLOAD, budget, HardStop(budget, _clock()), IMAGE)).warm_leads
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
    budget = SessionBudget.from_hours(2.0, NOW)
    peers = _peers(layout)

    async def unanswered(request: object) -> None:
        await asyncio.sleep(60)

    monkeypatch.setattr(peers.toolbox, "check_test", unanswered)
    monkeypatch.setattr(prelude_module, "PREFLIGHT_SECONDS", 0.01)
    prelude = _prelude(layout, _registry(layout, child_with_patch(tmp_path)), peers, recipe_kb=OneChampionKB())
    result = await prelude.root(WORKLOAD, budget, HardStop(budget, _clock()), IMAGE)

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
    budget = SessionBudget.from_hours(2.0, NOW)

    prelude = _prelude(layout, _registry(layout, child_with_patch(tmp_path)), _peers(layout), recipe_kb=recipe_kb)
    result = await prelude.root(WORKLOAD, budget, HardStop(budget, _clock()), IMAGE)

    assert result.warm_leads == () and result.start_commit == result.root_commit
    assert _warm_start_decision(layout) == {"leads": [], "unread": unread}


async def test_the_profile_names_the_model_type_and_architectures_of_the_models_config(tmp_path: Path) -> None:
    model = tmp_path / "models" / "Qwen3-8B"
    model.mkdir(parents=True)
    (model / "config.json").write_text(json.dumps({"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"]}))
    workload = replace(WORKLOAD, model=str(model))
    layout = _layout(tmp_path)
    peers = _peers(layout)
    budget = SessionBudget.from_hours(2.0, NOW)

    prelude = _prelude(layout, _registry(layout, child_with_patch(tmp_path)), peers)
    result = await prelude.root(workload, budget, HardStop(budget, _clock()), IMAGE)

    assert (result.profile.model_type, result.profile.architectures) == ("qwen3", ("Qwen3ForCausalLM",))
    assert await _blackboard(layout).session_profile() == result.profile


async def test_baseline_is_measured_through_the_toolbox(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    peers = _peers(layout)
    prelude = _prelude(layout, _registry(layout, child_with_patch(tmp_path)), peers)

    result = await _run(prelude, SessionBudget.from_hours(2.0, NOW))

    requests = peers.toolbox.requests
    assert len(requests) == BASELINE_TESTS
    baseline = await _blackboard(layout).baseline()
    assert baseline.root_commit == result.root_commit
    assert sorted(baseline.request_ids) == sorted(requests)
    assert baseline.values[GRADED_OUTPUT] == pytest.approx(1000.0)
    assert dict(baseline.threshold_pct) == {GRADED_OUTPUT: FLOOR_PCT}
    assert set(peers.toolbox.results) == set(requests)
    for request in requests.values():
        assert (request.branch_id, request.parent_commit) == (INTEGRATION_BRANCH, result.root_commit)
        assert (request.kind, request.purpose, request.delta) == (TestKind.TEST, Purpose.BASELINE, RecipeDelta())
        assert (request.accuracy, request.intent_fingerprint, request.priority) == (True, None, None)


async def test_measuring_the_baseline_reports_its_longest_root_test(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    peers = _peers(layout)
    prelude = _prelude(layout, _registry(layout, child_with_patch(tmp_path)), peers)

    longest = (await _run(prelude, SessionBudget.from_hours(2.0, NOW))).longest_root_test

    assert longest == FakeToolboxConfig().attempt_seconds
    logged = [json.loads(line) for line in layout.decisions.read_text().splitlines()]
    [established] = [entry for entry in logged if entry["reason"] == "baseline established"]
    assert established["detail"]["longest_test_seconds"] == longest


async def test_child_failing_before_its_launch_info_fails_prelude(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    peers = _peers(layout)
    with pytest.raises(PreludeFailed, match="exited with 3"):
        await _run(_prelude(layout, _registry(layout, None, exit_code=3), peers), SessionBudget.from_hours(2.0, NOW))
    assert json.loads(layout.prelude_child.read_text())["session_dir"] is None
    assert "root committed" not in [entry["reason"] for entry in _logged(layout)]


async def test_a_child_whose_baseline_failed_fails_prelude_with_its_reason(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    child = failed_child(tmp_path)
    peers = _peers(layout)

    with pytest.raises(PreludeFailed) as failed:
        await _run(_prelude(layout, _registry(layout, child, exit_code=1), peers), SessionBudget.from_hours(2.0, NOW))

    assert str(failed.value) == (
        f"cannot root the session on the baseline prelude in {child}: the child ended with "
        f"stop_reason='baseline_failed' and baseline_tput=0.0; its last baseline attempt failed with "
        f"{BASELINE_ERROR_CLASS!r}"
    )
    reasons = [entry["reason"] for entry in _logged(layout)]
    assert "child started" in reasons and "root committed" not in reasons
    assert peers.toolbox.root_profiles == []


async def test_baseline_tests_that_fail_to_measure_fail_prelude(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    peers = _peers(layout)
    peers.toolbox.script(lambda request: request.purpose is Purpose.BASELINE, [Fate.MEASUREMENT])
    prelude = _prelude(layout, _registry(layout, child_with_patch(tmp_path)), peers)
    with pytest.raises(PreludeFailed, match="the baseline anchors nothing: 0 of 3 root tests measured"):
        await _run(prelude, SessionBudget.from_hours(2.0, NOW))


async def test_a_baseline_below_the_accuracy_floor_fails_prelude(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    peers = _peers(layout, accuracy=0.0)
    prelude = _prelude(layout, _registry(layout, child_with_patch(tmp_path)), peers)
    with pytest.raises(PreludeFailed, match="the baseline anchors nothing: baseline accuracy"):
        await _run(prelude, SessionBudget.from_hours(2.0, NOW))
    assert len(peers.toolbox.results) == BASELINE_TESTS


async def test_a_baseline_measuring_no_throughput_fails_prelude(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    measure_no_throughput(monkeypatch)
    layout = _layout(tmp_path)
    peers = _peers(layout)
    prelude = _prelude(layout, _registry(layout, child_with_patch(tmp_path)), peers)
    with pytest.raises(PreludeFailed, match="the baseline anchors nothing: .*: a throughput must be positive"):
        await _run(prelude, SessionBudget.from_hours(2.0, NOW))


async def test_refused_baseline_test_fails_prelude_and_takes_the_accepted_ones_back(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    peers = _peers(layout)
    peers.toolbox.executor.gate.clear()
    accepted = iter(range(BASELINE_TESTS - 1))
    refusal = RequestRejected("the base image moved", code="base_image_mismatch")
    peers.toolbox.reject_if(lambda request: None if next(accepted, None) is not None else refusal)
    prelude = _prelude(layout, _registry(layout, child_with_patch(tmp_path)), peers)

    with pytest.raises(PreludeFailed, match="refused the baseline test: base_image_mismatch"):
        await _run(prelude, SessionBudget.from_hours(2.0, NOW))

    assert len(peers.toolbox.requests) == BASELINE_TESTS - 1
    assert set(peers.toolbox.withdrawn) == set(peers.toolbox.requests)
    assert peers.toolbox.results == {}


async def test_exhausted_budget_starts_no_child(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    processes = _registry(layout, None)
    with pytest.raises(PreludeFailed, match="at least one minute"):
        await _run(_prelude(layout, processes, _peers(layout)), SessionBudget.from_hours(2.0, NOW - 2 * 3600))
    assert processes.argv == ()


async def test_cancelled_prelude_leaves_no_process(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    processes = _registry(layout, None, linger=600.0)
    prelude = _prelude(layout, processes, _peers(layout))
    task = asyncio.create_task(_run(prelude, SessionBudget.from_hours(2.0, NOW)))
    for _ in range(200):
        if layout.pids.exists():
            break
        await asyncio.sleep(0.05)
    await asyncio.sleep(0.5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert await processes.leftovers() == ()


async def test_a_prelude_cancelled_near_the_hard_stop_leaves_close_its_share(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    times = [NOW]
    clock = _clock(times)
    budget = SessionBudget.from_hours(2.0, NOW)
    processes = StubbornRegistry(layout.pids, clock)
    task = asyncio.create_task(_run(_prelude(layout, processes, _peers(layout), clock=clock), budget))
    while not layout.pids.exists():
        await asyncio.sleep(0.05)
    times[0] = budget.hard_stop_at - NEAR_HARD_STOP_SECONDS
    started = time.monotonic()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert time.monotonic() - started < budget.close_grace_seconds / 2
    assert await processes.leftovers() == ()


def _submitted_after_the_child(peers: PreludePeers, processes: MbFakeChildRegistry) -> list[bool]:
    """Whether the child had exited as each root test was submitted, filled in as they are."""

    exited: list[bool] = []

    def note(request: TestRequest) -> None:
        if request.purpose is Purpose.BASELINE:
            exited.append(processes.child_exited())

    peers.toolbox.reject_if(note)
    return exited


async def test_the_root_tests_run_while_the_child_runs_on_the_cards_leased_for_it(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    processes, peers = _registry(layout, child_with_patch(tmp_path), linger=SHORT_CHILD), _peers(layout)
    exited = _submitted_after_the_child(peers, processes)

    result = await _run(_prelude(layout, processes, peers), SessionBudget.from_hours(2.0, NOW))

    assert exited == [False] * BASELINE_TESTS
    assert (await _blackboard(layout).baseline()).root_commit == result.root_commit
    assert peers.toolbox.leased == [("prelude", WORKLOAD.tp)] and peers.toolbox.released == ["lease-1"]
    assert processes.env is not None and processes.env[CHILD_CARDS_ENV] == "6"
    assert processes.env[TRACE_ANALYSIS_ROUTE_ENV] == CHILD_TRACE_ANALYSIS_ROUTE
    assert processes.env.get("HIP_VISIBLE_DEVICES") == os.environ.get("HIP_VISIBLE_DEVICES")


async def test_a_child_whose_lease_would_leave_no_test_its_cards_runs_before_the_root_tests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(TRACE_ANALYSIS_ROUTE_ENV, "report")
    layout = _layout(tmp_path)
    processes, peers = _registry(layout, child_with_patch(tmp_path), linger=SHORT_CHILD), _peers(layout)
    peers.toolbox.lease_refusal = RequestRejected("one test needs both cards", "lease_leaves_no_measurement")
    exited = _submitted_after_the_child(peers, processes)

    await _run(_prelude(layout, processes, peers, recipe_kb=OneChampionKB()), SessionBudget.from_hours(2.0, NOW))

    assert exited == [True] * BASELINE_TESTS and peers.replays == [("inference:champion", True)]
    assert processes.env is not None and processes.env[TRACE_ANALYSIS_ROUTE_ENV] == CHILD_TRACE_ANALYSIS_ROUTE
    assert processes.env.get(CHILD_CARDS_ENV) == os.environ.get(CHILD_CARDS_ENV) and peers.toolbox.released == []
    [first] = [entry for entry in _logged(layout) if entry["reason"] == CHILD_FIRST]
    assert first["detail"] == {"refusal": "one test needs both cards"}


async def test_a_failing_root_test_ends_prelude_without_waiting_for_the_child(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    processes, peers = _registry(layout, child_with_patch(tmp_path), linger=LONG_CHILD), _peers(layout)
    refusal = RequestRejected("the base image moved", code="base_image_mismatch")
    peers.toolbox.reject_if(lambda request: refusal if request.purpose is Purpose.BASELINE else None)

    with pytest.raises(PreludeFailed, match="refused the baseline test"):
        await asyncio.wait_for(_run(_prelude(layout, processes, peers), SessionBudget.from_hours(2.0, NOW)), 60)

    assert await processes.leftovers() == () and peers.toolbox.released == ["lease-1"]
    assert "root committed" not in [entry["reason"] for entry in _logged(layout)]


async def test_a_child_that_ran_on_another_checkout_than_the_root_fails_prelude(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout = _layout(tmp_path)
    processes, peers = _registry(layout, child_with_patch(tmp_path)), _peers(layout)
    monkeypatch.setenv(
        MB_FRAMEWORK_REPO_ENV, str(framework_checkout(tmp_path / "elsewhere", "https://example.invalid/fork"))
    )

    with pytest.raises(PreludeFailed, match="that the session committed before it started"):
        await _run(_prelude(layout, processes, peers), SessionBudget.from_hours(2.0, NOW))

    assert "root committed" not in [entry["reason"] for entry in _logged(layout)]


async def test_the_champion_is_replayed_while_the_child_runs_on_the_profile_known_before_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(MB_FRAMEWORK_VERSION_ENV, "0.5.2")
    layout = _layout(tmp_path)
    kb = OneChampionKB()
    processes, peers = _registry(layout, child_with_patch(tmp_path), linger=SHORT_CHILD), _peers(layout)

    result = await _run(_prelude(layout, processes, peers, recipe_kb=kb), SessionBudget.from_hours(2.0, NOW))

    assert peers.replays == [("inference:champion", False)]
    assert kb.searched and result.profile.framework_version == "0.5.2"
    assert (await _blackboard(layout).session_profile()).framework_version == "0.5.2"
    reasons = [entry["reason"] for entry in _logged(layout)]
    assert reasons.index("warm start read") < reasons.index("root committed")


async def test_an_agentx_champion_is_replayed_once_the_childs_run_anchored_the_baseline(tmp_path: Path) -> None:
    child = child_with_patch(tmp_path)
    with_agentx_baseline_run(child)
    layout, peers, _ = _agentx_prelude(tmp_path, child)

    await _run(
        _prelude(layout, _registry(layout, child), peers, recipe_kb=OneChampionKB()),
        SessionBudget.from_hours(2.0, NOW),
        AGENTX,
    )

    assert [exited for _, exited in peers.replays] == [True]
    reasons = [entry["reason"] for entry in _logged(layout)]
    assert reasons.index("baseline established") < reasons.index("warm start read")


def _logged(layout: SessionLayout) -> list[dict[str, Any]]:
    return [json.loads(line) for line in layout.decisions.read_text().splitlines()]


async def _rooted_session(tmp_path: Path, child: Path) -> SessionLayout:
    """A session in ``tmp_path`` whose PRELUDE rooted on ``child``, with the manifest naming its workload."""

    layout = create_session(tmp_path, WORKLOAD.model_basename, _clock())
    budget = SessionBudget.from_hours(2.0, NOW)
    await _prelude(layout, _registry(layout, child), _peers(layout)).root(
        WORKLOAD, budget, HardStop(budget, _clock()), IMAGE
    )
    atomic_write_json(layout.manifest, {"settings": {"workload": encode(WORKLOAD)}})
    return layout


async def test_a_reused_prelude_whose_child_now_gives_another_root_fails_and_commits_nothing(tmp_path: Path) -> None:
    child = child_with_patch(tmp_path)
    earlier = await _rooted_session(tmp_path / "earlier", child)
    reused = reused_prelude(earlier.root, WORKLOAD, IMAGE)
    moved = ("git", "-C", str(tmp_path / "framework"), "remote", "set-url", "origin", "https://example.invalid/moved")
    subprocess.run(moved, check=True)
    layout = create_session(tmp_path / "later", WORKLOAD.model_basename, _clock())

    with pytest.raises(PreludeFailed) as failed:
        await _prelude(layout, _registry(layout, None), _peers(layout)).reuse(reused, WORKLOAD, IMAGE)

    assert reused.root_recipe_digest in str(failed.value)
    assert re.search(r"root recipe \S+, not", str(failed.value))
    assert not layout.root_recipe.exists() and not layout.repo.exists()
    assert [entry["reason"] for entry in _logged(layout)] == ["child reused"]


async def test_a_reused_prelude_skips_decision_rows_of_a_kind_it_does_not_know(tmp_path: Path) -> None:
    child = child_with_patch(tmp_path)
    earlier = await _rooted_session(tmp_path / "earlier", child)
    rows = earlier.decisions.read_text().splitlines()
    unknown = json.dumps({"kind": "from-a-later-version", "reason": "root committed", "detail": {}})
    earlier.decisions.write_text("\n".join([unknown, *rows]) + "\n")

    reused = reused_prelude(earlier.root, WORKLOAD, IMAGE)

    assert reused.child_session == str(child)


async def test_a_prelude_of_a_workload_differing_only_in_its_kernel_optimizer_is_reused(tmp_path: Path) -> None:
    child = child_with_patch(tmp_path)
    earlier = await _rooted_session(tmp_path / "earlier", child)
    forge = replace(WORKLOAD, kernel_optimizer=KernelOptimizer.FORGE)

    reused = reused_prelude(earlier.root, forge, IMAGE)

    root = Recipe.from_document(json.loads(earlier.root_recipe.read_text()))
    assert reused == ReusedPrelude(str(earlier.root), str(child), root.digest())


async def test_a_reused_prelude_starts_the_root_profile_from_the_earlier_childs_trace(tmp_path: Path) -> None:
    child = child_with_patch(tmp_path)
    with_roofline_profile(child)
    earlier = await _rooted_session(tmp_path / "earlier", child)
    layout = create_session(tmp_path / "later", WORKLOAD.model_basename, _clock())
    peers = _peers(layout)

    result = await _prelude(layout, _registry(layout, None), peers).reuse(
        reused_prelude(earlier.root, WORKLOAD, IMAGE), WORKLOAD, IMAGE
    )

    [call] = peers.toolbox.root_profiles
    assert (call.commit_id, call.source_id) == (result.root_commit, ROOFLINE_RUN_ID)
    assert call.trace_dir == layout.root_profile / ROOFLINE_RUN_ID / "torch_trace"
    assert (call.trace_dir / TRACE_FILE).is_file()
    reasons = [entry["reason"] for entry in _logged(layout)]
    assert reasons[:3] == ["child reused", "root committed", "root profile started"]


def _agentx_prelude(tmp_path: Path, child: Path) -> tuple[SessionLayout, PreludePeers, Prelude]:
    """PRELUDE of an AgentX session on ``child``, on a toolbox that leases cards."""

    layout = _layout(tmp_path)
    peers = _peers(layout, rule=AgentXRule())
    return layout, peers, _prelude(layout, _registry(layout, child), peers)


async def test_an_agentx_baseline_is_the_childs_own_run_and_no_root_test_is_submitted(tmp_path: Path) -> None:
    child = child_with_patch(tmp_path)
    run = with_agentx_baseline_run(child)
    layout, peers, prelude = _agentx_prelude(tmp_path, child)

    result = await _run(prelude, SessionBudget.from_hours(2.0, NOW), AGENTX)

    reserve = result.longest_root_test
    assert peers.toolbox.requests == {} and peers.toolbox.leased == []
    assert reserve == AGENTX_RUN_SECONDS
    copy = layout.prelude / "baseline" / AGENTX_RUN_ID
    assert peers.toolbox.prelude_baselines == [copy]
    report = copy / AGENTX_BENCHMARK / "benchmark_report.json"
    assert report.read_bytes() == (run / AGENTX_BENCHMARK / "benchmark_report.json").read_bytes()
    baseline = await _blackboard(layout).baseline()
    assert (baseline.root_commit, baseline.request_ids, baseline.wall_seconds) == (result.root_commit, (), reserve)
    assert {key: baseline.values[key] for key in AGENTX_RUN_PERF} == pytest.approx(AGENTX_RUN_PERF)
    assert baseline.source.startswith(f"prelude-child:{run} sha256:")
    [established] = [entry for entry in _logged(layout) if entry["reason"] == "baseline established"]
    assert established["detail"] == {"request_ids": [], "source": baseline.source, "longest_test_seconds": reserve}


@pytest.mark.parametrize(
    ("anchor", "state", "refusal"),
    [
        ({"decision": "discarded"}, {}, "promoted no baseline that succeeded"),
        ({"extras": {"baseline_tput": 1.0}}, {}, "not the anchor's baseline_tput"),
        ({}, {"baseline_measure_round_dropped": True}, "cold start"),
        ({}, {"baseline_perf": {**AGENTX_RUN_PERF, "e2e_norm_intvty_p50": 120.0}}, "the child recorded 120.0"),
    ],
)
async def test_an_agentx_baseline_the_childs_run_cannot_stand_for_fails_prelude(
    tmp_path: Path, anchor: dict[str, Any], state: dict[str, Any], refusal: str
) -> None:
    child = child_with_patch(tmp_path)
    with_agentx_baseline_run(child, **anchor)
    document = json.loads((child / "state.json").read_text())
    write_child_session(child, {**document, **state})
    layout, peers, prelude = _agentx_prelude(tmp_path, child)

    with pytest.raises(PreludeFailed, match=refusal):
        await _run(prelude, SessionBudget.from_hours(2.0, NOW), AGENTX)

    assert peers.toolbox.requests == {}
    assert not [entry for entry in _logged(layout) if entry["reason"] == "baseline established"]


async def test_an_agentx_baseline_whose_run_is_gone_fails_prelude(tmp_path: Path) -> None:
    child = child_with_patch(tmp_path)
    run = with_agentx_baseline_run(child)
    _, _, prelude = _agentx_prelude(tmp_path, child)
    shutil.rmtree(run)

    with pytest.raises(PreludeFailed, match="cannot copy the child's baseline run"):
        await _run(prelude, SessionBudget.from_hours(2.0, NOW), AGENTX)


async def test_a_reused_prelude_anchors_an_agentx_baseline_on_the_earlier_childs_run(tmp_path: Path) -> None:
    child = child_with_patch(tmp_path)
    run = with_agentx_baseline_run(child)
    earlier = await _rooted_session(tmp_path / "earlier", child)
    layout = create_session(tmp_path / "later", WORKLOAD.model_basename, _clock())
    peers = _peers(layout, rule=AgentXRule())
    prelude = _prelude(layout, _registry(layout, None), peers)
    result = await prelude.reuse(reused_prelude(earlier.root, WORKLOAD, IMAGE), WORKLOAD, IMAGE)

    assert result.longest_root_test == AGENTX_RUN_SECONDS

    assert peers.toolbox.requests == {}
    assert (await _blackboard(layout).baseline()).source.startswith(f"prelude-child:{run} ")


#: The op shares :class:`AnalysingToolbox`'s TraceLens job finds in every profile.
ROOT_SHARES = {"gemm": 48.0, "attention": 31.0}


class AnalysingToolbox(LeasingToolbox):
    """The leasing toolbox with a TraceLens route whose job ends at once, leaving :data:`ROOT_SHARES` on the commit the
    profile test measured, or refused with ``analysis_refusal``; ``analysed`` holds each profile test analysed. A
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


def _hosted(tmp_path: Path, rule: Rule | None = None) -> tuple[SessionLayout, PreludePeers, MbFakeChildRegistry]:
    """A session whose toolbox measures in sandboxes elsewhere, with a child that would root it were one started."""

    layout = _layout(tmp_path)
    peers = _peers(layout, rule=rule, toolbox_type=AnalysingToolbox)
    peers.on_this_machine = False
    return layout, peers, _registry(layout, child_with_patch(tmp_path))


def _by_purpose(toolbox: LeasingToolbox, purpose: Purpose) -> list[str]:
    return [request_id for request_id, request in toolbox.requests.items() if request.purpose is purpose]


async def test_a_prelude_measuring_in_sandboxes_starts_no_child_and_roots_on_root_tests_and_a_root_profile(
    tmp_path: Path,
) -> None:
    layout, peers, processes = _hosted(tmp_path)

    result = await _run(_prelude(layout, processes, peers), SessionBudget.from_hours(2.0, NOW))

    assert processes.child is None and not layout.prelude_child.exists() and peers.toolbox.leased == []
    root = result.root_commit
    assert root == await peers.repo.head(INTEGRATION_BRANCH) and peers.opened == [(root, False)]
    baseline_tests, [profile] = (
        _by_purpose(peers.toolbox, Purpose.BASELINE),
        _by_purpose(peers.toolbox, Purpose.PROFILE),
    )
    assert len(baseline_tests) == BASELINE_TESTS and len(peers.toolbox.requests) == BASELINE_TESTS + 1
    request = peers.toolbox.requests[profile]
    assert (request.kind, request.branch_id, request.parent_commit) == (TestKind.PROFILE, INTEGRATION_BRANCH, root)
    assert peers.toolbox.analysed == [profile] and peers.toolbox.root_profiles == []
    baseline = await _blackboard(layout).baseline()
    assert (baseline.root_commit, sorted(baseline.request_ids)) == (root, sorted(baseline_tests))
    assert result.longest_root_test == FakeToolboxConfig().attempt_seconds
    session_profile = await _blackboard(layout).session_profile()
    assert session_profile == result.profile and dict(session_profile.roofline_ceiling) == {}
    shares = await _blackboard(layout).op_shares(root)
    assert shares is not None and (dict(shares.shares), shares.profile_request_id) == (ROOT_SHARES, profile)
    assert result.child_gpu_seconds_upper_bound == 0.0
    [rooted] = [entry["detail"] for entry in _logged(layout) if entry["reason"] == "root committed"]
    assert rooted == {"root_commit": root}
    [profiled] = [entry for entry in _logged(layout) if entry["kind"] == "profile"]
    assert (profiled["reason"], profiled["detail"]["commit"], profiled["detail"]["request_id"]) == (
        PROFILED,
        root,
        profile,
    )


async def test_an_agentx_prelude_measuring_in_sandboxes_anchors_its_baseline_on_one_root_test(tmp_path: Path) -> None:
    layout, peers, processes = _hosted(tmp_path, AgentXRule())

    result = await _run(_prelude(layout, processes, peers), SessionBudget.from_hours(2.0, NOW), AGENTX)

    assert processes.child is None and peers.toolbox.prelude_baselines == []
    [root_test], [profile] = _by_purpose(peers.toolbox, Purpose.BASELINE), _by_purpose(peers.toolbox, Purpose.PROFILE)
    baseline = await _blackboard(layout).baseline()
    assert (baseline.root_commit, baseline.request_ids, baseline.source) == (result.root_commit, (root_test,), "")
    assert result.longest_root_test == FakeToolboxConfig().attempt_seconds
    assert peers.toolbox.analysed == [profile]
    [established] = [entry for entry in _logged(layout) if entry["reason"] == "baseline established"]
    assert established["detail"]["request_ids"] == [root_test]


async def test_a_prelude_whose_backend_cannot_analyse_the_root_profile_records_it_and_goes_on(tmp_path: Path) -> None:
    layout, peers, processes = _hosted(tmp_path)
    assert isinstance(peers.toolbox, AnalysingToolbox)
    peers.toolbox.analysis_refusal = RequestRejected("no TraceLens on this backend", CODE_TOOL_UNAVAILABLE)

    result = await _run(_prelude(layout, processes, peers), SessionBudget.from_hours(2.0, NOW))

    assert (await _blackboard(layout).baseline()).root_commit == result.root_commit
    assert await _blackboard(layout).op_shares(result.root_commit) is None
    [unavailable] = [entry for entry in _logged(layout) if entry["kind"] == "profile"]
    assert (unavailable["reason"], unavailable["detail"]["code"]) == (PROFILE_UNAVAILABLE, CODE_TOOL_UNAVAILABLE)


async def test_root_tests_that_fail_to_measure_fail_a_prelude_measuring_in_sandboxes(
    tmp_path: Path,
) -> None:
    layout, peers, processes = _hosted(tmp_path)
    peers.toolbox.script(lambda request: request.purpose is Purpose.BASELINE, [Fate.MEASUREMENT])

    with pytest.raises(PreludeFailed, match="the baseline anchors nothing"):
        await _run(_prelude(layout, processes, peers), SessionBudget.from_hours(2.0, NOW))

    assert processes.child is None
    assert not [entry for entry in _logged(layout) if entry["reason"] == "baseline established"]


async def test_the_prelude_of_a_session_that_started_no_child_cannot_be_reused(tmp_path: Path) -> None:
    layout, peers, processes = _hosted(tmp_path)
    budget = SessionBudget.from_hours(2.0, NOW)
    await _prelude(layout, processes, peers).root(WORKLOAD, budget, HardStop(budget, _clock()), IMAGE)
    atomic_write_json(layout.manifest, {"settings": {"workload": encode(WORKLOAD)}})

    with pytest.raises(ValueError, match="rooted on no child session"):
        reused_prelude(layout.root, WORKLOAD, IMAGE)
