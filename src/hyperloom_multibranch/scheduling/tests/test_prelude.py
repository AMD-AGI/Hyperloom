# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""PRELUDE's contract: the child's command line, what a finished child roots the session on, and how it fails."""

from __future__ import annotations

import asyncio
import json
import re
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pytest

import hyperloom
from hyperloom.common.perf_metric import GRADED_OUTPUT
from hyperloom.contracts.measurement import Purpose, TestKind
from hyperloom.contracts.recipe import RecipeDelta
from hyperloom.measure.compile import LocalBlobStore
from hyperloom_multibranch.contracts.notebook import INTEGRATION_BRANCH, Role, Writer
from hyperloom_multibranch.contracts.toolbox import RequestRejected
from hyperloom_multibranch.notebook import LocalNotebook
from hyperloom_multibranch.notebook.knowledge.warm_start import RecipeDownloader
from hyperloom_multibranch.repo.git_repo import GitExperimentRepo
from hyperloom_multibranch.scheduling.budget import HardStop, SessionBudget
from hyperloom_multibranch.scheduling.clock import Clock
from hyperloom_multibranch.scheduling.context import SchedulerContext
from hyperloom_multibranch.scheduling.roster import Roster, RosterCaps
from hyperloom_multibranch.scheduling.decisions import DecisionLog
from hyperloom_multibranch.scheduling.forks import ForkDesk
from hyperloom_multibranch.scheduling.layout import SessionLayout, create_session
from hyperloom_multibranch.rulegate.service import BASELINE_TESTS, GateService
from hyperloom_multibranch.scheduling.prelude import (
    Prelude,
    PreludeFailed,
    PreludeResult,
    WarmStartSource,
    prelude_argv,
)
from hyperloom_multibranch.scheduling.processes import ProcessRegistry, RegisteredProcess
from hyperloom_multibranch.scheduling.requests import SessionTests
from hyperloom_multibranch.scheduling.settings import InferenceXShape, KernelOptimizer, Workload
from hyperloom_multibranch.scheduling.tests.peer_fakes import (
    OneChampionKB,
    LAUNCH_FLAG,
    MbFakeChildRegistry,
    RecordedReservations,
    RecordingToolbox,
    measure_no_throughput,
    session_gate,
)
from hyperloom_multibranch.toolbox.fake_model import Fate, FakeToolboxConfig
from hyperloom_multibranch.scheduling.tests.child_sessions import (
    PATCH_TEXT,
    ROOFLINE_PEAK,
    child_with_patch,
    write_child_session,
)

NOW = 1_000_000.0
#: The image the fake toolbox measures on, which the root recipe must carry.
IMAGE = FakeToolboxConfig().image_digest
STUBBORN_CHILD = ("sh", "-c", "trap '' TERM; sleep 60")
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


class StubbornRegistry(ProcessRegistry):
    """Runs a main-flow child that ignores SIGTERM; a failure double, not a placeholder."""

    async def spawn(self, argv: Sequence[str], *, branch: str, log: Path) -> RegisteredProcess:
        return await super().spawn(STUBBORN_CHILD, branch=branch, log=log)


@dataclass
class PreludePeers:
    """The repository, toolbox and rule gate PRELUDE works with, on the session's notebook; ``tests`` waits for the
    toolbox's outcomes without real sleeps."""

    repo: GitExperimentRepo
    toolbox: RecordingToolbox
    tests: SessionTests
    gate: GateService


async def _instant(seconds: float) -> None:
    await asyncio.sleep(0)


def _peers(layout: SessionLayout, accuracy: float | None = None) -> PreludePeers:
    """``accuracy`` is the eval score every baseline test measures, the fake's own when ``None``."""

    scores = {} if accuracy is None else {RecipeDelta().fingerprint(): accuracy}
    toolbox_notebook = LocalNotebook(layout.notebook, Writer(Role.TOOLBOX, "toolbox"), clock=_clock().now)
    toolbox = RecordingToolbox(toolbox_notebook, accuracy=scores, conc=WORKLOAD.conc)
    tests = SessionTests(toolbox, Clock(now=_clock().now, sleep=_instant))
    gate = session_gate(LocalNotebook(layout.notebook, Writer(Role.GATE, "gate"), clock=_clock().now), toolbox)
    return PreludePeers(GitExperimentRepo(layout.repo, LocalBlobStore(layout.blobs)), toolbox, tests, gate)


def _clock(times: list[float] | None = None) -> Clock:
    current = [NOW] if times is None else times
    return Clock(now=lambda: current[0])


def _layout(tmp_path: Path) -> SessionLayout:
    return create_session(tmp_path / "workspace", WORKLOAD.model_basename, _clock())


def _notebook(layout: SessionLayout) -> LocalNotebook:
    return LocalNotebook(layout.notebook, Writer(Role.SCHEDULER, "scheduler"), clock=_clock().now)


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
        layout, roster, _notebook(layout), decisions, processes, clock, desk, RecordedReservations()
    )
    warm_start = None if recipe_kb is None else WarmStartSource(recipe_kb, LocalBlobStore(layout.blobs))
    return Prelude(context, peers.repo, peers.gate, peers.tests, warm_start=warm_start)


def _warm_start_decision(layout: SessionLayout) -> dict[str, Any]:
    logged = [json.loads(line) for line in layout.decisions.read_text().splitlines()]
    return next(entry["detail"] for entry in logged if entry["reason"] == "warm start read")


async def _run(prelude: Prelude, budget: SessionBudget) -> PreludeResult:
    """Both PRELUDE steps, as the session runs them once the integration branch has its K."""

    result = await prelude.root(WORKLOAD, budget, HardStop(budget, _clock()), IMAGE)
    await prelude.measure_baseline(result.root_commit)
    return result


def _registry(
    layout: SessionLayout, session: Path | None, exit_code: int = 0, linger: float = 0.0
) -> MbFakeChildRegistry:
    child = "" if session is None else str(session)
    return MbFakeChildRegistry(layout.pids, _clock(), session=child, exit_code=exit_code, linger=linger)


def _optimize_options() -> set[str]:
    """Every option ``hyperloom optimize`` declares, read from the parser's source text without importing it."""

    source = (Path(hyperloom.__file__).parent / "inference_optimizer" / "cli" / "parser.py").read_text()
    start = source.index('sub.add_parser("optimize"')
    optimize = source[start : source.index("sub.add_parser(", start + 1)]
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
    for flag in ("--no-research-scout", "--degraded-kb"):
        assert flag in argv
    assert argv[argv.index("--enablement") + 1] == "launch"
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
    processes, peers = _registry(layout, child_with_patch(tmp_path)), _peers(layout)

    budget = SessionBudget.from_hours(2.0, NOW)
    result = await _prelude(layout, processes, peers).root(AGENTX, budget, HardStop(budget, _clock()), IMAGE)

    profile = await _notebook(layout).session_profile()
    assert profile == result.profile
    assert (profile.isl, profile.osl, dict(profile.roofline_ceiling)) == (None, None, {})
    assert not {"--isl", "--osl"} & set(processes.argv)


async def test_finished_child_roots_the_session_before_any_baseline_test(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    child = child_with_patch(tmp_path)
    processes, peers = _registry(layout, child), _peers(layout)

    budget = SessionBudget.from_hours(2.0, NOW)
    result = await _prelude(layout, processes, peers).root(WORKLOAD, budget, HardStop(budget, _clock()), IMAGE)

    assert float(processes.argv[processes.argv.index("--max-hours") + 1]) == pytest.approx(2.0)
    assert result.root_commit == await peers.repo.head(INTEGRATION_BRANCH)
    recipe = await peers.repo.read_recipe(result.root_commit)
    [component] = recipe.components
    texts = [await peers.repo.patch_text(result.root_commit, patch.path) for patch in component.patches]
    assert texts == [PATCH_TEXT] and recipe.base_image_digest == IMAGE
    assert json.loads(layout.root_recipe.read_text()) == recipe.to_document()
    assert json.loads(layout.prelude_child.read_text())["session_dir"] == str(child)
    assert await processes.leftovers() == ()
    root = (await _notebook(layout).lineage()).commit(result.root_commit)
    assert root is not None
    assert (root.branch, root.recipe_digest) == (INTEGRATION_BRANCH, recipe.digest())
    profile = await _notebook(layout).session_profile()
    assert profile == result.profile
    assert (profile.session_id, profile.hardware, profile.framework_version) == (layout.session_id, "mi355x", "0.4.1")
    assert dict(profile.roofline_ceiling) == {GRADED_OUTPUT: ROOFLINE_PEAK}
    assert (profile.isl, profile.osl) == (1024, 1024)
    assert profile.kernel_optimizer == "geak"
    assert (profile.model_type, profile.architectures) == ("", ())
    assert peers.toolbox.requests == {}


class MbFakeSlowChildRegistry(MbFakeChildRegistry):
    """The PRELUDE child stand-in whose run moves the clock in ``times`` on by ``seconds``; deleted together with
    :class:`MbFakeChildRegistry`."""

    def __init__(self, layout: SessionLayout, session: Path, times: list[float], seconds: float) -> None:
        super().__init__(layout.pids, _clock(times), session=str(session), exit_code=0, linger=0.0)
        self._times = times
        self._seconds = seconds

    async def spawn(self, argv: Sequence[str], *, branch: str, log: Path) -> RegisteredProcess:
        spawned = await super().spawn(argv, branch=branch, log=log)
        if LAUNCH_FLAG in argv:
            self._times[0] += self._seconds
        return spawned


async def test_the_child_gpu_time_is_bounded_by_its_wall_time_times_tp_and_recorded(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    times = [NOW]
    processes = MbFakeSlowChildRegistry(layout, child_with_patch(tmp_path), times, 1800.0)
    workload = replace(WORKLOAD, tp=4)

    budget = SessionBudget.from_hours(2.0, NOW)
    prelude = _prelude(layout, processes, _peers(layout), clock=_clock(times))
    result = await prelude.root(workload, budget, HardStop(budget, _clock(times)), IMAGE)

    assert result.child_gpu_seconds_upper_bound == 1800.0 * 4
    logged = [json.loads(line) for line in layout.decisions.read_text().splitlines()]
    [rooted] = [entry["detail"] for entry in logged if entry["reason"] == "root committed"]
    assert rooted["child_gpu_seconds_upper_bound"] == 1800.0 * 4


async def test_a_patch_off_the_framework_checkout_is_kept_in_the_prelude_dir_and_marks_the_root(
    tmp_path: Path,
) -> None:
    layout = _layout(tmp_path)
    child = child_with_patch(tmp_path)
    site_packages = tmp_path / "site-packages"
    site_packages.mkdir()
    state = json.loads((child / "state.json").read_text())
    state["enablement"]["patch_roots"] = {state["enablement"]["kept_patches"][0]: str(site_packages)}
    write_child_session(child, state)
    processes, peers = _registry(layout, child), _peers(layout)

    budget = SessionBudget.from_hours(2.0, NOW)
    result = await _prelude(layout, processes, peers).root(WORKLOAD, budget, HardStop(budget, _clock()), IMAGE)

    recipe = await peers.repo.read_recipe(result.root_commit)
    assert recipe.unexpressed == (f"patch:{site_packages}/1.patch", "setup_commands")
    assert [component.patches for component in recipe.components] == [()]
    copy = layout.prelude_patches / "0000-1.patch"
    assert copy.read_text() == PATCH_TEXT
    logged = [json.loads(line) for line in layout.decisions.read_text().splitlines()]
    detail = next(entry["detail"] for entry in logged if entry["reason"] == "root committed")
    assert detail["unexpressed"] == list(recipe.unexpressed)
    assert detail["unexpressed_patches"] == {str(copy): str(site_packages)}


async def test_the_recipe_kb_champions_for_the_profile_become_warm_start_leads_prepared_for_replay(
    tmp_path: Path,
) -> None:
    layout = _layout(tmp_path)
    kb = OneChampionKB()
    budget = SessionBudget.from_hours(2.0, NOW)

    prelude = _prelude(layout, _registry(layout, child_with_patch(tmp_path)), _peers(layout), recipe_kb=kb)
    result = await prelude.root(WORKLOAD, budget, HardStop(budget, _clock()), IMAGE)

    [lead] = result.warm_leads
    assert (lead.canonical_id, lead.server_args, lead.gain_pct) == (
        "inference:champion",
        "--enable-expert-parallel",
        12.0,
    )
    assert kb.searched[0]["match"]["hardware"] == result.profile.hardware
    assert lead.replay is not None and lead.replay.recipe is None
    assert lead.replay.launch.server_args == "--enable-expert-parallel"
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

    assert result.warm_leads == ()
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
    assert await _notebook(layout).session_profile() == result.profile


async def test_baseline_is_measured_through_the_toolbox(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    peers = _peers(layout)
    prelude = _prelude(layout, _registry(layout, child_with_patch(tmp_path)), peers)

    result = await _run(prelude, SessionBudget.from_hours(2.0, NOW))

    requests = peers.toolbox.requests
    assert len(requests) == BASELINE_TESTS
    baseline = await _notebook(layout).baseline()
    assert baseline.root_commit == result.root_commit
    assert sorted(baseline.request_ids) == sorted(requests)
    assert baseline.values[GRADED_OUTPUT] == pytest.approx(1000.0)
    assert dict(baseline.threshold_pct) == {GRADED_OUTPUT: 1.0}
    assert set(peers.toolbox.results) == set(requests)
    for request in requests.values():
        assert (request.branch_id, request.parent_commit) == (INTEGRATION_BRANCH, result.root_commit)
        assert (request.kind, request.purpose, request.delta) == (TestKind.TEST, Purpose.BASELINE, RecipeDelta())
        assert (request.accuracy, request.intent_fingerprint, request.priority) == (True, None, None)


async def test_measuring_the_baseline_reports_its_longest_root_test(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    peers = _peers(layout)
    prelude = _prelude(layout, _registry(layout, child_with_patch(tmp_path)), peers)
    budget = SessionBudget.from_hours(2.0, NOW)
    result = await prelude.root(WORKLOAD, budget, HardStop(budget, _clock()), IMAGE)

    longest = await prelude.measure_baseline(result.root_commit)

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
    assert not layout.repo.exists()


async def test_child_without_a_baseline_fails_prelude(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    child = write_child_session(
        tmp_path / "child",
        {"schema_version": 6, "stop_reason": "enablement_attempts_exhausted", "baseline_tput": 0.0},
    )
    peers = _peers(layout)
    with pytest.raises(PreludeFailed, match="enablement_attempts_exhausted"):
        await _run(_prelude(layout, _registry(layout, child, exit_code=1), peers), SessionBudget.from_hours(2.0, NOW))
    assert not layout.repo.exists()


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
