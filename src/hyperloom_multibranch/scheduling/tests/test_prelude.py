# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""PRELUDE's contract: the child's command line, what a finished child roots the session on, and how it fails."""

from __future__ import annotations

import asyncio
import json
import re
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pytest

import hyperloom
from hyperloom.common.io import atomic_write_json
from hyperloom.common.outside_facts.rule_pack import rule_pack_identity
from hyperloom.common.perf_metric import GRADED_OUTPUT
from hyperloom.contracts.measurement import Purpose, TestKind
from hyperloom.contracts.recipe import Recipe, RecipeDelta
from hyperloom.measure.compile import LocalBlobStore
from hyperloom_multibranch.contracts.notebook import INTEGRATION_BRANCH, Role, Writer
from hyperloom_multibranch.contracts.toolbox import RequestRejected
from hyperloom_multibranch.notebook import LocalNotebook
from hyperloom_multibranch.notebook.codec import encode
from hyperloom_multibranch.notebook.knowledge.warm_start import RecipeDownloader, ReplayRefusal
from hyperloom_multibranch.repo.git_repo import GitExperimentRepo
from hyperloom_multibranch.scheduling.budget import HardStop, SessionBudget
from hyperloom_multibranch.scheduling.clock import Clock
from hyperloom_multibranch.scheduling.context import SchedulerContext
from hyperloom_multibranch.scheduling.roster import Roster, RosterCaps
from hyperloom_multibranch.scheduling.decisions import DecisionLog
from hyperloom_multibranch.scheduling.forks import ForkDesk
from hyperloom_multibranch.scheduling.layout import SessionLayout, create_session
from hyperloom_multibranch.rulegate.service import BASELINE_TESTS, GateService
from hyperloom_multibranch.scheduling import prelude as prelude_module
from hyperloom_multibranch.scheduling.prelude import (
    PREFLIGHT_OUT_OF_TIME,
    Prelude,
    PreludeFailed,
    PreludeResult,
    PreludeServices,
    WarmStartSource,
    prelude_argv,
    reused_prelude,
)
from hyperloom_multibranch.scheduling.processes import ProcessRegistry, RegisteredProcess
from hyperloom_multibranch.scheduling.requests import SessionTests
from hyperloom_multibranch.scheduling.settings import InferenceXShape, KernelOptimizer, ReusedPrelude, Workload
from hyperloom_multibranch.scheduling.tests.peer_fakes import (
    BASE_IMAGE_DIGEST,
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
    POINT_CONFIG,
    ROOFLINE_PEAK,
    ROOFLINE_RUN_ID,
    TRACE_FILE,
    child_with_patch,
    with_roofline_profile,
    write_child_session,
)

NOW = 1_000_000.0
#: The image the fake toolbox measures on, which the root recipe must carry.
IMAGE = BASE_IMAGE_DIGEST
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
    repo = GitExperimentRepo(layout.repo, LocalBlobStore(layout.blobs))
    toolbox = RecordingToolbox(toolbox_notebook, accuracy=scores, conc=WORKLOAD.conc, repo=repo)
    tests = SessionTests(toolbox, Clock(now=_clock().now, sleep=_instant))
    gate = session_gate(LocalNotebook(layout.notebook, Writer(Role.GATE, "gate"), clock=_clock().now), toolbox)
    return PreludePeers(repo, toolbox, tests, gate)


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
    services = PreludeServices(peers.repo, peers.gate, peers.tests, peers.toolbox)
    return Prelude(context, services, warm_start=warm_start)


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
    assert reasons.index("root committed") + 1 == reasons.index("root profile started")
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

    assert (await _notebook(layout).baseline()).root_commit == result.root_commit
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
    assert lead.replay is not None and lead.replay.recipe is None and lead.refusal is None
    assert lead.replay.launch.server_args == "--enable-expert-parallel"
    assert result.champion is lead.replay
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

    assert result.warm_leads == () and result.champion is None
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
