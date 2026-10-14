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
from dataclasses import dataclass
from pathlib import Path

import pytest

import hyperloom
from hyperloom_multibranch.contracts.notebook import INTEGRATION_BRANCH, Role, Writer
from hyperloom_multibranch.contracts.scheduling import PriorityClass
from hyperloom_multibranch.notebook import LocalNotebook
from hyperloom_multibranch.scheduling.budget import SessionBudget
from hyperloom_multibranch.scheduling.clock import Clock
from hyperloom_multibranch.scheduling.decisions import DecisionLog
from hyperloom_multibranch.scheduling.layout import SessionLayout, create_session
from hyperloom_multibranch.scheduling.mb_fakes import MB_FAKE_PRIMARY_METRIC
from hyperloom_multibranch.scheduling.prelude import Prelude, PreludeFailed, PreludeResult, prelude_argv
from hyperloom_multibranch.scheduling.processes import ProcessRegistry, RegisteredProcess
from hyperloom_multibranch.scheduling.settings import Workload
from hyperloom_multibranch.scheduling.tests.peer_fakes import (
    BASELINE_TESTS,
    LAUNCH_FLAG,
    ROOT_COMMIT,
    MbFakeChildRegistry,
    MbFakeGate,
    MbFakeRepo,
    MbFakeToolbox,
)
from hyperloom_multibranch.scheduling.tests.child_sessions import (
    PATCH_TEXT,
    ROOFLINE_PEAK,
    child_with_patch,
    write_child_session,
)

NOW = 1_000_000.0
STUBBORN_CHILD = ("sh", "-c", "trap '' TERM; sleep 60")
#: Clock seconds left before the hard stop when the stubborn child's PRELUDE is cancelled.
NEAR_HARD_STOP_SECONDS = 0.4
WORKLOAD = Workload(model="/models/Llama-3-8B", framework="sglang", tp=1, isl=1024, osl=1024, conc=16, precision="bf16")


class StubbornRegistry(ProcessRegistry):
    """Runs a main-flow child that ignores SIGTERM; a failure double, not a placeholder."""

    async def spawn(self, argv: Sequence[str], *, branch: str, log: Path) -> RegisteredProcess:
        return await super().spawn(STUBBORN_CHILD, branch=branch, log=log)


@dataclass
class PreludePeers:
    """The repository, toolbox and rule gate stand-ins PRELUDE talks to, on the session's notebook."""

    repo: MbFakeRepo
    toolbox: MbFakeToolbox
    gate: MbFakeGate


def _peers(layout: SessionLayout) -> PreludePeers:
    toolbox = MbFakeToolbox(LocalNotebook(layout.notebook, Writer(Role.TOOLBOX, "toolbox"), clock=_clock().now))
    gate = MbFakeGate(LocalNotebook(layout.notebook, Writer(Role.GATE, "gate"), clock=_clock().now), toolbox)
    return PreludePeers(MbFakeRepo(), toolbox, gate)


def _clock(times: list[float] | None = None) -> Clock:
    current = [NOW] if times is None else times
    return Clock(now=lambda: current[0])


def _layout(tmp_path: Path) -> SessionLayout:
    return create_session(tmp_path / "workspace", WORKLOAD.model_basename, _clock())


def _notebook(layout: SessionLayout) -> LocalNotebook:
    return LocalNotebook(layout.notebook, Writer(Role.SCHEDULER, "scheduler"), clock=_clock().now)


def _prelude(
    layout: SessionLayout, processes: ProcessRegistry, peers: PreludePeers, clock: Clock | None = None
) -> Prelude:
    clock = clock or _clock()
    decisions = DecisionLog(layout.decisions, clock)
    return Prelude(layout, processes, _notebook(layout), peers.repo, peers.toolbox, peers.gate, decisions, clock)


async def _run(prelude: Prelude, budget: SessionBudget) -> PreludeResult:
    """Both PRELUDE steps, as the session runs them once the integration branch has its quota."""

    result = await prelude.root(WORKLOAD, budget)
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
    assert "--gpu-type" not in argv
    gpu = prelude_argv(Workload("m", "vllm", 8, 1, 1, 4, "fp8", gpu_type="mi355x"), 1.0, tmp_path / "l.json")
    assert gpu[gpu.index("--gpu-type") + 1] == "mi355x"


async def test_finished_child_roots_the_session_before_any_baseline_test(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    child = child_with_patch(tmp_path)
    processes, peers = _registry(layout, child), _peers(layout)

    result = await _prelude(layout, processes, peers).root(WORKLOAD, SessionBudget.from_hours(2.0, NOW))

    assert float(processes.argv[processes.argv.index("--max-hours") + 1]) == pytest.approx(2.0)
    assert result.root_commit == ROOT_COMMIT
    assert [text for _, text in peers.repo.stored] == [PATCH_TEXT]
    recipe = peers.repo.recipes[ROOT_COMMIT]
    assert json.loads(layout.root_recipe.read_text()) == recipe.to_document()
    assert json.loads(layout.prelude_child.read_text())["session_dir"] == str(child)
    assert await processes.leftovers() == ()
    root = (await _notebook(layout).lineage()).commit(ROOT_COMMIT)
    assert root is not None
    assert (root.branch, root.recipe_digest) == (INTEGRATION_BRANCH, recipe.digest())
    profile = await _notebook(layout).session_profile()
    assert profile == result.profile
    assert (profile.session_id, profile.hardware, profile.framework_version) == (layout.session_id, "mi355x", "0.4.1")
    assert dict(profile.roofline_ceiling) == {MB_FAKE_PRIMARY_METRIC: ROOFLINE_PEAK}
    assert peers.toolbox.submitted == []


async def test_baseline_is_measured_through_the_toolbox(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    peers = _peers(layout)
    prelude = _prelude(layout, _registry(layout, child_with_patch(tmp_path)), peers)

    await _run(prelude, SessionBudget.from_hours(2.0, NOW))

    assert len(peers.toolbox.submitted) == BASELINE_TESTS
    [baseline] = peers.gate.established
    assert baseline.root_commit == ROOT_COMMIT
    assert baseline.request_ids == tuple(request.request_id for request in peers.toolbox.submitted)
    for request in peers.toolbox.submitted:
        assert (request.branch, request.parent_commit) == (INTEGRATION_BRANCH, ROOT_COMMIT)
        assert (request.intent_fingerprint, request.priority) == ("baseline", PriorityClass.SESSION_MEASUREMENT.value)


async def test_child_failing_before_its_launch_info_fails_prelude(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    peers = _peers(layout)
    with pytest.raises(PreludeFailed, match="exited with 3"):
        await _run(_prelude(layout, _registry(layout, None, exit_code=3), peers), SessionBudget.from_hours(2.0, NOW))
    assert json.loads(layout.prelude_child.read_text())["session_dir"] is None
    assert peers.repo.recipes == {}


async def test_child_without_a_baseline_fails_prelude(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    child = write_child_session(
        tmp_path / "child",
        {"schema_version": 6, "stop_reason": "enablement_attempts_exhausted", "baseline_tput": 0.0},
    )
    peers = _peers(layout)
    with pytest.raises(PreludeFailed, match="enablement_attempts_exhausted"):
        await _run(_prelude(layout, _registry(layout, child, exit_code=1), peers), SessionBudget.from_hours(2.0, NOW))
    assert peers.repo.recipes == {}


async def test_withdrawn_baseline_test_fails_prelude(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    peers = _peers(layout)
    peers.toolbox.withdrawing = frozenset({ROOT_COMMIT})
    prelude = _prelude(layout, _registry(layout, child_with_patch(tmp_path)), peers)
    with pytest.raises(PreludeFailed, match="withdrawn"):
        await _run(prelude, SessionBudget.from_hours(2.0, NOW))


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
