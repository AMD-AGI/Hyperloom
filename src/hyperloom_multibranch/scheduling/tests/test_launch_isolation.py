# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""How a session's launches are recorded: from the function the agent runner launches a specialist with, and in each
of the four cases (a local run with and without bubblewrap, a hosted run with and without its layer's) as a command
each path really starts can reach this process: the record says unconfined exactly when the command reaches it."""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from hyperloom_common import bwrap
from hyperloom_common.bwrap import BwrapProbe
from hyperloom_common.tests._bwrap_host import usable_bwrap
from hyperloom.common import agent_isolation
from hyperloom.common.agent_isolation import (
    ACCOUNT,
    BWRAP,
    NONE,
    REFUSED,
    AccountIsolation,
    AgentIsolation,
    BwrapIsolation,
    IsolationRefused,
    NoIsolation,
)
from hyperloom.common.env_safety import SPECIALIST_USER_ENV, RunAs, RunAsError
from hyperloom.hosted.campaign import parse_campaign
from hyperloom.measure.local_templates import local_runner_template
from hyperloom.orchestrator.specialists import agent_runner
from hyperloom.orchestrator.specialists.launch_env import LaunchPlace, prepare_launch, resolve_launch_isolation
from hyperloom.sandbox.confined import RunnerPlan
from hyperloom.sandbox.local import LocalSandboxProvider, ProcessUsers
from hyperloom_multibranch.contracts.toolbox import BranchLease
from hyperloom_multibranch.scheduling import launch_isolation as launch_isolation_module
from hyperloom_multibranch.scheduling.hosted_run import HOST_LAYER_ENV, require_hosted_isolation
from hyperloom_multibranch.scheduling.launch_isolation import (
    HOSTED,
    LaunchIsolation,
    announce_launch_isolation,
    launch_isolation,
)
from hyperloom_multibranch.scheduling.local_peers import LocalCards, branch_sandboxes
from hyperloom_multibranch.scheduling.peers import PeersUnavailable
from hyperloom_multibranch.specialist import agent_watch, dev_sandbox
from hyperloom_multibranch.specialist.agent_sandbox import FrameworkProtection

ACCOUNT_RUN_AS = RunAs(user="hl-specialist", uid=1500, gid=1500, groups=(1500,), home="/home/hl", drop_tool="setpriv")
LAYER_PROBE = BwrapProbe("/opt/layer/bin/bwrap", (0, 13, 1), ())
#: Why a specialist runs unconfined where no account is named and no bubblewrap works.
SPECIALIST_WHY = (f"running them as an account: {SPECIALIST_USER_ENV} is not set", "no bwrap")
#: A shell line that prints whether it can read this test process's environment, as an unconfined command can.
REACHES = "if cat /proc/{pid}/environ >/dev/null 2>&1; then echo reached; else echo fenced; fi"
#: Why a sandbox on this host runs unconfined, as the choosers say it.
SANDBOX_WHY = ("running them as an account: an agent sandbox runs its commands as the app's own account", "no bwrap")
#: Why a test on this host runs unconfined, as the choosers say it.
TEST_WHY = ("running them as an account: a test sandbox runs as the app's own account", "no bwrap")


@pytest.fixture(autouse=True)
def unmarked(monkeypatch: pytest.MonkeyPatch) -> None:
    """A process no hosted run has marked, with no specialist account configured."""

    monkeypatch.setattr(agent_isolation, "_STATE", agent_isolation._Process())
    monkeypatch.setattr(bwrap, "_PROBES", {})
    monkeypatch.delenv(SPECIALIST_USER_ENV, raising=False)


def _runner_gives(monkeypatch: pytest.MonkeyPatch, answer: AgentIsolation | RunAsError) -> None:
    def resolve(gpu_lease: object | None) -> AgentIsolation:
        assert gpu_lease is None, "a session's specialists are not launched on the Ray actor path"
        if isinstance(answer, RunAsError):
            raise answer
        return answer

    monkeypatch.setattr(launch_isolation_module, "resolve_launch_isolation", resolve)


def test_a_specialist_is_recorded_by_the_function_the_agent_runner_launches_it_with() -> None:
    assert launch_isolation_module.resolve_launch_isolation is agent_runner.resolve_launch_isolation


@pytest.mark.parametrize("local", [True, False])
def test_a_specialist_the_runner_starts_as_an_account_is_recorded_so(
    local: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    _runner_gives(monkeypatch, AccountIsolation(ACCOUNT_RUN_AS))

    assert launch_isolation(local=local).specialists == ACCOUNT


def test_a_specialist_the_runner_starts_under_bubblewrap_is_recorded_so_and_not_as_unconfined(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _runner_gives(monkeypatch, BwrapIsolation(LAYER_PROBE, ("running them as an account: not set",)))

    isolation = launch_isolation(local=False)

    assert isolation == LaunchIsolation(BWRAP, HOSTED, HOSTED, False, ())
    assert isolation.problems() == ""


def test_a_specialist_the_runner_starts_with_no_tier_is_recorded_unconfined_naming_why(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _runner_gives(monkeypatch, NoIsolation(SPECIALIST_WHY))

    isolation = launch_isolation(local=False)

    assert (isolation.specialists, isolation.reasons, isolation.required) == (NONE, SPECIALIST_WHY, False)
    assert isolation.problems() == f"specialists run unconfined on this host ({'; '.join(SPECIALIST_WHY)})"


@pytest.mark.parametrize("refusal", [IsolationRefused, lambda why: RunAsError(why[0])], ids=["no-tier", "run-as"])
def test_a_specialist_the_runner_would_refuse_is_recorded_refused_with_the_runners_reason(
    refusal: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(SPECIALIST_USER_ENV, "hl-specialist")
    _runner_gives(monkeypatch, refusal(("running a specialist as another user needs root or CAP_SETUID",)))

    isolation = launch_isolation(local=False)

    assert (isolation.specialists, isolation.required) == (REFUSED, True)
    assert isolation.reasons == ("running a specialist as another user needs root or CAP_SETUID",)
    assert isolation.problems() == (
        "specialists will be refused (running a specialist as another user needs root or CAP_SETUID)"
    )


def _choosers_give(monkeypatch: pytest.MonkeyPatch, *, confined: bool) -> None:
    """The sandbox and test choosers give bubblewrap (``confined``), else unconfined with their reasons."""

    def sandbox() -> BwrapIsolation | NoIsolation:
        return BwrapIsolation(BwrapProbe("/usr/bin/bwrap", (0, 13, 1), ())) if confined else NoIsolation(SANDBOX_WHY)

    monkeypatch.setattr(launch_isolation_module, "choose_sandbox_isolation", sandbox)
    monkeypatch.setattr(
        launch_isolation_module, "local_test_isolation", lambda: (BWRAP, ()) if confined else (NONE, TEST_WHY)
    )


def test_sandboxes_and_tests_on_this_hosts_cards_are_recorded_under_bubblewrap_where_their_choosers_give_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _runner_gives(monkeypatch, AccountIsolation(ACCOUNT_RUN_AS))
    _choosers_give(monkeypatch, confined=True)

    isolation = launch_isolation(local=True)

    assert isolation == LaunchIsolation(ACCOUNT, BWRAP, BWRAP, False, ())
    assert isolation.problems() == ""


def test_sandboxes_and_tests_on_this_hosts_cards_are_recorded_unconfined_naming_why_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _runner_gives(monkeypatch, AccountIsolation(ACCOUNT_RUN_AS))
    _choosers_give(monkeypatch, confined=False)

    isolation = launch_isolation(local=True)

    assert isolation == LaunchIsolation(ACCOUNT, NONE, NONE, False, (*SANDBOX_WHY, TEST_WHY[0]))
    assert isolation.problems() == (
        f"agent and dev sandboxes and tests run unconfined on this host ({SANDBOX_WHY[0]}; no bwrap; {TEST_WHY[0]})"
    )


def test_sandboxes_and_tests_a_hosted_run_cannot_confine_here_are_recorded_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _runner_gives(monkeypatch, AccountIsolation(ACCOUNT_RUN_AS))
    monkeypatch.setattr(agent_isolation._STATE, "hosted", True)

    isolation = launch_isolation(local=True)

    assert (isolation.sandboxes, isolation.tests) == (REFUSED, REFUSED)
    assert agent_isolation.NO_LAYER_BWRAP in isolation.reasons
    assert isolation.problems().startswith("agent and dev sandboxes and tests will be refused (")


def test_sandboxes_and_tests_measured_in_a_grants_sandboxes_are_recorded_there_not_unconfined_here(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _runner_gives(monkeypatch, AccountIsolation(ACCOUNT_RUN_AS))

    isolation = launch_isolation(local=False)

    assert isolation == LaunchIsolation(ACCOUNT, HOSTED, HOSTED, False, ())
    assert isolation.problems() == ""


def test_a_hosted_runs_mark_makes_its_launches_required(monkeypatch: pytest.MonkeyPatch) -> None:
    _runner_gives(monkeypatch, AccountIsolation(ACCOUNT_RUN_AS))
    monkeypatch.setattr(agent_isolation._STATE, "hosted", True)

    assert launch_isolation(local=False).required


def test_the_record_reads_back_as_the_report_writes_it() -> None:
    isolation = LaunchIsolation(ACCOUNT, NONE, BWRAP, True, ("no bwrap",))

    assert isolation.to_dict() == {
        "specialists": ACCOUNT,
        "sandboxes": NONE,
        "tests": BWRAP,
        "required": True,
        "reasons": ["no bwrap"],
    }
    assert isolation.line() == "agent isolation: specialists account, sandboxes none, tests bwrap: no bwrap"
    assert LaunchIsolation(ACCOUNT, HOSTED, HOSTED, True, ()).line() == (
        "agent isolation: specialists account, sandboxes hosted, tests hosted"
    )


def _said(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [record for record in caplog.records if record.name == launch_isolation_module.__name__]


def test_a_session_that_launches_some_unconfined_says_so_once_at_warning(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger=launch_isolation_module.__name__):
        announce_launch_isolation(LaunchIsolation(NONE, NONE, NONE, False, SPECIALIST_WHY))

    [said] = _said(caplog)
    assert said.levelno == logging.WARNING
    assert said.getMessage() == (
        f"specialists and agent and dev sandboxes and tests run unconfined on this host ({'; '.join(SPECIALIST_WHY)}):"
        " they can read this account's files and credentials, as before"
    )


def test_a_session_whose_specialists_would_be_refused_says_so_and_not_that_they_run(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO, logger=launch_isolation_module.__name__):
        announce_launch_isolation(LaunchIsolation(REFUSED, HOSTED, HOSTED, True, ("needs root",)))

    [said] = _said(caplog)
    assert (said.levelno, said.getMessage()) == (logging.WARNING, "specialists will be refused (needs root)")


def test_a_session_that_launches_nothing_unconfined_says_how_at_info(caplog: pytest.LogCaptureFixture) -> None:
    isolation = LaunchIsolation(ACCOUNT, HOSTED, HOSTED, True, ())

    with caplog.at_level(logging.INFO, logger=launch_isolation_module.__name__):
        announce_launch_isolation(isolation)

    [said] = _said(caplog)
    assert (said.levelno, said.getMessage()) == (logging.INFO, isolation.line())


# -- what a command really launched on this host reaches --------------------------------------------------------------


async def test_the_record_of_the_sandboxes_is_what_a_dev_command_on_this_host_reaches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The dev sandbox the local backend builds runs one command: unconfined, it reads this process's environment,
    and the record must say so; confined, it cannot, and the record must not, unless that one sandbox's own paths
    could not be bound (on bubblewrap before 0.12.0 a path below the agent-writable /tmp, as ``tmp_path`` is), which it
    says itself. The watch the sandbox starts for this process is ended with the test: left running, it is a child
    every later test of this process counts."""

    watches: dict[int, tuple[subprocess.Popen[bytes], Path]] = {}
    monkeypatch.setattr(agent_watch, "_watches", watches)
    monkeypatch.setattr(agent_watch, "_sandboxes", {})
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "local-tmp"))
    (tmp_path / "local-tmp").mkdir()
    monkeypatch.setattr(dev_sandbox, "framework_protection", FrameworkProtection)
    worktree = SimpleNamespace(path=tmp_path / "worktree", linked=(), holds=lambda component: False)
    worktree.path.mkdir()
    setup: Any = SimpleNamespace(processes=None)
    sandboxes = branch_sandboxes(LocalCards((0,)), setup, lambda branch: worktree)  # type: ignore[arg-type,return-value]
    sandbox = sandboxes.dev("b1", BranchLease("lease-1", "b1", (0,), tmp_path / "lease", {}, 0))
    try:
        ran = await sandbox.run(REACHES.format(pid=os.getpid()), tmp_path, {}, 60)
    finally:
        await sandbox.stop()
        for process, _registry in watches.values():
            process.terminate()
            process.wait(timeout=30)

    assert ran.returncode == 0, ran.output
    reached = ran.output.split()[-1] == "reached"
    recorded, own = launch_isolation(local=True).sandboxes, getattr(sandbox, "isolation", None)
    own_paths_refused = recorded == BWRAP and isinstance(own, NoIsolation)
    assert reached == (recorded == NONE or own_paths_refused), ran.output
    assert not own_paths_refused or "below the agent-writable /tmp" in own.why, own


def test_the_record_of_the_tests_is_what_a_test_sandbox_on_this_host_reaches(tmp_path: Path) -> None:
    """A measurement sandbox made as the local toolbox makes one starts its command: unconfined, it reads this
    process's environment, and the record must say so; confined, it cannot, and the record must not."""

    line = REACHES.format(pid=os.getpid())
    probe = ("/bin/sh", "-c", f'({line}) > "$HYPERLOOM_SANDBOX_ROOT/out/reached"')
    template = replace(local_runner_template("runner-1", gpus=0), command=probe)
    provider = LocalSandboxProvider([template], base_dir=tmp_path / "sandboxes", users=ProcessUsers(plan=RunnerPlan()))
    sandbox = provider.create(template.name, role="measurement", timeout_sec=60, metadata={})
    try:
        process, status = sandbox._startup  # type: ignore[attr-defined]
        assert status.wait_exit(process, None) == 0
        seen = sandbox.read_file("/out/reached").decode().strip()
    finally:
        sandbox.kill()

    assert seen in ("reached", "fenced")
    assert (seen == "reached") == (launch_isolation(local=True).tests == NONE)


# -- the four cases, through the real choosers and real bubblewrap ---------------------------------------------------

#: How each case launches (specialists, sandboxes, tests) on this host's cards, and in a grant's sandboxes.
FOUR_CASES = {
    "local-bwrap": ((BWRAP, BWRAP, BWRAP), (BWRAP, HOSTED, HOSTED)),
    "local-no-bwrap": ((NONE, NONE, NONE), (NONE, HOSTED, HOSTED)),
    "hosted-layer-bwrap": ((BWRAP, BWRAP, BWRAP), (BWRAP, HOSTED, HOSTED)),
    "hosted-no-layer-bwrap": ((REFUSED, REFUSED, REFUSED), (REFUSED, HOSTED, HOSTED)),
}


def _spec(measurement: str) -> Any:
    return parse_campaign(
        {
            "schema": "hyperloom.campaign-spec/v1",
            "flow": "multibranch",
            "model": {"path": "/models/Llama-3-8B"},
            "framework": {"name": "vllm", "image": f"registry.example/vllm@sha256:{'2' * 64}"},
            "hardware": {"gpu_type": "mi355x", "tp": 1},
            "workload": {"conc": 16, "isl": 1024, "osl": 512, "precision": "fp8"},
            "budget": {"max_hours": 1},
            "placement": {"profile": "hosted-gpu", "measurement": measurement},
            "multibranch": {"workload": "inferencex"},
        }
    )


def _enter_case(case: str, local: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> PeersUnavailable | None:
    """Make this process ``case`` with no specialist account: a local run with the real bubblewrap on ``PATH`` or with
    none there, or a hosted run started (:func:`require_hosted_isolation`) with its layer holding the real bubblewrap
    or no layer named, ``PATH`` then holding no usable one. Returns the hosted start's refusal, if any."""

    no_bwrap = tmp_path / "no-bwrap-bin"
    no_bwrap.mkdir()
    if case == "local-bwrap":
        usable_bwrap()
        return None
    if case == "hosted-layer-bwrap":
        real = usable_bwrap()
        layer = tmp_path / "layer"
        (layer / "bin").mkdir(parents=True)
        (layer / "bin" / "bwrap").symlink_to(str(real.path))
        environ = {HOST_LAYER_ENV: str(layer)}
    else:
        environ = {}
    monkeypatch.setenv("PATH", str(no_bwrap))
    if case == "local-no-bwrap":
        return None
    try:
        require_hosted_isolation(_spec("local" if local else "hosted"), environ)
    except PeersUnavailable as refused:
        return refused
    return None


@pytest.mark.agent_isolation
@pytest.mark.parametrize("local", [True, False], ids=["measured-here", "measured-in-grant"])
@pytest.mark.parametrize("case", list(FOUR_CASES))
def test_the_record_in_each_case_is_what_each_path_chooses_and_the_hosted_start_refuses_only_a_real_gap(
    case: str, local: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    refused = _enter_case(case, local, tmp_path, monkeypatch)

    isolation = launch_isolation(local=local)

    expected = FOUR_CASES[case][0 if local else 1]
    assert (isolation.specialists, isolation.sandboxes, isolation.tests) == expected, isolation
    assert isolation.required == case.startswith("hosted")
    assert bool(isolation.reasons) == (NONE in expected or REFUSED in expected), isolation.reasons
    assert (isolation.problems() == "") == (expected.count(BWRAP) + expected.count(HOSTED) == 3)
    if case == "hosted-layer-bwrap":
        assert refused is None, "with the layer's bubblewrap nothing here runs unconfined, so the run starts"
    elif case == "hosted-no-layer-bwrap":
        assert refused is not None and agent_isolation.NO_LAYER_BWRAP in str(refused)


@pytest.mark.agent_isolation
@pytest.mark.parametrize("case", list(FOUR_CASES))
def test_the_record_of_the_specialists_is_what_a_specialist_the_runner_launches_reaches(
    case: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A specialist launched as the agent runner launches one, through the tier its chooser gives now: unconfined, it
    reads this process's environment, and the record must say so; under bubblewrap it cannot, and the record must
    not; where it must be isolated and cannot be, it is not launched at all, and the record says refused."""

    _enter_case(case, True, tmp_path, monkeypatch)
    recorded = launch_isolation(local=True).specialists
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    cmd = ["/bin/sh", "-c", REACHES.format(pid=os.getpid())]
    env = {"PATH": "/usr/bin:/bin"}
    try:
        tier = resolve_launch_isolation(None)
    except IsolationRefused:
        assert recorded == REFUSED
        return
    prepared = prepare_launch(tier, LaunchPlace(workspace), cmd, env)
    try:
        spawn = prepared.local(cmd, env)
        ran = subprocess.run(
            spawn.argv, env=spawn.env, capture_output=True, text=True, timeout=60, check=False, **spawn.kwargs
        )
    finally:
        prepared.release()

    assert ran.returncode == 0, ran.stderr
    assert (ran.stdout.split()[-1] == "reached") == (recorded == NONE), (recorded, ran.stdout, ran.stderr)
    assert recorded in (BWRAP, NONE)
