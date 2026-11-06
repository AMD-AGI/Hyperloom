# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""How a session's launches are recorded: from the function the agent runner launches a specialist with, and, for the
sandboxes and tests on this host, as a command each really starts can reach this process: the record says unconfined
exactly when the command reaches it."""

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

from hyperloom.common import agent_isolation
from hyperloom.common.agent_isolation import ACCOUNT, NONE, REFUSED
from hyperloom.common.env_safety import SPECIALIST_USER_ENV, RunAs, RunAsError
from hyperloom.measure.local_templates import local_runner_template
from hyperloom.orchestrator.specialists import agent_runner
from hyperloom.sandbox.local import LocalSandboxProvider
from hyperloom_multibranch.contracts.toolbox import BranchLease
from hyperloom_multibranch.scheduling import launch_isolation as launch_isolation_module
from hyperloom_multibranch.scheduling.launch_isolation import (
    HOSTED,
    LOCAL_SANDBOX,
    LOCAL_TEST,
    NO_ACCOUNT,
    LaunchIsolation,
    announce_launch_isolation,
    launch_isolation,
)
from hyperloom_multibranch.scheduling.local_peers import LocalCards, branch_sandboxes
from hyperloom_multibranch.specialist import agent_watch, dev_sandbox
from hyperloom_multibranch.specialist.agent_sandbox import FrameworkProtection

ACCOUNT_RUN_AS = RunAs(user="hl-specialist", uid=1500, gid=1500, groups=(1500,), home="/home/hl", drop_tool="setpriv")
#: A shell line that prints whether it can read this test process's environment, as an unconfined command can.
REACHES = "if cat /proc/{pid}/environ >/dev/null 2>&1; then echo reached; else echo fenced; fi"


@pytest.fixture(autouse=True)
def unmarked(monkeypatch: pytest.MonkeyPatch) -> None:
    """A process no hosted run has marked, with no specialist account configured."""

    monkeypatch.setattr(agent_isolation, "_STATE", agent_isolation._Process())
    monkeypatch.delenv(SPECIALIST_USER_ENV, raising=False)


def _runner_gives(monkeypatch: pytest.MonkeyPatch, answer: RunAs | None | RunAsError) -> None:
    def resolve(gpu_lease: object | None) -> RunAs | None:
        assert gpu_lease is None, "a session's specialists are not launched on the Ray actor path"
        if isinstance(answer, RunAsError):
            raise answer
        return answer

    monkeypatch.setattr(launch_isolation_module, "resolve_launch_user", resolve)


def test_a_specialist_is_recorded_by_the_function_the_agent_runner_launches_it_with() -> None:
    assert launch_isolation_module.resolve_launch_user is agent_runner.resolve_launch_user


@pytest.mark.parametrize("local", [True, False])
def test_a_specialist_the_runner_starts_as_an_account_is_recorded_so(
    local: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    _runner_gives(monkeypatch, ACCOUNT_RUN_AS)

    assert launch_isolation(local=local).specialists == ACCOUNT


def test_a_specialist_the_runner_starts_with_no_account_is_recorded_unconfined_naming_why(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _runner_gives(monkeypatch, None)

    isolation = launch_isolation(local=False)

    assert (isolation.specialists, isolation.reasons, isolation.required) == (NONE, (NO_ACCOUNT,), False)


def test_a_specialist_the_runner_would_refuse_is_recorded_refused_with_the_runners_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(SPECIALIST_USER_ENV, "hl-specialist")
    _runner_gives(monkeypatch, RunAsError("running a specialist as another user needs root or CAP_SETUID"))

    isolation = launch_isolation(local=False)

    assert (isolation.specialists, isolation.required) == (REFUSED, True)
    assert isolation.reasons == ("running a specialist as another user needs root or CAP_SETUID",)
    assert isolation.problems() == (
        "specialists will be refused (running a specialist as another user needs root or CAP_SETUID)"
    )


def test_sandboxes_and_tests_on_this_hosts_cards_are_recorded_unconfined_naming_why(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _runner_gives(monkeypatch, ACCOUNT_RUN_AS)

    isolation = launch_isolation(local=True)

    assert isolation == LaunchIsolation(ACCOUNT, NONE, NONE, False, (LOCAL_SANDBOX, LOCAL_TEST))
    assert isolation.problems() == (
        f"agent and dev sandboxes and tests run unconfined on this host ({LOCAL_SANDBOX}; {LOCAL_TEST})"
    )


def test_sandboxes_and_tests_measured_in_a_grants_sandboxes_are_recorded_there_not_unconfined_here(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _runner_gives(monkeypatch, ACCOUNT_RUN_AS)

    isolation = launch_isolation(local=False)

    assert isolation == LaunchIsolation(ACCOUNT, HOSTED, HOSTED, False, ())
    assert isolation.problems() == ""


def test_a_hosted_runs_mark_makes_its_launches_required(monkeypatch: pytest.MonkeyPatch) -> None:
    _runner_gives(monkeypatch, ACCOUNT_RUN_AS)
    monkeypatch.setattr(agent_isolation._STATE, "hosted", True)

    assert launch_isolation(local=False).required


def test_the_record_reads_back_as_the_report_writes_it() -> None:
    isolation = LaunchIsolation(ACCOUNT, NONE, HOSTED, True, (LOCAL_SANDBOX,))

    assert isolation.to_dict() == {
        "specialists": ACCOUNT,
        "sandboxes": NONE,
        "tests": HOSTED,
        "required": True,
        "reasons": [LOCAL_SANDBOX],
    }
    assert isolation.line() == f"agent isolation: specialists account, sandboxes none, tests hosted: {LOCAL_SANDBOX}"
    assert LaunchIsolation(ACCOUNT, HOSTED, HOSTED, True, ()).line() == (
        "agent isolation: specialists account, sandboxes hosted, tests hosted"
    )


def _said(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [record for record in caplog.records if record.name == launch_isolation_module.__name__]


def test_a_session_that_launches_some_unconfined_says_so_once_at_warning(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger=launch_isolation_module.__name__):
        announce_launch_isolation(LaunchIsolation(NONE, NONE, NONE, False, (NO_ACCOUNT, LOCAL_SANDBOX, LOCAL_TEST)))

    [said] = _said(caplog)
    assert said.levelno == logging.WARNING
    assert said.getMessage() == (
        f"specialists and agent and dev sandboxes and tests run unconfined on this host ({NO_ACCOUNT}; {LOCAL_SANDBOX};"
        f" {LOCAL_TEST}): they can read this account's files and credentials, as before"
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
    and the record must say so; confined, it cannot, and the record must not. The watch the sandbox starts for this
    process is ended with the test: left running, it is a child every later test of this process counts."""

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
    assert reached == (launch_isolation(local=True).sandboxes == NONE), ran.output


def test_the_record_of_the_tests_is_what_a_test_sandbox_on_this_host_reaches(tmp_path: Path) -> None:
    """A measurement sandbox made as the local toolbox makes one starts its command: unconfined, it reads this
    process's environment, and the record must say so; confined, it cannot, and the record must not."""

    line = REACHES.format(pid=os.getpid())
    probe = ("/bin/sh", "-c", f'({line}) > "$HYPERLOOM_SANDBOX_ROOT/out/reached"')
    template = replace(local_runner_template("runner-1", gpus=0), command=probe)
    provider = LocalSandboxProvider([template], base_dir=tmp_path / "sandboxes")
    sandbox = provider.create(template.name, role="measurement", timeout_sec=60, metadata={})
    try:
        process, status = sandbox._startup  # type: ignore[attr-defined]
        assert status.wait_exit(process, None) == 0
        seen = sandbox.read_file("/out/reached").decode().strip()
    finally:
        sandbox.kill()

    assert seen in ("reached", "fenced")
    assert (seen == "reached") == (launch_isolation(local=True).tests == NONE)
