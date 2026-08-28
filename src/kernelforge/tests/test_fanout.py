# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Isolated Implementer lanes and the one device lock their driver runs share."""

from __future__ import annotations

import asyncio
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

from kernelforge import cli
from kernelforge.config import Config
from kernelforge.loop import fanout
from kernelforge.llm.process_reaping import ReapReport
from kernelforge.loop.fanout import (
    SERIALIZED_DRIVER_NAME,
    DeviceBenchmarkLock,
    LanePlan,
    LaneResult,
    run_lanes,
)

DRIVER_NAME = "forge_driver.py"


def _workspace(tmp_path: Path, *, driver_source: str = "pass\n") -> Path:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "kernel.py").write_text("VALUE = 0\n")
    (workspace / DRIVER_NAME).write_text(driver_source)
    (workspace / "build").mkdir()
    (workspace / "build" / "cached.o").write_text("untracked build output\n")
    subprocess.run(["git", "init"], cwd=workspace, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "tests@example.com"],
        cwd=workspace,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "KernelForge Tests"],
        cwd=workspace,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "add", "kernel.py", DRIVER_NAME],
        cwd=workspace,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "initial"],
        cwd=workspace,
        check=True,
        capture_output=True,
    )
    return workspace


def _worktree_workspace(tmp_path: Path, *, detached: bool = False) -> Path:
    """A workspace that is a git worktree, as workspace/worktree.py sets up."""
    main = tmp_path / "main"
    main.mkdir()
    (main / "kernel.py").write_text("VALUE = 0\n")
    (main / DRIVER_NAME).write_text("pass\n")
    for command in (
        ["git", "init"],
        ["git", "config", "user.email", "tests@example.com"],
        ["git", "config", "user.name", "KernelForge Tests"],
        ["git", "add", "kernel.py", DRIVER_NAME],
        ["git", "commit", "-m", "initial"],
    ):
        subprocess.run(command, cwd=main, check=True, capture_output=True)
    workspace = tmp_path / "workspace"
    checkout = ["--detach", str(workspace)] if detached else ["-b", "campaign", str(workspace)]
    subprocess.run(
        ["git", "worktree", "add", *checkout],
        cwd=main,
        check=True,
        capture_output=True,
    )
    (workspace / "build").mkdir()
    (workspace / "build" / "cached.o").write_text("untracked build output\n")
    return workspace


def _git_output(workspace: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=workspace,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout


async def test_a_lane_of_a_worktree_workspace_gets_its_own_index(tmp_path):
    """`cp -a` copies the .git pointer file, so lanes would share one index.

    A single ``git add`` in a lane then stages the lane's edit into the
    canonical repository, and the canonical candidate fingerprint reads the
    lane's work as this round's candidate.
    """
    workspace = _worktree_workspace(tmp_path)

    async def session(_lane: LanePlan, lane_dir: Path, _driver: Path) -> None:
        (lane_dir / "kernel.py").write_text("VALUE = 1\n")
        subprocess.run(
            ["git", "add", "kernel.py"],
            cwd=lane_dir,
            check=True,
            capture_output=True,
        )

    results = await run_lanes(
        workspace_dir=str(workspace),
        lanes=[LanePlan("1", "stage an edit")],
        session=session,
        parent_dir=str(tmp_path),
        driver=DRIVER_NAME,
    )

    assert "VALUE = 1" in results[0].diff
    assert _git_output(workspace, "status", "--porcelain") == "?? build/\n"
    assert _git_output(workspace, "diff", "HEAD", "--", ".") == ""
    assert (workspace / "kernel.py").read_text() == "VALUE = 0\n"


async def test_a_lane_of_a_detached_worktree_diffs_against_the_same_commit(tmp_path):
    """A forge worktree is often checked out detached rather than on a branch."""
    workspace = _worktree_workspace(tmp_path, detached=True)

    async def session(_lane: LanePlan, lane_dir: Path, _driver: Path) -> None:
        (lane_dir / "kernel.py").write_text("VALUE = 1\n")

    results = await run_lanes(
        workspace_dir=str(workspace),
        lanes=[LanePlan("1", "edit the kernel")],
        session=session,
        parent_dir=str(tmp_path),
        driver=DRIVER_NAME,
    )

    assert "VALUE = 1" in results[0].diff
    assert _git_output(workspace, "status", "--porcelain") == "?? build/\n"


async def test_each_lane_edits_its_own_copy_of_the_workspace(tmp_path):
    """Lanes must not be able to see or overwrite each other's edits."""
    workspace = _workspace(tmp_path)

    async def session(lane: LanePlan, lane_dir: Path, _driver: Path) -> None:
        (lane_dir / "kernel.py").write_text(f"VALUE = {lane.lane_id}\n")

    results = await run_lanes(
        workspace_dir=str(workspace),
        lanes=[LanePlan("1", "raise the tile"), LanePlan("2", "stage through LDS")],
        session=session,
        parent_dir=str(tmp_path),
        driver=DRIVER_NAME,
    )

    assert [item.lane_id for item in results] == ["1", "2"]
    assert "VALUE = 1" in results[0].diff
    assert "VALUE = 2" in results[1].diff
    assert (workspace / "kernel.py").read_text() == "VALUE = 0\n"


async def test_a_lane_carries_the_build_state_a_bench_needs(tmp_path):
    """A git worktree would omit build outputs, so an in-session bench would fail."""
    workspace = _workspace(tmp_path)
    seen: list[bool] = []

    async def session(_lane: LanePlan, lane_dir: Path, _driver: Path) -> None:
        seen.append((lane_dir / "build" / "cached.o").is_file())

    await run_lanes(
        workspace_dir=str(workspace),
        lanes=[LanePlan("1", "plan")],
        session=session,
        parent_dir=str(tmp_path),
        driver=DRIVER_NAME,
    )

    assert seen == [True]


async def test_one_failed_lane_does_not_cost_the_round(tmp_path):
    """A lost session is a lost candidate, not a lost iteration."""
    workspace = _workspace(tmp_path)

    async def session(lane: LanePlan, lane_dir: Path, _driver: Path) -> None:
        if lane.lane_id == "1":
            raise RuntimeError("backend refused the session")
        (lane_dir / "kernel.py").write_text("VALUE = 2\n")

    results = await run_lanes(
        workspace_dir=str(workspace),
        lanes=[LanePlan("1", "a"), LanePlan("2", "b")],
        session=session,
        parent_dir=str(tmp_path),
        driver=DRIVER_NAME,
    )

    assert results[0].error.startswith("RuntimeError")
    assert results[0].produced_candidate is False
    assert results[1].produced_candidate is True


async def test_a_lane_that_staged_its_edit_still_reports_a_candidate(tmp_path):
    """``git add`` is routine in a session, so a staged edit is still a candidate."""
    workspace = _workspace(tmp_path)

    async def session(_lane: LanePlan, lane_dir: Path, _driver: Path) -> None:
        (lane_dir / "kernel.py").write_text("VALUE = 7\n")
        subprocess.run(
            ["git", "add", "kernel.py"],
            cwd=lane_dir,
            check=True,
            capture_output=True,
        )

    results = await run_lanes(
        workspace_dir=str(workspace),
        lanes=[LanePlan("1", "stage the edit")],
        session=session,
        parent_dir=str(tmp_path),
        driver=DRIVER_NAME,
    )

    assert results[0].produced_candidate is True
    assert "VALUE = 7" in results[0].diff


async def test_a_lane_whose_diff_cannot_be_read_is_reported_as_lost(tmp_path):
    """ "git failed" and "the agent changed nothing" must not read the same."""
    workspace = _workspace(tmp_path)

    async def session(_lane: LanePlan, lane_dir: Path, _driver: Path) -> None:
        (lane_dir / "kernel.py").write_text("VALUE = 7\n")
        shutil.rmtree(lane_dir / ".git")

    results = await run_lanes(
        workspace_dir=str(workspace),
        lanes=[LanePlan("1", "lose the repository")],
        session=session,
        parent_dir=str(tmp_path),
        driver=DRIVER_NAME,
    )

    assert results[0].diff == ""
    assert "GitError" in results[0].error
    assert "git diff HEAD" in results[0].error
    assert results[0].produced_candidate is False


async def test_a_lane_whose_workspace_stays_busy_is_reported_as_lost(
    tmp_path,
    monkeypatch,
):
    """A lane's leftovers are not just the lane's problem.

    The lanes share one device, and the round's candidates are measured on it
    afterwards. A lane command that survived the reaper is still benching while
    the next lane runs, so both the diff it produced and every number taken
    after it are suspect -- reporting the lane as lost is what keeps that out
    of the KEEP decision.
    """
    workspace = _workspace(tmp_path)

    async def contended_reap(lane_dir: Path) -> ReapReport:
        return ReapReport(
            directory=str(lane_dir),
            unkillable=(4321,),
            holding_device=(4321,),
        )

    monkeypatch.setattr(fanout, "_reap_lane_processes", contended_reap)

    async def session(_lane: LanePlan, lane_dir: Path, _driver: Path) -> None:
        (lane_dir / "kernel.py").write_text("VALUE = 3\n")

    results = await run_lanes(
        workspace_dir=str(workspace),
        lanes=[LanePlan("1", "leave something running")],
        session=session,
        parent_dir=str(tmp_path),
        driver=DRIVER_NAME,
    )

    assert results[0].produced_candidate is False
    assert "could not be cleared" in results[0].error
    assert "4321" in results[0].error
    # The same finding the round reads, carried as data rather than recovered
    # from the sentence above: the round has to decide on it, not parse it.
    assert results[0].contended is True
    assert results[0].reaped is not None
    assert results[0].reaped.blockers == (4321,)


async def test_a_failed_lane_reports_its_own_failure_not_its_teardown_s(
    tmp_path,
    monkeypatch,
):
    """The teardown runs for a lane that already failed, and finds leftovers.

    Of course it does -- the session died with its commands still running. What
    the round needs recorded is why the session died, so the teardown's finding
    must not overwrite it.
    """
    workspace = _workspace(tmp_path)

    async def contended_reap(lane_dir: Path) -> ReapReport:
        return ReapReport(directory=str(lane_dir), unkillable=(4321,))

    monkeypatch.setattr(fanout, "_reap_lane_processes", contended_reap)

    async def session(_lane: LanePlan, _lane_dir: Path, _driver: Path) -> None:
        raise RuntimeError("backend refused the session")

    results = await run_lanes(
        workspace_dir=str(workspace),
        lanes=[LanePlan("1", "fail outright")],
        session=session,
        parent_dir=str(tmp_path),
        driver=DRIVER_NAME,
    )

    assert results[0].error == "RuntimeError: backend refused the session"


async def test_a_failed_lane_still_reports_the_contention_it_left_behind(
    tmp_path,
    monkeypatch,
):
    """Failing for its own reason must not cost the round the device finding.

    A session that raised is the lane most likely to have left commands running,
    and the two facts are unrelated: why this lane produced nothing costs the
    round one candidate, while what is still on the device costs the round its
    measurement. Folding the second into the first is how it gets lost -- the
    error is already taken.
    """
    workspace = _workspace(tmp_path)

    async def contended_reap(lane_dir: Path) -> ReapReport:
        return ReapReport(
            directory=str(lane_dir),
            unkillable=(4321,),
            holding_device=(4321, 8765),
        )

    monkeypatch.setattr(fanout, "_reap_lane_processes", contended_reap)

    async def session(_lane: LanePlan, _lane_dir: Path, _driver: Path) -> None:
        raise RuntimeError("backend refused the session")

    results = await run_lanes(
        workspace_dir=str(workspace),
        lanes=[LanePlan("1", "fail with something still running")],
        session=session,
        parent_dir=str(tmp_path),
        driver=DRIVER_NAME,
    )

    assert results[0].error == "RuntimeError: backend refused the session"
    assert results[0].contended is True
    assert results[0].reaped is not None
    assert results[0].reaped.blockers == (4321, 8765)


async def test_a_lane_that_edits_nothing_reports_no_candidate(tmp_path):
    workspace = _workspace(tmp_path)

    async def session(_lane: LanePlan, _lane_dir: Path, _driver: Path) -> None:
        return None

    results = await run_lanes(
        workspace_dir=str(workspace),
        lanes=[LanePlan("1", "a")],
        session=session,
        parent_dir=str(tmp_path),
        driver=DRIVER_NAME,
    )

    assert results[0].produced_candidate is False
    # A lane that left nothing running says so, so a round reading the reports
    # can tell "clean" from "never asked".
    assert results[0].reaped is not None
    assert results[0].contended is False


async def test_lane_workspaces_are_removed_even_when_a_session_fails(tmp_path):
    workspace = _workspace(tmp_path)
    before = set(tmp_path.iterdir())

    async def session(_lane: LanePlan, _lane_dir: Path, _driver: Path) -> None:
        raise RuntimeError("boom")

    await run_lanes(
        workspace_dir=str(workspace),
        lanes=[LanePlan("1", "a")],
        session=session,
        parent_dir=str(tmp_path),
        driver=DRIVER_NAME,
    )

    # The device sentinel is campaign-scoped and is meant to outlive the round;
    # what must not survive it is the lane copies.
    left = set(tmp_path.iterdir()) - {fanout.campaign_device_lock_path(workspace)}
    assert left == before


async def test_lanes_refuse_to_copy_a_workspace_that_will_not_fit(
    tmp_path,
    monkeypatch,
):
    """A full filesystem must be a refusal with numbers, not a half-copied lane."""
    workspace = _workspace(tmp_path)
    monkeypatch.setattr(fanout, "_available_bytes", lambda _directory: 4096)
    started: list[str] = []

    async def session(lane: LanePlan, _lane_dir: Path, _driver: Path) -> None:
        started.append(lane.lane_id)

    with pytest.raises(RuntimeError) as error:
        await run_lanes(
            workspace_dir=str(workspace),
            lanes=[LanePlan("1", "a"), LanePlan("2", "b")],
            session=session,
            parent_dir=str(tmp_path),
            driver=DRIVER_NAME,
        )

    assert started == []
    assert "4096" in str(error.value)
    assert set(tmp_path.iterdir()) == {workspace}


# ─── The one device every lane's driver run has to queue for ───


def _recording_driver(log: Path, *, hold_sec: float = 0.3) -> str:
    """A driver that marks the window in which it would be holding the device."""
    return (
        "import os, sys, time\n"
        f"LOG = {str(log)!r}\n"
        f"HOLD = {hold_sec!r}\n"
        "with open(LOG, 'a') as handle:\n"
        "    handle.write('start %d\\n' % os.getpid())\n"
        "time.sleep(HOLD)\n"
        "with open(LOG, 'a') as handle:\n"
        "    handle.write('end %d %s\\n' % (os.getpid(), ' '.join(sys.argv[1:])))\n"
        "sys.exit(len(sys.argv) - 1)\n"
    )


def _device_events(log: Path) -> list[str]:
    """Just the start/end sequence, which is where an overlap becomes visible."""
    return [line.split()[0] for line in log.read_text().splitlines()]


async def _wait_for_every_lane(barrier: Path, lane_id: str, *, lanes: int) -> None:
    """Block this lane's session until every other lane's session has started.

    Without this the lanes could finish one after another and a serialized
    device would prove nothing: the windows would not have had the chance to
    overlap in the first place. A lane that waits here forever is a lane whose
    session was serialized, which is the failure this raises on.
    """
    (barrier / lane_id).write_text("started")
    deadline = time.monotonic() + 30.0
    while len(list(barrier.iterdir())) < lanes:
        if time.monotonic() > deadline:
            raise RuntimeError(
                f"lane {lane_id} waited for {lanes} concurrent sessions and saw {len(list(barrier.iterdir()))}"
            )
        await asyncio.sleep(0.01)


async def _run(*args: str, cwd: Path) -> int:
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        *args,
        cwd=str(cwd),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    await process.communicate()
    return process.returncode


async def test_two_lanes_never_hold_the_device_at_the_same_time(tmp_path):
    """The whole point of a lane round: sessions overlap, device runs do not.

    Every number a benchmark takes while another benchmark is on the same GPU
    is worthless, and the lane agent runs the driver from its own shell, in a
    process this loop never sees. The lock therefore has to be held by the
    process that runs the driver, which is why the lane is handed a wrapper
    rather than the driver itself.
    """
    log = tmp_path / "device.log"
    barrier = tmp_path / "barrier"
    barrier.mkdir()
    workspace = _workspace(tmp_path, driver_source=_recording_driver(log))

    async def session(lane: LanePlan, lane_dir: Path, driver: Path) -> None:
        await _wait_for_every_lane(barrier, lane.lane_id, lanes=2)
        assert await _run(str(driver), cwd=lane_dir) == 0

    results = await run_lanes(
        workspace_dir=str(workspace),
        lanes=[LanePlan("1", "a"), LanePlan("2", "b")],
        session=session,
        parent_dir=str(tmp_path),
        driver=DRIVER_NAME,
    )

    assert [item.error for item in results] == ["", ""]
    assert _device_events(log) == ["start", "end", "start", "end"]


async def test_the_serialized_driver_runs_the_lane_s_own_driver(tmp_path):
    """A wrapper that ran the canonical driver would measure the shared tree."""
    log = tmp_path / "device.log"
    workspace = _workspace(tmp_path, driver_source=_recording_driver(log))
    seen: dict[str, object] = {}

    async def session(_lane: LanePlan, lane_dir: Path, driver: Path) -> None:
        (lane_dir / DRIVER_NAME).write_text(_recording_driver(log, hold_sec=0.0).replace("start", "lane-start"))
        seen["exit"] = await _run(str(driver), "--bench-mode", cwd=lane_dir)

    await run_lanes(
        workspace_dir=str(workspace),
        lanes=[LanePlan("1", "a")],
        session=session,
        parent_dir=str(tmp_path),
        driver=DRIVER_NAME,
    )

    # The lane's own edit of the driver ran, its argument arrived, and its exit
    # status came back: one argument, so the recording driver exits 1.
    assert _device_events(log) == ["lane-start", "end"]
    assert log.read_text().splitlines()[1].endswith("--bench-mode")
    assert seen["exit"] == 1


async def test_the_serialized_driver_stays_out_of_the_lane_candidate(tmp_path):
    """A wrapper inside the lane's diff would be rejected as a driver edit.

    The candidate a lane produces is read as ``git diff HEAD -- .`` and refused
    outright when it touches the measurement surface, so the wrapper has to be
    invisible to that read even after the routine ``git add -A`` of a session.
    """
    workspace = _workspace(tmp_path)

    async def session(_lane: LanePlan, lane_dir: Path, _driver: Path) -> None:
        (lane_dir / "kernel.py").write_text("VALUE = 9\n")
        subprocess.run(["git", "add", "-A"], cwd=lane_dir, check=True, capture_output=True)

    results = await run_lanes(
        workspace_dir=str(workspace),
        lanes=[LanePlan("1", "a")],
        session=session,
        parent_dir=str(tmp_path),
        driver=DRIVER_NAME,
    )

    assert "VALUE = 9" in results[0].diff
    assert SERIALIZED_DRIVER_NAME not in results[0].diff
    assert DRIVER_NAME not in results[0].diff


async def test_a_lane_whose_driver_is_outside_the_workspace_is_refused(tmp_path):
    """A lane that cannot be given its own driver must not run at all."""
    workspace = _workspace(tmp_path)
    started: list[str] = []

    async def session(lane: LanePlan, _lane_dir: Path, _driver: Path) -> None:
        started.append(lane.lane_id)

    with pytest.raises(RuntimeError) as error:
        await run_lanes(
            workspace_dir=str(workspace),
            lanes=[LanePlan("1", "a")],
            session=session,
            parent_dir=str(tmp_path),
            driver=str(tmp_path / "elsewhere" / DRIVER_NAME),
        )

    assert started == []
    assert DRIVER_NAME in str(error.value)


async def test_a_lane_s_detached_driver_is_killed_before_the_round_returns(tmp_path):
    """An orphan holding the GPU corrupts the canonical KEEP decision itself.

    The lane agent runs the driver through its own shell, each command detached
    into its own session, so a command still running when the session ends
    survives it. The canonical validation and benchmark run right after this
    round returns, on the same device.
    """
    workspace = _workspace(tmp_path)
    leaked: list[subprocess.Popen] = []

    async def session(_lane: LanePlan, lane_dir: Path, _driver: Path) -> None:
        leaked.append(
            subprocess.Popen(
                [sys.executable, "-c", "import time; time.sleep(120)"],
                cwd=str(lane_dir),
                start_new_session=True,
            )
        )

    await run_lanes(
        workspace_dir=str(workspace),
        lanes=[LanePlan("1", "a")],
        session=session,
        parent_dir=str(tmp_path),
        driver=DRIVER_NAME,
    )

    assert leaked[0].wait(timeout=30) != 0


async def test_a_process_outside_the_lane_survives_the_round(tmp_path):
    """Reaping is scoped to the copies being deleted, not to the workspace."""
    workspace = _workspace(tmp_path)
    bystander = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(120)"],
        cwd=str(workspace),
        start_new_session=True,
    )

    async def session(_lane: LanePlan, _lane_dir: Path, _driver: Path) -> None:
        return None

    try:
        await run_lanes(
            workspace_dir=str(workspace),
            lanes=[LanePlan("1", "a")],
            session=session,
            parent_dir=str(tmp_path),
            driver=DRIVER_NAME,
        )

        assert bystander.poll() is None
    finally:
        bystander.kill()
        bystander.wait(timeout=30)


def _lane_repository(tmp_path: Path, lane_id: str, driver_source: str) -> Path:
    """One lane copy, as run_lanes leaves it: its own repository and driver."""
    lane_dir = tmp_path / "lanes" / lane_id
    lane_dir.mkdir(parents=True)
    (lane_dir / DRIVER_NAME).write_text(driver_source)
    subprocess.run(["git", "init"], cwd=lane_dir, check=True, capture_output=True)
    return lane_dir


async def test_four_lanes_sharing_one_lock_run_the_driver_one_at_a_time(tmp_path):
    """The lock, exercised the only way it is ever used: from other processes.

    An ``asyncio.Lock`` cannot serialize these runs, because the process that
    runs the driver is not this one -- the lane agent is a CLI subprocess and it
    invokes the driver from its own shell, so the timing happens in a
    grandchild. Four wrappers of one lock contend here for real.
    """
    log = tmp_path / "device.log"
    lock = DeviceBenchmarkLock(tmp_path / "sentinel")
    wrappers = []
    for lane_id in ("1", "2", "3", "4"):
        lane_dir = _lane_repository(tmp_path, lane_id, _recording_driver(log, hold_sec=0.2))
        wrappers.append(await lock.install(lane_dir=lane_dir, driver=lane_dir / DRIVER_NAME))
    processes = [subprocess.Popen([sys.executable, str(wrapper)], cwd=str(wrapper.parent)) for wrapper in wrappers]

    for process in processes:
        assert process.wait(timeout=120) == 0
    assert _device_events(log) == ["start", "end"] * 4


async def test_the_serialized_driver_is_hidden_from_the_lane_repository(tmp_path):
    """``git add -A`` is routine in a session and must not stage the wrapper."""
    lock = DeviceBenchmarkLock(tmp_path / "sentinel")
    lane_dir = _lane_repository(tmp_path, "1", "pass\n")

    wrapper = await lock.install(lane_dir=lane_dir, driver=lane_dir / DRIVER_NAME)
    subprocess.run(["git", "add", "-A"], cwd=lane_dir, check=True, capture_output=True)

    assert wrapper.is_file()
    assert wrapper.name not in _git_output(lane_dir, "status", "--porcelain")


def test_a_lane_result_without_a_diff_is_not_a_candidate():
    assert LaneResult("1", "plan").produced_candidate is False
    assert LaneResult("1", "plan", diff="   ").produced_candidate is False
    assert LaneResult("1", "plan", diff="@@ -1 +1 @@").produced_candidate is True


class _RecordingImplementer:
    """Stands in for make_agent_fn, recording what each lane was built with."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)

        async def agent(kernel_path: str, plan: str) -> str:
            return f"{kernel_path}:{plan}"

        return agent


def _campaign(tmp_path: Path) -> tuple[Config, Path, Path]:
    """A campaign workspace with a driver and a source file, plus one lane copy."""
    workspace = tmp_path / "workspace"
    (workspace / "src").mkdir(parents=True)
    (workspace / DRIVER_NAME).write_text("pass\n")
    (workspace / "src" / "kernel.py").write_text("VALUE = 0\n")
    lane_dir = tmp_path / "lanes" / "1"
    lane_dir.mkdir(parents=True)
    return Config(workspace=str(workspace)), workspace, lane_dir


def _lane_factory(config: Config, workspace: Path, implementer) -> object:
    return cli._make_lane_agent_factory(
        make_agent=implementer,
        config=config,
        workspace_dir=str(workspace),
        driver=str(workspace / DRIVER_NAME),
        source_files=[str(workspace / "src" / "kernel.py")],
        session_kwargs={"program_md": "", "kernel_backend_name": "ck"},
    )


def _serialized_driver(lane_dir: Path) -> str:
    """What the round hands the lane factory as its driver invocation."""
    return str(lane_dir / SERIALIZED_DRIVER_NAME)


def test_a_lane_session_gets_its_own_workspace_configuration(tmp_path):
    """A provider that requires the workspace as its cwd starts there.

    The codex provider declares requires_workspace_cwd, and the session resolves
    its cwd from config.workspace. Sharing one Config across lanes puts every
    lane's edits and shell commands in the canonical workspace, while each lane's
    diff is read from a copy nothing touched.
    """
    config, workspace, lane_dir = _campaign(tmp_path)
    implementer = _RecordingImplementer()

    _lane_factory(config, workspace, implementer)(str(lane_dir), _serialized_driver(lane_dir))

    assert implementer.calls[0]["config"].workspace == str(lane_dir)
    assert config.workspace == str(workspace)


async def test_a_lane_session_outside_its_lane_is_refused(tmp_path):
    """An edit outside the lane is an edit no lane diff would ever report."""
    config, workspace, lane_dir = _campaign(tmp_path)
    lane_agent = _lane_factory(config, workspace, _RecordingImplementer())(str(lane_dir), _serialized_driver(lane_dir))

    with pytest.raises(ValueError) as error:
        await lane_agent(str(workspace / "src" / "kernel.py"), "tune it")

    assert str(lane_dir) in str(error.value)
    assert str(workspace / "src") in str(error.value)


def test_a_lane_is_pointed_at_its_own_copy_of_every_source_file(tmp_path):
    """Declared source files become the prompt's entry points and target_files.

    Handing a lane the canonical paths points the agent explicitly at the
    campaign workspace's files, so only the anchor kernel names the lane's copy.
    """
    config, workspace, lane_dir = _campaign(tmp_path)
    implementer = _RecordingImplementer()

    _lane_factory(config, workspace, implementer)(str(lane_dir), _serialized_driver(lane_dir))

    assert implementer.calls[0]["source_files"] == [str(lane_dir / "src" / "kernel.py")]


def test_a_lane_is_given_its_own_copy_of_the_driver(tmp_path):
    config, workspace, lane_dir = _campaign(tmp_path)
    implementer = _RecordingImplementer()

    _lane_factory(config, workspace, implementer)(str(lane_dir), _serialized_driver(lane_dir))

    assert implementer.calls[0]["driver_script"] == str(lane_dir / DRIVER_NAME)


def test_a_relative_path_is_read_against_the_workspace_it_names(tmp_path):
    """Not against wherever forge was launched from.

    Resolving against the process cwd would rebind whatever happens to sit at
    the same relative position under it -- and far more often, refuse a path
    that named the workspace correctly.
    """
    config, workspace, lane_dir = _campaign(tmp_path)
    implementer = _RecordingImplementer()
    factory = cli._make_lane_agent_factory(
        make_agent=implementer,
        config=config,
        workspace_dir=str(workspace),
        driver=DRIVER_NAME,
        source_files=["src/kernel.py"],
        session_kwargs={},
    )

    factory(str(lane_dir), _serialized_driver(lane_dir))

    assert implementer.calls[0]["driver_script"] == str(lane_dir / DRIVER_NAME)
    assert implementer.calls[0]["source_files"] == [str(lane_dir / "src" / "kernel.py")]


def test_a_lane_is_told_to_run_the_driver_through_its_serialized_copy(tmp_path):
    """The factory's second argument is the wrapper, and it has to be used.

    It used to be accepted and dropped, which left the device lock resting on a
    single line of a per-invocation note.
    """
    config, workspace, lane_dir = _campaign(tmp_path)
    implementer = _RecordingImplementer()

    _lane_factory(config, workspace, implementer)(str(lane_dir), _serialized_driver(lane_dir))

    assert implementer.calls[0]["interposed_driver_path"] == _serialized_driver(lane_dir)


def test_a_driver_outside_the_workspace_stops_the_lane(tmp_path):
    """Keeping the canonical driver is the one thing lane isolation must prevent."""
    config, workspace, lane_dir = _campaign(tmp_path)
    outside = tmp_path / "elsewhere" / DRIVER_NAME
    outside.parent.mkdir()
    outside.write_text("pass\n")
    factory = cli._make_lane_agent_factory(
        make_agent=_RecordingImplementer(),
        config=config,
        workspace_dir=str(workspace),
        driver=str(outside),
        source_files=[str(workspace / "src" / "kernel.py")],
        session_kwargs={},
    )

    with pytest.raises(ValueError) as error:
        factory(str(lane_dir), _serialized_driver(lane_dir))

    assert str(outside) in str(error.value)
    assert str(workspace) in str(error.value)


def test_a_source_file_outside_the_workspace_stops_the_lane(tmp_path):
    """A path that cannot be rebound must not be handed over as it is."""
    config, workspace, lane_dir = _campaign(tmp_path)
    outside = tmp_path / "elsewhere" / "kernel.py"
    outside.parent.mkdir()
    outside.write_text("VALUE = 0\n")
    factory = cli._make_lane_agent_factory(
        make_agent=_RecordingImplementer(),
        config=config,
        workspace_dir=str(workspace),
        driver=str(workspace / DRIVER_NAME),
        source_files=[str(outside)],
        session_kwargs={},
    )

    with pytest.raises(ValueError) as error:
        factory(str(lane_dir), _serialized_driver(lane_dir))

    assert str(outside) in str(error.value)
    assert str(workspace) in str(error.value)


async def test_a_lane_session_inside_its_lane_runs(tmp_path):
    config, workspace, lane_dir = _campaign(tmp_path)
    lane_agent = _lane_factory(config, workspace, _RecordingImplementer())(str(lane_dir), _serialized_driver(lane_dir))
    kernel = lane_dir / "src" / "kernel.py"

    assert await lane_agent(str(kernel), "tune it") == f"{kernel}:tune it"


async def test_every_round_serializes_on_one_campaign_wide_sentinel(tmp_path):
    """A sentinel created per round serializes that round's lanes and nothing else.

    The device is the campaign's: an analysis-phase probe and the next round's
    lanes drive the same GPU, so the file they all flock has to be the same one
    in every round and outlive each of them.
    """
    workspace = _workspace(tmp_path)
    expected = fanout.campaign_device_lock_path(workspace)
    seen = []

    async def session(_lane: LanePlan, _lane_dir: Path, driver: Path) -> None:
        seen.append(driver.read_text())

    for _ in range(2):
        await run_lanes(
            workspace_dir=str(workspace),
            lanes=[LanePlan("1", "plan")],
            session=session,
            parent_dir=str(tmp_path),
            driver=DRIVER_NAME,
        )

    assert len(seen) == 2
    assert all(f"SENTINEL = {str(expected)!r}" in text for text in seen)
    # Outlives the round's lane copies, which are removed with it.
    assert expected.is_file()
    assert not expected.is_relative_to(workspace)
