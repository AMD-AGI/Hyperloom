# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Local agent runner: deadline path, heartbeat rule, status, and resource records."""

from __future__ import annotations

import asyncio as real_asyncio
import json
import os
import signal
import subprocess
import sys
import time as real_time
from pathlib import Path
from types import SimpleNamespace

import pytest

from hyperloom.common.deadline import Deadline
from hyperloom.orchestrator.specialists import agent_runner
from hyperloom.orchestrator.specialists import subprocess_ as ss
from hyperloom.orchestrator.specialists.agent_runner import LOST_AFTER_SEC, AgentConfigError, Liveness, McpEndpoint
from hyperloom.orchestrator.specialists.tests._agent_fixtures import (
    child_argv,
    make_runner,
    make_spec,
    wait_finished,
)
from hyperloom.orchestrator.specialists.tests.test_resource_sampling import add_proc, write_cgroup

HOUR = 3600.0


class FakeClock:
    """A virtual wall and monotonic clock; time passes only when something sleeps."""

    def __init__(self) -> None:
        self.mono = 5000.0
        self.wall = real_time.time()
        self.files: list[Path] = []
        self.keep_alive = True
        self.on_tick: list = []

    def monotonic(self) -> float:
        return self.mono

    def time(self) -> float:
        return self.wall

    async def sleep(self, seconds: float) -> None:
        self.mono += seconds
        self.wall += seconds
        if self.keep_alive:
            for path in self.files:
                if path.exists():
                    os.utime(path, (self.wall, self.wall))
        for hook in self.on_tick:
            hook()
        await real_asyncio.sleep(0)


class ClockedProcess:
    """A process stand-in that exits once the virtual clock reaches ``exit_at``."""

    pid = 4194000

    def __init__(self, clock: FakeClock, exit_at: float) -> None:
        self._clock = clock
        self._exit_at = exit_at
        self.returncode: int | None = None
        self.args: list[str] = []

    def poll(self) -> int | None:
        if self.returncode is None and self._clock.mono >= self._exit_at:
            self.returncode = 0
        return self.returncode


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> FakeClock:
    """Put the existing reap loop on the virtual clock; its own code path is the one under test."""
    fake = FakeClock()
    monkeypatch.setattr(ss, "time", SimpleNamespace(monotonic=fake.monotonic, time=fake.time, sleep=fake.sleep))
    monkeypatch.setattr(ss, "asyncio", SimpleNamespace(sleep=fake.sleep, CancelledError=real_asyncio.CancelledError))
    return fake


def clocked_runner(tmp_path: Path, clock: FakeClock, **kwargs):
    kwargs.setdefault("sleep", clock.sleep)
    return make_runner(
        tmp_path / "session", monotonic=clock.monotonic, wall=clock.time, poll_interval_seconds=5.0, **kwargs
    )


async def test_session_longer_than_four_hours_is_not_truncated(tmp_path: Path, clock: FakeClock) -> None:
    ws = tmp_path / "w"
    spec = make_spec(ws, deadline_unix=clock.wall + 6 * HOUR)
    clock.files = [spec.heartbeat_path]
    spec.heartbeat_path.write_text("{}")
    start = clock.mono
    runner = clocked_runner(tmp_path, clock, popen=lambda *a, **k: ClockedProcess(clock, start + 5 * HOUR))
    handle = await runner.launch(spec)
    await handle.session.task
    status = await runner.poll(handle)
    assert status.state == "exited"
    assert status.returncode == 0  # it ended by itself, 5 hours in
    assert not status.timed_out
    assert clock.mono - start >= 5 * HOUR


async def test_the_four_hour_cap_exists_for_an_unbounded_run(tmp_path: Path, clock: FakeClock) -> None:
    """Control: the same reap loop with no deadline kills at the 4 hour cap, so the test above is meaningful."""
    ws = tmp_path / "w"
    ws.mkdir()
    heartbeat = ws / "heartbeat.json"
    heartbeat.write_text("{}")
    clock.files = [heartbeat]
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"], start_new_session=True)
    try:
        dispatcher = ss.SpecialistSubprocessDispatcher(
            ss.SpecialistSubprocessConfig(poll_interval_seconds=5.0, heartbeat_stale_seconds=LOST_AFTER_SEC)
        )
        started = clock.mono
        outcome = await dispatcher._reap_loop(
            proc,
            ss.ReapWatch(workspace=ws, done_files=(ws / "specialist_done.json",), heartbeat_file=heartbeat),
            deadline=None,
            started=started,
        )
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait()
    assert outcome["timed_out"]
    assert 4 * HOUR <= clock.mono - started < 4 * HOUR + 60


async def test_session_is_killed_at_its_own_deadline_not_before_and_not_much_after(
    tmp_path: Path, clock: FakeClock
) -> None:
    ws = tmp_path / "w"
    spec = make_spec(ws, deadline_unix=clock.wall + 6 * HOUR)
    clock.files = [spec.heartbeat_path]
    spec.heartbeat_path.write_text("{}")
    runner = clocked_runner(tmp_path, clock)
    start = clock.mono
    handle = await runner.launch(spec)
    await handle.session.task
    await runner.collect(handle)
    status = await runner.poll(handle)
    assert status.state == "exited" and status.timed_out
    assert 6 * HOUR <= clock.mono - start < 6 * HOUR + 60
    record = json.loads((tmp_path / "session" / "agents" / "resources.jsonl").read_text().splitlines()[0])
    assert record["ended_by"] == "timeout"
    with pytest.raises(ProcessLookupError):
        os.kill(handle.pid, 0)


async def test_deadline_is_derived_from_the_absolute_wall_clock(tmp_path: Path, clock: FakeClock) -> None:
    spec = make_spec(tmp_path / "w", deadline_unix=clock.wall + 6 * HOUR + 123)
    clock.files = [spec.heartbeat_path]
    spec.heartbeat_path.write_text("{}")
    runner = clocked_runner(tmp_path, clock)
    handle = await runner.launch(spec)
    try:
        deadline: Deadline = handle.session.deadline
        assert deadline.at == pytest.approx(clock.mono + 6 * HOUR + 123, abs=1.0)
    finally:
        await runner.cancel(handle)
        await handle.session.task


async def test_silence_for_more_than_300_seconds_is_lost(tmp_path: Path, clock: FakeClock) -> None:
    ws = tmp_path / "w"
    spec = make_spec(ws, deadline_unix=clock.wall + 6 * HOUR)
    spec.heartbeat_path.write_text("{}")
    clock.keep_alive = False  # nothing ever refreshes the heartbeat or the process log
    runner = clocked_runner(tmp_path, clock)
    start = clock.mono
    handle = await runner.launch(spec)
    await handle.session.task
    status = await runner.poll(handle)
    assert status.state == "lost"
    assert "heartbeat stale" in status.error
    assert LOST_AFTER_SEC <= clock.mono - start < 2 * LOST_AFTER_SEC + 30
    await runner.collect(handle)
    record = json.loads((tmp_path / "session" / "agents" / "resources.jsonl").read_text().splitlines()[0])
    assert record["ended_by"] == "lost"


async def test_poll_reports_lost_while_the_process_is_still_up(tmp_path: Path) -> None:
    ws = tmp_path / "w"
    runner = make_runner(tmp_path / "session", poll_interval_seconds=2.0)  # the reap loop does not act first
    spec = make_spec(ws)
    handle = await runner.launch(spec)
    try:
        assert (await runner.poll(handle)).state == "running"
        old = real_time.time() - (LOST_AFTER_SEC + 100)
        for path in (spec.heartbeat_path, ws / "process.log"):
            path.touch()
            os.utime(path, (old, old))
        status = await runner.poll(handle)
        assert status.state == "lost"
        assert status.last_heartbeat_unix == pytest.approx(old, abs=1.0)
        assert status.returncode is None
    finally:
        await runner.cancel(handle)
        await handle.session.task


async def test_a_runner_given_a_longer_lost_window_keeps_a_session_quiet_for_less(tmp_path: Path) -> None:
    ws = tmp_path / "w"
    runner = make_runner(
        tmp_path / "session", poll_interval_seconds=2.0, liveness=Liveness(lost_after_seconds=3 * LOST_AFTER_SEC)
    )
    spec = make_spec(ws)
    handle = await runner.launch(spec)
    try:
        assert (await runner.poll(handle)).state == "running"
        quiet = real_time.time() - (LOST_AFTER_SEC + 100)
        for path in (spec.heartbeat_path, ws / "process.log"):
            path.touch()
            os.utime(path, (quiet, quiet))
        assert (await runner.poll(handle)).state == "running"
        silent = real_time.time() - (3 * LOST_AFTER_SEC + 100)
        for path in (spec.heartbeat_path, ws / "process.log"):
            os.utime(path, (silent, silent))
        assert (await runner.poll(handle)).error == f"no heartbeat for more than {3 * LOST_AFTER_SEC:.0f}s"
    finally:
        await runner.cancel(handle)
        await handle.session.task


async def test_fresh_heartbeat_keeps_a_quiet_process_running(tmp_path: Path) -> None:
    ws = tmp_path / "w"
    runner = make_runner(tmp_path / "session")
    spec = make_spec(ws)
    handle = await runner.launch(spec)
    try:
        spec.heartbeat_path.write_text("{}")
        os.utime(ws / "process.log", (real_time.time() - 1000, real_time.time() - 1000))
        status = await runner.poll(handle)
        assert status.state == "running"
        assert status.last_heartbeat_unix == pytest.approx(real_time.time(), abs=5.0)
    finally:
        await runner.cancel(handle)
        await wait_finished(runner, handle)
        await runner.collect(handle)


async def test_exit_code_and_done_file_are_reported(tmp_path: Path) -> None:
    ws = tmp_path / "w"
    runner = make_runner(tmp_path / "session")
    handle = await runner.launch(make_spec(ws, code="import sys; sys.exit(3)"))
    status = await wait_finished(runner, handle)
    assert (status.state, status.returncode) == ("exited", 3)
    await runner.collect(handle)
    record = json.loads((tmp_path / "session" / "agents" / "resources.jsonl").read_text().splitlines()[0])
    assert record["ended_by"] == "failed"


async def test_collect_of_a_running_session_is_refused(tmp_path: Path) -> None:
    runner = make_runner(tmp_path / "session")
    handle = await runner.launch(make_spec(tmp_path / "w"))
    try:
        with pytest.raises(RuntimeError, match="still running; cancel it before collect"):
            await runner.collect(handle)
    finally:
        await runner.cancel(handle)


async def test_resource_records_are_complete_and_appended(tmp_path: Path) -> None:
    ws = tmp_path / "w"
    (ws / "worktree").mkdir(parents=True)
    (ws / "worktree" / "big.bin").write_bytes(b"x" * 50000)
    cgroup = tmp_path / "cg"
    write_cgroup(cgroup, memory_max="4294967296", peak="1000000", oom=0, cpu_max="400000 100000")
    code = "import pathlib, time\nx = bytearray(20_000_000)\ntime.sleep(0.4)\npathlib.Path('specialist_done.json').write_text('{}')\n"
    runner = make_runner(tmp_path / "session", cgroup_root=cgroup)
    handle = await runner.launch(make_spec(ws, code=code, gpus=0))
    await wait_finished(runner, handle)
    await runner.collect(handle)
    await runner.finish_campaign()
    lines = [json.loads(x) for x in (tmp_path / "session" / "agents" / "resources.jsonl").read_text().splitlines()]
    specialist, container_at_end, container_final = lines
    assert specialist["kind"] == "specialist_resources"
    assert specialist["task_id"] == "task-1" and specialist["backend"] == "claude" and specialist["location"] == "local"
    assert specialist["peak_rss_bytes"] >= 20_000_000
    assert specialist["worktree_disk_bytes"] >= 50000
    assert specialist["selfcheck_venv_disk_bytes"] is None
    assert specialist["concurrent_specialists_at_start"] == 1
    assert specialist["ended_by"] == "normal"
    assert specialist["wall_seconds"] > 0
    assert specialist["cpu_seconds"] >= 0
    assert container_at_end["kind"] == container_final["kind"] == "container_resources"
    assert container_final["memory_max_bytes"] == 4294967296
    assert container_final["cpu_quota_us"] == 400000
    assert container_final["max_concurrent_specialists"] == 1
    assert oct((tmp_path / "session" / "agents" / "resources.jsonl").stat().st_mode & 0o777) == "0o600"


async def test_concurrency_at_start_counts_sessions_already_running(tmp_path: Path) -> None:
    runner = make_runner(tmp_path / "session")
    handles = [await runner.launch(make_spec(tmp_path / f"w{i}", task_id=f"t{i}")) for i in range(3)]
    for handle in handles:
        await runner.cancel(handle)
    for handle in handles:
        await wait_finished(runner, handle)
        await runner.collect(handle)
    rows = [json.loads(x) for x in (tmp_path / "session" / "agents" / "resources.jsonl").read_text().splitlines()]
    starts = sorted(r["concurrent_specialists_at_start"] for r in rows if r["kind"] == "specialist_resources")
    assert starts == [1, 2, 3]
    assert max(r["max_concurrent_specialists"] for r in rows if r["kind"] == "container_resources") == 3


async def test_a_process_killed_while_the_cgroup_counts_an_oom_is_recorded_as_oom_killed(tmp_path: Path) -> None:
    cgroup = tmp_path / "cg"
    write_cgroup(cgroup, memory_max="1000000", peak="1", oom=0, cpu_max=None)
    runner = make_runner(tmp_path / "session", cgroup_root=cgroup)
    handle = await runner.launch(make_spec(tmp_path / "w"))
    write_cgroup(cgroup, memory_max="1000000", peak="1", oom=1, cpu_max=None)
    os.kill(handle.pid, signal.SIGKILL)
    status = await wait_finished(runner, handle)
    assert status.returncode == -signal.SIGKILL
    await runner.collect(handle)
    record = json.loads((tmp_path / "session" / "agents" / "resources.jsonl").read_text().splitlines()[0])
    assert record["ended_by"] == "oom_killed"


async def test_the_same_sigkill_without_an_oom_event_is_not_called_oom(tmp_path: Path) -> None:
    cgroup = tmp_path / "cg"
    write_cgroup(cgroup, memory_max="1000000", peak="1", oom=0, cpu_max=None)
    runner = make_runner(tmp_path / "session", cgroup_root=cgroup)
    handle = await runner.launch(make_spec(tmp_path / "w"))
    os.kill(handle.pid, signal.SIGKILL)
    await wait_finished(runner, handle)
    await runner.collect(handle)
    record = json.loads((tmp_path / "session" / "agents" / "resources.jsonl").read_text().splitlines()[0])
    assert record["ended_by"] == "failed"


async def test_cancelled_session_is_recorded_as_cancelled(tmp_path: Path) -> None:
    runner = make_runner(tmp_path / "session")
    handle = await runner.launch(make_spec(tmp_path / "w"))
    await runner.cancel(handle)
    await wait_finished(runner, handle)
    await runner.collect(handle)
    record = json.loads((tmp_path / "session" / "agents" / "resources.jsonl").read_text().splitlines()[0])
    assert record["ended_by"] == "cancelled"


async def test_cpu_only_session_hides_the_gpus_unless_the_caller_pinned_them(tmp_path: Path) -> None:
    ws = tmp_path / "w"
    code = "import json, os, pathlib\npathlib.Path('gpu.json').write_text(json.dumps([os.environ.get('HIP_VISIBLE_DEVICES'), os.environ.get('CUDA_VISIBLE_DEVICES')]))\n"
    runner = make_runner(tmp_path / "session")
    handle = await runner.launch(make_spec(ws, code=code, gpus=0, env={"CUDA_VISIBLE_DEVICES": "3"}))
    await wait_finished(runner, handle)
    await runner.collect(handle)
    assert json.loads((ws / "gpu.json").read_text()) == ["", "3"]


async def test_child_runs_in_its_own_session_with_the_spec_cwd(tmp_path: Path) -> None:
    ws = tmp_path / "w"
    code = "import os, pathlib\npathlib.Path('who.txt').write_text(f'{os.getcwd()} {os.getsid(0) == os.getpid()}')\n"
    runner = make_runner(tmp_path / "session")
    handle = await runner.launch(make_spec(ws, code=code, cwd=ws / "worktree"))
    await wait_finished(runner, handle)
    await runner.collect(handle)
    assert (ws / "worktree" / "who.txt").read_text() == f"{ws / 'worktree'} True"


async def test_argv_is_not_rewritten_beyond_the_mcp_flag(tmp_path: Path) -> None:
    runner = make_runner(tmp_path / "session")
    spec = make_spec(tmp_path / "w", mcp=False)
    handle = await runner.launch(spec)
    try:
        assert tuple(handle.session.proc.args) == child_argv("import time; time.sleep(60)")
    finally:
        await runner.cancel(handle)
        await wait_finished(runner, handle)
        await runner.collect(handle)


async def run_fake_time(clock: FakeClock, handle, seconds: float) -> None:
    """Let virtual time pass while the supervisor tasks run, stopping early if the session ends."""
    target = clock.mono + seconds
    while clock.mono < target and not handle.session.task.done():
        await real_asyncio.sleep(0)


async def stop_and_collect(runner, handle) -> None:
    await runner.cancel(handle)
    await wait_finished(runner, handle)
    await runner.collect(handle)


async def test_a_session_blocked_in_a_long_mcp_call_is_not_lost(tmp_path: Path, clock: FakeClock) -> None:
    """The MCP service refreshes the session's heartbeat while a call made with its token is in flight."""
    beats: list = []
    spec = make_spec(tmp_path / "w", deadline_unix=clock.wall + 6 * HOUR, heartbeat_sink=beats.append)
    spec.heartbeat_path.write_text("{}")
    clock.keep_alive = False
    runner = clocked_runner(tmp_path, clock)
    handle = await runner.launch(spec)
    assert len(beats) == 1  # the sink got the session's beat at launch
    start = clock.mono
    clock.on_tick.append(beats[0])  # the call is in flight: the service beats every tick
    await run_fake_time(clock, handle, 1000.0)
    assert clock.mono - start >= 1000.0  # well past the 300 s window
    status = await runner.poll(handle)
    assert status.state == "running"
    assert not handle.session.outcome  # the reap loop never judged it stale
    await stop_and_collect(runner, handle)


async def test_a_hung_session_with_no_call_in_flight_is_still_lost(tmp_path: Path, clock: FakeClock) -> None:
    beats: list = []
    spec = make_spec(tmp_path / "w", deadline_unix=clock.wall + 6 * HOUR, heartbeat_sink=beats.append)
    spec.heartbeat_path.write_text("{}")
    clock.keep_alive = False
    runner = clocked_runner(tmp_path, clock)
    handle = await runner.launch(spec)  # the sink holds a beat, but nothing calls it
    await run_fake_time(clock, handle, 1000.0)
    status = await runner.poll(handle)
    assert status.state == "lost"
    assert handle.session.outcome["stale_heartbeat"]
    await runner.collect(handle)


async def test_cpu_work_in_the_process_tree_keeps_a_long_shell_command_alive(tmp_path: Path, clock: FakeClock) -> None:
    proc_root = tmp_path / "proc"
    spec = make_spec(tmp_path / "w", deadline_unix=clock.wall + 6 * HOUR)
    spec.heartbeat_path.write_text("{}")
    clock.keep_alive = False
    runner = clocked_runner(tmp_path, clock, proc_root=proc_root)
    handle = await runner.launch(spec)
    ticks = {"n": 0}

    def burn_cpu() -> None:  # a compiler child burning one CPU second per virtual second
        ticks["n"] += 5
        add_proc(proc_root, handle.pid, 1, rss_kb=1000, utime=ticks["n"] * 100, children=[])

    clock.on_tick.append(burn_cpu)
    await run_fake_time(clock, handle, 1000.0)
    status = await runner.poll(handle)
    assert status.state == "running"
    assert not handle.session.outcome
    await stop_and_collect(runner, handle)


async def test_a_tree_that_burns_no_cpu_is_lost_even_though_a_process_exists(tmp_path: Path, clock: FakeClock) -> None:
    proc_root = tmp_path / "proc"
    spec = make_spec(tmp_path / "w", deadline_unix=clock.wall + 6 * HOUR)
    spec.heartbeat_path.write_text("{}")
    clock.keep_alive = False
    runner = clocked_runner(tmp_path, clock, proc_root=proc_root)
    handle = await runner.launch(spec)
    add_proc(proc_root, handle.pid, 1, rss_kb=1000, utime=500, children=[])  # alive, but idle
    await run_fake_time(clock, handle, 1000.0)
    status = await runner.poll(handle)
    assert status.state == "lost"
    await runner.collect(handle)


async def test_beat_after_the_session_is_over_does_nothing(tmp_path: Path) -> None:
    beats: list = []
    runner = make_runner(tmp_path / "session")
    spec = make_spec(tmp_path / "w", heartbeat_sink=beats.append)
    handle = await runner.launch(spec)
    await stop_and_collect(runner, handle)
    before = spec.heartbeat_path.exists() and spec.heartbeat_path.stat().st_mtime
    beats[0]()
    after = spec.heartbeat_path.exists() and spec.heartbeat_path.stat().st_mtime
    assert before == after


async def test_a_failing_heartbeat_sink_does_not_stop_the_session(tmp_path: Path) -> None:
    def broken(beat) -> None:
        raise RuntimeError("sink down")

    runner = make_runner(tmp_path / "session")
    handle = await runner.launch(make_spec(tmp_path / "w", heartbeat_sink=broken))
    assert (await runner.poll(handle)).state == "running"
    await stop_and_collect(runner, handle)


ECHO_LINES = "import sys\nfor line in sys.stdin:\n    print('got ' + line.strip(), flush=True)\n"


async def until_logged(log: Path, text: str, timeout: float = 20.0) -> None:
    end = real_time.time() + timeout
    while text not in (log.read_text() if log.exists() else ""):
        assert real_time.time() < end, f"{text!r} never reached {log}"
        await real_asyncio.sleep(0.02)


async def test_a_session_launched_with_stdin_open_reads_each_message_until_its_stdin_is_closed(tmp_path: Path) -> None:
    ws = tmp_path / "w"
    runner = make_runner(tmp_path / "session")
    handle = await runner.launch(make_spec(ws, code=ECHO_LINES, stdin_open=True))
    await until_logged(ws / "process.log", "got do the task")
    await runner.send(handle, b"the next message\n")
    await until_logged(ws / "process.log", "got the next message")
    assert (await runner.poll(handle)).state == "running"
    await runner.close_stdin(handle)
    status = await wait_finished(runner, handle)
    assert (status.state, status.returncode) == ("exited", 0)
    await runner.collect(handle)


async def test_send_to_a_session_launched_without_stdin_open_is_refused(tmp_path: Path) -> None:
    runner = make_runner(tmp_path / "session")
    handle = await runner.launch(make_spec(tmp_path / "w"))
    try:
        with pytest.raises(AgentConfigError, match="stdin is not open"):
            await runner.send(handle, b"more\n")
    finally:
        await stop_and_collect(runner, handle)


async def test_a_done_file_left_by_an_earlier_session_is_refused_at_launch(tmp_path: Path) -> None:
    ws = tmp_path / "w"
    ws.mkdir()
    (ws / "specialist_done.json").write_text("{}")
    with pytest.raises(AgentConfigError, match="left over from an earlier session"):
        await make_runner(tmp_path / "session").launch(make_spec(ws))


async def test_codex_gets_a_private_codex_home_even_without_mcp_servers(tmp_path: Path) -> None:
    ws = tmp_path / "w"
    code = "import os, pathlib\npathlib.Path('home.txt').write_text(os.environ.get('CODEX_HOME', ''))\n"
    runner = make_runner(tmp_path / "session")
    handle = await runner.launch(make_spec(ws, code=code, backend="codex", mcp=False))
    await wait_finished(runner, handle)
    await runner.collect(handle)
    assert (ws / "home.txt").read_text() == str(ws / ".codex")


async def test_default_token_variable_matches_the_tool_layer() -> None:
    assert McpEndpoint("tools", "http://x").token_env == "HYPERLOOM_SPECIALIST_MCP_TOKEN"


async def test_a_fired_cancel_scope_stops_the_session_and_kills_its_tree(tmp_path: Path) -> None:
    from hyperloom.orchestrator.actions.cancel_channel import CancelScope, use_cancel_scope

    runner = make_runner(tmp_path / "session")
    scope = CancelScope()
    with use_cancel_scope(scope):
        handle = await runner.launch(make_spec(tmp_path / "w"))
        scope.cancel(reason="session_stopped")
        await real_asyncio.wait_for(handle.session.task, 10)
    assert handle.session.proc.poll() is not None  # not orphaned
    status = await runner.poll(handle)
    assert status.state == "exited" and "session_stopped" in status.error
    await runner.collect(handle)
    record = json.loads((tmp_path / "session" / "agents" / "resources.jsonl").read_text().splitlines()[0])
    assert record["ended_by"] == "cancelled"


async def test_cancelling_the_supervising_task_kills_the_child(tmp_path: Path) -> None:
    runner = make_runner(tmp_path / "session")
    handle = await runner.launch(make_spec(tmp_path / "w"))
    handle.session.task.cancel()
    with pytest.raises(real_asyncio.CancelledError):
        await handle.session.task
    assert handle.session.proc.poll() is not None


async def test_a_wall_budget_extension_left_by_an_earlier_run_does_not_move_the_new_deadline(tmp_path: Path) -> None:
    # The registry the runner clears is the one its own imported function sees, whatever else rebound module names.
    registry = agent_runner.clear_wall_budget_extension.__globals__["_WALL_BUDGET_EXTENSIONS"]
    registry["task-1"] = 3600.0
    runner = make_runner(tmp_path / "session")
    handle = await runner.launch(make_spec(tmp_path / "w"))
    assert "task-1" not in registry
    await stop_and_collect(runner, handle)
