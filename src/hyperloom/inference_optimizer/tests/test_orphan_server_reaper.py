# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the session-scoped orphaned serving-process reaper."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from hyperloom.orchestrator.actions.executors import _server_lifecycle as sl
from hyperloom.orchestrator.actions.executors._server_lifecycle import (
    reap_orphaned_servers,
    reap_recorded_servers,
    stop_agentx_servers,
)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_for_cmdline_marker(pid: int, marker: str, timeout: float = 5.0) -> None:
    """Block until ``/proc/<pid>/cmdline`` contains ``marker``."""
    deadline = time.time() + timeout
    proc_cmdline = Path(f"/proc/{pid}/cmdline")
    while time.time() < deadline:
        try:
            raw = proc_cmdline.read_bytes()
        except OSError:
            raw = b""
        if marker.encode() in raw:
            return
        time.sleep(0.02)
    raise AssertionError(f"pid {pid} cmdline never contained {marker!r}")


def _spawn_marker_process(marker: str) -> subprocess.Popen:
    """Spawn a long-lived child whose argv contains ``marker`` (cmdline match)."""
    proc = subprocess.Popen(
        [sys.executable, "-c", f"import time,sys; _={marker!r}; time.sleep(120)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    _wait_for_cmdline_marker(proc.pid, marker)
    return proc


def _write_pidfile(session_dir: Path, tag: str, pid: int) -> Path:
    run_dir = session_dir / "runs" / "explore" / "v0"
    run_dir.mkdir(parents=True, exist_ok=True)
    pidfile = run_dir / f"{tag}.pid"
    pidfile.write_text(str(pid), encoding="utf-8")
    return pidfile


def _proc_state(pid: int) -> str:
    """The ``/proc/<pid>/stat`` state letter, or ``""`` once the pid is gone."""
    try:
        return Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").rsplit(")", 1)[1].split()[0]
    except (OSError, IndexError):
        return ""


def _gone(pid: int, timeout: float = 5.0) -> bool:
    """Whether ``pid`` exits (or is left a zombie) within ``timeout``."""
    deadline = time.time() + timeout
    while _proc_state(pid) not in ("", "Z"):
        if time.time() >= deadline:
            return False
        time.sleep(0.05)
    return True


def _spawn_zombie_leader(tmp_path: Path, child_marker: str | None) -> tuple[subprocess.Popen, int | None]:
    """A setsid'd leader that exits and is never waited on, optionally leaving a child in its group."""
    info = tmp_path / "zombie_child.json"
    lines = ["import json", "import subprocess", "import sys"]
    if child_marker is not None:
        lines += [
            f"p = subprocess.Popen([sys.executable, '-c', {('import time; _=' + repr(child_marker) + '; time.sleep(120)')!r}])",
            f"open({str(info)!r}, 'w').write(json.dumps({{'child_pid': p.pid}}))",
        ]
    script = tmp_path / "zombie_leader.py"
    script.write_text("\n".join(lines), encoding="utf-8")
    leader = subprocess.Popen(
        [sys.executable, str(script)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    deadline = time.time() + 5.0
    while _proc_state(leader.pid) != "Z":
        if time.time() >= deadline:
            raise AssertionError("leader never exited")
        time.sleep(0.02)
    if child_marker is None:
        return leader, None
    child_pid = int(json.loads(info.read_text(encoding="utf-8"))["child_pid"])
    _wait_for_cmdline_marker(child_pid, child_marker)
    return leader, child_pid


def _spawn_dead_leader_with_live_child(tmp_path: Path, marker: str) -> tuple[int, int]:
    """Return ``(leader_pid, child_pid)`` where child remains in leader's pgid."""
    info = tmp_path / "child.json"
    leader_script = tmp_path / "leader.py"
    leader_script.write_text(
        "\n".join(
            [
                "import json",
                "import subprocess",
                "import sys",
                f"p = subprocess.Popen([sys.executable, '-c', {('import time; _=' + repr(marker) + '; time.sleep(120)')!r}])",
                f"open({str(info)!r}, 'w').write(json.dumps({{'child_pid': p.pid}}))",
                "sys.exit(0)",
            ]
        ),
        encoding="utf-8",
    )
    leader = subprocess.Popen(
        [sys.executable, str(leader_script)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    leader_pid = leader.pid
    leader.wait(timeout=5)
    deadline = time.time() + 5.0
    while time.time() < deadline:
        if info.exists():
            child_pid = int(json.loads(info.read_text(encoding="utf-8"))["child_pid"])
            return leader_pid, child_pid
        time.sleep(0.05)
    raise AssertionError("leader did not write child pid")


@pytest.mark.parametrize(
    ("marker", "tag"),
    [("sglang.launch_server", "sglang_8888"), ("atom.entrypoints.openai_server", "atom_8888")],
    ids=["sglang", "atom"],
)
def test_reap_kills_matching_orphan_and_clears_pidfile(tmp_path, marker, tag):
    """A live server whose cmdline matches is reaped and its pidfile removed."""
    proc = _spawn_marker_process(marker)
    pidfile = _write_pidfile(tmp_path, tag, proc.pid)
    try:
        reaped = reap_orphaned_servers(tmp_path)

        # Reap the zombie so the liveness probe reflects true termination (the reaper is not this process's
        # parent-waiter).
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            # The assertions below report a survivor; finally still cleans it up.
            pass

        assert proc.pid in reaped
        assert not _pid_alive(proc.pid)
        assert not pidfile.exists()
    finally:
        try:
            proc.kill()
        except OSError:
            # The reaper may already have terminated and collected the child.
            pass
        proc.wait(timeout=5)


def test_reap_spares_pid_whose_cmdline_does_not_match(tmp_path):
    """A recycled pid running an unrelated process must NOT be killed."""
    proc = _spawn_marker_process("totally-unrelated-process")
    pidfile = _write_pidfile(tmp_path, "sglang_8888", proc.pid)
    try:
        reaped = reap_orphaned_servers(tmp_path)

        assert proc.pid not in reaped
        assert _pid_alive(proc.pid), "unrelated pid must not be killed"
        # A spared pidfile is left so a later run can re-evaluate it.
        assert pidfile.exists()
    finally:
        try:
            proc.kill()
        except OSError:
            pass
        proc.wait(timeout=5)


def test_reap_no_pidfiles_is_noop(tmp_path):
    """Fresh session (no pidfiles) reaps nothing and never raises."""
    (tmp_path / "runs").mkdir(parents=True, exist_ok=True)
    assert reap_orphaned_servers(tmp_path) == []


def test_reap_dead_pid_clears_stale_pidfile(tmp_path):
    """A pidfile pointing at a dead pid is cleaned up without killing anything."""
    proc = _spawn_marker_process("sglang.launch_server")
    proc.kill()
    proc.wait(timeout=5)
    pidfile = _write_pidfile(tmp_path, "sglang_8888", proc.pid)

    reaped = reap_orphaned_servers(tmp_path)

    assert proc.pid not in reaped
    assert not pidfile.exists()


def test_reap_kills_group_when_recorded_leader_exited_but_child_survives(tmp_path):
    """The real leak shape: setsid leader exits, child remains in its pgid."""
    leader_pid, child_pid = _spawn_dead_leader_with_live_child(tmp_path, "sglang.launch_server")
    pidfile = _write_pidfile(tmp_path, "sglang_8888", leader_pid)
    try:
        reaped = reap_orphaned_servers(tmp_path)

        deadline = time.time() + 5.0
        while time.time() < deadline and _pid_alive(child_pid):
            time.sleep(0.05)

        assert leader_pid in reaped
        assert not _pid_alive(child_pid)
        assert not pidfile.exists()
    finally:
        try:
            os.kill(child_pid, 9)
        except OSError:
            pass


def test_reap_spares_group_when_recorded_leader_exited_but_child_is_unrelated(tmp_path):
    """A recycled pgid must not be killed unless a group member still looks like a server."""
    leader_pid, child_pid = _spawn_dead_leader_with_live_child(tmp_path, "totally-unrelated-process")
    pidfile = _write_pidfile(tmp_path, "sglang_8888", leader_pid)
    try:
        reaped = reap_orphaned_servers(tmp_path)

        assert leader_pid not in reaped
        assert _pid_alive(child_pid), "unrelated process group must not be killed"
        assert pidfile.exists()
    finally:
        try:
            os.kill(child_pid, 9)
        except OSError:
            pass


def test_reap_kills_sglang_ranks_whose_leader_is_an_unreaped_zombie(tmp_path):
    """An init that never waits leaves the exited leader a zombie while its renamed ranks still hold the GPUs."""
    leader, child_pid = _spawn_zombie_leader(tmp_path, "sglang::scheduler_TP0")
    pidfile = _write_pidfile(tmp_path, "agentx_server", leader.pid)
    try:
        reaped = reap_orphaned_servers(tmp_path)

        assert reaped == [leader.pid]
        assert _gone(child_pid)
        assert not pidfile.exists()
    finally:
        try:
            os.kill(child_pid, 9)
        except OSError:
            # The reaper may already have killed it.
            pass
        leader.wait(timeout=5)


def test_reap_treats_a_lone_zombie_leader_as_stale(tmp_path):
    """A zombie takes no signal and holds no memory: only its pidfile is left to clean up."""
    leader, _ = _spawn_zombie_leader(tmp_path, None)
    pidfile = _write_pidfile(tmp_path, "agentx_server", leader.pid)
    try:
        assert reap_orphaned_servers(tmp_path) == []
        assert not pidfile.exists()
    finally:
        leader.wait(timeout=5)


def test_reap_reports_a_group_that_outlives_sigkill_and_keeps_its_pidfile(tmp_path, monkeypatch):
    proc = _spawn_marker_process("sglang.launch_server")
    pidfile = _write_pidfile(tmp_path, "agentx_server", proc.pid)
    monkeypatch.setattr(sl, "_terminate_group", lambda _pgid: False)
    try:
        reap = reap_recorded_servers(tmp_path / "runs")

        assert reap.reaped == [proc.pid]
        assert reap.survivors == [proc.pid]
        # Left for the next reap, and for whoever has to explain why the GPUs are still taken.
        assert pidfile.exists()
    finally:
        proc.kill()
        proc.wait(timeout=5)


def test_stop_agentx_servers_leaves_other_pidfiles_alone(tmp_path):
    """A lifecycle server kept warm for the next round is recorded under its own name and must keep running."""
    client_booted = _spawn_marker_process("sglang.launch_server")
    warm = _spawn_marker_process("vllm.entrypoints.openai.api_server")
    client_pidfile = _write_pidfile(tmp_path, "agentx_server", client_booted.pid)
    warm_pidfile = _write_pidfile(tmp_path, "vllm_8888", warm.pid)
    try:
        assert stop_agentx_servers(tmp_path) == []

        client_booted.wait(timeout=5)
        assert not client_pidfile.exists()
        assert _pid_alive(warm.pid)
        assert warm_pidfile.exists()
    finally:
        for proc in (client_booted, warm):
            try:
                proc.kill()
            except OSError:
                # Either may already have exited.
                pass
            proc.wait(timeout=5)


def test_reap_never_signals_its_own_process_group(tmp_path, monkeypatch):
    """A pidfile naming a member of the reaper's own group would otherwise take the orchestrator down with it."""
    in_our_group = subprocess.Popen(
        [sys.executable, "-c", "import time; _='sglang.launch_server'; time.sleep(120)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    _wait_for_cmdline_marker(in_our_group.pid, "sglang.launch_server")
    pidfile = _write_pidfile(tmp_path, "agentx_server", in_our_group.pid)
    signalled: list[int] = []
    monkeypatch.setattr(sl, "_terminate_group", lambda pgid: signalled.append(pgid) or True)
    try:
        assert reap_orphaned_servers(tmp_path) == []
        assert signalled == []
        assert pidfile.exists()
    finally:
        in_our_group.kill()
        in_our_group.wait(timeout=5)
