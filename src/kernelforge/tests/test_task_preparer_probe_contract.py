"""Tests for the graph-replay probe and profile-contract subprocess helpers.

These two async helpers (`_count_graph_replays` / `_check_profile_contract`)
shell out to a child driver with ``start_new_session=True`` and a wall-clock
timeout, reaping the whole process group via ``_kill_process_group`` when the
child overruns. None of the normal unit tests spawn a real driver, so the
success, timeout, and cancellation branches were entirely uncovered. We drive
them here with a fake subprocess and a patched ``wait_for`` so no real process
is launched and the timeout path is exercised deterministically.
"""

from __future__ import annotations

import asyncio
import json
import os

from kernelforge.loop import task_preparer


class _FakeProc:
    """Minimal stand-in for an asyncio subprocess."""

    def __init__(self, *, out=b"", err=b"", returncode=0):
        self.pid = 4321
        self._out = out
        self._err = err
        self.returncode = returncode
        self.killed = False

    async def communicate(self):
        return self._out, self._err

    async def wait(self):
        return self.returncode

    def kill(self):
        self.killed = True


def _patch_spawn(monkeypatch, proc):
    async def _fake_create(*args, **kwargs):
        return proc

    monkeypatch.setattr(task_preparer.asyncio, "create_subprocess_exec", _fake_create)


def _run(coro):
    return asyncio.run(coro)


def _patch_probe_shards(monkeypatch, tmp_path, payloads):
    """Seed the graph probe with the supplied shard payloads."""

    def _fake_mkstemp(prefix=""):
        """Create one fake probe output path and its shard files."""
        target = tmp_path / f"{prefix}out"
        target.write_text("")
        for pid, payload in enumerate(payloads, start=100):
            content = payload if isinstance(payload, str) else json.dumps(payload)
            (tmp_path / f"{prefix}out.{pid}").write_text(content)
        fd = os.open(target, os.O_RDONLY)
        return fd, str(target)

    monkeypatch.setattr(task_preparer.tempfile, "mkstemp", _fake_mkstemp)


# ---------------------------------------------------------------------------
# _count_graph_replays
# ---------------------------------------------------------------------------


def test_graph_probe_sitecustomize_records_rank_identity():
    """Every structured shard identifies its distributed rank context."""
    source = task_preparer._GRAPH_PROBE_SITECUSTOMIZE
    assert '"rank": os.environ.get("RANK")' in source
    assert '"world_size": os.environ.get("WORLD_SIZE")' in source
    assert "else _ancestor_pids()" in source


def test_count_graph_replays_reads_replay_file(monkeypatch, tmp_path):
    """A non-distributed process reports its own replay count."""
    proc = _FakeProc(out=b"stdout-tail", err=b"stderr-tail")
    _patch_spawn(monkeypatch, proc)
    _patch_probe_shards(
        monkeypatch,
        tmp_path,
        [{"replays": 42, "rank": None, "world_size": None}],
    )

    replays, tail = _run(task_preparer._count_graph_replays("driver.py", 1, 1, timeout_sec=5))
    assert replays == 42
    assert "stdout-tail" in tail and "stderr-tail" in tail


def test_count_graph_replays_accepts_legacy_integer_shard(monkeypatch, tmp_path):
    """A legacy single-process integer shard remains readable."""
    proc = _FakeProc()
    _patch_spawn(monkeypatch, proc)
    _patch_probe_shards(monkeypatch, tmp_path, ["17"])

    replays, _ = _run(task_preparer._count_graph_replays("driver.py", 1, 1, timeout_sec=5))
    assert replays == 17


def test_count_graph_replays_uses_minimum_complete_rank_count(
    monkeypatch,
    tmp_path,
):
    """A complete rank set is scored by its least replayed worker.

    The launcher parent is unranked and must not lower the worker minimum.
    """
    proc = _FakeProc(out=b"", err=b"")
    _patch_spawn(monkeypatch, proc)
    _patch_probe_shards(
        monkeypatch,
        tmp_path,
        [
            {"replays": 0, "rank": None, "world_size": None},
            {"replays": 30, "rank": "0", "world_size": "4"},
            {"replays": 30, "rank": "1", "world_size": "4"},
            {"replays": 5, "rank": "2", "world_size": "4"},
            {"replays": 30, "rank": "3", "world_size": "4"},
        ],
    )

    replays, _ = _run(task_preparer._count_graph_replays("driver.py", 1, 10, timeout_sec=5))
    assert replays == 5


def test_count_graph_replays_ignores_ranked_helper_process(monkeypatch, tmp_path):
    """Only the root worker shard represents one distributed rank."""
    proc = _FakeProc()
    _patch_spawn(monkeypatch, proc)
    _patch_probe_shards(
        monkeypatch,
        tmp_path,
        [
            {
                "replays": 5,
                "rank": "0",
                "world_size": "2",
                "pid": 100,
                "ppid": 50,
            },
            {
                "replays": 40,
                "rank": "0",
                "world_size": "2",
                "pid": 101,
                "ppid": 300,
                "ancestors": [300, 100, 50],
            },
            {
                "replays": 30,
                "rank": "1",
                "world_size": "2",
                "pid": 200,
                "ppid": 50,
            },
        ],
    )

    replays, _ = _run(task_preparer._count_graph_replays("driver.py", 1, 10, timeout_sec=5))
    assert replays == 5


def test_count_graph_replays_rejects_incomplete_rank_set(monkeypatch, tmp_path):
    """An unranked launcher shard cannot substitute for a missing worker."""
    proc = _FakeProc()
    _patch_spawn(monkeypatch, proc)
    _patch_probe_shards(
        monkeypatch,
        tmp_path,
        [
            {"replays": 0, "rank": None, "world_size": None},
            {"replays": 30, "rank": "1", "world_size": "4"},
            {"replays": 30, "rank": "2", "world_size": "4"},
            {"replays": 30, "rank": "3", "world_size": "4"},
        ],
    )

    replays, tail = _run(task_preparer._count_graph_replays("driver.py", 1, 10, timeout_sec=5))
    assert replays == -1
    assert "missing ranks: [0]" in tail


def test_count_graph_replays_nonzero_exit_fails(monkeypatch):
    """Replay shards cannot make a crashing benchmark pass."""
    proc = _FakeProc(out=b"partial output", err=b"worker failed", returncode=3)
    _patch_spawn(monkeypatch, proc)

    replays, tail = _run(task_preparer._count_graph_replays("driver.py", 1, 1, timeout_sec=5))
    assert replays == -1
    assert "benchmark exited 3" in tail
    assert "worker failed" in tail


def test_count_graph_replays_timeout_reaps_group(monkeypatch):
    proc = _FakeProc()
    _patch_spawn(monkeypatch, proc)

    killed = {"called": False}

    def _fake_kill(p):
        killed["called"] = True

    monkeypatch.setattr(task_preparer, "_kill_process_group", _fake_kill)

    async def _fake_wait_for(awaitable, timeout):
        # Close the coroutine we were handed, then simulate the timeout.
        if asyncio.iscoroutine(awaitable):
            awaitable.close()
        raise asyncio.TimeoutError

    monkeypatch.setattr(task_preparer.asyncio, "wait_for", _fake_wait_for)

    replays, tail = _run(task_preparer._count_graph_replays("driver.py", 1, 1, timeout_sec=1))
    assert replays == -1
    assert "timed out" in tail
    assert killed["called"] is True


def test_count_graph_replays_spawn_error_returns_minus_one(monkeypatch):
    async def _boom(*a, **k):
        raise OSError("cannot spawn")

    monkeypatch.setattr(task_preparer.asyncio, "create_subprocess_exec", _boom)

    replays, tail = _run(task_preparer._count_graph_replays("driver.py", 1, 1, timeout_sec=1))
    assert replays == -1
    assert "OSError" in tail


# ---------------------------------------------------------------------------
# _check_profile_contract
# ---------------------------------------------------------------------------


def test_profile_contract_success(monkeypatch):
    proc = _FakeProc(out=b"ok", err=b"", returncode=0)
    _patch_spawn(monkeypatch, proc)

    ok, detail = _run(task_preparer._check_profile_contract("driver.py", timeout_sec=5))
    assert ok is True
    assert detail == "verified"


def test_profile_contract_nonzero_exit(monkeypatch):
    proc = _FakeProc(out=b"", err=b"boom", returncode=3)
    _patch_spawn(monkeypatch, proc)

    ok, msg = _run(task_preparer._check_profile_contract("driver.py", timeout_sec=5))
    assert ok is False
    assert "exited 3" in msg


def test_profile_contract_timeout_reaps_group(monkeypatch):
    proc = _FakeProc()
    _patch_spawn(monkeypatch, proc)

    killed = {"called": False}
    monkeypatch.setattr(
        task_preparer,
        "_kill_process_group",
        lambda p: killed.__setitem__("called", True),
    )

    async def _fake_wait_for(awaitable, timeout):
        if asyncio.iscoroutine(awaitable):
            awaitable.close()
        raise asyncio.TimeoutError

    monkeypatch.setattr(task_preparer.asyncio, "wait_for", _fake_wait_for)

    ok, msg = _run(task_preparer._check_profile_contract("driver.py", timeout_sec=1))
    assert ok is False
    assert "timed out" in msg
    assert killed["called"] is True
