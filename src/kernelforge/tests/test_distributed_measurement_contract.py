# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What the graph probe concludes about a multi-rank measurement.

The probe asks two things of a collective task: that the declared ranks really
launched, and that every one of them measured inside ``dist_harness``. It does
not re-derive how the run behaved. The harness owns device binding, per-rank
seeding, barrier placement, the cross-rank reduction and the teardown, so a rank
that ran inside it holds those by construction -- and a rank that did not is
refused for that rather than for whichever of them it broke first.
"""

from __future__ import annotations

import json

from kernelforge.loop import task_preparer
from kernelforge.loop.task_preparer import (
    PROBE_DISTRIBUTED_VIOLATION,
    PROBE_FAILED,
    _read_graph_probe_shards,
)


def _shard(tmp_path, name: str, payload: dict | str) -> None:
    body = payload if isinstance(payload, str) else json.dumps(payload)
    (tmp_path / f"probe.{name}").write_text(body, encoding="utf-8")


def _worker(rank: int, **overrides) -> dict:
    """A conforming worker shard for a four-rank run."""
    payload = {
        "replays": 30,
        "rank": str(rank),
        "local_rank": str(rank),
        "world_size": "4",
        "pid": 100 + rank,
        "ppid": 50,
        "ancestors": [50],
        "harness": True,
    }
    payload.update(overrides)
    return payload


def _read(tmp_path, ranks: int | None = 4):
    return _read_graph_probe_shards(str(tmp_path / "probe"), expected_world_size=ranks)


def test_a_conforming_four_rank_run_is_scored_by_its_slowest_worker(tmp_path):
    for rank in range(4):
        _shard(tmp_path, str(rank), _worker(rank, replays=30 if rank else 12))

    replays, reason = _read(tmp_path)

    assert (replays, reason) == (12, "")


def test_a_single_process_run_is_refused_for_a_multi_rank_task(tmp_path):
    """The driver never re-execed under torchrun, so nothing collective ran."""
    _shard(tmp_path, "solo", {"replays": 30, "rank": None, "world_size": None})

    replays, reason = _read(tmp_path)

    assert replays == PROBE_DISTRIBUTED_VIOLATION
    assert "single-process" in reason
    assert "4 ranks" in reason


def test_a_rank_count_that_disagrees_with_the_task_is_refused(tmp_path):
    for rank in range(2):
        _shard(tmp_path, str(rank), _worker(rank, world_size="2", local_rank=str(rank)))

    replays, reason = _read(tmp_path)

    assert replays == PROBE_DISTRIBUTED_VIOLATION
    assert "launched 2 ranks" in reason


def test_a_driver_that_launched_its_own_ranks_is_refused(tmp_path):
    """Reaching the right rank count by hand forfeits every other guarantee.

    Nothing else about such a run is worth reading: the properties the number
    rests on are the harness's, so their absence is the finding, not whichever
    of them the driver happened to break.
    """
    for rank in range(4):
        _shard(tmp_path, str(rank), _worker(rank, harness=False))

    replays, reason = _read(tmp_path)

    assert replays == PROBE_DISTRIBUTED_VIOLATION
    assert "did not measure inside dist_harness" in reason
    assert "dist_harness.run()" in reason


def test_one_rank_outside_the_harness_refuses_the_whole_run(tmp_path):
    """A collective is one measurement; a rank measured elsewhere is not part of it."""
    for rank in range(4):
        _shard(tmp_path, str(rank), _worker(rank, harness=rank != 2))

    replays, reason = _read(tmp_path)

    assert replays == PROBE_DISTRIBUTED_VIOLATION
    assert "[2]" in reason


def test_a_shard_that_never_reported_the_harness_is_refused(tmp_path):
    """Absent is not the same as true, and must not be read as it.

    A shard predating the harness, or written by a process that died before the
    field was set, says nothing about where the measurement happened.
    """
    for rank in range(4):
        shard = _worker(rank)
        del shard["harness"]
        _shard(tmp_path, str(rank), shard)

    replays, _reason = _read(tmp_path)

    assert replays == PROBE_DISTRIBUTED_VIOLATION


def test_a_single_rank_task_is_not_held_to_the_distributed_contract(tmp_path):
    """The same shards, with no rank count declared, are judged on replays only."""
    _shard(tmp_path, "solo", {"replays": 30, "rank": None, "world_size": None})

    assert _read(tmp_path, ranks=None) == (30, "")


def test_a_malformed_shard_is_a_probe_failure_not_a_driver_verdict(tmp_path):
    """The two are different stages and must not be reported as one."""
    _shard(tmp_path, "broken", "{not json")

    replays, reason = _read(tmp_path)

    assert replays == PROBE_FAILED
    assert "invalid graph probe shard" in reason


def test_the_probe_tells_the_driver_and_itself_the_same_rank_count(monkeypatch):
    """One declared number reaches both the launcher and the observer."""
    captured: dict = {}

    async def _fake_create(*args, **kwargs):
        captured.update(kwargs.get("env") or {})
        raise RuntimeError("stop before running")

    monkeypatch.setattr(task_preparer.asyncio, "create_subprocess_exec", _fake_create)

    task_preparer.asyncio.run(task_preparer._count_graph_replays("driver.py", 1, 1, timeout_sec=5, require_ranks=4))

    assert captured["FORGE_NPROC_PER_NODE"] == "4"
    assert captured["GRAPH_PROBE_EXPECT_RANKS"] == "4"


def test_a_single_rank_probe_declares_no_rank_count(monkeypatch):
    """The distributed observation is opt-in, so the single-GPU path is unchanged.

    Both names are cleared first: the probe inherits the caller's environment,
    and a rank count left over from whatever ran before would otherwise decide
    this test's answer.
    """
    captured: dict = {}

    async def _fake_create(*args, **kwargs):
        captured.update(kwargs.get("env") or {})
        raise RuntimeError("stop before running")

    monkeypatch.setattr(task_preparer.asyncio, "create_subprocess_exec", _fake_create)
    monkeypatch.delenv("FORGE_NPROC_PER_NODE", raising=False)
    monkeypatch.delenv("GRAPH_PROBE_EXPECT_RANKS", raising=False)

    task_preparer.asyncio.run(task_preparer._count_graph_replays("driver.py", 1, 1, timeout_sec=5))

    assert "GRAPH_PROBE_EXPECT_RANKS" not in captured
    assert "FORGE_NPROC_PER_NODE" not in captured


def test_the_probe_survives_a_torch_that_carries_no_cuda(tmp_path, monkeypatch):
    """The probe runs in every process on the path, so it may not raise from one.

    A module named torch with no ``cuda`` is a real shape -- a test stub, or a
    package caught mid-import -- and reading the attribute outside the guard
    turned the lazy install into an uncaught AttributeError in whatever process
    happened to import torch next.
    """
    import sys
    import types

    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
    monkeypatch.setenv("GRAPH_PROBE_OUT", str(tmp_path / "probe"))
    namespace: dict = {}

    exec(compile(task_preparer._GRAPH_PROBE_SITECUSTOMIZE, "sitecustomize.py", "exec"), namespace)

    assert namespace["_graph_ready"] is False
    # The hook stays installed and must swallow the same shape on every import.
    assert namespace["_hooked"]("torch") is not None


def test_the_probe_reports_where_a_rank_measured(tmp_path, monkeypatch):
    """The shard's ``harness`` field is what the module under test actually writes.

    Asserting the field by hand everywhere else would let the probe and the
    reader drift apart silently, so the recorded source is executed once.
    """
    namespace: dict = {}
    monkeypatch.setenv("GRAPH_PROBE_EXPECT_RANKS", "4")
    monkeypatch.setenv("GRAPH_PROBE_OUT", str(tmp_path / "probe"))
    exec(compile(task_preparer._GRAPH_PROBE_SITECUSTOMIZE, "sitecustomize.py", "exec"), namespace)

    assert namespace["_harness_measured"]() is False

    class _Harness:
        MEASURED = [1]

    monkeypatch.setitem(__import__("sys").modules, "dist_harness", _Harness)

    assert namespace["_harness_measured"]() is True
