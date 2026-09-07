# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What the graph probe concludes about a multi-rank measurement.

A collective task is only worth an end-to-end validation if the micro number it
produced describes the collective it claims to. These checks cover the failure
modes that still print a plausible speedup: the ranks never launched, two of
them shared a device, the per-rank times were averaged instead of maxed, or the
process group was left standing. Each is judged from what the run reported, so
a driver written in any style passes as long as it actually did the work.
"""

from __future__ import annotations

import json

import pytest

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
        "device": rank,
        "dist_live": False,
        # The spelling torch actually reports for a ReduceOp at runtime.
        "reduce_ops": ["RedOpType.MAX", "RedOpType.MIN"],
        "gathers": 0,
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


def test_two_ranks_sharing_one_device_are_refused(tmp_path):
    """Intra-device copies are not the multi-GPU path the speedup claims."""
    for rank in range(4):
        _shard(tmp_path, str(rank), _worker(rank, device=0 if rank < 2 else rank))

    replays, reason = _read(tmp_path)

    assert replays == PROBE_DISTRIBUTED_VIOLATION
    assert "share GPU" in reason


@pytest.mark.parametrize("sum_op", ["RedOpType.SUM", "ReduceOp.SUM"])
def test_an_averaging_reduction_is_refused(tmp_path, sum_op):
    """A mean over ranks hides the laggard that bounds the collective.

    Both spellings are accepted as input because the probe records whatever
    ``str()`` gives it, and that has differed across torch versions.
    """
    for rank in range(4):
        _shard(tmp_path, str(rank), _worker(rank, reduce_ops=[sum_op]))

    replays, reason = _read(tmp_path)

    assert replays == PROBE_DISTRIBUTED_VIOLATION
    assert "slowest" in reason


def test_gathering_every_rank_time_is_accepted_instead_of_max(tmp_path):
    """Taking the max in Python off a gather is the same guarantee."""
    for rank in range(4):
        _shard(tmp_path, str(rank), _worker(rank, reduce_ops=["RedOpType.SUM"], gathers=5))

    replays, reason = _read(tmp_path)

    assert (replays, reason) == (30, "")


def test_a_run_that_reduced_nothing_is_not_second_guessed(tmp_path):
    """No observed reduction is no evidence, and must not become a rejection.

    Code that bound ``all_reduce`` before the probe patched it reports an empty
    op set; rejecting on that would fail drivers for how they imported, not for
    what they measured.
    """
    for rank in range(4):
        _shard(tmp_path, str(rank), _worker(rank, reduce_ops=[]))

    replays, reason = _read(tmp_path)

    assert (replays, reason) == (30, "")


def test_a_rank_that_leaves_its_process_group_standing_is_refused(tmp_path):
    for rank in range(4):
        _shard(tmp_path, str(rank), _worker(rank, dist_live=rank == 2))

    replays, reason = _read(tmp_path)

    assert replays == PROBE_DISTRIBUTED_VIOLATION
    assert "still initialized" in reason


def test_unreported_devices_do_not_fail_a_run_that_is_otherwise_sound(tmp_path):
    """An unobservable property must not become a rejection."""
    for rank in range(4):
        _shard(tmp_path, str(rank), _worker(rank, device=None))

    replays, reason = _read(tmp_path)

    assert (replays, reason) == (30, "")


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


@pytest.mark.parametrize("declared", [2, 4, 8])
def test_the_probe_tells_the_driver_and_itself_the_same_rank_count(monkeypatch, declared):
    """One declared number reaches both the launcher and the observer."""
    captured: dict = {}

    async def _fake_create(*args, **kwargs):
        captured.update(kwargs.get("env") or {})
        raise RuntimeError("stop before running")

    monkeypatch.setattr(task_preparer.asyncio, "create_subprocess_exec", _fake_create)

    task_preparer.asyncio.run(
        task_preparer._count_graph_replays("driver.py", 1, 1, timeout_sec=5, require_ranks=declared)
    )

    assert captured["FORGE_NPROC_PER_NODE"] == str(declared)
    assert captured["GRAPH_PROBE_EXPECT_RANKS"] == str(declared)


def test_a_single_rank_probe_leaves_torch_distributed_alone(monkeypatch):
    """The added observation is opt-in, so the single-GPU path is unchanged."""
    captured: dict = {}

    async def _fake_create(*args, **kwargs):
        captured.update(kwargs.get("env") or {})
        raise RuntimeError("stop before running")

    monkeypatch.setattr(task_preparer.asyncio, "create_subprocess_exec", _fake_create)

    task_preparer.asyncio.run(task_preparer._count_graph_replays("driver.py", 1, 1, timeout_sec=5))

    assert "GRAPH_PROBE_EXPECT_RANKS" not in captured
    assert "FORGE_NPROC_PER_NODE" not in captured
