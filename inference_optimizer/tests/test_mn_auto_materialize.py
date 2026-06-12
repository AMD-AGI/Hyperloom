# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for ``Coordinator._maybe_materialize_mn_explore``.

The multi-node bridge that turns a specialist ``proposal_set`` into a benchmarked
``explore`` task. Single-node must be a strict no-op (the Orchestration LLM drives
explore there); multi-node must deterministically enqueue an explore grid built
from the proposals. We invoke the method unbound on a light fake ``self`` so no
real Coordinator / cluster is needed.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from inference_optimizer.orchestrator.action_executors import (
    _multi_node_env as mne,
)
from inference_optimizer.orchestrator.coordinator import Coordinator


class _FakeTasks:
    def __init__(self):
        self.calls = []

    async def create_or_return_existing(self, *, kind, params, idempotency_key):
        self.calls.append(
            {"kind": kind, "params": params, "idempotency_key": idempotency_key}
        )
        return SimpleNamespace(task_id="explore-task-1"), False


def _fake_self(**state_overrides):
    state = SimpleNamespace(
        baseline_config_path="/cfg.yaml",
        current_best={"extra_server_args": "--base-arg 1"},
        baseline_tput=123.0,
        last_baseline={"benchmark_script": "bench.sh"},
    )
    for k, v in state_overrides.items():
        setattr(state, k, v)
    return SimpleNamespace(
        _MN_AUTO_EXPLORE_GRID_CAP=Coordinator._MN_AUTO_EXPLORE_GRID_CAP,
        shared_state=state,
        tasks=_FakeTasks(),
    )


def _task(task_id="task-abcdef1234"):
    return SimpleNamespace(task_id=task_id, params={})


def _run(self_obj, *, domain, proposals, task=None):
    asyncio.run(
        Coordinator._maybe_materialize_mn_explore(
            self_obj, task=task or _task(), domain=domain, proposals=proposals,
        )
    )


def test_single_node_is_strict_noop(monkeypatch):
    monkeypatch.setattr(mne, "is_multi_node", lambda: False)
    s = _fake_self()
    _run(s, domain="moe", proposals=[{"name": "v1", "extra_args": "--x"}])
    assert s.tasks.calls == []  # never enqueues on single-node


def test_empty_proposals_noop(monkeypatch):
    monkeypatch.setattr(mne, "is_multi_node", lambda: True)
    s = _fake_self()
    _run(s, domain="moe", proposals=[])
    assert s.tasks.calls == []


def test_proposals_with_no_args_or_envs_are_dropped(monkeypatch):
    # Research-only proposals (no server-arg, no env) have nothing for restart to
    # apply -> dropped. If all are dropped, no explore task is enqueued.
    monkeypatch.setattr(mne, "is_multi_node", lambda: True)
    s = _fake_self()
    _run(s, domain="moe", proposals=[
        {"name": "research-only", "reason": "investigate later"},
        "not-a-dict",  # non-dict entries are skipped
    ])
    assert s.tasks.calls == []


def test_multi_node_builds_explore_grid(monkeypatch):
    monkeypatch.setattr(mne, "is_multi_node", lambda: True)
    s = _fake_self()
    proposals = [
        {"name": "arg-variant", "extra_args": "--enable-foo", "reason": "r1"},
        {"name": "env-variant", "extra_envs": {"MORI_DISPATCH": "2"}},
        {"extra_args": "--no-name"},  # name falls back to domain-task-idx
        {"name": "drop-me", "reason": "no args/envs"},  # dropped
    ]
    _run(s, domain="moe", proposals=proposals, task=_task("task-abcdef1234"))

    assert len(s.tasks.calls) == 1
    call = s.tasks.calls[0]
    assert call["kind"] == "explore"
    assert call["idempotency_key"] == "mn-auto-explore-task-abcdef1234"
    params = call["params"]
    assert params["source"] == "coordinator_internal_mn"
    assert params["reason"] == "mn_auto_materialize:moe"
    grid = params["grid"]
    # 3 applicable (research-only dropped); each carries provenance.
    assert len(grid) == 3
    names = [g["name"] for g in grid]
    assert "arg-variant" in names and "env-variant" in names
    # The unnamed variant gets a deterministic fallback name.
    assert any(n.startswith("moe-task-abc") for n in names)
    env_row = next(g for g in grid if g["name"] == "env-variant")
    assert env_row["extra_envs"] == {"MORI_DISPATCH": "2"}
    assert all(g["provenance"] == "specialist:moe" for g in grid)
    # Baseline context threaded through from shared_state.
    assert params["config_path"] == "/cfg.yaml"
    assert params["base_extra_args"] == "--base-arg 1"
    assert params["base_tput"] == 123.0
    assert params["benchmark_script"] == "bench.sh"


def test_grid_capped_at_grid_cap(monkeypatch):
    monkeypatch.setattr(mne, "is_multi_node", lambda: True)
    s = _fake_self()
    proposals = [
        {"name": f"v{i}", "extra_args": f"--flag {i}"} for i in range(20)
    ]
    _run(s, domain="params", proposals=proposals)
    grid = s.tasks.calls[0]["params"]["grid"]
    assert len(grid) == Coordinator._MN_AUTO_EXPLORE_GRID_CAP


def test_string_extra_envs_ignored(monkeypatch):
    # extra_envs must be a dict; a non-dict is coerced to {} (and the variant is
    # then dropped if it also has no args).
    monkeypatch.setattr(mne, "is_multi_node", lambda: True)
    s = _fake_self()
    _run(s, domain="moe", proposals=[{"name": "v", "extra_envs": "MORI=1"}])
    assert s.tasks.calls == []


def test_enqueue_failure_is_swallowed(monkeypatch):
    # Bookkeeping must never be blocked by an enqueue error.
    monkeypatch.setattr(mne, "is_multi_node", lambda: True)
    s = _fake_self()

    async def _boom(**kwargs):
        raise RuntimeError("queue down")

    s.tasks.create_or_return_existing = _boom
    # Must not raise.
    _run(s, domain="moe", proposals=[{"name": "v", "extra_args": "--x"}])
