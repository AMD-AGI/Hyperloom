# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the automatic bench of untested specialist proposals."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest


class _FakeTasks:
    def __init__(self, queued=(), running=()):
        self._queued = list(queued)
        self._running = list(running)
        self.calls: list[dict[str, Any]] = []

    async def queued(self):
        return list(self._queued)

    async def running(self):
        return list(self._running)

    async def create_or_return_existing(self, *, kind, params, idempotency_key, **kwargs):
        self.calls.append({"kind": kind, "params": dict(params), "idempotency_key": idempotency_key, **kwargs})
        return SimpleNamespace(task_id="auto-explore-1"), False


def _build_fake(
    *,
    phase="FRAMEWORK_AGENT",
    proposals=None,
    queued=(),
    running=(),
    budget_paused=False,
    admit_frozen=False,
    baseline_tput=100.0,
):
    from hyperloom.orchestrator.phases.framework import FrameworkPhase
    from hyperloom.orchestrator.state.shared_state import SharedState

    state = SharedState()
    state.phase = phase
    state.framework_agent_phase_done = False
    state.baseline_tput = baseline_tput
    state.baseline_config_path = "/cfg.yaml"
    state.last_baseline = {"benchmark_script": "bench.sh"}
    state.macro_cycle = 0
    if proposals is not None:
        state.specialist_rounds = proposals

    tasks = _FakeTasks(queued=queued, running=running)

    fp = FrameworkPhase.__new__(FrameworkPhase)
    fp.shared_state = state
    fp.tasks = tasks
    fp.session_dir = None
    fp.dispatcher = SimpleNamespace(
        _admit_frozen=admit_frozen,
        _dispatch_paused_for_phase_budget=lambda: budget_paused,
    )
    fp._registry_lanes_ttl = lambda kind: ([], 1800)

    return fp, tasks


def _proposal_row(name, extra_args="--flag 1", extra_envs=None):
    return {
        "task_id": "t-1",
        "domain": "serving_specialist",
        "gap_canonical_id": "gap.s1",
        "cycle": 0,
        "proposal_set": [
            {
                "name": name,
                "extra_args": extra_args,
                "extra_envs": extra_envs or {},
                "reason": "test",
                "atomic": False,
                "args_mode": "append",
                "remove_args": [],
                "unset_envs": [],
            }
        ],
    }


def test_auto_bench_enqueues_explore_when_idle(tmp_path):
    """When the proposal queue is non-empty and no explore is running, the pump enqueues one."""
    fp, tasks = _build_fake(
        proposals=[_proposal_row("v1", "--enable-x")],
    )

    asyncio.run(fp._maybe_bench_untested_proposals())

    assert any(c["kind"] == "explore" for c in tasks.calls), "expected an explore to be enqueued"
    call = next(c for c in tasks.calls if c["kind"] == "explore")
    grid = call["params"]["grid"]
    assert len(grid) == 1
    assert grid[0]["name"] == "v1"
    assert call["params"]["source"] == "coordinator_internal"
    assert call["lease_ttl_sec"] > 0


def test_auto_bench_noop_when_explore_is_running(tmp_path):
    """No enqueue when an explore task is already running."""
    running_explore = SimpleNamespace(kind="explore", task_id="e-1", params={})
    fp, tasks = _build_fake(
        proposals=[_proposal_row("v1", "--enable-x")],
        running=[running_explore],
    )

    asyncio.run(fp._maybe_bench_untested_proposals())

    assert not any(c["kind"] == "explore" for c in tasks.calls)


def test_auto_bench_noop_when_explore_is_queued(tmp_path):
    """No enqueue when an explore task is queued."""
    queued_explore = SimpleNamespace(kind="explore", task_id="e-2", params={})
    fp, tasks = _build_fake(
        proposals=[_proposal_row("v1", "--enable-x")],
        queued=[queued_explore],
    )

    asyncio.run(fp._maybe_bench_untested_proposals())

    assert not any(c["kind"] == "explore" for c in tasks.calls)


def test_auto_bench_noop_outside_framework_agent(tmp_path):
    """Outside FRAMEWORK_AGENT the method is a no-op."""
    fp, tasks = _build_fake(
        phase="ENABLEMENT",
        proposals=[_proposal_row("v1", "--enable-x")],
    )

    asyncio.run(fp._maybe_bench_untested_proposals())

    assert not tasks.calls


def test_auto_bench_noop_when_admit_frozen(tmp_path):
    """No enqueue while a phase transition is pending."""
    fp, tasks = _build_fake(
        proposals=[_proposal_row("v1", "--enable-x")],
        admit_frozen=True,
    )

    asyncio.run(fp._maybe_bench_untested_proposals())

    assert not tasks.calls


def test_auto_bench_noop_when_queue_empty(tmp_path):
    """No enqueue when the proposal queue is empty."""
    fp, tasks = _build_fake(proposals=[])

    asyncio.run(fp._maybe_bench_untested_proposals())

    assert not tasks.calls


def test_auto_bench_cuts_at_grid_cap(tmp_path):
    """The auto bench grid is cut at _AUTO_EXPLORE_GRID_CAP (4)."""
    from hyperloom.orchestrator.phases.framework import FrameworkPhase

    proposals = [_proposal_row(f"v{i}", f"--flag {i}") for i in range(10)]
    fp, tasks = _build_fake(proposals=proposals)

    asyncio.run(fp._maybe_bench_untested_proposals())

    call = next((c for c in tasks.calls if c["kind"] == "explore"), None)
    assert call is not None
    assert len(call["params"]["grid"]) == FrameworkPhase._AUTO_EXPLORE_GRID_CAP


def test_auto_bench_idempotency_key_derived_from_fingerprints(tmp_path):
    """The idempotency key is stable for the same set of proposals."""
    fp, tasks = _build_fake(
        proposals=[_proposal_row("v1", "--enable-x")],
    )

    asyncio.run(fp._maybe_bench_untested_proposals())
    asyncio.run(fp._maybe_bench_untested_proposals())

    keys = [c["idempotency_key"] for c in tasks.calls if c["kind"] == "explore"]
    assert len(set(keys)) == 1, "idempotency key must be stable across calls"


def test_auto_bench_tested_fingerprints_excluded(tmp_path):
    """Proposals whose fingerprint is already in explore_search.tested are excluded."""
    from hyperloom.orchestrator.actions.executors._proposal_identity import (
        controls_of,
        effective_fingerprint,
        normalize_proposal,
    )

    extra_args = "--enable-x"
    proposal = _proposal_row("v1", extra_args)["proposal_set"][0]
    fields = normalize_proposal(proposal)
    fp_str = effective_fingerprint(fields["extra_args"], fields["extra_envs"], controls=controls_of(fields))

    fp, tasks = _build_fake(proposals=[_proposal_row("v1", extra_args)])
    # tested dict: key is the fingerprint, value has extra_server_args (explore-round format).
    fp.shared_state.explore_search = {
        "tested": {
            fp_str: {
                "fingerprint": fp_str,
                "extra_server_args": extra_args,
                "extra_envs": {},
                "outcome": "KEEP",
            }
        }
    }

    asyncio.run(fp._maybe_bench_untested_proposals())

    assert not any(c["kind"] == "explore" for c in tasks.calls), "already-tested fingerprint must be excluded"
