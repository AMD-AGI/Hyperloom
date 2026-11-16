# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Contract tests for benching untested specialist proposals automatically."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest


class _Tasks:
    def __init__(self, queued=(), running=()) -> None:
        self._queued, self._running = list(queued), list(running)
        self.created: list[dict[str, Any]] = []
        self.ended: dict[str, str] = {}

    async def queued(self):
        return self._queued

    async def running(self):
        return self._running

    def queued_context_sync(self):
        return self._queued

    def running_context_sync(self):
        return self._running

    def with_key_prefix_sync(self, kind, key_prefix):
        return [
            SimpleNamespace(kind=c["kind"], params=c["params"], state=self.ended.get(c["idempotency_key"], "queued"))
            for c in self.created
            if c["kind"] == kind and c["idempotency_key"].startswith(key_prefix)
        ]

    def end_every_grid(self, state: str) -> None:
        self.ended.update((c["idempotency_key"], state) for c in self.created)

    async def create_or_return_existing(self, spec):
        existing = any(c["idempotency_key"] == spec.idempotency_key for c in self.created)
        self.created.append(vars(spec))
        return SimpleNamespace(task_id=f"t-{len(self.created)}"), existing


def _round(*names: str) -> dict[str, Any]:
    return {
        "task_id": "spec-1",
        "domain": "serving_specialist",
        "gap_canonical_id": "gap.s1",
        "cycle": 0,
        "proposal_set": [{"name": name, "extra_args": f"--{name}", "reason": "why"} for name in names],
    }


def _phase(*, phase="FRAMEWORK_AGENT", rounds=(), queued=(), running=(), frozen=False, paused=False):
    from hyperloom.orchestrator.phases.framework import FrameworkPhase
    from hyperloom.orchestrator.state.shared_state import SharedState

    state = SharedState()
    state.phase = phase
    state.baseline_tput = 100.0
    state.baseline_config_path = "/cfg.yaml"
    state.baseline_benchmark_script = "bench.sh"
    state.specialist_rounds = list(rounds)
    dispatcher = SimpleNamespace(
        admission_frozen=frozen,
        dispatch_paused_for_phase_budget=lambda: paused,
        registry_lanes_ttl=lambda kind: (["benchmark_lane"], 1800),
    )
    coord = SimpleNamespace(shared_state=state, tasks=_Tasks(queued, running), dispatcher=dispatcher)
    coord.phase_framework = FrameworkPhase(coord)
    return coord.phase_framework


def _explores(fp) -> list[dict[str, Any]]:
    return [c for c in fp._coord.tasks.created if c["kind"] == "explore"]


async def test_enqueues_one_explore_grid_from_the_queue():
    fp = _phase(rounds=[_round("enable-x")])
    await fp._maybe_bench_untested_proposals()

    (call,) = _explores(fp)
    assert [v["name"] for v in call["params"]["grid"]] == ["enable-x"]
    assert call["params"]["grid"][0]["provenance"] == "specialist:serving"
    assert call["params"]["source"] == "coordinator_internal"
    assert call["params"]["config_path"] == "/cfg.yaml"
    assert call["params"]["benchmark_script"] == "bench.sh"
    assert call["requires_lanes"] == ["benchmark_lane"] and call["lease_ttl_sec"] > 0


async def test_grid_is_cut_at_the_cap():
    from hyperloom.orchestrator.phases.framework import FrameworkPhase

    fp = _phase(rounds=[_round(*(f"v{i}" for i in range(10)))])
    await fp._maybe_bench_untested_proposals()
    assert len(_explores(fp)[0]["params"]["grid"]) == FrameworkPhase._AUTO_EXPLORE_GRID_CAP


async def test_same_queue_head_maps_to_the_same_idempotency_key():
    fp = _phase(rounds=[_round("enable-x")])
    await fp._maybe_bench_untested_proposals()
    await fp._maybe_bench_untested_proposals()
    keys = {c["idempotency_key"] for c in _explores(fp)}
    assert len(keys) == 1


@pytest.mark.parametrize("ended", ["failed", "cancelled", "succeeded"])
async def test_a_grid_that_ended_without_a_verdict_hands_the_head_to_the_rows_behind_it(ended):
    fp = _phase(rounds=[_round(*(f"v{i}" for i in range(6)))])
    await fp._maybe_bench_untested_proposals()
    fp._coord.tasks.end_every_grid(ended)

    await fp._maybe_bench_untested_proposals()

    first, second = _explores(fp)
    assert [v["name"] for v in first["params"]["grid"]] == ["v0", "v1", "v2", "v3"]
    assert [v["name"] for v in second["params"]["grid"]] == ["v4", "v5"]


async def test_proposals_a_finished_grid_carried_are_not_benched_again():
    fp = _phase(rounds=[_round("a", "b")])
    await fp._maybe_bench_untested_proposals()
    fp._coord.tasks.end_every_grid("failed")

    await fp._maybe_bench_untested_proposals()

    assert len(_explores(fp)) == 1
    assert [row["name"] for row in fp.shared_state.untested_proposal_rows()] == ["a", "b"]


async def test_a_grid_of_an_earlier_cycle_does_not_hold_back_this_cycles_proposals():
    fp = _phase(rounds=[_round("a")])
    await fp._maybe_bench_untested_proposals()
    fp._coord.tasks.end_every_grid("failed")
    state = fp._coord.shared_state
    state.macro_cycle = 1
    state.specialist_rounds = [{**_round("a"), "cycle": 1}]

    await fp._maybe_bench_untested_proposals()

    assert [c["idempotency_key"].split("-")[2] for c in _explores(fp)] == ["c0", "c1"]


async def test_already_benched_fingerprints_are_skipped():
    fp = _phase(rounds=[_round("enable-x")])
    fp._coord.shared_state.explore_search = {"tested": {"fp": {"extra_server_args": "--enable-x", "outcome": "KEEP"}}}
    await fp._maybe_bench_untested_proposals()
    assert not _explores(fp)


async def test_no_enqueue_when_it_must_not_run():
    explore = SimpleNamespace(kind="explore", params={})
    blocked = [
        _phase(rounds=[_round("x")], queued=[explore]),
        _phase(rounds=[_round("x")], running=[explore]),
        _phase(rounds=[_round("x")], phase="ENABLEMENT"),
        _phase(rounds=[_round("x")], frozen=True),
        _phase(rounds=[_round("x")], paused=True),
        _phase(rounds=[]),
    ]
    for fp in blocked:
        await fp._maybe_bench_untested_proposals()
        assert not _explores(fp)
