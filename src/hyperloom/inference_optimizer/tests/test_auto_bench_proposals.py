# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Contract tests for benching untested specialist proposals automatically."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any


class _Tasks:
    def __init__(self, queued=(), running=()) -> None:
        self._queued, self._running = list(queued), list(running)
        self.created: list[dict[str, Any]] = []

    async def queued(self):
        return self._queued

    async def running(self):
        return self._running

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


async def test_dormant_levers_of_several_patches_get_one_bundle_round_per_set():
    """A set is benched once, a grown set again, and one patch's set never: its integrate round measured it."""
    from hyperloom.orchestrator.phases.dispatch_order import LEVER_BUNDLE_REASON

    fp = _phase()
    state = fp._coord.shared_state

    def dormant(switch: str, patch: str) -> None:
        state.record_authored_framework_levers([{"switch": switch}], default_on=False, specialist_task_id=patch)

    dormant("HL_ONE", "t-1")
    assert await fp.enqueue_dormant_lever_bundle() is False
    dormant("HL_TWO", "t-2")
    assert await fp.enqueue_dormant_lever_bundle() is True
    assert await fp.enqueue_dormant_lever_bundle() is False
    dormant("HL_THREE", "t-3")
    assert await fp.enqueue_dormant_lever_bundle() is True

    rounds = _explores(fp)
    assert {c["params"]["reason"] for c in rounds} == {LEVER_BUNDLE_REASON}
    assert [c["params"]["grid"][0]["framework_lever_bundle"] for c in (rounds[0], rounds[-1])] == [
        ["HL_ONE", "HL_TWO"],
        ["HL_ONE", "HL_THREE", "HL_TWO"],
    ]
    assert rounds[0]["params"]["config_path"] == "/cfg.yaml"


async def test_a_phase_on_its_way_out_gets_no_bundle_round():
    """It would be cancelled by the transition the moment it was queued."""
    fp = _phase(frozen=True)
    state = fp._coord.shared_state
    state.record_authored_framework_levers([{"switch": "HL_ONE"}], default_on=False, specialist_task_id="t-1")
    state.record_authored_framework_levers([{"switch": "HL_TWO"}], default_on=False, specialist_task_id="t-2")

    assert await fp.enqueue_dormant_lever_bundle() is False
    assert not _explores(fp)


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


def _awaiting_screen(state, switch: str, patch: str) -> None:
    from hyperloom.orchestrator.actions.executors._integrate_vocabulary import ATTRIBUTION_CYCLE_PENDING

    state.record_authored_framework_levers([{"switch": switch}], default_on=False, specialist_task_id=patch)
    state.record_framework_lever_attribution(switch, gain_pct=None, source=ATTRIBUTION_CYCLE_PENDING)


async def test_levers_awaiting_the_screen_are_batched_one_arm_per_patch():
    from hyperloom.orchestrator.actions.executors.cycle_bench_arm import CYCLE_ARMS_PARAM
    from hyperloom.orchestrator.actions.executors.cycle_bench_arm import CYCLE_BATCH_REASON

    fp = _phase()
    state = fp._coord.shared_state
    state.record_authored_framework_levers([{"switch": "HL_SCREENED"}], default_on=False, specialist_task_id="t-0")
    for switch, patch in (("HL_A", "t-1"), ("HL_B", "t-1"), ("HL_C", "t-2")):
        _awaiting_screen(state, switch, patch)

    assert await fp.enqueue_cycle_batch() is True

    (batch,) = _explores(fp)
    assert batch["params"]["reason"] == CYCLE_BATCH_REASON
    assert "band_pct" not in batch["params"]
    assert batch["params"][CYCLE_ARMS_PARAM] == [
        {"name": "t-1", "envs": {"HL_A": "1", "HL_B": "1"}},
        {"name": "t-2", "envs": {"HL_C": "1"}},
    ]


async def test_a_lone_pending_patch_waits_only_while_the_serving_lanes_have_other_work():
    busy = SimpleNamespace(kind="integrate_patch", params={}, requires_lanes=("server_lifecycle", "benchmark_lane"))
    fp = _phase(queued=[busy])
    _awaiting_screen(fp._coord.shared_state, "HL_A", "t-1")
    assert await fp.enqueue_cycle_batch() is False

    idle = _phase()
    _awaiting_screen(idle._coord.shared_state, "HL_A", "t-1")
    assert await idle.enqueue_cycle_batch() is True


async def test_no_second_batch_while_one_is_queued_or_the_phase_is_leaving():
    from hyperloom.orchestrator.actions.executors.cycle_bench_arm import CYCLE_ARMS_PARAM

    queued = SimpleNamespace(
        kind="explore", params={CYCLE_ARMS_PARAM: [{"name": "t-0", "envs": {}}]}, requires_lanes=()
    )
    for fp in (_phase(queued=[queued]), _phase(frozen=True)):
        for switch, patch in (("HL_A", "t-1"), ("HL_C", "t-2")):
            _awaiting_screen(fp._coord.shared_state, switch, patch)
        assert await fp.enqueue_cycle_batch() is False
        assert _explores(fp) == []


def test_a_cycle_batch_is_priced_per_arm_and_its_two_bases():
    from hyperloom.orchestrator.actions.executors.cycle_bench_arm import CYCLE_ARM_MINUTES
    from hyperloom.orchestrator.loop.time_budget import explore_admission_minutes

    arms = [{"name": "a"}, {"name": "b"}, {"name": "c"}]
    assert explore_admission_minutes(78.0, {"cycle_arms": arms}) == 5 * CYCLE_ARM_MINUTES


async def test_a_phase_on_its_way_out_hands_unscreened_levers_back_to_replay_attribution():
    fp = _phase(frozen=True)
    state = fp._coord.shared_state
    state.record_authored_framework_levers([{"switch": "HL_DONE"}], default_on=False, specialist_task_id="t-0")
    for switch, patch in (("HL_A", "t-1"), ("HL_C", "t-2")):
        _awaiting_screen(state, switch, patch)

    assert await fp.enqueue_cycle_batch() is False

    rows = [(r["switch"], r["attribution_source"]) for r in state.authored_framework_levers]
    assert rows == [("HL_DONE", ""), ("HL_A", "cycle_bench_inconclusive"), ("HL_C", "cycle_bench_inconclusive")]
    assert _explores(fp) == []
