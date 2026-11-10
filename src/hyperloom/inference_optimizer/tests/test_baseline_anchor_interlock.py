# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Once a cumulative gain is validated against the baseline anchor, a later re-baseline does not move the anchor."""

from __future__ import annotations

from typing import Any

import pytest

from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType
from hyperloom.orchestrator.actions.executors._accuracy_gate import ENABLEMENT_REVALIDATION_REASON
from hyperloom.orchestrator.loop.coordinator import Coordinator
from hyperloom.orchestrator.roles import Backend, MockBackend, ScriptedPlan
from hyperloom.orchestrator.state.task_registry import Task

ANCHOR = 55.23
HIGHER_REBASELINE = 55.43


def _backends() -> dict[str, Backend]:
    idle = ScriptedPlan(
        turns=[],
        default_intent=Intent(type=IntentType.SEND_MESSAGE, payload={"topic": "heartbeat", "body_md": "ok"}),
    )
    return {name: MockBackend(idle, name=name) for name in ("orchestration", "critic")}


@pytest.fixture
async def coord(session_dir):
    c = Coordinator(session_dir, backends=_backends())
    try:
        yield c
    finally:
        await c.stop()


def _baseline_task(task_id: str, **params: Any) -> Task:
    return Task(task_id=task_id, kind="baseline", state="running", params=params, idempotency_key=f"idem-{task_id}")


def _stamp_a_validated_gain(coord: Coordinator) -> None:
    state = coord.shared_state
    state.optimization_stack = [{"action": "explore", "variant_name": "v1", "tput": 60.0}]
    state.current_best = {"action": "explore", "variant_name": "v1", "tput": 60.0}
    state.cumulative_gain_validated = 8.6
    state.cumulative_gain_validated_ts = "2026-10-13T10:00:00+00:00"
    state.cumulative_gain_validated_stack_len = 1


async def _promote_baseline(coord: Coordinator, tput: float, task: Task, **result: Any) -> None:
    await coord.writeback.promote_to_shared_state("baseline", {"output_throughput": tput, **result}, task=task)


async def test_a_higher_re_baseline_does_not_move_an_anchor_a_gain_was_validated_against(coord: Coordinator):
    """The Kimi-K3 session: a re-baseline after a stamped gain moved the anchor 55.23 -> 55.43, and the report then
    classified the stamp as inconsistent."""
    coord.shared_state.baseline_tput = ANCHOR
    _stamp_a_validated_gain(coord)

    await _promote_baseline(coord, HIGHER_REBASELINE, _baseline_task("t-rebaseline"))

    assert coord.shared_state.baseline_tput == pytest.approx(ANCHOR)
    held = coord.shared_state.last_baseline["extras"]["anchor_held"]
    assert held == {"reason": "anchor_load_bearing", "prior": ANCHOR, "measured": HIGHER_REBASELINE}


async def test_a_higher_re_baseline_still_moves_an_anchor_no_gain_was_validated_against(coord: Coordinator):
    coord.shared_state.baseline_tput = ANCHOR

    await _promote_baseline(coord, HIGHER_REBASELINE, _baseline_task("t-rebaseline"))

    assert coord.shared_state.baseline_tput == pytest.approx(HIGHER_REBASELINE)


async def test_the_first_anchor_lands(coord: Coordinator):
    await _promote_baseline(coord, ANCHOR, _baseline_task("t-first"))

    assert coord.shared_state.baseline_tput == pytest.approx(ANCHOR)


async def test_a_revalidation_moves_an_anchor_a_gain_was_validated_against(coord: Coordinator):
    """The enablement patch changed the stack, so the prior anchor describes nothing reproducible."""
    coord.shared_state.baseline_tput = ANCHOR
    _stamp_a_validated_gain(coord)

    await _promote_baseline(
        coord, HIGHER_REBASELINE, _baseline_task("t-revalidate", reason=ENABLEMENT_REVALIDATION_REASON)
    )

    assert coord.shared_state.baseline_tput == pytest.approx(HIGHER_REBASELINE)


async def test_a_hot_pass_corrects_a_cold_anchor_before_any_gain_is_validated(coord: Coordinator):
    coord.shared_state.baseline_tput = 1000.0
    coord.shared_state.baseline_measure_round_dropped = True

    await _promote_baseline(coord, 980.0, _baseline_task("t-hot"), measure_round_runtime_sec=400.0)

    assert coord.shared_state.baseline_tput == pytest.approx(980.0)
