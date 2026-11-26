# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Contract tests, part 2 (harness graph and timeline): the task rows, phase history, timeline events and task
states the graph and the timeline are drawn from are the ones Hyperloom writes.

Each test writes a session artifact with the real producer (see :mod:`.producers`) or compares the Inspector's
names with the producer's constants. A failure means a producer changed what it writes or the names it uses: update
the Inspector reader the failure message names, then this test. A test marked ``xfail(strict=True)`` pins a known
break; it starts failing ("XPASS(strict)") once the Inspector is fixed, and then the marker comes off.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hyperloom.inspector import derive, graph
from hyperloom.inspector.collectors import parse_ts
from hyperloom.inspector.tests import producers
from hyperloom.inspector.tests.producers import contract_message

pytestmark = pytest.mark.inspector_contract


# A measured baseline result, as the baseline executor returns it to its recorder.
_MEASURED = {"status": "succeeded", "output_throughput": 1500.0, "ttft_mean_ms": 100.0, "e2el_mean_ms": 2000.0}
_MEASURED["tpot_mean_ms"] = 10.0


async def test_task_history_written_by_registry_spans_the_task_bar(tmp_path: Path) -> None:
    from hyperloom.inspector.watcher import _task_span

    session = producers.new_session(tmp_path)
    async with producers.coordinator_db(session) as coord:
        task = await producers.run_task(coord, producers.TaskRun(kind="explore"))
    row = producers.watch(session).tasks[0]
    running, ended = parse_ts(task.history[0]["ts"]), parse_ts(task.updated_at)
    assert _task_span(row, ended) == (running, ended), contract_message(
        "TaskRegistry.transition", "watcher.py (_task_span)", row
    )


def test_state_json_written_by_shared_state_drives_phases(tmp_path: Path) -> None:
    from hyperloom.inspector.watcher import _phase_bands

    session = producers.new_session(tmp_path)
    state = producers.new_state(model_name="m")
    producers.record_phase(state, "BASELINE", "start", 1000.0)
    producers.record_phase(state, "EXPLORE", "baseline measured", 2000.0)
    state.agent_last_active["orchestration"] = 1500.0
    state.save(session)
    summary = producers.watch(session).summary_at(0)
    history = summary["phase_history"]
    where = "watcher.py (_phase_bands)"
    bands = [(b["phase"], b["start"], b["end"]) for b in _phase_bands(history, 3000.0)]
    assert bands == [("BASELINE", 1000.0, 2000.0), ("EXPLORE", 2000.0, None)], contract_message(
        "phase_chain.make_history_row / machine_state.record_phase_transition", where, history
    )
    assert summary["agent_last_active"] == {"orchestration": 1500.0}, contract_message(
        "SharedState.agent_last_active", "derive.py (agent_staleness)", summary
    )


def test_every_task_health_is_ranked() -> None:
    from hyperloom.orchestrator.state.task_states import TRANSITIONS

    health = {s: derive.task_health({"state": s}, None, None, None, 900.0)["health"] for s in TRANSITIONS}
    unranked = {s: h for s, h in health.items() if h not in graph._HEALTH_RANK}
    assert not unranked, contract_message("task_states.TRANSITIONS", "derive.py (task_health), graph.py", unranked)


def test_baseline_timeline_event_is_a_timeline_step(tmp_path: Path) -> None:
    session = producers.new_session(tmp_path)
    producers.record_baseline(session, "t-base", _MEASURED)
    steps = producers.watch(session).timeline_view([], float("inf"))["steps"]
    assert [(s["kind"], s["status"]) for s in steps] == [("baseline", "succeeded")] and steps[0]["start"], (
        contract_message("event_timeline.build_envelope / sbd_v6.write_timeline_event_at", "watcher.py", steps)
    )


def test_the_ended_task_states_are_the_orchestrators() -> None:
    from hyperloom.orchestrator.state.task_registry import TERMINAL_STATES

    from hyperloom.inspector.collectors import ENDED_STATES

    assert ENDED_STATES == TERMINAL_STATES
