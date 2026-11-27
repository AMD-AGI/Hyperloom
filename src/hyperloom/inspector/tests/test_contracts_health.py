# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Contract tests, part 3 (task health): a task the Inspector shows as healthy is one Hyperloom's own code
considers healthy, and a failed variant's evidence is found where Hyperloom writes it.

Each test feeds the Inspector what the producer emits (the specialist runner's failure reasons, the grid runner's
progress notes and slot names, the outcome vocabulary, the task states) and checks the health the Inspector
derives. A failure means a producer changed what it emits: update the Inspector reader the failure message names,
then this test. A test marked ``xfail(strict=True)`` pins a known break; it starts failing ("XPASS(strict)") once
the Inspector is fixed, and then the marker comes off.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hyperloom.inspector import derive
from hyperloom.inspector.tests import producers
from hyperloom.inspector.tests.producers import contract_message

pytestmark = pytest.mark.inspector_contract


_DERIVE = "derive.py (result_health, task_health)"


_UNHEALTHY = ("crashed", "failed")


# How a specialist subprocess can end badly, as the dispatcher reaps it (SpecialistSubprocessResult fields).
_ENDINGS = {
    "exit_code": {"exit_code": 3},
    "timeout": {"timed_out": True, "error": "killed after 3600s"},
    "stale_heartbeat": {"stale_heartbeat": True, "error": "no heartbeat for 900s"},
    "error": {"error": "claude: rate limited"},
}


def _backend_error(ending: str, done: bool) -> str:
    """``runner._subprocess_backend_error`` for the ending, with or without a ``specialist_done.json`` emitted."""
    payload = {"summary": "s"} if done else None
    return producers.specialist_backend_error(done_payload=payload, **_ENDINGS[ending])


def _inspector_health(runner_status: str, reason: str, round_reason: str | None) -> str:
    # The result envelope inference_optimizer/cli/executors.py (_executor) builds from SpecialistRunResult.
    result = {"runner_status": runner_status, "error": reason}
    return derive.task_health({"state": "succeeded"}, result, round_reason, None, 900.0)["health"]


@pytest.mark.parametrize("ending", sorted(_ENDINGS))
def test_a_specialist_that_died_without_a_done_file_is_unhealthy(ending: str) -> None:
    # SpecialistRunner: no specialist_done.json and a backend error -> runner status "stale" (runner.py, SpecialistRunner._finalize).
    reason = _backend_error(ending, done=False)
    assert reason and producers.producer_says_failed("stale", reason), (
        f"{ending}: the producer no longer calls this a failure"
    )
    health = _inspector_health("stale", reason, reason)
    assert health in _UNHEALTHY, contract_message(
        "runner._subprocess_backend_error", _DERIVE, f"{reason!r} -> {health}"
    )


@pytest.mark.parametrize("ending", ["timeout", "stale_heartbeat", "error"])
def test_a_specialist_that_died_after_its_done_file_is_unhealthy(ending: str) -> None:
    # SpecialistRunner: a specialist_done.json and a backend error -> runner status "partial" (runner.py, SpecialistRunner._finalize).
    reason = _backend_error(ending, done=True)
    assert reason and producers.producer_says_failed("partial", reason), (
        f"{ending}: the producer no longer calls this a failure"
    )
    health = _inspector_health("partial", reason, reason)
    assert health in _UNHEALTHY, contract_message(
        "runner._subprocess_backend_error", _DERIVE, f"{reason!r} -> {health}"
    )


@pytest.mark.parametrize("ending", sorted(_ENDINGS))
def test_the_specialist_round_reason_alone_marks_the_task_unhealthy(ending: str) -> None:
    # No failing runner status: only state.json's specialist_rounds reason says the subprocess failed, so the
    # Inspector's own reading of the reason (derive._subprocess_failed) decides.
    reason = _backend_error(ending, done=False)
    health = derive.task_health({"state": "succeeded"}, None, reason, None, 900.0)["health"]
    assert health in _UNHEALTHY, contract_message(
        "runner._subprocess_backend_error", _DERIVE, f"{reason!r} -> {health}"
    )


async def test_failed_variant_outcome_is_matched_to_its_progress_note(tmp_path: Path) -> None:
    from hyperloom.orchestrator.actions.executors.grid._grid_base import TS_FAILED

    session = producers.new_session(tmp_path)
    notes = producers.variant_progress_notes(["a", "b"], ["succeeded", "failed"], [100.0, None])
    outcome = {"variant_name": "b", "outcome": TS_FAILED, "reason": "server died", "stage": "server_start"}
    async with producers.coordinator_db(session) as coord:
        task = await producers.run_task(coord, producers.TaskRun(kind="explore", progress=notes))
        await producers.announce_result(coord, task, {"status": "succeeded", "per_variant_outcomes": [outcome]})
    row = producers.watch(session).tasks_at(float("inf"))[0]
    failed = [u for u in row["progress"]["units"] if u["status"] == "failed"]
    assert [(u["label"], u.get("reason"), u.get("stage")) for u in failed] == [("b", "server died", "server_start")], (
        contract_message("_grid_base.TS_FAILED / _variant_progress_note", "derive.py (task_progress)", row["progress"])
    )


@pytest.mark.parametrize("name", ["mnbt_8192", "max-num-seqs=64 tp=2"])
async def test_a_failed_variant_s_server_log_is_found(tmp_path: Path, name: str) -> None:
    from hyperloom.inference_optimizer.session.session_paths import runs_dir
    from hyperloom.orchestrator.actions.executors.grid._grid_base import _safe, variant_slot

    session = producers.new_session(tmp_path)
    notes = producers.variant_progress_notes([name], ["failed"], [None])
    async with producers.coordinator_db(session) as coord:
        task = await producers.run_task(coord, producers.TaskRun(kind="explore", progress=notes))
    # explore.py names the slot f"v{idx:02d}_{_safe(gv.name)}" inline; run_grid puts the variant's own slot below it.
    slot = runs_dir(session, "explore", task.task_id) / f"v00_{_safe(name)}"
    log = variant_slot(slot, 0, name) / "suite_run_0" / "server.log"
    log.parent.mkdir(parents=True)
    log.write_text("INFO start\nRuntimeError: HIP out of memory\n", encoding="utf-8")
    units = producers.watch(session).tasks_at(float("inf"))[0]["progress"]["units"]
    assert units and units[0]["error"] == "RuntimeError: HIP out of memory", contract_message(
        "_grid_base._safe / variant_slot", "watcher.py (_unit_error)", units
    )


def test_every_task_state_has_a_health() -> None:
    from hyperloom.orchestrator.state.task_states import TRANSITIONS

    health = {s: derive.task_health({"state": s}, None, None, None, 900.0)["health"] for s in TRANSITIONS}
    assert health["succeeded"] == "ok", contract_message("task_states.TRANSITIONS", "derive.py (task_health)", health)
