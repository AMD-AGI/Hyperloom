# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A server argv the installed parser refuses costs a probe, not a launch, and it terminates by name."""

from __future__ import annotations

import asyncio
from pathlib import Path

from hyperloom.common.bringup import LadderStage
from hyperloom.orchestrator.actions.executors.tests.test_baseline_suite_round import _play, suite  # noqa: F401
from hyperloom.orchestrator.bringup import ARGV_INVALID, is_argv_invalid, load_boot_observation
from hyperloom.orchestrator.rehearsal import STAGE_FAILED, LaunchAttempt

from .test_bringup_round_scenario import _lane, registry  # noqa: F401

_REFUSAL = "argument --attention-backend: invalid choice: 'fa4'"


def test_a_refused_argv_is_recorded_at_the_parse_rung_and_ends_the_run_without_a_repair_round(
    monkeypatch, suite, registry
):
    result, rounds = _play(
        monkeypatch,
        suite,
        LaunchAttempt(outcome=STAGE_FAILED, stage="argv_parse", message=_REFUSAL),
        baseline_double_run=False,
    )

    assert result["status"] == "failed"
    assert result["error_class"] == ARGV_INVALID
    assert _REFUSAL in result["error"]
    assert len(rounds.calls) == 1
    assert not Path(result["workspace"], "server.log").exists()

    loaded = load_boot_observation(result["boot_observation_path"])
    assert loaded.observation is not None, loaded.degraded
    assert loaded.observation.stage_failed == LadderStage.ARGV_PARSE
    assert is_argv_invalid(loaded.observation)
    assert _REFUSAL in loaded.observation.failure_text

    tasks, round_store, _locks = registry
    lane = _lane(suite.session, tasks, round_store, result["error"])
    lane.shared_state.enablement.launch_observation_path = result["boot_observation_path"]

    async def _ticks() -> None:
        for _tick in range(3):
            assert await lane._maybe_enqueue_enablement_specialist() == ""
            assert lane.shared_state.stop_reason == ARGV_INVALID
        assert await round_store.held() is None

    asyncio.run(_ticks())


def test_a_state_recorded_with_argv_repairs_still_loads():
    """Sessions written before the drop repair was removed carry its ledger; it is ignored."""
    from hyperloom.orchestrator.state._shared_state.enablement_round import EnablementRound

    loaded = EnablementRound.from_dict({"argv_repairs": ["0" * 64], "launch_log": "boot failed"})
    assert loaded.launch_log == "boot failed"
    assert not hasattr(loaded, "argv_repairs")
