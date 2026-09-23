# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The loop driven end to end over mock backends."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType
from hyperloom.inference_optimizer.session.paths import make_session_dir
from hyperloom.orchestrator.phases import machine_state as ps


def _coordinator(session_dir: Path):
    from hyperloom.orchestrator.loop.coordinator import Coordinator
    from hyperloom.orchestrator.roles import (
        MockBackend,
        MockCriticBackend,
        ScriptedPlan,
    )

    # Orchestration says nothing: what the phases do on their own is the point.
    silent = ScriptedPlan(
        turns=[],
        default_intent=Intent(
            type=IntentType.SEND_MESSAGE,
            payload={"topic": "heartbeat", "body_md": "ok"},
        ),
    )
    return Coordinator(
        session_dir,
        backends={
            "orchestration": MockBackend(silent, name="orch"),
            "critic": MockCriticBackend(),
        },
    )


@pytest.fixture
def session_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    return make_session_dir()


def _chain(state: Any) -> list[tuple[str, str, str]]:
    return [
        (str(row.get("from_phase") or ""), str(row.get("to_phase") or ""), str(row.get("reason") or ""))
        for row in state.phase_history
    ]


@pytest.mark.asyncio
async def test_a_baseline_carries_the_run_into_the_optimisation_phase_with_work(session_dir: Path):
    """Entering the phase must produce dispatches, not just a history row."""
    coord = _coordinator(session_dir)
    try:
        coord.shared_state.baseline_tput = 1500.0
        coord.shared_state.max_minutes = 180
        coord.shared_state.save(session_dir)

        for tick in range(1, 25):
            await coord.tick(tick)

        state = coord.shared_state
        assert state.phase == ps.PHASE_FRAMEWORK_AGENT
        assert state.phase_budget_pct[ps.PHASE_FRAMEWORK_AGENT] > 0.0
        queued = await coord.tasks.queued()
        running = await coord.tasks.running()
        kinds = {str((t.params or {}).get("task_kind") or "") for t in (*queued, *running)}
        assert kinds & {"candidate_discovery", "framework_local_explore"}
    finally:
        await coord.stop()


@pytest.mark.asyncio
async def test_both_arms_dry_walks_the_rest_of_the_chain(session_dir: Path, monkeypatch: pytest.MonkeyPatch):
    """With nothing left to try, the run reaches CLOSE through every phase."""
    from hyperloom.orchestrator.kernel import request_handlers
    from hyperloom.orchestrator.state.attempt_ledger import record_config_attempt

    def _no_geak_runner(tool_name: str) -> Path:
        raise FileNotFoundError(tool_name)

    # KERNEL entry runs out of band; a real GEAK subprocess would decide how long the phase is held.
    monkeypatch.setattr(request_handlers, "_kernel_agent_tool_path", _no_geak_runner)

    coord = _coordinator(session_dir)
    try:
        state = coord.shared_state
        state.baseline_tput = 1500.0
        state.max_minutes = 180
        # Source arm: no local exploration, and discovery past its retries.
        state.framework_local_explore_enabled = False
        state.framework_agent_empty_discoveries = 99
        # Config arm: a run of benched variants past the streak floor, none adopted.
        for i in range(ps.DEFAULT_PLATEAU_EXPLORE_EMPTY_STREAK + 1):
            record_config_attempt(
                state,
                task_id=f"explore-{i}",
                round_id=f"round-{i}",
                fingerprint=f"fp-{i}",
                variant_name=f"variant-{i}",
                outcome="REVERT",
                gain_pct=0.01,
                before_tput=1500.0,
                after_tput=1500.15,
                error_class="",
                provenance="default_grid",
            )
        state.save(session_dir)

        for tick in range(1, 12):
            await coord.tick(tick)

        assert state.phase == ps.PHASE_CLOSE
        assert state.geak_result["error_class"] == "runner_not_found"
        visited = [to_phase for _, to_phase, _ in _chain(state)]
        assert visited[:2] == [ps.PHASE_PRELUDE, ps.PHASE_FRAMEWORK_AGENT]
        assert visited[-3:] == [ps.PHASE_KERNEL_AGENT, ps.PHASE_SWEEP, ps.PHASE_CLOSE]
        reasons = {reason for _, _, reason in _chain(state)}
        assert "optimize_no_more_leverage" in reasons
    finally:
        await coord.stop()
