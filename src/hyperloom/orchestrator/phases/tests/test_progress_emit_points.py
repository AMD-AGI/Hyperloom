# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The legacy flow's emit points for progress events: a phase switch and a current_best promotion."""

from __future__ import annotations

from pathlib import Path

import pytest

from hyperloom.orchestrator.phases.machine import Transition
from hyperloom.plugs import Implementation, ManifestSources, PlugHost, activated, builtin_registry, resolve_manifest


class _Spy:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def emit(self, event_type, data) -> None:
        self.events.append((event_type, dict(data)))

    def flush(self) -> None:
        pass


@pytest.fixture
def spy_host():
    spy = _Spy()
    registry = builtin_registry()
    registry.register(Implementation("events", "spy", lambda ctx: spy))
    host = PlugHost(resolve_manifest(registry, ManifestSources(cli_plugs=("events=spy",))), registry, session_dir=None)
    host.start()
    with activated(host):
        yield spy


@pytest.fixture
def coord(tmp_path: Path):
    from hyperloom.orchestrator.loop.coordinator import AdvisoryServices, Coordinator
    from hyperloom.orchestrator.roles.agent_role import default_role_registry
    from hyperloom.orchestrator.roles.mock_backend import MockBackend, MockTurn, ScriptedPlan
    from hyperloom.orchestrator.state.shared_state import SharedState

    state = SharedState(session_id="progress-test", baseline_tput=1000.0, model_path="/models/m")
    state.max_minutes = 120
    state.save(tmp_path)
    idle = ScriptedPlan(turns=[MockTurn(intents=[])])
    return Coordinator(
        session_dir=tmp_path,
        backends={name: MockBackend(idle) for name in ("orchestration", "critic")},
        role_registry=default_role_registry(),
        advisory=AdvisoryServices(knowledge_plane=None),
    )


async def test_a_phase_switch_emits_phase_changed_once(coord, spy_host):
    coord.shared_state.phase = "PRELUDE"

    await coord.phase_machine._commit_transition(Transition("PRELUDE", "FRAMEWORK_AGENT", "baseline_done", {}))

    changed = [d for t, d in spy_host.events if t == "hyperloom.phase_changed"]
    assert changed == [{"from": "PRELUDE", "to": "FRAMEWORK_AGENT", "macro_cycle": coord.shared_state.macro_cycle}]


def test_a_current_best_promotion_emits_best_updated(coord, spy_host):
    winner = {"name": "v", "output_throughput": 1200.0, "extra_server_args": "--tp 8", "request_id": "req-3"}

    assert coord.writeback.lift_to_current_best("explore", 1200.0, winner) is True

    updated = [d for t, d in spy_host.events if t == "hyperloom.best_updated"]
    assert updated == [
        {
            "metric": "output_throughput",
            "value": 1200.0,
            "gain_pct": 20.0,
            "patch_count": len(coord.shared_state.optimization_stack),
            "request_id": "req-3",
        }
    ]


def test_a_winner_that_does_not_beat_the_anchor_emits_no_best_updated(coord, spy_host):
    coord.shared_state.current_best = {"tput": 1300.0, "output_throughput": 1300.0}

    assert coord.writeback.lift_to_current_best("explore", 1200.0, {"name": "v", "output_throughput": 1200.0}) is False

    assert [t for t, _ in spy_host.events if t == "hyperloom.best_updated"] == []
