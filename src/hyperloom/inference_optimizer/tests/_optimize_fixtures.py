# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared fixtures for the merged optimisation phase."""

from __future__ import annotations

from typing import Any

from hyperloom.orchestrator.actions.executors._grid_base import VariantResult
from hyperloom.orchestrator.state.shared_state import SharedState


def variant_result(**overrides: Any) -> VariantResult:
    """A real ``VariantResult`` with plausible defaults."""
    fields: dict[str, Any] = {
        "name": "v1",
        "extra_server_args": "",
        "extra_envs": {},
        "status": "succeeded",
        "output_throughput": 1000.0,
        "ttft_mean_ms": 10.0,
        "tpot_mean_ms": 2.0,
        "error": "",
        "nonfatal_warnings": [],
    }
    fields.update(overrides)
    return VariantResult(**fields)


def optimize_state(
    *,
    source_no_keep: int = 0,
    source_exhausted: bool = False,
    config_keep_gain_pct: float = 5.0,
    config_empty_rounds: int = 0,
    **overrides: Any,
) -> SharedState:
    """A real ``SharedState`` positioned in the optimisation phase."""
    from hyperloom.orchestrator.lever import LEVER_CONFIG, LEVER_SOURCE_PATCH
    from hyperloom.orchestrator.phases.machine_state import PHASE_FRAMEWORK_AGENT

    state = SharedState()
    state.phase = PHASE_FRAMEWORK_AGENT
    state.macro_cycle = 0
    state.baseline_tput = 1500.0
    state.framework_agent_phase_done = source_exhausted

    # Seed the unified attempts ledger used by per_lever_dryness.
    # Patch arm: N trailing no-keep source-patch attempts.
    for _ in range(source_no_keep):
        state.record_attempt({"lever_kind": LEVER_SOURCE_PATCH, "outcome": "REVERT", "adopted": False})
    # Config arm: one KEEP row carrying the requested gain, then empty rounds after it.
    state.record_attempt(
        {"lever_kind": LEVER_CONFIG, "outcome": "KEEP", "adopted": True, "gain_pct": config_keep_gain_pct}
    )
    for _ in range(config_empty_rounds):
        state.record_attempt({"lever_kind": LEVER_CONFIG, "outcome": "REVERT", "adopted": False})

    for key, value in overrides.items():
        setattr(state, key, value)
    return state


def fake_coordinator(cls: type[Any], session_dir: Any, **state: Any) -> Any:
    """A Coordinator built without ``__init__`` that holds only the given state; the rest resolves for real."""
    from hyperloom.inference_optimizer.protocol.action_surfaces import ACTION_CATALOGUE
    from hyperloom.orchestrator.phases.framework import FrameworkPhase

    coord = cls.__new__(cls)
    coord.session_dir = session_dir
    # The real catalogue: a stubbed one can only ever agree with the test.
    coord.action_registry = ACTION_CATALOGUE
    coord.phase_framework = FrameworkPhase(coord)
    for key, value in state.items():
        setattr(coord, key, value)
    return coord
