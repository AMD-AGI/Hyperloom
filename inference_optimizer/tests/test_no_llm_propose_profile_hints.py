# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Post-PR-321 review Finding 1 regression — Coordinator sequence-denial hints must not steer the LLM toward proposing ``profile`` / ``roofline``.

Pins that ``_sequence_denial_for_action`` (P2_10) and
``_sequence_denial_for_request`` (P2_11) no longer block on an empty
``last_profile_trace``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from inference_optimizer.orchestrator.coordinator import Coordinator


@dataclass
class _BareState:
    baseline_tput: float = 100.0
    last_profile_trace: str = ""
    last_trace_analyze: dict[str, Any] = field(default_factory=dict)
    cumulative_gain_validated: float = 0.0
    last_roofline_tput: float = 0.0
    auto_roofline_pending_task_id: str = ""
    enable_roofline: bool = True
    stop_reason: str = ""
    optimization_stack: list = field(default_factory=list)


@pytest.fixture
def coord(tmp_path: Path) -> Coordinator:
    c = Coordinator.__new__(Coordinator)
    c.session_dir = tmp_path
    c.shared_state = _BareState()
    c.role_registry = {"kernel": object()}
    c._compare_against_gpu = ""
    # Short-circuit ``_target_analysis_baseline_exists`` so the gate doesn't mask the profile-prereq branch under test.
    c._target_analysis_baseline_exists = lambda: True  # type: ignore[assignment]
    return c


@pytest.mark.parametrize(
    "action",
    ["explore", "sweep", "report", "integrate"],
)
def test_sequence_denial_action_no_longer_blocks_on_profile(
    coord: Coordinator, action: str,
):
    """P2_10: with baseline done and empty ``last_profile_trace``, explore-family actions are no longer sequence-denied."""
    coord.shared_state.last_profile_trace = ""
    coord.shared_state.baseline_tput = 100.0
    assert coord._sequence_denial_for_action(action) is None


@pytest.mark.parametrize(
    "req_kind", ["run_optimization"],
)
def test_sequence_denial_request_no_longer_blocks_on_profile(
    coord: Coordinator, req_kind: str,
):
    """P2_11: the request-layer profile-prereq deny was demoted into ``run_optimization_handler``, so kernel requests aren't pre-denied on an empty ``last_profile_trace``."""
    coord.shared_state.last_profile_trace = ""
    coord.shared_state.baseline_tput = 100.0
    assert coord._sequence_denial_for_request("kernel", req_kind) is None
