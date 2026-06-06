"""Post-PR-321 review Finding 1 regression — Coordinator runtime hints
must not tell the LLM to propose ``profile`` / ``roofline``.

Both names are absent from ``PHASE_LLM_PROPOSABLE_ACTIONS``, so any
LLM-emitted ``propose_action`` / ``delegate`` is denied by PolicyGate
R1 with ``rule='phase_incompatible'``. The sequence-denial hints in
coordinator.py previously emitted phrasing like *"propose/delegate
`profile`"* even after the single-path refactor. If the PRELUDE /
watermark auto-analysis fails (or the pending field gets stuck mid-
restart), those hints would point the LLM at a guaranteed denial and
trip a policy loop — there is no manual recovery path for the LLM
other than ``recover``.

This test pins the sequence-denial sites:

  * ``_sequence_denial_for_action()`` — the action-layer profile-prereq
    deny was removed (P2_10), so explore-family actions are no longer
    blocked on an empty ``last_profile_trace``.
  * ``_sequence_denial_for_request()`` — the request-layer profile-prereq
    deny was demoted (P2_11) to a data-contract check inside
    ``run_optimization_handler``, so kernel requests are no longer
    pre-denied on an empty ``last_profile_trace``.

Both checks guarantee the LLM is never steered into the forbidden
``propose/delegate profile`` phrasing via these sequencing hints.
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
    # ``_sequence_denial_for_action`` calls
    # ``_target_analysis_baseline_exists`` — short-circuit it so the
    # target_analysis gate doesn't mask the profile-prereq denial branch
    # the sequence-denial tests below target.
    c._target_analysis_baseline_exists = lambda: True  # type: ignore[assignment]
    return c


@pytest.mark.parametrize(
    "action",
    ["explore", "sweep", "report", "integrate"],
)
def test_sequence_denial_action_no_longer_blocks_on_profile(
    coord: Coordinator, action: str,
):
    """The action-layer profile-prereq deny was removed (P2_10): with an
    empty ``last_profile_trace`` (and baseline done), explore-family
    actions are no longer sequence-denied for a missing profile."""
    coord.shared_state.last_profile_trace = ""
    coord.shared_state.baseline_tput = 100.0
    assert coord._sequence_denial_for_action(action) is None


@pytest.mark.parametrize(
    "req_kind", ["run_optimization"],
)
def test_sequence_denial_request_no_longer_blocks_on_profile(
    coord: Coordinator, req_kind: str,
):
    """The REQUEST-layer profile-prereq deny was demoted alongside the
    action-layer one: with baseline done and an empty
    ``last_profile_trace``, the request layer no longer pre-denies kernel
    requests on a missing profile. The data dependency on a fresh
    trace_analyze (and the candidates artifact it produces) is enforced
    inside ``run_optimization_handler`` instead, so the LLM hint surface
    cannot leak forbidden ``propose/delegate profile`` phrasing."""
    coord.shared_state.last_profile_trace = ""
    coord.shared_state.baseline_tput = 100.0
    assert coord._sequence_denial_for_request("kernel", req_kind) is None
