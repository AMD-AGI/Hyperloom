"""Post-PR-321 review Finding 1 regression — Coordinator runtime hints
must not tell the LLM to propose ``profile`` / ``roofline``.

PolicyGate denies LLM-emitted ``propose_action`` / ``delegate`` against
either action with ``rule='analysis_action_not_llm_proposable'``. The
sequence-denial hints in coordinator.py previously emitted phrasing like
*"propose/delegate `profile`"* even after the single-path refactor. If
the PRELUDE / watermark auto-analysis fails (or the pending field gets
stuck mid-restart), those hints would point the LLM at a guaranteed
denial and trip a policy loop — there is no manual recovery path for the
LLM other than ``recover``.

This test pins the sequence-denial sites:

  * ``_sequence_denial_for_action()`` — execution_order denial for
    sequence-gated actions when last_profile_trace is empty.
  * ``_sequence_denial_for_request()`` — execution_order denial for
    trace_analyze / run_optimization requests.

The accepted phrasing must reference (a) waiting for the Coordinator-
internal analysis task and (b) ``recover`` as the escape hatch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from inference_optimizer.orchestrator.coordinator import Coordinator
from inference_optimizer.orchestrator.policy import PolicyDenied


_FORBIDDEN_SUBSTRINGS = (
    "propose/delegate `profile`",
    "propose/delegate only `profile`",
    "propose/delegate `roofline`",
    "propose `profile`",
    "delegate `profile`",
    "propose `roofline`",
    "delegate `roofline`",
)


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


def _assert_no_forbidden(blob: str, where: str) -> None:
    for s in _FORBIDDEN_SUBSTRINGS:
        assert s not in blob, (
            f"{where}: hint still tells the LLM to {s!r}; "
            f"PolicyGate will deny with analysis_action_not_llm_proposable. "
            f"Full text: {blob!r}"
        )


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
def test_sequence_denial_request_hint_does_not_tell_llm_to_propose_profile(
    coord: Coordinator, req_kind: str,
):
    """Same contract for the REQUEST-layer denial that fires on
    run_optimization when ``last_profile_trace`` is empty. (Note:
    ``trace_analyze`` is the prereq itself and early-returns ``None``
    from ``_sequence_denial_for_request``; only ``run_optimization``
    reaches the profile-prereq branch.)"""
    coord.shared_state.last_profile_trace = ""
    coord.shared_state.baseline_tput = 100.0
    denied = coord._sequence_denial_for_request("kernel", req_kind)
    assert isinstance(denied, PolicyDenied), (
        f"expected the profile-prereq denial to fire for request={req_kind!r}"
    )
    assert denied.rule == "execution_order"
    _assert_no_forbidden(
        denied.hint, f"_sequence_denial_for_request({req_kind!r})",
    )
