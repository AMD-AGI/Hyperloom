# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Post-PR-321 review Finding 1 regression — Coordinator runtime hints
must not tell the LLM to propose ``profile`` / ``roofline``.

PolicyGate denies LLM-emitted ``propose_action`` / ``delegate`` against
either action with ``rule='analysis_action_not_llm_proposable'``. Three
sites in coordinator.py previously emitted hints like *"propose/delegate
`profile`"* even after the single-path refactor. If the
PRELUDE / watermark auto-analysis fails (or the pending field gets stuck
mid-restart), those hints would point the LLM at a guaranteed denial and
trip a policy loop — there is no manual recovery path for the LLM other
than ``recover``.

This test pins all three sites:

  * ``_required_next_step()`` — the TODO surfaced in the prompt every tick.
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
    # ``_required_next_step`` calls ``_target_analysis_baseline_exists``
    # — short-circuit it so we skip the TODO 0 branch and reach the
    # profile-trace branch that owns the offending hint.
    c._target_analysis_baseline_exists = lambda: True  # type: ignore[assignment]
    return c


def _assert_no_forbidden(blob: str, where: str) -> None:
    for s in _FORBIDDEN_SUBSTRINGS:
        assert s not in blob, (
            f"{where}: hint still tells the LLM to {s!r}; "
            f"PolicyGate will deny with analysis_action_not_llm_proposable. "
            f"Full text: {blob!r}"
        )


def test_required_next_step_does_not_tell_llm_to_propose_profile(
    coord: Coordinator,
):
    """``_required_next_step`` is rendered into the orchestration prompt
    every tick. When ``last_profile_trace`` is empty (the PRELUDE /
    watermark analysis hasn't landed yet), the hint must point at the
    auto-enqueued Coordinator task or ``recover`` — never at a manual
    propose/delegate that PolicyGate will deny."""
    coord.shared_state.last_profile_trace = ""
    coord.shared_state.baseline_tput = 100.0
    todo = coord._required_next_step()
    assert todo, "expected a TODO when last_profile_trace is empty"
    _assert_no_forbidden(todo, "_required_next_step")
    # Positive contract: must mention the auto-enqueue or recovery
    # path so the LLM knows what to do instead.
    low = todo.lower()
    assert any(tok in low for tok in ("auto-enqueue", "coordinator", "recover")), (
        "_required_next_step: must point the LLM at the Coordinator-"
        "internal analysis task or `recover`; got: %r" % todo
    )


@pytest.mark.parametrize(
    "action",
    ["explore", "sweep", "report", "integrate"],
)
def test_sequence_denial_action_hint_does_not_tell_llm_to_propose_profile(
    coord: Coordinator, action: str,
):
    """``_sequence_denial_for_action`` returns the PolicyDenied that the
    LLM reads in its inbox. When the gate fires on empty
    ``last_profile_trace``, the hint must NOT tell the LLM to propose
    profile (PolicyGate denies that too)."""
    coord.shared_state.last_profile_trace = ""
    coord.shared_state.baseline_tput = 100.0
    denied = coord._sequence_denial_for_action(action)
    assert isinstance(denied, PolicyDenied), (
        f"expected the profile-prereq denial to fire for action={action!r}"
    )
    assert denied.rule == "execution_order"
    _assert_no_forbidden(denied.hint, f"_sequence_denial_for_action({action!r})")
    assert "analysis_action_not_llm_proposable" in denied.hint or (
        "auto-enqueue" in denied.hint.lower() or "coordinator" in denied.hint.lower()
    ), (
        f"_sequence_denial_for_action({action!r}): hint must name the "
        f"Coordinator-internal analysis path; got: {denied.hint!r}"
    )


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
