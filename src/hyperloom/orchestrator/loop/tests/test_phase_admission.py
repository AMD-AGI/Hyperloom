# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The phase contract the admission gate holds an agent-proposed action to.

An agent may propose or delegate only what its phase lists as proposable; the
Critic approves on its own reading and the phase barrier only sweeps queued work
at a transition, so anything admitted here runs.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from hyperloom.inference_optimizer.protocol.action_surfaces import ACTION_CATALOGUE
from hyperloom.orchestrator.loop.dispatcher import DispatcherCollaborator
from hyperloom.orchestrator.policy.gate import PolicyDenied
from hyperloom.orchestrator.state.shared_state import SharedState


def _dispatcher(phase: str, *, baseline_tput: float = 800.0) -> DispatcherCollaborator:
    """A dispatcher over the coordinator surface admission reads, with an unbounded budget."""
    state = SharedState(session_id="s")
    state.phase = phase
    state.baseline_tput = baseline_tput
    no_tasks = SimpleNamespace(queued_context_sync=list, running_context_sync=list)
    return DispatcherCollaborator(SimpleNamespace(shared_state=state, action_registry=ACTION_CATALOGUE, tasks=no_tasks))


def test_a_baseline_proposed_in_the_optimisation_phase_is_refused():
    denied = _dispatcher("FRAMEWORK_AGENT").admission_denial_for_action("baseline")

    assert isinstance(denied, PolicyDenied)
    assert denied.rule == "phase_incompatible"
    assert "'baseline'" in str(denied) and "'FRAMEWORK_AGENT'" in str(denied)
    assert denied.hint == "FRAMEWORK_AGENT proposes: explore, integrate_patch, specialist"


@pytest.mark.parametrize(
    ("phase", "action"),
    [
        ("PRELUDE", "explore"),
        ("PRELUDE", "specialist"),
        ("ENABLEMENT", "explore"),
        ("ENABLEMENT", "specialist"),
        ("ENABLEMENT", "integrate_patch"),
        ("FRAMEWORK_AGENT", "target_analysis"),
        ("FRAMEWORK_AGENT", "report"),
        ("KERNEL_AGENT", "explore"),
        ("KERNEL_AGENT", "specialist"),
        ("SWEEP", "explore"),
        ("CLOSE", "integrate_patch"),
    ],
)
def test_an_action_outside_the_phase_contract_is_refused(phase: str, action: str):
    denied = _dispatcher(phase).admission_denial_for_action(action)

    assert denied is not None and denied.rule == "phase_incompatible"


def test_a_phase_that_proposes_nothing_says_so():
    denied = _dispatcher("SWEEP").admission_denial_for_action("explore")

    assert denied is not None and denied.hint == "SWEEP proposes: (none)"


@pytest.mark.parametrize(
    ("phase", "action"),
    [
        ("FRAMEWORK_AGENT", "explore"),
        ("FRAMEWORK_AGENT", "integrate_patch"),
        ("FRAMEWORK_AGENT", "specialist"),
        ("ENABLEMENT", "target_analysis"),
        ("KERNEL_AGENT", "integrate"),
        ("CLOSE", "report"),
        ("CLOSE", "session_breakdown"),
    ],
)
def test_an_action_the_phase_proposes_is_admitted(phase: str, action: str):
    assert _dispatcher(phase).admission_denial_for_action(action) is None


def test_the_prelude_baseline_is_admitted_before_any_measurement():
    assert _dispatcher("PRELUDE", baseline_tput=0.0).admission_denial_for_action("baseline") is None


def test_enablement_keeps_its_baseline_for_the_coordinator():
    """ENABLEMENT runs its baseline only as the Coordinator's revalidation of a KEEP."""
    denied = _dispatcher("ENABLEMENT").admission_denial_for_action("baseline")

    assert denied is not None and denied.rule == "phase_incompatible"
    assert denied.hint == "ENABLEMENT proposes: target_analysis"


@pytest.mark.parametrize("action", ["baseline", "explore", "specialist"])
def test_no_phase_entered_yet_holds_nothing_to_a_contract(action: str):
    """Before the phase machine enters PRELUDE there is no phase whose contract could apply."""
    assert _dispatcher("").admission_denial_for_action(action) is None
