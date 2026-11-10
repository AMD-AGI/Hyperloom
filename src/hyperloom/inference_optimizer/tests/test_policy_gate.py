# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What the gate keeps closed to an agent, and what the prompt is allowed to offer.

The pure helpers (presence checks, GPU probing, lane ceilings, path allowlists,
free-form descriptions) are exercised in
``test_policy_helpers_coverage_unit.py``; the bring-up round's own lifecycle in
``test_policy_advisory_projection.py``. What is left here is the agent-facing
surface: the request kinds, action names and payload paths an agent may reach
through, and the standing agreement that the prompt never advertises a lever the
gate then refuses.
"""

from __future__ import annotations


import pytest

from hyperloom.inference_optimizer.protocol.action_surfaces import (
    COORDINATOR_INTERNAL_ACTIONS,
    COORDINATOR_OWNED_KERNEL_REQUEST_KINDS,
)
from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType
from hyperloom.orchestrator.kernel.request_handlers import KERNEL_REQUEST_HANDLERS
from hyperloom.orchestrator.phases.machine_state import PHASE_NAMES, allowed_actions_for
from hyperloom.orchestrator.policy.denial import PolicyDenied
from hyperloom.orchestrator.policy.gate import PolicyGate
from hyperloom.orchestrator.policy.projection import ResourceFacts
from hyperloom.orchestrator.prompts.prompt_builder import _section_phase_semantics
from hyperloom.orchestrator.roles.agent_role import default_role_registry
from hyperloom.orchestrator.state.shared_state import SharedState

_PROPOSE_CHANNELS = (IntentType.DELEGATE, IntentType.PROPOSE_ACTION)


def _llm_gate(resources: ResourceFacts | None = None) -> PolicyGate:
    """A gate an orchestration agent emits into.

    Args:
        resources: The resource facts the round rule judges against; ``None``
            leaves that rule refusing nothing.

    Returns:
        PolicyGate: The gate under test.
    """
    return PolicyGate(
        role_registry=default_role_registry(),
        shared_state=SharedState(session_id="t", phase="KERNEL_AGENT"),
        resources=resources or ResourceFacts(),
    )


def _emit(gate: PolicyGate, intent_type: IntentType, payload: dict) -> None:
    """Emit one intent as the orchestration agent.

    Args:
        gate: The gate under test.
        intent_type: The channel the intent arrives on.
        payload: The intent payload.
    """
    gate.validate_intent("orchestration", Intent(type=intent_type, payload=payload))


# -- Coordinator-owned kernel lanes ---------------------------------------
@pytest.mark.parametrize("kind", sorted(COORDINATOR_OWNED_KERNEL_REQUEST_KINDS))
def test_a_coordinator_owned_request_kind_is_refused(kind: str) -> None:
    """A direct request skips the lane's own entry gate and accounting."""
    with pytest.raises(PolicyDenied) as excinfo:
        _emit(_llm_gate(), IntentType.REQUEST, {"target_agent": "kernel_agent", "kind": kind})
    assert excinfo.value.rule == "request_kind"


@pytest.mark.parametrize("kind", sorted(set(KERNEL_REQUEST_HANDLERS) - COORDINATOR_OWNED_KERNEL_REQUEST_KINDS))
def test_the_llm_requestable_kinds_still_pass(kind: str) -> None:
    """The narrowing refuses the owned lanes and nothing beside them."""
    _emit(_llm_gate(), IntentType.REQUEST, {"target_agent": "kernel_agent", "kind": kind})


def test_an_unregistered_kind_reaches_the_auto_reject() -> None:
    """The handler lookup answers a typo with the valid-kind vocabulary."""
    _emit(_llm_gate(), IntentType.REQUEST, {"target_agent": "kernel_agent", "kind": "no_such_kind"})


# -- Coordinator-managed actions ------------------------------------------
def test_a_coordinator_managed_action_is_not_proposable() -> None:
    """Each has a registered executor, so a proposal really ran a second copy."""
    gate = _llm_gate()
    for channel in _PROPOSE_CHANNELS:
        for action_name in sorted(COORDINATOR_INTERNAL_ACTIONS):
            with pytest.raises(PolicyDenied) as excinfo:
                _emit(gate, channel, {"action_name": action_name, "predicted_gain_pct": 1.0, "params": {}})
            assert excinfo.value.rule == "coordinator_managed_action", (channel, action_name)


def test_the_coordinator_still_dispatches_its_own_internal_actions() -> None:
    """The guard sits on the agent channels; dispatch replay must pass."""
    gate = _llm_gate()
    for action_name in sorted(COORDINATOR_INTERNAL_ACTIONS):
        gate.validate_dispatched_task(action_name, {})


def test_phase_semantics_prompt_names_every_internal_action() -> None:
    """The orchestration prompt must name the actions PolicyGate will deny.

    Telling the model that ``framework`` is Coordinator-managed while the
    runtime denies ``framework_agent`` invites a proposal that costs a tick
    and gets rejected as coordinator_managed_action.
    """
    rendered = "\n".join(_section_phase_semantics(kernel_enabled=True))

    missing = sorted(a for a in COORDINATOR_INTERNAL_ACTIONS if a not in rendered)
    assert not missing, f"Coordinator-internal actions absent from the prompt: {missing}"


# -- A bring-up round holds the machine -----------------------------------
def _round_in_flight() -> ResourceFacts:
    """Facts in which a bring-up round holds the machine."""
    return ResourceFacts(excluding_round_id="round-1", excluding_round_holder="task-abc")


def test_baseline_is_refused_on_both_agent_channels_while_a_round_holds_the_machine() -> None:
    """A second bring-up fights the first for the same cards and ports."""
    gate = _llm_gate(_round_in_flight())
    for channel in _PROPOSE_CHANNELS:
        with pytest.raises(PolicyDenied) as excinfo:
            _emit(gate, channel, {"action_name": "baseline", "params": {}})
        assert excinfo.value.rule == "enablement_round_in_flight", channel


def test_the_round_holders_own_bring_up_is_dispatched_and_no_other_row_is() -> None:
    """Dispatch replay is guarded too; the holder is admitted by its own id."""
    gate = _llm_gate(_round_in_flight())
    gate.validate_dispatched_task("baseline", {"reason": "enablement_revalidation"}, task_id="task-abc")

    with pytest.raises(PolicyDenied) as excinfo:
        gate.validate_dispatched_task("baseline", {}, task_id="forged-row")
    assert excinfo.value.rule == "enablement_round_in_flight"


def test_the_round_rule_refuses_nothing_without_a_snapshot() -> None:
    """No ledger means no resource facts, and the acquire remains the authority."""
    _emit(_llm_gate(), IntentType.DELEGATE, {"action_name": "baseline", "params": {}})


# -- send_message recipient shape -----------------------------------------
def test_send_message_to_a_single_recipient_is_admitted() -> None:
    """``to`` names one recipient; the inbox filter and the events row agree."""
    _emit(_llm_gate(), IntentType.SEND_MESSAGE, {"to": "critic", "topic": "observation", "body_md": "x"})


def test_send_message_without_to_broadcasts() -> None:
    """An absent ``to`` is the broadcast form and stays admitted."""
    _emit(_llm_gate(), IntentType.SEND_MESSAGE, {"topic": "observation", "body_md": "x"})


def test_send_message_to_a_list_of_recipients_is_admitted() -> None:
    """A JSON array is how a model addresses several agents at once; each becomes its own events row."""
    _emit(_llm_gate(), IntentType.SEND_MESSAGE, {"to": ["critic", "specialist:t1"], "topic": "observation"})


@pytest.mark.parametrize("to", [[], ["critic", ""], ["critic", 3], 3, {"agent": "critic"}, ""])
def test_send_message_with_a_to_naming_no_recipient_is_denied(to) -> None:
    """A ``to`` that is not a name or a list of names is denied, not crashed on.

    Before the check a raw list reached the events INSERT as ``to_agent`` and faulted the running phase event; a
    list is fanned out now, but one holding an empty or non-string name still names nobody.
    """
    with pytest.raises(PolicyDenied) as excinfo:
        _emit(_llm_gate(), IntentType.SEND_MESSAGE, {"to": to, "topic": "observation", "body_md": "x"})
    assert excinfo.value.rule == "payload"


# -- The prompt and the gate agree ----------------------------------------
def test_the_prompt_never_advertises_an_action_the_gate_denies() -> None:
    """Ties the rendered per-phase sets to what the propose channel accepts."""
    gate = _llm_gate()
    denied: list[tuple[str, str, str]] = []
    for phase in PHASE_NAMES:
        for action_name in allowed_actions_for(phase):
            try:
                _emit(
                    gate,
                    IntentType.PROPOSE_ACTION,
                    {"action_name": action_name, "predicted_gain_pct": 1.0, "params": {}},
                )
            except PolicyDenied as exc:
                if exc.rule in {"coordinator_managed_action", "propose_action_source"}:
                    denied.append((phase, action_name, exc.rule or ""))
    assert not denied, f"prompt advertises actions the gate refuses: {denied}"
