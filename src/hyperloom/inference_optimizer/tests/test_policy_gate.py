# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What the gate keeps closed to an agent, and what the prompt is allowed to offer.

The pure helpers (presence checks, GPU probing, lane ceilings, path allowlists,
free-form descriptions) are exercised in
``test_policy_helpers_coverage_unit.py``; the bring-up round's own lifecycle in
``test_policy_advisory_projection.py``; which action names a phase lets an agent
propose in ``loop/tests/test_phase_admission.py``. What is left here is the
agent-facing surface: the request kinds, the bring-up round and the message
shapes an agent may reach through, and that a refusal is a denial which leaves
no row in the failure ledger.
"""

from __future__ import annotations


import pytest

from hyperloom.inference_optimizer.protocol.action_surfaces import (
    COORDINATOR_INTERNAL_ACTIONS,
    COORDINATOR_OWNED_KERNEL_REQUEST_KINDS,
)
from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType
from hyperloom.orchestrator.kernel.request_handlers import KERNEL_REQUEST_HANDLERS
from hyperloom.orchestrator.loop.coordinator import Coordinator
from hyperloom.orchestrator.policy.denial import PolicyDenied
from hyperloom.orchestrator.policy.gate import PolicyGate
from hyperloom.orchestrator.policy.projection import ResourceFacts
from hyperloom.orchestrator.prompts.prompt_builder import _section_phase_semantics
from hyperloom.orchestrator.state.shared_state import SharedState

from .conftest import silent_backends

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


# -- A refusal is a denial, never a recorded failure -----------------------
def _denied_rules(coordinator: Coordinator) -> list[str]:
    """The rule of every ``policy_denied`` observation in the orchestration inbox."""
    inbox = coordinator.bus.inbox_context_sync("orchestration")
    return [m.payload["rule"] for m in inbox if m.payload.get("kind") == "policy_denied"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kind", "rule"),
    [("run_fusion", "request_kind"), ("gemm_tuning", "phase_incompatible"), ("WebSearch", "tool_whitelist_role")],
)
async def test_a_refused_request_kind_is_denied_and_leaves_no_failure_row(session_dir, kind: str, rule: str) -> None:
    """Past the gate the router would answer with a failed reply, which becomes a gap and a ``what_failed`` prior."""
    coordinator = Coordinator(session_dir, backends=silent_backends())
    try:
        await coordinator.router.handle_intent(
            "orchestration", Intent(type=IntentType.REQUEST, payload={"target_agent": "kernel_agent", "kind": kind})
        )

        assert _denied_rules(coordinator) == [rule]
        assert coordinator.shared_state.last_action_failures == []
        assert await coordinator.bus.tail(topic="request") == []
        assert await coordinator.bus.tail(topic="response") == []
    finally:
        await coordinator.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("channel", _PROPOSE_CHANNELS, ids=lambda channel: channel.value)
async def test_an_action_no_phase_offers_is_denied_and_leaves_no_failure_row(session_dir, channel: IntentType) -> None:
    """Admission refuses the name before a task exists, so nothing reaches the failure ledger."""
    coordinator = Coordinator(session_dir, backends=silent_backends())
    try:
        payload = {"action_name": "profile", "predicted_gain_pct": 1.0, "params": {}}
        await coordinator.router.handle_intent("orchestration", Intent(type=channel, payload=payload))

        assert _denied_rules(coordinator) == ["phase_incompatible"]
        assert coordinator.shared_state.last_action_failures == []
        assert await coordinator.tasks.by_state("queued") == []
    finally:
        await coordinator.stop()


# -- Coordinator-managed actions ------------------------------------------
def test_the_coordinator_still_dispatches_its_own_internal_actions() -> None:
    """Dispatch replay must pass the actions only the Coordinator creates."""
    gate = _llm_gate()
    for action_name in sorted(COORDINATOR_INTERNAL_ACTIONS):
        gate.validate_dispatched_task(action_name, {})


def test_phase_semantics_prompt_names_every_internal_action() -> None:
    """The orchestration prompt must name the actions admission will deny.

    Telling the model that ``framework`` is Coordinator-managed while the
    runtime denies ``framework_agent`` invites a proposal that costs a tick
    and gets rejected as phase_incompatible.
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
    _emit(_llm_gate(), IntentType.SEND_MESSAGE, {"to": "kernel_agent", "topic": "observation", "body_md": "x"})


def test_send_message_without_to_broadcasts() -> None:
    """An absent ``to`` is the broadcast form and stays admitted."""
    _emit(_llm_gate(), IntentType.SEND_MESSAGE, {"topic": "observation", "body_md": "x"})


def test_send_message_to_a_list_of_recipients_is_admitted() -> None:
    """A JSON array is how a model addresses several agents at once; each becomes its own events row."""
    _emit(_llm_gate(), IntentType.SEND_MESSAGE, {"to": ["kernel_agent", "specialist:t1"], "topic": "observation"})


@pytest.mark.parametrize("to", [[], ["kernel_agent", ""], ["kernel_agent", 3], 3, {"agent": "kernel_agent"}, ""])
def test_send_message_with_a_to_naming_no_recipient_is_denied(to) -> None:
    """A ``to`` that is not a name or a list of names is denied, not crashed on.

    Before the check a raw list reached the events INSERT as ``to_agent`` and faulted the running phase event; a
    list is fanned out now, but one holding an empty or non-string name still names nobody.
    """
    with pytest.raises(PolicyDenied) as excinfo:
        _emit(_llm_gate(), IntentType.SEND_MESSAGE, {"to": to, "topic": "observation", "body_md": "x"})
    assert excinfo.value.rule == "payload"
