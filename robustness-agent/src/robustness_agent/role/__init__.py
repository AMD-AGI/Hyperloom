"""Robustness reactor role layer.

Bridges the Coordinator's Backend protocol (or a multi-cli inbox/outbox
transport in later milestones) to the agent's symptom -> decision pipeline.

Only the wire-format primitives (envelope + prompt_inputs) are
re-exported at package level. Higher-level pieces (:class:`Reactor`,
:class:`RobustnessAgentBackend`) live in submodules to avoid circular
imports between ``role`` and ``decision``.
"""

from .envelope import (
    BackendTurnResult,
    Intent,
    IntentType,
    build_alert,
    build_envelope_dict,
    build_escalate,
    build_heartbeat,
    build_send_message,
    build_update_state,
)
from .prompt_inputs import (
    InboxItem,
    ReactorContext,
    SharedStateSnapshot,
    from_coordinator_prompt,
)

__all__ = [
    "BackendTurnResult",
    "InboxItem",
    "Intent",
    "IntentType",
    "ReactorContext",
    "SharedStateSnapshot",
    "build_alert",
    "build_envelope_dict",
    "build_escalate",
    "build_heartbeat",
    "build_send_message",
    "build_update_state",
    "from_coordinator_prompt",
]
