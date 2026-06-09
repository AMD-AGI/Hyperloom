# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Robustness reactor role layer.

Only the wire-format primitives (envelope + prompt_inputs) are re-exported here;
the reactor lives in ``role.reactor`` to keep this module import-light for hosts
that just need the JSON-IO surface.
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
