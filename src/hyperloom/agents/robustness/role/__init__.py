# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Robustness reactor role layer.

Only the wire-format primitives (envelope + prompt_inputs) are re-exported here;
the reactor lives in ``role.reactor`` to keep this module import-light for hosts
that just need the JSON-IO surface.
"""

from .envelope import (
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
    ConversationProgress,
    InboxItem,
    PhaseBudgetRow,
    ReactorContext,
    SharedStateSnapshot,
    from_coordinator_prompt,
)

__all__ = [
    "ConversationProgress",
    "InboxItem",
    "Intent",
    "IntentType",
    "PhaseBudgetRow",
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
