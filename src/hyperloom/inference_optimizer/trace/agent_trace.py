# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What a finished agent-CLI run left in its logs, whichever CLI wrote them."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class AgentTrace:
    """One agent-CLI run's trace.

    Attributes:
        usage: Cumulative token usage on the canonical counters (plus the
            model name when the log names it), or ``None`` when the log
            carried none.
        response: The reply text that lands in ``conversations.jsonl``.
        tool_calls: ``{"tool", "query"}`` rows for the intel ledger.
        turn_usages: Per-turn usage for fine-grained tracing.
        error: The failure the CLI itself recorded, prefixed with the CLI's
            name; empty when it recorded none.
    """

    usage: dict[str, Any] | None = None
    response: str | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    turn_usages: list[dict[str, Any]] = field(default_factory=list)
    error: str = ""
