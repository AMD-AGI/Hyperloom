# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Mock Critic backend — auto-approves every proposal it sees.

Used in P0 main-path tests. Emits ``review_verdict{verdict="approve"}`` per
visible proposal, or a heartbeat when none are present.
"""

from __future__ import annotations

import re
from typing import Any

from ...protocol.intent import Intent, IntentType
from .base import BackendTurnResult


# DESIGN §13.1 inbox rendering format.
_PROPOSAL_RE = re.compile(
    r"^\s*seq=(\d+)\s+msg_id=([a-f0-9]+)\s+from=(\w+)\s+topic=proposal\s+payload=(.*)$",
    re.MULTILINE,
)


class MockCriticBackend:
    """Always-approve Critic adapter. Implements :class:`Backend`."""

    def __init__(self, name: str = "critic-mock"):
        """Initialise the mock Critic backend.

        Args:
            name (str): Human-readable backend name used in logs and metadata.
        """
        self.name = name
        self.calls: list[dict[str, Any]] = []
        # Track approved proposals so a re-rendered inbox doesn't double-emit.
        self._approved_msg_ids: set[str] = set()

    async def run(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        tools: list[str] | None = None,
        max_turns: int = 1,
    ) -> BackendTurnResult:
        """Auto-approve every visible proposal, else emit a heartbeat.

        Scans the rendered inbox in ``prompt`` for proposal rows and emits one
        ``review_verdict`` with ``verdict="approve"`` per not-yet-approved
        proposal. When no proposal is visible, emits a single heartbeat message
        so the reactor loop always sees signal of life.

        Args:
            prompt (str): The composed turn prompt containing the rendered inbox.
            system_prompt (str | None): Unused; accepted for protocol parity.
            tools (list[str] | None): Unused; accepted for protocol parity.
            max_turns (int): Unused; accepted for protocol parity.

        Returns:
            BackendTurnResult: The approval and/or heartbeat intents for this
            turn.
        """
        self.calls.append({"prompt": prompt})
        intents: list[Intent] = []
        for match in _PROPOSAL_RE.finditer(prompt):
            seq, msg_id, from_agent, raw_payload = match.groups()
            if msg_id in self._approved_msg_ids:
                continue
            self._approved_msg_ids.add(msg_id)
            intents.append(Intent(
                type=IntentType.REVIEW_VERDICT,
                payload={
                    "target_proposal_msg_id": msg_id,
                    "verdict": "approve",
                    "reasoning": "(mock critic — auto-approve)",
                    "source": "mock",
                },
            ))
        if not intents:
            intents.append(Intent(
                type=IntentType.SEND_MESSAGE,
                payload={"topic": "heartbeat", "body_md": "ok (mock critic)"},
            ))
        return BackendTurnResult(intents=intents, raw_text="(mock critic)")


__all__ = ["MockCriticBackend"]
