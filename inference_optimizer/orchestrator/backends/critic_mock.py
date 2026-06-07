# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Mock Critic backend — auto-approves every proposal it sees.

Used in P0 main-path tests so Coordinator + Orchestration + Kernel can run
end-to-end without a real Codex Critic. The full Critic Review Protocol
(§18) — verdict ∈ {approve, reject, redirect, advise, needs_review},
KB-evidence reasoning, brier calibration — is implemented later by the
Critic owner.

Behaviour:

* When the prompt contains an inbox row with ``topic=proposal`` and a
  parseable ``msg_id``, emit ``review_verdict{verdict="approve",
  source="mock"}`` for that proposal.
* If multiple proposals are visible, emit one verdict per proposal in
  the same turn.
* Otherwise emit a ``send_message{topic="heartbeat"}`` so the reactor
  loop sees signal of life (Critic always speaks at least once per tick).
"""

from __future__ import annotations

import re
from typing import Any

from ...protocol.intent import Intent, IntentType
from .base import BackendTurnResult


# DESIGN §13.1 inbox rendering format. Coordinator._compose_prompt emits:
#     seq=12 msg_id=<hex32> from=orchestration topic=proposal payload={...}
_PROPOSAL_RE = re.compile(
    r"^\s*seq=(\d+)\s+msg_id=([a-f0-9]+)\s+from=(\w+)\s+topic=proposal\s+payload=(.*)$",
    re.MULTILINE,
)


class MockCriticBackend:
    """Always-approve Critic adapter. Implements :class:`Backend`."""

    def __init__(self, name: str = "critic-mock"):
        self.name = name
        self.calls: list[dict[str, Any]] = []
        # Track which proposals we've already approved so we don't double-emit
        # if the same proposal appears in two consecutive inbox windows.
        self._approved_msg_ids: set[str] = set()

    async def run(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        tools: list[str] | None = None,
        max_turns: int = 1,
    ) -> BackendTurnResult:
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
