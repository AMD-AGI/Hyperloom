# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Row-scanning mock backend and the always-approve mock Critic built on it."""

from __future__ import annotations

import re
from typing import Any, Callable

from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType
from .base import BackendTurnResult


# Coordinator inbox row format: ``seq=<n> msg_id=<hex> from=<agent> topic=<t> payload=<...>``.
_PROPOSAL_RE = re.compile(
    r"^\s*seq=(\d+)\s+msg_id=([a-f0-9]+)\s+from=(\w+)\s+topic=proposal\s+payload=(.*)$",
    re.MULTILINE,
)


class MockRowScanBackend:
    """Row-scanning reactor mock: one intent per matched inbox row, else idle."""

    def __init__(
        self,
        *,
        name: str,
        row_regex: re.Pattern[str],
        intent_builder: Callable[[re.Match[str]], Intent],
        idle_body: str,
        raw_text: str,
    ):
        """Initialise the row-scan mock backend."""
        self.name = name
        self._row_regex = row_regex
        self._intent_builder = intent_builder
        self._idle_body = idle_body
        self._raw_text = raw_text
        self.calls: list[dict[str, Any]] = []
        # Track handled rows so reactor fan-out re-renders don't double-emit.
        self._answered_ids: set[str] = set()

    async def run(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        tools: list[str] | None = None,
        max_turns: int = 1,
    ) -> BackendTurnResult:
        """Emit one intent per not-yet-seen matched row, else an idle message."""
        self.calls.append({"prompt": prompt})
        intents: list[Intent] = []
        for match in self._row_regex.finditer(prompt):
            key = match.group(2)
            if key in self._answered_ids:
                continue
            self._answered_ids.add(key)
            intents.append(self._intent_builder(match))
        if not intents:
            intents.append(
                Intent(
                    type=IntentType.SEND_MESSAGE,
                    payload={"topic": "observation", "body_md": self._idle_body},
                )
            )
        return BackendTurnResult(intents=intents, raw_text=self._raw_text)


def MockCriticBackend(name: str = "critic-mock") -> MockRowScanBackend:
    """Build the always-approve mock Critic backend."""

    def _approve(match: re.Match[str]) -> Intent:
        msg_id = match.group(2)
        return Intent(
            type=IntentType.REVIEW_VERDICT,
            payload={
                "target_proposal_msg_id": msg_id,
                "verdict": "approve",
                "reasoning": "(mock critic — auto-approve)",
                "source": "mock",
            },
        )

    return MockRowScanBackend(
        name=name,
        row_regex=_PROPOSAL_RE,
        intent_builder=_approve,
        idle_body="ok (mock critic, no proposals)",
        raw_text="(mock critic)",
    )


__all__ = [
    "MockCriticBackend",
    "MockRowScanBackend",
]
