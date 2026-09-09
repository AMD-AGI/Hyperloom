# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""MockBackend — scripted-turn LLM stub for unit / e2e tests."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable

from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType
from .base import BackendTurnResult


@dataclass
class MockTurn:
    """One scripted turn the mock backend will play back."""

    intents: list[Intent] = field(default_factory=list)
    raw_text: str = "(mock turn)"
    raise_error: BaseException | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScriptedPlan:
    """Sequence of pre-recorded turns."""

    turns: list[MockTurn]
    loop_last: bool = False
    default_intent: Intent | None = None


class MockBackend:
    """Implements :class:`Backend` by playing back a :class:`ScriptedPlan`."""

    def __init__(self, plan: ScriptedPlan, *, name: str = "mock"):
        """Initialise the mock backend with a scripted plan."""
        self.plan = plan
        self.name = name
        self._cursor = 0
        self.calls: list[dict[str, Any]] = []

    async def run(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        tools: list[str] | None = None,
        disallowed_tools: list[str] | None = None,
        max_turns: int = 1,
    ) -> BackendTurnResult:
        """Record the call and play back the next scripted turn."""
        self.calls.append(
            {
                "prompt": prompt,
                "system_prompt": system_prompt,
                "tools": list(tools or []),
                "max_turns": max_turns,
            }
        )
        turn = self._next_turn()
        if turn.raise_error is not None:
            raise turn.raise_error
        return BackendTurnResult(
            intents=list(turn.intents),
            raw_text=turn.raw_text,
            metadata=dict(turn.metadata),
        )

    def _next_turn(self) -> MockTurn:
        """Return the next turn to play, applying loop/default/fallback rules."""
        if self._cursor < len(self.plan.turns):
            t = self.plan.turns[self._cursor]
            self._cursor += 1
            return t
        if self.plan.loop_last and self.plan.turns:
            return self.plan.turns[-1]
        if self.plan.default_intent is not None:
            return MockTurn(intents=[self.plan.default_intent])
        # Out of script and no fallback → emit a heartbeat so the reactor keeps ticking.
        return MockTurn(
            intents=[
                Intent(type=IntentType.SEND_MESSAGE, payload={"topic": "heartbeat", "body_md": "ok"}),
            ]
        )


# Coordinator inbox row format: ``seq=<n> msg_id=<hex> from=<agent> topic=<t> payload=<...>``.
_PROPOSAL_RE = re.compile(
    r"^\s*seq=(\d+)\s+msg_id=([a-f0-9]+)\s+from=(\w+)\s+topic=proposal\s+payload=(.*)$",
    re.MULTILINE,
)


class MockRowScanBackend:
    """Row-scanning reactor mock: one intent per matched inbox row, else heartbeat."""

    def __init__(
        self,
        *,
        name: str,
        row_regex: re.Pattern[str],
        intent_builder: Callable[[re.Match[str]], Intent],
        heartbeat_body: str,
        raw_text: str,
        dedup_key: Callable[[re.Match[str]], str] = lambda m: m.group(2),
    ):
        """Initialise the row-scan mock backend."""
        self.name = name
        self._row_regex = row_regex
        self._intent_builder = intent_builder
        self._heartbeat_body = heartbeat_body
        self._raw_text = raw_text
        self._dedup_key = dedup_key
        self.calls: list[dict[str, Any]] = []
        # Track handled rows so reactor fan-out re-renders don't double-emit.
        self._answered_ids: set[str] = set()

    async def run(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        tools: list[str] | None = None,
        disallowed_tools: list[str] | None = None,
        max_turns: int = 1,
    ) -> BackendTurnResult:
        """Emit one intent per not-yet-seen matched row, else a heartbeat."""
        self.calls.append({"prompt": prompt})
        intents: list[Intent] = []
        for match in self._row_regex.finditer(prompt):
            key = self._dedup_key(match)
            if key in self._answered_ids:
                continue
            self._answered_ids.add(key)
            intents.append(self._intent_builder(match))
        if not intents:
            intents.append(
                Intent(
                    type=IntentType.SEND_MESSAGE,
                    payload={"topic": "heartbeat", "body_md": self._heartbeat_body},
                )
            )
        return BackendTurnResult(intents=intents, raw_text=self._raw_text)


def auto_approve_critic(name: str = "critic-mock") -> MockRowScanBackend:
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
        heartbeat_body="ok (mock critic)",
        raw_text="(mock critic)",
    )


__all__ = [
    "MockBackend",
    "MockRowScanBackend",
    "MockTurn",
    "ScriptedPlan",
    "auto_approve_critic",
]
