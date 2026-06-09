# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""MockBackend — scripted-turn LLM stub for P0 / unit / e2e tests.

Deterministic, offline, token-free playback of pre-recorded turns.

Usage::

    plan = ScriptedPlan([
        MockTurn(intents=[
            Intent(IntentType.PROPOSE_ACTION, payload={...}),
        ]),
        MockTurn(intents=[
            Intent(IntentType.REQUEST, payload={"target_agent": "kernel", ...}),
        ]),
    ])
    backend = MockBackend(plan)
    result = await backend.run(prompt="...")  # → first MockTurn
    result = await backend.run(prompt="...")  # → second MockTurn
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from ...protocol.intent import Intent
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
    """Sequence of pre-recorded turns.

    ``loop_last`` repeats the final turn after the script is exhausted;
    otherwise ``default_intent`` (if set) is used.
    """

    turns: list[MockTurn]
    loop_last: bool = False
    default_intent: Intent | None = None

    @classmethod
    def from_intents(cls, *intents_per_turn: Iterable[Intent]) -> "ScriptedPlan":
        turns = [MockTurn(intents=list(its)) for its in intents_per_turn]
        return cls(turns=turns)


class MockBackend:
    """Implements :class:`Backend` by playing back a :class:`ScriptedPlan`."""

    def __init__(self, plan: ScriptedPlan, *, name: str = "mock"):
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
        max_turns: int = 1,
    ) -> BackendTurnResult:
        self.calls.append({
            "prompt": prompt,
            "system_prompt": system_prompt,
            "tools": list(tools or []),
            "max_turns": max_turns,
        })
        turn = self._next_turn()
        if turn.raise_error is not None:
            raise turn.raise_error
        return BackendTurnResult(
            intents=list(turn.intents),
            raw_text=turn.raw_text,
            metadata=dict(turn.metadata),
        )

    def _next_turn(self) -> MockTurn:
        if self._cursor < len(self.plan.turns):
            t = self.plan.turns[self._cursor]
            self._cursor += 1
            return t
        if self.plan.loop_last and self.plan.turns:
            return self.plan.turns[-1]
        if self.plan.default_intent is not None:
            return MockTurn(intents=[self.plan.default_intent])
        # Out of script and no fallback → emit a heartbeat so the reactor keeps ticking.
        from ...protocol.intent import Intent as _Intent
        from ...protocol.intent import IntentType as _IT
        return MockTurn(intents=[
            _Intent(type=_IT.SEND_MESSAGE, payload={"topic": "heartbeat", "body_md": "ok"}),
        ])

    @property
    def remaining_turns(self) -> int:
        return max(0, len(self.plan.turns) - self._cursor)


__all__ = ["MockBackend", "MockTurn", "ScriptedPlan"]
