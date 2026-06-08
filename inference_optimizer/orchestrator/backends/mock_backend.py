# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""MockBackend — scripted-turn LLM stub for P0 / unit / e2e tests.

Avoids burning tokens during dev, makes tests deterministic and offline.
The Coordinator / SubAgentRunner / Critic Review tests all use this.

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
    """Sequence of pre-recorded turns. Loops if ``loop_last=True``.

    If ``default_intent`` is provided, it's used after the script is
    exhausted (only when ``loop_last`` is False).
    """

    turns: list[MockTurn]
    loop_last: bool = False
    default_intent: Intent | None = None

    @classmethod
    def from_intents(cls, *intents_per_turn: Iterable[Intent]) -> "ScriptedPlan":
        """Build a plan from one intent iterable per scripted turn.

        Args:
            *intents_per_turn (Iterable[Intent]): One iterable of intents for
                each turn, in playback order.

        Returns:
            ScriptedPlan: A plan whose turns replay the given intent groups in
            order.
        """
        turns = [MockTurn(intents=list(its)) for its in intents_per_turn]
        return cls(turns=turns)


class MockBackend:
    """Implements :class:`Backend` by playing back a :class:`ScriptedPlan`."""

    def __init__(self, plan: ScriptedPlan, *, name: str = "mock"):
        """Initialise the mock backend with a scripted plan.

        Args:
            plan (ScriptedPlan): The sequence of turns to play back.
            name (str): Human-readable backend name used in logs and metadata.
        """
        self.plan = plan
        self.name = name
        self._cursor = 0
        # Recorded so tests can assert what the reactor sent us
        self.calls: list[dict[str, Any]] = []

    async def run(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        tools: list[str] | None = None,
        max_turns: int = 1,
    ) -> BackendTurnResult:
        """Record the call and play back the next scripted turn.

        Args:
            prompt (str): The composed turn prompt (recorded for assertions).
            system_prompt (str | None): Optional system prompt (recorded).
            tools (list[str] | None): Optional tool names (recorded).
            max_turns (int): Maximum sub-turns (recorded).

        Returns:
            BackendTurnResult: The intents, raw text, and metadata of the next
            scripted turn.

        Raises:
            BaseException: Whatever ``MockTurn.raise_error`` holds for the
                played-back turn, to simulate backend failures.
        """
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
        """Return the next turn to play, applying loop/default/fallback rules.

        Advances the cursor while scripted turns remain. Once exhausted, repeats
        the last turn when ``loop_last`` is set, otherwise replays
        ``default_intent`` when present, and finally falls back to a heartbeat
        turn so the reactor keeps ticking.

        Returns:
            MockTurn: The turn to replay for this invocation.
        """
        if self._cursor < len(self.plan.turns):
            t = self.plan.turns[self._cursor]
            self._cursor += 1
            return t
        if self.plan.loop_last and self.plan.turns:
            return self.plan.turns[-1]
        if self.plan.default_intent is not None:
            return MockTurn(intents=[self.plan.default_intent])
        # Out of script and no fallback → silently emit a heartbeat so the
        # reactor can keep ticking without exploding mid-test.
        from ...protocol.intent import Intent as _Intent
        from ...protocol.intent import IntentType as _IT
        return MockTurn(intents=[
            _Intent(type=_IT.SEND_MESSAGE, payload={"topic": "heartbeat", "body_md": "ok"}),
        ])

    @property
    def remaining_turns(self) -> int:
        """Number of scripted turns not yet played.

        Returns:
            int: Count of remaining scripted turns (zero once exhausted).
        """
        return max(0, len(self.plan.turns) - self._cursor)


__all__ = ["MockBackend", "MockTurn", "ScriptedPlan"]
