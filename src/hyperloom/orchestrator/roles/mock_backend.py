# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""MockBackend — scripted-turn LLM stub for unit / e2e tests.

Deterministic, offline, token-free playback of pre-recorded turns.

Usage::

    plan = ScriptedPlan([
        MockTurn(intents=[
            Intent(IntentType.PROPOSE_ACTION, payload={...}),
        ]),
        MockTurn(intents=[
            Intent(IntentType.REQUEST, payload={"target_agent": "kernel_agent", ...}),
        ]),
    ])
    backend = MockBackend(plan)
    result = await backend.run(prompt="...")  # → first MockTurn
    result = await backend.run(prompt="...")  # → second MockTurn
"""

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
    """Sequence of pre-recorded turns.

    ``loop_last`` repeats the final turn after the script is exhausted;
    otherwise ``default_intent`` (if set) is used.
    """

    turns: list[MockTurn]
    loop_last: bool = False
    default_intent: Intent | None = None


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
        # Out of script and no fallback → emit a heartbeat so the reactor keeps ticking.
        return MockTurn(
            intents=[
                Intent(type=IntentType.SEND_MESSAGE, payload={"topic": "heartbeat", "body_md": "ok"}),
            ]
        )


# Coordinator inbox row format: ``seq=<n> msg_id=<hex> from=<agent> topic=<t> payload=<...>``.
# Group 2 is the msg_id used for dedup; group 4 is the raw payload string.
_PROPOSAL_RE = re.compile(
    r"^\s*seq=(\d+)\s+msg_id=([a-f0-9]+)\s+from=(\w+)\s+topic=proposal\s+payload=(.*)$",
    re.MULTILINE,
)
_REQUEST_RE = re.compile(
    r"^\s*seq=(\d+)\s+msg_id=([a-f0-9]+)\s+from=(\w+)\s+topic=request\s+payload=(.*)$",
    re.MULTILINE,
)
_KIND_RE = re.compile(r"['\"]kind['\"]\s*:\s*['\"]([\w-]+)['\"]")


class MockRowScanBackend:
    """Row-scanning reactor mock: one intent per matched inbox row, else heartbeat.

    Generalises the always-approve Critic and auto-respond Kernel mocks, which
    share the same shape: scan the rendered inbox in ``prompt`` for rows matching
    ``row_regex`` and emit one intent (built by ``intent_builder``) per
    not-yet-seen row — keyed by ``dedup_key`` (msg_id by default) so reactor
    fan-out re-renders don't double-emit. When no row matches, emit a single
    heartbeat ``send_message`` so the reactor loop always sees signal of life.
    Use :func:`auto_approve_critic` / :func:`auto_respond_kernel` to build the
    concrete role mocks. Implements :class:`Backend`.
    """

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
        """Initialise the row-scan mock backend.

        Args:
            name (str): Human-readable backend name used in logs and metadata.
            row_regex (re.Pattern[str]): Multiline regex matched against the
                rendered inbox; each match yields one intent.
            intent_builder (Callable[[re.Match[str]], Intent]): Builds the intent
                for a matched (not-yet-seen) row.
            heartbeat_body (str): ``body_md`` of the fallback heartbeat message
                emitted when no row matches.
            raw_text (str): Raw text stamped on the returned turn result.
            dedup_key (Callable[[re.Match[str]], str]): Extracts the dedup key
                from a match (defaults to the msg_id capture group).
        """
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
        max_turns: int = 1,
    ) -> BackendTurnResult:
        """Emit one intent per not-yet-seen matched row, else a heartbeat.

        Args:
            prompt (str): The composed turn prompt containing the rendered inbox.
            system_prompt (str | None): Unused; accepted for protocol parity.
            tools (list[str] | None): Unused; accepted for protocol parity.
            max_turns (int): Unused; accepted for protocol parity.

        Returns:
            BackendTurnResult: The per-row intents and/or heartbeat for this turn.
        """
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
    """Build the always-approve mock Critic backend.

    Emits ``review_verdict{verdict="approve"}`` per visible proposal row, or a
    heartbeat when none are present.

    Args:
        name (str): Human-readable backend name used in logs and metadata.

    Returns:
        MockRowScanBackend: A backend configured to auto-approve proposals.
    """

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


def auto_respond_kernel(name: str = "kernel-mock") -> MockRowScanBackend:
    """Build the auto-respond mock Kernel backend.

    Emits one ``response{status="ok"}`` per visible request row (echoing the
    request ``kind`` as ``<kind>_done``), or a heartbeat when none are present.

    Args:
        name (str): Human-readable backend name used in logs and metadata.

    Returns:
        MockRowScanBackend: A backend configured to auto-respond to requests.
    """

    def _respond(match: re.Match[str]) -> Intent:
        msg_id = match.group(2)
        raw_payload = match.group(4)
        kind_match = _KIND_RE.search(raw_payload)
        kind = kind_match.group(1) if kind_match else "unknown"
        return Intent(
            type=IntentType.RESPONSE,
            payload={
                "in_reply_to": msg_id,
                "kind": f"{kind}_done",
                "status": "ok",
                "result": {"source": "mock", "chosen": ["mock_kernel_1"]},
            },
        )

    return MockRowScanBackend(
        name=name,
        row_regex=_REQUEST_RE,
        intent_builder=_respond,
        heartbeat_body="ok (mock kernel)",
        raw_text="(mock kernel)",
    )


__all__ = [
    "MockBackend",
    "MockRowScanBackend",
    "MockTurn",
    "ScriptedPlan",
    "auto_approve_critic",
    "auto_respond_kernel",
]
