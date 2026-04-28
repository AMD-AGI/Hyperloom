"""MockBackend — scripted ``Intent`` emission for tests / dry-run.

Two modes of use:

1. **Default (no script)**: emits one ``send_message`` per call so the dry-run
   keeps producing heartbeats until the clock fires ``time_exhausted``.

2. **Scripted**: pass a list of :class:`ScriptStep`. Each call pops the next
   step in order. After the script is exhausted, falls back to default mode.

The mock deliberately bypasses :func:`parse_claude_trajectory` /
:func:`parse_codex_validated_json` because those are still stubs (see
IMPLEMENTATION-CHECKLIST Phase 2). Once those land, ClaudeBackend /
CodexBackend slot in alongside this with no Conductor changes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from ..intent_parser import Intent, IntentType
from .base import Backend


@dataclass
class ScriptStep:
    """One scripted reply: the mock returns ``intents`` on the matching call."""

    intents: list[Intent]
    only_if_agent: str | None = None  # None = matches any agent
    note: str = ""  # for human-readable test logs


@dataclass
class MockBackend(Backend):
    """Scripted intents for offline runs."""

    script: list[ScriptStep] = field(default_factory=list)
    default_topic: str = "heartbeat"
    calls: list[dict] = field(default_factory=list)

    async def run(
        self,
        prompt: str,
        *,
        agent_name: str,
        allowed_tools: Sequence[str] = (),
        max_turns: int = 10,
        extra: dict | None = None,
    ) -> list[Intent]:
        self.calls.append(
            {
                "agent": agent_name,
                "prompt_chars": len(prompt),
                "allowed_tools": tuple(allowed_tools),
                "extra": dict(extra or {}),
            }
        )
        for i, step in enumerate(self.script):
            if step.only_if_agent and step.only_if_agent != agent_name:
                continue
            self.script.pop(i)
            return list(step.intents)
        return self._default_intents(agent_name)

    def _default_intents(self, agent_name: str) -> list[Intent]:
        return [
            Intent(
                type=IntentType.SEND_MESSAGE,
                payload={
                    "to": "*",
                    "topic": self.default_topic,
                    "body_md": f"[mock] {agent_name} alive call#{len(self.calls)}",
                    "priority": 1,
                },
            )
        ]
