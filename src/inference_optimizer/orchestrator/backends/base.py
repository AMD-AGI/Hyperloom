"""Backend ABC — DESIGN §5.1 / §10.5.

A reactor calls ``backend.run(...)`` with a fully composed prompt and a
list of allowed tools. The backend is responsible for whatever it takes
(Claude tool_use, Codex validated_json_output, mock scripts) to return
a list of validated :class:`Intent` objects.

This abstraction lets us:
    - swap MockBackend for ClaudeBackend without touching the Conductor;
    - run end-to-end tests offline (no API keys);
    - extend with a CodexBackend later for the no-tools roles.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Sequence

from ..intent_parser import Intent


class BackendError(RuntimeError):
    """Raised when the backend fails irrecoverably (no intents, parse fail)."""


@dataclass
class BackendCall:
    """One invocation record (used for tests + telemetry)."""

    agent_name: str
    prompt: str
    allowed_tools: tuple[str, ...] = ()
    max_turns: int = 10
    extra: dict = field(default_factory=dict)


class Backend(ABC):
    """Anything that produces :class:`Intent` from a prompt."""

    @abstractmethod
    async def run(
        self,
        prompt: str,
        *,
        agent_name: str,
        allowed_tools: Sequence[str] = (),
        max_turns: int = 10,
        extra: dict | None = None,
    ) -> list[Intent]:
        """Return a non-empty list of validated intents."""
