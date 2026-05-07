"""Backend protocol — what the Coordinator needs from any LLM provider.

Concrete implementations (Claude SDK, Codex, multi-CLI bridge, mock)
return a :class:`BackendTurnResult` carrying the intents emitted in this
turn. The Coordinator handles validation, PolicyGate, and persistence —
backends only need to produce intents.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from ..intent_parser import Intent


class BackendError(RuntimeError):
    """Backend invocation failed (network, schema, etc.).

    Coordinator catches this, surfaces a ``policy_denied``-style observation
    so the next reactor turn sees the failure context.
    """


@dataclass
class BackendTurnResult:
    """One turn's output from a backend."""

    intents: list[Intent] = field(default_factory=list)
    raw_text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Backend(Protocol):
    """Async LLM backend protocol used by the Coordinator reactor loop.

    Backends are stateful (they hold conversation continuation, tool
    config, etc.) but each ``run`` invocation is one logical turn — given
    a prompt it produces a :class:`BackendTurnResult`.
    """

    async def run(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        tools: list[str] | None = None,
        max_turns: int = 1,
    ) -> BackendTurnResult: ...


__all__ = ["Backend", "BackendError", "BackendTurnResult"]
