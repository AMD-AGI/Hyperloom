# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What the body lends a brain for one call: a model, and an agent that reads and edits the framework source.

There is no port that measures. A brain may think, read and write code; whether a candidate is any good is what
the body finds out after the call, so the same brain runs unchanged against the live body and an offline dataset.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, runtime_checkable

from hyperloom.brain_api.state import PatchFile
from hyperloom.brain_api.wire import json_mapping


class PortError(RuntimeError):
    """A port call that did not complete; ``retryable`` says whether the same call may succeed again."""

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


@dataclass(frozen=True)
class LlmRequest:
    """One model call; ``seed`` pins sampling where the backend supports it."""

    prompt: str
    system_prompt: str = ""
    max_tokens: int = 0
    seed: int | None = None


@dataclass(frozen=True)
class LlmReply:
    """The model's text, why it stopped, and what it cost."""

    text: str
    finish_reason: str = ""
    usage: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "usage", json_mapping(self.usage, "LlmReply.usage"))


@dataclass(frozen=True)
class AgentRequest:
    """A tool-using agent run on a scratch copy of the framework source.

    ``writes_source`` lets the agent edit the copy; its edits come back as patches. The body chooses the agent's
    tools; none of them launches a server or runs a benchmark.
    """

    prompt: str
    system_prompt: str = ""
    max_turns: int = 0
    wall_budget_sec: float = 0.0
    writes_source: bool = False
    seed: int | None = None


@dataclass(frozen=True)
class AgentReply:
    """The agent's final text, the source edits it left, and why it stopped."""

    text: str
    patches: tuple[PatchFile, ...] = ()
    finish_reason: str = ""
    usage: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "patches", tuple(self.patches))
        object.__setattr__(self, "usage", json_mapping(self.usage, "AgentReply.usage"))


@runtime_checkable
class Ports(Protocol):
    """The ports of one call. The body counts their cost and holds their credentials."""

    async def llm(self, request: LlmRequest) -> LlmReply:
        """Run one model call; raises :class:`PortError`."""
        ...

    async def agent(self, request: AgentRequest) -> AgentReply:
        """Run one agent; raises :class:`PortError`."""
        ...
