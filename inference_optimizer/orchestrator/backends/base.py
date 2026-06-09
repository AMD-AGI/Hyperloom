# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Backend protocol — what the Coordinator needs from any LLM provider.

Concrete implementations return a :class:`BackendTurnResult` carrying the
turn's intents; the Coordinator handles validation, PolicyGate, persistence.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from ...protocol.intent import Intent

log = logging.getLogger(__name__)


def parse_call_timeout_env(env_name: str, *, default: float) -> float:
    """Read a per-call wall-clock timeout from ``env_name``, default on miss/error.

    Returns ``default`` (logging a WARNING) when the env var is unset, empty,
    or not a positive finite float — a malformed knob must not be fatal.
    """
    raw = os.environ.get(env_name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw)
    except ValueError:
        log.warning(
            "%s=%r is not a float; using default %.1fs",
            env_name, raw, default,
        )
        return default
    if value <= 0 or not math.isfinite(value):
        log.warning(
            "%s=%r is not a positive finite number; using default %.1fs",
            env_name, raw, default,
        )
        return default
    return value


class BackendError(RuntimeError):
    """Backend invocation failed (network, schema, etc.)."""


@dataclass
class BackendTurnResult:
    """One turn's output from a backend."""

    intents: list[Intent] = field(default_factory=list)
    raw_text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Backend(Protocol):
    """Async LLM backend protocol used by the Coordinator reactor loop.

    Backends are stateful but each ``run`` invocation is one logical turn.
    """

    async def run(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        tools: list[str] | None = None,
        max_turns: int = 1,
    ) -> BackendTurnResult: ...


__all__ = ["Backend", "BackendError", "BackendTurnResult", "parse_call_timeout_env"]
