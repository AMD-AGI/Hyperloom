# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tell \"the model never answered\" apart from \"the model answered nothing\"."""

from __future__ import annotations

import asyncio
import logging
import os
import random
from typing import Any, Callable

log = logging.getLogger("forge_fusion")

# Why the model could not be reached. Only ``api_error`` recovers on its own.
API_ERROR = "api_error"
AUTH = "auth"
CONTEXT_LENGTH = "context_length"
NOT_CONFIGURED = "not_configured"
TIMEOUT = "timeout"

DEFAULT_ATTEMPTS = 5
DEFAULT_BASE_DELAY_SEC = 5.0
DEFAULT_MAX_DELAY_SEC = 120.0
_DELAY_FACTOR = 3.0
# Wall-clock ceiling for the whole retry chain.
DEFAULT_DEADLINE_SEC = 1800.0

# Kinds a retry can still fix.
RETRYABLE_KINDS = frozenset({API_ERROR, TIMEOUT})

_AUTH_MARKERS = (
    "unauthorized",
    "forbidden",
    "invalid api key",
    "invalid_api_key",
    "authentication",
    "missing subscription key",
    "permission denied",
)
_CONTEXT_MARKERS = (
    "context length",
    "context_length",
    "prompt is too long",
    "maximum context",
    "too many total text bytes",
)
_TIMEOUT_MARKERS = ("timed out", "timeout")

# Attribute an agent backend sets truthy on the exception it raises for a workspace-safety VERDICT, and falsy on the
# same exception class raised because the guard could not read or query the workspace.
from kernelforge.agent_backends.base import AGENT_SAFETY_REJECTION_ATTR


class LlmUnavailableError(RuntimeError):
    """The model was never reached, so the run learned nothing."""

    def __init__(self, message: str, *, kind: str = API_ERROR, attempts: int = 0) -> None:
        super().__init__(message)
        self.kind = kind
        self.attempts = attempts

    @property
    def retryable(self) -> bool:
        """Whether waiting longer could have produced an answer."""
        return self.kind in RETRYABLE_KINDS

    def to_dict(self) -> dict[str, Any]:
        """The machine-readable form embedded in the run manifest."""
        return {
            "stage": "discovery",
            "class": "llm_unavailable",
            "kind": self.kind,
            "attempts": self.attempts,
            "message": str(self)[:2000],
        }


def _status_code(error: BaseException) -> int | None:
    """HTTP status carried by an OpenAI-SDK style exception, when present."""
    for source in (error, getattr(error, "response", None)):
        status = getattr(source, "status_code", None)
        if isinstance(status, int):
            return status
    return None


def classify_llm_error(error: BaseException) -> str:
    """Classify why a completion failed, deciding whether a retry can help."""
    status = _status_code(error)
    if status in (401, 403):
        return AUTH
    if status == 413:
        return CONTEXT_LENGTH
    lowered = str(error).lower()
    if any(marker in lowered for marker in _AUTH_MARKERS):
        return AUTH
    if any(marker in lowered for marker in _CONTEXT_MARKERS):
        return CONTEXT_LENGTH
    if any(marker in lowered for marker in _TIMEOUT_MARKERS):
        return TIMEOUT
    return API_ERROR


def _error_chain(error: BaseException) -> list[BaseException]:
    """The error plus the wrappers the backends flattened it into, once each."""
    chain: list[BaseException] = []
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append(current)
        current = current.__cause__ or current.__context__
    return chain


def is_agent_safety_error(error: BaseException) -> bool:
    """Recognize a provider's workspace-safety VERDICT through wrapper chains."""
    return any(bool(getattr(current, AGENT_SAFETY_REJECTION_ATTR, False)) for current in _error_chain(error))


def is_agent_timeout_error(error: BaseException) -> bool:
    """Whether a failed agent run ran out of clock, seen through the same chain."""
    return any(
        isinstance(current, (asyncio.TimeoutError, TimeoutError)) or classify_llm_error(current) == TIMEOUT
        for current in _error_chain(error)
    )


def retry_delay(
    attempt: int,
    *,
    base_sec: float = DEFAULT_BASE_DELAY_SEC,
    max_sec: float = DEFAULT_MAX_DELAY_SEC,
    rng: Callable[[], float] = random.random,
) -> float:
    """Exponential backoff with full jitter, for a 1-based attempt number."""
    ceiling = min(max_sec, base_sec * (_DELAY_FACTOR ** max(0, attempt - 1)))
    return ceiling * (0.5 + 0.5 * rng())


def env_setting(name: str, default: float, *, cast: Callable[[str], Any]) -> Any:
    """Read one operator override, ignoring anything unparseable or negative."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = cast(raw)
    except ValueError:
        log.warning("ignoring unparseable %s=%r", name, raw)
        return default
    return value if value >= 0 else default


__all__ = [
    "AGENT_SAFETY_REJECTION_ATTR",
    "API_ERROR",
    "AUTH",
    "CONTEXT_LENGTH",
    "DEFAULT_ATTEMPTS",
    "DEFAULT_BASE_DELAY_SEC",
    "DEFAULT_DEADLINE_SEC",
    "DEFAULT_MAX_DELAY_SEC",
    "LlmUnavailableError",
    "NOT_CONFIGURED",
    "RETRYABLE_KINDS",
    "TIMEOUT",
    "classify_llm_error",
    "env_setting",
    "is_agent_safety_error",
    "is_agent_timeout_error",
    "retry_delay",
]
