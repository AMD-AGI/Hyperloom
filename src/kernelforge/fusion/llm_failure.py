# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tell "the model never answered" apart from "the model answered nothing".

Discovery asks the model one question — which op chains in this model are worth
fusing — and its answer decides the run's verdict. An empty answer and an
unasked question are indistinguishable once they reach the caller as ``""``, so
conflating them turns a gateway outage into a published ``no_opportunity`` on a
model the diagnosis just flagged as launch-bound. The task exits 0, the manifest
looks normal, and no failure dashboard shows anything.

So a call that never reached the model raises :class:`LlmUnavailableError`
instead of returning a string, and the caller is forced to decide what that
means rather than defaulting into a business conclusion.
"""

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
# Wall-clock ceiling for the whole retry chain. The attempt count alone does not
# bound it: each attempt may sit on the client's own read timeout (900s by
# default), so five of them could hold discovery for over an hour. 0 lifts it.
DEFAULT_DEADLINE_SEC = 1800.0

# Kinds a retry can still fix. A timeout belongs here: it means the request never
# came back THIS time, which is exactly the transient gateway degradation the
# retry chain exists for. Excluding it dropped the retry the previous
# implementation had, and turned a single slow response into a published
# "the model was unreachable".
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

# Attribute an agent backend sets truthy on the exception it raises for a
# workspace-safety VERDICT, and falsy on the same exception class raised because
# the guard could not read or query the workspace. Re-exported rather than
# redeclared: it is published with the provider base classes that have to set it,
# where a backend outside this repository can find it, and one spelling means the
# producer and the consumer cannot drift apart.
from kernelforge.agent_backends.base import AGENT_SAFETY_REJECTION_ATTR


class LlmUnavailableError(RuntimeError):
    """The model was never reached, so the run learned nothing.

    Distinct from an empty proposal list on purpose: this is a fact about the
    gateway, never about the kernel being analyzed.
    """

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
    """Classify why a completion failed, deciding whether a retry can help.

    Everything is treated as a transient ``api_error`` except credentials and an
    over-long prompt. A bare, reason-less 400 — which is what the AMD Vertex
    path returns while it is degraded — is indistinguishable from a genuinely
    malformed request, and the costs are not symmetric: retrying a malformed
    request wastes four calls and still ends in "never answered", while giving
    up on a transient one publishes a wrong verdict about a real model.
    """
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
    """Recognize a provider's workspace-safety VERDICT through wrapper chains.

    A backend raises its safety class for two unrelated things: a verdict about
    what the session did to the workspace, which is identical on every retry, and
    a failure of the guard's own bookkeeping -- a snapshot it could not read, a Git
    query that timed out on NFS -- which is weather. Matching the class name made
    the second one fatal, so a stalled ``git`` call abandoned a recipe. The
    provider therefore marks the verdict explicitly with
    ``AGENT_SAFETY_REJECTION_ATTR``; an attribute rather than a shared base class so
    no fusion stage has to import a provider package to classify one of its
    errors. Walked through ``__cause__``/``__context__`` because the backends
    flatten these into wrappers of their own. Shared by discovery and authoring:
    both have to refuse to retry a rejection that is decided the same way
    every time.
    """
    return any(bool(getattr(current, AGENT_SAFETY_REJECTION_ATTR, False)) for current in _error_chain(error))


def is_agent_timeout_error(error: BaseException) -> bool:
    """Whether a failed agent run ran out of clock, seen through the same chain.

    Chain-aware for the same reason :func:`is_agent_safety_error` is: a backend
    that times out runs its rollback on the way out, and a rollback that itself
    fails replaces the timeout with its own exception, leaving the expired clock
    visible only in ``__context__``.
    """
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
    """Exponential backoff with full jitter, for a 1-based attempt number.

    The gateway degradations this rides out last minutes, so the ceiling has to
    grow past the ~30s window that four fixed 3s-step retries covered — that
    window was shorter than the outage every time it mattered. The jitter stops
    a whole batch of pods from retrying in lockstep against the gateway they
    are all waiting on.
    """
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
