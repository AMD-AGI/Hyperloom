# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Process-wide observers of the single HTTP LLM requests the ``llm_config`` helpers issue.

``hyperloom.common`` may not import the trace packages, so a ledger that wants one row per gateway request registers
an observer here. An observer fault never reaches the caller's request.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

PROTOCOL_OPENAI_CHAT = "openai_chat"
PROTOCOL_ANTHROPIC_MESSAGES = "anthropic_messages"

_ERROR_MESSAGE_MAX = 500


@dataclass(frozen=True)
class LLMRequestRecord:
    """One finished (or failed) HTTP request; ``usage`` uses Hyperloom's uncached ``input_tokens`` semantics."""

    protocol: str
    component: str
    operation: str
    model: str | None
    response_id: str | None
    start: float
    end: float
    first_token: float | None
    stop_reason: str | None
    usage: dict[str, int]
    streamed: bool
    error_type: str | None = None
    error_message: str | None = None


LLMRequestObserver = Callable[[LLMRequestRecord], None]

_OBSERVERS: list[LLMRequestObserver] = []


def add_llm_request_observer(observer: LLMRequestObserver) -> None:
    """Register ``observer`` for every later request; registering the same callable twice is a no-op."""
    if observer not in _OBSERVERS:
        _OBSERVERS.append(observer)


def remove_llm_request_observer(observer: LLMRequestObserver) -> None:
    """Drop ``observer`` if registered."""
    if observer in _OBSERVERS:
        _OBSERVERS.remove(observer)


def _notify(record: LLMRequestRecord) -> None:
    for observer in tuple(_OBSERVERS):
        try:
            observer(record)
        except Exception:
            log.debug("llm request observer %r failed", observer, exc_info=True)


@dataclass
class RequestObservation:
    """Mutable facts about one in-flight request, filled in by the helper that issues it."""

    protocol: str
    component: str
    operation: str
    model: str | None
    streamed: bool
    clock: Callable[[], float] = time.time
    start: float = 0.0
    first_token: float | None = None
    response_id: str | None = None
    stop_reason: str | None = None
    usage: dict[str, int] = field(default_factory=dict)

    def mark_first_token(self) -> None:
        if self.first_token is None:
            self.first_token = self.clock()

    def update(
        self,
        *,
        response_id: Any = None,
        model: Any = None,
        stop_reason: Any = None,
        usage: dict[str, int] | None = None,
    ) -> None:
        """Fold response fields in; ``None`` never overwrites a value already seen."""
        if isinstance(response_id, str) and response_id:
            self.response_id = response_id
        if isinstance(model, str) and model:
            self.model = model
        if isinstance(stop_reason, str) and stop_reason:
            self.stop_reason = stop_reason
        if usage:
            self.usage.update(usage)

    def _record(self, error: BaseException | None) -> LLMRequestRecord:
        return LLMRequestRecord(
            protocol=self.protocol,
            component=self.component,
            operation=self.operation,
            model=self.model,
            response_id=self.response_id,
            start=self.start,
            end=self.clock(),
            first_token=self.first_token,
            stop_reason=self.stop_reason,
            usage=dict(self.usage),
            streamed=self.streamed,
            error_type=None if error is None else type(error).__name__,
            error_message=None if error is None else str(error)[:_ERROR_MESSAGE_MAX],
        )


@contextmanager
def observed_request(
    *,
    protocol: str,
    component: str,
    operation: str,
    model: Any,
    streamed: bool = False,
) -> Iterator[RequestObservation]:
    """Time the enclosed request and hand its record to every observer, failed when the body raises."""
    observation = RequestObservation(
        protocol=protocol,
        component=component,
        operation=operation,
        model=model if isinstance(model, str) and model else None,
        streamed=streamed,
    )
    if not _OBSERVERS:
        yield observation
        return
    observation.start = observation.clock()
    try:
        yield observation
    except BaseException as exc:
        _notify(observation._record(exc))
        raise
    _notify(observation._record(None))


__all__ = [
    "LLMRequestObserver",
    "LLMRequestRecord",
    "PROTOCOL_ANTHROPIC_MESSAGES",
    "PROTOCOL_OPENAI_CHAT",
    "RequestObservation",
    "add_llm_request_observer",
    "observed_request",
    "remove_llm_request_observer",
]
