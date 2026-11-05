# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Process-wide observers of the single HTTP LLM requests the ``llm_config`` helpers issue.

``hyperloom.common`` may not import the trace packages, so a ledger that wants one row per gateway request registers
an observer here. An observer's I/O or data fault never reaches the caller's request. The reader that folds a streamed Messages
reply back into one body lives here too: it is what marks an observation's first token.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Iterator, Mapping
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


def _notify(record: LLMRequestRecord) -> None:
    for observer in tuple(_OBSERVERS):
        try:
            observer(record)
        except (OSError, ValueError):
            log.debug("llm request observer %r failed", observer, exc_info=True)


@dataclass
class RequestObservation:
    """Mutable facts about one in-flight request, filled in by the helper that issues it."""

    protocol: str
    component: str
    operation: str
    model: str | None
    streamed: bool
    start: float = 0.0
    first_token: float | None = None
    response_id: str | None = None
    stop_reason: str | None = None
    usage: dict[str, int] = field(default_factory=dict)

    def mark_first_token(self) -> None:
        if self.first_token is None:
            self.first_token = time.time()

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
            end=time.time(),
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
    observation.start = time.time()
    try:
        yield observation
    except BaseException as exc:
        _notify(observation._record(exc))
        raise
    _notify(observation._record(None))


class AnthropicStream:
    """A streamed Messages reply, folded back into the body a whole reply carries.

    A line ends only at CR, LF or CRLF, as server-sent events end it: httpx's ``iter_lines`` also ends one at U+0085,
    U+2028 and U+2029, which JSON carries unescaped. A last line with no LF yet waits for more bytes, since its CR may
    be the first half of a CRLF. Bytes that arrive after ``deadline`` (``time.monotonic``) raise ``TimeoutError``.
    """

    def __init__(self, observation: RequestObservation, deadline: float | None = None) -> None:
        self._observation = observation
        self._deadline = deadline
        self._pending = b""
        self._data: list[bytes] = []
        self._body: dict[str, Any] = {}
        self._stopped = False

    def feed(self, chunk: bytes) -> None:
        if self._deadline is not None and time.monotonic() > self._deadline:
            raise TimeoutError("anthropic messages reply outlasted its timeout_s")
        lines = (self._pending + chunk).splitlines(keepends=True)
        self._pending = lines.pop() if lines and not lines[-1].endswith(b"\n") else b""
        for line in lines:
            self._line(line.rstrip(b"\r\n"))

    def finish(self) -> dict[str, Any]:
        """Take the last line and return the reply's body; a stream that ended before ``message_stop`` raises."""
        self._line(self._pending.rstrip(b"\r\n"))
        self._line(b"")
        if not self._stopped:
            raise RuntimeError("anthropic messages stream ended before message_stop")
        return self._body

    def _line(self, line: bytes) -> None:
        if line.startswith(b"data:"):
            self._data.append(line[len(b"data:") :].strip())
        elif not line and self._data:
            data, self._data = b"\n".join(self._data), []
            try:
                event = json.loads(data)
            except ValueError as exc:
                raise RuntimeError(f"anthropic messages stream carried a non-JSON event: {exc!r}") from exc
            try:
                self._fold(event)
            except (KeyError, IndexError, TypeError) as exc:
                raise RuntimeError(f"anthropic messages stream carried a malformed event: {data[:200]!r}") from exc

    def _fold(self, event: Mapping[str, Any]) -> None:
        kind = event["type"]
        if kind == "message_start":
            self._body = event["message"]
        elif kind == "content_block_start":
            self._body["content"].append(event["content_block"])
        elif kind == "content_block_delta" and event["delta"]["type"] == "text_delta":
            self._body["content"][event["index"]]["text"] += event["delta"]["text"]
            self._observation.mark_first_token()
        elif kind == "message_delta":
            self._body.update(event["delta"])
            self._body["usage"] = {**self._body["usage"], **event["usage"]}
        elif kind == "message_stop":
            self._stopped = True
        elif kind == "error":
            raise RuntimeError(
                f"anthropic messages stream error: {event['error']['type']}: {event['error']['message']}"
            )


__all__ = [
    "AnthropicStream",
    "LLMRequestObserver",
    "LLMRequestRecord",
    "PROTOCOL_ANTHROPIC_MESSAGES",
    "PROTOCOL_OPENAI_CHAT",
    "RequestObservation",
    "add_llm_request_observer",
    "observed_request",
]
