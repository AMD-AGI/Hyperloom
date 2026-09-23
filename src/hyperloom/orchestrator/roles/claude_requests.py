# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Per-request usage and timing for one Claude Agent SDK ``query()`` call.

One call makes one model request per tool round trip, and the call-level ``ResultMessage.usage`` only reports their
sum. With partial messages enabled, each request streams ``message_start`` .. ``message_stop``; without them, the
SDK's ``AssistantMessage`` rows (one per content block, sharing a message id) are the fallback. A request is timed
from the boundary that let it start (the stream opening, the previous request ending, or a tool result arriving).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from ..trace.trajectory_trace import (
    EVENT_LLM_REQUEST,
    STATUS_COMPLETED,
    STATUS_FAILED,
    record_event,
)
from .base import safe_int

TIMING_STREAM = "stream_events"
TIMING_ASSISTANT_MESSAGE = "assistant_message"

_USAGE_KEYS: tuple[str, ...] = (
    "input_tokens",
    "output_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
)


def _usage_counts(usage: Any) -> dict[str, int]:
    """The canonical counters a usage mapping reports (absent keys are left out, so a later delta cannot zero them)."""
    if not isinstance(usage, dict):
        return {}
    return {key: safe_int(usage.get(key)) for key in _USAGE_KEYS if usage.get(key) is not None}


def _iso(epoch_s: float) -> str:
    return datetime.fromtimestamp(epoch_s, timezone.utc).isoformat(timespec="microseconds")


def _elapsed_ms(start: float, end: float | None) -> int | None:
    return None if end is None else max(0, int((end - start) * 1000))


@dataclass
class _Request:
    message_id: str | None
    model: str | None
    start: float
    timing_source: str
    first_token: float | None = None
    end: float | None = None
    stop_reason: str | None = None
    usage: dict[str, int] = field(default_factory=dict)


class ClaudeRequestTracker:
    """Fold one call's SDK message stream into per-request usage and timing."""

    def __init__(self, *, clock: Callable[[], float] = time.time) -> None:
        self._clock = clock
        self._boundary = clock()
        # Keyed by ``parent_tool_use_id``, so a sub-agent's stream cannot close the parent's request.
        self._open: dict[str | None, _Request] = {}
        self._requests: list[_Request] = []
        self._by_id: dict[str, _Request] = {}

    def observe(self, message: Any) -> None:
        """Account one SDK message."""
        kind = type(message).__name__
        if kind == "StreamEvent":
            self._on_stream_event(getattr(message, "event", None), getattr(message, "parent_tool_use_id", None))
        elif kind == "AssistantMessage":
            self._on_assistant_message(message)
        elif kind == "UserMessage":
            self._boundary = self._clock()

    def _remember(self, request: _Request) -> None:
        self._requests.append(request)
        if request.message_id:
            self._by_id[request.message_id] = request

    def _on_stream_event(self, event: Any, parent: str | None) -> None:
        if not isinstance(event, dict):
            return
        event_type = event.get("type")
        now = self._clock()
        if event_type == "message_start":
            message = event.get("message") if isinstance(event.get("message"), dict) else {}
            request = _Request(
                message_id=message.get("id"),
                model=message.get("model"),
                start=self._boundary,
                timing_source=TIMING_STREAM,
            )
            request.usage.update(_usage_counts(message.get("usage")))
            self._open[parent] = request
            self._remember(request)
            return
        request = self._open.get(parent)
        if request is None:
            return
        if event_type == "content_block_delta" and request.first_token is None:
            request.first_token = now
        elif event_type == "message_delta":
            request.usage.update(_usage_counts(event.get("usage")))
            delta = event.get("delta") if isinstance(event.get("delta"), dict) else {}
            request.stop_reason = delta.get("stop_reason") or request.stop_reason
        elif event_type == "message_stop":
            request.end = now
            self._open.pop(parent, None)
            self._boundary = now

    def _on_assistant_message(self, message: Any) -> None:
        message_id = getattr(message, "message_id", None)
        if not isinstance(message_id, str) or not message_id:
            return
        now = self._clock()
        request = self._by_id.get(message_id)
        if request is None:
            request = _Request(
                message_id=message_id,
                model=getattr(message, "model", None),
                start=self._boundary,
                timing_source=TIMING_ASSISTANT_MESSAGE,
            )
            self._remember(request)
        if request.timing_source == TIMING_ASSISTANT_MESSAGE:
            request.end = now
            request.usage.update(_usage_counts(getattr(message, "usage", None)))
            request.stop_reason = getattr(message, "stop_reason", None) or request.stop_reason
        request.model = request.model or getattr(message, "model", None)

    def records(self) -> list[dict[str, Any]]:
        """One dict per request seen, in order; ``complete`` is False for a request the stream cut off."""
        out: list[dict[str, Any]] = []
        for index, request in enumerate(self._requests):
            out.append(
                {
                    "request_index": index,
                    "message_id": request.message_id,
                    "model": request.model,
                    "timing_source": request.timing_source,
                    "complete": request.end is not None,
                    "start": request.start,
                    "end": request.end,
                    "latency_ms": _elapsed_ms(request.start, request.end),
                    "ttft_ms": _elapsed_ms(request.start, request.first_token),
                    "stop_reason": request.stop_reason,
                    **request.usage,
                }
            )
        return out

    def record_trajectory(self, *, attempt: int, fallback_model: str | None) -> None:
        """Append one ``llm.request`` trajectory row per request (a no-op outside a trajectory scope)."""
        now = self._clock()
        for record in self.records():
            start, end = record.pop("start"), record.pop("end")
            model = record["model"] or fallback_model
            record_event(
                EVENT_LLM_REQUEST,
                status=STATUS_COMPLETED if record["complete"] else STATUS_FAILED,
                start_ts=_iso(start),
                ts=_iso(end if end is not None else now),
                attributes={**record, "name": model, "model": model, "attempt": attempt},
            )


__all__ = [
    "ClaudeRequestTracker",
    "TIMING_ASSISTANT_MESSAGE",
    "TIMING_STREAM",
]
