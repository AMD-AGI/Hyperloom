# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The AgentX profile window, opened on the replay's phase clock rather than at a fixed offset from the client's start.

aiperf's setup and warmup take minutes and vary between replays, so a window at a fixed offset profiles the warmup
(1-2 requests running instead of the concurrency). The window opens ``agentx_open_after_sec`` after the first
profiling-phase request, the profiling phase's ``start_ns`` on aiperf's progress API, and closes
``agentx_duration_sec`` later or when the replay ends. When it opens it reads the server's running-request count once.
``profile/capture.json`` records the profiling start, both window instants, that count, and a control request the
server refused, which leaves no trace rather than failing the round.
"""

from __future__ import annotations

import http.client
import json
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import asdict, dataclass

from inference_suite.contracts.config import ProfileConfig
from inference_suite.contracts.layout import RunLayout
from inference_suite.frameworks import Framework
from inference_suite.profile.inject import start_body
from inference_suite.runtime.phases import Phase, PhaseClock
from inference_suite.telemetry.kv_poll import json_number, parse_prometheus_text, running_requests

#: aiperf's name for the measured phase on its progress API.
PROFILING_PHASE = "profiling"
START_PROFILE_PATH = "/start_profile"
STOP_PROFILE_PATH = "/stop_profile"
METRICS_PATH = "/metrics"
#: A control request answers once the profiler has started or flushed, not after the window.
_CONTROL_TIMEOUT_SEC = 30.0
_NS_PER_SEC = 1e9
#: A loopback request routed through the host's proxy would never reach the server.
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
_HTTP_ERRORS = (urllib.error.URLError, http.client.HTTPException, OSError)


def profiling_start_unix(phases: Mapping[str, object] | None) -> float | None:
    """When the replay sent its first profiling-phase request, from aiperf's progress ``phases``; ``None`` before."""
    stats = None if phases is None else phases.get(PROFILING_PHASE)
    start_ns = json_number(stats.get("start_ns")) if isinstance(stats, Mapping) else None
    return None if start_ns is None or start_ns <= 0 else start_ns / _NS_PER_SEC


def _request(url: str, body: Mapping[str, object] | None) -> bytes:
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with _OPENER.open(request, timeout=_CONTROL_TIMEOUT_SEC) as response:
        return response.read()


@dataclass(frozen=True)
class WindowCapture:
    """What the window did, as ``profile/capture.json`` records it; every instant is Unix seconds."""

    profiling_start_unix: float | None
    window_open_unix: float | None
    window_close_unix: float | None
    running_requests_in_window: int | None
    error: str | None

    def to_document(self) -> dict[str, object]:
        return asdict(self)


class AgentxWindow:
    """Opens and closes the server's profiler as the replay's phases go by; fed by the load at every poll."""

    def __init__(self, profile: ProfileConfig, framework: Framework, base_url: str, clock: PhaseClock) -> None:
        if profile.agentx_open_after_sec is None or profile.agentx_duration_sec is None:
            raise ValueError("an AgentX profile window needs agentx_open_after_sec and agentx_duration_sec")
        self._open_after = profile.agentx_open_after_sec
        self._duration = profile.agentx_duration_sec
        self._body = start_body(profile, framework)
        self._base_url = base_url.rstrip("/")
        self._clock = clock
        self._profiling_start: float | None = None
        self._opened: float | None = None
        self._closed: float | None = None
        self._running: int | None = None
        self._error: str | None = None

    def hear(self, phases: Mapping[str, object] | None, now_unix: float) -> None:
        """Open the window once the profiling phase is ``agentx_open_after_sec`` old; close it once it has run out."""
        if self._profiling_start is None:
            self._profiling_start = profiling_start_unix(phases)
        if self._closed is not None or self._error is not None or self._profiling_start is None:
            return
        if self._opened is None:
            if now_unix >= self._profiling_start + self._open_after:
                self._open(now_unix)
        elif now_unix >= self._opened + self._duration:
            self._close(now_unix)

    def _open(self, now_unix: float) -> None:
        try:
            _request(self._base_url + START_PROFILE_PATH, self._body)
        except _HTTP_ERRORS as exc:
            self._error = f"{START_PROFILE_PATH}: {exc}"
            return
        self._opened = now_unix
        self._clock.mark(Phase.PROFILE_OPEN)
        self._running = self._running_requests()

    def _running_requests(self) -> int | None:
        try:
            text = _request(self._base_url + METRICS_PATH, None).decode("utf-8", "ignore")
        except _HTTP_ERRORS:
            return None
        running = running_requests(parse_prometheus_text(text))
        return None if running is None else int(running)

    def _close(self, now_unix: float) -> None:
        self._closed = now_unix
        self._clock.mark(Phase.PROFILE_CLOSE)
        try:
            _request(self._base_url + STOP_PROFILE_PATH, {})
        except _HTTP_ERRORS as exc:
            self._error = f"{STOP_PROFILE_PATH}: {exc}"

    def finish(self, layout: RunLayout, now_unix: float) -> WindowCapture:
        """Close a window the replay outlived, write ``profile/capture.json`` and return what it holds."""
        if self._opened is not None and self._closed is None:
            self._close(now_unix)
        capture = WindowCapture(self._profiling_start, self._opened, self._closed, self._running, self._error)
        layout.profile_capture.parent.mkdir(parents=True, exist_ok=True)
        layout.profile_capture.write_text(json.dumps(capture.to_document(), indent=2) + "\n", encoding="utf-8")
        return capture


__all__ = ["PROFILING_PHASE", "AgentxWindow", "WindowCapture", "profiling_start_unix"]
