# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The fixed load's profile window: opened once the warmup pass is over, closed once the measured pass has ended.

The window sits where InferenceX's ``--profile`` put it: ``/start_profile`` right before the measured requests,
``/stop_profile`` right after them. Which steps it captures is the server's to bound (SGLang's ``start_step`` and
``num_steps`` in the body, vLLM's ``--profiler-config`` iterations), and the server counts only steps that run requests,
so the measured client's setup before its first request moves nothing. The answer to ``/stop_profile`` is not read, as
InferenceX did not read it: SGLang refuses to stop a window its ``num_steps`` already closed, and the trace dirs say
whether anything was captured.
"""

from __future__ import annotations

import time

from inference_suite.contracts.config import ProfileConfig
from inference_suite.contracts.layout import RunLayout
from inference_suite.frameworks import Framework
from inference_suite.profile.inject import (
    SERVER_REQUEST_ERRORS,
    START_PROFILE_PATH,
    STOP_PROFILE_PATH,
    WindowCapture,
    server_request,
    start_body,
)
from inference_suite.runtime.phases import Phase, PhaseClock


class FixedWindow:
    """Sends the server's profiler its start and stop around the measured pass, and records when it did."""

    def __init__(self, profile: ProfileConfig, framework: Framework, base_url: str, clock: PhaseClock) -> None:
        self._body = start_body(profile, framework)
        self._base_url = base_url.rstrip("/")
        self._clock = clock
        self._opened: float | None = None
        self._closed: float | None = None
        self._error: str | None = None

    def open(self) -> None:
        try:
            server_request(self._base_url + START_PROFILE_PATH, self._body)
        except SERVER_REQUEST_ERRORS as exc:
            self._error = f"{START_PROFILE_PATH}: {exc}"
            return
        self._opened = time.time()
        self._clock.mark(Phase.PROFILE_OPEN)

    def close(self) -> None:
        if self._opened is None or self._closed is not None:
            return
        self._closed = time.time()
        self._clock.mark(Phase.PROFILE_CLOSE)
        try:
            server_request(self._base_url + STOP_PROFILE_PATH, {})
        except SERVER_REQUEST_ERRORS:  # see the module docstring: the trace dirs, not this answer, are the verdict
            pass

    def finish(self, layout: RunLayout) -> WindowCapture:
        """Write ``profile/capture.json`` and return what it holds."""
        capture = WindowCapture(None, self._opened, self._closed, None, self._error)
        capture.write(layout)
        return capture


__all__ = ["FixedWindow"]
