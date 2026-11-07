# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Contract of the AgentX profile window: it counts from the measured mark and closes by the finished mark."""

from __future__ import annotations

import http.server
import json
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest

from inference_suite.contracts import ProfileConfig, RunLayout
from inference_suite.frameworks import Framework
from inference_suite.profile.agentx_window import AgentxWindow
from inference_suite.runtime.phases import Phase, PhaseClock

_START = 1_800_000_000.0
_PROFILE = ProfileConfig(enabled=True, agentx_open_after_sec=30, agentx_duration_sec=60)
_METRICS = (
    'sglang:num_running_reqs{tp_rank="0",dp_rank="0"} 7\n'
    'sglang:num_running_reqs{tp_rank="1",dp_rank="0"} 7\n'
    'sglang:num_running_reqs{tp_rank="0",dp_rank="1"} 6\n'
)


class _Server:
    """A served engine that records each control request it answers."""

    def __init__(self, refuse: str | None) -> None:
        self.requests: list[tuple[str, object]] = []
        self.refuse = refuse

    def handler(self) -> type[http.server.BaseHTTPRequestHandler]:
        server = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def _answer(self, body: bytes) -> None:
                status = 500 if self.path == server.refuse else 200
                self.send_response(status)
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:
                server.requests.append((self.path, None))
                self._answer(_METRICS.encode())

            def do_POST(self) -> None:
                length = int(self.headers["Content-Length"])
                server.requests.append((self.path, json.loads(self.rfile.read(length))))
                self._answer(b"{}")

            def log_message(self, *args: object) -> None:
                pass

        return Handler


@pytest.fixture(params=[None])
def engine(request: pytest.FixtureRequest) -> Iterator[tuple[_Server, str]]:
    served = _Server(request.param)
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), served.handler())
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield served, f"http://127.0.0.1:{httpd.server_address[1]}"
    finally:
        httpd.shutdown()


def _measured(clock: PhaseClock, at: float = _START) -> PhaseClock:
    clock.mark(Phase.MEASURED, at)
    return clock


def test_the_window_opens_its_offset_after_the_measured_mark_not_after_the_warmup(engine, tmp_path) -> None:
    served, url = engine
    clock = PhaseClock()
    window = AgentxWindow(_PROFILE, Framework.SGLANG, url, clock)

    clock.mark(Phase.WARMUP, _START - 480)
    window.hear(_START + 20)
    assert served.requests == []
    _measured(clock)
    window.hear(_START + 29)
    assert served.requests == []
    window.hear(_START + 30)

    assert served.requests == [("/start_profile", {}), ("/metrics", None)]
    assert clock.first(Phase.PROFILE_OPEN).mono == _START + 30
    window.hear(_START + 89)
    assert [path for path, _ in served.requests][-1] == "/metrics"
    window.hear(_START + 90)
    assert served.requests[-1] == ("/stop_profile", {})
    assert clock.first(Phase.PROFILE_CLOSE).mono == _START + 90

    capture = window.finish(RunLayout(tmp_path))
    assert capture.window_open_unix - capture.profiling_start_unix == pytest.approx(30, abs=1e-3)
    assert capture.window_close_unix - capture.window_open_unix == pytest.approx(60, abs=1e-3)
    assert capture.window_open_unix == clock.first(Phase.PROFILE_OPEN).ts
    assert capture.running_requests_in_window == 13
    assert capture.error is None
    assert json.loads((tmp_path / "profile" / "capture.json").read_text()) == capture.to_document()


def test_the_finished_mark_closes_an_open_window_at_the_next_poll(engine, tmp_path) -> None:
    served, url = engine
    clock = _measured(PhaseClock())
    window = AgentxWindow(_PROFILE, Framework.VLLM, url, clock)
    window.hear(_START + 30)
    clock.mark(Phase.FINISHED, _START + 44)

    window.hear(_START + 45)

    assert served.requests[-1] == ("/stop_profile", {})
    assert clock.first(Phase.PROFILE_CLOSE).mono == _START + 45
    window.hear(_START + 200)
    assert [path for path, _ in served.requests].count("/stop_profile") == 1


def test_a_replay_that_finishes_before_the_offset_opens_no_window(engine, tmp_path) -> None:
    served, url = engine
    clock = _measured(PhaseClock())
    clock.mark(Phase.FINISHED, _START + 10)
    window = AgentxWindow(_PROFILE, Framework.SGLANG, url, clock)

    window.hear(_START + 3600)
    capture = window.finish(RunLayout(tmp_path))

    assert served.requests == []
    assert capture.window_open_unix is None and capture.window_close_unix is None


def test_a_replay_that_never_reaches_profiling_opens_no_window(engine, tmp_path) -> None:
    served, url = engine
    clock = PhaseClock()
    clock.mark(Phase.WARMUP, _START)
    window = AgentxWindow(_PROFILE, Framework.SGLANG, url, clock)
    window.hear(_START + 3600)

    capture = window.finish(RunLayout(tmp_path))

    assert served.requests == []
    assert capture.window_open_unix is None and capture.profiling_start_unix is None


@pytest.mark.parametrize("engine", ["/start_profile"], indirect=True)
def test_a_refused_start_is_recorded_and_never_retried(engine, tmp_path) -> None:
    served, url = engine
    window = AgentxWindow(_PROFILE, Framework.SGLANG, url, _measured(PhaseClock()))
    for offset in (30, 31, 120):
        window.hear(_START + offset)

    capture = window.finish(RunLayout(tmp_path))

    assert [path for path, _ in served.requests] == ["/start_profile"]
    assert capture.window_open_unix is None
    assert capture.error is not None and capture.error.startswith("/start_profile: HTTP Error 500")


def test_sglang_carries_the_configured_start_body(engine) -> None:
    served, url = engine
    profile = ProfileConfig(
        enabled=True, start_body={"record_shapes": True}, agentx_open_after_sec=0, agentx_duration_sec=1
    )
    AgentxWindow(profile, Framework.SGLANG, url, _measured(PhaseClock())).hear(_START)

    assert served.requests[0] == ("/start_profile", {"record_shapes": True})


def test_the_window_needs_both_agentx_bounds(tmp_path: Path) -> None:
    profile = ProfileConfig(enabled=True, num_prompts=8, delay_iters=1, max_iters=2)

    with pytest.raises(ValueError, match="agentx_open_after_sec"):
        AgentxWindow(profile, Framework.SGLANG, "http://127.0.0.1:1", PhaseClock())
