# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Contract of the AgentX profile window: it counts from the first profiling request, never from the client's start."""

from __future__ import annotations

import http.server
import json
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest

from inference_suite.contracts import ProfileConfig, RunLayout
from inference_suite.frameworks import Framework
from inference_suite.profile.agentx_window import AgentxWindow, profiling_start_unix
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


def _phases(*, warmup_ns: int | None = None, profiling_ns: int | None = None) -> dict[str, object]:
    phases: dict[str, object] = {}
    if warmup_ns is not None:
        phases["warmup"] = {"start_ns": warmup_ns}
    phases["profiling"] = {"start_ns": profiling_ns}
    return phases


def _ns(unix: float) -> int:
    return int(unix * 1e9)


def test_the_window_opens_its_offset_after_the_first_profiling_request_not_after_the_warmup(engine, tmp_path) -> None:
    served, url = engine
    clock = PhaseClock()
    window = AgentxWindow(_PROFILE, Framework.SGLANG, url, clock)

    window.hear(_phases(warmup_ns=_ns(_START)), _START + 500)
    assert served.requests == []
    window.hear(_phases(warmup_ns=_ns(_START), profiling_ns=_ns(_START + 480)), _START + 509)
    assert served.requests == []
    window.hear(_phases(warmup_ns=_ns(_START), profiling_ns=_ns(_START + 480)), _START + 510)

    assert served.requests == [("/start_profile", {}), ("/metrics", None)]
    assert clock.first(Phase.PROFILE_OPEN) is not None
    window.hear(_phases(profiling_ns=_ns(_START + 480)), _START + 569)
    assert [path for path, _ in served.requests][-1] == "/metrics"
    window.hear(_phases(profiling_ns=_ns(_START + 480)), _START + 570)
    assert served.requests[-1] == ("/stop_profile", {})
    assert clock.first(Phase.PROFILE_CLOSE) is not None

    capture = window.finish(RunLayout(tmp_path), _START + 600)
    assert (capture.profiling_start_unix, capture.window_open_unix, capture.window_close_unix) == (
        _START + 480,
        _START + 510,
        _START + 570,
    )
    assert capture.running_requests_in_window == 13
    assert capture.error is None
    assert json.loads((tmp_path / "profile" / "capture.json").read_text()) == capture.to_document()


def test_a_replay_that_ends_inside_the_window_closes_it_when_the_load_finishes(engine, tmp_path) -> None:
    served, url = engine
    window = AgentxWindow(_PROFILE, Framework.VLLM, url, PhaseClock())
    window.hear(_phases(profiling_ns=_ns(_START)), _START + 30)

    capture = window.finish(RunLayout(tmp_path), _START + 45)

    assert served.requests[-1] == ("/stop_profile", {})
    assert (capture.window_open_unix, capture.window_close_unix) == (_START + 30, _START + 45)


def test_a_replay_that_never_reaches_profiling_opens_no_window(engine, tmp_path) -> None:
    served, url = engine
    window = AgentxWindow(_PROFILE, Framework.SGLANG, url, PhaseClock())
    window.hear(None, _START)
    window.hear(_phases(warmup_ns=_ns(_START)), _START + 3600)

    capture = window.finish(RunLayout(tmp_path), _START + 3600)

    assert served.requests == []
    assert capture.window_open_unix is None and capture.profiling_start_unix is None


@pytest.mark.parametrize("engine", ["/start_profile"], indirect=True)
def test_a_refused_start_is_recorded_and_never_retried(engine, tmp_path) -> None:
    served, url = engine
    window = AgentxWindow(_PROFILE, Framework.SGLANG, url, PhaseClock())
    for offset in (30, 31, 120):
        window.hear(_phases(profiling_ns=_ns(_START)), _START + offset)

    capture = window.finish(RunLayout(tmp_path), _START + 200)

    assert [path for path, _ in served.requests] == ["/start_profile"]
    assert capture.window_open_unix is None
    assert capture.error is not None and capture.error.startswith("/start_profile: HTTP Error 500")


def test_sglang_carries_the_configured_start_body(engine) -> None:
    served, url = engine
    profile = ProfileConfig(
        enabled=True, start_body={"record_shapes": True}, agentx_open_after_sec=0, agentx_duration_sec=1
    )
    AgentxWindow(profile, Framework.SGLANG, url, PhaseClock()).hear(_phases(profiling_ns=_ns(_START)), _START)

    assert served.requests[0] == ("/start_profile", {"record_shapes": True})


@pytest.mark.parametrize("start_ns", [None, 0, True, "1"])
def test_a_profiling_phase_without_a_start_has_not_begun(start_ns) -> None:
    assert profiling_start_unix({"profiling": {"start_ns": start_ns}}) is None
    assert profiling_start_unix({"profiling": "x"}) is None


def test_the_window_needs_both_agentx_bounds(tmp_path: Path) -> None:
    profile = ProfileConfig(enabled=True, num_prompts=8, delay_iters=1, max_iters=2)

    with pytest.raises(ValueError, match="agentx_open_after_sec"):
        AgentxWindow(profile, Framework.SGLANG, "http://127.0.0.1:1", PhaseClock())
