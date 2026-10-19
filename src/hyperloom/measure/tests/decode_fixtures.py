# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A fake OpenAI-compatible server for the decode pass: it answers ``/v1/completions`` deterministically."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

HTTP_ERROR = 500


def expected_tokens(prompt: list[int], max_tokens: int) -> list[int]:
    """What the fake server answers for ``prompt``: a function of the prompt only, as a greedy decode is."""
    return [(sum(prompt) + step) % 1000 for step in range(max_tokens)]


class FakeServer:
    """Records every request; ``mode`` is ``ok``, ``error`` (HTTP 500) or ``no_ids`` (a body with no token ids)."""

    def __init__(self, mode: str = "ok", delay_sec: float = 0.0) -> None:
        self.mode, self.delay_sec = mode, delay_sec
        self.requests: list[dict[str, Any]] = []
        self.in_flight = self.peak = 0
        self.lock = threading.Lock()
        self.port = 0


def _handler(state: FakeServer) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args: Any) -> None:
            return

        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            with state.lock:
                state.requests.append({"path": self.path, **body})
                state.in_flight += 1
                state.peak = max(state.peak, state.in_flight)
            time.sleep(state.delay_sec)
            with state.lock:
                state.in_flight -= 1
            code, payload = 200, {"choices": [{"token_ids": expected_tokens(body["prompt"], body["max_tokens"])}]}
            if state.mode == "error":
                code, payload = HTTP_ERROR, {"error": "boom"}
            if state.mode == "no_ids":
                payload = {"choices": [{"text": "hello"}]}
            raw = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    return Handler


@contextmanager
def fake_server(mode: str = "ok", delay_sec: float = 0.0) -> Iterator[FakeServer]:
    state = FakeServer(mode, delay_sec)
    server = ThreadingHTTPServer(("127.0.0.1", 0), _handler(state))
    state.port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
