# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The stand-in's router on its own, in front of an envd of the test's making: it relays a path to the envd of the
sandbox it looked up, and refuses a target that is not a path and an answer whose header would split its own."""

from __future__ import annotations

import http.client
import threading
from collections.abc import Callable, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from hyperloom.sandbox.tests.e2b_standin.proxy import serve_proxy

_SANDBOX = "sbx-router"
_BODY = b"envd"

Serve = Callable[[ThreadingHTTPServer], ThreadingHTTPServer]


@pytest.fixture
def serve() -> Iterator[Serve]:
    """Serve a server for the test; each is shut down after it."""
    started: list[ThreadingHTTPServer] = []

    def start(server: ThreadingHTTPServer) -> ThreadingHTTPServer:
        threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True).start()
        started.append(server)
        return server

    yield start
    for server in started:
        server.shutdown()
        server.server_close()


def _envd(headers: list[tuple[str, str]], asked: list[str]) -> ThreadingHTTPServer:
    """An envd that notes each target it is asked for and answers with ``headers`` and a short body."""

    class Envd(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args: object) -> None:
            pass

        def do_GET(self) -> None:
            asked.append(self.path)
            self.close_connection = True  # the router may leave an answer unread: there is no next request to wait for
            self.send_response(200)
            for name, value in headers:
                self.send_header(name, value)
            self.send_header("Content-Length", str(len(_BODY)))
            self.end_headers()
            self.wfile.write(_BODY)

    return ThreadingHTTPServer(("127.0.0.1", 0), Envd)


def _router(serve: Serve, headers: list[tuple[str, str]], asked: list[str]) -> int:
    """The port of a router whose one sandbox is an envd answering with ``headers``."""
    envd = serve(_envd(headers, asked))
    box = SimpleNamespace(port=envd.server_address[1])
    plane = SimpleNamespace(lookup=lambda sandbox_id: box if sandbox_id == _SANDBOX else None)
    return serve(serve_proxy(plane)).server_address[1]


def _ask(port: int, target: str, sandbox: str = _SANDBOX) -> tuple[int, dict[str, str], bytes]:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        conn.request("GET", target, headers={"E2b-Sandbox-Id": sandbox})
        response = conn.getresponse()
        return response.status, dict(response.getheaders()), response.read()
    finally:
        conn.close()


def test_a_path_with_its_query_reaches_the_envd_of_the_sandbox_and_its_answer_comes_back(serve: Serve) -> None:
    asked: list[str] = []
    port = _router(serve, [("X-Envd", "yes")], asked)

    status, headers, body = _ask(port, "/files?path=%2Fwork%2Fa.txt")

    assert (status, headers["X-Envd"], body) == (200, "yes", _BODY)
    assert asked == ["/files?path=%2Fwork%2Fa.txt"]


@pytest.mark.parametrize("target", ["http://elsewhere.invalid/health", "elsewhere.invalid:80", "*"])
def test_a_target_that_is_not_a_path_on_the_sandbox_is_refused_and_reaches_nothing(serve: Serve, target: str) -> None:
    asked: list[str] = []
    port = _router(serve, [], asked)

    status, _, body = _ask(port, target)

    assert status == 400 and b"not a path on the sandbox" in body
    assert asked == []


def test_an_answer_with_a_line_break_inside_a_header_is_refused_rather_than_relayed(serve: Serve) -> None:
    port = _router(serve, [("X-Folded", "first\r\n second")], [])

    status, headers, body = _ask(port, "/health")

    assert status == 502 and b"line break inside a header" in body
    assert "X-Folded" not in headers


def test_a_sandbox_that_does_not_exist_gets_the_502_e2b_gives(serve: Serve) -> None:
    asked: list[str] = []
    port = _router(serve, [], asked)

    status, _, body = _ask(port, "/health", sandbox="sbx-nope")

    assert status == 502 and b"The sandbox sbx-nope was not found" in body
    assert asked == []
