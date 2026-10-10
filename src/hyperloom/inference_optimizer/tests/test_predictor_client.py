# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Contract tests for the PrimaTune predictor HTTP client."""

from __future__ import annotations

import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from hyperloom.orchestrator.predictor.client import predict


@pytest.fixture
def service():
    """A local predictor stand-in; set ``reply`` to (status, body) and read ``requests``."""
    state = {"reply": (200, {}), "delay": 0.0, "requests": []}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            state["requests"].append((self.path, json.loads(self.rfile.read(length))))
            time.sleep(state["delay"])
            status, body = state["reply"]
            data = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    state["url"] = f"http://127.0.0.1:{server.server_address[1]}"
    yield state
    server.shutdown()
    server.server_close()


def _answer(**fields):
    return {"schema": "primatune.predictor_response.v1", "parsed": True, **fields}


def test_posts_the_request_and_reads_every_action_best_first(service):
    service["reply"] = (
        200,
        _answer(
            actions=[
                {"server_args": {"--kv-cache-dtype": "fp8"}, "envs": {}, "rationale": "fp8 KV\n  halves the reads."},
                {"server_args": {}, "envs": {}},
                {"source_change": "fuse the rmsnorm"},
            ],
            meta={"samples": 8},
        ),
    )
    answer = predict({"schema": "x"}, endpoint=service["url"] + "/", timeout_sec=5)

    assert service["requests"] == [("/v1/predict", {"schema": "x"})]
    assert answer.parsed and answer.meta == {"samples": 8}
    assert [a.server_args for a in answer.config_actions] == [{"--kv-cache-dtype": "fp8"}]
    assert answer.config_actions[0].rationale == "fp8 KV halves the reads."
    assert answer.source_change == "fuse the rmsnorm"
    assert len(answer.actions) == 2


def test_a_single_action_service_is_read_too(service):
    service["reply"] = (200, _answer(action={"envs": {"VLLM_ROCM_USE_AITER": "1"}}))
    answer = predict({}, endpoint=service["url"], timeout_sec=5)
    assert [a.envs for a in answer.config_actions] == [{"VLLM_ROCM_USE_AITER": "1"}]


@pytest.mark.parametrize(
    ("reply", "error"),
    [
        ((200, {"schema": "primatune.predictor_response.v1", "parsed": False}), "predictor declined"),
        ((500, {"detail": "boom"}), "HTTP 500"),
        ((200, b"not json"), "malformed response body"),
        ((200, [1, 2]), "expected an object"),
        ((200, {"schema": "other.v1", "parsed": True}), "unexpected response schema"),
    ],
)
def test_every_failure_is_an_unparsed_answer(service, reply, error):
    service["reply"] = reply
    answer = predict({}, endpoint=service["url"], timeout_sec=5)
    assert not answer.parsed and not answer.actions
    assert error in answer.error


def test_a_slow_service_times_out_into_no_answer(service):
    service["delay"] = 1.0
    answer = predict({}, endpoint=service["url"], timeout_sec=0.2)
    assert not answer.parsed and "transport error" in answer.error


def test_an_unreachable_or_non_http_endpoint_is_no_answer():
    assert not predict({}, endpoint="http://127.0.0.1:9", timeout_sec=1).parsed
    assert "unsupported URL scheme" in predict({}, endpoint="file:///etc", timeout_sec=1).error
    assert "transport error" in predict({}, endpoint="http://127.0.0.1:99999x", timeout_sec=1).error


def test_a_listener_that_does_not_speak_http_is_no_answer():
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)

    def _reply() -> None:
        conn, _ = listener.accept()
        conn.recv(4096)
        conn.sendall(b"SSH-2.0-not-http\r\n\r\n")
        conn.close()

    threading.Thread(target=_reply, daemon=True).start()
    answer = predict({}, endpoint=f"http://127.0.0.1:{listener.getsockname()[1]}", timeout_sec=5)
    listener.close()
    assert not answer.parsed and "transport error" in answer.error
