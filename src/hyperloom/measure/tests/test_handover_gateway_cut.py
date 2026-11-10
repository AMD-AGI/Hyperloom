# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Make-before-break through the gateway's cuts of a create that queues for a node.

On the hosted service the blocking ``POST /v2/sandboxes`` of a sandbox that waits for a node is cut by the gateway with
a 502 after about 600 s, and the provider sends it again with the same idempotency key to go on waiting (hosted smoke
#12 saw two such cuts in a row). That is the normal path of a queued create, not a failed one: the sandbox a finished
test handed over must stay until the new sandbox runs, through every cut its timeout was sized for.

The pool here runs the real E2B provider against a canned HTTP service; one second stands for the gateway's 600.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from hyperloom.measure.sandboxes import (
    DEFAULT_HANDOVER_TIMEOUT_SEC,
    AttemptTag,
    Handover,
    PoolLimits,
    SandboxPool,
)
from hyperloom.sandbox.e2b import E2BSandboxProvider, RetryPolicy
from hyperloom.sandbox.socket import IDEMPOTENCY_KEY

#: How long the gateway let a blocking create wait before it answered 502, as measured on hosted smoke #12.
_GATEWAY_CUT_MEASURED_SEC = 600.0
#: The test's stand-in for that cut.
_CUT_SEC = 1.0
_BAD_GATEWAY = b"<html><head><title>502 Bad Gateway</title></head></html>"


class _Gateway:
    """A sandbox service behind a gateway: each create waits ``_CUT_SEC`` and gets a 502 until ``cuts`` have been
    answered, then its sandbox runs. Records ``("running", id)``, ``("close", id)`` and every create's key, in order."""

    def __init__(self) -> None:
        self.url = ""
        self.cuts = 0
        self.events: list[tuple[str, str]] = []
        self.keys: list[str] = []
        self.live: dict[str, dict[str, str]] = {}
        self._lock = threading.Lock()
        self._count = 0

    def create(self, metadata: dict[str, str]) -> tuple[int, bytes]:
        with self._lock:
            self.keys.append(metadata[IDEMPOTENCY_KEY])
            cut = self.cuts > 0
            self.cuts -= 1 if cut else 0
        if cut:
            time.sleep(_CUT_SEC)
            return 502, _BAD_GATEWAY
        with self._lock:
            self._count += 1
            sandbox_id = f"sbx-{self._count}"
            self.live[sandbox_id] = metadata
            self.events.append(("running", sandbox_id))
        body = {"templateID": "runner-1", "sandboxID": sandbox_id, "clientID": "c", "envdVersion": "0.10.0"}
        return 201, json.dumps(body).encode()

    def close(self, sandbox_id: str) -> None:
        with self._lock:
            if self.live.pop(sandbox_id, None) is not None:
                self.events.append(("close", sandbox_id))

    def listing(self, path: str) -> bytes:
        """``GET /v2/sandboxes`` (this run's live sandboxes) or ``GET /sandboxes/<id>`` (one of them)."""
        with self._lock:
            listed = [_listed(sandbox_id, metadata) for sandbox_id, metadata in self.live.items()]
        if path.startswith("/sandboxes/"):
            [one] = [item for item in listed if item["sandboxID"] == path.rsplit("/", 1)[-1]]
            return json.dumps(one).encode()
        return json.dumps(listed).encode()


def _listed(sandbox_id: str, metadata: dict[str, str]) -> dict[str, object]:
    return {
        "templateID": "runner-1",
        "sandboxID": sandbox_id,
        "clientID": "c",
        "envdVersion": "0.10.0",
        "alias": "runner-1",
        "startedAt": "2026-11-09T00:00:00Z",
        "endAt": "2026-11-09T01:00:00Z",
        "cpuCount": 1,
        "memoryMB": 512,
        "diskSizeMB": 1024,
        "state": "running",
        "metadata": metadata,
    }


@pytest.fixture
def gateway(monkeypatch: pytest.MonkeyPatch) -> Iterator[_Gateway]:
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "E2B_API_KEY", "E2B_DEBUG"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    service = _Gateway()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args: object) -> None:
            pass

        def _send(self, status: int, payload: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Length", str(len(payload)))
            if payload.startswith((b"{", b"[")):
                self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(payload)

        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
            self._send(*service.create(dict(body.get("metadata") or {})))

        def do_DELETE(self) -> None:
            service.close(self.path.rsplit("/", 1)[-1])
            self._send(204, b"")

        def do_GET(self) -> None:
            self._send(200, service.listing(self.path.split("?", 1)[0]))

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True).start()
    service.url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        yield service
    finally:
        server.shutdown()
        server.server_close()


def test_a_kept_sandbox_stays_through_two_gateway_cuts_until_the_next_sandbox_runs(gateway: _Gateway) -> None:
    provider = E2BSandboxProvider(
        {"measurement": "e2b_measurement_key_000000000000"},
        api_url=gateway.url,
        sandbox_url=gateway.url,
        retry=RetryPolicy(budget_sec=30.0, first_delay_sec=0.01, max_delay_sec=0.05),
    )
    # The default timeout, scaled as the gateway's cut is: what the kept sandbox would get on the service.
    timeout = DEFAULT_HANDOVER_TIMEOUT_SEC * _CUT_SEC / _GATEWAY_CUT_MEASURED_SEC
    limits = PoolLimits(max_parallel_tests=1, request_timeout_sec=60, handover=Handover(3, 2, timeout_sec=timeout))
    pool = SandboxPool(provider, {1: "runner-1"}, run_id="run-1", limits=limits)
    try:
        first = pool.acquire(1, AttemptTag("req-1", 1), timeout_sec=60)
        pool.note_upcoming({"req-1", "req-2"})
        pool.hand_over(first)
        assert pool.parked == first.sandbox_id
        gateway.cuts = 2
        second = pool.acquire(1, AttemptTag("req-2", 1), timeout_sec=60)
        assert gateway.keys[1:] == [pool.idempotency_key("req-2", 1)] * 3, "two cut sends and the one that ran"
        assert gateway.events == [
            ("running", first.sandbox_id),
            ("running", second.sandbox_id),
            ("close", first.sandbox_id),
        ], "the kept sandbox was closed before the sandbox the cut creates waited for ran"
        assert pool.parked is None and set(gateway.live) == {second.sandbox_id}
        pool.release(second)
        assert gateway.live == {}
    finally:
        pool.close()


@pytest.mark.parametrize("timeout", [0, -1.0])
def test_a_handover_without_a_positive_timeout_is_refused(timeout: float) -> None:
    with pytest.raises(ValueError, match="positive timeout_sec"):
        Handover(3, 2, timeout_sec=timeout)
