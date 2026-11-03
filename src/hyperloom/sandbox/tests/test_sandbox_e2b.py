# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The E2B provider's failure handling, against a canned HTTP service.

The sandbox contract suite covers the provider's normal behaviour against a stand-in service; this file covers what a
stand-in cannot honestly produce: a particular status code, a server error that clears on the third try, a data plane
that refuses with 403 or cannot be reached. The contract suite and the stand-in arrive in a later PR.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import e2b
import httpx
import pytest

from hyperloom.sandbox import CreateOptions, OutputSinks, RunSettings
from hyperloom.sandbox.e2b import E2BSandboxProvider, RetryPolicy, _LineSink
from hyperloom.sandbox.socket import (
    IDEMPOTENCY_KEY,
    WORKSPACE_KEY,
    SandboxCreateFailed,
    SandboxDenied,
    SandboxGone,
    SandboxQuotaExceeded,
    SandboxUnavailable,
)

KEY_DEV = "e2b_dev_key_0123456789abcdef"
KEY_MEASUREMENT = "e2b_measurement_key_000000000000"
FAST = RetryPolicy(budget_sec=2.0, first_delay_sec=0.01, max_delay_sec=0.05)
SANDBOX = {"templateID": "dev", "sandboxID": "sbx-1", "clientID": "c", "envdVersion": "0.10.0"}


def _listed(sandbox_id: str, metadata: dict[str, str]) -> dict[str, Any]:
    return {
        **SANDBOX,
        "sandboxID": sandbox_id,
        "alias": "dev",
        "startedAt": "2026-10-20T00:00:00Z",
        "endAt": "2026-10-20T01:00:00Z",
        "cpuCount": 1,
        "memoryMB": 512,
        "diskSizeMB": 1024,
        "state": "running",
        "metadata": metadata,
    }


@dataclass
class _Seen:
    method: str
    path: str
    query: dict[str, list[str]]
    headers: dict[str, str]
    body: Any


@dataclass
class _Service:
    """A control plane that answers each request from a script and records what it was asked."""

    url: str = ""
    seen: list[_Seen] = field(default_factory=list)
    _script: dict[tuple[str, str], list[tuple[int, Any, dict[str, str], float]]] = field(default_factory=dict)

    def reply(
        self, method: str, path: str, *answers: int | tuple[int, Any], headers: dict[str, str] | None = None
    ) -> None:
        """Answer ``method path`` with each answer in turn (the last one repeats).

        An answer is a status (an error status carries the JSON error body E2B sends) or ``(status, body)``, where a
        ``bytes`` body is sent as it is.
        """
        script = [
            (a, {"code": a, "message": f"status {a}"} if a >= 400 else None) if isinstance(a, int) else a
            for a in answers
        ]
        self._script[(method, path)] = [(status, body, headers or {}, 0.0) for status, body in script]

    def delay(self, method: str, path: str, seconds: float) -> None:
        status, body, headers, _ = self._script[(method, path)][0]
        self._script[(method, path)] = [(status, body, headers, seconds)]

    def calls(self, method: str, path: str) -> list[_Seen]:
        return [s for s in self.seen if (s.method, s.path) == (method, path)]


@pytest.fixture
def service(monkeypatch: pytest.MonkeyPatch) -> Iterator[_Service]:
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "E2B_API_KEY", "E2B_DEBUG"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    svc = _Service()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args: object) -> None:
            pass

        def _answer(self) -> None:
            url = urlparse(self.path)
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b""
            svc.seen.append(
                _Seen(
                    self.command,
                    url.path,
                    parse_qs(url.query),
                    {k.lower(): v for k, v in self.headers.items()},
                    json.loads(raw or b"null"),
                )
            )
            script = svc._script.get((self.command, url.path), [(404, {"code": 404, "message": "no script"}, {}, 0.0)])
            status, body, headers, delay = script.pop(0) if len(script) > 1 else script[0]
            time.sleep(delay)
            payload = body if isinstance(body, bytes) else b"" if body is None else json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Length", str(len(payload)))
            if payload:
                self.send_header("Content-Type", "application/json")
            for name, value in headers.items():
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(payload)

        do_GET = do_POST = do_DELETE = _answer

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True).start()
    svc.url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        yield svc
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture
def provider(service: _Service) -> E2BSandboxProvider:
    keys = {"dev": KEY_DEV, "measurement": KEY_MEASUREMENT}
    return E2BSandboxProvider(keys, api_url=service.url, sandbox_url=service.url, retry=FAST)


def _create(provider: E2BSandboxProvider, **kw: Any):
    return provider.create("dev", role="dev", timeout_sec=60, metadata=kw.pop("metadata", {}), **kw)


def test_create_sends_the_request_with_the_roles_key(service: _Service, provider: E2BSandboxProvider) -> None:
    service.reply("POST", "/v2/sandboxes", (201, SANDBOX))
    sb = provider.create(
        "dev", role="dev", timeout_sec=90, metadata={"a": "b"}, options=CreateOptions(envs={"X": "1"}, wait_sec=5)
    )
    assert sb.sandbox_id == "sbx-1"
    [call] = service.calls("POST", "/v2/sandboxes")
    assert call.headers["x-api-key"] == KEY_DEV
    assert (call.body["templateID"], call.body["timeout"]) == ("dev", 90)
    assert (call.body["metadata"], call.body["envVars"]) == ({"a": "b"}, {"X": "1"})
    measurement = provider.create("dev", role="measurement", timeout_sec=60, metadata={})
    assert measurement.sandbox_id == "sbx-1"
    assert service.calls("POST", "/v2/sandboxes")[-1].headers["x-api-key"] == KEY_MEASUREMENT


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (400, SandboxDenied),
        (401, SandboxDenied),
        (403, SandboxDenied),
        (429, SandboxUnavailable),
        (500, SandboxUnavailable),
        (503, SandboxUnavailable),
    ],
)
def test_a_refused_create_is_one_of_the_socket_failures_and_not_replayed(
    service: _Service, provider: E2BSandboxProvider, status: int, expected: type[Exception]
) -> None:
    service.reply("POST", "/v2/sandboxes", (status, {"code": status, "message": "pending_timeout"}))
    with pytest.raises(expected) as refused:
        _create(provider)
    assert type(refused.value) is expected, "none of these is a quota refusal"
    assert len(service.calls("POST", "/v2/sandboxes")) == 1


def test_create_with_an_idempotency_key_is_replayed_after_a_server_error(
    service: _Service, provider: E2BSandboxProvider
) -> None:
    service.reply("POST", "/v2/sandboxes", 500, 502, (201, SANDBOX))
    sb = _create(provider, metadata={IDEMPOTENCY_KEY: "k-1"})
    assert sb.sandbox_id == "sbx-1"
    assert len(service.calls("POST", "/v2/sandboxes")) == 3


#: Agent Sandbox's answer to a create whose sandbox failed while starting (joint hosted run 4), word for word: the E2B
#: error body (the SDK requires its ``code``) with the reason code beside it, and in ``message``, which the SDK keeps.
CREATE_FAILED_MESSAGE = "sandbox_create_failed: workload sbx-00000000000000000000000000000000 phase Failed"
CREATE_FAILED_BODY = {"code": 500, "reason": "sandbox_create_failed", "message": CREATE_FAILED_MESSAGE}
#: Its answer to every replay of that key afterwards: the key stays bound to the failed sandbox, whose id it names (the
#: SDK's message of joint run 4, word for word, in the same E2B error body).
KEY_SPENT_MESSAGE = "idempotency_key_spent: sbx-00000000000000000000000000000000"
KEY_SPENT_BODY = {"code": 410, "reason": "idempotency_key_spent", "message": KEY_SPENT_MESSAGE}
#: Its answer to the original create whose sandbox was not running within its pending timeout: the reason code first,
#: then SaFE's last phase and status.
PENDING_TIMEOUT_MESSAGE = (
    "pending_timeout: workload sbx-00000000000000000000000000000000 phase Pending, status Scheduling"
)
#: Its answer to a replay of that key: the key is spent, then the original reason.
KEY_SPENT_PENDING_BODY = {
    "code": 410,
    "reason": "idempotency_key_spent",
    "message": f"{KEY_SPENT_MESSAGE}: {PENDING_TIMEOUT_MESSAGE}",
}
#: How a gateway cuts a create that is still pending: its 180 s limit, or the service's context cancelled under it.
GATEWAY_CUTS = [
    (504, {"code": 504, "message": "context canceled"}),
    (504, b"<html>504 Gateway Time-out</html>"),
]


def _keys_sent(service: _Service) -> list[str]:
    return [call.body["metadata"][IDEMPOTENCY_KEY] for call in service.calls("POST", "/v2/sandboxes")]


@pytest.mark.parametrize(
    "body",
    [
        CREATE_FAILED_BODY,
        {"code": 500, "message": "workload sbx-1 phase Failed"},
        {"code": 500, "message": "sandbox_create_failed: workload sbx-1 was deleted while pending, status Scheduling"},
    ],
    ids=["sandbox_create_failed", "phase_failed", "deleted_while_pending"],
)
def test_a_create_whose_sandbox_failed_is_final_and_not_replayed(
    service: _Service, provider: E2BSandboxProvider, body: dict[str, Any]
) -> None:
    service.reply("POST", "/v2/sandboxes", (500, body), (201, SANDBOX))
    with pytest.raises(SandboxCreateFailed) as failed:
        _create(provider, metadata={IDEMPOTENCY_KEY: "k-1"})
    assert str(failed.value) == f"500: {body['message']}", "the platform's own message is kept"
    assert _keys_sent(service) == ["k-1"], "a replay of the key would only find the failed sandbox or make another"


@pytest.mark.parametrize("status", [500, 503, 504])
def test_a_create_whose_sandbox_timed_out_pending_is_unavailable_and_not_replayed(
    service: _Service, provider: E2BSandboxProvider, status: int
) -> None:
    service.reply(
        "POST", "/v2/sandboxes", (status, {"code": status, "message": PENDING_TIMEOUT_MESSAGE}), (201, SANDBOX)
    )
    with pytest.raises(SandboxUnavailable) as timed_out:
        _create(provider, metadata={IDEMPOTENCY_KEY: "k-1"})
    assert type(timed_out.value) is SandboxUnavailable, "the sandbox never ran: it is not a failed sandbox"
    assert PENDING_TIMEOUT_MESSAGE in str(timed_out.value), "the service's reason is kept"
    assert _keys_sent(service) == ["k-1"], "the key is spent: a replay would only be answered with a 410"


@pytest.mark.parametrize("cut", GATEWAY_CUTS, ids=["context_canceled", "gateway_page"])
def test_a_create_cut_by_the_gateway_is_replayed_and_finds_the_pending_sandbox(
    service: _Service, provider: E2BSandboxProvider, cut: tuple[int, Any]
) -> None:
    service.reply("POST", "/v2/sandboxes", cut, (201, SANDBOX))
    sb = _create(provider, metadata={IDEMPOTENCY_KEY: "k-1"})
    assert sb.sandbox_id == "sbx-1"
    assert _keys_sent(service) == ["k-1", "k-1"], "the replay carries the same key"


@pytest.mark.parametrize(
    ("answer", "says"),
    [((410, KEY_SPENT_BODY), KEY_SPENT_MESSAGE), ((500, CREATE_FAILED_BODY), "phase Failed")],
    ids=["key_spent", "sandbox_create_failed"],
)
def test_a_replay_after_a_cut_that_finds_the_sandbox_failed_stops_there(
    service: _Service, provider: E2BSandboxProvider, answer: tuple[int, Any], says: str
) -> None:
    """The pending sandbox failed while the gateway had cut the create: the replay is answered with the key spent (or the
    failure itself), which ends this create; it is not the key of an earlier process, to be replaced by another."""
    service.reply("POST", "/v2/sandboxes", GATEWAY_CUTS[0], answer, (201, SANDBOX))
    with pytest.raises(SandboxCreateFailed, match=says):
        _create(provider, metadata={IDEMPOTENCY_KEY: "k-1"})
    assert _keys_sent(service) == ["k-1", "k-1"]


def test_a_replay_answered_with_the_key_spent_and_a_pending_timeout_is_unavailable(
    service: _Service, provider: E2BSandboxProvider
) -> None:
    """The gateway cut the create while its sandbox was pending, and the sandbox then timed out: the replay's 410 carries
    the original reason, which makes this create unavailable, not failed."""
    service.reply("POST", "/v2/sandboxes", GATEWAY_CUTS[0], (410, KEY_SPENT_PENDING_BODY), (201, SANDBOX))
    with pytest.raises(SandboxUnavailable, match="pending_timeout") as timed_out:
        _create(provider, metadata={IDEMPOTENCY_KEY: "k-1"})
    assert type(timed_out.value) is SandboxUnavailable
    assert KEY_SPENT_PENDING_BODY["message"] in str(timed_out.value), "the key and the reason are kept"
    assert _keys_sent(service) == ["k-1", "k-1"]


@pytest.mark.parametrize("body", [KEY_SPENT_BODY, KEY_SPENT_PENDING_BODY], ids=["key_spent", "pending_timeout"])
def test_a_key_spent_before_the_first_send_is_gone_for_the_caller_to_replace(
    service: _Service, provider: E2BSandboxProvider, body: dict[str, Any]
) -> None:
    service.reply("POST", "/v2/sandboxes", (410, body))
    with pytest.raises(SandboxGone) as gone:
        _create(provider, metadata={IDEMPOTENCY_KEY: "k-1"})
    assert type(gone.value) is SandboxGone, "an earlier process spent the key; this create made nothing"
    assert _keys_sent(service) == ["k-1"]


def test_a_503_on_create_is_final_even_with_an_idempotency_key(service: _Service, provider: E2BSandboxProvider) -> None:
    service.reply("POST", "/v2/sandboxes", (503, {"code": 503, "message": "no_capacity"}))
    with pytest.raises(SandboxUnavailable, match="no_capacity"):
        _create(provider, metadata={IDEMPOTENCY_KEY: "k-1"})
    assert len(service.calls("POST", "/v2/sandboxes")) == 1


@pytest.mark.parametrize(
    "body",
    [
        {"code": 410, "message": "idempotency_key_spent: its sandbox sbx-old has ended"},
        {"code": 410, "reason": "idempotency_key_spent", "sandboxID": "sbx-old"},
    ],
)
def test_a_create_with_a_spent_idempotency_key_is_gone_and_not_replayed(
    service: _Service, provider: E2BSandboxProvider, body: dict[str, Any]
) -> None:
    service.reply("POST", "/v2/sandboxes", (410, body))
    with pytest.raises(SandboxGone):
        _create(provider, metadata={IDEMPOTENCY_KEY: "k-1"})
    assert len(service.calls("POST", "/v2/sandboxes")) == 1


def test_a_410_from_a_call_on_a_sandbox_is_gone(service: _Service, provider: E2BSandboxProvider) -> None:
    service.reply("POST", "/v2/sandboxes", (201, SANDBOX))
    sb = _create(provider)
    service.reply("GET", "/sandboxes/sbx-1", (410, {"code": 410, "message": "gone"}))
    with pytest.raises(SandboxGone):
        sb.info()


def test_a_429_over_the_grant_quota_is_back_pressure_and_not_replayed(
    service: _Service, provider: E2BSandboxProvider
) -> None:
    service.reply("POST", "/v2/sandboxes", (429, {"code": 429, "message": "grant_quota_exceeded: 4 of 4 running"}))
    with pytest.raises(SandboxQuotaExceeded) as refused:
        _create(provider, metadata={IDEMPOTENCY_KEY: "k-1"})
    assert isinstance(refused.value, SandboxUnavailable), "a caller that knows only the socket sees unavailable"
    assert len(service.calls("POST", "/v2/sandboxes")) == 1


@pytest.mark.parametrize(
    ("status", "body"),
    [
        (429, {"code": 429, "message": "slow down"}),
        (503, {"code": 503, "message": "pending_timeout"}),
        (503, {"code": 503, "message": "no_capacity"}),
    ],
)
def test_a_rate_limit_or_no_capacity_is_not_quota_back_pressure(
    service: _Service, provider: E2BSandboxProvider, status: int, body: Any
) -> None:
    service.reply("POST", "/v2/sandboxes", (status, body))
    with pytest.raises(SandboxUnavailable) as refused:
        _create(provider, metadata={IDEMPOTENCY_KEY: "k-1"})
    assert type(refused.value) is SandboxUnavailable
    assert len(service.calls("POST", "/v2/sandboxes")) == 1


def test_create_stops_waiting_after_wait_sec(service: _Service, provider: E2BSandboxProvider) -> None:
    service.reply("POST", "/v2/sandboxes", (201, SANDBOX))
    service.delay("POST", "/v2/sandboxes", 3.0)
    started = time.monotonic()
    with pytest.raises(SandboxUnavailable):
        _create(provider, options=CreateOptions(wait_sec=0.3))
    assert time.monotonic() - started < 2.0


def test_a_gateway_page_instead_of_an_error_body_is_unavailable(
    service: _Service, provider: E2BSandboxProvider
) -> None:
    service.reply("GET", "/v2/sandboxes", (502, b"<html>Bad Gateway</html>"))
    with pytest.raises(SandboxUnavailable):
        provider.find({}, role="dev")


def test_control_calls_are_retried_while_the_failure_is_transient(
    service: _Service, provider: E2BSandboxProvider
) -> None:
    service.reply("POST", "/v2/sandboxes", (201, SANDBOX))
    sb = _create(provider)
    service.reply("DELETE", "/sandboxes/sbx-1", 503, (429, b"<html>slow down</html>"), 204)
    sb.kill()
    assert len(service.calls("DELETE", "/sandboxes/sbx-1")) == 3
    service.reply("GET", "/sandboxes/sbx-1", 500, (200, _listed("sbx-1", {"a": "b"})))
    assert sb.info().metadata == {"a": "b"}


def test_a_server_error_that_cannot_change_is_not_retried(service: _Service, provider: E2BSandboxProvider) -> None:
    service.reply("GET", "/v2/sandboxes", 501)
    with pytest.raises(SandboxUnavailable):
        provider.find({}, role="dev")
    assert len(service.calls("GET", "/v2/sandboxes")) == 1


def test_the_retry_budget_ends_in_unavailable(service: _Service) -> None:
    short = E2BSandboxProvider(
        {"dev": KEY_DEV},
        api_url=service.url,
        sandbox_url=service.url,
        retry=RetryPolicy(budget_sec=0.3, first_delay_sec=0.05, max_delay_sec=0.1),
    )
    service.reply("GET", "/v2/sandboxes", 503)
    with pytest.raises(SandboxUnavailable):
        short.find({}, role="dev")
    assert len(service.calls("GET", "/v2/sandboxes")) > 1


def test_an_unreachable_service_is_unavailable() -> None:
    nowhere = E2BSandboxProvider(
        {"dev": KEY_DEV},
        api_url="http://127.0.0.1:9",
        sandbox_url="http://127.0.0.1:9",
        retry=RetryPolicy(budget_sec=0.2, first_delay_sec=0.05, max_delay_sec=0.1),
    )
    with pytest.raises(SandboxUnavailable):
        nowhere.find({}, role="dev")


def test_an_unknown_sandbox_is_gone_and_a_repeated_kill_is_quiet(
    service: _Service, provider: E2BSandboxProvider
) -> None:
    service.reply("POST", "/v2/sandboxes/nope/connect", (404, {"code": 404, "message": "nope"}))
    with pytest.raises(SandboxGone):
        provider.connect("nope", role="dev")
    service.reply("POST", "/v2/sandboxes", (201, SANDBOX))
    sb = _create(provider)
    service.reply("GET", "/sandboxes/sbx-1", (404, {"code": 404, "message": "gone"}))
    with pytest.raises(SandboxGone):
        sb.info()
    service.reply("DELETE", "/sandboxes/sbx-1", (404, {"code": 404, "message": "gone"}))
    sb.kill()
    service.reply("POST", "/sandboxes/sbx-1/timeout", (404, {"code": 404, "message": "gone"}))
    with pytest.raises(SandboxGone):
        sb.set_timeout(10)


def test_find_pages_through_the_listing_and_filters_running_sandboxes(
    service: _Service, provider: E2BSandboxProvider
) -> None:
    service.reply("GET", "/v2/sandboxes", (200, [_listed("sbx-b", {"k": "v"})]), headers={"x-next-token": "t2"})
    service._script[("GET", "/v2/sandboxes")].append((200, [_listed("sbx-a", {"k": "v"})], {}, 0.0))
    found = provider.find({"agent-sandbox.io/idempotency-key": "k 1", "k": "v"}, role="dev")
    assert [i.sandbox_id for i in found] == ["sbx-a", "sbx-b"]
    first, second = service.calls("GET", "/v2/sandboxes")
    assert first.query["state"] == ["running"]
    assert unquote(first.query["metadata"][0]).count("=") == 2
    assert second.query["nextToken"] == ["t2"]


def test_a_role_without_a_key_is_denied_before_any_request(service: _Service, provider: E2BSandboxProvider) -> None:
    with pytest.raises(SandboxDenied, match="subrun"):
        provider.create("dev", role="subrun", timeout_sec=60, metadata={})
    with pytest.raises(SandboxDenied):
        provider.find({}, role="subrun")
    assert service.seen == []


def test_keys_stay_out_of_repr_errors_and_logs(
    service: _Service, provider: E2BSandboxProvider, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    service.reply("POST", "/v2/sandboxes", (401, {"code": 401, "message": "Invalid API key"}))
    with pytest.raises(SandboxDenied) as denied:
        _create(provider)
    shown = [repr(provider), str(provider), str(denied.value), repr(denied.value), caplog.text]
    assert not any(KEY_DEV in text or KEY_MEASUREMENT in text for text in shown)


def test_from_environ_reads_the_keys_and_both_addresses(service: _Service) -> None:
    environ = {
        "HYPERLOOM_SANDBOX_KEY_DEV": KEY_DEV,
        "HYPERLOOM_SANDBOX_KEY_SUBRUN": "",
        "E2B_API_URL": service.url,
        "E2B_SANDBOX_URL": service.url,
    }
    provider = E2BSandboxProvider.from_environ(environ, retry=FAST)
    assert repr(provider) == "E2BSandboxProvider(roles=['dev'])"
    service.reply("GET", "/v2/sandboxes", (200, []))
    assert provider.find({}, role="dev") == []
    assert service.calls("GET", "/v2/sandboxes")[0].headers["x-api-key"] == KEY_DEV


#: Addresses that resolve nowhere, for a provider that is refused before it sends anything.
NOWHERE = {"E2B_API_URL": "http://api.invalid", "E2B_SANDBOX_URL": "http://sandboxes.invalid"}


@pytest.mark.parametrize("missing", sorted(NOWHERE))
@pytest.mark.parametrize("given", ["unset", "empty"])
def test_from_environ_refuses_a_missing_service_url_before_any_request(
    monkeypatch: pytest.MonkeyPatch, missing: str, given: str
) -> None:
    """Without the URL the SDK would take its own default, the public E2B service, and send it the role's key."""
    for name in (*NOWHERE, "E2B_DOMAIN"):
        monkeypatch.delenv(name, raising=False)
    sent: list[httpx.Request] = []

    def send(client: httpx.Client, request: httpx.Request, *args: Any, **kwargs: Any) -> httpx.Response:
        sent.append(request)
        raise httpx.ConnectError("intercepted", request=request)

    monkeypatch.setattr(httpx.Client, "send", send)
    environ = {"HYPERLOOM_SANDBOX_KEY_DEV": KEY_DEV, **NOWHERE}
    if given == "unset":
        del environ[missing]
    else:
        environ[missing] = ""
    with pytest.raises(ValueError, match=f"^{missing} is not set"):
        E2BSandboxProvider.from_environ(environ, retry=FAST).find({}, role="dev")
    assert [str(request.url) for request in sent] == []


@pytest.mark.parametrize("blank", ["api_url", "sandbox_url"])
def test_a_blank_service_url_is_refused(blank: str) -> None:
    urls = {"api_url": NOWHERE["E2B_API_URL"], "sandbox_url": NOWHERE["E2B_SANDBOX_URL"], blank: " "}
    variable = "E2B_API_URL" if blank == "api_url" else "E2B_SANDBOX_URL"
    with pytest.raises(ValueError, match=f"^{variable} is not set"):
        E2BSandboxProvider({"dev": KEY_DEV}, **urls)


#: The workspace a SaFE key used directly names, and the variable that sets it.
WORKSPACE = "top-models"
WORKSPACE_ENV = "HYPERLOOM_SANDBOX_WORKSPACE"


def test_the_workspace_goes_with_every_create_and_a_workspace_the_caller_names_wins(service: _Service) -> None:
    keys = {"dev": KEY_DEV, "measurement": KEY_MEASUREMENT}
    provider = E2BSandboxProvider(keys, api_url=service.url, sandbox_url=service.url, retry=FAST, workspace=WORKSPACE)
    service.reply("POST", "/v2/sandboxes", (201, SANDBOX))
    given = {"a": "b"}
    provider.create("dev", role="dev", timeout_sec=60, metadata=given)
    provider.create("dev", role="measurement", timeout_sec=60, metadata={IDEMPOTENCY_KEY: "k-1"})
    provider.create("dev", role="dev", timeout_sec=60, metadata={WORKSPACE_KEY: "other"})
    assert [call.body["metadata"] for call in service.calls("POST", "/v2/sandboxes")] == [
        {"a": "b", WORKSPACE_KEY: WORKSPACE},
        {IDEMPOTENCY_KEY: "k-1", WORKSPACE_KEY: WORKSPACE},
        {WORKSPACE_KEY: "other"},
    ]
    assert given == {"a": "b"}, "the caller's mapping is not changed"


@pytest.mark.parametrize(
    ("named", "sent"),
    [
        ({}, {"a": "b"}),
        ({WORKSPACE_ENV: ""}, {"a": "b"}),
        ({WORKSPACE_ENV: WORKSPACE}, {"a": "b", WORKSPACE_KEY: WORKSPACE}),
    ],
    ids=["unset_as_on_claw", "empty", "set_for_a_safe_key"],
)
def test_from_environ_sends_a_workspace_only_when_the_variable_names_one(
    service: _Service, named: dict[str, str], sent: dict[str, str]
) -> None:
    environ = {"HYPERLOOM_SANDBOX_KEY_DEV": KEY_DEV, "E2B_API_URL": service.url, "E2B_SANDBOX_URL": service.url}
    provider = E2BSandboxProvider.from_environ({**environ, **named}, retry=FAST)
    service.reply("POST", "/v2/sandboxes", (201, SANDBOX))
    _create(provider, metadata={"a": "b"})
    [call] = service.calls("POST", "/v2/sandboxes")
    assert call.body["metadata"] == sent


def test_a_blank_workspace_is_refused() -> None:
    with pytest.raises(ValueError, match="workspace is blank"):
        E2BSandboxProvider(
            {"dev": KEY_DEV}, api_url=NOWHERE["E2B_API_URL"], sandbox_url=NOWHERE["E2B_SANDBOX_URL"], workspace="  "
        )


#: How the service refuses a create by a key whose user may create in several workspaces when it names none (a 400
#: ``workspace_required``), in the E2B error body (the SDK requires its ``code`` and ``message``, and keeps only
#: ``message``): the reason alone, or first, as the service words its other reasons.
WORKSPACE_REQUIRED_BODIES = [
    {"code": 400, "message": "workspace_required"},
    {"code": 400, "reason": "workspace_required", "message": "workspace_required: the user is in 5 workspaces"},
]


@pytest.mark.parametrize("body", WORKSPACE_REQUIRED_BODIES, ids=["bare", "with_reason"])
def test_a_create_refused_for_want_of_a_workspace_names_the_variable_and_is_not_replayed(
    service: _Service, provider: E2BSandboxProvider, body: dict[str, Any]
) -> None:
    service.reply("POST", "/v2/sandboxes", (400, body), (201, SANDBOX))
    with pytest.raises(SandboxDenied) as refused:
        _create(provider, metadata={IDEMPOTENCY_KEY: "k-1"})
    said = str(refused.value)
    assert WORKSPACE_ENV in said and "'dev' key" in said and "named none" in said
    assert "workspace_required" in said, "the service's own reason is kept"
    assert KEY_DEV not in said
    assert len(service.calls("POST", "/v2/sandboxes")) == 1


def test_a_workspace_refused_although_one_was_sent_says_which_and_another_refusal_is_left_as_it_is(
    service: _Service,
) -> None:
    keys = {"measurement": KEY_MEASUREMENT}
    provider = E2BSandboxProvider(keys, api_url=service.url, sandbox_url=service.url, retry=FAST, workspace=WORKSPACE)
    service.reply("POST", "/v2/sandboxes", (400, WORKSPACE_REQUIRED_BODIES[0]))
    with pytest.raises(SandboxDenied, match=f"named '{WORKSPACE}': set {WORKSPACE_ENV}"):
        provider.create("t", role="measurement", timeout_sec=60, metadata={})
    service.reply("POST", "/v2/sandboxes", (400, {"code": 400, "message": "template_not_allowed"}))
    with pytest.raises(SandboxDenied) as refused:
        provider.create("t", role="measurement", timeout_sec=60, metadata={})
    assert str(refused.value) == "400: template_not_allowed"


class _Refusing:
    """A stand-in for the SDK's sandbox handle whose data plane fails the way a sealed or vanished sandbox does."""

    def __init__(self, error: Exception) -> None:
        self.sandbox_id = "sbx-1"
        self.files = self.commands = self
        self._error = error

    def write(self, *args: object, **kw: object) -> None:
        raise self._error

    def run(self, *args: object, **kw: object) -> None:
        raise self._error


def _sandbox_failing_with(provider: E2BSandboxProvider, error: Exception):
    from hyperloom.sandbox.e2b import _E2BSandbox

    return _E2BSandbox(_Refusing(error), provider._control("dev"))


@pytest.mark.parametrize(
    "error",
    [
        e2b.SandboxException("403: path is not writable in a sealed sandbox"),
        e2b.SandboxException("Code.PERMISSION_DENIED: commands are not allowed"),
        e2b.AuthenticationException("bad token"),
    ],
)
def test_a_refusal_by_the_data_plane_is_denied(provider: E2BSandboxProvider, error: Exception) -> None:
    sandbox = _sandbox_failing_with(provider, error)
    with pytest.raises(SandboxDenied):
        sandbox.write_file("/out/x", b"x")
    with pytest.raises(SandboxDenied):
        sandbox.run("echo hi")


def test_an_unreachable_data_plane_is_gone_only_when_the_service_says_so(
    service: _Service, provider: E2BSandboxProvider
) -> None:
    sandbox = _sandbox_failing_with(provider, e2b.SandboxUnreachableException("cannot reach it"))
    service.reply("GET", "/sandboxes/sbx-1", (404, {"code": 404, "message": "gone"}))
    with pytest.raises(SandboxGone):
        sandbox.write_file("/work/x", b"x")
    service.reply("GET", "/sandboxes/sbx-1", (200, _listed("sbx-1", {})))
    with pytest.raises(SandboxUnavailable):
        sandbox.write_file("/work/x", b"x")


def test_a_sandbox_that_is_not_running_is_gone_and_a_bad_argument_is_not_hidden(provider: E2BSandboxProvider) -> None:
    with pytest.raises(SandboxGone):
        _sandbox_failing_with(provider, e2b.SandboxNotRunningException("killed")).run("true")
    with pytest.raises(e2b.InvalidArgumentException):
        _sandbox_failing_with(provider, e2b.InvalidArgumentException("bad path")).write_file("/x", b"")


def test_a_command_timeout_is_unavailable_and_its_settings_reach_the_sdk(provider: E2BSandboxProvider) -> None:
    with pytest.raises(SandboxUnavailable):
        _sandbox_failing_with(provider, e2b.TimeoutException("deadline")).run(
            "sleep 9", settings=RunSettings(timeout_sec=1), sinks=OutputSinks()
        )


def test_a_retry_policy_must_make_sense() -> None:
    with pytest.raises(ValueError):
        RetryPolicy(first_delay_sec=0)
    with pytest.raises(ValueError):
        RetryPolicy(first_delay_sec=2, max_delay_sec=1)


def test_chunks_reach_the_callback_as_whole_lines() -> None:
    lines: list[str] = []
    sink = _LineSink(lines.append)
    for chunk in ("he", "llo\nwor", "ld\n\nlast"):
        sink.feed(chunk)
    sink.flush()
    assert lines == ["hello\n", "world\n", "\n", "last"]
    _LineSink(None).feed("ignored")
