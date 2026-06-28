# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Unit coverage for the urllib-based HTTPKBClient transport."""

from __future__ import annotations

import io
import urllib.error

import pytest

from runtime.errors import (
    KBConflictError,
    KBNotFoundError,
    KBTransportError,
    KBValidationError,
)
from runtime.kb_client import HTTPKBClient


class _Resp:
    def __init__(self, body: bytes = b'{"ok": true}', status: int = 200):
        self._body = body
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return self._body


def _http_error(code: int, body: bytes = b"err"):
    return urllib.error.HTTPError(
        url="http://kb.local/api/kb/upsert",
        code=code,
        msg="x",
        hdrs={},  # type: ignore[arg-type]
        fp=io.BytesIO(body),
    )


def test_init_requires_base_url() -> None:
    with pytest.raises(ValueError):
        HTTPKBClient("")


def test_init_strips_slash_and_token(monkeypatch) -> None:
    monkeypatch.delenv("KB_SERVICE_TOKEN", raising=False)
    client = HTTPKBClient("http://kb.local/", token="abc", timeout_ms=2000)
    assert client.base_url == "http://kb.local"
    assert client.token == "abc"
    assert client.timeout_s == 2.0


def test_endpoint_methods_build_bodies(monkeypatch) -> None:
    client = HTTPKBClient("http://kb.local")
    seen = []

    def fake_request(path, body):
        seen.append((path, body))
        return {"echo": path}

    monkeypatch.setattr(client, "_request", fake_request)

    client.list(scope_filter={"model": "m"}, kind="pitfall", metadata_filter={"a": 1}, limit=5)
    client.upsert({"topic": "t"})
    client.batch_insert([{"x": 1}], on_conflict="skip")
    client.add_edges([{"src": "a", "dst": "b"}])

    paths = [p for p, _ in seen]
    assert paths == [
        "/api/kb/list",
        "/api/kb/upsert",
        "/api/kb/batch_insert",
        "/api/kb/edges/add",
    ]
    list_body = seen[0][1]
    assert list_body["kind"] == "pitfall"
    assert list_body["metadata_filter"] == {"a": 1}
    assert list_body["limit"] == 5
    assert seen[2][1] == {"items": [{"x": 1}], "on_conflict": "skip"}


def test_request_success(monkeypatch) -> None:
    client = HTTPKBClient("http://kb.local", token="tok")
    monkeypatch.setattr(
        "runtime.kb_client.urllib.request.urlopen",
        lambda req, timeout=None: _Resp(b'{"result": 1}', status=200),
    )
    out = client.upsert({"topic": "t"})
    assert out == {"result": 1}


def test_request_empty_body(monkeypatch) -> None:
    client = HTTPKBClient("http://kb.local")
    monkeypatch.setattr(
        "runtime.kb_client.urllib.request.urlopen",
        lambda req, timeout=None: _Resp(b"", status=200),
    )
    assert client.upsert({"topic": "t"}) == {}


def test_request_404(monkeypatch) -> None:
    client = HTTPKBClient("http://kb.local")

    def boom(req, timeout=None):
        raise _http_error(404)

    monkeypatch.setattr("runtime.kb_client.urllib.request.urlopen", boom)
    with pytest.raises(KBNotFoundError):
        client.upsert({"topic": "t"})


def test_request_409(monkeypatch) -> None:
    client = HTTPKBClient("http://kb.local")

    def boom(req, timeout=None):
        raise _http_error(409)

    monkeypatch.setattr("runtime.kb_client.urllib.request.urlopen", boom)
    with pytest.raises(KBConflictError):
        client.upsert({"topic": "t"})


def test_request_validation_4xx(monkeypatch) -> None:
    client = HTTPKBClient("http://kb.local")

    def boom(req, timeout=None):
        raise _http_error(422)

    monkeypatch.setattr("runtime.kb_client.urllib.request.urlopen", boom)
    with pytest.raises(KBValidationError):
        client.upsert({"topic": "t"})


def test_request_5xx_retries_then_transport(monkeypatch) -> None:
    slept = []
    client = HTTPKBClient("http://kb.local", retry_max=2, sleep_fn=slept.append)

    def boom(req, timeout=None):
        raise _http_error(503)

    monkeypatch.setattr("runtime.kb_client.urllib.request.urlopen", boom)
    with pytest.raises(KBTransportError):
        client.upsert({"topic": "t"})
    # retry_max=2 -> two backoff sleeps before giving up.
    assert len(slept) == 2


def test_request_urlerror_retries_then_transport(monkeypatch) -> None:
    slept = []
    client = HTTPKBClient("http://kb.local", retry_max=1, sleep_fn=slept.append)

    def boom(req, timeout=None):
        raise urllib.error.URLError("network down")

    monkeypatch.setattr("runtime.kb_client.urllib.request.urlopen", boom)
    with pytest.raises(KBTransportError):
        client.upsert({"topic": "t"})
    assert len(slept) == 1


def test_request_recovers_after_one_failure(monkeypatch) -> None:
    client = HTTPKBClient("http://kb.local", retry_max=3, sleep_fn=lambda _s: None)
    calls = {"n": 0}

    def flaky(req, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise urllib.error.URLError("transient")
        return _Resp(b'{"ok": 1}', status=200)

    monkeypatch.setattr("runtime.kb_client.urllib.request.urlopen", flaky)
    assert client.upsert({"topic": "t"}) == {"ok": 1}
    assert calls["n"] == 2


def test_backoff_grows(monkeypatch) -> None:
    monkeypatch.setattr("runtime.kb_client.random.random", lambda: 0.5)
    client = HTTPKBClient("http://kb.local", backoff_base=1.0)
    b1 = client._backoff_for(1)
    b2 = client._backoff_for(2)
    assert b2 > b1
