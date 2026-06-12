# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Supplementary coverage for RemoteRecipeClient transport + error decoding."""

from __future__ import annotations

import httpx
import pytest
import respx

from inference_optimizer.recipe_kb import (
    RemoteRecipeClient,
    RemoteRecipeClientError,
)
from inference_optimizer.recipe_kb import remote_client as rc
from inference_optimizer.recipe_snapshot_constants import PATH_HEALTH


KB_URL = "http://kb-test.local"


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(rc.time, "sleep", lambda *_a, **_k: None)


@pytest.fixture
def _client(monkeypatch):
    for key in ("CORTEX_KB_URL", "CORTEX_KB_HTTP_TIMEOUT_SEC",
                "CORTEX_KB_RETRY_ATTEMPTS", "KB_SERVICE_TOKEN"):
        monkeypatch.delenv(key, raising=False)
    return RemoteRecipeClient(kb_url=KB_URL, foreground=False, retry_attempts=2)


# ---- _parse_error_envelope ----
def test_parse_error_envelope_non_json():
    resp = httpx.Response(400, text="plain text error")
    cat, code, msg, details = rc._parse_error_envelope(resp)
    assert cat == "unknown"
    assert "plain text error" in msg


def test_parse_error_envelope_business():
    resp = httpx.Response(409, json={"detail": {"error": {
        "code": "CONFLICT", "message": "exists", "details": {"x": 1}}}})
    cat, code, msg, details = rc._parse_error_envelope(resp)
    assert cat == "business"
    assert code == "CONFLICT"
    assert details == {"x": 1}


def test_parse_error_envelope_validation_list():
    resp = httpx.Response(422, json={"detail": [
        {"loc": ["body", "limit"], "msg": "must be int"},
        "not-a-mapping",
    ]})
    cat, code, msg, details = rc._parse_error_envelope(resp)
    assert cat == "validation"
    assert "body.limit" in msg


def test_parse_error_envelope_unknown_dict():
    resp = httpx.Response(400, json={"weird": "shape"})
    cat, code, msg, details = rc._parse_error_envelope(resp)
    assert cat == "unknown"


# ---- transport retry behaviour ----
def test_transport_retries_then_succeeds(_client):
    with respx.mock(base_url=KB_URL) as mock:
        mock.get(PATH_HEALTH).mock(side_effect=[
            httpx.Response(503, json={"detail": "warming"}),
            httpx.Response(200, json={"status": "ok"}),
        ])
        assert _client.health() is True


def test_transport_exhausted_raises(_client):
    with respx.mock(base_url=KB_URL) as mock:
        mock.get(PATH_HEALTH).mock(
            return_value=httpx.Response(503, json={"detail": "warming"}),
        )
        # health() swallows -> False; use a raising read path instead.
        with pytest.raises(RemoteRecipeClientError) as ei:
            _client._ensure_transport().request("GET", PATH_HEALTH)
    assert ei.value.category == "transport"


def test_transport_connect_error_retries_then_exhausts(_client):
    with respx.mock(base_url=KB_URL) as mock:
        mock.get(PATH_HEALTH).mock(side_effect=httpx.ConnectError("boom"))
        with pytest.raises(RemoteRecipeClientError) as ei:
            _client._ensure_transport().request("GET", PATH_HEALTH)
    assert ei.value.category == "transport"


def test_request_204_returns_empty_dict(_client):
    with respx.mock(base_url=KB_URL) as mock:
        mock.get("/x").mock(return_value=httpx.Response(204))
        assert _client._ensure_transport().request("GET", "/x") == {}


def test_request_non_json_raises(_client):
    with respx.mock(base_url=KB_URL) as mock:
        mock.get("/x").mock(return_value=httpx.Response(200, text="not json"))
        with pytest.raises(RemoteRecipeClientError):
            _client._ensure_transport().request("GET", "/x")


def test_request_bare_list_wrapped(_client):
    with respx.mock(base_url=KB_URL) as mock:
        mock.get("/x").mock(return_value=httpx.Response(200, json=[1, 2, 3]))
        assert _client._ensure_transport().request("GET", "/x") == {"_value": [1, 2, 3]}


def test_request_business_400_raises(_client):
    with respx.mock(base_url=KB_URL) as mock:
        mock.get("/x").mock(return_value=httpx.Response(
            400, json={"detail": {"error": {"code": "BAD", "message": "no"}}}))
        with pytest.raises(RemoteRecipeClientError) as ei:
            _client._ensure_transport().request("GET", "/x")
    assert ei.value.category == "business"


def test_transport_close_idempotent(_client):
    # build then close the underlying client; second close is a no-op
    _client._ensure_transport()._ensure_client()
    _client._ensure_transport().close()
    _client._ensure_transport().close()
