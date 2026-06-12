# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for the InferenceX HTTP client (env resolvers, SSL context,
and the never-raises ``fetch_rows`` retry / decode contract)."""
from __future__ import annotations

import ssl

import pytest

from inference_optimizer.baseline_comparison import inferencex_client as ix


# ---- env resolvers --------------------------------------------------------
def test_base_url_default_and_override(monkeypatch):
    monkeypatch.delenv("INFERENCEX_BASE_URL", raising=False)
    assert ix._base_url() == ix.DEFAULT_BASE_URL
    monkeypatch.setenv("INFERENCEX_BASE_URL", "http://local/api")
    assert ix._base_url() == "http://local/api"


def test_timeout_sec_variants(monkeypatch):
    monkeypatch.delenv("INFERENCEX_TIMEOUT_SEC", raising=False)
    assert ix._timeout_sec() == ix.DEFAULT_TIMEOUT_SEC
    monkeypatch.setenv("INFERENCEX_TIMEOUT_SEC", "0.1")  # clamped to 0.5 floor
    assert ix._timeout_sec() == 0.5
    monkeypatch.setenv("INFERENCEX_TIMEOUT_SEC", "not-a-float")
    assert ix._timeout_sec() == ix.DEFAULT_TIMEOUT_SEC


def test_max_attempts_variants(monkeypatch):
    monkeypatch.delenv("INFERENCEX_MAX_ATTEMPTS", raising=False)
    assert ix._max_attempts() == ix.DEFAULT_MAX_ATTEMPTS
    monkeypatch.setenv("INFERENCEX_MAX_ATTEMPTS", "0")  # clamped to >=1
    assert ix._max_attempts() == 1
    monkeypatch.setenv("INFERENCEX_MAX_ATTEMPTS", "bad")
    assert ix._max_attempts() == ix.DEFAULT_MAX_ATTEMPTS


def test_insecure_flag(monkeypatch):
    monkeypatch.setenv("INFERENCEX_INSECURE", "true")
    assert ix._insecure() is True
    monkeypatch.setenv("INFERENCEX_INSECURE", "0")
    assert ix._insecure() is False


def test_build_ssl_context(monkeypatch):
    monkeypatch.setenv("INFERENCEX_INSECURE", "0")
    assert ix._build_ssl_context() is None
    monkeypatch.setenv("INFERENCEX_INSECURE", "1")
    ctx = ix._build_ssl_context()
    assert isinstance(ctx, ssl.SSLContext)
    assert ctx.check_hostname is False
    assert ctx.verify_mode == ssl.CERT_NONE


# ---- fetch_rows -----------------------------------------------------------
def test_fetch_rows_empty_model():
    rows, warning = ix.fetch_rows("")
    assert rows is None
    assert "empty" in warning


def test_fetch_rows_success(monkeypatch):
    monkeypatch.setattr(ix, "_fetch_raw", lambda url: b'[{"gpu": "mi300x"}]')
    rows, warning = ix.fetch_rows("llama")
    assert rows == [{"gpu": "mi300x"}]
    assert warning == ""


def test_fetch_rows_non_list_payload(monkeypatch):
    monkeypatch.setattr(ix, "_fetch_raw", lambda url: b'{"not": "a list"}')
    rows, warning = ix.fetch_rows("llama")
    assert rows is None
    assert "JSON array" in warning


def test_fetch_rows_decode_error(monkeypatch):
    monkeypatch.setattr(ix, "_fetch_raw", lambda url: b"\xff\xfe not json")
    rows, warning = ix.fetch_rows("llama")
    assert rows is None
    assert "decode error" in warning


def test_fetch_rows_retries_then_fails(monkeypatch):
    monkeypatch.setattr(ix.time, "sleep", lambda s: None)
    monkeypatch.setenv("INFERENCEX_MAX_ATTEMPTS", "2")

    def _boom(url):
        raise ix.InferenceXFetchError("HTTP 503")

    monkeypatch.setattr(ix, "_fetch_raw", _boom)
    rows, warning = ix.fetch_rows("llama")
    assert rows is None
    assert "503" in warning


def test_fetch_rows_success_after_retry(monkeypatch):
    monkeypatch.setattr(ix.time, "sleep", lambda s: None)
    monkeypatch.setenv("INFERENCEX_MAX_ATTEMPTS", "3")
    calls = {"n": 0}

    def _flaky(url):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ix.InferenceXFetchError("transient")
        return b"[]"

    monkeypatch.setattr(ix, "_fetch_raw", _flaky)
    rows, warning = ix.fetch_rows("llama")
    assert rows == []
    assert warning == ""
    assert calls["n"] == 2


# ---- _fetch_raw -----------------------------------------------------------
class _FakeResp:
    def __init__(self, *, code=200, body=b"[]", encoding=""):
        self._code = code
        self._body = body
        self.headers = {"Content-Encoding": encoding}

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def getcode(self):
        return self._code

    def read(self):
        return self._body


def test_fetch_raw_plain_200(monkeypatch):
    monkeypatch.setattr(ix.urllib.request, "urlopen",
                        lambda req, timeout=None, context=None: _FakeResp(body=b"[1]"))
    assert ix._fetch_raw("http://x") == b"[1]"


def test_fetch_raw_gzip_200(monkeypatch):
    import gzip
    payload = gzip.compress(b"[2]")
    monkeypatch.setattr(
        ix.urllib.request, "urlopen",
        lambda req, timeout=None, context=None: _FakeResp(body=payload, encoding="gzip"),
    )
    assert ix._fetch_raw("http://x") == b"[2]"


def test_fetch_raw_non_200_raises(monkeypatch):
    monkeypatch.setattr(ix.urllib.request, "urlopen",
                        lambda req, timeout=None, context=None: _FakeResp(code=404))
    with pytest.raises(ix.InferenceXFetchError):
        ix._fetch_raw("http://x")


def test_fetch_raw_http_error(monkeypatch):
    from urllib.error import HTTPError

    def _raise(req, timeout=None, context=None):
        raise HTTPError("http://x", 500, "boom", {}, None)

    monkeypatch.setattr(ix.urllib.request, "urlopen", _raise)
    with pytest.raises(ix.InferenceXFetchError):
        ix._fetch_raw("http://x")


def test_fetch_raw_url_error(monkeypatch):
    from urllib.error import URLError

    def _raise(req, timeout=None, context=None):
        raise URLError("down")

    monkeypatch.setattr(ix.urllib.request, "urlopen", _raise)
    with pytest.raises(ix.InferenceXFetchError):
        ix._fetch_raw("http://x")


def test_fetch_raw_timeout(monkeypatch):
    import socket

    def _raise(req, timeout=None, context=None):
        raise socket.timeout()

    monkeypatch.setattr(ix.urllib.request, "urlopen", _raise)
    with pytest.raises(ix.InferenceXFetchError):
        ix._fetch_raw("http://x")


def test_fetch_raw_transport_error(monkeypatch):
    def _raise(req, timeout=None, context=None):
        raise OSError("reset")

    monkeypatch.setattr(ix.urllib.request, "urlopen", _raise)
    with pytest.raises(ix.InferenceXFetchError):
        ix._fetch_raw("http://x")
