# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for the InferenceX HTTP client (env resolvers, SSL context,
and the ``_fetch_raw`` transport)."""

from __future__ import annotations

import ssl

import pytest

from hyperloom.inference_optimizer.baseline_comparison import inferencex_client as ix


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
    monkeypatch.setattr(ix.urllib.request, "urlopen", lambda req, timeout=None, context=None: _FakeResp(body=b"[1]"))
    assert ix._fetch_raw("http://x") == b"[1]"


def test_fetch_raw_gzip_200(monkeypatch):
    import gzip

    payload = gzip.compress(b"[2]")
    monkeypatch.setattr(
        ix.urllib.request,
        "urlopen",
        lambda req, timeout=None, context=None: _FakeResp(body=payload, encoding="gzip"),
    )
    assert ix._fetch_raw("http://x") == b"[2]"


def test_fetch_raw_non_200_raises(monkeypatch):
    monkeypatch.setattr(ix.urllib.request, "urlopen", lambda req, timeout=None, context=None: _FakeResp(code=404))
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


# ---- fetch_rows -----------------------------------------------------------
def _bench_row(**overrides) -> dict:
    base = {
        "hardware": "b300",
        "precision": "fp8",
        "isl": 1024,
        "osl": 1024,
        "conc": 64,
        "decode_tp": 2,
        "is_multinode": False,
        "disagg": False,
        "metrics": {"tput_per_gpu": 100.0},
    }
    base.update(overrides)
    return base


def test_fetch_rows_empty_name_returns_none():
    assert ix.fetch_rows("") is None
    assert ix.fetch_rows("   ") is None


def test_fetch_rows_plain_list(monkeypatch):
    import json

    payload = [{"hardware": "b300"}]
    monkeypatch.setattr(ix, "_fetch_raw", lambda _url: json.dumps(payload).encode("utf-8"))
    assert ix.fetch_rows("MiniMax-M2.5") == payload


def test_fetch_rows_gzip_body(monkeypatch):
    import gzip
    import json

    gz = gzip.compress(json.dumps([{"hardware": "b300"}]).encode("utf-8"))
    monkeypatch.setattr(ix, "_fetch_raw", lambda _url: gz)
    assert ix.fetch_rows("MiniMax-M2.5") == [{"hardware": "b300"}]


def test_fetch_rows_structured_error_returns_empty(monkeypatch):
    import json

    monkeypatch.setattr(ix, "_fetch_raw", lambda _url: json.dumps({"error": "bad model"}).encode("utf-8"))
    assert ix.fetch_rows("MiniMax-M2.5") == []


def test_fetch_rows_wrapped_list_key(monkeypatch):
    import json

    monkeypatch.setattr(ix, "_fetch_raw", lambda _url: json.dumps({"data": [{"hardware": "h200"}]}).encode("utf-8"))
    assert ix.fetch_rows("MiniMax-M2.5") == [{"hardware": "h200"}]


def test_fetch_rows_unexpected_dict_returns_empty(monkeypatch):
    import json

    monkeypatch.setattr(ix, "_fetch_raw", lambda _url: json.dumps({"unexpected": 1}).encode("utf-8"))
    assert ix.fetch_rows("MiniMax-M2.5") == []


def test_fetch_rows_bad_json_returns_none(monkeypatch):
    monkeypatch.setattr(ix, "_fetch_raw", lambda _url: b"not-json{")
    assert ix.fetch_rows("MiniMax-M2.5") is None


def test_fetch_rows_retries_then_none(monkeypatch):
    calls = {"n": 0}

    def _boom(_url):
        calls["n"] += 1
        raise ix.InferenceXFetchError("HTTP 503")

    monkeypatch.setenv("INFERENCEX_MAX_ATTEMPTS", "3")
    monkeypatch.setattr(ix, "_fetch_raw", _boom)
    assert ix.fetch_rows("MiniMax-M2.5") is None
    assert calls["n"] == 3


# ---- find_reference_rows --------------------------------------------------
def test_find_reference_rows_strict_shape_and_hardware():
    rows = [_bench_row(), _bench_row(hardware="mi300x"), _bench_row(isl=8192), _bench_row(osl=8192)]
    out = ix.find_reference_rows(rows, hardware="b300", isl=1024, osl=1024)
    assert len(out) == 1
    assert out[0]["hardware"] == "b300"


def test_find_reference_rows_case_insensitive_hardware():
    out = ix.find_reference_rows([_bench_row(hardware="B300")], hardware="b300", isl=1024, osl=1024)
    assert len(out) == 1


def test_find_reference_rows_precision_filter_and_fallback():
    rows = [_bench_row(precision="fp8"), _bench_row(precision="fp4")]
    assert len(ix.find_reference_rows(rows, hardware="b300", isl=1024, osl=1024, precision="fp4")) == 1
    # unavailable precision degrades to "any" instead of dropping everything
    assert len(ix.find_reference_rows(rows, hardware="b300", isl=1024, osl=1024, precision="bf16")) == 2


def test_find_reference_rows_excludes_disagg_and_multinode():
    rows = [
        _bench_row(),
        _bench_row(disagg=True, metrics={"tput_per_gpu": 999999.0}),
        _bench_row(is_multinode=True, metrics={"tput_per_gpu": 888888.0}),
    ]
    out = ix.find_reference_rows(rows, hardware="b300", isl=1024, osl=1024)
    assert len(out) == 1
    assert out[0]["metrics"]["tput_per_gpu"] == 100.0


def test_find_reference_rows_missing_topology_fields_treated_single_node():
    row = {"hardware": "b300", "isl": 1024, "osl": 1024, "metrics": {"tput_per_gpu": 5.0}}
    assert len(ix.find_reference_rows([row], hardware="b300", isl=1024, osl=1024)) == 1


def test_find_reference_rows_empty_when_no_shape_match():
    assert ix.find_reference_rows([_bench_row()], hardware="b300", isl=2048, osl=2048) == []
