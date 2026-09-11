# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the InferenceX HTTP client (env resolvers and transport)."""

from __future__ import annotations

import gzip
import json
from urllib.parse import parse_qs, urlsplit

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


def test_find_reference_rows_precision_is_hard_filter():
    rows = [_bench_row(precision="fp8"), _bench_row(precision="fp4")]
    # Exact precision keeps only that precision.
    assert len(ix.find_reference_rows(rows, hardware="b300", isl=1024, osl=1024, precision="fp4")) == 1
    # Unavailable precision drops everything (never substitutes another precision).
    assert ix.find_reference_rows(rows, hardware="b300", isl=1024, osl=1024, precision="bf16") == []
    # Empty precision leaves the precision dimension unconstrained.
    assert len(ix.find_reference_rows(rows, hardware="b300", isl=1024, osl=1024, precision="")) == 2


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


@pytest.mark.parametrize("isl,osl", [(None, None), (1024, 8192)])
def test_agentx_matches_explicit_agentic_rows_without_fixed_lengths(isl, osl):
    agentic = _bench_row(benchmark_type="agentic_traces", isl=None, osl=None)
    rows = [agentic, _bench_row(benchmark_type="single_turn"), _bench_row(isl=None, osl=None)]

    assert ix.find_reference_rows(rows, hardware="b300", isl=isl, osl=osl, benchmark_mode="agentx") == [agentic]


def test_agentx_still_filters_hardware_precision_and_topology():
    matching = _bench_row(benchmark_type="agentic_traces", isl=None, osl=None, precision="fp4")
    rows = [
        matching,
        *(
            {**matching, **change}
            for change in (
                {"hardware": "h100"},
                {"precision": "fp8"},
                {"is_multinode": True},
                {"disagg": True},
            )
        ),
    ]

    assert ix.find_reference_rows(
        rows, hardware="B300", isl=None, osl=None, precision="FP4", benchmark_mode="agentx"
    ) == [matching]


def test_synthetic_keeps_legacy_rows_but_excludes_other_workloads():
    legacy = _bench_row()
    single_turn = _bench_row(benchmark_type="single_turn")
    rows = [legacy, single_turn, _bench_row(benchmark_type="agentic_traces"), _bench_row(benchmark_type="unknown")]

    assert ix.find_reference_rows(rows, hardware="b300", isl=1024, osl=1024) == [legacy, single_turn]


# ---- fetch_agentic_interactivity -------------------------------------------
def test_agentic_interactivity_joins_by_normalized_id_not_response_order(monkeypatch):
    calls = []
    payload = {
        "20": {"id": 20, "p90_e2e_norm_intvty": 8.5, "p75_e2e_norm_intvty": 99},
        "10": {"id": 10, "p90_e2e_norm_intvty": 4.25},
        "999": {"id": 999, "p90_e2e_norm_intvty": 1000},
    }

    def fetch(url):
        calls.append(url)
        return json.dumps(payload).encode()

    monkeypatch.setenv("INFERENCEX_BASE_URL", "https://reference.test/api/v1")
    monkeypatch.setattr(ix, "_fetch_raw", fetch)
    assert ix.fetch_agentic_interactivity(["0010", 20, "20"]) == {"10": 4.25, "20": 8.5}
    assert len(calls) == 1
    assert urlsplit(calls[0]).path == "/api/v1/derived-agentic-metrics"
    assert parse_qs(urlsplit(calls[0]).query) == {"ids": ["10,20"]}


@pytest.mark.parametrize("invalid", [None, 0, -1, True, "12.0", float("nan"), float("inf")])
def test_agentic_interactivity_keeps_unavailable_values_missing(monkeypatch, invalid):
    payload = {"1": {"id": 1, "p90_e2e_norm_intvty": invalid, "p90_intvty": 100, "mean_tpot": 0.01}}
    monkeypatch.setattr(ix, "_fetch_raw", lambda url: json.dumps(payload).encode())

    assert ix.fetch_agentic_interactivity([1, 2]) == {"1": None}


def test_agentic_interactivity_batches_no_more_than_200_ids(monkeypatch):
    requested = []

    def fetch(url):
        ids = parse_qs(urlsplit(url).query)["ids"][0].split(",")
        requested.append(ids)
        return json.dumps({key: {"id": int(key), "p90_e2e_norm_intvty": float(key)} for key in ids}).encode()

    monkeypatch.setattr(ix, "_fetch_raw", fetch)
    ids = list(range(1, 402))
    result = ix.fetch_agentic_interactivity([*ids, "1"])
    assert [len(batch) for batch in requested] == [200, 200, 1]
    assert result == {str(key): float(key) for key in ids}


def test_agentic_interactivity_empty_input_does_not_fetch(monkeypatch):
    monkeypatch.setattr(ix, "_fetch_raw", lambda url: pytest.fail("empty input must not request the API"))
    assert ix.fetch_agentic_interactivity([]) == {}


@pytest.mark.parametrize("value", [True, 0, -1, 1.5, "1.0", "1,2", "", "abc"])
def test_agentic_interactivity_rejects_invalid_request_ids(monkeypatch, value):
    monkeypatch.setattr(ix, "_fetch_raw", lambda url: pytest.fail("invalid IDs must not request the API"))
    with pytest.raises(ValueError, match="benchmark ID"):
        ix.fetch_agentic_interactivity([value])


def test_agentic_interactivity_retries_transport_then_parses_gzip(monkeypatch):
    calls = []
    monkeypatch.setenv("INFERENCEX_MAX_ATTEMPTS", "2")

    def fetch(url):
        calls.append(url)
        if len(calls) == 1:
            raise ix.InferenceXFetchError("HTTP 503")
        return gzip.compress(json.dumps({"1": {"id": 1, "p90_e2e_norm_intvty": 8.0}}).encode())

    monkeypatch.setattr(ix, "_fetch_raw", fetch)
    assert ix.fetch_agentic_interactivity([1]) == {"1": 8.0}
    assert len(calls) == 2


@pytest.mark.parametrize(
    "body",
    [
        b"not-json",
        b"[]",
        b'{"error":"unavailable"}',
        b'{"1":{"id":2,"p90_e2e_norm_intvty":8}}',
        b'{"1":8}',
    ],
)
def test_agentic_interactivity_distinguishes_invalid_response_from_no_data(monkeypatch, body):
    monkeypatch.setattr(ix, "_fetch_raw", lambda url: body)
    assert ix.fetch_agentic_interactivity([1]) is None


def test_agentic_interactivity_empty_response_is_valid_no_data(monkeypatch):
    monkeypatch.setattr(ix, "_fetch_raw", lambda url: b"{}")
    assert ix.fetch_agentic_interactivity([1]) == {}


def test_agentic_interactivity_failed_batch_does_not_return_partial_success(monkeypatch):
    calls = []
    monkeypatch.setenv("INFERENCEX_MAX_ATTEMPTS", "2")

    def fetch(url):
        calls.append(url)
        if parse_qs(urlsplit(url).query)["ids"] == ["201"]:
            raise ix.InferenceXFetchError("HTTP 503")
        return json.dumps({"1": {"id": 1, "p90_e2e_norm_intvty": 8.0}}).encode()

    monkeypatch.setattr(ix, "_fetch_raw", fetch)
    assert ix.fetch_agentic_interactivity(list(range(1, 202))) is None
    assert len(calls) == 3


@pytest.mark.parametrize("encoding", ["", "gzip"])
def test_agentic_interactivity_rejects_truncated_gzip_response(monkeypatch, encoding):
    body = gzip.compress(b'{"1":{"id":1,"p90_e2e_norm_intvty":8.0}}')[:-8]
    monkeypatch.setattr(ix.urllib.request, "urlopen", lambda req, **kwargs: _FakeResp(body=body, encoding=encoding))
    assert ix.fetch_agentic_interactivity([1]) is None


def test_agentic_interactivity_rejects_non_http_base_url(monkeypatch):
    monkeypatch.setenv("INFERENCEX_BASE_URL", "file:///reference")
    monkeypatch.setattr(ix.urllib.request, "urlopen", lambda *args, **kwargs: pytest.fail("unsafe URL opened"))
    assert ix.fetch_agentic_interactivity([1]) is None
