# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for the optional ``/v2/reasoning/assess`` client."""

from __future__ import annotations

import json
import urllib.error
from io import BytesIO

import pytest

from runtime.kb_assess_client import ASSESS_PATH, KBAssessClient


def test_from_env_returns_none_when_unset(monkeypatch):
    monkeypatch.delenv("CORTEX_KB_URL", raising=False)
    assert KBAssessClient.from_env() is None


def test_from_env_builds_client_when_url_set(monkeypatch):
    monkeypatch.setenv("CORTEX_KB_URL", "http://kb.local/")
    monkeypatch.setenv("CORTEX_KB_HTTP_TIMEOUT_SEC", "1.5")
    client = KBAssessClient.from_env()
    assert client is not None
    assert client.base_url == "http://kb.local"  # trailing slash stripped
    assert client.timeout_sec == 1.5


def test_assess_posts_to_v2_reasoning_assess(monkeypatch):
    captured = {}

    class _Resp:
        status = 200

        def read(self):
            return json.dumps({"reasonable": "supported", "verdicts": []}).encode()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def _fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["body"] = json.loads(req.data.decode())
        captured["timeout"] = timeout
        return _Resp()

    monkeypatch.setattr("runtime.kb_assess_client.urllib.request.urlopen", _fake_urlopen)

    client = KBAssessClient("http://kb.local", timeout_sec=2.0)
    out = client.assess(
        focus={"model": "Qwen3-14B"}, params={"kv_cache_dtype": "fp8"},
        envs={"VLLM_USE_AITER": "1"}, args="--quantization fp8")

    assert out == {"reasonable": "supported", "verdicts": []}
    assert captured["url"] == f"http://kb.local{ASSESS_PATH}"
    assert captured["body"]["focus"] == {"model": "Qwen3-14B"}
    assert captured["body"]["params"] == {"kv_cache_dtype": "fp8"}
    assert captured["body"]["envs"] == {"VLLM_USE_AITER": "1"}
    assert captured["body"]["args"] == "--quantization fp8"
    assert captured["timeout"] == 2.0


def test_assess_swallows_errors_and_returns_none(monkeypatch):
    def _boom(req, timeout=None):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr("runtime.kb_assess_client.urllib.request.urlopen", _boom)
    client = KBAssessClient("http://kb.local")
    assert client.assess(focus={"model": "m"}, params={"x": 1}) is None


def test_empty_base_url_raises():
    with pytest.raises(ValueError):
        KBAssessClient("")
