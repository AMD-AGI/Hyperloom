# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for ``orchestrator.predictor.config`` and ``.client``.

The client is exercised against a real loopback HTTP server rather than a
patched ``urlopen``: the failure modes worth covering here are status codes,
truncated bodies and timeouts, and a mock of ``urlopen`` would assert the mock.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from hyperloom.orchestrator.predictor import client as cl
from hyperloom.orchestrator.predictor import config as cfg


class _Handler(BaseHTTPRequestHandler):
    """Replies with whatever the server was configured to return."""

    def do_POST(self):  # noqa: N802 - BaseHTTPRequestHandler's spelling
        length = int(self.headers.get("Content-Length") or 0)
        self.server.last_body = json.loads(self.rfile.read(length) or "{}")
        self.server.last_path = self.path
        status, payload = self.server.reply
        raw = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *args):  # noqa: D102 - silence the test log
        pass


@pytest.fixture
def service():
    """A loopback predictor whose reply each test sets."""
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    server.reply = (200, {"schema": "primatune.predictor_response.v1", "parsed": False})
    server.last_body = None
    server.last_path = None
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    server.endpoint = f"http://127.0.0.1:{server.server_port}"
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _predict(service, request=None):
    return cl.predict(request or {"schema": "x"}, endpoint=service.endpoint, timeout_sec=5.0)


class TestConfig:
    def test_defaults_to_shadow_and_disabled(self, monkeypatch):
        for name in (cfg.ENV_ENDPOINT, cfg.ENV_MODE, cfg.ENV_MAX_CHAIN):
            monkeypatch.delenv(name, raising=False)
        conf = cfg.load()
        assert conf.mode == cfg.MODE_SHADOW
        assert conf.enabled is False  # no endpoint
        assert conf.enqueues is False

    def test_endpoint_enables_shadow_but_not_enqueue(self, monkeypatch):
        monkeypatch.setenv(cfg.ENV_ENDPOINT, "http://host:8000")
        monkeypatch.delenv(cfg.ENV_MODE, raising=False)
        conf = cfg.load()
        assert conf.enabled is True
        assert conf.enqueues is False

    def test_active_mode_enqueues(self, monkeypatch):
        monkeypatch.setenv(cfg.ENV_ENDPOINT, "http://host:8000")
        monkeypatch.setenv(cfg.ENV_MODE, "active")
        assert cfg.load().enqueues is True

    def test_off_mode_wins_over_an_endpoint(self, monkeypatch):
        monkeypatch.setenv(cfg.ENV_ENDPOINT, "http://host:8000")
        monkeypatch.setenv(cfg.ENV_MODE, "off")
        assert cfg.load().enabled is False

    def test_unknown_mode_falls_back_to_shadow(self, monkeypatch):
        """A typo must not silently start enqueueing."""
        monkeypatch.setenv(cfg.ENV_ENDPOINT, "http://host:8000")
        monkeypatch.setenv(cfg.ENV_MODE, "activate")
        assert cfg.load().mode == cfg.MODE_SHADOW

    def test_bad_numbers_fall_back_without_raising(self, monkeypatch):
        monkeypatch.setenv(cfg.ENV_MAX_CHAIN, "lots")
        monkeypatch.setenv(cfg.ENV_BUDGET_PCT, "-5")
        conf = cfg.load()
        assert conf.max_chain == cfg.DEFAULT_MAX_CHAIN
        assert conf.budget_pct == cfg.DEFAULT_BUDGET_PCT

    def test_only_sglang_and_vllm_are_supported(self):
        conf = cfg.PredictorConfig()
        assert conf.supports("vllm") and conf.supports("SGLang")
        assert not conf.supports("atom")
        assert not conf.supports("")
        assert not conf.supports(None)


class TestPredictSuccess:
    def test_returns_the_parsed_action(self, service):
        service.reply = (
            200,
            {
                "schema": "primatune.predictor_response.v1",
                "parsed": True,
                "action": {
                    "server_args": {"--max-num-batched-tokens": "16384"},
                    "envs": {"VLLM_ROCM_USE_AITER": "1"},
                    "source_change": None,
                },
                "meta": {"model": "m", "dropped_flags": ["--bogus"]},
            },
        )
        out = _predict(service)
        assert out.parsed is True
        assert out.server_args == {"--max-num-batched-tokens": "16384"}
        assert out.envs == {"VLLM_ROCM_USE_AITER": "1"}
        assert out.has_config is True
        assert out.has_source_change is False
        assert out.meta["dropped_flags"] == ["--bogus"]

    def test_posts_the_request_to_the_predict_path(self, service):
        _predict(service, {"schema": "hyperloom.predictor_request.v1", "session_id": "s1"})
        assert service.last_path == "/v1/predict"
        assert service.last_body["session_id"] == "s1"

    def test_trailing_slash_on_the_endpoint_is_tolerated(self, service):
        cl.predict({}, endpoint=service.endpoint + "/", timeout_sec=5.0)
        assert service.last_path == "/v1/predict"

    def test_source_change_only_is_actionable(self, service):
        service.reply = (
            200,
            {
                "schema": "primatune.predictor_response.v1",
                "parsed": True,
                "action": {"source_change": "Fuse RoPE into the KV write at tuned_gemm.py:395"},
            },
        )
        out = _predict(service)
        assert out.has_source_change is True
        assert out.has_config is False
        assert out.is_empty is False

    def test_empty_action_is_not_actionable(self, service):
        service.reply = (200, {"schema": "primatune.predictor_response.v1", "parsed": True, "action": {}})
        assert _predict(service).is_empty is True


class TestPredictFailure:
    def test_declined_answer_is_not_an_error(self, service):
        service.reply = (200, {"schema": "primatune.predictor_response.v1", "parsed": False})
        out = _predict(service)
        assert out.parsed is False
        assert out.is_empty is True

    def test_server_error_collapses_to_no_action(self, service):
        service.reply = (503, {"error": "busy"})
        out = _predict(service)
        assert out.parsed is False
        assert "503" in out.error

    def test_malformed_body_collapses_to_no_action(self, service):
        service.reply = (200, b"{not json")
        assert _predict(service).parsed is False

    def test_non_object_body_collapses_to_no_action(self, service):
        service.reply = (200, [1, 2, 3])
        assert _predict(service).parsed is False

    def test_foreign_schema_is_refused(self, service):
        service.reply = (200, {"schema": "someone.else.v1", "parsed": True, "action": {"envs": {"A": "1"}}})
        out = _predict(service)
        assert out.parsed is False
        assert "schema" in out.error

    def test_unreachable_endpoint_collapses_to_no_action(self):
        out = cl.predict({}, endpoint="http://127.0.0.1:1", timeout_sec=2.0)
        assert out.parsed is False
        assert out.error

    def test_non_http_scheme_is_refused_before_any_request(self):
        """A file:// endpoint must not become a local read."""
        out = cl.predict({}, endpoint="file:///etc/passwd", timeout_sec=2.0)
        assert out.parsed is False
        assert "unsafe" in out.error

    def test_missing_endpoint_is_refused(self):
        assert cl.predict({}, endpoint="", timeout_sec=2.0).parsed is False

    def test_unserialisable_request_is_refused_before_any_request(self):
        out = cl.predict({"bad": object()}, endpoint="http://127.0.0.1:9", timeout_sec=2.0)
        assert out.parsed is False
        assert "JSON" in out.error
