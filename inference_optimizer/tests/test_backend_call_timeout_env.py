"""Per-backend ``call_timeout_s`` env-var override tests.

``parse_call_timeout_env`` (orchestrator/backends/base.py) reads
``INFERENCE_OPTIMIZER_*_CALL_TIMEOUT_SEC`` so operators can bump the
asyncio wall-clock cap on a ``ClaudeBackend`` / ``CodexBackend`` call
without code changes. The helper returns ``default`` for unset / empty
/ malformed / non-positive / non-finite values so a typo in the env can
never crash backend boot.

These tests pin both contracts:

1. Helper-level: unset/empty/garbage/negative/nan -> default; happy path
   returns the parsed float verbatim.
2. Backend-level: ``ClaudeBackend()`` and ``CodexBackend()`` instantiated
   under a chosen env value pick up that value as ``call_timeout_s``,
   and instantiated without the env fall back to 120.0.

Backend tests use ``dataclasses.fields(...).default_factory`` -- the
``call_timeout_s`` value is read at instantiation, not at import, so
``monkeypatch.setenv`` after the module is already imported still wins.
"""

from __future__ import annotations

import math

import pytest

from inference_optimizer.orchestrator.backends.base import parse_call_timeout_env


CLAUDE_ENV = "INFERENCE_OPTIMIZER_CLAUDE_CALL_TIMEOUT_SEC"
CODEX_ENV = "INFERENCE_OPTIMIZER_CODEX_CALL_TIMEOUT_SEC"
PROBE_ENV = "INFERENCE_OPTIMIZER_TIMEOUT_ENV_PROBE_SEC"


class TestParseCallTimeoutEnv:
    def test_unset_returns_default(self, monkeypatch):
        monkeypatch.delenv(PROBE_ENV, raising=False)
        assert parse_call_timeout_env(PROBE_ENV, default=120.0) == 120.0

    def test_empty_returns_default(self, monkeypatch):
        monkeypatch.setenv(PROBE_ENV, "")
        assert parse_call_timeout_env(PROBE_ENV, default=77.0) == 77.0

    def test_whitespace_returns_default(self, monkeypatch):
        monkeypatch.setenv(PROBE_ENV, "   ")
        assert parse_call_timeout_env(PROBE_ENV, default=42.0) == 42.0

    def test_happy_path_float(self, monkeypatch):
        monkeypatch.setenv(PROBE_ENV, "240")
        assert parse_call_timeout_env(PROBE_ENV, default=120.0) == 240.0

    def test_happy_path_fractional(self, monkeypatch):
        monkeypatch.setenv(PROBE_ENV, "12.5")
        assert parse_call_timeout_env(PROBE_ENV, default=120.0) == 12.5

    def test_garbage_returns_default(self, monkeypatch, caplog):
        monkeypatch.setenv(PROBE_ENV, "not-a-float")
        with caplog.at_level("WARNING"):
            assert parse_call_timeout_env(PROBE_ENV, default=120.0) == 120.0
        assert any("not a float" in rec.message for rec in caplog.records)

    def test_zero_returns_default(self, monkeypatch, caplog):
        monkeypatch.setenv(PROBE_ENV, "0")
        with caplog.at_level("WARNING"):
            assert parse_call_timeout_env(PROBE_ENV, default=120.0) == 120.0

    def test_negative_returns_default(self, monkeypatch, caplog):
        monkeypatch.setenv(PROBE_ENV, "-30")
        with caplog.at_level("WARNING"):
            assert parse_call_timeout_env(PROBE_ENV, default=120.0) == 120.0
        assert any("not a positive finite" in rec.message for rec in caplog.records)

    def test_nan_returns_default(self, monkeypatch, caplog):
        monkeypatch.setenv(PROBE_ENV, "nan")
        with caplog.at_level("WARNING"):
            result = parse_call_timeout_env(PROBE_ENV, default=120.0)
        assert result == 120.0
        assert not math.isnan(result)

    @pytest.mark.parametrize("raw", ["inf", "-inf", "infinity"])
    def test_infinity_returns_default(self, monkeypatch, caplog, raw):
        """``math.isfinite`` rejects both ``+inf`` and ``-inf`` -- a wall-clock
        timeout can never be infinite. This pins the CodeQL fix that swapped
        the NaN-only ``value != value`` check for ``not math.isfinite(value)``.
        """
        monkeypatch.setenv(PROBE_ENV, raw)
        with caplog.at_level("WARNING"):
            result = parse_call_timeout_env(PROBE_ENV, default=120.0)
        assert result == 120.0
        assert math.isfinite(result)
        assert any("not a positive finite" in rec.message for rec in caplog.records)


class TestClaudeBackendHonoursEnv:
    def test_default_when_env_unset(self, monkeypatch):
        monkeypatch.delenv(CLAUDE_ENV, raising=False)
        backend = _new_claude_backend_with_seams()
        assert backend.call_timeout_s == 120.0

    def test_env_override_picked_up_at_instantiation(self, monkeypatch):
        monkeypatch.setenv(CLAUDE_ENV, "300")
        backend = _new_claude_backend_with_seams()
        assert backend.call_timeout_s == 300.0

    def test_bad_env_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv(CLAUDE_ENV, "garbage")
        backend = _new_claude_backend_with_seams()
        assert backend.call_timeout_s == 120.0


class TestCodexBackendHonoursEnv:
    def test_default_when_env_unset(self, monkeypatch):
        monkeypatch.delenv(CODEX_ENV, raising=False)
        from inference_optimizer.orchestrator.backends.codex import CodexBackend

        backend = CodexBackend(
            api_key_env="OPENAI_API_KEY_TEST",
            base_url_env="OPENAI_BASE_URL_TEST",
            client_factory=object,
        )
        assert backend.call_timeout_s == 120.0

    def test_env_override_picked_up_at_instantiation(self, monkeypatch):
        monkeypatch.setenv(CODEX_ENV, "240")
        from inference_optimizer.orchestrator.backends.codex import CodexBackend

        backend = CodexBackend(
            api_key_env="OPENAI_API_KEY_TEST",
            base_url_env="OPENAI_BASE_URL_TEST",
            client_factory=object,
        )
        assert backend.call_timeout_s == 240.0

    def test_bad_env_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv(CODEX_ENV, "negative-one")
        from inference_optimizer.orchestrator.backends.codex import CodexBackend

        backend = CodexBackend(
            api_key_env="OPENAI_API_KEY_TEST",
            base_url_env="OPENAI_BASE_URL_TEST",
            client_factory=object,
        )
        assert backend.call_timeout_s == 120.0


def _new_claude_backend_with_seams():
    """Build a ClaudeBackend that skips real SDK import + MCP server.

    ``__post_init__`` accepts ``enable_mcp_emit_intent=False`` and a
    minimal pair of seams to skip both the real SDK import and the MCP
    server build (mirroring ``test_claude_backend.py``'s pattern).
    """
    from inference_optimizer.orchestrator.backends.claude import ClaudeBackend

    return ClaudeBackend(
        sdk_query_factory=lambda *a, **kw: iter(()),
        sdk_options_cls=type("FakeOpts", (), {"__init__": lambda self, **kw: None}),
        enable_mcp_emit_intent=False,
    )


@pytest.fixture(autouse=True)
def _clear_probe_env(monkeypatch):
    """Ensure tests never leak the probe env to each other."""
    for key in (CLAUDE_ENV, CODEX_ENV, PROBE_ENV):
        monkeypatch.delenv(key, raising=False)
    yield
