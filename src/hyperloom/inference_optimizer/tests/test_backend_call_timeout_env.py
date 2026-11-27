# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Per-backend ``call_timeout_s`` env-var override tests."""

from __future__ import annotations

import math

import pytest

from hyperloom.orchestrator.roles.base import parse_call_timeout_env
from hyperloom.orchestrator.roles.claude import ClaudeBackend


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
        """Both ``+inf`` and ``-inf`` fail the positive-finite guard."""
        monkeypatch.setenv(PROBE_ENV, raw)
        with caplog.at_level("WARNING"):
            result = parse_call_timeout_env(PROBE_ENV, default=120.0)
        assert result == 120.0
        assert math.isfinite(result)
        assert any("not a positive finite" in rec.message for rec in caplog.records)


@pytest.mark.usefixtures("claude_sdk")
class TestClaudeBackendHonoursEnv:
    def test_default_when_env_unset(self, monkeypatch):
        monkeypatch.delenv(CLAUDE_ENV, raising=False)
        backend = ClaudeBackend()
        assert backend.call_timeout_s == 120.0

    def test_env_override_picked_up_at_instantiation(self, monkeypatch):
        monkeypatch.setenv(CLAUDE_ENV, "300")
        backend = ClaudeBackend()
        assert backend.call_timeout_s == 300.0

    def test_bad_env_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv(CLAUDE_ENV, "garbage")
        backend = ClaudeBackend()
        assert backend.call_timeout_s == 120.0


class TestCodexBackendHonoursEnv:
    def test_default_when_env_unset(self, monkeypatch, tmp_path):
        monkeypatch.delenv(CODEX_ENV, raising=False)
        assert _new_codex_backend(tmp_path).call_timeout_s == 300.0

    def test_env_override_picked_up_at_instantiation(self, monkeypatch, tmp_path):
        monkeypatch.setenv(CODEX_ENV, "240")
        assert _new_codex_backend(tmp_path).call_timeout_s == 240.0

    def test_bad_env_falls_back_to_default(self, monkeypatch, tmp_path):
        monkeypatch.setenv(CODEX_ENV, "negative-one")
        assert _new_codex_backend(tmp_path).call_timeout_s == 300.0


def _new_codex_backend(tmp_path):
    """Build a CodexBackend against a throwaway session-private cwd."""
    from hyperloom.orchestrator.roles.codex import CodexBackend

    return CodexBackend(cwd=tmp_path / "codex")


@pytest.fixture(autouse=True)
def _clear_probe_env(monkeypatch):
    """Ensure tests never leak the probe env to each other."""
    for key in (CLAUDE_ENV, CODEX_ENV, PROBE_ENV):
        monkeypatch.delenv(key, raising=False)
    yield
