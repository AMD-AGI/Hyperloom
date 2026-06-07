# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for ``action_executors._robustness_pulse``.

The pulse is a best-effort subprocess shim; the autouse fixture sets
``PYTEST_CURRENT_TEST`` so the production code refuses to spawn a real
subprocess. We exercise the gating and request-shaping helpers, plus
the ``pulse()`` coroutine via dependency injection over ``asyncio``
subprocess primitives.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

from inference_optimizer.orchestrator.action_executors import _robustness_pulse as rp


# ---------------------------------------------------------------------------
# _enabled()
# ---------------------------------------------------------------------------

class TestEnabled:
    def test_disabled_inside_pytest_by_default(self):
        # PYTEST_CURRENT_TEST is always set by pytest, so the gate is False.
        assert rp._enabled() is False

    def test_disabled_when_env_flag_off(self, monkeypatch):
        monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
        monkeypatch.setenv("HYPERLOOM_GRID_ROBUSTNESS_PULSE", "0")
        assert rp._enabled() is False
        for val in ("false", "no", "off", "  "):
            monkeypatch.setenv("HYPERLOOM_GRID_ROBUSTNESS_PULSE", val)
            assert rp._enabled() is False

    def test_enabled_when_env_unset_outside_pytest(self, monkeypatch):
        monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
        monkeypatch.delenv("HYPERLOOM_GRID_ROBUSTNESS_PULSE", raising=False)
        assert rp._enabled() is True


# ---------------------------------------------------------------------------
# _resolve_session_dir()
# ---------------------------------------------------------------------------

class TestResolveSessionDir:
    def test_returns_none_when_no_env(self, monkeypatch):
        monkeypatch.delenv("ROBUSTNESS_AGENT_SESSION_DIR", raising=False)
        monkeypatch.delenv("SESSION_DIR", raising=False)
        assert rp._resolve_session_dir() is None

    def test_prefers_robustness_env_when_dir_exists(self, monkeypatch, tmp_path):
        primary = tmp_path / "robustness_session"
        primary.mkdir()
        monkeypatch.setenv("ROBUSTNESS_AGENT_SESSION_DIR", str(primary))
        monkeypatch.setenv("SESSION_DIR", str(tmp_path / "other"))
        assert rp._resolve_session_dir() == primary

    def test_falls_back_to_session_dir_env(self, monkeypatch, tmp_path):
        monkeypatch.delenv("ROBUSTNESS_AGENT_SESSION_DIR", raising=False)
        backup = tmp_path / "session"
        backup.mkdir()
        monkeypatch.setenv("SESSION_DIR", str(backup))
        assert rp._resolve_session_dir() == backup

    def test_returns_none_when_dir_missing(self, monkeypatch, tmp_path):
        monkeypatch.setenv(
            "ROBUSTNESS_AGENT_SESSION_DIR", str(tmp_path / "ghost"),
        )
        monkeypatch.delenv("SESSION_DIR", raising=False)
        assert rp._resolve_session_dir() is None


# ---------------------------------------------------------------------------
# _build_request()
# ---------------------------------------------------------------------------

class TestBuildRequest:
    def test_request_carries_session_dir_and_disables_llm_rca(self, tmp_path):
        sd = tmp_path / "abc-def"
        sd.mkdir()
        req = rp._build_request(sd, tick_index=7)
        assert req["kind"] == "coordinator_inbox"
        assert req["session_id"] == "abc-def"
        assert "session_id=abc-def" in req["raw_prompt"]
        assert req["context"] == {"tick_index": 7}
        assert req["options"]["session_dir"] == str(sd)
        assert req["options"]["llm_rca_enabled"] is False

    def test_request_falls_back_to_default_session_id(self, tmp_path):
        # Path("") .name == "" so the helper substitutes "default".
        req = rp._build_request(Path(""), tick_index=0)
        assert req["session_id"] == "default"


# ---------------------------------------------------------------------------
# pulse()
# ---------------------------------------------------------------------------

class _FakeProc:
    def __init__(self, *, returncode: int = 0, stderr: bytes = b""):
        self.returncode = returncode
        self._stderr = stderr
        self.killed = False
        self.waited = False

    async def communicate(self):
        return b"", self._stderr

    def kill(self):
        self.killed = True

    async def wait(self):
        self.waited = True


class _SlowProc(_FakeProc):
    async def communicate(self):
        await asyncio.sleep(10)
        return b"", self._stderr


def _spawn_factory(proc):
    async def _spawn(*args, **kwargs):
        # Stash captured args on the proc for assertions.
        proc.captured_args = args
        proc.captured_kwargs = kwargs
        return proc
    return _spawn


@pytest.fixture
def _enable_pulse(monkeypatch):
    """Force the gate to True regardless of pytest's $PYTEST_CURRENT_TEST."""
    monkeypatch.setattr(rp, "_enabled", lambda: True)


class TestPulse:
    @pytest.mark.asyncio
    async def test_disabled_returns_false(self, monkeypatch):
        monkeypatch.setattr(rp, "_enabled", lambda: False)
        assert await rp.pulse(tick_index=0) is False

    @pytest.mark.asyncio
    async def test_returns_false_when_session_dir_missing(
        self, monkeypatch, _enable_pulse,
    ):
        monkeypatch.delenv("ROBUSTNESS_AGENT_SESSION_DIR", raising=False)
        monkeypatch.delenv("SESSION_DIR", raising=False)
        assert await rp.pulse(tick_index=0) is False

    @pytest.mark.asyncio
    async def test_clean_exit_returns_true_and_cleans_tempfile(
        self, monkeypatch, tmp_path, _enable_pulse,
    ):
        sd = tmp_path / "sess"
        sd.mkdir()
        monkeypatch.setenv("ROBUSTNESS_AGENT_SESSION_DIR", str(sd))
        proc = _FakeProc(returncode=0)
        monkeypatch.setattr(
            asyncio, "create_subprocess_exec", _spawn_factory(proc),
        )

        observed: list[str] = []
        orig_unlink = os.unlink

        def tracking_unlink(path):
            observed.append(path)
            return orig_unlink(path)

        monkeypatch.setattr(rp.os, "unlink", tracking_unlink)
        assert await rp.pulse(tick_index=3) is True
        # Request file path is deleted in `finally`.
        assert observed and observed[0].endswith(".json")
        # The subprocess command targets the runtime CLI.
        cmd = proc.captured_args
        assert cmd[1:5] == ("-m", "robustness_agent.runtime.cli", "tick", "--request")

    @pytest.mark.asyncio
    async def test_nonzero_exit_returns_false(
        self, monkeypatch, tmp_path, _enable_pulse,
    ):
        sd = tmp_path / "sess2"
        sd.mkdir()
        monkeypatch.setenv("ROBUSTNESS_AGENT_SESSION_DIR", str(sd))
        proc = _FakeProc(returncode=2, stderr=b"boom")
        monkeypatch.setattr(
            asyncio, "create_subprocess_exec", _spawn_factory(proc),
        )
        assert await rp.pulse(tick_index=1) is False

    @pytest.mark.asyncio
    async def test_subprocess_spawn_failure_returns_false(
        self, monkeypatch, tmp_path, _enable_pulse,
    ):
        sd = tmp_path / "sess3"
        sd.mkdir()
        monkeypatch.setenv("ROBUSTNESS_AGENT_SESSION_DIR", str(sd))

        async def boom(*args, **kwargs):
            raise FileNotFoundError("robustness_agent missing")

        monkeypatch.setattr(asyncio, "create_subprocess_exec", boom)
        assert await rp.pulse(tick_index=0) is False

    @pytest.mark.asyncio
    async def test_tempfile_failure_returns_false(
        self, monkeypatch, tmp_path, _enable_pulse,
    ):
        sd = tmp_path / "sess4"
        sd.mkdir()
        monkeypatch.setenv("ROBUSTNESS_AGENT_SESSION_DIR", str(sd))

        class _BoomTemp:
            def __init__(self, *args, **kwargs):
                raise OSError("no disk space")

        monkeypatch.setattr(rp.tempfile, "NamedTemporaryFile", _BoomTemp)
        assert await rp.pulse(tick_index=0) is False

    @pytest.mark.asyncio
    async def test_timeout_kills_subprocess(
        self, monkeypatch, tmp_path, _enable_pulse,
    ):
        sd = tmp_path / "sess5"
        sd.mkdir()
        monkeypatch.setenv("ROBUSTNESS_AGENT_SESSION_DIR", str(sd))
        slow = _SlowProc(returncode=0)
        monkeypatch.setattr(
            asyncio, "create_subprocess_exec", _spawn_factory(slow),
        )
        result = await rp.pulse(tick_index=0, timeout_s=0.05)
        assert result is False
        assert slow.killed is True
        assert slow.waited is True

    @pytest.mark.asyncio
    async def test_timeout_handles_process_already_gone(
        self, monkeypatch, tmp_path, _enable_pulse,
    ):
        sd = tmp_path / "sess6"
        sd.mkdir()
        monkeypatch.setenv("ROBUSTNESS_AGENT_SESSION_DIR", str(sd))

        class _GoneProc(_SlowProc):
            def kill(self):  # noqa: D401
                raise ProcessLookupError()

        gone = _GoneProc(returncode=0)
        monkeypatch.setattr(
            asyncio, "create_subprocess_exec", _spawn_factory(gone),
        )
        assert await rp.pulse(tick_index=0, timeout_s=0.05) is False
        assert gone.waited is True
