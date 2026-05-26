"""Unit tests for the legacy log-tailer and server-health monitors.

Both classes were thin enough that the M1 reactor still uses them
indirectly via the integration tests, but their error/edge branches are
not otherwise exercised. This module fills in the obvious gaps so the
monitor logic stays trustworthy if/when the legacy daemon is revived.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from robustness_agent.models import Alert, ServerHealthStatus, Severity
from robustness_agent.monitors.log_tailer import LogTailer
from robustness_agent.monitors.server_health import ServerHealthMonitor


# ---------------------------------------------------------------------------
# LogTailer
# ---------------------------------------------------------------------------

class TestLogTailer:
    @pytest.mark.asyncio
    async def test_check_returns_empty_when_no_path(self):
        tailer = LogTailer()
        assert await tailer.check() == []

    @pytest.mark.asyncio
    async def test_check_returns_empty_when_path_missing(self, tmp_path):
        tailer = LogTailer(log_path=tmp_path / "ghost.log")
        assert await tailer.check() == []

    @pytest.mark.asyncio
    async def test_oom_pattern_emits_critical_alert(self, tmp_path):
        log_path = tmp_path / "server.log"
        log_path.write_text(
            "starting up...\nCUDA out of memory at line 12\n",
        )
        tailer = LogTailer(log_path=log_path)
        alerts = await tailer.check()
        assert len(alerts) == 1
        a = alerts[0]
        assert a.check_name == "log_error_oom"
        assert a.severity == Severity.CRITICAL
        assert "out of memory" in a.detail.lower()

    @pytest.mark.asyncio
    async def test_dedup_window_suppresses_repeats(self, tmp_path):
        log_path = tmp_path / "server.log"
        log_path.write_text("CUDA out of memory line a\n")
        tailer = LogTailer(log_path=log_path)
        first = await tailer.check()
        assert len(first) == 1
        # Append a fresh OOM line, but the dedup window should still apply.
        with log_path.open("a") as fh:
            fh.write("CUDA out of memory line b\n")
        second = await tailer.check()
        assert second == []

    @pytest.mark.asyncio
    async def test_set_log_path_resets_position(self, tmp_path):
        original = tmp_path / "a.log"
        original.write_text("nothing interesting\n")
        tailer = LogTailer(log_path=original)
        await tailer.check()  # advance file_pos
        assert tailer._file_pos > 0
        new_path = tmp_path / "b.log"
        new_path.write_text("HIP out of memory at end\n")
        tailer.set_log_path(new_path)
        assert tailer._file_pos == 0
        alerts = await tailer.check()
        assert any(alert.check_name == "log_error_oom" for alert in alerts)

    @pytest.mark.asyncio
    async def test_read_truncates_to_max_lines(self, tmp_path, monkeypatch):
        # Force max_lines low so we exercise the truncation branch in
        # ``_read_new_lines``.
        log_path = tmp_path / "lots.log"
        log_path.write_text("noise\n" * 50 + "Segmentation fault here\n")
        tailer = LogTailer(log_path=log_path, max_lines_per_check=2)
        alerts = await tailer.check()
        # Even truncated, the last line with the fault must be in the
        # window we inspect.
        assert any(a.check_name == "log_error_segfault" for a in alerts)

    @pytest.mark.asyncio
    async def test_handles_read_failure_gracefully(
        self, tmp_path, monkeypatch,
    ):
        log_path = tmp_path / "x.log"
        log_path.write_text("baseline\n")
        tailer = LogTailer(log_path=log_path)

        async def boom(*args, **kwargs):
            raise OSError("disk vanished")

        monkeypatch.setattr(tailer, "_read_new_lines", boom)
        assert await tailer.check() == []  # swallowed


# ---------------------------------------------------------------------------
# ServerHealthMonitor
# ---------------------------------------------------------------------------

class _FakeResponse:
    def __init__(self, status_code: int = 200):
        self.status_code = status_code


class _FakeClient:
    def __init__(self, *, response=None, exc: Exception | None = None,
                 latency_s: float = 0.0):
        self._response = response
        self._exc = exc
        self._latency = latency_s

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def get(self, url, timeout=None):
        if self._latency:
            await asyncio.sleep(self._latency)
        if self._exc is not None:
            raise self._exc
        return self._response


def _patch_async_client(monkeypatch, **kwargs):
    monkeypatch.setattr(
        "robustness_agent.monitors.server_health.httpx.AsyncClient",
        lambda *a, **k: _FakeClient(**kwargs),
    )


class TestServerHealthMonitor:
    @pytest.mark.asyncio
    async def test_check_returns_noop_when_url_blank(self):
        mon = ServerHealthMonitor()
        status, alerts = await mon.check()
        assert status.reachable is False
        assert status.error == "no server URL configured"
        assert alerts == []

    @pytest.mark.asyncio
    async def test_healthy_response_clears_failure_streak(self, monkeypatch):
        _patch_async_client(monkeypatch, response=_FakeResponse(200))
        mon = ServerHealthMonitor(server_url="http://server:8000")
        mon._consecutive_failures = 2  # pretend prior outage
        status, alerts = await mon.check()
        assert status.reachable is True
        assert mon._consecutive_failures == 0
        assert alerts == []

    @pytest.mark.asyncio
    async def test_non_200_emits_warning_then_critical(self, monkeypatch):
        _patch_async_client(monkeypatch, response=_FakeResponse(500))
        mon = ServerHealthMonitor(server_url="http://server:8000")
        for streak in (1, 2):
            _, alerts = await mon.check()
            assert alerts[0].severity == Severity.WARNING
            assert mon._consecutive_failures == streak
        _, alerts = await mon.check()
        assert mon._consecutive_failures == 3
        assert alerts[0].severity == Severity.CRITICAL

    @pytest.mark.asyncio
    async def test_timeout_classified(self, monkeypatch):
        _patch_async_client(monkeypatch, exc=httpx.TimeoutException("slow"))
        mon = ServerHealthMonitor(server_url="http://server:8000")
        status, alerts = await mon.check()
        assert status.reachable is False
        assert status.error == "timeout"
        assert alerts and alerts[0].check_name == "server_health_fail"

    @pytest.mark.asyncio
    async def test_connect_error_classified(self, monkeypatch):
        _patch_async_client(monkeypatch, exc=httpx.ConnectError("ECONNREFUSED"))
        mon = ServerHealthMonitor(server_url="http://server:8000")
        status, _ = await mon.check()
        assert status.error.startswith("connect error")

    @pytest.mark.asyncio
    async def test_unknown_exception_recorded_as_message(self, monkeypatch):
        _patch_async_client(monkeypatch, exc=RuntimeError("kaboom"))
        mon = ServerHealthMonitor(server_url="http://server:8000")
        status, _ = await mon.check()
        assert "kaboom" in status.error

    @pytest.mark.asyncio
    async def test_slow_response_emits_extra_warning(self, monkeypatch):
        # Drive the implementation's slow-probe branch by stubbing
        # ``_probe`` to return a healthy-but-slow status. The slow-path
        # alert lives outside ``_probe`` so the monitor still appends it.
        async def fake_probe(self):
            return ServerHealthStatus(
                url=self._server_url + "/health",
                reachable=True,
                response_time_ms=6000.0,
                status_code=200,
                error="",
            )

        monkeypatch.setattr(ServerHealthMonitor, "_probe", fake_probe)
        mon = ServerHealthMonitor(server_url="http://server:8000")
        status, alerts = await mon.check()
        assert status.reachable is True
        assert any(a.check_name == "server_slow_response" for a in alerts)

    def test_set_server_url_resets_streak(self):
        mon = ServerHealthMonitor(server_url="http://a")
        mon._consecutive_failures = 4
        mon.set_server_url("http://b")
        assert mon._consecutive_failures == 0
        assert mon._server_url == "http://b"
