# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Tests for process, GPU, server health, and log-tailer monitors.

The process / GPU monitors are exercised via the shared FakeProvider
fixture from ``conftest.py``. The log tailer and server health
monitors are tested in isolation with patched httpx clients and tmp
files.
"""

from __future__ import annotations

import asyncio
import time

import httpx
import pytest

from robustness_agent.config import Config
from robustness_agent.models import (
    GpuSnapshot,
    ProcessInfo,
    ServerHealthStatus,
    Severity,
)
from robustness_agent.monitors.gpu_monitor import GpuMonitor
from robustness_agent.monitors.log_tailer import LogTailer
from robustness_agent.monitors.process_monitor import ProcessMonitor
from robustness_agent.monitors.server_health import ServerHealthMonitor

from .conftest import FakeProvider


# ---------------------------------------------------------------------------
# ProcessMonitor (integration via FakeProvider)
# ---------------------------------------------------------------------------


class TestProcessMonitor:

    @pytest.fixture
    def monitor(self, config: Config, fake_provider: FakeProvider) -> ProcessMonitor:
        return ProcessMonitor(config, fake_provider)

    @pytest.mark.asyncio
    async def test_no_alerts_when_clean(
        self, monitor: ProcessMonitor, fake_provider: FakeProvider,
    ) -> None:
        fake_provider.processes = [
            ProcessInfo(pid=1, state="S", cmd="python3 -m sglang.srt", rss_mb=1000),
        ]
        alerts = await monitor.check()
        assert len(alerts) == 0

    @pytest.mark.asyncio
    async def test_zombie_server_detected(
        self, monitor: ProcessMonitor, fake_provider: FakeProvider,
    ) -> None:
        fake_provider.processes = [
            ProcessInfo(pid=1, state="Z", cmd="python3 -m sglang.srt", rss_mb=0),
        ]
        alerts = await monitor.check()
        zombie_alerts = [a for a in alerts if a.check_name == "server_zombie"]
        assert len(zombie_alerts) == 1
        assert zombie_alerts[0].severity == Severity.CRITICAL

    @pytest.mark.asyncio
    async def test_server_disappeared(
        self, monitor: ProcessMonitor, fake_provider: FakeProvider,
    ) -> None:
        fake_provider.processes = [
            ProcessInfo(pid=1, state="S", cmd="python3 -m sglang.srt", rss_mb=1000),
        ]
        await monitor.check()

        fake_provider.processes = []
        alerts = await monitor.check()
        disappeared = [a for a in alerts if a.check_name == "server_disappeared"]
        assert len(disappeared) == 1

    @pytest.mark.asyncio
    async def test_benchmark_timeout(
        self, monitor: ProcessMonitor, fake_provider: FakeProvider,
    ) -> None:
        monitor.notify_benchmark_started()
        monitor._benchmark_started_at = time.time() - 600
        fake_provider.processes = [
            ProcessInfo(pid=2, state="S", cmd="benchmark_serving --model foo", rss_mb=500),
        ]
        alerts = await monitor.check()
        timeout_alerts = [a for a in alerts if a.check_name == "benchmark_timeout"]
        assert len(timeout_alerts) == 1


# ---------------------------------------------------------------------------
# GpuMonitor (integration via FakeProvider)
# ---------------------------------------------------------------------------


class TestGpuMonitor:

    @pytest.fixture
    def monitor(self, config: Config, fake_provider: FakeProvider) -> GpuMonitor:
        return GpuMonitor(config, fake_provider)

    @pytest.mark.asyncio
    async def test_vram_critical(
        self, monitor: GpuMonitor, fake_provider: FakeProvider,
    ) -> None:
        fake_provider.gpu_snapshots = [
            GpuSnapshot(
                gpu_id=0, utilization=80,
                vram_used_mb=63000, vram_total_mb=65536,
                temperature_c=70, power_watts=300,
            ),
        ]
        alerts = await monitor.check()
        vram = [a for a in alerts if a.check_name == "gpu_vram_critical"]
        assert len(vram) == 1

    @pytest.mark.asyncio
    async def test_temperature_warning(
        self, monitor: GpuMonitor, fake_provider: FakeProvider,
    ) -> None:
        fake_provider.gpu_snapshots = [
            GpuSnapshot(
                gpu_id=0, utilization=50,
                vram_used_mb=10000, vram_total_mb=65536,
                temperature_c=90, power_watts=300,
            ),
        ]
        alerts = await monitor.check()
        temp = [a for a in alerts if a.check_name == "gpu_temperature_high"]
        assert len(temp) == 1

    @pytest.mark.asyncio
    async def test_ecc_error(
        self, monitor: GpuMonitor, fake_provider: FakeProvider,
    ) -> None:
        fake_provider.gpu_snapshots = [
            GpuSnapshot(
                gpu_id=0, utilization=80,
                vram_used_mb=10000, vram_total_mb=65536,
                temperature_c=70, power_watts=300,
                ecc_errors=5,
            ),
        ]
        alerts = await monitor.check()
        ecc = [a for a in alerts if a.check_name == "gpu_ecc_error"]
        assert len(ecc) == 1
        assert ecc[0].severity == Severity.CRITICAL

    @pytest.mark.asyncio
    async def test_utilization_drop(
        self, monitor: GpuMonitor, fake_provider: FakeProvider,
    ) -> None:
        monitor.set_baseline_utilization(0, 80.0)
        fake_provider.gpu_snapshots = [
            GpuSnapshot(
                gpu_id=0, utilization=20,
                vram_used_mb=10000, vram_total_mb=65536,
                temperature_c=70, power_watts=300,
            ),
        ]
        alerts = await monitor.check()
        drop = [a for a in alerts if a.check_name == "gpu_utilization_drop"]
        assert len(drop) == 1

    @pytest.mark.asyncio
    async def test_healthy_gpu_no_alerts(
        self, monitor: GpuMonitor, fake_provider: FakeProvider,
    ) -> None:
        fake_provider.gpu_snapshots = [
            GpuSnapshot(
                gpu_id=0, utilization=50,
                vram_used_mb=20000, vram_total_mb=65536,
                temperature_c=65, power_watts=250,
            ),
        ]
        alerts = await monitor.check()
        assert len(alerts) == 0


# ---------------------------------------------------------------------------
# LogTailer (unit-level)
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
        with log_path.open("a") as fh:
            fh.write("CUDA out of memory line b\n")
        second = await tailer.check()
        assert second == []

    @pytest.mark.asyncio
    async def test_set_log_path_resets_position(self, tmp_path):
        original = tmp_path / "a.log"
        original.write_text("nothing interesting\n")
        tailer = LogTailer(log_path=original)
        await tailer.check()
        assert tailer._file_pos > 0
        new_path = tmp_path / "b.log"
        new_path.write_text("HIP out of memory at end\n")
        tailer.set_log_path(new_path)
        assert tailer._file_pos == 0
        alerts = await tailer.check()
        assert any(alert.check_name == "log_error_oom" for alert in alerts)

    @pytest.mark.asyncio
    async def test_read_truncates_to_max_lines(self, tmp_path, monkeypatch):
        log_path = tmp_path / "lots.log"
        log_path.write_text("noise\n" * 50 + "Segmentation fault here\n")
        tailer = LogTailer(log_path=log_path, max_lines_per_check=2)
        alerts = await tailer.check()
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
        assert await tailer.check() == []


# ---------------------------------------------------------------------------
# ServerHealthMonitor (unit-level)
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
        mon._consecutive_failures = 2
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
