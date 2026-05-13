"""Tests for process, GPU, and server health monitors."""

from __future__ import annotations

import time

import pytest

from robustness_agent.config import Config
from robustness_agent.models import GpuSnapshot, ProcessInfo, Severity
from robustness_agent.monitors.gpu_monitor import GpuMonitor
from robustness_agent.monitors.process_monitor import ProcessMonitor

from .conftest import FakeProvider


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
