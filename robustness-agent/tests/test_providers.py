"""Tests for metrics providers."""

from __future__ import annotations

import pytest

from robustness_agent.config import Config
from robustness_agent.providers.hybrid import HybridProvider, create_provider
from robustness_agent.providers.local import LocalProvider, _RingBuffer
from robustness_agent.models import GpuSnapshot


class TestRingBuffer:

    def test_push_and_get(self) -> None:
        buf = _RingBuffer(max_age_seconds=60)
        snap = GpuSnapshot(
            gpu_id=0, utilization=80, vram_used_mb=1000,
            vram_total_mb=2000, temperature_c=70, power_watts=300,
            timestamp=100.0,
        )
        buf.push(snap)
        result = buf.get(0, 120)
        assert len(result) == 1
        assert result[0].utilization == 80

    def test_expiry(self) -> None:
        buf = _RingBuffer(max_age_seconds=10)
        old = GpuSnapshot(
            gpu_id=0, utilization=50, vram_used_mb=500,
            vram_total_mb=1000, temperature_c=60, power_watts=200,
            timestamp=1.0,
        )
        new = GpuSnapshot(
            gpu_id=0, utilization=90, vram_used_mb=900,
            vram_total_mb=1000, temperature_c=80, power_watts=350,
            timestamp=100.0,
        )
        buf.push(old)
        buf.push(new)
        result = buf.get(0, 20)
        assert len(result) == 1
        assert result[0].utilization == 90

    def test_empty_gpu(self) -> None:
        buf = _RingBuffer()
        assert buf.get(99, 60) == []


class TestCreateProvider:

    @pytest.mark.asyncio
    async def test_no_url_returns_local(self, tmp_path) -> None:
        cfg = Config(session_dir=tmp_path)
        provider = await create_provider(cfg)
        assert isinstance(provider, LocalProvider)

    @pytest.mark.asyncio
    async def test_with_url_returns_hybrid(self, tmp_path) -> None:
        cfg = Config(session_dir=tmp_path, robust_analyzer_url="http://fake:8085")
        provider = await create_provider(cfg)
        assert isinstance(provider, HybridProvider)
