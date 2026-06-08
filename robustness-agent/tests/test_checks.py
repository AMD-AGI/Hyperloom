# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Tests for disk, stall, and event checks."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pytest

from robustness_agent.checks.disk_check import DiskCheck
from robustness_agent.checks.event_check import EventCheck
from robustness_agent.checks.stall_check import StallCheck
from robustness_agent.conductor import ConductorReader
from robustness_agent.config import Config
from robustness_agent.models import ConductorEvent, DiskSnapshot

from .conftest import FakeProvider


class TestDiskCheck:

    @pytest.mark.asyncio
    async def test_disk_critical(self, config: Config, fake_provider: FakeProvider) -> None:
        fake_provider.disks = [
            DiskSnapshot(mount="/", total_gb=100, used_gb=97, available_gb=3),
        ]
        check = DiskCheck(config, fake_provider)
        alerts = await check.check()
        assert any(a.check_name == "disk_critical" for a in alerts)

    @pytest.mark.asyncio
    async def test_disk_ok(self, config: Config, fake_provider: FakeProvider) -> None:
        fake_provider.disks = [
            DiskSnapshot(mount="/", total_gb=100, used_gb=50, available_gb=50),
        ]
        check = DiskCheck(config, fake_provider)
        alerts = await check.check()
        assert len(alerts) == 0


class TestStallCheck:

    @pytest.mark.asyncio
    async def test_agent_stall_detected(
        self, config: Config, conductor_db: Path,
    ) -> None:
        conn = sqlite3.connect(str(conductor_db))
        conn.execute(
            "INSERT INTO events (agent, intent_type, payload, timestamp, topic) VALUES (?, ?, ?, ?, ?)",
            ("orchestration", "send_message", "{}", time.time() - 600, "heartbeat"),
        )
        conn.commit()
        conn.close()

        reader = ConductorReader(conductor_db)
        reader.connect()
        check = StallCheck(config, reader)
        alerts = await check.check()
        stall = [a for a in alerts if a.check_name == "agent_stall"]
        assert len(stall) == 1
        assert stall[0].evidence["agent"] == "orchestration"
        reader.close()

    @pytest.mark.asyncio
    async def test_no_stall_when_active(
        self, config: Config, conductor_db: Path,
    ) -> None:
        conn = sqlite3.connect(str(conductor_db))
        conn.execute(
            "INSERT INTO events (agent, intent_type, payload, timestamp, topic) VALUES (?, ?, ?, ?, ?)",
            ("orchestration", "send_message", "{}", time.time(), "heartbeat"),
        )
        conn.commit()
        conn.close()

        reader = ConductorReader(conductor_db)
        reader.connect()
        check = StallCheck(config, reader)
        alerts = await check.check()
        assert len(alerts) == 0
        reader.close()


class TestEventCheck:

    def test_keep_revert_bouncing(self, config: Config) -> None:
        check = EventCheck(config)
        now = time.time()
        events = [
            ConductorEvent(i, "orchestration", "update_state",
                           {"decision": d, "action_name": "kernel-opt"},
                           now - (10 - i), "state")
            for i, d in enumerate(["keep", "revert", "keep", "revert", "keep"])
        ]
        alerts = check.process_events(events)
        bouncing = [a for a in alerts if a.check_name == "keep_revert_bouncing"]
        assert len(bouncing) >= 1

    def test_family_failure_tracking(self, config: Config) -> None:
        check = EventCheck(config)
        now = time.time()
        events = [
            ConductorEvent(i, "orchestration", "delegate",
                           {"status": "failed", "family": "deep_kernel"},
                           now - i, "task")
            for i in range(3)
        ]
        alerts = check.process_events(events)
        family_fail = [a for a in alerts if a.check_name == "family_repeated_failure"]
        assert len(family_fail) == 1
        assert family_fail[0].evidence["family"] == "deep_kernel"
