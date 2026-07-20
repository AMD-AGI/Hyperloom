# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the lifecycle -> driver wiring (S4).

Verifies that _maybe_pump_targeted_build writes plan.json and spawns the driver
argv for real components (empty build_command), while preserving the fake/explicit
build_command path from S2.  One opt-in e2e marker is registered for real ROCm.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from hyperloom.orchestrator.bus.resource_lock import ResourceLockManager, SqliteLeaseBackend
from hyperloom.orchestrator.bus.storage import SqliteConnection
from hyperloom.orchestrator.bus.storage.schema import ensure_schema
from hyperloom.orchestrator.framework.build_actions import TargetedBuildAction
from hyperloom.orchestrator.loop.build_lifecycle import (
    BuildLifecycleCollaborator,
    _driver_command,
)
from hyperloom.orchestrator.state.shared_state import SharedState
from hyperloom.orchestrator.state.task_registry import TaskRegistry


class _FakeCoord:
    def __init__(self, session_dir, db):
        self.session_dir = session_dir
        self.tasks = TaskRegistry(db)
        self.locks = ResourceLockManager(SqliteLeaseBackend(db))
        self.shared_state = SharedState()


@pytest.fixture
def coord(tmp_path):
    db = SqliteConnection(tmp_path / "coordinator.db")
    ensure_schema(db.raw)
    fc = _FakeCoord(tmp_path, db)
    yield fc
    db.close()


@pytest.fixture
def bl(coord):
    return BuildLifecycleCollaborator(coord)


def _real_action(**kw):
    base = dict(gap_id="g", framework="vllm", component="aiter", capability="fp4_moe",
                ref="v0.1.0", repo_url="https://github.com/ROCm/aiter", gpu_arch="gfx950")
    base.update(kw)
    return TargetedBuildAction(**base)


def _fake_action(**kw):
    """Action with an explicit build_command (S2 fake-builder path)."""
    import sys as _sys
    base = dict(gap_id="g", framework="vllm", component="aiter", capability="fp4_moe",
                build_command=(_sys.executable, "-c", "print('fake')"))
    base.update(kw)
    return TargetedBuildAction(**base)


# ---------------------------------------------------------------------------
# _driver_command
# ---------------------------------------------------------------------------

def test_driver_command_real_component_uses_driver_module(tmp_path):
    action = _real_action()
    attempt_root = str(tmp_path / "attempt")
    cmd = _driver_command(action, attempt_root)
    assert cmd[0] == sys.executable
    assert "targeted_build" in " ".join(cmd)
    assert "--attempt-root" in cmd
    # plan.json must be written
    plan = (tmp_path / "attempt" / "plan.json")
    assert plan.exists()
    data = json.loads(plan.read_text())
    assert data["component"] == "aiter"


def test_driver_command_explicit_build_command_passthrough(tmp_path):
    action = _fake_action()
    attempt_root = str(tmp_path / "attempt2")
    cmd = _driver_command(action, attempt_root)
    assert cmd == list(action.build_command)
    # No plan.json written for fake path
    assert not (tmp_path / "attempt2" / "plan.json").exists()


# ---------------------------------------------------------------------------
# pump: real component spawns driver argv
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_pump_real_component_writes_plan_json(coord, bl, tmp_path):
    """Pump for an aiter action with empty build_command writes plan.json."""
    action = _real_action()
    tid = await bl.enqueue_targeted_build(action)

    spawned_commands: list[list[str]] = []

    def _capture_spawn(action_inner, *, attempt_root, command, run=None, **kw):
        spawned_commands.append(list(command) if command else [])
        plan = Path(attempt_root) / "plan.json"
        assert plan.exists(), "plan.json must exist before spawn_build is called"
        assert json.loads(plan.read_text())["component"] == "aiter"
        raise RuntimeError("captured")  # abort spawn so we don't actually exec

    import hyperloom.orchestrator.loop.build_lifecycle as blc_mod
    original = blc_mod.spawn_build
    blc_mod.spawn_build = _capture_spawn
    try:
        await bl._maybe_pump_targeted_build(tick=0)
    except RuntimeError:
        pass
    finally:
        blc_mod.spawn_build = original

    assert spawned_commands
    cmd = spawned_commands[0]
    assert sys.executable in cmd
    assert "targeted_build" in " ".join(cmd)
    assert "--attempt-root" in cmd


@pytest.mark.asyncio
async def test_pump_fake_component_runs_literal_command(coord, bl):
    """Pump for an action with explicit build_command uses that command directly."""
    action = _fake_action()
    await bl.enqueue_targeted_build(action)

    spawned_commands: list[list[str]] = []

    def _capture_spawn(action_inner, *, attempt_root, command, run=None, **kw):
        spawned_commands.append(list(command) if command else [])
        raise RuntimeError("captured")

    import hyperloom.orchestrator.loop.build_lifecycle as blc_mod
    original = blc_mod.spawn_build
    blc_mod.spawn_build = _capture_spawn
    try:
        await bl._maybe_pump_targeted_build(tick=0)
    except RuntimeError:
        pass
    finally:
        blc_mod.spawn_build = original

    assert spawned_commands
    assert spawned_commands[0] == list(action.build_command)


# ---------------------------------------------------------------------------
# e2e marker: skips off a real ROCm host
# ---------------------------------------------------------------------------

@pytest.mark.targeted_build_e2e
def test_aiter_build_e2e_real_rocm(tmp_path):
    """Opt-in real-compile test.  Skipped unless a ROCm host is present.

    This test is excluded from CI via ``-m 'not targeted_build_e2e'``.
    Enable with ``pytest -m targeted_build_e2e``.
    """
    try:
        import torch

        if not getattr(torch.version, "hip", None):
            pytest.skip("not a ROCm torch — skipping real AITER compile")
    except ImportError:
        pytest.skip("torch not importable — skipping real AITER compile")

    from hyperloom.orchestrator.framework.targeted_build import run_aiter_build

    action = TargetedBuildAction(
        gap_id="e2e",
        framework="vllm",
        component="aiter",
        capability="fp4_moe",
        ref="",
        repo_url="https://github.com/ROCm/aiter",
        gpu_arch="gfx950",
        max_jobs=8,
    )
    result = run_aiter_build(action, str(tmp_path / "e2e_attempt"))
    assert result.ok, f"e2e AITER build failed: {result.failure_class} - {result.failure_summary}"
    assert result.installed_versions.get("aiter_ref")
