# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for :mod:`hyperloom.inference_optimizer.cli.executors`."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from hyperloom.inference_optimizer.cli import executors as cli_executors
from hyperloom.inference_optimizer.protocol.action_surfaces import KERNEL_AGENT_OWNED_ACTIONS
from hyperloom.inference_optimizer.cli.executors import (
    _build_specialist_executor,
    _register_executors,
    _REAL_EXECUTORS_FULL,
)


def test_recover_executor_is_not_registered() -> None:
    assert "recover" not in _REAL_EXECUTORS_FULL


def _spec_args() -> argparse.Namespace:
    return argparse.Namespace(
        claude_model="claude-opus-4-6",
        specialist_model=None,
        specialist_max_turns=3,
        specialist_mcp_config=None,
    )


def test_build_specialist_executor_refuses_to_start_without_a_claude_cli(monkeypatch, tmp_path):
    """Specialists only run as CLI subprocesses, so a missing CLI stops the run at startup."""
    import shutil

    monkeypatch.setattr(shutil, "which", lambda _n: "")
    monkeypatch.delenv("GEAK_CLAUDE_BIN", raising=False)
    monkeypatch.delenv("HYPERLOOM_CLAUDE_CLI_PATH", raising=False)
    with pytest.raises(RuntimeError, match="none was found in"):
        _build_specialist_executor(
            _spec_args(),
            session_dir=tmp_path,
            knowledge_plane=None,
        )


def test_build_specialist_executor_subprocess_uses_the_recorded_claude(monkeypatch, tmp_path, caplog):
    """A CLI recorded in GEAK_CLAUDE_BIN keeps subprocess dispatch when ``claude`` is not on PATH."""
    import shutil

    recorded = tmp_path / "claude"
    recorded.write_text("#!/bin/sh\n", encoding="utf-8")
    recorded.chmod(0o755)
    monkeypatch.setattr(shutil, "which", lambda _n: "")
    monkeypatch.delenv("HYPERLOOM_CLAUDE_CLI_PATH", raising=False)
    monkeypatch.setenv("GEAK_CLAUDE_BIN", str(recorded))
    with caplog.at_level(logging.WARNING, logger=cli_executors.log.name):
        executor = _build_specialist_executor(
            _spec_args(),
            session_dir=tmp_path,
            knowledge_plane=None,
        )
    assert callable(executor)
    assert not any("claude" in rec.message for rec in caplog.records)


def test_build_specialist_executor_subprocess_with_knowledge_plane(monkeypatch, tmp_path):
    """subprocess path with a KnowledgePlane generates an MCP config."""
    import shutil

    monkeypatch.setattr(shutil, "which", lambda _n: "/usr/bin/claude")

    class _KP:
        def specialist_mcp_url(self) -> str:
            return "http://pr-monitor.invalid/mcp"

    executor = _build_specialist_executor(
        _spec_args(),
        session_dir=tmp_path,
        knowledge_plane=_KP(),
    )
    assert callable(executor)


def test_build_specialist_executor_subprocess_kp_missing_methods(monkeypatch, tmp_path):
    """subprocess path tolerates a KnowledgePlane lacking the MCP-url methods."""
    import shutil

    monkeypatch.setattr(shutil, "which", lambda _n: "/usr/bin/claude")

    executor = _build_specialist_executor(
        _spec_args(),
        session_dir=tmp_path,
        knowledge_plane=object(),
    )
    assert callable(executor)


class _FakeSub:
    """Minimal stand-in for Coordinator.sub recording registered executors."""

    def __init__(self) -> None:
        self.executor_registry: dict[str, object] = {}

    def register_executor(self, kind: str, fn: object) -> None:
        self.executor_registry[kind] = fn


def _fake_coordinator() -> SimpleNamespace:
    return SimpleNamespace(sub=_FakeSub(), shared_state=SimpleNamespace())


def test_register_executors_wires_full_set():
    coord = _fake_coordinator()
    _register_executors(coord, session_dir=None)
    reg = coord.sub.executor_registry
    for kind in _REAL_EXECUTORS_FULL:
        assert kind in reg
    assert "target_analysis" in reg
    # Both arms land their diffs through this one kind; a missing key silently drops every discovered PR candidate and
    # every authored patch.
    assert "integrate_patch" in reg
    assert "framework_agent" not in reg
    assert "framework" not in reg
    assert "roofline" in reg


def test_register_executors_covers_every_phase_allowed_action():
    """Every action a phase may enqueue resolves to a registered executor."""
    from hyperloom.orchestrator.phases.machine_state import PHASE_ALLOWED_ACTIONS

    coord = _fake_coordinator()

    async def _spec(ctx):
        return {}

    _register_executors(coord, session_dir=None, specialist_executor=_spec)
    reg = coord.sub.executor_registry

    expected: set[str] = set()
    for actions in PHASE_ALLOWED_ACTIONS.values():
        expected |= set(actions)
    # Kernel-owned actions never become tasks: PolicyGate denies delegate / propose_action for them, and the
    # Coordinator routes them over the bus.
    expected -= KERNEL_AGENT_OWNED_ACTIONS
    missing = sorted(kind for kind in expected if kind not in reg)
    assert not missing, f"phase-allowed actions with no executor: {missing}"


def test_register_executors_never_wires_kernel_owned_actions(caplog):
    """Kernel-owned actions are REQUEST-only, so they get no executor at all."""
    coord = _fake_coordinator()
    with caplog.at_level(logging.DEBUG, logger=cli_executors.log.name):
        _register_executors(coord, session_dir=None)
    reg = coord.sub.executor_registry
    assert "roofline" in reg
    assert not (set(reg) & KERNEL_AGENT_OWNED_ACTIONS)


def test_register_executors_registers_optional_specialist():
    coord = _fake_coordinator()

    async def _spec(ctx):
        return {}

    _register_executors(coord, specialist_executor=_spec, session_dir=Path("."))
    assert coord.sub.executor_registry["specialist"] is _spec


async def _spec_stub(ctx):
    return {}


def _fully_wired_registry() -> dict[str, object]:
    coord = _fake_coordinator()
    _register_executors(coord, specialist_executor=_spec_stub, session_dir=None)
    return coord.sub.executor_registry


def test_every_coordinator_internal_action_has_an_executor():
    """Producer/consumer binding for the kinds the Coordinator enqueues itself."""
    from hyperloom.inference_optimizer.protocol.action_surfaces import (
        COORDINATOR_INTERNAL_ACTIONS,
    )

    registry = _fully_wired_registry()

    missing = sorted(COORDINATOR_INTERNAL_ACTIONS - set(registry))
    assert not missing, f"Coordinator-internal kinds with no executor: {missing}"


def test_no_executor_is_registered_under_an_unknown_action_name():
    """The reverse direction: a stale key left behind by a rename."""
    from hyperloom.inference_optimizer.protocol.action_surfaces import (
        ACTION_CATALOGUE,
        INTERNAL_ONLY_ACTION_NAMES,
    )

    registry = _fully_wired_registry()
    catalogue = {meta.name for meta in ACTION_CATALOGUE.values()}
    uncatalogued_by_design = INTERNAL_ONLY_ACTION_NAMES - catalogue

    phantom = sorted(set(registry) - catalogue - uncatalogued_by_design)
    assert not phantom, f"executor keys absent from ACTION_CATALOGUE: {phantom}"


def test_conditional_registrations_are_exactly_the_documented_exceptions():
    """Pin which kinds may legitimately be absent, so the exception set cannot drift."""
    minimal = _fake_coordinator()
    _register_executors(minimal, specialist_executor=None, session_dir=None)

    optional = set(_fully_wired_registry()) - set(minimal.sub.executor_registry)
    assert optional == {"specialist"}
