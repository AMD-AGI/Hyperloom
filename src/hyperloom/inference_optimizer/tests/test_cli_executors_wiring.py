# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for :mod:`hyperloom.inference_optimizer.cli.executors`.

Cover the specialist-executor factory (subprocess and in-process branches) and
the ``_register_executors`` wiring / ``_noop_prep`` stub without launching any
real ``claude`` subprocess.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path
from types import SimpleNamespace

from hyperloom.inference_optimizer.cli import executors as cli_executors
from hyperloom.inference_optimizer.cli.executors import (
    _build_specialist_executor,
    _noop_prep,
    _register_executors,
    _REAL_EXECUTORS_FULL,
)


def _spec_args(dispatch_mode: str) -> argparse.Namespace:
    return argparse.Namespace(
        claude_model="claude-opus-4-6",
        specialist_model=None,
        specialist_max_turns=3,
        specialist_per_turn_max_seconds=120.0,
        specialist_dispatch_mode=dispatch_mode,
        specialist_mcp_config=None,
    )


def test_noop_prep_returns_success_envelope():
    ctx = SimpleNamespace(task=SimpleNamespace(kind="rewrite_kernel"))
    out = asyncio.run(_noop_prep(ctx))
    assert out == {"status": "succeeded", "kind": "rewrite_kernel", "note": "noop-stub"}


def test_build_specialist_executor_inprocess_when_no_claude(monkeypatch, tmp_path):
    """dispatch_mode=inprocess builds the in-process backend runner."""
    import shutil

    monkeypatch.setattr(shutil, "which", lambda _n: "")
    executor = _build_specialist_executor(
        _spec_args("inprocess"),
        session_dir=tmp_path,
        knowledge_plane=None,
    )
    assert callable(executor)


def test_build_specialist_executor_subprocess_fallback_warns(monkeypatch, tmp_path, caplog):
    """subprocess requested but no claude binary -> warns + falls back."""
    import shutil

    monkeypatch.setattr(shutil, "which", lambda _n: "")
    with caplog.at_level(logging.WARNING, logger=cli_executors.log.name):
        executor = _build_specialist_executor(
            _spec_args("subprocess"),
            session_dir=tmp_path,
            knowledge_plane=None,
        )
    assert callable(executor)
    assert any("claude" in rec.message for rec in caplog.records)


def test_build_specialist_executor_subprocess_with_knowledge_plane(monkeypatch, tmp_path):
    """subprocess path with a KnowledgePlane generates an MCP config."""
    import shutil

    monkeypatch.setattr(shutil, "which", lambda _n: "/usr/bin/claude")

    class _KP:
        def specialist_mcp_url(self) -> str:
            return "http://pr-monitor.invalid/mcp"

    executor = _build_specialist_executor(
        _spec_args("subprocess"),
        session_dir=tmp_path,
        knowledge_plane=_KP(),
    )
    assert callable(executor)


def test_mcp_servers_from_explicit_config(tmp_path):
    cfg = tmp_path / "mcp.json"
    cfg.write_text(
        json.dumps({"mcpServers": {"recipe_kb": {"type": "http"}, "pr_monitor": {"type": "http"}}}),
        encoding="utf-8",
    )
    assert set(cli_executors._mcp_servers_from_config(str(cfg))) == {"recipe_kb", "pr_monitor"}


def test_mcp_servers_from_absent_config_is_not_authoritative():
    assert cli_executors._mcp_servers_from_config(None) is None


def test_mcp_servers_from_empty_explicit_config_is_authoritative(tmp_path):
    cfg = tmp_path / "mcp.json"
    cfg.write_text(json.dumps({"mcpServers": {}}), encoding="utf-8")
    assert cli_executors._mcp_servers_from_config(str(cfg)) == ()


def test_build_specialist_executor_subprocess_kp_missing_methods(monkeypatch, tmp_path):
    """subprocess path tolerates a KnowledgePlane lacking the MCP-url methods."""
    import shutil

    monkeypatch.setattr(shutil, "which", lambda _n: "/usr/bin/claude")

    executor = _build_specialist_executor(
        _spec_args("subprocess"),
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


def test_register_executors_wires_full_set_and_kernel_noops():
    coord = _fake_coordinator()
    _register_executors(coord, no_kernel=False, session_dir=None)
    reg = coord.sub.executor_registry
    for kind in _REAL_EXECUTORS_FULL:
        assert kind in reg
    assert "target_analysis" in reg
    assert "integrate_patch" in reg
    # Must match the kind the FRAMEWORK phase enqueues; a stale "framework"
    # key silently drops every discovered PR candidate.
    assert "framework_agent" in reg
    assert "framework" not in reg
    assert "roofline" in reg
    assert any(fn is _noop_prep for fn in reg.values())


def test_register_executors_covers_every_phase_allowed_action():
    """Every action a phase may enqueue resolves to a registered executor.

    ``SubAgentRunner.run_task`` fails a task with ``no_executor`` when
    ``task.kind`` has no registry entry, so a name that drifts between the
    enqueue site and the registration site turns into a silent phase-wide
    failure. Deriving the expectation from ``PHASE_ALLOWED_ACTIONS`` keeps the
    two in lockstep instead of re-listing kinds by hand.
    """
    from hyperloom.orchestrator.phases.machine_state import PHASE_ALLOWED_ACTIONS

    coord = _fake_coordinator()

    async def _spec(ctx):  # noqa: ANN001, ANN202 - test stub
        return {}

    _register_executors(coord, no_kernel=False, session_dir=None, specialist_executor=_spec)
    reg = coord.sub.executor_registry

    expected: set[str] = set()
    for actions in PHASE_ALLOWED_ACTIONS.values():
        expected |= set(actions)
    missing = sorted(kind for kind in expected if kind not in reg)
    assert not missing, f"phase-allowed actions with no executor: {missing}"


def test_register_executors_no_kernel_skips_noops_and_debug_log(caplog):
    coord = _fake_coordinator()
    with caplog.at_level(logging.DEBUG, logger=cli_executors.log.name):
        _register_executors(coord, no_kernel=True, session_dir=None)
    reg = coord.sub.executor_registry
    assert "roofline" in reg
    assert not any(fn is _noop_prep for fn in reg.values())


def test_register_executors_registers_optional_specialist():
    coord = _fake_coordinator()

    async def _spec(ctx):  # noqa: ANN001, ANN202 - test stub
        return {}

    _register_executors(coord, no_kernel=True, specialist_executor=_spec, session_dir=Path("."))
    assert coord.sub.executor_registry["specialist"] is _spec


async def _spec_stub(ctx):  # noqa: ANN001, ANN202 - test stub
    return {}


def _fully_wired_registry() -> dict[str, object]:
    coord = _fake_coordinator()
    _register_executors(coord, no_kernel=False, specialist_executor=_spec_stub, session_dir=None)
    return coord.sub.executor_registry


def test_every_coordinator_internal_action_has_an_executor():
    """Producer/consumer binding for the kinds the Coordinator enqueues itself.

    These actions are never proposed by an agent, so a missing executor
    surfaces only as a silently failed task at runtime. That is how the
    FRAMEWORK phase came to enqueue ``framework_agent`` against a registry
    that only knew ``framework``, failing every discovered PR candidate. The
    expectation is read off the production vocabulary, so adding an action
    cannot leave this guard behind.
    """
    from hyperloom.inference_optimizer.protocol.action_surfaces import (
        COORDINATOR_INTERNAL_ACTIONS,
    )

    registry = _fully_wired_registry()

    missing = sorted(COORDINATOR_INTERNAL_ACTIONS - set(registry))
    assert not missing, f"Coordinator-internal kinds with no executor: {missing}"


def test_no_executor_is_registered_under_an_unknown_action_name():
    """The reverse direction: a stale key left behind by a rename.

    A registration whose name is not in the action catalogue can never be
    enqueued, so it is dead weight that also makes the real gap harder to see.
    """
    from hyperloom.orchestrator.actions.registry import ActionRegistry

    registry = _fully_wired_registry()
    catalogue = {meta.name for meta in ActionRegistry().load().all()}

    phantom = sorted(set(registry) - catalogue)
    assert not phantom, f"executor keys with no actions/_meta/*.yaml: {phantom}"


def test_conditional_registrations_are_exactly_the_documented_exceptions():
    """Pin which kinds may legitimately be absent, so the exception set cannot drift.

    Only two conditions remove an executor: ``--no-kernel`` drops the
    kernel-owned stubs, and a zero research-lane capacity drops the
    specialist. Anything else disappearing is a wiring bug.
    """
    from hyperloom.inference_optimizer.protocol.action_surfaces import (
        KERNEL_AGENT_OWNED_ACTIONS,
    )

    minimal = _fake_coordinator()
    _register_executors(minimal, no_kernel=True, specialist_executor=None, session_dir=None)

    optional = set(_fully_wired_registry()) - set(minimal.sub.executor_registry)
    assert optional == set(KERNEL_AGENT_OWNED_ACTIONS) | {"specialist"}
