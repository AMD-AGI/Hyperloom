# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for :mod:`hyperloom.inference_optimizer.cli.executors`.

Cover the specialist-executor factory (subprocess and in-process branches) and
the ``_register_executors`` wiring / ``_noop_prep`` stub without launching any
real ``claude`` subprocess.
"""

from __future__ import annotations

import argparse
import asyncio
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

        def cortex_specialist_mcp_url(self) -> str:
            return "http://cortex.invalid/mcp"

        def cortex_specialist_mcp_headers(self) -> dict:
            return {"authorization": "Bearer x"}

    executor = _build_specialist_executor(
        _spec_args("subprocess"),
        session_dir=tmp_path,
        knowledge_plane=_KP(),
    )
    assert callable(executor)


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
    assert "framework" in reg
    assert "roofline" in reg
    assert any(fn is _noop_prep for fn in reg.values())


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
