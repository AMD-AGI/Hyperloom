# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""KnowledgePlane facade tests."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from hyperloom.orchestrator.knowledge.knowledge_plane import KnowledgePlane
from hyperloom.orchestrator.phases.machine import Transition


def test_plane_disabled_pr_returns_empty_mcp_url():
    plane = KnowledgePlane.from_clients(pr_monitor_mcp_url="http://pr.test/mcp/")
    assert plane.pr_monitor_enabled is False
    assert plane.specialist_mcp_url() == ""
    assert KnowledgePlane().pr_monitor_enabled is False


def test_plane_enabled_pr_returns_mcp_url():
    plane = KnowledgePlane.from_clients(
        pr_monitor_enabled=True,
        pr_monitor_mcp_url="http://pr.test/mcp/",
    )
    assert plane.pr_monitor_enabled is True
    assert plane.specialist_mcp_url() == "http://pr.test/mcp/"


@pytest.mark.asyncio
async def test_on_enter_the_optimisation_phase_runs_without_plane(tmp_path: Path):
    """plane=None must not raise."""
    from hyperloom.orchestrator.loop.coordinator import Coordinator

    coord = Coordinator.__new__(Coordinator)
    coord.knowledge_plane = None
    coord.shared_state = _make_bare_shared_state()
    coord.session_dir = tmp_path
    await coord.phase_framework.on_enter_framework(
        Transition(from_phase="PRELUDE", to_phase="FRAMEWORK", reason="test", evidence={})
    )


def _make_bare_shared_state():
    from dataclasses import dataclass

    from hyperloom.orchestrator.state.shared_state import SharedState

    @dataclass
    class _SS(SharedState):
        save_count: int = 0

        def save(self, _session_dir):
            self.save_count += 1

    return _SS()


def _build_args(**overrides) -> argparse.Namespace:
    base = dict(
        pr_monitor_enabled=True,
        pr_degraded_reason=None,
        kb_degraded_reason=None,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def test_bootstrap_reports_ir3_auto_degrade(capsys):
    from hyperloom.inference_optimizer.cli.kb import _bootstrap_knowledge_plane

    args = _build_args(pr_monitor_enabled=False, pr_degraded_reason="ir3_auto")
    plane = _bootstrap_knowledge_plane(args, recipe_kb_client=None)
    assert "PR Monitor       : DISABLED (ir3_auto)" in capsys.readouterr().out
    assert plane.pr_monitor_enabled is False


def test_bootstrap_disables_pr_monitor_without_an_mcp_url(monkeypatch, capsys):
    from hyperloom.inference_optimizer.cli.kb import _bootstrap_knowledge_plane

    monkeypatch.setenv("KNOWLEDGE_STORE_MODE", "remote")
    monkeypatch.delenv("KB_STORE_URL", raising=False)
    plane = _bootstrap_knowledge_plane(_build_args(degraded_kb=True), recipe_kb_client=None)
    assert "PR Monitor       : DISABLED (no MCP URL configured)" in capsys.readouterr().out
    assert plane.pr_monitor_enabled is False
    assert plane.specialist_mcp_url() == ""


def _parse_optimize_args(extra: list[str]) -> argparse.Namespace:
    """Pin the dest-name + default contract the bootstrap reads."""
    from hyperloom.inference_optimizer.cli.parser import _build_parser

    parser = _build_parser()
    return parser.parse_args(["optimize", "--degraded-kb", *extra])


def test_cli_pr_monitor_controls_have_expected_defaults():
    args = _parse_optimize_args([])
    assert args.degraded_pr is False
    assert not hasattr(args, "pr_monitor_url")
    assert not hasattr(args, "pr_monitor_mcp_url")


def test_cli_degraded_pr_sets_flag_true():
    args = _parse_optimize_args(["--degraded-pr"])
    assert args.degraded_pr is True


@pytest.mark.parametrize("flag", ["--pr-monitor-url", "--pr-monitor-mcp-url"])
def test_removed_independent_pr_endpoint_flags_are_rejected(flag: str):
    with pytest.raises(SystemExit):
        _parse_optimize_args([flag, "https://legacy.example/pr-monitor"])


def test_cli_args_round_trip_into_bootstrap_knowledge_plane(monkeypatch):
    """Argparse ``args`` values propagate into the KnowledgePlane."""
    from hyperloom.inference_optimizer.cli.kb import _bootstrap_knowledge_plane

    monkeypatch.setenv("KB_STORE_URL", "https://kb.example/knowledge-base")

    args = _parse_optimize_args([])
    plane = _bootstrap_knowledge_plane(args, recipe_kb_client=None)
    assert plane.pr_monitor_enabled is True
    assert plane.pr_monitor_mcp_url == "https://kb.example/knowledge-base/pr-monitor/mcp/"
