# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""KnowledgePlane facade tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from hyperloom.orchestrator.knowledge.knowledge_plane import KnowledgePlane
from hyperloom.orchestrator.knowledge.pr_monitor import PRMonitorClient


def test_plane_disabled_pr_returns_empty_mcp_url():
    plane = KnowledgePlane.from_clients(
        pr_monitor=PRMonitorClient.from_args(enabled=False),
    )
    assert plane.pr_monitor_enabled is False
    assert plane.specialist_mcp_url() == ""


def test_plane_enabled_pr_returns_mcp_url():
    plane = KnowledgePlane.from_clients(
        pr_monitor=PRMonitorClient.from_args(enabled=True),
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
    await coord._on_enter_framework(from_phase="PRELUDE")


def _make_bare_shared_state():
    from dataclasses import dataclass, field

    @dataclass
    class _SS:
        baseline_tput: float = 0.0
        last_roofline_tput: float = 0.0
        last_trace_analyze: dict = field(default_factory=dict)
        cumulative_gain_validated: float = 0.0
        auto_roofline_pending_task_id: str = ""
        phase_history: list = field(default_factory=list)
        save_count: int = 0

        def save(self, _session_dir):
            self.save_count += 1

    return _SS()


def _build_args(**overrides) -> argparse.Namespace:
    base = dict(
        pr_monitor_enabled=True,
        pr_monitor_url=None,
        pr_monitor_mcp_url=None,
        pr_degraded_reason=None,
        kb_degraded_reason=None,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def test_bootstrap_writes_status_marker_when_disabled(tmp_path: Path):
    from hyperloom.inference_optimizer.cli.kb import _bootstrap_knowledge_plane
    from hyperloom.inference_optimizer.session.session_paths import pr_monitor_status_json

    args = _build_args(pr_monitor_enabled=False)
    _bootstrap_knowledge_plane(args, recipe_kb_client=None, session_dir=tmp_path)
    marker = pr_monitor_status_json(tmp_path)
    assert marker.exists()
    payload = json.loads(marker.read_text())
    assert payload["enabled"] is False
    assert payload["reachable"] is False


def test_bootstrap_marker_records_ir3_auto_degrade(tmp_path: Path, monkeypatch):
    from hyperloom.inference_optimizer.cli.kb import _bootstrap_knowledge_plane
    from hyperloom.inference_optimizer.session.session_paths import pr_monitor_status_json
    from hyperloom.orchestrator.knowledge import pr_monitor as pr_mod

    class _Stub:
        def __init__(self, enabled):
            self.enabled = enabled

    monkeypatch.setattr(
        pr_mod.PRMonitorClient,
        "from_args",
        classmethod(lambda cls, **kw: _Stub(enabled=kw.get("enabled", True))),
    )
    args = _build_args(pr_monitor_enabled=False, pr_degraded_reason="ir3_auto")
    _bootstrap_knowledge_plane(args, recipe_kb_client=None, session_dir=tmp_path)
    payload = json.loads(pr_monitor_status_json(tmp_path).read_text())
    assert payload["enabled"] is False
    assert "ir3_auto" in payload.get("status_text", "")


def _parse_optimize_args(extra: list[str]) -> argparse.Namespace:
    """Pin the dest-name + default contract the bootstrap reads."""
    from hyperloom.inference_optimizer.cli.parser import _build_parser

    parser = _build_parser()
    return parser.parse_args(["optimize", "--degraded-kb", *extra])


def test_cli_pr_monitor_flags_have_expected_dest_and_defaults():
    """PR-monitor flags land under the dest names the bootstrap reads."""
    args = _parse_optimize_args([])
    # dest is ``degraded_pr`` (store_true, default False).
    assert args.degraded_pr is False
    assert args.pr_monitor_url is None
    assert args.pr_monitor_mcp_url is None
    assert isinstance(args.pr_feed_window_days, int)
    assert args.pr_feed_window_days > 0


def test_cli_degraded_pr_sets_flag_true():
    args = _parse_optimize_args(["--degraded-pr"])
    assert args.degraded_pr is True


def test_cli_pr_monitor_url_override_reaches_namespace():
    args = _parse_optimize_args(
        [
            "--pr-monitor-url",
            "https://localhost:8080/v1",
        ]
    )
    assert args.pr_monitor_url == "https://localhost:8080/v1"


def test_cli_pr_monitor_mcp_url_override_reaches_namespace():
    args = _parse_optimize_args(
        [
            "--pr-monitor-mcp-url",
            "https://localhost:8080/mcp/",
        ]
    )
    assert args.pr_monitor_mcp_url == "https://localhost:8080/mcp/"


def test_cli_pr_monitor_url_defaults_to_primus_cortex_pr_api_env(monkeypatch):
    """--pr-monitor-url default resolves from the canonical $PRIMUS_CORTEX_PR_API env."""
    monkeypatch.setenv("PRIMUS_CORTEX_PR_API", "https://global.example/pr-monitor")
    monkeypatch.delenv("PR_MONITOR_URL", raising=False)
    args = _parse_optimize_args([])
    assert args.pr_monitor_url == "https://global.example/pr-monitor"


def test_cli_pr_monitor_url_ignores_legacy_pr_monitor_url_env(monkeypatch):
    """The removed legacy $PR_MONITOR_URL env no longer feeds --pr-monitor-url."""
    monkeypatch.delenv("PRIMUS_CORTEX_PR_API", raising=False)
    monkeypatch.setenv("PR_MONITOR_URL", "https://legacy.example/pr-monitor/v1")
    args = _parse_optimize_args([])
    assert args.pr_monitor_url is None


def test_cli_pr_monitor_url_flag_wins_over_primus_cortex_pr_api_env(monkeypatch):
    monkeypatch.setenv("PRIMUS_CORTEX_PR_API", "https://env.example/pr-monitor")
    args = _parse_optimize_args(["--pr-monitor-url", "https://flag.example/pr-monitor"])
    assert args.pr_monitor_url == "https://flag.example/pr-monitor"


def test_cli_pr_feed_window_days_override_reaches_namespace():
    args = _parse_optimize_args(["--pr-feed-window-days", "7"])
    assert args.pr_feed_window_days == 7


def test_cli_args_round_trip_into_bootstrap_knowledge_plane(
    tmp_path: Path,
    monkeypatch,
):
    """Argparse ``args`` values propagate into the KnowledgePlane."""
    from hyperloom.inference_optimizer.cli.kb import _bootstrap_knowledge_plane
    from hyperloom.orchestrator.knowledge import pr_monitor as pr_mod

    constructed_enabled: list[bool] = []

    class _Stub:
        def __init__(self, enabled: bool):
            self.enabled = enabled
            constructed_enabled.append(enabled)

        def healthz(self) -> bool:
            return True

        def reset_cache(self) -> None:
            pass

    monkeypatch.setattr(
        pr_mod.PRMonitorClient,
        "from_args",
        classmethod(lambda cls, **kw: _Stub(enabled=kw.get("enabled", True))),
    )

    args = _parse_optimize_args(
        [
            "--pr-monitor-url",
            "https://my-pr-monitor.example/v1",
            "--pr-monitor-mcp-url",
            "https://my-pr-monitor.example/mcp/",
            "--pr-feed-window-days",
            "14",
        ]
    )
    plane = _bootstrap_knowledge_plane(
        args,
        recipe_kb_client=None,
        session_dir=tmp_path,
    )
    assert constructed_enabled[-1] is True
    # The plane does not store pr_feed_window_days; only the MCP URL round-trips.
    assert plane.pr_monitor_mcp_url == "https://my-pr-monitor.example/mcp/"
