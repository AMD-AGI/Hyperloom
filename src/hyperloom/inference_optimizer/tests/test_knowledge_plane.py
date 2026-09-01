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


def test_collect_kb_provenance_surfaces_pr_monitor_disabled_warning(tmp_path: Path):
    from hyperloom.inference_optimizer.breakdown.collectors import collect_kb_provenance
    from hyperloom.inference_optimizer.session.session_paths import pr_monitor_status_json

    marker = pr_monitor_status_json(tmp_path)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({"enabled": False, "reachable": False, "mcp_url": "", "status_text": "disabled"}))
    warnings_list: list = []
    collect_kb_provenance(tmp_path, state={}, manifest={}, warnings=warnings_list)
    assert "pr_monitor:disabled" in warnings_list


def test_collect_kb_provenance_no_warning_when_plane_healthy(tmp_path: Path):
    from hyperloom.inference_optimizer.breakdown.collectors import collect_kb_provenance
    from hyperloom.inference_optimizer.session.session_paths import pr_monitor_status_json

    marker = pr_monitor_status_json(tmp_path)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(
        json.dumps({"enabled": True, "reachable": True, "mcp_url": "http://pr.test/mcp/", "status_text": "ok"})
    )
    warnings_list: list = []
    collect_kb_provenance(tmp_path, state={}, manifest={}, warnings=warnings_list)
    assert not any(w.startswith("pr_monitor:") for w in warnings_list)


def test_collect_kb_provenance_no_warning_when_marker_missing(tmp_path: Path):
    from hyperloom.inference_optimizer.breakdown.collectors import collect_kb_provenance

    out = collect_kb_provenance(tmp_path, state={}, manifest={}, warnings=[])
    assert out["recipe_snapshot_reads"]["count"] == 0
    assert out["recipe_snapshot_reads"]["hits"] == 0


def test_collect_kb_provenance_attributes_recipe_reads_per_source(
    tmp_path: Path,
):
    """Composite per-path provenance is aggregated into by_source / best_config_by_source."""
    from hyperloom.inference_optimizer.breakdown.collectors import collect_kb_provenance
    from hyperloom.inference_optimizer.session.session_paths import recipe_snapshot_audit_jsonl

    audit = recipe_snapshot_audit_jsonl(tmp_path)
    audit.parent.mkdir(parents=True, exist_ok=True)
    audit.write_text(
        "\n".join(
            json.dumps(r)
            for r in [
                {
                    "method": "get_recipe",
                    "remote": "composite",
                    "resolution": "remote",
                    "hit": True,
                    "result": {"sources": ["gbrain", "recipe_kb"], "best_config_source": "gbrain"},
                },
                {
                    "method": "get_recipe",
                    "remote": "composite",
                    "resolution": "remote",
                    "hit": True,
                    "result": {"sources": ["recipe_kb"], "best_config_source": "recipe_kb"},
                },
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    out = collect_kb_provenance(tmp_path, state={}, manifest={}, warnings=[])
    rs = out["recipe_snapshot_reads"]
    assert rs["by_source"] == {"gbrain": 1, "recipe_kb": 2}
    assert rs["best_config_by_source"] == {"gbrain": 1, "recipe_kb": 1}


def test_collect_kb_provenance_surfaces_warm_start_recipe_source(
    tmp_path: Path,
):
    """The applied warm recipe's KB path is surfaced from the WarmStartContext source tag."""
    from hyperloom.inference_optimizer.breakdown.collectors import collect_kb_provenance

    state = {
        "warm_start_recipe": {"raw": "{}", "tier": "exact", "recipe": {}},
        "warm_start_context": {"match": {"source": "recipe_kb"}},
    }
    out = collect_kb_provenance(tmp_path, state=state, manifest={}, warnings=[])
    assert out["warm_start_recipe_source"] == "recipe_kb"


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
    assert isinstance(args.pr_feed_window_days, int)
    assert args.pr_feed_window_days > 0


def test_cli_degraded_pr_sets_flag_true():
    args = _parse_optimize_args(["--degraded-pr"])
    assert args.degraded_pr is True


@pytest.mark.parametrize("flag", ["--pr-monitor-url", "--pr-monitor-mcp-url"])
def test_removed_independent_pr_endpoint_flags_are_rejected(flag: str):
    with pytest.raises(SystemExit):
        _parse_optimize_args([flag, "https://legacy.example/pr-monitor"])


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
    monkeypatch.setenv("KB_STORE_URL", "https://kb.example/knowledge-base")

    args = _parse_optimize_args(["--pr-feed-window-days", "14"])
    plane = _bootstrap_knowledge_plane(
        args,
        recipe_kb_client=None,
        session_dir=tmp_path,
    )
    assert constructed_enabled[-1] is True
    assert plane.pr_monitor_mcp_url == "https://kb.example/knowledge-base/pr-monitor/mcp/"
