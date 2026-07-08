# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""KnowledgePlane facade tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from hyperloom.orchestrator.knowledge.knowledge_plane import KnowledgePlane
from hyperloom.orchestrator.knowledge.pr_monitor import PRMonitorClient


# 1. KnowledgePlane facade basics
def test_plane_disabled_pr_returns_empty_mcp_url():
    plane = KnowledgePlane.from_clients(
        pr_monitor=PRMonitorClient.from_args(enabled=False),
    )
    assert plane.pr_monitor_enabled is False
    assert plane.specialist_mcp_url() == ""
    assert plane.cortex_enabled is False


def test_plane_enabled_pr_returns_mcp_url():
    plane = KnowledgePlane.from_clients(
        pr_monitor=PRMonitorClient.from_args(enabled=True),
        pr_monitor_mcp_url="http://pr.test/mcp/",
    )
    assert plane.pr_monitor_enabled is True
    assert plane.specialist_mcp_url() == "http://pr.test/mcp/"


def test_plane_cortex_enabled_when_url_set():
    plane = KnowledgePlane.from_clients(
        pr_monitor=PRMonitorClient.from_args(enabled=False),
        cortex_kb_mcp_url="http://gbrain.test/mcp",
        cortex_kb_mcp_headers={"Authorization": "Bearer t"},
    )
    assert plane.cortex_enabled is True
    assert plane.cortex_specialist_mcp_url() == "http://gbrain.test/mcp"
    assert plane.cortex_specialist_mcp_headers() == {"Authorization": "Bearer t"}


def test_plane_reset_round_caches_noop():
    plane = KnowledgePlane.from_clients(
        pr_monitor=PRMonitorClient.from_args(),
    )
    plane.reset_round_caches()  # must not raise


# 2. EXPLORE-entry hook — no longer pre-warms; just runs cycle reprofile
@pytest.mark.asyncio
async def test_on_enter_explore_runs_without_plane(tmp_path: Path):
    """plane=None must not raise."""
    from hyperloom.orchestrator.loop.coordinator import Coordinator

    coord = Coordinator.__new__(Coordinator)
    coord.knowledge_plane = None
    coord.shared_state = _make_bare_shared_state()
    await coord._on_enter_explore(from_phase="PRELUDE")


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


# 3. _bootstrap_knowledge_plane status marker
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
    from hyperloom.inference_optimizer.cli import _bootstrap_knowledge_plane
    from hyperloom.inference_optimizer.session.session_paths import pr_monitor_status_json

    args = _build_args(pr_monitor_enabled=False)
    _bootstrap_knowledge_plane(args, cortex_client=None, session_dir=tmp_path)
    marker = pr_monitor_status_json(tmp_path)
    assert marker.exists()
    payload = json.loads(marker.read_text())
    assert payload["enabled"] is False
    assert payload["reachable"] is False


def test_bootstrap_marker_records_ir3_auto_degrade(tmp_path: Path, monkeypatch):
    from hyperloom.inference_optimizer.cli import _bootstrap_knowledge_plane
    from hyperloom.inference_optimizer.session.session_paths import pr_monitor_status_json
    from hyperloom.orchestrator.knowledge import pr_monitor as pr_mod

    class _Stub:
        def __init__(self, url, enabled):
            self.base_url = url or "https://example.test"
            self.enabled = enabled

    monkeypatch.setattr(
        pr_mod.PRMonitorClient,
        "from_args",
        classmethod(lambda cls, **kw: _Stub(url=kw.get("url") or "", enabled=kw.get("enabled", True))),
    )
    args = _build_args(pr_monitor_enabled=False, pr_degraded_reason="ir3_auto")
    _bootstrap_knowledge_plane(args, cortex_client=None, session_dir=tmp_path)
    payload = json.loads(pr_monitor_status_json(tmp_path).read_text())
    assert payload["enabled"] is False
    assert "ir3_auto" in payload.get("status_text", "")


# 4. breakdown.warnings wiring
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
    marker.write_text(json.dumps({"enabled": True, "reachable": True, "mcp_url": "http://pr.test/mcp/", "status_text": "ok"}))
    warnings_list: list = []
    collect_kb_provenance(tmp_path, state={}, manifest={}, warnings=warnings_list)
    assert not any(w.startswith("pr_monitor:") for w in warnings_list)


def test_collect_kb_provenance_no_warning_when_marker_missing(tmp_path: Path):
    from hyperloom.inference_optimizer.breakdown.collectors import collect_kb_provenance

    warnings_list: list = []
    collect_kb_provenance(tmp_path, state={}, manifest={}, warnings=warnings_list)
    assert not any(w.startswith("pr_monitor:") for w in warnings_list)
