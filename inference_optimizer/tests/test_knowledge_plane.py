# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""v0.8 §3.6 / M4 — KnowledgePlane integration tests (KB_gaps/Gap-02)."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest


# Fixtures — minimal PR Monitor / cortex client doubles
@dataclass
class _StubPRClient:
    """Minimal stand-in for :class:`PRMonitorClient`."""

    enabled: bool = True
    base_url: str = "https://pr-monitor.test"
    healthz_ok: bool = True
    pr_feed_payload: list = field(default_factory=list)
    list_repos_payload: list = field(default_factory=list)

    def healthz(self) -> bool:
        return self.healthz_ok

    def reset_cache(self) -> None:
        return None

    def list_repos(self) -> list:
        return list(self.list_repos_payload)

    def pr_feed_warm(
        self,
        repos: list,
        *,
        keywords: list,
        window_days: int,
        per_repo_limit: int,
        total_budget_sec: float,
    ) -> tuple[list, list]:
        return list(self.pr_feed_payload), []


# 1. KnowledgePlane.pr_feed_warm_all_domains
def test_pr_feed_warm_all_domains_returns_entry_per_known_domain():
    """KB_gaps/Gap-02 — every known specialist domain appears in the result map."""
    from inference_optimizer.orchestrator.knowledge_plane import (
        KnowledgePlane,
        load_domain_repos,
    )
    from inference_optimizer.orchestrator.specialist_domains import (
        SPECIALIST_DOMAIN_KEYS,
    )

    plane = KnowledgePlane.from_clients(
        cortex_kb=None,
        pr_monitor=None,
        domain_repos=load_domain_repos(),
    )
    out = plane.pr_feed_warm_all_domains()
    for domain in SPECIALIST_DOMAIN_KEYS:
        assert domain in out, f"missing {domain!r} in pr_feed_warm_all_domains"
        prs, warns = out[domain]
        assert isinstance(prs, list)
        assert isinstance(warns, list)


def test_pr_feed_warm_all_domains_stashes_warnings():
    """Aggregated warnings land on ``last_warnings`` for one-pass surfacing."""
    from inference_optimizer.orchestrator.knowledge_plane import (
        KnowledgePlane,
        load_domain_repos,
    )

    plane = KnowledgePlane.from_clients(
        cortex_kb=None,
        pr_monitor=None,
        domain_repos=load_domain_repos(),
    )
    plane.pr_feed_warm_all_domains()
    # pr_monitor=None → every domain yields a ``pr_monitor:disabled`` warning.
    assert plane.last_warnings, "warnings should be aggregated"
    assert any("pr_monitor:disabled" in w for w in plane.last_warnings)


def test_pr_feed_warm_all_domains_isolates_per_domain_failures():
    """A failure on one domain must NOT abort the rest of the batch (KB_design §3.14 R-03)."""
    from inference_optimizer.orchestrator.knowledge_plane import (
        KnowledgePlane,
        load_domain_repos,
    )
    from inference_optimizer.orchestrator.specialist_domains import (
        SPECIALIST_DOMAIN_KEYS,
    )

    plane = KnowledgePlane.from_clients(
        cortex_kb=None,
        pr_monitor=None,
        domain_repos=load_domain_repos(),
    )

    real_pr_feed_warm = plane.pr_feed_warm
    call_log: list = []

    def _flaky_pr_feed_warm(domain: str, **kwargs):
        call_log.append(domain)
        if domain == "serving_specialist":
            raise RuntimeError("synthetic PR monitor outage")
        return real_pr_feed_warm(domain, **kwargs)

    plane.pr_feed_warm = _flaky_pr_feed_warm  # type: ignore[assignment]
    out = plane.pr_feed_warm_all_domains()

    assert set(call_log) == set(SPECIALIST_DOMAIN_KEYS)
    prs, warns = out["serving_specialist"]
    assert prs == []
    assert any("serving_specialist" in w for w in warns)


# 2. Coordinator EXPLORE phase entry auto-warm
class _FakePlane:
    """Lighter-than-KnowledgePlane double that just records calls."""

    pr_monitor_enabled = True
    cortex_enabled = True

    def __init__(self):
        self.warm_calls: int = 0
        self.last_kwargs: dict | None = None

    def pr_feed_warm_all_domains(self, **kwargs):
        self.warm_calls += 1
        self.last_kwargs = kwargs
        return {"serving_specialist": ([], [])}

    def pr_feed_warm(self, domain, **_kw):
        return [], []


def _make_bare_shared_state():
    """Minimal SharedState stand-in for the EXPLORE-entry tests."""
    from dataclasses import dataclass, field

    @dataclass
    class _SS:
        baseline_tput: float = 0.0
        last_roofline_tput: float = 0.0
        last_trace_analyze: dict[str, Any] = field(default_factory=dict)
        cumulative_gain_validated: float = 0.0
        auto_roofline_pending_task_id: str = ""
        phase_history: list[dict[str, Any]] = field(default_factory=list)
        save_count: int = 0

        def save(self, _session_dir):
            self.save_count += 1

    return _SS()


@pytest.mark.asyncio
async def test_on_enter_explore_warms_pr_feed(tmp_path: Path):
    """The EXPLORE-entry hook calls ``pr_feed_warm_all_domains`` exactly once."""
    from inference_optimizer.orchestrator.coordinator import Coordinator

    coord = Coordinator.__new__(Coordinator)
    plane = _FakePlane()
    coord.knowledge_plane = plane
    coord.shared_state = _make_bare_shared_state()

    await coord._on_enter_explore(from_phase="PRELUDE")
    assert plane.warm_calls == 1


@pytest.mark.asyncio
async def test_on_enter_explore_graceful_when_plane_is_none(tmp_path: Path):
    """``--degraded-kb`` runs have plane=None; the hook must short-circuit."""
    from inference_optimizer.orchestrator.coordinator import Coordinator

    coord = Coordinator.__new__(Coordinator)
    coord.knowledge_plane = None
    coord.shared_state = _make_bare_shared_state()
    await coord._on_enter_explore(from_phase="PRELUDE")


@pytest.mark.asyncio
async def test_on_enter_explore_swallows_warmup_exceptions(tmp_path: Path):
    """A raising ``pr_feed_warm_all_domains`` must log + continue, not crash."""
    from inference_optimizer.orchestrator.coordinator import Coordinator

    class _BadPlane:
        pr_monitor_enabled = True
        cortex_enabled = True

        def pr_feed_warm_all_domains(self, **_kw):
            raise RuntimeError("synthetic warmup outage")

    coord = Coordinator.__new__(Coordinator)
    coord.knowledge_plane = _BadPlane()
    coord.shared_state = _make_bare_shared_state()
    await coord._on_enter_explore(from_phase="PRELUDE")


@pytest.mark.asyncio
async def test_on_phase_entered_only_explore_fires_pr_feed_warmup(tmp_path: Path):
    """PR-feed warmup fires on EXPLORE and only EXPLORE, never KERNEL/SWEEP/CLOSE."""
    from inference_optimizer.orchestrator.coordinator import Coordinator

    coord = Coordinator.__new__(Coordinator)
    coord.session_dir = tmp_path
    plane = _FakePlane()
    coord.knowledge_plane = plane

    # State that lets the non-EXPLORE hooks short-circuit; the assertion
    # is "plane.warm_calls stays 0 for non-EXPLORE".
    @dataclass
    class _BareState:
        kernel_enabled: bool = False
        last_profile_trace: str = ""
        baseline_config_path: str = ""
        warm_start_recipe: dict | None = None
        cortex_session_id: str = ""
        cortex_session_summary: dict = field(default_factory=dict)
        closing_report_task_id: str = ""
        stop_reason: str = ""
        current_best: dict = field(default_factory=dict)
        last_baseline: dict = field(default_factory=dict)
        phase_history: list = field(default_factory=list)
        close_sequence_done: bool = False

        def save(self, _session_dir: Path | None) -> None:
            return None

        def set_stop_reason(self, value: str) -> str:
            self.stop_reason = value
            return value
    coord.shared_state = _BareState()
    coord.role_registry = {}   # _kernel_enabled() reads role_registry
    coord.cortex_kb = None

    class _StubTaskRegistry:
        async def create_or_return_existing(self, **kwargs):
            from inference_optimizer.orchestrator.task_registry import Task
            import uuid as _uuid
            return Task(
                task_id=_uuid.uuid4().hex,
                kind=kwargs["kind"],
                state="succeeded",  # terminal so _wait_for_task_terminal short-circuits
                params=kwargs["params"],
                idempotency_key=kwargs["idempotency_key"],
            ), False

        async def get(self, task_id):
            from inference_optimizer.orchestrator.task_registry import Task
            return Task(
                task_id=task_id, kind="report", state="succeeded",
                params={}, idempotency_key="",
            )
    coord.tasks = _StubTaskRegistry()

    await coord._on_phase_entered(from_phase="PRELUDE", to_phase="KERNEL")
    await coord._on_phase_entered(from_phase="PRELUDE", to_phase="SWEEP")
    await coord._on_phase_entered(from_phase="PRELUDE", to_phase="CLOSE")
    assert plane.warm_calls == 0

    await coord._on_phase_entered(from_phase="PRELUDE", to_phase="EXPLORE")
    assert plane.warm_calls == 1


# 3. _bootstrap_knowledge_plane status marker + breakdown.warnings wiring
def _build_args(**overrides) -> argparse.Namespace:
    base = dict(
        pr_monitor_enabled=True,
        pr_monitor_url=None,
        pr_monitor_mcp_url=None,
        pr_feed_window_days=30,
        pr_degraded_reason=None,
        kb_degraded_reason=None,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def test_bootstrap_writes_status_marker_when_disabled(tmp_path: Path):
    """``--degraded-pr``: marker declares enabled=False for ``pr_monitor:disabled``."""
    from inference_optimizer.cli import _bootstrap_knowledge_plane
    from inference_optimizer.session_paths import pr_monitor_status_json

    args = _build_args(pr_monitor_enabled=False)
    _bootstrap_knowledge_plane(
        args, cortex_client=None, session_dir=tmp_path,
    )
    marker = pr_monitor_status_json(tmp_path)
    assert marker.exists(), "status marker should be written"
    payload = json.loads(marker.read_text())
    assert payload["enabled"] is False
    assert payload["reachable"] is False


def test_bootstrap_marker_records_ir3_auto_degrade(
    tmp_path: Path, monkeypatch,
):
    """IR-3 auto-degrade: marker shows ``enabled=False`` + ``ir3_auto`` in status_text."""
    from inference_optimizer.cli import _bootstrap_knowledge_plane
    from inference_optimizer.session_paths import pr_monitor_status_json
    from inference_optimizer.orchestrator import pr_monitor as pr_mod

    class _Stub:
        def __init__(self, url: str, enabled: bool):
            self.base_url = url or "https://example.test"
            self.enabled = enabled

        def healthz(self) -> bool:
            return False

        def reset_cache(self) -> None:
            pass

    monkeypatch.setattr(
        pr_mod.PRMonitorClient,
        "from_args",
        classmethod(
            lambda cls, **kw: _Stub(
                url=kw.get("url") or "", enabled=kw.get("enabled", True),
            ),
        ),
    )
    args = _build_args(
        pr_monitor_enabled=False,
        pr_degraded_reason="ir3_auto",
        pr_monitor_url="https://pr-monitor.test",
    )
    _bootstrap_knowledge_plane(
        args, cortex_client=None, session_dir=tmp_path,
    )
    marker = pr_monitor_status_json(tmp_path)
    payload = json.loads(marker.read_text())
    assert payload["enabled"] is False
    assert payload["reachable"] is False
    assert "ir3_auto" in payload.get("status_text", "")


def test_collect_kb_provenance_surfaces_pr_monitor_disabled_warning(
    tmp_path: Path,
):
    """The breakdown collector emits ``pr_monitor:disabled`` when the marker says so."""
    from inference_optimizer.breakdown.collectors import collect_kb_provenance
    from inference_optimizer.session_paths import pr_monitor_status_json

    marker = pr_monitor_status_json(tmp_path)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({
        "enabled": False,
        "url": "",
        "reachable": False,
        "mcp_url": "",
        "window_days": 30,
        "status_text": "disabled (--degraded-pr)",
    }))

    warnings_list: list = []
    collect_kb_provenance(
        tmp_path,
        state={},
        manifest={},
        warnings=warnings_list,
    )
    assert "pr_monitor:disabled" in warnings_list


def test_collect_kb_provenance_surfaces_pr_monitor_unreachable_warning(
    tmp_path: Path,
):
    from inference_optimizer.breakdown.collectors import collect_kb_provenance
    from inference_optimizer.session_paths import pr_monitor_status_json

    marker = pr_monitor_status_json(tmp_path)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({
        "enabled": True,
        "url": "https://pr-monitor.test",
        "reachable": False,
        "mcp_url": "https://pr-monitor.test/mcp/",
        "window_days": 30,
        "status_text": "unreachable at https://pr-monitor.test",
    }))

    warnings_list: list = []
    collect_kb_provenance(
        tmp_path,
        state={},
        manifest={},
        warnings=warnings_list,
    )
    assert any(w.startswith("pr_monitor:unreachable") for w in warnings_list)


def test_collect_kb_provenance_no_warning_when_plane_healthy(
    tmp_path: Path,
):
    """Happy path: marker says enabled + reachable → no warning."""
    from inference_optimizer.breakdown.collectors import collect_kb_provenance
    from inference_optimizer.session_paths import pr_monitor_status_json

    marker = pr_monitor_status_json(tmp_path)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({
        "enabled": True,
        "url": "https://pr-monitor.test",
        "reachable": True,
        "mcp_url": "https://pr-monitor.test/mcp/",
        "window_days": 30,
        "status_text": "REST https://pr-monitor.test (window=30d)",
    }))

    warnings_list: list = []
    collect_kb_provenance(
        tmp_path,
        state={},
        manifest={},
        warnings=warnings_list,
    )
    assert not any(w.startswith("pr_monitor:") for w in warnings_list), \
        f"healthy plane should emit no pr_monitor warning, got: {warnings_list}"


def test_collect_kb_provenance_no_warning_when_marker_missing(
    tmp_path: Path,
):
    """A missing marker must not produce spurious warnings — absence ≠ failure."""
    from inference_optimizer.breakdown.collectors import collect_kb_provenance
    warnings_list: list = []
    collect_kb_provenance(
        tmp_path,
        state={},
        manifest={},
        warnings=warnings_list,
    )
    assert not any(w.startswith("pr_monitor:") for w in warnings_list)


def test_collect_kb_provenance_summarises_recipe_snapshot_reads(
    tmp_path: Path,
):
    """The recipe-snapshot read audit is summarised into kb_provenance."""
    from inference_optimizer.breakdown.collectors import collect_kb_provenance
    from inference_optimizer.session_paths import recipe_snapshot_audit_jsonl

    audit = recipe_snapshot_audit_jsonl(tmp_path)
    audit.parent.mkdir(parents=True, exist_ok=True)
    audit.write_text("\n".join(json.dumps(r) for r in [
        {"method": "get_recipe", "remote": "gbrain", "resolution": "remote", "hit": True},
        {"method": "get_recipe", "remote": "gbrain", "resolution": "remote_miss", "hit": False},
        {"method": "get_recipe", "remote": "cortex", "resolution": "local", "hit": True},
    ]) + "\n", encoding="utf-8")

    out = collect_kb_provenance(tmp_path, state={}, manifest={}, warnings=[])
    rs = out["recipe_snapshot_reads"]
    assert rs["count"] == 3
    assert rs["hits"] == 2
    assert rs["by_remote"] == {"gbrain": 2, "cortex": 1}
    assert rs["by_resolution"] == {"remote": 1, "remote_miss": 1, "local": 1}
    assert len(rs["tail"]) == 3


def test_collect_kb_provenance_recipe_reads_empty_when_no_audit(
    tmp_path: Path,
):
    from inference_optimizer.breakdown.collectors import collect_kb_provenance
    out = collect_kb_provenance(tmp_path, state={}, manifest={}, warnings=[])
    assert out["recipe_snapshot_reads"]["count"] == 0
    assert out["recipe_snapshot_reads"]["hits"] == 0


# 4. KB_gaps/Gap-16 — CLI flag plumbing reaches _bootstrap_knowledge_plane
def _parse_optimize_args(extra: list[str]) -> argparse.Namespace:
    """Pin the dest-name + default contract the bootstrap reads."""
    from inference_optimizer.cli import _build_parser
    parser = _build_parser()
    return parser.parse_args(["optimize", "--degraded-kb", *extra])


def test_cli_pr_monitor_flags_have_expected_dest_and_defaults():
    """KB_gaps/Gap-16 — PR-monitor flags land under the dest names the bootstrap reads."""
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
    args = _parse_optimize_args([
        "--pr-monitor-url", "https://localhost:8080/v1",
    ])
    assert args.pr_monitor_url == "https://localhost:8080/v1"


def test_cli_pr_monitor_mcp_url_override_reaches_namespace():
    args = _parse_optimize_args([
        "--pr-monitor-mcp-url", "https://localhost:8080/mcp/",
    ])
    assert args.pr_monitor_mcp_url == "https://localhost:8080/mcp/"


def test_cli_pr_feed_window_days_override_reaches_namespace():
    args = _parse_optimize_args(["--pr-feed-window-days", "7"])
    assert args.pr_feed_window_days == 7


def test_cli_args_round_trip_into_bootstrap_knowledge_plane(
    tmp_path: Path, monkeypatch,
):
    """KB_gaps/Gap-16 — argparse ``args`` values propagate into the KnowledgePlane."""
    from inference_optimizer.cli import _bootstrap_knowledge_plane
    from inference_optimizer.orchestrator import pr_monitor as pr_mod

    constructed_urls: list[str] = []

    class _Stub:
        def __init__(self, url: str, enabled: bool):
            self.base_url = url or "https://default.test"
            self.enabled = enabled
            constructed_urls.append(self.base_url)

        def healthz(self) -> bool:
            return True

        def reset_cache(self) -> None:
            pass

    monkeypatch.setattr(
        pr_mod.PRMonitorClient,
        "from_args",
        classmethod(
            lambda cls, **kw: _Stub(
                url=kw.get("url") or "",
                enabled=kw.get("enabled", True),
            ),
        ),
    )

    args = _parse_optimize_args([
        "--pr-monitor-url", "https://my-pr-monitor.example/v1",
        "--pr-monitor-mcp-url", "https://my-pr-monitor.example/mcp/",
        "--pr-feed-window-days", "14",
    ])
    plane = _bootstrap_knowledge_plane(
        args, cortex_client=None, session_dir=tmp_path,
    )
    assert "my-pr-monitor.example" in constructed_urls[-1]
    assert plane.pr_feed_window_days == 14
    assert plane.pr_monitor_mcp_url == "https://my-pr-monitor.example/mcp/"
