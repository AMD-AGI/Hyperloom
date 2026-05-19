"""v0.8 §3.6 / M4 — KnowledgePlane **integration** tests.

KB_gaps/Gap-02 root cause: ``cli._bootstrap_knowledge_plane`` was
defined but never invoked from ``_run_optimize``, leaving the
specialist sub-agent layer (Gap-01) with a ``None`` plane and the
PR Monitor / Cortex readonly surface effectively unused.

This module covers what was missing from M4 PR6 ("call the
bootstrap function + wire it into the Coordinator + EXPLORE
phase-entry warmup + breakdown signal"):

* :meth:`KnowledgePlane.pr_feed_warm_all_domains` batch warmer.
* Coordinator ``_on_enter_explore`` hook that fires on EXPLORE entry.
* ``--no-pr-monitor`` → ``pr_monitor:disabled`` warning in
  ``breakdown.warnings``.
* ``--pr-monitor-url`` unreachable → ``pr_monitor:unreachable:<url>``
  warning.

These tests do **not** mock the cli helpers or the breakdown
collector. They exercise the real wiring with a minimal in-memory
``PRMonitorClient`` double.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest


# ===========================================================================
# Fixtures — minimal PR Monitor / cortex client doubles
# ===========================================================================
@dataclass
class _StubPRClient:
    """Minimal stand-in for :class:`PRMonitorClient`.

    Implements only the surface KnowledgePlane reaches for. Behaviour
    is parameterised by the ``healthz_ok`` + ``enabled`` flags so tests
    can flip between ``--no-pr-monitor``, ``REST unreachable``, and the
    happy path.
    """

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


# ===========================================================================
# 1. KnowledgePlane.pr_feed_warm_all_domains
# ===========================================================================
def test_pr_feed_warm_all_domains_returns_entry_per_known_domain():
    """KB_design §3.6 + KB_gaps/Gap-02 PR 5.4 — every known specialist
    domain must appear in the result map, even when PR Monitor is
    disabled or yields empty PRs."""
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
    # Every known domain has an entry, even if (empty list, warnings).
    for domain in SPECIALIST_DOMAIN_KEYS:
        assert domain in out, f"missing {domain!r} in pr_feed_warm_all_domains"
        prs, warns = out[domain]
        assert isinstance(prs, list)
        assert isinstance(warns, list)


def test_pr_feed_warm_all_domains_stashes_warnings():
    """Aggregated warnings land on ``last_warnings`` so the breakdown
    collector can surface them in one pass."""
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
    # With pr_monitor=None, every domain produces a ``pr_monitor:disabled``
    # warning (KnowledgePlane.pr_feed_warm fast-path).
    assert plane.last_warnings, "warnings should be aggregated"
    assert any("pr_monitor:disabled" in w for w in plane.last_warnings)


def test_pr_feed_warm_all_domains_isolates_per_domain_failures():
    """A failure on one domain must NOT abort the rest of the batch
    (defense in depth: KB_design §3.14 R-03 fail-soft contract)."""
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
        if domain == "framework_specialist":
            raise RuntimeError("synthetic PR monitor outage")
        return real_pr_feed_warm(domain, **kwargs)

    plane.pr_feed_warm = _flaky_pr_feed_warm  # type: ignore[assignment]
    out = plane.pr_feed_warm_all_domains()

    # Every domain still got a turn.
    assert set(call_log) == set(SPECIALIST_DOMAIN_KEYS)
    # The poisoned domain has a non-empty warnings entry.
    prs, warns = out["framework_specialist"]
    assert prs == []
    assert any("framework_specialist" in w for w in warns)


# ===========================================================================
# 2. Coordinator EXPLORE phase entry auto-warm
# ===========================================================================
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
        return {"framework_specialist": ([], [])}

    def pr_feed_warm(self, domain, **_kw):
        return [], []


@pytest.mark.asyncio
async def test_on_enter_explore_warms_pr_feed(tmp_path: Path):
    """Coordinator's EXPLORE-entry hook must call
    ``KnowledgePlane.pr_feed_warm_all_domains`` exactly once per
    transition, regardless of where it came from (PRELUDE or
    resume-inferred)."""
    from inference_optimizer.orchestrator.coordinator import Coordinator

    coord = Coordinator.__new__(Coordinator)
    plane = _FakePlane()
    coord.knowledge_plane = plane
    # Patch in just enough infrastructure for the dispatcher methods
    # this test touches. The hook itself is pure dispatch + 1 plane
    # call.

    await coord._on_enter_explore(from_phase="PRELUDE")
    assert plane.warm_calls == 1


@pytest.mark.asyncio
async def test_on_enter_explore_graceful_when_plane_is_none(tmp_path: Path):
    """``--no-cortex`` runs have plane=None; the hook must short-circuit
    instead of raising."""
    from inference_optimizer.orchestrator.coordinator import Coordinator

    coord = Coordinator.__new__(Coordinator)
    coord.knowledge_plane = None
    # Should not raise:
    await coord._on_enter_explore(from_phase="PRELUDE")


@pytest.mark.asyncio
async def test_on_enter_explore_swallows_warmup_exceptions(tmp_path: Path):
    """If ``pr_feed_warm_all_domains`` raises (e.g. network gone),
    the hook must log + continue. EXPLORE phase still functions; the
    per-dispatch warmup in ``_handle_delegate`` is the second line of
    defence."""
    from inference_optimizer.orchestrator.coordinator import Coordinator

    class _BadPlane:
        pr_monitor_enabled = True
        cortex_enabled = True

        def pr_feed_warm_all_domains(self, **_kw):
            raise RuntimeError("synthetic warmup outage")

    coord = Coordinator.__new__(Coordinator)
    coord.knowledge_plane = _BadPlane()
    # Should not raise:
    await coord._on_enter_explore(from_phase="PRELUDE")


@pytest.mark.asyncio
async def test_on_phase_entered_only_explore_fires_warmup(tmp_path: Path):
    """The dispatcher table currently wires only EXPLORE. Transitions
    to KERNEL / SWEEP / CLOSE must not call into the plane (Gap-04/05/06
    will add their own branches)."""
    from inference_optimizer.orchestrator.coordinator import Coordinator

    coord = Coordinator.__new__(Coordinator)
    plane = _FakePlane()
    coord.knowledge_plane = plane

    await coord._on_phase_entered(from_phase="PRELUDE", to_phase="KERNEL")
    await coord._on_phase_entered(from_phase="PRELUDE", to_phase="SWEEP")
    await coord._on_phase_entered(from_phase="PRELUDE", to_phase="CLOSE")
    assert plane.warm_calls == 0

    await coord._on_phase_entered(from_phase="PRELUDE", to_phase="EXPLORE")
    assert plane.warm_calls == 1


# ===========================================================================
# 3. _bootstrap_knowledge_plane status marker + breakdown.warnings wiring
# ===========================================================================
def _build_args(**overrides) -> argparse.Namespace:
    base = dict(
        pr_monitor_enabled=True,
        pr_monitor_url=None,
        pr_monitor_mcp_url=None,
        pr_feed_window_days=30,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def test_bootstrap_writes_status_marker_when_disabled(tmp_path: Path):
    """``--no-pr-monitor`` path: marker must declare enabled=False so
    the breakdown collector surfaces ``pr_monitor:disabled``."""
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


def test_bootstrap_writes_status_marker_when_unreachable(
    tmp_path: Path, monkeypatch,
):
    """PR Monitor REST returns non-200: marker reflects reachable=False
    but enabled=True (so downstream knows the operator wanted PR
    Monitor on, just can't reach it)."""
    from inference_optimizer.cli import _bootstrap_knowledge_plane
    from inference_optimizer.session_paths import pr_monitor_status_json
    from inference_optimizer.orchestrator import pr_monitor as pr_mod

    class _UnreachableClient:
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
            lambda cls, **kw: _UnreachableClient(
                url=kw.get("url") or "", enabled=kw.get("enabled", True),
            ),
        ),
    )
    args = _build_args(
        pr_monitor_enabled=True,
        pr_monitor_url="https://pr-monitor.test",
    )
    _bootstrap_knowledge_plane(
        args, cortex_client=None, session_dir=tmp_path,
    )
    marker = pr_monitor_status_json(tmp_path)
    payload = json.loads(marker.read_text())
    assert payload["enabled"] is True
    assert payload["reachable"] is False
    assert "pr-monitor.test" in str(payload.get("url"))


def test_collect_kb_provenance_surfaces_pr_monitor_disabled_warning(
    tmp_path: Path,
):
    """Roundtrip: the breakdown collector must emit
    ``pr_monitor:disabled`` to ``warnings`` when the marker says so."""
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
        "status_text": "disabled (--no-pr-monitor)",
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
    # Either bare ``pr_monitor:unreachable`` or with the URL appended is
    # acceptable; we assert the prefix.
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
    """Resume path without the marker (e.g. v0.6 session resumed pre-Gap-02)
    must not produce spurious warnings — absence ≠ failure."""
    from inference_optimizer.breakdown.collectors import collect_kb_provenance
    # No marker written.
    warnings_list: list = []
    collect_kb_provenance(
        tmp_path,
        state={},
        manifest={},
        warnings=warnings_list,
    )
    assert not any(w.startswith("pr_monitor:") for w in warnings_list)
