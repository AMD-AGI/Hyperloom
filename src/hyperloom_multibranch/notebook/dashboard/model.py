# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What the dashboard shows, as JSON-ready data built from one notebook snapshot.

* ``branches``: each branch's progress, its head's gain over the baseline, and its frontier score when the progress
  rows carry one (pass :class:`FrontierScoring`'s rows to show them);
* ``in_flight``: tests requested and neither measured nor withdrawn yet, and the claims held now. The toolbox's own
  queue, once it reports one, is passed in as ``queue`` and shown as it is;
* ``lineage`` and ``splits``: every commit, every branch, and the branches split off for a correctness fix;
* ``sandboxes``: per measurement environment, the tests run there and their GPU time;
* ``quarantined``: the nodes and GPU sets out of service.

The run entry point decides when to rebuild it; :func:`write_dashboard` writes ``dashboard.json`` and
``dashboard.html`` atomically, so a reader never sees half a page.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hyperloom.common.io import atomic_write_json, atomic_write_text
from hyperloom_multibranch.contracts.notebook.records import BranchProgress, NotebookSnapshot
from hyperloom_multibranch.contracts.notebook.vocab import ClaimStatus, LineageKind
from hyperloom_multibranch.notebook.codec import encode
from hyperloom_multibranch.notebook.dashboard.page import render_html
from hyperloom_multibranch.notebook.knowledge.breakdown import measurement_usage

SCHEMA_VERSION = "multibranch-dashboard-1"
DASHBOARD_JSON = "dashboard.json"
DASHBOARD_HTML = "dashboard.html"


def _iso(at: float) -> str:
    return datetime.fromtimestamp(at, timezone.utc).isoformat()


def _gain(values: Mapping[str, float], baseline: Mapping[str, float], metric: str) -> float | None:
    value, anchor = values.get(metric), baseline.get(metric)
    return (value / anchor - 1.0) * 100.0 if value and anchor else None


def _branches(snapshot: NotebookSnapshot, rows: Sequence[BranchProgress], metric: str) -> list[dict[str, Any]]:
    baseline = snapshot.baseline.values if snapshot.baseline is not None else {}
    return [
        {
            "branch": row.branch,
            "running": row.running,
            "start_commit": row.start_commit,
            "head_commit": row.head_commit,
            "tests": row.tests,
            "commits": row.commits,
            "gpu_hours": row.gpu_seconds / 3600.0,
            "head_value": dict(row.head_value),
            "gain_pct": _gain(row.head_value, baseline, metric),
            "frontier": encode(row.frontier),
        }
        for row in rows
    ]


def _in_flight(snapshot: NotebookSnapshot) -> dict[str, Any]:
    waiting = [record for record in snapshot.records if record.in_flight]
    return {
        "tests": [
            {
                "request_id": record.request_id,
                "kind": record.request.kind.value,
                "branch": record.request.branch,
                "parent_commit": record.request.parent_commit,
                "purpose": record.request.purpose,
                "requested_at": _iso(record.requested_at),
                "waiting_s": max(0.0, snapshot.taken_at - record.requested_at),
            }
            for record in waiting
        ],
        "claims": [
            {
                "branch": claim.holder,
                "parent_commit": claim.key.parent_commit,
                "target": claim.intent.target,
                "mechanism": claim.intent.mechanism,
                "request_id": claim.request_id,
                "expires_in_s": max(0.0, claim.expires_at - snapshot.taken_at),
            }
            for claim in snapshot.claims
            if claim.status is ClaimStatus.HELD
        ],
    }


def build_dashboard(
    snapshot: NotebookSnapshot,
    progress: Sequence[BranchProgress] | None = None,
    *,
    metric: str = "",
    queue: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The dashboard of ``snapshot``; ``progress`` defaults to the snapshot's rows, ``metric`` names the gains."""

    rows = snapshot.progress if progress is None else progress
    profile, baseline = snapshot.profile, snapshot.baseline
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": _iso(snapshot.taken_at),
        "metric": metric,
        "session": encode(profile),
        "baseline": encode(baseline),
        "branches": _branches(snapshot, rows, metric),
        "in_flight": _in_flight(snapshot),
        "queue": dict(queue) if queue is not None else None,
        "lineage": encode(snapshot.lineage),
        "splits": [
            {"branch": node.branch, "from_branch": node.from_branch, "start_commit": node.start_commit}
            for node in snapshot.lineage.branches
            if node.origin is LineageKind.SPLIT
        ],
        "sandboxes": encode(measurement_usage(snapshot.records)),
        "quarantined": encode(snapshot.quarantined),
    }


@dataclass(frozen=True)
class DashboardFiles:
    json: Path
    html: Path


def write_dashboard(directory: Path, dashboard: Mapping[str, Any]) -> DashboardFiles:
    """Write the dashboard data and its page into ``directory``, each atomically."""

    files = DashboardFiles(directory / DASHBOARD_JSON, directory / DASHBOARD_HTML)
    atomic_write_json(files.json, dict(dashboard), indent=2, sort_keys=True, make_parents=True)
    atomic_write_text(files.html, render_html(dashboard), make_parents=True)
    return files


__all__ = ["DASHBOARD_HTML", "DASHBOARD_JSON", "SCHEMA_VERSION", "DashboardFiles", "build_dashboard", "write_dashboard"]
