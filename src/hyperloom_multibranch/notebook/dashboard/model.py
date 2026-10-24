# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What the dashboard shows, as JSON-ready data built from one notebook snapshot.

* ``branches``: each branch's progress, its head's gain over the baseline, and its frontier score when the progress
  rows carry one (pass :class:`FrontierScoring`'s rows to show them);
* ``stranded``: the branches that lead integration by more than the merge bar while commits of theirs failed to merge,
  with those commits (:func:`~hyperloom_multibranch.notebook.rsi.stranded.stranded_leads`);
* ``spend``: the session's GPU hours and LLM dollars so far, its specialists', judge's and resolver's together, and the
  tokens of the LLM calls whose model the price sheet does not list, which the dollars leave out;
* ``in_flight``: tests requested and neither measured nor withdrawn yet, and the claims held now;
* ``directions``: every direction an intent named, the states it was tried on, its best gain, the running branches
  whose head carries one of its commits and the branches holding a claim on it;
* ``hypotheses``: every hypothesis a specialist logged, at its latest version;
* ``queue``: the toolbox's queue lengths and latest cold start, when the caller passes ``Toolbox.queue_state()``;
* ``lineage`` and ``splits``: every commit, every branch, and the branches split off for a correctness fix;
* ``sandboxes``: per measurement environment, the tests run there and their GPU time;
* ``environment_changes``: where a node's image, GPU model, power cap or partition mode moved between its results;
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
from hyperloom_multibranch.contracts.toolbox import QueueState
from hyperloom_multibranch.notebook.codec import encode
from hyperloom_multibranch.notebook.dashboard.page import render_html
from hyperloom_multibranch.notebook.knowledge.breakdown import measurement_usage, session_spend
from hyperloom_multibranch.notebook.rsi.analysis import direction_occupancy, hypothesis_ledger
from hyperloom_multibranch.notebook.rsi.stranded import stranded_leads

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
            "judged_since_commit": row.judged_since_commit,
            "gpu_hours": row.gpu_seconds / 3600.0,
            "llm_usd": row.llm_usd,
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
                "purpose": record.request.purpose.value,
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


def _directions(snapshot: NotebookSnapshot) -> list[dict[str, Any]]:
    return [
        {
            "target": item.target,
            "mechanism": item.mechanism,
            "layer": item.layer.value,
            "tried_on": list(item.tried_on),
            "outcomes": list(item.outcomes),
            "best_gain_pct": item.best_gain_pct,
            "committed_on": list(item.committed_on),
            "claimed_by": list(item.claimed_by),
        }
        for item in direction_occupancy(snapshot)
    ]


def _hypotheses(snapshot: NotebookSnapshot) -> list[dict[str, Any]]:
    return [
        {
            "branch": row.branch,
            "status": row.status.value,
            "statement": row.statement,
            "evidence": list(row.evidence),
            "updated_at": _iso(row.updated_at),
            "entry_id": row.latest_entry,
        }
        for row in hypothesis_ledger(snapshot)
    ]


def _environment_changes(snapshot: NotebookSnapshot) -> list[dict[str, Any]]:
    return [
        {
            "node": change.node,
            "request_id": change.request_id,
            "at": _iso(change.at),
            "changed": {name: list(values) for name, values in change.changed.items()},
        }
        for change in snapshot.environment_changes
    ]


def _queue(queue: QueueState) -> dict[str, Any]:
    # String keys only: dashboard.json must read back equal to what was built.
    return {
        "queued": queue.queued,
        "running": queue.running,
        "queued_by_priority": {str(priority): count for priority, count in sorted(queue.queued_by_priority.items())},
        "queued_by_branch": dict(sorted(queue.queued_by_branch.items())),
        "running_by_branch": dict(sorted(queue.running_by_branch.items())),
        "cold_start_seconds": queue.cold_start_seconds,
    }


def build_dashboard(
    snapshot: NotebookSnapshot,
    progress: Sequence[BranchProgress] | None = None,
    *,
    metric: str = "",
    queue: QueueState | None = None,
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
        "stranded": encode(stranded_leads(snapshot, metric)),
        "spend": session_spend(snapshot),
        "in_flight": _in_flight(snapshot),
        "directions": _directions(snapshot),
        "hypotheses": _hypotheses(snapshot),
        "queue": _queue(queue) if queue is not None else None,
        "lineage": encode(snapshot.lineage),
        "splits": [
            {"branch": node.branch, "from_branch": node.from_branch, "start_commit": node.start_commit}
            for node in snapshot.lineage.branches
            if node.origin is LineageKind.SPLIT
        ],
        "sandboxes": encode(measurement_usage(snapshot.records)),
        "environment_changes": _environment_changes(snapshot),
        "quarantined": encode(snapshot.quarantined),
    }


@dataclass(frozen=True)
class DashboardFiles:
    json: Path
    html: Path


def write_dashboard(directory: Path, dashboard: Mapping[str, Any]) -> DashboardFiles:
    """Write the dashboard data and its page into ``directory``, each atomically."""

    files = DashboardFiles(directory / DASHBOARD_JSON, directory / DASHBOARD_HTML)
    atomic_write_json(files.json, dict(dashboard))
    atomic_write_text(files.html, render_html(dashboard))
    return files


__all__ = ["DASHBOARD_HTML", "DASHBOARD_JSON", "SCHEMA_VERSION", "DashboardFiles", "build_dashboard", "write_dashboard"]
