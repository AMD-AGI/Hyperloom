# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Turn finished single-lineage Hyperloom sessions into search trees the replay and the priors can learn from.

Before multi-branch, a session was one line of KEEPs. Its ``reports/optimization_journal.json`` holds every settled
decision with its time and throughput, its ``manifest.json`` the task it optimized, and its ``session_breakdown.json``
the noise band it graded under and each baseline measurement. The tree has one work branch running the whole
session: each KEEP is a commit whose cumulative gain is its throughput over the journal's baseline. Such a tree tells
the replay when a line of work stops paying, and nothing about branch caps, which only multi-branch sessions record.
Those sessions measured on one host, so their dispersion is a lower bound for hosted measurement.

Their Experiences come from the ``hyperloom-sbd-v6`` collection mapping, not from here.
"""

from __future__ import annotations

import json
import statistics
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hyperloom.common.timeutil import parse_iso_unix_or_zero
from hyperloom_multibranch.contracts.notebook import INTEGRATION_BRANCH
from hyperloom_multibranch.contracts.notebook.vocab import CommitOrigin, LineageKind
from hyperloom_multibranch.notebook.knowledge.breakdown import FILENAME as BREAKDOWN_FILENAME
from hyperloom_multibranch.notebook.rsi.archive import (
    SINGLE_HOST,
    MeasurementNoise,
    SearchTree,
    TreeArchive,
    TreeBranch,
    TreeCommit,
)
from hyperloom_multibranch.notebook.rsi.similarity import LABEL_WEIGHTS, SIZE_WEIGHTS, TaskConditions

SOURCE = "hyperloom-journal"
WORK_BRANCH = "s1"
JOURNAL_PATH = Path("reports") / "optimization_journal.json"
MANIFEST = "manifest.json"
# The journal's own vocabulary: an adopted change, and the rows that anchor the baseline rather than improve it.
_ADOPTED = "KEEP"
_ANCHOR_KINDS = frozenset({"baseline", "profile"})


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _positive(value: Any) -> float | None:
    number = value if isinstance(value, (int, float)) and not isinstance(value, bool) else None
    return float(number) if number is not None and number > 0 else None


def task_from_manifest(manifest: Mapping[str, Any]) -> TaskConditions:
    """The task conditions of a session manifest's ``task_config``, in the Experience KB identity vocabulary."""

    task = _mapping(manifest.get("task_config") or _mapping(manifest.get("metadata")).get("task_config"))
    architecture = _mapping(task.get("architecture"))
    architectures = architecture.get("architectures") or []
    labels = {
        "model": task.get("model_name"),
        "gpu": task.get("gpu_type"),
        "framework": task.get("framework_name"),
        "model_type": architecture.get("model_type"),
        "architecture": architectures[0] if architectures else architecture.get("model_class"),
        "framework_version": task.get("framework_version"),
        "precision": task.get("precision"),
    }
    return TaskConditions(
        labels={key: str(value) for key, value in labels.items() if key in LABEL_WEIGHTS and value not in (None, "")},
        sizes={key: size for key in SIZE_WEIGHTS if (size := _positive(task.get(key))) is not None},
    )


def noise_from_breakdown(breakdown: Mapping[str, Any]) -> MeasurementNoise:
    """The noise band an old session graded under and the dispersion of its baseline measurements, when it had two."""

    guard = _mapping(_mapping(_mapping(breakdown.get("metadata")).get("grading")).get("tput_guard"))
    values = [
        value
        for event in breakdown.get("timeline") or ()
        if _mapping(event).get("type") == "baseline"
        for action in _mapping(_mapping(event).get("ext")).get("actions") or ()
        if (value := _positive(_mapping(_mapping(action).get("measurement")).get("throughput_tok_s_per_gpu")))
    ]
    dispersion = statistics.stdev(values) / statistics.fmean(values) * 100.0 if len(values) >= 2 else None
    band = guard.get("noise_pct")
    return MeasurementNoise(
        dispersion_pct=dispersion,
        noise_band_pct=float(band) if isinstance(band, (int, float)) and not isinstance(band, bool) else None,
        samples=len(values),
        environment=SINGLE_HOST,
    )


def _read(path: Path) -> Mapping[str, Any]:
    return _mapping(json.loads(path.read_text(encoding="utf-8"))) if path.exists() else {}


def _commits(rows: Iterable[Mapping[str, Any]], start: float, baseline: float) -> tuple[TreeCommit, ...]:
    commits = []
    for row in rows:
        after, when = _positive(row.get("throughput_after")), parse_iso_unix_or_zero(str(row.get("ts") or ""))
        adopted = str(row.get("outcome") or "").upper() == _ADOPTED and row.get("kind") not in _ANCHOR_KINDS
        if adopted and after is not None and when > 0:
            commits.append(TreeCommit(when - start, (after / baseline - 1.0) * 100.0, CommitOrigin.COMMIT))
    return tuple(sorted(commits, key=lambda commit: commit.t_s))


def tree_from_hyperloom_session(session_dir: Path) -> SearchTree:
    """Read one session directory; raises ``FileNotFoundError`` without a journal, ``ValueError`` when it is unusable."""

    session_dir = Path(session_dir)
    journal = _mapping(json.loads((session_dir / JOURNAL_PATH).read_text(encoding="utf-8")))
    rows = [row for row in journal.get("entries") or () if isinstance(row, Mapping)]
    times = sorted(when for row in rows if (when := parse_iso_unix_or_zero(str(row.get("ts") or ""))) > 0)
    baseline = _positive(journal.get("baseline_throughput"))
    if not times or baseline is None:
        raise ValueError(f"{session_dir / JOURNAL_PATH} has no timed decisions or no measured baseline")
    span = times[-1] - times[0]
    commits = _commits(rows, times[0], baseline)
    integration = TreeBranch(INTEGRATION_BRANCH, LineageKind.ROOT, "", 0.0, None, 0.0, 0.0)
    work = TreeBranch(
        WORK_BRANCH,
        LineageKind.OPEN,
        INTEGRATION_BRANCH,
        0.0,
        span,
        0.0,
        span,
        commits,
        tuple(when - times[0] for when in times),
    )
    return SearchTree(
        session_id=str(journal.get("session_id") or session_dir.name),
        source=SOURCE,
        task=task_from_manifest(_read(session_dir / MANIFEST)),
        budget_seconds=span,
        final_gain_pct=max((commit.gain_pct for commit in commits), default=0.0),
        branches=(integration, work),
        noise=noise_from_breakdown(_read(session_dir / BREAKDOWN_FILENAME)),
    )


@dataclass(frozen=True)
class BackfillResult:
    session_dir: Path
    session_id: str
    error: str = ""


def backfill(session_dirs: Iterable[Path], archive: TreeArchive) -> list[BackfillResult]:
    """Archive the tree of every session that has one; a session that cannot be read is reported, not raised."""

    results = []
    for session_dir in session_dirs:
        try:
            tree = tree_from_hyperloom_session(session_dir)
            archive.put(tree)
        except (OSError, KeyError, TypeError, ValueError) as exc:
            results.append(BackfillResult(Path(session_dir), "", f"{type(exc).__name__}: {exc}"))
            continue
        results.append(BackfillResult(Path(session_dir), tree.session_id))
    return results


__all__ = [
    "JOURNAL_PATH",
    "SOURCE",
    "WORK_BRANCH",
    "BackfillResult",
    "backfill",
    "noise_from_breakdown",
    "task_from_manifest",
    "tree_from_hyperloom_session",
]
