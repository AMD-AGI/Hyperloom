# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Archived search trees: what each past session's branches did over time, what they cost, how noisy it measured.

A tree keeps what the replay, the frontier priors and the dispersion priors need: when each branch opened and stopped
and why it opened, when its tests were requested, when its commits landed and the cumulative gain they reached, the GPU
time it spent, the session's task conditions, its baseline's dispersion and where it measured. Trees are written once,
under ``<archive>/trees/<session_id>.json``; the archive is a shared folder the deployment names.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from hyperloom_multibranch.notebook import NotRecorded
from hyperloom_multibranch.notebook.codec import decode, encode
from hyperloom_multibranch.notebook.local.files import encode_json, publish_once
from hyperloom_multibranch.notebook.records import BranchNode, LineageView, NotebookSnapshot, TestRecord
from hyperloom_multibranch.notebook.vocab import CommitOrigin, LineageKind
from hyperloom_multibranch.rsi.similarity import TaskConditions, task_conditions, task_similarity

log = logging.getLogger(__name__)

TREE_FORMAT = "multibranch.search-tree.v1"
SOURCE_NOTEBOOK = "multibranch-notebook"
ARCHIVE_ENV = "HYPERLOOM_RSI_ARCHIVE"
_FILE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:@-]{0,127}")


@dataclass(frozen=True)
class TreeCommit:
    t_s: float
    gain_pct: float
    origin: CommitOrigin


@dataclass(frozen=True)
class TreeBranch:
    """One branch on the session clock (seconds since the session started); gains are over the session's baseline."""

    name: str
    origin: LineageKind
    parent_branch: str
    opened_s: float
    closed_s: float | None
    fork_gain_pct: float
    gpu_seconds: float
    commits: tuple[TreeCommit, ...] = ()
    test_times: tuple[float, ...] = ()

    def gain_at(self, t_s: float) -> float:
        """The cumulative gain of the branch's latest commit at ``t_s``, or the gain it forked at."""

        landed = [commit.gain_pct for commit in self.commits if commit.t_s <= t_s]
        return landed[-1] if landed else self.fork_gain_pct

    @property
    def final_gain_pct(self) -> float:
        return self.gain_at(math.inf)

    def tests_by(self, t_s: float) -> int:
        return sum(1 for when in self.test_times if when <= t_s)


@dataclass(frozen=True)
class MeasurementNoise:
    """How noisy the session measured its throughput, and where.

    ``dispersion_pct`` is the baseline's dispersion over ``samples`` measurements; ``noise_band_pct`` the band the
    session graded under, when it recorded one; ``environment`` the measurement backends it used.
    """

    dispersion_pct: float | None = None
    noise_band_pct: float | None = None
    samples: int = 0
    environment: str = ""


@dataclass(frozen=True)
class SearchTree:
    """One past session as the RSI replays it."""

    session_id: str
    source: str
    task: TaskConditions
    budget_seconds: float
    final_gain_pct: float
    branches: tuple[TreeBranch, ...]
    noise: MeasurementNoise = field(default_factory=MeasurementNoise)
    policy_versions: Mapping[str, str] = field(default_factory=dict)


def tree_document(tree: SearchTree) -> dict[str, object]:
    return {"format": TREE_FORMAT, "tree": encode(tree)}


def tree_from_document(document: Mapping[str, object]) -> SearchTree:
    """The tree :func:`tree_document` wrote; raises ``ValueError`` or ``TypeError`` on anything else."""

    if document.get("format") != TREE_FORMAT:
        raise ValueError(f"unsupported search tree format {document.get('format')!r}")
    return decode(SearchTree, document.get("tree"))


def session_start(snapshot: NotebookSnapshot) -> float:
    """When the session started: its earliest recorded commit, branch or request."""

    times = [
        *(commit.at for commit in snapshot.lineage.commits),
        *(branch.opened_at for branch in snapshot.lineage.branches),
        *(record.requested_at for record in snapshot.records),
    ]
    return min((at for at in times if at > 0), default=snapshot.taken_at)


def _parent_branch(node: BranchNode, view: LineageView) -> str:
    if node.origin is LineageKind.ROOT:
        return ""
    start = view.commit(node.start_commit)
    return node.from_branch or (start.branch if start is not None else "")


def _environment(snapshot: NotebookSnapshot) -> str:
    baseline = snapshot.baseline
    measured = [record for record in snapshot.records if record.result is not None]
    anchored = [record for record in measured if baseline is not None and record.request_id in baseline.request_ids]
    backends = {record.result.sandbox.backend or "unknown" for record in anchored or measured if record.result}
    return "+".join(sorted(backends))


def tree_from_snapshot(
    snapshot: NotebookSnapshot,
    metric: str,
    *,
    budget_seconds: float | None = None,
    bottlenecks: Mapping[str, float] | None = None,
) -> SearchTree:
    """The session's search tree, its gains read off ``metric``; the budget defaults to the time the notebook spans."""

    profile, baseline = snapshot.profile, snapshot.baseline
    if profile is None or baseline is None:
        raise NotRecorded("a search tree needs the session profile and the baseline")
    anchor = baseline.values.get(metric)
    if not anchor:
        raise ValueError(f"the baseline records no {metric}")
    start = session_start(snapshot)
    view = snapshot.lineage

    def gain(commit_id: str) -> float:
        value = snapshot.value_of(commit_id).get(metric)
        return (value / anchor - 1.0) * 100.0 if value else 0.0

    tests: dict[str, list[TestRecord]] = defaultdict(list)
    for record in snapshot.records:
        tests[record.request.branch].append(record)
    branches = tuple(
        TreeBranch(
            name=node.branch,
            origin=node.origin,
            parent_branch=_parent_branch(node, view),
            opened_s=max(0.0, node.opened_at - start),
            closed_s=None if node.stopped_at is None else node.stopped_at - start,
            fork_gain_pct=gain(node.start_commit),
            gpu_seconds=sum(record.result.gpu_seconds for record in tests[node.branch] if record.result is not None),
            commits=tuple(
                TreeCommit(commit.at - start, gain(commit.commit_id), commit.origin)
                for commit in view.commits
                if commit.branch == node.branch and commit.origin is not CommitOrigin.ROOT
            ),
            test_times=tuple(sorted(record.requested_at - start for record in tests[node.branch])),
        )
        for node in view.branches
    )
    return SearchTree(
        session_id=profile.session_id,
        source=SOURCE_NOTEBOOK,
        task=task_conditions(profile, bottlenecks),
        budget_seconds=max(0.0, snapshot.taken_at - start) if budget_seconds is None else budget_seconds,
        final_gain_pct=max((gain(commit.commit_id) for commit in view.commits), default=0.0),
        branches=branches,
        noise=MeasurementNoise(
            dispersion_pct=baseline.dispersion_pct.get(metric),
            samples=len(baseline.request_ids),
            environment=_environment(snapshot),
        ),
        policy_versions=dict(snapshot.fingerprints[-1].policy_versions) if snapshot.fingerprints else {},
    )


@dataclass(frozen=True)
class WeightedTree:
    tree: SearchTree
    weight: float


def similar_trees(history: Sequence[SearchTree], task: TaskConditions) -> tuple[WeightedTree, ...]:
    """The archived trees that share a condition with ``task``, most similar first."""

    weighted = (WeightedTree(tree, task_similarity(task, tree.task)) for tree in history)
    return tuple(sorted((item for item in weighted if item.weight > 0), key=lambda item: -item.weight))


class TreeArchive:
    """The shared folder of archived search trees."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root) / "trees"

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> TreeArchive | None:
        """The archive :data:`ARCHIVE_ENV` names, or ``None`` when it names none."""

        root = str((os.environ if env is None else env).get(ARCHIVE_ENV) or "").strip()
        return cls(Path(root)) if root else None

    def put(self, tree: SearchTree) -> None:
        """Archive ``tree`` once; archiving it again is a no-op, another tree under its session id is an error."""

        if not _FILE_NAME.fullmatch(tree.session_id):
            raise ValueError(f"session id {tree.session_id!r} cannot name an archived tree")
        path = self.root / f"{tree.session_id}.json"
        if publish_once(path, encode_json(tree_document(tree))):
            return
        if tree_from_document(json.loads(path.read_text(encoding="utf-8"))) != tree:
            raise FileExistsError(f"{path} already holds another tree for session {tree.session_id}")

    def load(self) -> list[SearchTree]:
        """Every readable archived tree; one this version cannot read is skipped with a warning."""

        trees = []
        for path in sorted(self.root.glob("*.json")):
            try:
                trees.append(tree_from_document(json.loads(path.read_text(encoding="utf-8"))))
            except (OSError, KeyError, TypeError, ValueError) as exc:
                log.warning("skipping unreadable search tree %s: %s", path, exc)
        return trees


__all__ = [
    "ARCHIVE_ENV",
    "SOURCE_NOTEBOOK",
    "TREE_FORMAT",
    "MeasurementNoise",
    "SearchTree",
    "TreeArchive",
    "TreeBranch",
    "TreeCommit",
    "WeightedTree",
    "session_start",
    "similar_trees",
    "tree_document",
    "tree_from_document",
    "tree_from_snapshot",
]
