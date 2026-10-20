# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Archived search trees: what each past session's branches did over time, what they cost, how noisy it measured.

A tree keeps what the replay, the frontier priors and their calibration, the dispersion priors and the history digest
need: when each branch opened and stopped and why, when its tests were requested (its kernel-layer ones apart), the
directions they tried, when its commits landed and the cumulative gain they reached, the GPU time it spent, the
session's task conditions, its roofline bound, its baseline's dispersion and where it measured. A tree an older version
wrote, or one backfilled from a session that recorded no intents, reads with no stop reasons, no directions, no kernel
test times and no roofline bound. Trees are written once,
under ``<archive>/trees/<session_id>.json``; the archive is a shared folder the deployment names. CLOSE archives its
session with :func:`archive_session`, which says why when it archives nothing.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

from hyperloom_multibranch.contracts.notebook import INTEGRATION_BRANCH, NotRecorded
from hyperloom_multibranch.contracts.notebook.peers import Intent, IntentLayer, SandboxFacts
from hyperloom_multibranch.contracts.notebook.records import BranchNode, LineageView, NotebookSnapshot, TestRecord
from hyperloom_multibranch.contracts.notebook.vocab import COMMITTING, CommitOrigin, Conclusion, LineageKind
from hyperloom_multibranch.notebook.codec import decode, encode
from hyperloom_multibranch.notebook.local.files import encode_json, publish_once
from hyperloom_multibranch.notebook.rsi.similarity import TaskConditions, task_conditions, task_similarity

log = logging.getLogger(__name__)

TREE_FORMAT = "multibranch.search-tree.v1"
SOURCE_NOTEBOOK = "multibranch-notebook"
ARCHIVE_ENV = "HYPERLOOM_RSI_ARCHIVE"
SINGLE_HOST = "local"
_FILE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:@-]{0,127}")


@dataclass(frozen=True)
class TreeCommit:
    t_s: float
    gain_pct: float
    origin: CommitOrigin


@dataclass(frozen=True)
class TreeDirection:
    """One intent a branch tested: how many tests it took, how many committed, and the best gain any of them was judged
    at over its parent, among those that kept accuracy; ``None`` when none was."""

    target: str
    mechanism: str
    layer: IntentLayer | None
    tests: int
    commits: int = 0
    best_gain_pct: float | None = None


@dataclass(frozen=True)
class TreeBranch:
    """One branch on the session clock (seconds since the session started); gains are over the session's baseline.

    ``stop_reason`` is the reason the scheduler gave when it stopped the branch; ``directions`` are the intents its tests
    tried, in the order it first tried them; ``kernel_test_times`` when its kernel-layer tests were requested, which the
    frontier calibration reads the observation period off. ``measured_times`` and ``judged_times`` are when its tests
    were measured and judged; ``None`` in a tree archived before they were kept.
    """

    name: str
    origin: LineageKind
    parent_branch: str
    opened_s: float
    closed_s: float | None
    fork_gain_pct: float
    gpu_seconds: float
    commits: tuple[TreeCommit, ...] = ()
    test_times: tuple[float, ...] = ()
    stop_reason: str = ""
    directions: tuple[TreeDirection, ...] = ()
    kernel_test_times: tuple[float, ...] = ()
    measured_times: tuple[float, ...] | None = None
    judged_times: tuple[float, ...] | None = None

    @property
    def spent_times(self) -> tuple[float, ...]:
        """When the branch's tests were measured; their request times in a tree archived before that was kept."""

        return self.measured_times if self.measured_times is not None else self.test_times

    def judged_by(self, t_s: float) -> int | None:
        """How many of the branch's tests were judged by ``t_s``; ``None`` in a tree archived before that was kept."""

        return None if self.judged_times is None else sum(1 for when in self.judged_times if when <= t_s)

    def gain_at(self, t_s: float) -> float:
        """The cumulative gain of the branch's latest commit at ``t_s``, or the gain it forked at."""

        landed = [commit.gain_pct for commit in self.commits if commit.t_s <= t_s]
        return landed[-1] if landed else self.fork_gain_pct

    @property
    def final_gain_pct(self) -> float:
        return self.gain_at(math.inf)

    def tests_by(self, t_s: float) -> int:
        return sum(1 for when in self.test_times if when <= t_s)

    def as_of(self, t_s: float) -> TreeBranch:
        """The branch as it stood at ``t_s``: its commits and tests by then, open unless it had stopped by then.

        The GPU time is the share of the branch's tests requested by then, the archive keeping only the total; the
        directions, which the archive does not date, are left out.
        """

        tests = tuple(when for when in self.test_times if when <= t_s)
        return replace(
            self,
            closed_s=self.closed_s if self.closed_s is not None and self.closed_s <= t_s else None,
            gpu_seconds=self.gpu_seconds * len(tests) / len(self.test_times) if self.test_times else 0.0,
            commits=tuple(commit for commit in self.commits if commit.t_s <= t_s),
            test_times=tests,
            directions=(),
            kernel_test_times=tuple(when for when in self.kernel_test_times if when <= t_s),
            measured_times=_by(self.measured_times, t_s),
            judged_times=_by(self.judged_times, t_s),
        )


def _by(times: tuple[float, ...] | None, t_s: float) -> tuple[float, ...] | None:
    return None if times is None else tuple(when for when in times if when <= t_s)


@dataclass(frozen=True)
class MeasurementNoise:
    """How noisy the session measured its throughput, and where.

    ``dispersion_pct`` is the baseline's dispersion over ``samples`` measurements; ``noise_band_pct`` the band the
    session graded under, when it recorded one; ``environment`` where it measured: :data:`SINGLE_HOST` when every
    baseline measurement ran on one node, whatever its backend, else the backends it used, joined by ``+``.
    """

    dispersion_pct: float | None = None
    noise_band_pct: float | None = None
    samples: int = 0
    environment: str = ""


@dataclass(frozen=True)
class SearchTree:
    """One past session as the RSI replays it.

    ``roofline_ceiling_pct`` is the roofline bound PRELUDE recorded for the gain metric, as a gain over the baseline;
    ``None`` when it recorded none, or the tree is older. ``longest_root_test_seconds`` is the longest of the root tests
    the baseline anchored on, start to finish, which paces the frontier's window; ``None`` likewise.
    """

    session_id: str
    source: str
    task: TaskConditions
    budget_seconds: float
    final_gain_pct: float
    branches: tuple[TreeBranch, ...]
    noise: MeasurementNoise = field(default_factory=MeasurementNoise)
    policy_versions: Mapping[str, str] = field(default_factory=dict)
    roofline_ceiling_pct: float | None = None
    longest_root_test_seconds: float | None = None


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


def longest_root_test_seconds(snapshot: NotebookSnapshot) -> float | None:
    """The longest start-to-finish time of the root tests the baseline anchored on, the one PRELUDE logs; ``None``
    before a baseline or when none of them recorded both times."""

    baseline = snapshot.baseline
    anchored = set(baseline.request_ids) if baseline is not None else set()
    walls = [
        record.result.finished_at - record.result.started_at
        for record in snapshot.records
        if record.request_id in anchored
        and record.result is not None
        and record.result.started_at is not None
        and record.result.finished_at is not None
    ]
    return max(walls, default=None)


def _measured_times(records: Sequence[TestRecord], start: float) -> tuple[float, ...]:
    return tuple(sorted(item.measured_at - start for item in records if item.measured and item.measured_at is not None))


def _judged_times(records: Sequence[TestRecord], start: float) -> tuple[float, ...]:
    return tuple(sorted(item.judged_at - start for item in records if item.judged and item.judged_at is not None))


def _parent_branch(node: BranchNode, view: LineageView) -> str:
    if node.origin is LineageKind.ROOT:
        return ""
    start = view.commit(node.start_commit)
    return node.from_branch or (start.branch if start is not None else "")


def _sandbox(record: TestRecord) -> SandboxFacts | None:
    return record.result.sandbox if record.result is not None else None


def _backend(record: TestRecord) -> str:
    sandbox = _sandbox(record)
    return (sandbox.backend if sandbox is not None else "") or "unknown"


def _environment(snapshot: NotebookSnapshot) -> str:
    baseline = snapshot.baseline
    measured = [record for record in snapshot.records if record.measured]
    anchored = [record for record in measured if baseline is not None and record.request_id in baseline.request_ids]
    sampled = anchored or measured
    # A hosted backend run on one machine (MB1) still measured on a single host.
    nodes = {sandbox.node_name for sandbox in map(_sandbox, sampled) if sandbox is not None and sandbox.node_name}
    if len(nodes) == 1:
        return SINGLE_HOST
    return "+".join(sorted({_backend(record) for record in sampled}))


def _kept_gain(record: TestRecord) -> float | None:
    judged = record.judgement
    if judged is None or not judged.accuracy_ok or record.conclusion is Conclusion.ACCURACY_FAILED:
        return None
    return judged.gain_pct


def _directions(records: Sequence[TestRecord], intents: Mapping[str, Intent]) -> tuple[TreeDirection, ...]:
    tried: dict[str, list[TestRecord]] = {}
    for record in sorted(records, key=lambda item: item.requested_at):
        if record.withdrawn_at is None and record.request.intent_fingerprint in intents:
            tried.setdefault(record.request.intent_fingerprint, []).append(record)
    directions = []
    for fingerprint, tests in tried.items():
        intent = intents[fingerprint]
        gains = [gain for gain in map(_kept_gain, tests) if gain is not None]
        directions.append(
            TreeDirection(
                target=intent.target,
                mechanism=intent.mechanism,
                layer=intent.layer,
                tests=len(tests),
                commits=sum(1 for record in tests if record.conclusion in COMMITTING),
                best_gain_pct=max(gains) if gains else None,
            )
        )
    return tuple(directions)


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
    kernel = {fingerprint for fingerprint, intent in snapshot.intents.items() if intent.layer is IntentLayer.KERNEL}
    ceiling = profile.roofline_ceiling.get(metric)
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
            stop_reason=node.stop_reason or "",
            directions=() if node.branch == INTEGRATION_BRANCH else _directions(tests[node.branch], snapshot.intents),
            kernel_test_times=tuple(
                sorted(
                    record.requested_at - start
                    for record in tests[node.branch]
                    if record.request.intent_fingerprint in kernel
                )
            ),
            measured_times=_measured_times(tests[node.branch], start),
            judged_times=_judged_times(tests[node.branch], start),
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
        roofline_ceiling_pct=(ceiling / anchor - 1.0) * 100.0 if ceiling else None,
        longest_root_test_seconds=longest_root_test_seconds(snapshot),
    )


@dataclass(frozen=True)
class WeightedTree:
    tree: SearchTree
    weight: float


def similar_trees(
    history: Sequence[SearchTree], task: TaskConditions, *, minimum: float = 0.0
) -> tuple[WeightedTree, ...]:
    """The archived trees that share a condition with ``task``, at least ``minimum`` similar, most similar first."""

    weighted = (WeightedTree(tree, task_similarity(task, tree.task)) for tree in history)
    kept = (item for item in weighted if item.weight > 0 and item.weight >= minimum)
    return tuple(sorted(kept, key=lambda item: -item.weight))


class TreeConflict(FileExistsError):
    """The archive already holds another tree under the session's id."""


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
            raise TreeConflict(f"{path} already holds another tree for session {tree.session_id}")

    def load(self) -> list[SearchTree]:
        """Every readable archived tree; one this version cannot read is skipped with a warning."""

        trees = []
        for path in sorted(self.root.glob("*.json")):
            try:
                trees.append(tree_from_document(json.loads(path.read_text(encoding="utf-8"))))
            except (OSError, KeyError, TypeError, ValueError) as exc:
                log.warning("skipping unreadable search tree %s: %s", path, exc)
        return trees


ARCHIVED = "archived"
SKIPPED = "skipped"


@dataclass(frozen=True)
class ArchiveResult:
    """What :func:`archive_session` did: ``archived``, or ``skipped`` with the ``reason``."""

    status: str
    reason: str = ""
    session_id: str = ""


def archive_session(
    snapshot: NotebookSnapshot,
    metric: str,
    archive: TreeArchive | None = None,
    *,
    budget_seconds: float | None = None,
    bottlenecks: Mapping[str, float] | None = None,
) -> ArchiveResult:
    """CLOSE: archive the session's search tree in ``archive``, by default the one :data:`ARCHIVE_ENV` names.

    ``budget_seconds`` is the session's configured budget; later sessions compare their progress with this one's at
    the same fraction of its budget, so it beats the default, the time the notebook spans. Nothing is archived, and the
    result says why, when no archive is configured, the session lacks its profile or baseline, its id cannot name a
    file, the archive already holds another tree for it, or the archive cannot be written.
    """

    target = archive if archive is not None else TreeArchive.from_env()
    if target is None:
        return ArchiveResult(SKIPPED, "no_archive_configured")
    try:
        tree = tree_from_snapshot(snapshot, metric, budget_seconds=budget_seconds, bottlenecks=bottlenecks)
    except NotRecorded:
        return ArchiveResult(SKIPPED, "no_profile_or_baseline")
    except ValueError:
        return ArchiveResult(SKIPPED, "baseline_lacks_metric", snapshot.profile.session_id if snapshot.profile else "")
    try:
        target.put(tree)
    except ValueError:
        return ArchiveResult(SKIPPED, "session_id_not_a_file_name", tree.session_id)
    except TreeConflict:
        log.warning("the archive already holds another search tree for session %s", tree.session_id)
        return ArchiveResult(SKIPPED, "another_tree_archived", tree.session_id)
    except OSError as exc:
        log.warning("could not archive the search tree of session %s in %s: %s", tree.session_id, target.root, exc)
        return ArchiveResult(SKIPPED, "archive_unwritable", tree.session_id)
    return ArchiveResult(ARCHIVED, session_id=tree.session_id)


__all__ = [
    "ARCHIVED",
    "ARCHIVE_ENV",
    "SINGLE_HOST",
    "SKIPPED",
    "SOURCE_NOTEBOOK",
    "TREE_FORMAT",
    "ArchiveResult",
    "MeasurementNoise",
    "SearchTree",
    "TreeArchive",
    "TreeBranch",
    "TreeCommit",
    "TreeConflict",
    "TreeDirection",
    "WeightedTree",
    "archive_session",
    "longest_root_test_seconds",
    "session_start",
    "similar_trees",
    "tree_document",
    "tree_from_document",
    "tree_from_snapshot",
]
