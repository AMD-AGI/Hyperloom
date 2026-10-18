# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What the scheduler calls on the notebook's knowledge: once at CLOSE, and every round of EXPLORE.

* :func:`close_knowledge` is CLOSE's one call after the Recipe KB write-back: the session's Experiences and the
  specialists' learnings go to the Experience KB, and the session's search tree to the archive. A step that cannot be
  done says why and never stops CLOSE; :meth:`KnowledgeClose.document` is what the breakdown records of the steps
  (``build_breakdown(..., knowledge=...)``).
* :func:`session_frontier` is one session's frontier scoring: windows sized to its budget, the archived sessions as
  the prior, and the scores log.
* :func:`judge_history` is the history digest the judge reads every round, as text; empty while nothing is archived.

Passed no KB or archive, each step uses the environment's: the Experience KB ``HYPERLOOM_KB_URL`` names, and the
archive ``HYPERLOOM_RSI_ARCHIVE`` names.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

from hyperloom_kb.collect import CollectionTarget, CollectReport
from hyperloom_multibranch.contracts.notebook import NotRecorded
from hyperloom_multibranch.contracts.notebook.interfaces import SnapshotQuery
from hyperloom_multibranch.contracts.notebook.records import NotebookSnapshot
from hyperloom_multibranch.notebook.codec import encode
from hyperloom_multibranch.notebook.knowledge.experience import publish_experiences
from hyperloom_multibranch.notebook.knowledge.learnings import publish_learnings
from hyperloom_multibranch.notebook.rsi.archive import SKIPPED, SearchTree, TreeArchive, archive_session
from hyperloom_multibranch.notebook.rsi.frontier import FrontierPolicy, FrontierScoring
from hyperloom_multibranch.notebook.rsi.history import MAX_CHARS, session_history

log = logging.getLogger(__name__)

WRITTEN = "written"
DISABLED = "disabled"
FAILED = "failed"
NO_KB = "no_experience_kb_configured"
NO_PROFILE = "no_session_profile"
EXPERIENCE_RECEIPT = "experience_kb_experiences.json"
LEARNING_RECEIPT = "experience_kb_learnings.json"
#: The scores log's usual name in the session's reports folder.
FRONTIER_LOG = "frontier.jsonl"
#: What a step raises when its target or its input lets it down; CLOSE records it and goes on.
STEP_ERRORS = (LookupError, OSError, RuntimeError, ValueError)


@dataclass(frozen=True)
class CloseStep:
    """What one knowledge step of CLOSE did: ``status``, why when it wrote nothing or not all, and how many by status.

    ``counts`` holds, for a KB step, the Experiences by write status (``created``, ``unchanged``, ...) and the units
    skipped and failed.
    """

    status: str
    reason: str = ""
    counts: Mapping[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class KnowledgeClose:
    """CLOSE's knowledge steps: the Experiences, the learnings, and the archived search tree."""

    experiences: CloseStep
    learnings: CloseStep
    archive: CloseStep

    def document(self) -> dict[str, object]:
        """The steps as JSON-ready data: the breakdown's ``close.knowledge``."""

        steps = {"experiences": self.experiences, "learnings": self.learnings, "archive": self.archive}
        return {name: encode(step) for name, step in steps.items()}


def _failed(what: str, exc: Exception) -> CloseStep:
    log.warning("CLOSE could not %s: %s", what, exc)
    return CloseStep(FAILED, f"{type(exc).__name__}: {exc}")


def _published(what: str, publish: Callable[[], CollectReport]) -> CloseStep:
    try:
        report = publish()
    except NotRecorded:
        return CloseStep(SKIPPED, NO_PROFILE)
    except STEP_ERRORS as exc:
        return _failed(what, exc)
    if not report.enabled:
        return CloseStep(DISABLED, NO_KB)
    if report.blocked_reason:
        return CloseStep(SKIPPED, report.blocked_reason)
    counts = {
        **Counter(item.status for item in report.collected),
        "skipped": len(report.skipped),
        "errors": len(report.errors),
    }
    reason = report.errors[0].error if report.errors else ""
    return CloseStep(FAILED if report.errors and not report.collected else WRITTEN, reason, counts)


def _archived(
    snapshot: NotebookSnapshot,
    metric: str,
    archive: TreeArchive | None,
    *,
    budget_seconds: float | None,
    bottlenecks: Mapping[str, float] | None,
) -> CloseStep:
    try:
        result = archive_session(snapshot, metric, archive, budget_seconds=budget_seconds, bottlenecks=bottlenecks)
    except STEP_ERRORS as exc:
        return _failed("archive the search tree", exc)
    return CloseStep(result.status, result.reason)


@dataclass(frozen=True)
class CloseTargets:
    """Where CLOSE's knowledge goes: ``experience_kb`` takes the Experiences, ``learning_kb`` the learnings and
    ``archive`` the search tree, each the environment's when ``None``; ``receipts`` is the folder for the two
    collection receipts, none without it."""

    experience_kb: CollectionTarget | None = None
    learning_kb: CollectionTarget | None = None
    archive: TreeArchive | None = None
    receipts: Path | None = None

    def receipt(self, name: str) -> Path | None:
        return self.receipts / name if self.receipts is not None else None


def close_knowledge(
    snapshot: NotebookSnapshot,
    metric: str,
    *,
    budget_seconds: float | None = None,
    targets: CloseTargets = CloseTargets(),
    bottlenecks: Mapping[str, float] | None = None,
) -> KnowledgeClose:
    """CLOSE, after the Recipe KB write-back: what the session leaves the sessions after it.

    ``metric`` is the recorded-value throughput metric. ``budget_seconds`` is the session's configured budget, which
    later sessions compare the archived tree at shares of; without it the archive uses the time the notebook spans.
    ``targets`` says where each step writes, the environment's KB and archive by default.
    """

    return KnowledgeClose(
        experiences=_published(
            "publish the Experiences",
            partial(
                publish_experiences,
                snapshot,
                metric,
                kb=targets.experience_kb,
                receipt=targets.receipt(EXPERIENCE_RECEIPT),
            ),
        ),
        learnings=_published(
            "publish the learnings",
            partial(publish_learnings, snapshot, kb=targets.learning_kb, receipt=targets.receipt(LEARNING_RECEIPT)),
        ),
        archive=_archived(snapshot, metric, targets.archive, budget_seconds=budget_seconds, bottlenecks=bottlenecks),
    )


def archived_history(archive: TreeArchive | None = None) -> tuple[SearchTree, ...]:
    """Every search tree ``archive`` holds, by default the environment's archive; none without one."""

    target = archive if archive is not None else TreeArchive.from_env()
    if target is None:
        return ()
    try:
        return tuple(target.load())
    except OSError as exc:
        log.warning("could not read the archive %s: %s", target.root, exc)
        return ()


def session_frontier(
    notebook: SnapshotQuery,
    metric: str,
    *,
    budget_seconds: float | None = None,
    log_path: Path | None = None,
    history: Sequence[SearchTree] | None = None,
) -> FrontierScoring:
    """One session's frontier scoring: windows sized to ``budget_seconds``, ``history`` as the prior, the scores log.

    ``history`` defaults to the environment's archive, read once here. ``log_path`` (:data:`FRONTIER_LOG` in the
    session's reports folder, say) gets one line per round that scored a branch.
    """

    policy = FrontierPolicy(metric) if budget_seconds is None else FrontierPolicy.for_budget(metric, budget_seconds)
    trees = archived_history() if history is None else tuple(history)
    return FrontierScoring(notebook, policy, trees, log=log_path)


def judge_history(
    snapshot: NotebookSnapshot,
    history: Sequence[SearchTree],
    metric: str,
    *,
    budget_seconds: float,
    max_chars: int = MAX_CHARS,
) -> str:
    """How similar archived sessions went, for the judge this round, in at most ``max_chars``; empty without any.

    Pass the scoring's own history (``session_frontier(...).history``), so the archive is read once a session.
    """

    if not history:
        return ""
    return session_history(snapshot, history, metric, budget_seconds=budget_seconds).render(max_chars)


__all__ = [
    "DISABLED",
    "EXPERIENCE_RECEIPT",
    "FAILED",
    "FRONTIER_LOG",
    "LEARNING_RECEIPT",
    "NO_KB",
    "NO_PROFILE",
    "STEP_ERRORS",
    "WRITTEN",
    "CloseStep",
    "CloseTargets",
    "KnowledgeClose",
    "archived_history",
    "close_knowledge",
    "judge_history",
    "session_frontier",
]
