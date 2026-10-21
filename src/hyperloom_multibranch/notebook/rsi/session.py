# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What the scheduler calls on the notebook's knowledge: once at CLOSE, and every round of EXPLORE.

* :func:`close_knowledge` is CLOSE's one call after the Recipe KB write-back: the session's Experiences and the
  specialists' learnings go to the Experience KB, and the session's search tree to the archive. The Experiences the
  shared declaration cannot hold, kernel changes and an AgentX session's, are a step of their own, and the last step
  writes the frontier calibration over the archived sessions like this one beside the receipts. A step that cannot
  be done says why and never stops CLOSE; :meth:`KnowledgeClose.document` is what the breakdown records of the steps
  (``build_breakdown(..., knowledge=...)``).
* :func:`session_frontier_policy` chooses its policy once the session has a profile: the calibration's recommendation
  for sessions like it, or the defaults; :func:`session_frontier_scorer` chooses the same way among the scorers of
  :mod:`rsi.scorers` too, crowning a challenger that passes the promotion gate; :func:`session_challenger` is the
  exploration pick for a 1-in-4 session, the best pooled challenger that is not clearly worse, without crowning;
  :func:`named_frontier_choice` is the choice a label names, as a manifest records it or ``--frontier-scorer`` asks
  for it. The scheduler holds the chosen scorer in one ``ScorerGuard`` a session, which stands ``frontier-v1`` in
  where it fails.
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

from hyperloom_kb import NoOpExperienceKB
from hyperloom_kb.collect import CollectionTarget, CollectReport
from hyperloom.common.io import WriteOptions, atomic_write_text
from hyperloom_multibranch.contracts.notebook import NotRecorded
from hyperloom_multibranch.contracts.notebook.records import NotebookSnapshot
from hyperloom_multibranch.notebook.codec import encode
from hyperloom_multibranch.notebook.knowledge.experience import publish_experiences, publish_multibranch_experiences
from hyperloom_multibranch.notebook.knowledge.learnings import publish_learnings
from hyperloom_multibranch.notebook.rsi.archive import SKIPPED, SearchTree, TreeArchive, archive_session, similar_trees
from hyperloom_multibranch.notebook.rsi.calibration import (
    DEFAULT_POLICY,
    VARIATIONS,
    PolicyChoice,
    calibrate,
    frontier_policy_for,
    frontier_scorer_for,
)
from hyperloom_multibranch.notebook.rsi.expression import ExpressionError
from hyperloom_multibranch.notebook.rsi.frontier import FRONTIER_V1_LABEL, FrontierPolicy
from hyperloom_multibranch.notebook.rsi.history import MAX_CHARS, session_history
from hyperloom_multibranch.notebook.rsi.scorers import (
    ChampionRecord,
    ScorerPool,
    current_champion,
    registered_scorers,
    scorer_named,
)
from hyperloom_multibranch.notebook.rsi.similarity import task_conditions

log = logging.getLogger(__name__)

WRITTEN = "written"
DISABLED = "disabled"
FAILED = "failed"
NO_KB = "no_experience_kb_configured"
NO_PROFILE = "no_session_profile"
EXPERIENCE_RECEIPT = "experience_kb_experiences.json"
MULTIBRANCH_RECEIPT = "experience_kb_multibranch_experiences.json"
LEARNING_RECEIPT = "experience_kb_learnings.json"
#: The frontier calibration's report, written beside the receipts at every CLOSE.
CALIBRATION_REPORT = "frontier-calibration.md"
NO_REPORTS = "no_reports_folder"
#: The scores log's usual name in the session's reports folder.
FRONTIER_LOG = "frontier.jsonl"
NO_PROFILE_EVIDENCE = f"{DEFAULT_POLICY}: the session has no profile yet"
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
    """CLOSE's knowledge steps: the Experiences, the learnings, the archived search tree, the Experiences only the
    notebook's own declaration holds (``multibranch_experiences``), and the frontier calibration's report."""

    experiences: CloseStep
    learnings: CloseStep
    archive: CloseStep
    multibranch_experiences: CloseStep = CloseStep(DISABLED, NO_KB)
    calibration: CloseStep = CloseStep(SKIPPED, NO_REPORTS)

    def document(self) -> dict[str, object]:
        """The steps as JSON-ready data: the breakdown's ``close.knowledge``."""

        steps = {
            "experiences": self.experiences,
            "multibranch_experiences": self.multibranch_experiences,
            "learnings": self.learnings,
            "archive": self.archive,
            "calibration": self.calibration,
        }
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


def _calibrated(
    snapshot: NotebookSnapshot, archive: TreeArchive | None, path: Path | None, bottlenecks: Mapping[str, float] | None
) -> CloseStep:
    """The frontier calibration over the archived sessions like this one, this one included, written to ``path``."""

    if path is None:
        return CloseStep(SKIPPED, NO_REPORTS)
    if snapshot.profile is None:
        return CloseStep(SKIPPED, NO_PROFILE)
    try:
        task = task_conditions(snapshot.profile, bottlenecks)
        report = calibrate(tuple(item.tree for item in similar_trees(archived_history(archive), task)))
        atomic_write_text(path, report.render() + "\n", options=WriteOptions(make_parents=True))
    except STEP_ERRORS as exc:
        return _failed("write the frontier calibration", exc)
    return CloseStep(WRITTEN, report.note)


@dataclass(frozen=True)
class CloseTargets:
    """Where CLOSE's knowledge goes: ``experience_kb`` takes the Experiences, ``learning_kb`` the learnings and
    ``archive`` the search tree, each the environment's when ``None``; ``receipts`` is the folder for the collection
    receipts, none without it. ``multibranch_kb`` takes the Experiences under the notebook's own declaration."""

    experience_kb: CollectionTarget | None = None
    learning_kb: CollectionTarget | None = None
    archive: TreeArchive | None = None
    receipts: Path | None = None
    multibranch_kb: CollectionTarget | None = None

    def receipt(self, name: str) -> Path | None:
        return self.receipts / name if self.receipts is not None else None

    def multibranch(self) -> CollectionTarget | None:
        """``multibranch_kb``; else the environment's beside the environment's ``experience_kb``, and nowhere beside
        an ``experience_kb`` the caller chose, so a caller who chose where Experiences go is never written behind."""

        if self.multibranch_kb is not None or self.experience_kb is None:
            return self.multibranch_kb
        return NoOpExperienceKB(NO_KB)


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
        multibranch_experiences=_published(
            "publish the Experiences under the notebook's own declaration",
            partial(
                publish_multibranch_experiences,
                snapshot,
                metric,
                kb=targets.multibranch(),
                receipt=targets.receipt(MULTIBRANCH_RECEIPT),
            ),
        ),
        calibration=_calibrated(snapshot, targets.archive, targets.receipt(CALIBRATION_REPORT), bottlenecks),
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


def session_frontier_policy(
    snapshot: NotebookSnapshot, history: Sequence[SearchTree], defaults: FrontierPolicy
) -> PolicyChoice:
    """The policy the session ``snapshot`` records scores with: :func:`calibration.frontier_policy_for` on its task, as
    the scoring reads it, once PRELUDE recorded the session's profile; ``defaults`` before."""

    if snapshot.profile is None:
        return PolicyChoice(defaults, DEFAULT_POLICY, NO_PROFILE_EVIDENCE)
    return frontier_policy_for(history, task_conditions(snapshot.profile), defaults)


def _environment_pool() -> ScorerPool | None:
    archive = TreeArchive.from_env()
    return None if archive is None else ScorerPool.beside(archive)


def session_frontier_scorer(
    snapshot: NotebookSnapshot, history: Sequence[SearchTree], defaults: FrontierPolicy, pool: ScorerPool | None = None
) -> PolicyChoice:
    """How the session ``snapshot`` records scores, once PRELUDE recorded its profile:
    :func:`calibration.frontier_scorer_for` on its task over every scorer :mod:`rsi.scorers` registers in ``pool``, by
    default the one beside the environment's archive, against the pool's champion; ``defaults`` before. A scorer that
    takes the champion's place is crowned in the pool, so the sessions after this one start from it."""

    if snapshot.profile is None:
        return PolicyChoice(defaults, DEFAULT_POLICY, NO_PROFILE_EVIDENCE)
    target = pool if pool is not None else _environment_pool()
    scorers = registered_scorers(target, snapshot.taken_at)
    held = current_champion(target)
    choice = frontier_scorer_for(history, task_conditions(snapshot.profile), defaults, scorers, held)
    holder = held.scorer.label if held is not None else FRONTIER_V1_LABEL
    if target is not None and choice.scorer.label != holder:
        _crown(target, ChampionRecord(choice.scorer.label, snapshot.taken_at, holder, choice.evidence))
    return choice


def _crown(pool: ScorerPool, record: ChampionRecord) -> None:
    # A champion the pool cannot record costs the next sessions the ratchet, not this one its start.
    try:
        pool.crown(record)
    except (OSError, ValueError) as exc:
        log.warning("could not crown %s in %s: %s", record.label, pool.root, exc)


def session_challenger(
    snapshot: NotebookSnapshot, history: Sequence[SearchTree], defaults: FrontierPolicy, pool: ScorerPool | None = None
) -> PolicyChoice | None:
    """The scorer a 1-in-4 exploration session runs with: the pooled challenger :func:`authoring.parents` would offer
    first after the champion, under ``defaults``, or ``None`` when the session has no profile, there is no pool, or no
    challenger qualifies. Does not crown; promotion stays :func:`session_frontier_scorer`."""

    if snapshot.profile is None:
        return None
    target = pool if pool is not None else _environment_pool()
    if target is None:
        return None
    # authoring imports CloseStep from this module; load the survey helpers only here.
    from hyperloom_multibranch.notebook.rsi.authoring import parents, survey

    found = survey(tuple(item.tree for item in similar_trees(history, task_conditions(snapshot.profile))), target)
    label = next((item for item in parents(found) if item != found.champion), None)
    if label is None:
        return None
    compared = found.standings.comparison(label)
    lead = (
        f"lead {compared.difference_low} to {compared.difference_high} over {found.champion}"
        if compared is not None and compared.difference_low is not None
        else f"not clearly worse than {found.champion}"
    )
    return named_frontier_choice(label, defaults, f"{label}: exploration challenger, {lead}", target)


def named_frontier_choice(
    label: str, defaults: FrontierPolicy, evidence: str, pool: ScorerPool | None = None
) -> PolicyChoice | None:
    """The choice ``label`` names, as ``--frontier-policy`` or a manifest's ``policy_versions`` gives it, with
    ``evidence``: the defaults, a variation of them, or a scorer of :mod:`rsi.scorers` (looked up in ``pool``, by
    default the one beside the environment's archive) under them; ``None`` when nothing goes by that label."""

    if label == DEFAULT_POLICY:
        return PolicyChoice(defaults, label, evidence)
    variation = next((item for item in VARIATIONS[1:] if item.label == label), None)
    if variation is not None:
        return PolicyChoice(variation.apply(defaults), label, evidence)
    try:
        scorer = scorer_named(label, pool if pool is not None else _environment_pool())
    except ExpressionError as exc:
        log.warning("the frontier scorer %s no longer reads: %s", label, exc)
        return None
    return None if scorer is None else PolicyChoice(defaults, label, evidence, scorer)


def judge_history(
    snapshot: NotebookSnapshot,
    history: Sequence[SearchTree],
    metric: str,
    *,
    budget_seconds: float,
    max_chars: int = MAX_CHARS,
) -> str:
    """How similar archived sessions went, for the judge this round, in at most ``max_chars``; empty without any.

    Pass the scoring's own history, so the archive is read once a session.
    """

    if not history:
        return ""
    return session_history(snapshot, history, metric, budget_seconds=budget_seconds).render(max_chars)


__all__ = [
    "CALIBRATION_REPORT",
    "DISABLED",
    "EXPERIENCE_RECEIPT",
    "FAILED",
    "FRONTIER_LOG",
    "LEARNING_RECEIPT",
    "NO_KB",
    "NO_PROFILE",
    "NO_REPORTS",
    "STEP_ERRORS",
    "WRITTEN",
    "CloseStep",
    "CloseTargets",
    "KnowledgeClose",
    "archived_history",
    "close_knowledge",
    "judge_history",
    "named_frontier_choice",
    "session_challenger",
    "session_frontier_policy",
    "session_frontier_scorer",
]
