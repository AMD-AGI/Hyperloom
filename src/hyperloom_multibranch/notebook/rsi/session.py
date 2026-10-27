# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What the scheduler calls on the notebook's knowledge: once at CLOSE, and every round of EXPLORE.

* :func:`close_knowledge` is CLOSE's one call after the Recipe KB write-back: the session's Experiences and the
  specialists' learnings go to the Experience KB, and the session's search tree to the archive. The Experiences the
  shared declaration cannot hold, kernel changes and an AgentX session's, are a step of their own. The learnings go
  to the RSI lesson pool beside the archive too, revising and retiring the past sessions' lessons they name, and the
  pooled lessons the session showed get their standing from what its branches did next (:mod:`rsi.lesson_close`).
  Beside the receipts goes the stop audit of this session's stops and probations against the stall odds the judge
  read (:mod:`rsi.stops`). A session that
  played a scheduling playbook is kept as its trial beside the playbook pool, which may crown or retire a playbook
  (:mod:`rsi.ladder`); then the scheduling replay reruns over the archive with this session's tree in it, against the
  champion, and says which of its readings moved since the last CLOSE. Then the specialists' outcomes, the menu's
  branches beside its control, go beside the receipts and the archive (:mod:`rsi.outcomes`), and, given the judge's
  transcript, its exchanges beside the archive for R2 to ask again (:mod:`rsi.decisions`). A step that cannot be
  done says why and never stops CLOSE; :meth:`KnowledgeClose.document` is what the breakdown records of the steps
  (``CloseOutcome.knowledge`` of ``build_breakdown``).
* :func:`judge_history` is the stall odds the judge reads every round, as text (:mod:`rsi.history`): how often a
  stalled branch and a new one went on to commit, here and in the archived sessions, and the stop bar they set;
  :func:`judge_open_areas` is where a new branch could open, and whether one should open at all
  (:mod:`rsi.open_areas`). :func:`session_menus` gives the briefs their direction menus (:mod:`rsi.menu`).

Passed no KB or archive, each step uses the environment's: the Experience KB ``HYPERLOOM_KB_URL`` names, and the
archive ``HYPERLOOM_RSI_ARCHIVE`` names.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

from hyperloom_kb import NoOpExperienceKB
from hyperloom_kb.collect import CollectionTarget, CollectReport
from hyperloom.common.io import WriteOptions, atomic_write_text
from hyperloom.common.jsonio import read_jsonl
from hyperloom_multibranch.contracts.notebook import NotRecorded
from hyperloom_multibranch.contracts.notebook.records import NotebookSnapshot
from hyperloom_multibranch.notebook.codec import encode
from hyperloom_multibranch.notebook.knowledge.experience import publish_experiences, publish_multibranch_experiences
from hyperloom_multibranch.notebook.knowledge.learnings import publish_learnings
from hyperloom_multibranch.notebook.learning_search import LEARNINGS_LOG
from hyperloom_multibranch.notebook.rsi.archive import (
    NO_ARCHIVE,
    SKIPPED,
    SearchTree,
    TreeArchive,
    archive_session,
    tree_from_snapshot,
)
from hyperloom_multibranch.notebook.rsi.decisions import archive_decisions
from hyperloom_multibranch.notebook.rsi.history import MAX_CHARS, session_stall_odds, stall_odds
from hyperloom_multibranch.notebook.rsi.lesson_close import publish_lessons, session_standings
from hyperloom_multibranch.notebook.rsi.lessons import LessonPool
from hyperloom_multibranch.notebook.rsi.menu import MENU_LOG, DirectionMenus, MenuLog
from hyperloom_multibranch.notebook.rsi.ladder import (
    moved,
    record_standing,
    record_trial,
    replay_standing,
    settle,
    standing_report,
    standings,
    trial_of,
)
from hyperloom_multibranch.notebook.rsi.open_areas import OpenAreas, open_areas
from hyperloom_multibranch.notebook.rsi.outcomes import OUTCOMES_REPORT, OutcomePool, specialist_outcomes
from hyperloom_multibranch.notebook.rsi.playbook import FALLBACK_KEY, H0, PLAYBOOK_KEY, Playbook, PlaybookPool
from hyperloom_multibranch.notebook.rsi.similarity import task_conditions, workload_kind
from hyperloom_multibranch.notebook.rsi.stops import stop_audit

log = logging.getLogger(__name__)

WRITTEN = "written"
DISABLED = "disabled"
FAILED = "failed"
NO_KB = "no_experience_kb_configured"
NO_PROFILE = "no_session_profile"
EXPERIENCE_RECEIPT = "experience_kb_experiences.json"
MULTIBRANCH_RECEIPT = "experience_kb_multibranch_experiences.json"
LEARNING_RECEIPT = "experience_kb_learnings.json"
#: The stop audit, written beside the receipts at every CLOSE.
STOP_AUDIT_REPORT = "stop-audit.md"
#: The scheduling replay rerun over the archive, written beside the receipts at every CLOSE.
SCHEDULING_REPLAY_REPORT = "scheduling-replay.md"
NO_REPORTS = "no_reports_folder"
#: The scores log's usual name in the session's reports folder.
FRONTIER_LOG = "frontier.jsonl"
NO_RETRIEVAL_LOG = "no_retrieval_log"
NO_LESSON_SHOWN = "no_pooled_lesson_shown"
ALREADY_RECORDED = "already_recorded"
NO_PLAYBOOK_PLAYED = "no_playbook_played"
NO_TREES = "no_archived_tree"
NO_TRANSCRIPT = "no_judge_transcript"
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
    notebook's own declaration holds (``multibranch_experiences``), the learnings into the RSI lesson pool
    (``lessons``), the standing of the pooled lessons the session showed, the stop audit,
    whose counts are the stops and probations by what came of them, the playbook trial, the scheduling replay rerun,
    and the specialists' outcomes."""

    experiences: CloseStep
    learnings: CloseStep
    archive: CloseStep
    multibranch_experiences: CloseStep = CloseStep(DISABLED, NO_KB)
    lessons: CloseStep = CloseStep(SKIPPED, NO_ARCHIVE)
    lesson_standings: CloseStep = CloseStep(SKIPPED, NO_ARCHIVE)
    stop_audit: CloseStep = CloseStep(SKIPPED, NO_REPORTS)
    playbook_trial: CloseStep = CloseStep(SKIPPED, NO_PLAYBOOK_PLAYED)
    scheduling_replay: CloseStep = CloseStep(SKIPPED, NO_REPORTS)
    specialist_outcomes: CloseStep = CloseStep(SKIPPED, NO_REPORTS)
    judge_decisions: CloseStep = CloseStep(SKIPPED, NO_TRANSCRIPT)

    def steps(self) -> dict[str, CloseStep]:
        """The steps by their names in the breakdown's ``close.knowledge``."""

        return {
            "experiences": self.experiences,
            "multibranch_experiences": self.multibranch_experiences,
            "learnings": self.learnings,
            "archive": self.archive,
            "lessons": self.lessons,
            "lesson_standings": self.lesson_standings,
            "stop_audit": self.stop_audit,
            "playbook_trial": self.playbook_trial,
            "scheduling_replay": self.scheduling_replay,
            "specialist_outcomes": self.specialist_outcomes,
            "judge_decisions": self.judge_decisions,
        }

    def document(self) -> dict[str, object]:
        """The steps as JSON-ready data: the breakdown's ``close.knowledge``."""

        return {name: encode(step) for name, step in self.steps().items()}


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
    cross_image: Collection[str] = (),
) -> CloseStep:
    try:
        result = archive_session(
            snapshot, metric, archive, budget_seconds=budget_seconds, bottlenecks=bottlenecks, cross_image=cross_image
        )
    except STEP_ERRORS as exc:
        return _failed("archive the search tree", exc)
    return CloseStep(result.status, result.reason)


def _trialled(
    snapshot: NotebookSnapshot, metric: str, archive: TreeArchive | None, bottlenecks: Mapping[str, float] | None
) -> CloseStep:
    """The session kept as a trial of the playbook it played, beside the pool, and what that settles: a playbook
    crowned or retired (:func:`~hyperloom_multibranch.notebook.rsi.ladder.settle`)."""

    versions = snapshot.fingerprints[-1].policy_versions if snapshot.fingerprints else {}
    played = versions.get(PLAYBOOK_KEY)
    if not played:
        return CloseStep(SKIPPED, NO_PLAYBOOK_PLAYED)
    target = archive if archive is not None else TreeArchive.from_env()
    if target is None:
        return CloseStep(SKIPPED, NO_ARCHIVE)
    try:
        tree = tree_from_snapshot(snapshot, metric, bottlenecks=bottlenecks)
        pool = PlaybookPool.beside(target)
        audit = stop_audit(tree, stall_odds(archived_history(target), tree.task, session=tree))
        trial = trial_of(tree, audit, played, pool.champion().label, fell_back=bool(versions.get(FALLBACK_KEY)))
        written = record_trial(pool, trial)
        settled = settle(pool, snapshot.taken_at)
    except NotRecorded:
        return CloseStep(SKIPPED, NO_PROFILE)
    except STEP_ERRORS as exc:
        return _failed("keep the playbook trial", exc)
    reason = "; ".join(settled) or ("" if written else ALREADY_RECORDED)
    return CloseStep(WRITTEN if written else SKIPPED, reason, {"trials": int(written), "settled": len(settled)})


def _audited(
    snapshot: NotebookSnapshot,
    metric: str,
    archive: TreeArchive | None,
    path: Path | None,
    bottlenecks: Mapping[str, float] | None,
) -> CloseStep:
    """The stop audit of the session, read against the stall odds of the archive and the session, written to
    ``path``."""

    if path is None:
        return CloseStep(SKIPPED, NO_REPORTS)
    try:
        tree = tree_from_snapshot(snapshot, metric, bottlenecks=bottlenecks)
        audit = stop_audit(tree, stall_odds(archived_history(archive), tree.task, session=tree))
        atomic_write_text(path, audit.render() + "\n", options=WriteOptions(make_parents=True))
    except NotRecorded:
        return CloseStep(SKIPPED, NO_PROFILE)
    except STEP_ERRORS as exc:
        return _failed("write the stop audit", exc)
    return CloseStep(WRITTEN, audit.note(), audit.counts())


def _decided(snapshot: NotebookSnapshot, archive: TreeArchive | None, transcript: Path | None) -> CloseStep:
    """The judge's exchanges of the session's ``transcript``, kept beside the archive for R2 and the offline
    generations (:mod:`rsi.decisions`)."""

    if transcript is None or not transcript.is_file():
        return CloseStep(SKIPPED, NO_TRANSCRIPT)
    target = archive if archive is not None else TreeArchive.from_env()
    if target is None:
        return CloseStep(SKIPPED, NO_ARCHIVE)
    if snapshot.profile is None:
        return CloseStep(SKIPPED, NO_PROFILE)
    try:
        kept = archive_decisions(transcript, target, snapshot.profile.session_id)
    except STEP_ERRORS as exc:
        return _failed("keep the judge's decisions", exc)
    if kept is None:
        return CloseStep(SKIPPED, ALREADY_RECORDED)
    return CloseStep(WRITTEN, counts={"calls": kept})


def _outcomes(snapshot: NotebookSnapshot, archive: TreeArchive | None, receipts: Path | None) -> CloseStep:
    """The specialists' outcomes, the menu's branches beside its control, read against the menus the session's log
    kept beside the receipts; written there, and kept beside the archive for the studies to add up."""

    if receipts is None:
        return CloseStep(SKIPPED, NO_REPORTS)
    if snapshot.profile is None:
        return CloseStep(SKIPPED, NO_PROFILE)
    target = archive if archive is not None else TreeArchive.from_env()
    try:
        found = specialist_outcomes(snapshot, MenuLog(receipts / MENU_LOG).menus(), archived_history(target))
        report = receipts / OUTCOMES_REPORT
        atomic_write_text(report, found.render() + "\n", options=WriteOptions(make_parents=True))
        kept = OutcomePool.beside(target).record(found) if target is not None else False
    except STEP_ERRORS as exc:
        return _failed("count the specialists' outcomes", exc)
    tests = {f"{name}_tests": group.tests for name, group in found.groups.items()}
    return CloseStep(WRITTEN, found.note(), {"kept": int(kept), **tests})


def _replayed(snapshot: NotebookSnapshot, archive: TreeArchive | None, path: Path | None) -> CloseStep:
    """The scheduling replay rerun over the archived sessions of this session's workload kind, its own tree among
    them, against the champion, written to ``path``; its standing kept beside the playbook pool for the next CLOSE to
    read what moved against. A gain in interactivity and one in throughput do not average."""

    if path is None:
        return CloseStep(SKIPPED, NO_REPORTS)
    target = archive if archive is not None else TreeArchive.from_env()
    if target is None:
        return CloseStep(SKIPPED, NO_ARCHIVE)
    if snapshot.profile is None:
        return CloseStep(SKIPPED, NO_PROFILE)
    kind = workload_kind(task_conditions(snapshot.profile))
    trees = tuple(tree for tree in archived_history(target) if workload_kind(tree.task) == kind)
    if not trees:
        return CloseStep(SKIPPED, NO_TREES)
    try:
        pool = PlaybookPool.beside(target)
        previous = next((item for item in reversed(standings(pool)) if item.at < snapshot.taken_at), None)
        standing, comparisons = replay_standing(
            trees, pool, session_id=snapshot.profile.session_id, at=snapshot.taken_at
        )
        record_standing(pool, standing)
        atomic_write_text(
            path, standing_report(standing, comparisons, previous) + "\n", options=WriteOptions(make_parents=True)
        )
    except STEP_ERRORS as exc:
        return _failed("rerun the scheduling replay", exc)
    shifted = moved(previous, standing)
    return CloseStep(WRITTEN, "; ".join(shifted), {"sessions": standing.sessions, "moved": len(shifted)})


@dataclass(frozen=True)
class CloseTargets:
    """Where CLOSE's knowledge goes: ``experience_kb`` takes the Experiences, ``learning_kb`` the learnings and
    ``archive`` the search tree, each the environment's when ``None``; ``receipts`` is the folder for the collection
    receipts, none without it, where the session's retrieval log lies too. ``multibranch_kb`` takes the Experiences
    under the notebook's own declaration, and ``lessons`` the lessons, by default the pool beside the archive.
    ``judge_transcript`` is the judge's transcript of the session, whose exchanges go beside the archive
    (:mod:`rsi.decisions`), none without it."""

    experience_kb: CollectionTarget | None = None
    learning_kb: CollectionTarget | None = None
    archive: TreeArchive | None = None
    receipts: Path | None = None
    multibranch_kb: CollectionTarget | None = None
    lessons: LessonPool | None = None
    judge_transcript: Path | None = None

    def receipt(self, name: str) -> Path | None:
        return self.receipts / name if self.receipts is not None else None

    def lesson_pool(self) -> LessonPool | None:
        """``lessons``; else the pool beside ``archive``, or beside the environment's archive, which CLOSE writes to
        even when the session read nothing from it."""

        if self.lessons is not None:
            return self.lessons
        archive = self.archive if self.archive is not None else TreeArchive.from_env()
        return LessonPool.beside(archive) if archive is not None else None

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
    cross_image: Collection[str] = (),
) -> KnowledgeClose:
    """CLOSE, after the Recipe KB write-back: what the session leaves the sessions after it.

    ``metric`` is the recorded-value throughput metric. ``budget_seconds`` is the session's configured budget, which
    later sessions compare the archived tree at shares of; without it the archive uses the time the notebook spans.
    ``targets`` says where each step writes, the environment's KB and archive by default. ``cross_image`` names, by
    canonical id, the Recipe KB champions PRELUDE prepared from another image than the session's, whose replays the
    archived tree marks.
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
        archive=_archived(
            snapshot,
            metric,
            targets.archive,
            budget_seconds=budget_seconds,
            bottlenecks=bottlenecks,
            cross_image=cross_image,
        ),
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
        lessons=_lessons(snapshot, targets.lesson_pool()),
        lesson_standings=_standings(snapshot, targets.lesson_pool(), targets.receipt(LEARNINGS_LOG)),
        stop_audit=_audited(snapshot, metric, targets.archive, targets.receipt(STOP_AUDIT_REPORT), bottlenecks),
        playbook_trial=_trialled(snapshot, metric, targets.archive, bottlenecks),
        scheduling_replay=_replayed(snapshot, targets.archive, targets.receipt(SCHEDULING_REPLAY_REPORT)),
        specialist_outcomes=_outcomes(snapshot, targets.archive, targets.receipts),
        judge_decisions=_decided(snapshot, targets.archive, targets.judge_transcript),
    )


def _lessons(snapshot: NotebookSnapshot, pool: LessonPool | None) -> CloseStep:
    """The session's learnings into ``pool``."""

    if pool is None:
        return CloseStep(SKIPPED, NO_ARCHIVE)
    try:
        published = publish_lessons(snapshot, pool)
    except NotRecorded:
        return CloseStep(SKIPPED, NO_PROFILE)
    except STEP_ERRORS as exc:
        return _failed("publish the lessons", exc)
    if published.written:
        status = WRITTEN
    else:
        status = FAILED if published.reasons else SKIPPED
    return CloseStep(status, "; ".join(published.reasons), published.counts)


def _standings(snapshot: NotebookSnapshot, pool: LessonPool | None, log_path: Path | None) -> CloseStep:
    """The standing of each pooled lesson the session's retrieval log at ``log_path`` says it showed."""

    if pool is None:
        return CloseStep(SKIPPED, NO_ARCHIVE)
    if log_path is None or not log_path.is_file():
        return CloseStep(SKIPPED, NO_RETRIEVAL_LOG)
    try:
        standings = session_standings(snapshot, pool, read_jsonl(log_path, skip_malformed=True, skip_non_dict=True))
        if not standings.lessons:
            return CloseStep(SKIPPED, NO_LESSON_SHOWN)
        written = pool.record_standings(standings)
    except NotRecorded:
        return CloseStep(SKIPPED, NO_PROFILE)
    except STEP_ERRORS as exc:
        return _failed("record the lessons' standings", exc)
    totals: Counter[str] = Counter()
    for standing in standings.lessons.values():
        totals.update(encode(standing))
    counts = {"lessons": len(standings.lessons), **totals}
    return CloseStep(WRITTEN if written else SKIPPED, "" if written else ALREADY_RECORDED, counts)


def archived_history(archive: TreeArchive | None = None) -> tuple[SearchTree, ...]:
    """Every search tree ``archive`` holds, by default the archive a running session reads
    (:meth:`TreeArchive.read_from_env`); none without one."""

    target = archive if archive is not None else TreeArchive.read_from_env()
    if target is None:
        return ()
    try:
        return tuple(target.load())
    except OSError as exc:
        log.warning("could not read the archive %s: %s", target.root, exc)
        return ()


def session_menus(metric: str, reports: Path | None, archive: TreeArchive | None = None) -> DirectionMenus | None:
    """The session's direction menus, for ``SessionBriefs(menus=...)``: over ``archive``, by default the one a running
    session reads, read once, each menu logged to ``<reports>/menus.jsonl`` for CLOSE to read; ``None`` without an
    archive, so a session that reads none is briefed as one without the RSI."""

    target = archive if archive is not None else TreeArchive.read_from_env()
    if target is None:
        return None
    log_to = MenuLog(reports / MENU_LOG) if reports is not None else None
    return DirectionMenus(archived_history(target), metric, log_to)


def judge_history(
    snapshot: NotebookSnapshot,
    history: Sequence[SearchTree],
    metric: str,
    *,
    budget_seconds: float,
    max_chars: int = MAX_CHARS,
) -> str:
    """The stall odds the stop rule rests on, for the judge this round, in at most ``max_chars``: from the archived
    sessions and this one's own branches so far; empty before the session has a profile and a baseline, and without
    archived sessions, so a session that reads no archive is judged as one without the RSI.

    Pass the scoring's own history, so the archive is read once a session. ``budget_seconds`` is kept for the
    scheduler's call; the odds count judged tests, not shares of the budget.
    """

    del budget_seconds
    if not history:
        return ""
    odds = session_stall_odds(snapshot, history, metric)
    return odds.render(max_chars) if odds is not None else ""


def judge_open_areas(
    snapshot: NotebookSnapshot,
    history: Sequence[SearchTree],
    metric: str,
    *,
    seconds_left: float,
    playbook: Playbook = H0,
) -> OpenAreas:
    """Where a new branch could open this round, for the judge of a session that plays ``playbook``
    (:mod:`rsi.open_areas`), and whether one should open at all in the ``seconds_left`` of EXPLORE: the time a new
    branch takes to its first commit is the playbook's ``last_open_minutes``, else the median the stall odds of the
    archived ``history`` read, and unknown without either. Pass the scoring's own history, as to
    :func:`judge_history`."""

    odds = session_stall_odds(snapshot, history, metric) if history else None
    first_commit = odds.first_commit_minutes if odds is not None else None
    return open_areas(snapshot, seconds_left=seconds_left, first_commit_minutes=first_commit, playbook=playbook)


__all__ = [
    "DISABLED",
    "EXPERIENCE_RECEIPT",
    "FAILED",
    "FRONTIER_LOG",
    "LEARNING_RECEIPT",
    "NO_KB",
    "ALREADY_RECORDED",
    "NO_LESSON_SHOWN",
    "NO_PLAYBOOK_PLAYED",
    "NO_PROFILE",
    "NO_REPORTS",
    "NO_RETRIEVAL_LOG",
    "NO_TRANSCRIPT",
    "NO_TREES",
    "SCHEDULING_REPLAY_REPORT",
    "STEP_ERRORS",
    "STOP_AUDIT_REPORT",
    "WRITTEN",
    "CloseStep",
    "CloseTargets",
    "KnowledgeClose",
    "archived_history",
    "close_knowledge",
    "judge_history",
    "judge_open_areas",
    "session_menus",
]
