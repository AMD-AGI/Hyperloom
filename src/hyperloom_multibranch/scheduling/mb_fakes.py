# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Placeholders the MB1 scheduler runs on where MB2, or another owner, has not delivered the real thing yet.

Every name here carries the ``MB_FAKE_`` / ``mb_fake_`` prefix so ``git grep -nE 'MbFake|mb_fake_|MB_FAKE_'`` lists
what is left to replace, and each docstring or comment names what it stands in for and what replaces it.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

from hyperloom.contracts.recipe import RecipeDelta
from hyperloom_kb import NoOpExperienceKB
from hyperloom_multibranch.contracts.notebook.records import NotebookSnapshot
from hyperloom_multibranch.notebook.knowledge.learning_recall import LearningRecall
from hyperloom_multibranch.notebook.learning_search import RetrievalLog
from hyperloom_multibranch.notebook.rsi.session import (
    DISABLED,
    CloseStep,
    CloseTargets,
    KnowledgeClose,
    close_knowledge,
)

#: Stands in for the image digest of the machine a session runs on while whoever launches the session does not name
#: it (``HYPERLOOM_ACTIVE_IMAGE_REF`` and ``HYPERLOOM_ACTIVE_IMAGE_DIGEST``, board #8); replaced by
#: ``hyperloom.measure.pins.active_image`` alone once the pod launcher sets both.
MB_FAKE_BASE_IMAGE_DIGEST = "sha256:" + "0" * 64
#: Why CLOSE wrote nothing to the Experience KB and the specialists recall no past session's learnings: MB1 runs on no
#: Experience KB, by @zgong's decision (board #22). The online Experience KB service replaces it in a later milestone.
MB_FAKE_EXPERIENCE_KB = "mb_fake_experience_kb"


class MbFakeSplitLanding(Protocol):
    """Lands an admitted fork the scheduler names by its branch: cuts the branch at the fork's parent commit, commits
    the fix on it and records its ``SPLIT``; true once it has, false when it will not.

    Stands in for the rule gate's landing method proposed on board #34 (request A), which keys the fork by the branch
    the scheduler reserved for it; replaced by that method, and the stand-ins implementing this, once it lands.
    """

    async def land_split(self, branch: str) -> bool: ...


async def mb_fake_settle_voided(branch: str, reason: str) -> None:
    """Tell the rule gate that the fork or repair reserved as ``branch`` was voided for ``reason``: nothing is told.

    Stands in for the gate's void callback proposed on board #59 (request B), which turns the request's
    ``SPLIT_PENDING`` or ``REPAIR_REQUESTED`` conclusion into a record-only one; replaced by that callback once it
    lands. Until then only the scheduler's decision records the void, and a voided repair stays requested in the
    notebook.
    """

    del branch, reason


@dataclass(frozen=True)
class MbFakeRegressionOutcome:
    """What the regression set found: ``passed``, or why not in ``reason``.

    Stands in for the ``RegressionOutcome`` proposed on board #61 (request D); replaced by it, together with
    :class:`MbFakeRegressionCheck`, once the regression set lands.
    """

    passed: bool
    reason: str = ""


class MbFakeRegressionCheck(Protocol):
    """Runs the regression set of ``base_commit``'s ancestry on ``base_commit`` with ``candidate_delta`` applied.

    The integrator asks it before a merge, FINAL for every candidate. Stands in for the rule gate's
    ``RegressionCheck`` proposed on board #61 (request D); replaced by it, and the stand-ins implementing this, once it
    lands.
    """

    async def check(self, base_commit: str, candidate_delta: RecipeDelta) -> MbFakeRegressionOutcome: ...


def mb_fake_learning_recall(log: RetrievalLog) -> LearningRecall:
    """The specialists' learnings recall on no Experience KB: this session's learnings only, logged to ``log``.

    Stands in for the recall on the online Experience KB service (:data:`MB_FAKE_EXPERIENCE_KB`); replaced by
    ``LearningRecall.from_env`` once a session uses that service.
    """

    return LearningRecall(None, log=log)


def mb_fake_close_knowledge(
    snapshot: NotebookSnapshot, metric: str, *, budget_seconds: float, receipts: Path
) -> KnowledgeClose:
    """CLOSE's knowledge steps with the Experiences and the learnings written nowhere: both steps are disabled for
    :data:`MB_FAKE_EXPERIENCE_KB`, whatever the environment configures; the search tree goes to the environment's
    archive as usual, and the collection receipts to ``receipts``.

    Stands in for publishing to the online Experience KB service; replaced by ``close_knowledge`` on that service
    once a session uses it.
    """

    nowhere = NoOpExperienceKB(MB_FAKE_EXPERIENCE_KB)
    targets = CloseTargets(experience_kb=nowhere, learning_kb=nowhere, receipts=receipts)
    done = close_knowledge(snapshot, metric, budget_seconds=budget_seconds, targets=targets)
    return replace(done, experiences=_faked(done.experiences), learnings=_faked(done.learnings))


def _faked(step: CloseStep) -> CloseStep:
    return replace(step, reason=MB_FAKE_EXPERIENCE_KB) if step.status == DISABLED else step


__all__ = [
    "MB_FAKE_BASE_IMAGE_DIGEST",
    "MB_FAKE_EXPERIENCE_KB",
    "MbFakeRegressionCheck",
    "MbFakeRegressionOutcome",
    "MbFakeSplitLanding",
    "mb_fake_close_knowledge",
    "mb_fake_learning_recall",
    "mb_fake_settle_voided",
]
