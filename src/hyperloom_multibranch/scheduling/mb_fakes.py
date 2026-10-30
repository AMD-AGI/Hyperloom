# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Placeholders the scheduler runs on where another owner has not delivered the real thing yet.

Every name here carries the ``MB_FAKE_`` / ``mb_fake_`` prefix so ``git grep -nE 'MbFake|mb_fake_|MB_FAKE_'`` lists
what is left to replace, and each docstring or comment names what it stands in for and what replaces it.
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import replace

from hyperloom_kb import NoOpExperienceKB
from hyperloom_multibranch.contracts.blackboard.records import BlackboardSnapshot
from hyperloom_multibranch.blackboard.rsi.session import (
    DISABLED,
    CloseStep,
    CloseTargets,
    KnowledgeClose,
    close_knowledge,
)
from hyperloom_multibranch.scheduling.layout import SessionLayout

#: Why CLOSE wrote nothing to the Experience KB: MB1 runs on no Experience KB, by @zgong's decision (board #22). The
#: online Experience KB service replaces it in a later milestone.
MB_FAKE_EXPERIENCE_KB = "mb_fake_experience_kb"


def mb_fake_close_knowledge(
    snapshot: BlackboardSnapshot,
    metric: str,
    *,
    budget_seconds: float,
    layout: SessionLayout,
    cross_image: Collection[str],
) -> KnowledgeClose:
    """CLOSE's knowledge steps with the Experiences and the learnings written nowhere: both steps are disabled for
    :data:`MB_FAKE_EXPERIENCE_KB`, whatever the environment configures; the search tree goes to the environment's
    archive as usual, marking the replays of the champions ``cross_image`` names, the collection receipts to the
    session's reports, and the judge's transcript beside the archive.

    Stands in for publishing to the online Experience KB service; replaced by ``close_knowledge`` on that service
    once a session uses it.
    """

    nowhere = NoOpExperienceKB(MB_FAKE_EXPERIENCE_KB)
    targets = CloseTargets(
        experience_kb=nowhere, learning_kb=nowhere, receipts=layout.reports, judge_transcript=layout.judge
    )
    done = close_knowledge(snapshot, metric, budget_seconds=budget_seconds, targets=targets, cross_image=cross_image)
    return replace(done, experiences=_faked(done.experiences), learnings=_faked(done.learnings))


def _faked(step: CloseStep) -> CloseStep:
    return replace(step, reason=MB_FAKE_EXPERIENCE_KB) if step.status == DISABLED else step


__all__ = [
    "MB_FAKE_EXPERIENCE_KB",
    "mb_fake_close_knowledge",
]
