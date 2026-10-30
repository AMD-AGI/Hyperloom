# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What a specialist does with a judged candidate: retest it, commit it, fork for its fix, diagnose what blocked it, or
only record it."""

from __future__ import annotations

from enum import Enum

from hyperloom_multibranch.contracts.judgement import AccuracyVerdict, Judgement, Verdict


class Action(Enum):
    """``DIAGNOSE`` hands a blocked gain to the specialist, which sends a repair request when it finds a fix."""

    RETEST = "retest"
    COMMIT = "commit"
    REQUEST_FORK = "request_fork"
    DIAGNOSE = "diagnose"
    RECORD_ONLY = "record_only"


# (verdict, fixes correctness with the accuracy gate passed) -> action. A fix that costs performance goes to a new
# branch, so the branch it was found on keeps what it had.
_COMMIT_TABLE = {
    (Verdict.NEEDS_CONFIRM, False): Action.RETEST,
    (Verdict.NEEDS_CONFIRM, True): Action.RETEST,
    (Verdict.IMPROVED, False): Action.COMMIT,
    (Verdict.IMPROVED, True): Action.COMMIT,
    (Verdict.BLOCKED, False): Action.DIAGNOSE,
    (Verdict.BLOCKED, True): Action.DIAGNOSE,
    (Verdict.INCONCLUSIVE, False): Action.RECORD_ONLY,
    (Verdict.INCONCLUSIVE, True): Action.RECORD_ONLY,
    (Verdict.FLAT, False): Action.RECORD_ONLY,
    (Verdict.FLAT, True): Action.COMMIT,
    (Verdict.REGRESSED, False): Action.RECORD_ONLY,
    (Verdict.REGRESSED, True): Action.REQUEST_FORK,
}


def decide(judgement: Judgement) -> Action:
    """Apply the commit table to one candidate.

    ``fixes_correctness`` means the repro case fails on the parent and passes on the candidate, and review found its
    trigger reachable under the target workload; it counts only when the accuracy gate passed.
    """
    passed = judgement.accuracy is AccuracyVerdict.PASSED
    return _COMMIT_TABLE[judgement.verdict, judgement.fixes_correctness and passed]


__all__ = ["Action", "decide"]
