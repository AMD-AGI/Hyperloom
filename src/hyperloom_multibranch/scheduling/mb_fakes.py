# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Placeholders the MB1 scheduler runs on where MB2, or another owner, has not delivered the real thing yet.

Every name here carries the ``MB_FAKE_`` / ``mb_fake_`` prefix so ``git grep -nE 'MbFake|mb_fake_|MB_FAKE_'`` lists
what is left to replace, and each docstring or comment names what it stands in for and what replaces it.
"""

from __future__ import annotations

from hyperloom.common.perf_metric import GRADED_OUTPUT
from hyperloom_multibranch.contracts.fork import Admission, AdmissionStatus
from hyperloom_multibranch.scheduling.roster import Roster

#: Stands in for the image digest of the root recipe. The main flow builds no image; replaced when enablement or
#: hosted measurement produce one.
MB_FAKE_BASE_IMAGE_DIGEST = "sha256:" + "0" * 64

#: The metric the session's values are read in: the InferenceX rule's throughput. It feeds the frontier policy, final
#: selection, the dashboard and the final recipe's metric. Stands in for the per-workload choice (AgentX judges P50
#: interactivity); replaced once the workload names the benchmark it is judged on.
MB_FAKE_PRIMARY_METRIC = GRADED_OUTPUT


def mb_fake_refused_fork(refusal: str) -> Admission:
    """The answer to a fork or repair the roster cannot take now (``refusal`` says why): always a rejection.

    Stands in for the MB2 branch request policy, which keeps such a request pending and starts it first when a slot
    frees up; replaced when MB2 schedules that queue. A request is never left pending here.
    """

    return Admission(AdmissionStatus.REJECTED, refusal)


def mb_fake_branches_to_add(roster: Roster) -> int:
    """How many branches to open beyond the opening batch: always 0.

    Stands in for the MB2 later batches and refill of stopped slots; replaced when MB2 schedules them.
    """

    del roster
    return 0


def mb_fake_similar_branches(roster: Roster) -> tuple[str, ...]:
    """Branches to stop because they duplicate another: always none.

    Stands in for the MB2 similarity stop; replaced when MB2 compares branch trajectories.
    """

    del roster
    return ()


def mb_fake_regression_passes(commit: str) -> bool:
    """Whether ``commit`` passes the regression set: always true.

    Stands in for the MB2 regression set run before a merge or a final pick; replaced when that set exists.
    """

    del commit
    return True


__all__ = [
    "MB_FAKE_BASE_IMAGE_DIGEST",
    "MB_FAKE_PRIMARY_METRIC",
    "mb_fake_branches_to_add",
    "mb_fake_refused_fork",
    "mb_fake_regression_passes",
    "mb_fake_similar_branches",
]
