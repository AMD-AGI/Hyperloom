# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Contract of the MB1 stand-ins for MB2 policies and for the Experience KB: what the scheduler may rely on until they
are replaced."""

from __future__ import annotations

from pathlib import Path

import pytest

from hyperloom_kb import ReadStatus
from hyperloom_multibranch.contracts.fork import AdmissionStatus, ForkAdmission
from hyperloom_multibranch.scheduling.clock import Clock
from hyperloom_multibranch.notebook.learning_search import RetrievalLog
from hyperloom_multibranch.notebook.tests.factories import profile
from hyperloom_multibranch.scheduling.mb_fakes import (
    MB_FAKE_BASE_IMAGE_DIGEST,
    mb_fake_branches_to_add,
    mb_fake_learning_recall,
    mb_fake_refused_fork,
    mb_fake_regression_passes,
    mb_fake_similar_branches,
)
from hyperloom_multibranch.scheduling.roster import Roster, RosterCaps


def _roster(tmp_path: Path) -> Roster:
    caps = RosterCaps(max_branches=1, max_k_per_branch=2, max_parallel_tests=4)
    return Roster(caps, tmp_path / "MB-roster.json", Clock(now=lambda: 0.0))


def test_a_fork_the_roster_cannot_take_is_rejected_never_pending() -> None:
    assert mb_fake_refused_fork("the session is stopping") == ForkAdmission(
        AdmissionStatus.REJECTED, "the session is stopping"
    )


def test_mb2_policies_are_inert(tmp_path: Path) -> None:
    roster = _roster(tmp_path)
    assert mb_fake_branches_to_add(roster) == 0
    assert mb_fake_similar_branches(roster) == ()
    assert mb_fake_regression_passes("any-commit")


def test_base_image_digest_has_the_digest_shape() -> None:
    algorithm, _, digest = MB_FAKE_BASE_IMAGE_DIGEST.partition(":")
    assert algorithm == "sha256"
    assert len(digest) == 64


def test_the_specialists_recall_no_past_learnings_whatever_experience_kb_the_environment_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HYPERLOOM_KB_URL", "http://127.0.0.1:9/kb")
    recall = mb_fake_learning_recall(RetrievalLog(tmp_path / "learnings.jsonl"))

    past = recall.past(profile(), "kv cache", limit=5)

    assert (past.status, past.items) == (ReadStatus.DISABLED.value, ())
