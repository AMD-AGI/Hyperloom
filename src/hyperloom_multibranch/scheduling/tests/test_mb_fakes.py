# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Contract of the session's stand-ins for other owners' interfaces and for the Experience KB: what the scheduler may
rely on until they are replaced."""

from __future__ import annotations

from pathlib import Path

import pytest

from hyperloom_kb import ReadStatus
from hyperloom_multibranch.notebook.learning_search import RetrievalLog
from hyperloom_multibranch.notebook.tests.factories import profile
from hyperloom_multibranch.scheduling.mb_fakes import (
    MB_FAKE_BASE_IMAGE_DIGEST,
    mb_fake_learning_recall,
    mb_fake_settle_voided,
)
from hyperloom_multibranch.scheduling.local_peers import MbFakeNoSplitLanding


async def test_no_fork_is_landed_and_a_void_tells_the_gate_nothing() -> None:
    assert not await MbFakeNoSplitLanding().land_split("b03")
    assert await mb_fake_settle_voided("b03", "parent b01 stopped: flat") is None


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
