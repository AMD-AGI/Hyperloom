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
from hyperloom_multibranch.scheduling.mb_fakes import mb_fake_learning_recall


def test_the_specialists_recall_no_past_learnings_whatever_experience_kb_the_environment_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HYPERLOOM_KB_URL", "http://127.0.0.1:9/kb")
    recall = mb_fake_learning_recall(RetrievalLog(tmp_path / "learnings.jsonl"))

    past = recall.past(profile(), "kv cache", limit=5)

    assert (past.status, past.items) == (ReadStatus.DISABLED.value, ())
