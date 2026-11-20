# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for framework-agent pure helpers in ``models``."""

from __future__ import annotations

from hyperloom.agents.framework.models import Candidate


def test_candidate_slug() -> None:
    assert Candidate(ref="feature/Foo@1").slug == "feature-foo-1"
    assert Candidate(ref="PR:123").slug == "pr-123"
    assert Candidate(ref="release/v0.8.x").slug == "release-v0.8.x"
    assert Candidate(ref="!!!").slug == "candidate"
