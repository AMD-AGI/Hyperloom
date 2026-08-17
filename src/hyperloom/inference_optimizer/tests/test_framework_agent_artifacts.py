# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""FRAMEWORK candidate artifacts and outcome classification tests."""

from __future__ import annotations

import json
from pathlib import Path

from hyperloom.orchestrator.framework.artifacts import (
    candidate_slug,
    summarize_candidate_outcomes,
)


def test_candidate_slug_sanitizes_url():
    slug = candidate_slug("https://github.com/ROCm/vllm/pull/1234")
    assert "/" not in slug and ":" not in slug
    assert slug.strip("-") == slug
    assert slug


def test_candidate_slug_empty_defaults():
    assert candidate_slug("") == "candidate"
    assert candidate_slug("///") == "candidate"


def test_candidate_slug_caps_length():
    assert len(candidate_slug("x" * 500)) == 96
