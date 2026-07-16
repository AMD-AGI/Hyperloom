# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the quality-reference establish gate (R1).

Only a genuine ``baseline`` task may establish/overwrite the image-quality
reference; other kinds (e.g. ``replay_warm_recipe``) compare against it.
"""

from __future__ import annotations

from hyperloom.orchestrator.actions.executors.baseline import (
    _should_establish_quality_ref,
)


def test_genuine_baseline_establishes_reference():
    """A real ``baseline`` task is the only kind allowed to write the ref."""
    assert _should_establish_quality_ref("baseline") is True


def test_warm_replay_does_not_establish_reference():
    """``replay_warm_recipe`` must compare, never re-establish (R1 core)."""
    assert _should_establish_quality_ref("replay_warm_recipe") is False


def test_other_optimization_kinds_do_not_establish():
    """Explore/sweep/profile-style kinds also compare, never establish."""
    for kind in ("explore", "sweep", "conc_sweep", "roofline", "profile"):
        assert _should_establish_quality_ref(kind) is False


def test_missing_or_empty_kind_does_not_establish():
    """Defensive: ``None`` / empty kind must not be treated as a baseline."""
    assert _should_establish_quality_ref(None) is False
    assert _should_establish_quality_ref("") is False
