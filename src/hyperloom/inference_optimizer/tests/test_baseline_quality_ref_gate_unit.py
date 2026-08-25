# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the quality-reference establish gate.

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
    """``replay_warm_recipe`` must compare, never re-establish."""
    assert _should_establish_quality_ref("replay_warm_recipe") is False


def test_other_optimization_kinds_do_not_establish():
    """Explore/sweep/profile-style kinds also compare, never establish."""
    for kind in ("explore", "sweep", "conc_sweep", "roofline", "profile"):
        assert _should_establish_quality_ref(kind) is False


def test_missing_or_empty_kind_does_not_establish():
    """Defensive: ``None`` / empty kind must not be treated as a baseline."""
    assert _should_establish_quality_ref(None) is False
    assert _should_establish_quality_ref("") is False


def test_quality_ref_exempt_baseline_does_not_establish():
    """Synthetic kernel-lane re-baselines opt out of establishing the ref.

    Integrate re-baseline and stack validation both carry ``kind="baseline"``
    literally, but are throughput-only probes against
    an already-anchored baseline.
    """
    assert _should_establish_quality_ref("baseline", {"quality_ref_exempt": True}) is False


def test_plain_params_still_establish():
    """Params without the exemption leave a genuine baseline untouched."""
    assert _should_establish_quality_ref("baseline", {}) is True
    assert _should_establish_quality_ref("baseline", None) is True
    assert _should_establish_quality_ref("baseline", {"framework": "sglang"}) is True
    assert _should_establish_quality_ref("baseline", {"quality_ref_exempt": False}) is True
