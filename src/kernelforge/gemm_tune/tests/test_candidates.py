# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Projecting a gemm session into independent per-tuner candidates."""

from __future__ import annotations

from kernelforge.gemm_tune.candidates import (
    TunerCandidate,
    failed_tuner_records,
    is_candidate,
    per_tuner_candidates,
)
from kernelforge.gemm_tune.tuners.base import TuneResult


def _ok(name: str, **kw) -> TuneResult:
    base = dict(
        tuner_name=name,
        status="ok",
        artifact_path=f"/csv/{name}.csv",
        env_var=f"AITER_CONFIG_{name.upper()}",
        env_value=f"/csv/{name}.csv",
        improved_shapes=2,
        total_shapes=5,
        best_micro_speedup=1.2,
    )
    base.update(kw)
    return TuneResult(**base)


# --- is_candidate: mirrors the report builder's promotion rule ----------------


def test_ok_with_improvement_is_a_candidate() -> None:
    assert is_candidate(_ok("a8w8")) is True


def test_ok_without_improvement_is_not_a_candidate() -> None:
    r = _ok("a8w8", improved_shapes=0, best_micro_speedup=1.0)
    assert is_candidate(r) is False


def test_partial_output_with_improvement_is_a_candidate() -> None:
    r = _ok("a8w8", status="partial_output")
    assert is_candidate(r) is True


def test_a_forced_candidate_counts_even_without_micro_gain() -> None:
    """Split-K tuning delivers e2e-only benefit and reports no_improvement."""
    r = _ok("splitk", status="no_improvement", improved_shapes=0, best_micro_speedup=1.0, candidate=True)
    assert is_candidate(r) is True


def test_a_forced_candidate_that_failed_is_not_a_candidate() -> None:
    r = _ok("splitk", status="failed", candidate=True)
    assert is_candidate(r) is False


def test_a_failed_tuner_is_never_a_candidate() -> None:
    assert is_candidate(_ok("a8w8", status="failed")) is False


# --- per_tuner_candidates: each deployable tuner, on its own ------------------


def test_the_moe_and_dense_tuners_become_two_independent_candidates() -> None:
    """The core of the change: KEEP one, REVERT the other."""
    results = [_ok("fmoe_ck"), _ok("a8w8_blockscale")]
    cands = per_tuner_candidates(results)
    assert [c.tuner for c in cands] == ["fmoe_ck", "a8w8_blockscale"]
    # Each carries only its own env, never a sibling's.
    assert cands[0].env == {"AITER_CONFIG_FMOE_CK": "/csv/fmoe_ck.csv"}
    assert cands[1].env == {"AITER_CONFIG_A8W8_BLOCKSCALE": "/csv/a8w8_blockscale.csv"}


def test_input_order_is_preserved() -> None:
    cands = per_tuner_candidates([_ok("b"), _ok("a")])
    assert [c.tuner for c in cands] == ["b", "a"]


def test_a_non_candidate_tuner_is_dropped() -> None:
    winner = _ok("a8w8")
    loser = _ok("fmoe_ck", improved_shapes=0, best_micro_speedup=1.0)
    cands = per_tuner_candidates([winner, loser])
    assert [c.tuner for c in cands] == ["a8w8"]


def test_a_candidate_with_neither_env_nor_artifact_is_dropped() -> None:
    """Nothing to apply -- the integrate lane could not land it."""
    r = _ok("empty", artifact_path="", env_var="", env_value="", env_vars={}, candidate=True, status="no_improvement")
    assert per_tuner_candidates([r]) == []


def test_env_merges_the_primary_pair_and_the_extra_map() -> None:
    r = _ok("a8w8", env_vars={"EXTRA": "1"})
    (cand,) = per_tuner_candidates([r])
    assert cand.env == {"AITER_CONFIG_A8W8": "/csv/a8w8.csv", "EXTRA": "1"}


def test_a_candidate_with_only_an_env_map_still_lands() -> None:
    r = _ok("a8w8", env_var="", env_value="", artifact_path="", env_vars={"ONLY": "x"})
    (cand,) = per_tuner_candidates([r])
    assert cand.env == {"ONLY": "x"} and cand.artifact_path == ""


def test_a_candidate_with_only_an_artifact_still_lands() -> None:
    r = _ok("a8w8", env_var="", env_value="", env_vars={})
    (cand,) = per_tuner_candidates([r])
    assert cand.artifact_path == "/csv/a8w8.csv" and cand.env == {}


def test_non_tuneresult_entries_are_skipped() -> None:
    assert per_tuner_candidates([{"tuner": "x"}, None, _ok("a8w8")]) == [c for c in per_tuner_candidates([_ok("a8w8")])]


def test_candidate_serializes_to_a_dict() -> None:
    (cand,) = per_tuner_candidates([_ok("a8w8", best_micro_speedup=1.23456)])
    d = cand.to_dict()
    assert d == {
        "tuner": "a8w8",
        "env": {"AITER_CONFIG_A8W8": "/csv/a8w8.csv"},
        "artifact_path": "/csv/a8w8.csv",
        "best_micro_speedup": 1.2346,
        "improved_shapes": 2,
        "requires_e2e_validation": True,
    }


def test_candidate_is_frozen() -> None:
    cand = TunerCandidate(tuner="a", env={}, artifact_path="", best_micro_speedup=1.0, improved_shapes=0)
    import dataclasses

    import pytest

    with pytest.raises(dataclasses.FrozenInstanceError):
        cand.tuner = "b"  # type: ignore[misc]


# --- failed_tuner_records: a crash stays visible past a sibling's win ---------


def test_a_failure_is_reported_even_when_a_sibling_won() -> None:
    results = [
        _ok("a8w8"),
        _ok("fmoe_ck", status="failed", error="boom", error_class="RuntimeError"),
    ]
    failed = failed_tuner_records(results)
    assert failed == [{"tuner": "fmoe_ck", "error_class": "RuntimeError", "error": "boom"}]


def test_no_failures_yields_no_records() -> None:
    assert failed_tuner_records([_ok("a8w8")]) == []


def test_non_tuneresult_entries_are_ignored() -> None:
    assert failed_tuner_records([None, {"status": "failed"}]) == []
