"""Regression tests for cumulative extra_sglang_args merge/dedupe."""

from __future__ import annotations

from inference_optimizer.orchestrator.coordinator import (
    _dedupe_extra_sglang_args,
    _merge_cumulative_extra_sglang_args,
)


_CLEAN_STACK = (
    "--schedule-policy lpm --cuda-graph-bs 1 2 4 8 16 24 32 48 64 80 "
    "--mem-fraction-static 0.92 --page-size 16"
)


def test_merge_does_not_double_stack_when_candidate_is_cumulative() -> None:
    base = "--schedule-policy lpm --cuda-graph-bs 1 2 4 8 16 24 32 48 64 80 --mem-fraction-static 0.92"
    candidate = _CLEAN_STACK
    merged = _merge_cumulative_extra_sglang_args(base, candidate, candidate)
    assert merged == _CLEAN_STACK
    assert "0.92 2 4 8" not in merged


def test_merge_appends_delta_candidate() -> None:
    base = "--schedule-policy lpm"
    candidate = "--page-size 16"
    merged = _merge_cumulative_extra_sglang_args(base, candidate, candidate)
    assert merged == "--schedule-policy lpm --page-size 16"


def test_dedupe_preserves_cuda_graph_bs_multi_values() -> None:
    args = (
        "--schedule-policy lpm --cuda-graph-bs 1 2 4 8 16 24 32 48 64 80 "
        "--mem-fraction-static 0.92 --page-size 16"
    )
    assert _dedupe_extra_sglang_args(args) == args


def test_dedupe_last_wins_single_value_flags() -> None:
    args = "--mem-fraction-static 0.90 --mem-fraction-static 0.92"
    assert _dedupe_extra_sglang_args(args) == "--mem-fraction-static 0.92"
