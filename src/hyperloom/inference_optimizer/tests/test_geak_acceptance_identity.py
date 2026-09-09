# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit coverage for GEAK acceptance identity in the breakdown collectors.

Each test here is anchored to a case measured on the recorded campaign at
``/shared_nfs/hyperloom-claw``. The behaviours under test are the ones a
reviewer flagged on Hyperloom PR #1209:

* **Two lanes.** An acceptance lands in ``accepted_kernels`` *or*
  ``accepted_heads``. Measured: 8 of 11 sessions with an acceptance carry it in
  ``accepted_heads`` alone, 3 in ``accepted_kernels`` alone, 0 in both. Reading
  one lane loses most of them.
* **Alias twins.** GEAK records one acceptance as two journey rows: the
  candidate-slot row carries the measurement, the resolved-symbol row does not.
  The symbol is on the *unmeasured* twin, and ``name`` holds it while
  ``kernel_id`` holds an underscore-stripped slug.
* **Shared admission.** A journey row declares no ``kind``, so a library
  selection (``kind="env"``) would be credited as an authored kernel. The kind
  is recovered by joining the symbol back to the run's ``result.json`` lane, and
  what the join cannot resolve is recorded as unresolved rather than guessed.
"""

from __future__ import annotations

from typing import Any

from hyperloom.inference_optimizer.breakdown.collectors.attribution import (
    _geak_kernel_names,
)
from hyperloom.inference_optimizer.breakdown.collectors.geak import (
    _collapse_journey_aliases,
    _kind_source_counts,
    _stamp_journey_kind,
)
from hyperloom.orchestrator.loop.coordinator_helpers import (
    _geak_accepted_kernel_specs,
    geak_is_cand_tag,
    geak_spec_is_env,
    geak_spec_kind,
    geak_spec_name,
)


def _spec(name: str, delta: float, **extra: Any) -> dict[str, Any]:
    return {"short_name": name, "e2e_delta_pct": delta, **extra}


# --------------------------------------------------------------------------
# B1 — attribution must read both acceptance lanes
# --------------------------------------------------------------------------


def test_geak_kernel_names_reads_the_heads_lane() -> None:
    # Kimi-K3/20260816T122327Z: the acceptance is in accepted_heads alone.
    entry: dict[str, Any] = {
        "accepted_kernels": [],
        "accepted_heads": [{"short_name": "_fwd_grouped_kernel_stage1", "kind": "authored"}],
    }
    assert _geak_kernel_names(entry) == ["_fwd_grouped_kernel_stage1"]


def test_geak_kernel_names_merges_both_lanes_without_duplicates() -> None:
    entry: dict[str, Any] = {
        "accepted_kernels": [{"short_name": "a_kernel", "kind": "authored"}],
        "accepted_heads": [
            {"short_name": "a_kernel", "kind": "authored"},
            {"short_name": "b_kernel", "kind": "authored"},
        ],
    }
    assert _geak_kernel_names(entry) == ["a_kernel", "b_kernel"]


def test_geak_kernel_names_drops_env_selections() -> None:
    # Qwen3-14B-FP8/20260814T163051Z: c1_ck is a ck library pick, not a kernel.
    entry: dict[str, Any] = {
        "accepted_kernels": [],
        "accepted_heads": [
            {"short_name": "ck_gemm_a8w8_blockscale_bpreshuffle", "kind": "env"},
            {"short_name": "real_kernel", "kind": "authored"},
        ],
    }
    assert _geak_kernel_names(entry) == ["real_kernel"]


def test_geak_kernel_names_accepts_bare_string_specs() -> None:
    # A lane may hold plain strings; those declare no kind and are not env.
    entry: dict[str, Any] = {"accepted_kernels": ["plain_kernel"], "accepted_heads": []}
    assert _geak_kernel_names(entry) == ["plain_kernel"]


def test_geak_kernel_names_empty_when_no_lane_has_content() -> None:
    assert _geak_kernel_names({"accepted_kernels": [], "accepted_heads": []}) == []


# --------------------------------------------------------------------------
# B2 — the alias twin collapses onto the resolved symbol
# --------------------------------------------------------------------------


def test_collapse_keeps_the_symbol_from_the_unmeasured_twin() -> None:
    # GLM-5.2-MXFP4/20260814T163244Z. The measured row is the slot tag; the
    # symbol rides the twin that carries no gpu_pct.
    # Both rows repeat the same e2e_gain_pct: that is what pairs the twin.
    rows = [
        {
            "kernel_id": "c0_triton",
            "name": "c0_triton",
            "gpu_pct": 47.3,
            "e2e_gain_pct": 29.994,
        },
        {
            "kernel_id": "dsa_sparse_attn_prefill_main_kernel",
            "name": "dsa_sparse_attn_prefill_main_kernel",
            "gpu_pct": None,
            "e2e_gain_pct": 29.994,
        },
    ]
    out = _collapse_journey_aliases(rows)
    assert len(out) == 1
    assert out[0]["name"] == "dsa_sparse_attn_prefill_main_kernel"
    assert out[0]["kernel_id"] == "dsa_sparse_attn_prefill_main_kernel"
    # The measurement survives the rename.
    assert out[0]["gpu_pct"] == 47.3
    # The slot tag is kept as an alias so a later join can still use it.
    assert "c0_triton" in out[0]["aliases"]


def test_collapse_prefers_name_over_the_slugged_kernel_id() -> None:
    # MiniMax-M3-MXFP8/20260731T182731Z. GEAK strips the leading underscore
    # when it builds kernel_id, so only `name` holds the true symbol.
    rows = [
        {
            "kernel_id": "c0_flydsl",
            "name": "c0_flydsl",
            "gpu_pct": 12.0,
            "e2e_gain_pct": 40.626,
        },
        {
            "kernel_id": "mxfp8_linear_kernel",
            "name": "_mxfp8_linear_kernel",
            "gpu_pct": None,
            "e2e_gain_pct": 40.626,
        },
    ]
    out = _collapse_journey_aliases(rows)
    assert out[0]["name"] == "_mxfp8_linear_kernel"
    assert "mxfp8_linear_kernel" in out[0]["aliases"]


def test_collapse_leaves_a_lone_measured_row_alone() -> None:
    rows = [
        {
            "kernel_id": "solo_kernel",
            "name": "solo_kernel",
            "gpu_pct": 3.0,
            "e2e_gain_pct": 1.5,
        }
    ]
    out = _collapse_journey_aliases(rows)
    assert len(out) == 1
    assert out[0]["name"] == "solo_kernel"


def test_collapse_keeps_two_real_kernels_that_share_a_gain() -> None:
    # Rows are grouped by gain, so two distinct kernels that happen to land on
    # the same number must not fold into one. Both are measured, so neither is
    # a twin: the twin is defined by the missing measurement, not by the gain.
    rows = [
        {"kernel_id": "kernel_a", "name": "kernel_a", "gpu_pct": 10.0, "e2e_gain_pct": 2.0},
        {"kernel_id": "kernel_b", "name": "kernel_b", "gpu_pct": 20.0, "e2e_gain_pct": 2.0},
    ]
    out = _collapse_journey_aliases(rows)
    assert sorted(r["name"] for r in out) == ["kernel_a", "kernel_b"]


def test_specs_keeps_two_distinct_kernels_that_share_op_kind_and_gain() -> None:
    result = {
        "accepted_kernels": [
            _spec("kernel_a", 12.31, op_kind="prefill_attn"),
            _spec("kernel_b", 12.31, op_kind="prefill_attn"),
        ]
    }
    assert [geak_spec_name(s) for s in _geak_accepted_kernel_specs(result)] == [
        "kernel_a",
        "kernel_b",
    ]


def test_collapse_skips_unrelated_measured_unmeasured_pair_at_same_gain() -> None:
    rows = [
        {
            "kernel_id": "kernel_a",
            "name": "kernel_a",
            "gpu_pct": 10.0,
            "e2e_gain_pct": 5.0,
            "op_kind": "prefill_attn",
        },
        {
            "kernel_id": "kernel_b",
            "name": "kernel_b",
            "gpu_pct": None,
            "e2e_gain_pct": 5.0,
            "op_kind": "prefill_attn",
        },
    ]
    out = _collapse_journey_aliases(rows)
    assert sorted(r["name"] for r in out) == ["kernel_a", "kernel_b"]


def test_collapse_skips_three_kernel_mixed_group_at_same_gain() -> None:
    rows = [
        {
            "kernel_id": "c0_triton",
            "name": "c0_triton",
            "gpu_pct": 10.0,
            "e2e_gain_pct": 5.0,
            "op_kind": "prefill_attn",
        },
        {
            "kernel_id": "sym_a",
            "name": "sym_a",
            "gpu_pct": None,
            "e2e_gain_pct": 5.0,
            "op_kind": "prefill_attn",
        },
        {
            "kernel_id": "kernel_c",
            "name": "kernel_c",
            "gpu_pct": 8.0,
            "e2e_gain_pct": 5.0,
            "op_kind": "prefill_attn",
        },
    ]
    out = _collapse_journey_aliases(rows)
    assert len(out) == 3


def test_collapse_keeps_two_unmeasured_rows_that_share_a_gain() -> None:
    rows = [
        {"kernel_id": "kernel_a", "name": "kernel_a", "gpu_pct": None, "e2e_gain_pct": 2.0},
        {"kernel_id": "kernel_b", "name": "kernel_b", "gpu_pct": None, "e2e_gain_pct": 2.0},
    ]
    out = _collapse_journey_aliases(rows)
    assert sorted(r["name"] for r in out) == ["kernel_a", "kernel_b"]


def _acceptance_specs(result: dict[str, Any]) -> list[dict[str, Any]]:
    from hyperloom.orchestrator.phases.kernel import KernelPhase

    return KernelPhase._geak_acceptance_specs(result)


def test_acceptance_specs_collapse_the_alias_twin_onto_the_kernel_symbol() -> None:
    result = {
        "accepted_kernels": [_spec("c0_triton", 12.31, op_kind="prefill_attn")],
        "accepted_heads": [_spec("_dsa_prefill_kernel", 12.31, op_kind="prefill_attn")],
    }
    out = _acceptance_specs(result)
    assert [geak_spec_name(row) for row in out] == ["_dsa_prefill_kernel"]
    assert out[0]["alias_collapsed"] is True


def test_acceptance_specs_keep_two_rows_that_merely_both_lack_a_delta() -> None:
    """No measured delta is no evidence of twinning.

    Reading the delta as ``float(x or 0.0)`` mapped absent and zero onto the
    same number, so two unrelated env selections on one op_kind collapsed into
    one acceptance for carrying no measurement at all.
    """
    result = {
        "accepted_heads": [
            {"short_name": "ck_gemm_a8w8", "kind": "env", "op_kind": "gemm"},
            {"short_name": "hipblaslt_gemm", "kind": "env", "op_kind": "gemm"},
        ]
    }
    assert sorted(geak_spec_name(row) for row in _acceptance_specs(result)) == [
        "ck_gemm_a8w8",
        "hipblaslt_gemm",
    ]


def test_acceptance_specs_keep_a_row_whose_delta_is_not_a_number() -> None:
    """An unreadable delta is a row with no delta, not an absent acceptance."""
    result = {"accepted_heads": [_spec("odd_kernel", "n/a", op_kind="gemm")]}  # type: ignore[arg-type]
    out = _acceptance_specs(result)
    assert [geak_spec_name(row) for row in out] == ["odd_kernel"]


def test_acceptance_specs_still_distinguish_a_measured_zero_from_an_absent_delta() -> None:
    result = {
        "accepted_heads": [
            _spec("measured_zero", 0.0, op_kind="gemm"),
            {"short_name": "no_delta", "op_kind": "gemm"},
        ]
    }
    assert len(_acceptance_specs(result)) == 2


def test_cand_tag_recognises_slot_tags_only() -> None:
    assert geak_is_cand_tag("c0_triton")
    assert geak_is_cand_tag("cand_c1_flydsl")
    assert not geak_is_cand_tag("_mxfp8_linear_kernel")
    assert not geak_is_cand_tag("")


# --------------------------------------------------------------------------
# B3 — one admission test, shared with the ledger
# --------------------------------------------------------------------------


def test_stamp_recovers_kind_by_joining_the_ledger() -> None:
    rows = [{"name": "_mxfp8_linear_kernel", "kernel_id": "mxfp8_linear_kernel"}]
    result = {
        "accepted_kernels": [],
        "accepted_heads": [{"short_name": "_mxfp8_linear_kernel", "kind": "authored"}],
    }
    out = _stamp_journey_kind(rows, result)
    assert out[0]["kind"] == "authored"
    assert out[0]["kind_source"] == "result_json"


def test_stamp_joins_through_an_alias() -> None:
    rows = [
        {
            "name": "_fwd_grouped_kernel_stage1",
            "kernel_id": "fwd_grouped_kernel_stage1",
            "aliases": ["c0_triton"],
        }
    ]
    result = {"accepted_kernels": [{"short_name": "c0_triton", "kind": "authored"}]}
    out = _stamp_journey_kind(rows, result)
    assert out[0]["kind_source"] == "result_json"


def test_stamp_drops_only_known_env_rows() -> None:
    rows = [
        {"name": "ck_gemm_a8w8_blockscale_bpreshuffle"},
        {"name": "real_kernel"},
    ]
    result = {
        "accepted_heads": [
            {"short_name": "ck_gemm_a8w8_blockscale_bpreshuffle", "kind": "env"},
            {"short_name": "real_kernel", "kind": "authored"},
        ]
    }
    out = _stamp_journey_kind(rows, result)
    assert [r["name"] for r in out] == ["real_kernel"]


def test_stamp_marks_an_unjoinable_row_absent_and_keeps_it() -> None:
    # Guessing either way is wrong: "authored" inflates the kernel bucket with
    # library picks, "env" deletes real kernels recovered from dead runs.
    rows = [{"name": "orphan_kernel"}]
    out = _stamp_journey_kind(rows, {"accepted_kernels": [], "accepted_heads": []})
    assert len(out) == 1
    assert out[0]["kind"] is None
    assert out[0]["kind_source"] == "absent"


def test_stamp_distinguishes_undeclared_from_absent() -> None:
    rows = [{"name": "listed_kernel"}]
    result = {"accepted_kernels": [{"short_name": "listed_kernel"}]}
    out = _stamp_journey_kind(rows, result)
    assert out[0]["kind"] is None
    assert out[0]["kind_source"] == "result_json_undeclared"


def test_kind_source_counts_cover_every_admitted_row() -> None:
    # The counter must sum to the row count on every path, including the
    # ``result`` path, whose rows carry no kind_source. Measured on the
    # campaign: 3 of 4 rows on that path declare no kind at all, so treating
    # the path as "always declared" empties the counter where it is needed.
    rows: list[Any] = [
        {"name": "a", "kind_source": "result_json"},
        {"name": "b", "kind_source": "absent"},
        {"short_name": "c", "kind": "authored"},  # result path, declared
        {"short_name": "d"},  # result path, undeclared
    ]
    counts = _kind_source_counts(rows)
    assert sum(counts.values()) == len(rows)
    assert counts["result_json"] == 2
    assert counts["absent"] == 1
    assert counts["result_json_undeclared"] == 1


def test_kind_source_counts_empty_for_no_rows() -> None:
    assert _kind_source_counts([]) == {}


# --------------------------------------------------------------------------
# The shared helpers the two collectors now agree on
# --------------------------------------------------------------------------


def test_spec_kind_returns_none_for_absent_and_for_empty() -> None:
    # None and "" must not be distinguishable downstream: both mean undeclared.
    assert geak_spec_kind({"short_name": "k"}) is None
    assert geak_spec_kind({"short_name": "k", "kind": ""}) is None
    assert geak_spec_kind({"short_name": "k", "kind": "  "}) is None
    assert geak_spec_kind({"short_name": "k", "kind": "AUTHORED"}) == "authored"
    assert geak_spec_kind("bare_string") is None


def test_spec_is_env_is_true_only_for_a_declared_env() -> None:
    assert geak_spec_is_env({"short_name": "k", "kind": "env"})
    assert not geak_spec_is_env({"short_name": "k", "kind": "authored"})
    assert not geak_spec_is_env({"short_name": "k"})
    assert not geak_spec_is_env("bare_string")


def test_spec_name_reads_every_spelling_including_a_bare_string() -> None:
    assert geak_spec_name({"short_name": "s", "kernel_id": "k"}) == "s"
    assert geak_spec_name({"kernel_id": "k"}) == "k"
    assert geak_spec_name({"cand_tag": "c0_triton"}) == "c0_triton"
    assert geak_spec_name("  bare_string  ") == "bare_string"
    assert geak_spec_name(None) == ""
