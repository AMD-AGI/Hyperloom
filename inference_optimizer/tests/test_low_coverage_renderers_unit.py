# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for the data-provenance, sweep, and kernel-lifecycle breakdown
renderers (previously near-uncovered)."""
from __future__ import annotations

from inference_optimizer.breakdown.reporters._renderers import (
    data_provenance as dp,
    kernel_lifecycle as kl,
    sweep as sw,
)


# ---- data_provenance ------------------------------------------------------
def test_data_provenance_skipped_when_empty():
    out = dp.render({})
    assert out.skipped is True


def test_sources_summary_variants():
    assert dp._sources_summary([]) == "—"
    summ = dp._sources_summary([
        {"found": True, "required": True},
        {"found": False, "required": True},
        {"found": True, "required": False},
    ])
    assert "found" in summ and "required" in summ


def test_data_provenance_full_table():
    out = dp.render({"data_provenance": [
        {"section": "roofline", "status": "empty", "populated": False,
         "sources": [{"found": False, "required": True}],
         "missing_required": ["trace.json"]},
        {"section": "sweep", "status": "partial", "populated": True,
         "sources": [{"found": True, "required": True}],
         "missing_required": []},
        {"section": "kernels", "status": "ok", "populated": True,
         "sources": []},
        "not-a-dict",  # skipped by the dict guard
    ]})
    assert out.skipped is False
    assert "roofline" in out.markdown_block
    assert any("empty" in f for f in out.key_facts)
    assert any("partial" in f for f in out.key_facts)


# ---- sweep ----------------------------------------------------------------
def test_sweep_skipped_when_no_variants():
    out = sw.render({})
    assert out.skipped is True
    assert out.decisions[0].kind == "not_attempted"


def test_sweep_full_with_best_point():
    variants = [
        {"conc": 32, "isl": 128, "osl": 128, "status": "success",
         "output_throughput": 800.0, "ttft": 50, "e2el": 1000},
        {"conc": 64, "isl": 128, "osl": 128, "status": "success",
         "output_throughput": 950.0, "ttft": 40, "e2el": 900},
        {"conc": 128, "isl": 128, "osl": 128, "status": "failed",
         "output_throughput": None, "error": "oom" * 40},
    ]
    out = sw.render({"sweep": {"grid_size": 3, "all_variants": variants}})
    assert out.skipped is False
    assert out.decisions[0].kind == "kept"
    assert any("Best point" in f for f in out.key_facts)


def test_sweep_truncates_over_max_rows():
    variants = [
        {"conc": i, "status": "success", "output_throughput": float(i)}
        for i in range(sw._MAX_ROWS + 5)
    ]
    out = sw.render({"sweep": {"all_variants": variants}})
    assert f"first {sw._MAX_ROWS}" in out.markdown_block


def test_sweep_non_dict_variant_coerced():
    out = sw.render({"sweep": {"all_variants": ["raw-variant"]}})
    assert out.skipped is False


def test_sweep_ot_handles_bad_value():
    assert sw.render  # ensure module import
    # exercise the inner sort key indirectly via a non-numeric throughput
    out = sw.render({"sweep": {"all_variants": [
        {"status": "success", "output_throughput": "NaNish"},
    ]}})
    assert out.skipped is False


# ---- kernel_lifecycle -----------------------------------------------------
def test_kernel_lifecycle_skipped_when_none():
    out = kl.render({})
    assert out.skipped is True


def test_short_name_and_fmt_speedup_and_lane():
    long = "k" * 100
    assert "..." in kl._short_name(long)
    assert kl._short_name("") == ""
    assert kl._short_name("short") == "short"
    assert kl._fmt_speedup(None) == "—"
    assert kl._fmt_speedup("bad") == "—"
    assert kl._fmt_speedup(1.25) == "1.25x"
    assert kl._lane_summary(None) == "—"
    assert "att" in kl._lane_summary(
        {"best_speedup": 1.2, "attempts": 3, "decision": "KEEP"})


def test_kernel_lifecycle_full_with_adopted_and_residual():
    detected = [
        {"kernel_id": "k1", "name": "gemm", "gpu_pct": 40.0,
         "duration_us": 100.0, "call_count": 10,
         "selected_for_optimization": True,
         "geak": {"best_speedup": 1.3, "attempts": 2, "decision": "KEEP"},
         "oob": {"best_speedup": 1.1, "attempts": 1, "decision": "REVERT"},
         "final_decision": "kept", "adopted_by": "geak",
         "bandwidth_util_pct": 55.0, "compute_util_pct": 70.0},
        {"kernel_id": "k2", "name": "attn", "final_decision": "reverted",
         "geak": {"attempts": 1}},
        {"kernel_id": "k3", "final_decision": "rejected"},
        # residual long-tail kernel (not selected / touched / decided)
        {"kernel_id": "k4", "name": "elementwise", "gpu_pct": 1.0,
         "duration_us": 5.0, "final_decision": "not_optimized"},
        "anon-string-id",  # coerced into {"kernel_id": ...}
        {"name": "no-id-dropped"},  # filtered out (no kernel_id)
    ]
    out = kl.render({"kernel_lifecycle": {"detected": detected}})
    assert out.skipped is False
    assert any("adopted" in f.lower() or "Adopted" in f for f in out.key_facts)
    kinds = {d.kind for d in out.decisions}
    assert "kept" in kinds and "reverted" in kinds and "rejected" in kinds
    assert "residual" in out.markdown_block


def test_kernel_lifecycle_selected_but_no_lane_stall():
    out = kl.render({"kernel_lifecycle": {"detected": [
        {"kernel_id": "k1", "selected_for_optimization": True},
    ]}})
    assert any("stalled" in f for f in out.key_facts)


def test_kernel_lifecycle_no_decisions_not_attempted():
    out = kl.render({"kernel_lifecycle": {"detected": [
        {"kernel_id": "k1", "name": "x"},
    ]}})
    assert any(d.kind == "not_attempted" for d in out.decisions)
