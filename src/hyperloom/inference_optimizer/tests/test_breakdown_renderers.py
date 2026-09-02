# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the data-provenance and kernel-lifecycle breakdown
renderers."""

from __future__ import annotations

from hyperloom.inference_optimizer.breakdown.reporters._renderers import (
    data_provenance as dp,
    kernel_lifecycle as kl,
    optimizations as opt,
)


# ---- optimizations --------------------------------------------------------
def test_a_missing_read_model_is_not_rendered_as_an_empty_one():
    """Silence here is what let a records-less session read as no-gain."""
    out = opt.render(
        {
            "optimizations": {
                "available": False,
                "unavailable_reason": "no operations were recorded for this session",
                "entries": [],
                "validation": {"method": "unavailable"},
            }
        }
    )

    assert out.skipped is False
    assert any("no operations were recorded" in fact for fact in out.key_facts)
    assert any("absent, not empty" in warning for warning in out.warnings)


def test_the_table_says_what_it_does_not_add_up_to():
    out = opt.render(
        {
            "optimizations": {
                "entries": [{"validated": True, "gain_pct": 9.0}],
                "summary_by_source": {"explore": {"keeps": 1, "total_gain_pct": 9.0}},
                "validation": {
                    "validated_total_gain_pct": 10.0,
                    "attributed_total_gain_pct": 9.0,
                    "unattributed_gain_pct": 1.0,
                    "unmeasured_keep_count": 1,
                    "projected_keep_count": 2,
                    "stale_evidence_count": 3,
                },
            }
        }
    )

    joined = " ".join(out.warnings)
    assert "belongs to no adopted step" in joined
    assert "1 adopted step(s) recorded neither" in joined
    assert "2 adopted step(s) recorded no finishing throughput" in joined
    assert "3 adoption(s) cite measurements" in joined


def test_a_gain_with_no_owner_says_whether_it_is_really_ownerless():
    """The unattributed figure is only trustworthy if nothing went missing.

    A change recorded as integrated with no adoption behind it puts its gain in
    the same bucket, so the reader has to be told the bucket is overstated
    rather than left to read it as drift.
    """
    out = opt.render(
        {
            "optimizations": {
                "entries": [{"validated": True, "gain_pct": 9.0}],
                "summary_by_source": {"explore": {"keeps": 1, "total_gain_pct": 9.0}},
                "validation": {
                    "validated_total_gain_pct": 19.0,
                    "attributed_total_gain_pct": 9.0,
                    "unattributed_gain_pct": 10.0,
                    "unclaimed_integration_count": 1,
                },
            }
        }
    )

    joined = " ".join(out.warnings)
    assert "1 change(s) are recorded as integrated with nothing crediting them" in joined
    assert "overstates" in joined


def test_a_clean_ledger_carries_no_reconciliation_notes():
    out = opt.render(
        {
            "optimizations": {
                "entries": [{"validated": True, "gain_pct": 10.0}],
                "summary_by_source": {"explore": {"keeps": 1, "total_gain_pct": 10.0}},
                "validation": {
                    "method": "recorded_session_validation",
                    "validated_total_gain_pct": 10.0,
                    "ledger_total_gain_pct": 10.0,
                    "reconciliation_gap_pct": 0.0,
                    "attributed_total_gain_pct": 10.0,
                    "unattributed_gain_pct": 0.0,
                    "unmeasured_keep_count": 0,
                    "projected_keep_count": 0,
                    "stale_evidence_count": 0,
                    "unscored_keep_count": 0,
                },
            }
        }
    )

    assert out.warnings == []


def test_a_total_that_only_checks_itself_says_so():
    """Summing the ledger and calling it the session total proves nothing."""
    out = opt.render(
        {
            "optimizations": {
                "entries": [{"validated": True, "gain_pct": 10.0}],
                "summary_by_source": {"explore": {"keeps": 1, "total_gain_pct": 10.0}},
                "validation": {
                    "method": "ledger_sum",
                    "validated_total_gain_pct": 10.0,
                    "ledger_total_gain_pct": 10.0,
                    "reconciliation_gap_pct": None,
                },
            }
        }
    )

    assert any("cannot be checked against each other" in w for w in out.warnings)


def test_a_ledger_that_disagrees_with_the_run_is_reported():
    out = opt.render(
        {
            "optimizations": {
                "entries": [{"validated": True, "gain_pct": 8.0}],
                "summary_by_source": {"explore": {"keeps": 1, "total_gain_pct": 8.0}},
                "validation": {
                    "method": "recorded_session_validation",
                    "validated_total_gain_pct": 12.0,
                    "ledger_total_gain_pct": 8.0,
                    "reconciliation_gap_pct": 4.0,
                    "unscored_keep_count": 1,
                },
            }
        }
    )

    joined = " ".join(out.warnings)
    assert "adopted steps add up to" in joined
    assert "no accuracy gate having ruled on them" in joined


# ---- data_provenance ------------------------------------------------------
def test_data_provenance_skipped_when_empty():
    out = dp.render({})
    assert out.skipped is True


def test_sources_summary_variants():
    assert dp._sources_summary([]) == "—"
    summ = dp._sources_summary(
        [
            {"found": True, "required": True},
            {"found": False, "required": True},
            {"found": True, "required": False},
        ]
    )
    assert "found" in summ and "required" in summ


def test_data_provenance_full_table():
    out = dp.render(
        {
            "data_provenance": [
                {
                    "section": "roofline",
                    "status": "empty",
                    "populated": False,
                    "sources": [{"found": False, "required": True}],
                    "missing_required": ["trace.json"],
                },
                {
                    "section": "sweep",
                    "status": "partial",
                    "populated": True,
                    "sources": [{"found": True, "required": True}],
                    "missing_required": [],
                },
                {"section": "kernels", "status": "ok", "populated": True, "sources": []},
                "not-a-dict",
            ]
        }
    )
    assert out.skipped is False
    assert "roofline" in out.markdown_block
    assert any("empty" in f for f in out.key_facts)
    assert any("partial" in f for f in out.key_facts)


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
    assert "att" in kl._lane_summary({"best_speedup": 1.2, "attempts": 3, "decision": "KEEP"})


def test_kernel_lifecycle_full_with_adopted_and_residual():
    detected = [
        {
            "kernel_id": "k1",
            "name": "gemm",
            "gpu_pct": 40.0,
            "duration_us": 100.0,
            "call_count": 10,
            "selected_for_optimization": True,
            "geak": {"best_speedup": 1.3, "attempts": 2, "decision": "KEEP"},
            "forge": {"best_speedup": 1.1, "attempts": 1, "decision": "REVERT"},
            "final_decision": "kept",
            "adopted_by": "geak",
            "bandwidth_util_pct": 55.0,
            "compute_util_pct": 70.0,
        },
        {"kernel_id": "k2", "name": "attn", "final_decision": "reverted", "geak": {"attempts": 1}},
        {"kernel_id": "k3", "final_decision": "rejected"},
        # residual long-tail kernel
        {
            "kernel_id": "k4",
            "name": "elementwise",
            "gpu_pct": 1.0,
            "duration_us": 5.0,
            "final_decision": "not_optimized",
        },
        "anon-string-id",  # coerced into {"kernel_id": ...}
        {"name": "no-id-dropped"},  # filtered out
    ]
    out = kl.render({"kernel_lifecycle": {"detected": detected}})
    assert out.skipped is False
    assert any("adopted" in f.lower() or "Adopted" in f for f in out.key_facts)
    kinds = {d.kind for d in out.decisions}
    assert "kept" in kinds and "reverted" in kinds and "rejected" in kinds
    assert "residual" in out.markdown_block


def test_kernel_lifecycle_selected_but_no_lane_stall():
    out = kl.render(
        {
            "kernel_lifecycle": {
                "detected": [
                    {"kernel_id": "k1", "selected_for_optimization": True},
                ]
            }
        }
    )
    assert any("stalled" in f for f in out.key_facts)


def test_kernel_lifecycle_no_decisions_not_attempted():
    out = kl.render(
        {
            "kernel_lifecycle": {
                "detected": [
                    {"kernel_id": "k1", "name": "x"},
                ]
            }
        }
    )
    assert any(d.kind == "not_attempted" for d in out.decisions)
