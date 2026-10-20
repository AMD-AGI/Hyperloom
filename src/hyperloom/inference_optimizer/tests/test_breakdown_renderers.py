# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the optimizations and kernel-lifecycle breakdown renderers."""

from __future__ import annotations

from hyperloom.inference_optimizer.breakdown.reporters._renderers import (
    _kernels,
    capability_summary,
    kernel_lifecycle as kl,
    optimizations as opt,
)


# ---- optimizations --------------------------------------------------------
def _ledger(**validation) -> dict:
    validation.setdefault("attribution", {"available": True, "by_source": {}})
    return {"outcome": {"validation": validation}}


def test_a_missing_ledger_is_not_rendered_as_a_session_that_kept_nothing():
    """Silence here is what let a records-less session read as no-gain."""
    out = opt.render({"outcome": {"validation": {"attribution": {"available": False}}}})

    assert out.skipped is False
    assert any("no adoption can be reported" in fact for fact in out.key_facts)
    assert any("absent, not empty" in warning for warning in out.warnings)


def test_a_ledger_that_kept_nothing_is_skipped_not_warned_about():
    # A run that adopted nothing still closed its ledger, so there is no
    # data-quality finding to report -- only an empty section.
    out = opt.render(_ledger(adoption_count=0))

    assert out.skipped is True
    assert out.warnings == []


def test_the_table_splits_adoptions_by_the_source_that_earned_them():
    out = opt.render(
        _ledger(
            adoption_count=3,
            attributed_gain_pct=9.0,
            validated_at_stack_len=3,
            validated_total_gain_pct=10.0,
            attribution={
                "available": True,
                "by_source": {
                    "framework_agent": {"keep_count": 2, "total_gain_pct": 6.0, "unmeasured_keep_count": 1},
                    "kernel": {"keep_count": 1, "total_gain_pct": 3.0, "unmeasured_keep_count": 0},
                },
            },
        )
    )

    assert out.skipped is False
    assert any("3 adoption(s) recorded in the stack ledger." in fact for fact in out.key_facts)
    assert any("sum to +9.0" in fact for fact in out.key_facts)
    # The whole-stack figure names the stack it was measured on, so a reader
    # cannot mistake it for a measurement of the stack that shipped.
    assert any("last measured at length 3" in fact for fact in out.key_facts)
    assert "framework_agent" in out.markdown_block
    assert "kernel" in out.markdown_block


def test_each_source_that_kept_something_becomes_its_own_decision():
    out = opt.render(
        _ledger(
            adoption_count=2,
            attribution={
                "available": True,
                "by_source": {
                    "framework_agent": {"keep_count": 2, "total_gain_pct": 6.0},
                    # A source that ran and kept nothing is a row, not a decision.
                    "kernel": {"keep_count": 0, "total_gain_pct": 0.0},
                },
            },
        )
    )

    subjects = {d.subject for d in out.decisions}
    assert subjects == {"optimizations:framework_agent"}
    assert [d.metric_pct for d in out.decisions] == [6.0]


def test_the_ledgers_own_findings_are_the_sections_warnings():
    # The renderer does not re-derive what is wrong with the ledger; the
    # collector computed it where the rows were read, and this passes it
    # through so one wording serves every reader.
    notes = ["the whole-stack measurement and the sum of the adoptions differ by +4.00 pp"]
    out = opt.render(_ledger(adoption_count=1, notes=notes))

    assert out.warnings == notes


def test_a_reconciling_ledger_carries_no_findings():
    out = opt.render(_ledger(adoption_count=1, notes=[]))

    assert out.warnings == []


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


def _kernel_event(**ext) -> dict:
    """A ``kernel`` timeline event carrying the given ``ext`` blocks."""
    return {"type": "kernel", "ext": ext}


def test_kernel_lifecycle_full_with_adopted_and_residual():
    """Every verdict the gate can reach, plus a long-tail kernel nothing touched."""
    out = kl.render(
        {
            "timeline": [
                _kernel_event(
                    forge={
                        "discovered_kernels": [
                            {
                                "kernel_id": "k1",
                                "name": "gemm",
                                "gpu_pct": 40.0,
                                "duration_us": 100.0,
                                "call_count": 10,
                                "bandwidth_util_pct": 55.0,
                                "compute_util_pct": 70.0,
                                "selected": True,
                            },
                            {"kernel_id": "k2", "name": "attn", "gpu_pct": 20.0, "selected": True},
                            {"kernel_id": "k3", "name": "norm", "gpu_pct": 5.0, "selected": True},
                            # Never selected and never dispatched against: the
                            # residual long tail.
                            {"kernel_id": "k4", "name": "elementwise", "gpu_pct": 1.0, "duration_us": 5.0},
                        ],
                    },
                    attempts=[
                        {"route": "forge", "kernel_id": "k1", "speedup": 1.1, "outcome": "adopted"},
                        {"route": "geak", "kernel_id": "k1", "speedup": 1.3, "outcome": "adopted"},
                        {"route": "geak", "kernel_id": "k2", "speedup": 1.05, "outcome": "adopted"},
                        {"route": "geak", "kernel_id": "k3", "speedup": 0.9, "outcome": "rejected"},
                    ],
                    integrate=[
                        {"kernel_id": "k1", "decision": "KEEP"},
                        {"kernel_id": "k2", "decision": "REVERT"},
                    ],
                )
            ]
        }
    )

    assert out.skipped is False
    assert any("Adopted" in f for f in out.key_facts)
    kinds = {d.kind for d in out.decisions}
    assert "kept" in kinds and "reverted" in kinds and "rejected" in kinds
    assert "residual" in out.markdown_block


def test_kernel_lifecycle_selected_but_no_lane_stall():
    """A kernel the analysis nominated that no route ever dispatched against."""
    out = kl.render(
        {"timeline": [_kernel_event(forge={"discovered_kernels": [{"kernel_id": "k1", "selected": True}]})]}
    )

    assert any("stalled" in f for f in out.key_facts)


def test_kernel_lifecycle_no_decisions_not_attempted():
    out = kl.render({"timeline": [_kernel_event(forge={"discovered_kernels": [{"kernel_id": "k1", "name": "x"}]})]})

    assert any(d.kind == "not_attempted" for d in out.decisions)


def test_kernel_lifecycle_merges_a_kernel_across_two_visits():
    """A kernel discovered in one visit and gated in a later one is one kernel."""
    out = kl.render(
        {
            "timeline": [
                _kernel_event(
                    forge={
                        "discovered_kernels": [{"kernel_id": "k1", "name": "gemm", "gpu_pct": 40.0, "selected": True}],
                    },
                    attempts=[{"route": "forge", "kernel_id": "k1", "speedup": 1.4}],
                ),
                _kernel_event(integrate=[{"kernel_id": "k1", "decision": "KEEP"}]),
            ]
        }
    )

    assert "1 kernel(s) detected" in out.key_facts[0]
    assert "adopted=1" in out.key_facts[0]


# ---- capability counting --------------------------------------------------


def test_a_specialist_dispatch_seen_as_row_and_run_is_one_round():
    breakdown = {
        "timeline": [
            {
                "type": "phase",
                "ext": {
                    "actions": {
                        "rows": [{"action": "specialist", "task_id": "t1"}, {"action": "specialist", "task_id": "t2"}]
                    }
                },
            },
            {
                "type": "framework_agent",
                "ext": {
                    "runs": [
                        {"role": "discovery", "run_id": "t1"},
                        {"role": None, "run_id": "t2"},
                        {"role": "authoring", "run_id": "t3"},
                        {"role": "config", "run_id": "t4"},
                    ]
                },
            },
        ]
    }

    assert capability_summary.capability_rows(breakdown)["specialist"]["extras"]["rounds"] == 3


def _attempt(outcome: str, decision: str | None = None, gain: float | None = None) -> dict:
    e2e = {"decision": decision, "e2e_gain_pct": gain} if decision else {}
    return {"source_kind": "geak_authored_kernel", "route": "geak", "kernel_id": "k", "outcome": outcome, "e2e": e2e}


def _counters(*attempts: dict) -> dict:
    breakdown = {"timeline": [_kernel_event(attempts=list(attempts))]}
    return _kernels.source_counters(breakdown, _kernels.GEAK_SOURCES)


def test_each_attempt_lands_in_exactly_one_outcome_bucket():
    got = _counters(
        _attempt("rejected", "REJECTED"),
        _attempt("rejected"),
        _attempt("rejected"),
        _attempt("rejected", "KEEP"),
        _attempt("rejected", "REVERT"),
        _attempt("adopted", "KEEP"),
        _attempt("adopted"),
    )

    buckets = {k: got[k] for k in ("keeps", "micro_only_keeps", "reverts", "rejected", "failed")}
    assert buckets == {"keeps": 1, "micro_only_keeps": 1, "reverts": 1, "rejected": 4, "failed": 0}
    assert got["attempted"] == 7


def test_capability_e2e_gain_is_the_best_kept_gain_only():
    got = _counters(_attempt("adopted", "KEEP", 3.0), _attempt("rejected", "REVERT", 9.0))
    assert got["e2e_gain_pct"] == 3.0

    assert _counters(_attempt("rejected", "REVERT", 9.0))["e2e_gain_pct"] is None
