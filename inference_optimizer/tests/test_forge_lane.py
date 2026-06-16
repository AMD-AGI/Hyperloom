# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""forge is an independent third lane (geak / oob / forge).

These tests pin the contract that forge is NOT folded into the oob lane: it
gets its own ``forge_invocations`` section, its own capability-summary row, its
own attribution bucket, and its own ``adopted_by`` value — everywhere the
breakdown splits invocations by lane.
"""
from __future__ import annotations

from inference_optimizer.breakdown import collectors
from inference_optimizer.breakdown.recorder import instrument


def test_invocation_section_forge_is_own_lane() -> None:
    assert instrument._invocation_section("geak") == "geak_invocations"
    assert instrument._invocation_section("forge") == "forge_invocations"
    # claude/codex stay on oob; forge is never routed there.
    assert instrument._invocation_section("claude") == "oob_invocations"
    assert instrument._invocation_section("codex") == "oob_invocations"


def test_capability_summary_has_distinct_forge_row() -> None:
    forge_invs = [{"kernel_id": "k1", "decision": "KEEP", "micro_speedup": 1.4}]
    cap = collectors.collect_capability_summary(
        {}, [], [], [], forge_invocations=forge_invs,
    )
    # forge gets its own row with its own KEEP tally.
    assert cap["forge"]["attempts"] == 1
    assert cap["forge"]["keeps"] == 1
    assert cap["forge"]["status"] == "kept"
    # oob stays empty — forge was not folded into it.
    assert cap["oob"]["attempts"] == 0
    assert cap["oob"]["keeps"] == 0


def test_optimized_kernels_includes_forge_attempts() -> None:
    forge_invs = [{
        "kernel_id": "k1", "backend": "forge", "decision": "KEEP",
        "micro_speedup": 1.6, "attempt_id": "a1", "ts": "2026-06-16T00:00:00Z",
    }]
    rows = collectors._collect_optimized_kernels([], [], {}, forge_invs)
    by_kid = {r["kernel_id"]: r for r in rows}
    assert "k1" in by_kid
    assert by_kid["k1"]["backend"] == "forge"
    assert by_kid["k1"]["best_micro_speedup"] == 1.6
    assert by_kid["k1"]["successful_attempts"] == 1


def test_attribution_splits_kernel_gain_to_forge() -> None:
    state = {"gain_per_stack_entry": [{"action": "kernel_opt", "delta_pct": 10.0}]}
    forge_invs = [{"kernel_id": "k1", "decision": "KEEP"}]
    adopted = [{"kernel_id": "k1", "e2e_gain_pct": 10.0}]
    out = collectors.collect_attribution(
        state, [], [], adopted, [], forge_invocations=forge_invs,
    )
    sb = out["source_breakdown"]
    # The adopted kernel's gain is attributed to forge, not oob.
    assert sb["forge_pct_of_total"] == 10.0
    assert sb["oob_pct_of_total"] == 0.0


def test_attribution_backward_compatible_without_forge() -> None:
    # Legacy 5-arg call (no forge) still works and never errors.
    out = collectors.collect_attribution({}, [], [], [], [])
    assert "source_breakdown" in out
    assert out["source_breakdown"]["forge_pct_of_total"] == 0.0
