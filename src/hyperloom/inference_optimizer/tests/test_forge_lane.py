# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""forge is an independent kernel backend lane.

These tests pin the contract that forge gets its own ``forge_invocations`` section, its own capability-summary row, its
own attribution bucket, and its own ``adopted_by`` value — everywhere the
breakdown splits invocations by lane.
"""

from __future__ import annotations

from hyperloom.inference_optimizer.breakdown import collectors
from hyperloom.inference_optimizer.breakdown.recorder import instrument


def test_invocation_section_forge_is_own_lane() -> None:
    assert instrument._invocation_section("geak") == "geak_invocations"
    assert instrument._invocation_section("forge") == "forge_invocations"
    assert instrument._invocation_section("claude") is None
    assert instrument._invocation_section("codex") is None


def test_capability_summary_has_distinct_forge_row() -> None:
    forge_invs = [{"kernel_id": "k1", "decision": "KEEP", "micro_speedup": 1.4}]
    cap = collectors.collect_capability_summary(
        {},
        [],
        [],
        forge_invocations=forge_invs,
    )
    assert cap["forge"]["attempts"] == 1
    assert cap["forge"]["keeps"] == 1
    assert cap["forge"]["status"] == "kept"


def test_optimized_kernels_includes_forge_attempts() -> None:
    forge_invs = [
        {
            "kernel_id": "k1",
            "backend": "forge",
            "decision": "KEEP",
            "micro_speedup": 1.6,
            "attempt_id": "a1",
            "ts": "2026-06-16T00:00:00Z",
        }
    ]
    rows = collectors._collect_optimized_kernels([], {}, forge_invs)
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
        state,
        [],
        adopted,
        [],
        forge_invocations=forge_invs,
    )
    sb = out["source_breakdown"]
    assert sb["forge_pct_of_total"] == 10.0


def test_attribution_backward_compatible_without_forge() -> None:
    out = collectors.collect_attribution({}, [], [], [])
    assert "source_breakdown" in out
    assert out["source_breakdown"]["forge_pct_of_total"] == 0.0


def test_capability_not_reverse_inferred_from_optimization_stack() -> None:
    # A session whose optimization_stack lists an "explore" entry but has NO
    # explore_attempts record must report explore as not_attempted (the stack
    # entry may be seeded / warm-replayed), never fabricated as kept.
    state = {
        "optimization_stack": [
            {"action": "explore", "variant_name": "v1", "source": "seeded_from_current_best"}
        ],
        # no explore_attempts key => no real attempt evidence
    }
    cap = collectors.collect_capability_summary(state, [], [])
    assert cap["explore"]["status"] == "not_attempted"
    assert cap["explore"]["attempts"] == 0
    assert cap["explore"]["keeps"] == 0


def test_kernel_gain_without_forge_keep_is_unattributed_not_credited_to_forge() -> None:
    # Kernel-lane gain exists but there is NO Forge KEEP evidence. The gain must
    # stay unattributed rather than being reverse-inferred onto Forge.
    state = {"gain_per_stack_entry": [{"action": "kernel_opt", "delta_pct": 12.0}]}
    out = collectors.collect_attribution(
        state,
        [],
        [],  # no adopted kernels
        [],
        forge_invocations=[],  # no Forge KEEP
    )
    sb = out["source_breakdown"]
    assert sb["forge_pct_of_total"] == 0.0
    assert sb["kernel_unattributed_pct_of_total"] == 12.0
