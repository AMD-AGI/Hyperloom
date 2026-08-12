# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json

from hyperloom.inference_optimizer.breakdown import exporter
from hyperloom.inference_optimizer.breakdown.collectors import (
    collect_attribution,
    collect_optimizations,
    collect_recorded_optimizations,
    collect_v4_optimizations,
)


def test_collect_optimizations_unifies_warm_framework_and_explore():
    state = {
        "session_id": "s1",
        "baseline_tput": 100.0,
        "cumulative_gain_validated": 70.0,
        "cumulative_gain_validated_stack_len": 3,
        "optimization_stack": [
            {
                "action": "replay_warm_recipe",
                "variant_name": "warm",
                "tput": 150.0,
                "ts": "1970-01-01T00:00:10+00:00",
            },
            {
                "action": "integrate_patch",
                "source_phase": "FRAMEWORK_AGENT",
                "framework_agent_authoring": True,
                "variant_name": "framework-patch",
                "tput": 160.0,
                "ts": "1970-01-01T00:00:20+00:00",
                "patch_path": "/tmp/framework.patch",
            },
            {
                "action": "integrate_patch",
                "source_phase": "EXPLORE",
                "domain": "serving_specialist",
                "variant_name": "explore-config",
                "tput": 170.0,
                "ts": "1970-01-01T00:00:30+00:00",
                "operation_kind": "env",
            },
        ],
        # Legacy cumulative ledger: the attribution collector promotes it.
        "gain_per_stack_entry": [50.0, 60.0, 70.0],
        "phase_history": [
            {"to_phase": "PRELUDE", "ts_unix": 0.0},
            {"to_phase": "FRAMEWORK_AGENT", "ts_unix": 15.0},
            {"to_phase": "EXPLORE", "ts_unix": 25.0},
        ],
    }
    warnings: list[str] = []
    attribution = collect_attribution(state, [], [], warnings)

    result = collect_optimizations(state, attribution, [], [], warnings)

    assert [entry["source"] for entry in result["entries"]] == [
        "warm_replay",
        "framework_agent",
        "explore",
    ]
    assert [entry["gain_pct"] for entry in result["entries"]] == [50.0, 10.0, 10.0]
    assert result["entries"][1]["action"] == "integrate_patch"
    assert result["entries"][1]["variant_name"] == "framework-patch"
    assert result["entries"][1]["optimization_kind"] == "framework_patch"
    assert result["entries"][2]["optimization_kind"] == "env"
    summary = result["summary_by_source"]
    assert summary["warm_replay"] == {"keeps": 1, "total_gain_pct": 50.0}
    assert summary["framework_agent"] == {"keeps": 1, "total_gain_pct": 10.0}
    assert summary["explore"] == {"keeps": 1, "total_gain_pct": 10.0}
    source_breakdown = attribution["source_breakdown"]
    assert source_breakdown["framework_pct_of_total"] == 10.0
    assert source_breakdown["explore_pct_of_total"] == 10.0
    assert source_breakdown["unattributed_pct_of_total"] == 0.0


def test_fusion_after_warm_replay_is_attributed_to_kernel_agent():
    state = {
        "session_id": "warm-then-fusion",
        "baseline_tput": 100.0,
        "cumulative_gain_validated": 80.0,
        "cumulative_gain_validated_stack_len": 2,
        "optimization_stack": [
            {
                "action": "replay_warm_recipe",
                "source_phase": "PRELUDE",
                "variant_name": "warm",
                "tput": 120.0,
                "gain_pct": 20.0,
                "ts": "1970-01-01T00:00:10+00:00",
            },
            {
                "action": "fusion",
                "source_phase": "KERNEL_AGENT",
                "variant_name": "forge_fusion",
                "backend": "forge",
                "tput": 180.0,
                # The stack entry retains integrate's increment relative to
                # warm=120; the authoritative ledger remains baseline-relative.
                "gain_pct": 50.0,
                "ts": "1970-01-01T00:00:20+00:00",
            },
        ],
        "gain_per_stack_entry": [20.0, 80.0],
        "phase_history": [
            {"to_phase": "PRELUDE", "ts_unix": 0.0},
            {"to_phase": "KERNEL_AGENT", "ts_unix": 15.0},
        ],
    }
    warnings: list[str] = []

    attribution = collect_attribution(state, [], [], warnings)
    result = collect_optimizations(state, attribution, [], [], warnings)

    source_breakdown = attribution["source_breakdown"]
    assert source_breakdown["replay_warm_recipe_pct_of_total"] == 20.0
    assert source_breakdown["kernel_unattributed_pct_of_total"] == 60.0
    assert source_breakdown["unattributed_pct_of_total"] == 0.0
    assert attribution["phase_breakdown"]["kernel_agent"]["total_gain_pct"] == 60.0
    assert not any("actions: fusion" in warning for warning in warnings)
    assert [entry["source"] for entry in result["entries"]] == [
        "warm_replay",
        "kernel_agent",
    ]
    assert result["entries"][1]["optimization_kind"] == "kernel_fusion"


def test_prebaseline_enablement_is_not_framework_gain_before_warm_replay():
    """A runnable-baseline patch is config provenance, not a Framework gain."""
    state = {
        "session_id": "enablement-before-warm",
        "baseline_tput": 100.0,
        "cumulative_gain_validated": 10.0,
        "cumulative_gain_validated_stack_len": 2,
        "optimization_stack": [
            {
                "action": "integrate_patch",
                "source_phase": "PRELUDE",
                "baseline_enablement": True,
                "attribution_eligible": False,
                "variant_name": "make-model-runnable",
                "tput": 100.0,
                "ts": "1970-01-01T00:00:10+00:00",
            },
            {
                "action": "replay_warm_recipe",
                "source_phase": "PRELUDE",
                "variant_name": "warm",
                "tput": 110.0,
                "ts": "1970-01-01T00:00:20+00:00",
            },
        ],
        "gain_per_stack_entry": [None, 10.0],
        "phase_history": [
            {"to_phase": "PRELUDE", "ts_unix": 0.0},
        ],
    }
    warnings: list[str] = []

    attribution = collect_attribution(state, [], [], warnings)
    result = collect_optimizations(state, attribution, [], [], warnings)

    assert attribution["gain_per_stack_entry"][1]["cum_gain_before"] == 0.0
    assert attribution["source_breakdown"]["framework_pct_of_total"] == 0.0
    assert attribution["source_breakdown"]["replay_warm_recipe_pct_of_total"] == 10.0
    assert [entry["source"] for entry in result["entries"]] == ["warm_replay"]
    assert result["entries"][0]["throughput_before"] == 100.0
    assert result["entries"][0]["gain_pct"] == 10.0


def test_integrate_patch_with_unknown_phase_is_visible_as_unattributed():
    state = {
        "session_id": "unknown-integrate-owner",
        "baseline_tput": 100.0,
        "cumulative_gain_validated": 10.0,
        "cumulative_gain_validated_stack_len": 1,
        "optimization_stack": [
            {
                "action": "integrate_patch",
                "source_phase": "",
                "variant_name": "unknown-owner",
                "tput": 110.0,
                "ts": "1970-01-01T00:00:10+00:00",
            },
        ],
        "gain_per_stack_entry": [10.0],
        "phase_history": [{"to_phase": "KERNEL_AGENT", "ts_unix": 0.0}],
    }
    warnings: list[str] = []

    attribution = collect_attribution(state, [], [], warnings)
    result = collect_optimizations(state, attribution, [], [], warnings)

    assert attribution["source_breakdown"]["unattributed_pct_of_total"] == 10.0
    assert any("reported as unattributed" in note for note in attribution["notes"])
    assert any("reported as unattributed" in warning for warning in warnings)
    assert result["entries"][0]["source"] == "unattributed"
    assert result["summary_by_source"]["unattributed"] == {
        "keeps": 1,
        "total_gain_pct": 10.0,
    }
    assert result["entries"][0]["source_method"] == "unknown"
    assert result["entries"][0]["optimization_kind"] != "kernel_patch"
    assert result["validation"]["attributed_total_gain_pct"] == 0.0
    assert result["validation"]["attribution_gap_pct"] == 10.0
    assert attribution["phase_breakdown"]["unattributed"]["total_gain_pct"] == 10.0
    assert attribution["phase_breakdown"]["kernel_agent"]["total_gain_pct"] == 0.0


def test_ownerless_integrate_patch_ignores_framework_and_explore_timeline():
    for timeline_phase in ("EXPLORE", "FRAMEWORK_AGENT"):
        state = {
            "session_id": f"ownerless-{timeline_phase.lower()}",
            "baseline_tput": 100.0,
            "cumulative_gain_validated": 10.0,
            "cumulative_gain_validated_stack_len": 1,
            "optimization_stack": [
                {
                    "action": "integrate_patch",
                    "variant_name": "ownerless",
                    "tput": 110.0,
                    "ts": "1970-01-01T00:00:10+00:00",
                },
            ],
            "gain_per_stack_entry": [10.0],
            "phase_history": [{"to_phase": timeline_phase, "ts_unix": 0.0}],
        }
        warnings: list[str] = []

        attribution = collect_attribution(state, [], [], warnings)
        result = collect_optimizations(state, attribution, [], [], warnings)

        assert attribution["source_breakdown"]["unattributed_pct_of_total"] == 10.0
        assert attribution["source_breakdown"]["explore_pct_of_total"] == 0.0
        assert attribution["source_breakdown"]["framework_pct_of_total"] == 0.0
        assert attribution["phase_breakdown"]["unattributed"]["total_gain_pct"] == 10.0
        assert result["entries"][0]["source"] == "unattributed"
        assert result["validation"]["attribution_gap_pct"] == 10.0
        assert result["validation"]["source_breakdown"]["unattributed_gain_pct"] == 10.0
        assert any("reported as unattributed" in warning for warning in warnings)


def test_integrate_patch_with_only_kernel_completion_phase_is_unattributed():
    state = {
        "session_id": "kernel-integrate-owner",
        "baseline_tput": 100.0,
        "cumulative_gain_validated": 10.0,
        "cumulative_gain_validated_stack_len": 1,
        "optimization_stack": [
            {
                "action": "integrate_patch",
                "source_phase": "KERNEL_AGENT",
                "backend": "forge",
                "engine": "forge",
                "final_overlay": "/tmp/not-kernel-ownership.patch",
                "variant_name": "unknown-owner",
                "tput": 110.0,
                "ts": "1970-01-01T00:00:10+00:00",
            },
        ],
        "gain_per_stack_entry": [10.0],
    }
    warnings: list[str] = []

    attribution = collect_attribution(state, [], [], warnings)
    result = collect_optimizations(state, attribution, [], [], warnings)

    assert attribution["source_breakdown"]["kernel_unattributed_pct_of_total"] == 0.0
    assert attribution["source_breakdown"]["unattributed_pct_of_total"] == 10.0
    assert result["entries"][0]["source"] == "unattributed"
    assert result["entries"][0]["source_method"] == "unknown"
    assert result["entries"][0]["optimization_kind"] != "kernel_patch"
    assert result["validation"]["attributed_total_gain_pct"] == 0.0
    assert result["validation"]["attribution_gap_pct"] == 10.0
    assert attribution["phase_breakdown"]["unattributed"]["total_gain_pct"] == 10.0
    assert attribution["phase_breakdown"]["kernel_agent"]["total_gain_pct"] == 0.0


def test_integrate_patch_domain_survives_legacy_ledger_and_owns_all_breakdowns():
    state = {
        "session_id": "cross-phase-serving-config",
        "baseline_tput": 100.0,
        "cumulative_gain_validated": 10.0,
        "cumulative_gain_validated_stack_len": 1,
        "optimization_stack": [
            {
                "action": "integrate_patch",
                "source_phase": "KERNEL_AGENT",
                "domain": "serving_specialist",
                "operation_kind": "param",
                "variant_name": "fp8-per-channel-int4-quickreduce-stack",
                "candidate_extra_server_args": "--quantization fp8_per_channel",
                "tput": 110.0,
                "ts": "1970-01-01T00:00:10+00:00",
            },
        ],
        "gain_per_stack_entry": [10.0],
        "phase_history": [{"to_phase": "KERNEL_AGENT", "ts_unix": 0.0}],
    }
    warnings: list[str] = []

    attribution = collect_attribution(state, [], [], warnings)
    result = collect_optimizations(state, attribution, [], [], warnings)

    assert attribution["source_breakdown"]["explore_pct_of_total"] == 10.0
    assert attribution["source_breakdown"]["kernel_unattributed_pct_of_total"] == 0.0
    assert result["entries"][0]["source"] == "explore"
    assert result["entries"][0]["optimization_kind"] == "param"
    assert result["entries"][0]["kernel_id"] is None
    assert attribution["phase_breakdown"]["explore"]["total_gain_pct"] == 10.0
    assert attribution["phase_breakdown"]["explore"]["by_domain"] == {
        "serving_specialist": 10.0,
    }
    assert attribution["phase_breakdown"]["kernel_agent"]["total_gain_pct"] == 0.0


def test_framework_proposal_integrate_patch_overrides_kernel_completion_phase():
    state = {
        "session_id": "cross-phase-framework-config",
        "baseline_tput": 100.0,
        "cumulative_gain_validated": 10.0,
        "cumulative_gain_validated_stack_len": 1,
        "optimization_stack": [
            {
                "action": "integrate_patch",
                "source_phase": "KERNEL_AGENT",
                "framework_agent_authoring": True,
                "provenance": "specialist:serving_specialist",
                "domain": "serving_specialist",
                "operation_kind": "param",
                "variant_name": "fp8-per-channel-int4-quickreduce-stack",
                "candidate_extra_server_args": "--quantization fp8_per_channel",
                "tput": 110.0,
                "ts": "1970-01-01T00:00:10+00:00",
            },
        ],
        "gain_per_stack_entry": [10.0],
    }

    attribution = collect_attribution(state, [], [], [])
    result = collect_optimizations(state, attribution, [], [], [])

    assert attribution["source_breakdown"]["framework_pct_of_total"] == 10.0
    assert result["entries"][0]["source"] == "framework_agent"
    assert result["entries"][0]["optimization_kind"] == "serving_config"


def test_direct_framework_action_ignores_delayed_completion_phase():
    state = {
        "session_id": "cross-phase-direct-framework",
        "baseline_tput": 100.0,
        "cumulative_gain_validated": 10.0,
        "cumulative_gain_validated_stack_len": 1,
        "optimization_stack": [
            {
                "action": "framework",
                "source_phase": "KERNEL_AGENT",
                "variant_name": "PR:123",
                "tput": 110.0,
                "ts": "1970-01-01T00:00:10+00:00",
            },
        ],
        "gain_per_stack_entry": [10.0],
    }

    attribution = collect_attribution(state, [], [], [])
    result = collect_optimizations(state, attribution, [], [], [])

    assert attribution["source_breakdown"]["framework_pct_of_total"] == 10.0
    assert result["entries"][0]["source"] == "framework_agent"
    assert result["entries"][0]["source_method"] == "action_family"


def test_framework_stack_label_is_credited_by_both_collectors():
    """Both readers must credit the exact label the promote path stamps.

    The fixture takes its label from the writer's own constant instead of
    spelling it out, so renaming one side without the other fails here rather
    than silently routing FRAMEWORK gain into ``unattributed``. The entry
    deliberately carries no ``source_phase``: that field lets the optimizations
    collector resolve the owner without consulting the action label at all, and
    would therefore hide exactly the mismatch this test exists to catch.
    """
    from hyperloom.orchestrator.loop.writeback import _FRAMEWORK_STACK_ACTION

    state = {
        "session_id": "framework-label-contract",
        "baseline_tput": 100.0,
        "cumulative_gain_validated": 10.0,
        "cumulative_gain_validated_stack_len": 1,
        "optimization_stack": [
            {
                "action": _FRAMEWORK_STACK_ACTION,
                "variant_name": "https://example.com/pull/1",
                "tput": 110.0,
                "ts": "1970-01-01T00:00:10+00:00",
            },
        ],
        "gain_per_stack_entry": [10.0],
    }

    attribution = collect_attribution(state, [], [], [])
    result = collect_optimizations(state, attribution, [], [], [])

    assert attribution["source_breakdown"]["framework_pct_of_total"] == 10.0
    assert attribution["source_breakdown"]["unattributed_pct_of_total"] == 0.0
    assert attribution["phase_breakdown"]["framework"]["total_gain_pct"] == 10.0
    assert result["entries"][0]["source"] == "framework_agent"
    assert result["entries"][0]["source_method"] == "action_family"


def test_phase_breakdown_schema_declares_every_emitted_bucket():
    """The declared shape must cover the keys the collector actually writes.

    ``session_breakdown.json`` is a published contract, and the TypedDict is
    what downstream code reads it through, so a bucket the producer emits but
    the schema omits shows up as an empty section rather than an error. The
    KERNEL_AGENT bucket sat in exactly that state.
    """
    from hyperloom.inference_optimizer.breakdown.schema import PhaseBreakdown

    state = {
        "session_id": "phase-bucket-contract",
        "baseline_tput": 100.0,
        "cumulative_gain_validated": 10.0,
        "cumulative_gain_validated_stack_len": 1,
        "optimization_stack": [
            {
                "action": "kernel_opt",
                "source_phase": "KERNEL_AGENT",
                "variant_name": "k1",
                "kernel_id": "fused_moe",
                "tput": 110.0,
                "ts": "1970-01-01T00:00:10+00:00",
            },
        ],
        "gain_per_stack_entry": [10.0],
    }

    emitted = set(collect_attribution(state, [], [], [])["phase_breakdown"])
    undeclared = sorted(emitted - set(PhaseBreakdown.__annotations__))

    assert not undeclared, f"phase_breakdown buckets missing from the schema: {undeclared}"


def test_integrate_patch_projects_source_manifest_and_changed_files():
    state = {
        "session_id": "integrate-patch-artifacts",
        "baseline_tput": 100.0,
        "cumulative_gain_validated": 10.0,
        "cumulative_gain_validated_stack_len": 1,
        "optimization_stack": [
            {
                "action": "integrate_patch",
                "source_phase": "EXPLORE",
                "domain": "serving_specialist",
                "variant_name": "quantization-patch",
                "source_manifest": (
                    "/session/optimization_stack/src/spec-1/manifest.json"
                ),
                "target_files": [
                    "vllm/model_executor/layers/quantization/foo.py",
                    "vllm/model_executor/layers/quantization/bar.py",
                ],
                "tput": 110.0,
                "ts": "1970-01-01T00:00:10+00:00",
            },
        ],
        "gain_per_stack_entry": [10.0],
    }

    attribution = collect_attribution(state, [], [], [])
    result = collect_optimizations(state, attribution, [], [], [])

    assert result["entries"][0]["artifacts"] == [
        {
            "kind": "source_manifest",
            "path": "/session/optimization_stack/src/spec-1/manifest.json",
        },
        {
            "kind": "target_file",
            "path": "vllm/model_executor/layers/quantization/foo.py",
        },
        {
            "kind": "target_file",
            "path": "vllm/model_executor/layers/quantization/bar.py",
        },
    ]


def test_integrate_with_kernel_evidence_remains_kernel_agent():
    state = {
        "session_id": "kernel-integrate-k001",
        "baseline_tput": 100.0,
        "cumulative_gain_validated": 10.0,
        "cumulative_gain_validated_stack_len": 1,
        "optimization_stack": [
            {
                "action": "integrate",
                "source_phase": "KERNEL_AGENT",
                "kernel_id": "k001",
                "backend": "forge",
                "patch_path": "/runs/k001.patch",
                "target_file": "vllm/_aiter_ops.py",
                "variant_name": "k001",
                "tput": 110.0,
                "ts": "1970-01-01T00:00:10+00:00",
            },
        ],
        "gain_per_stack_entry": [10.0],
        "phase_history": [{"to_phase": "KERNEL_AGENT", "ts_unix": 0.0}],
    }

    attribution = collect_attribution(state, [], [], [])
    result = collect_optimizations(state, attribution, [], [], [])

    assert result["entries"][0]["source"] == "kernel_agent"
    assert result["entries"][0]["source_method"] == "action_family"
    assert result["entries"][0]["kernel_id"] == "k001"
    assert result["entries"][0]["backend"] == "forge"


def test_collect_optimizations_splits_kernel_backend():
    state = {
        "session_id": "s2",
        "baseline_tput": 100.0,
        "cumulative_gain_validated": 20.0,
        "cumulative_gain_validated_stack_len": 2,
        "optimization_stack": [
            {
                "action": "geak_e2e",
                "variant_name": "geak",
                "tput": 110.0,
                "source": "geak_e2e",
                "ts": "1970-01-01T00:00:10+00:00",
            },
            {
                "action": "gemm_tuning",
                "variant_name": "forge-gemm",
                "tput": 120.0,
                "backend": "forge",
                "kernel_id": "k1",
                "source_phase": "KERNEL_AGENT",
                "accepted_heads": ["mla"],
                "extra_server_args_is_invariant": True,
                "candidate_flags": {"fast_math": True},
                "ts": "1970-01-01T00:00:20+00:00",
            },
        ],
        "gain_per_stack_entry": [10.0, 20.0],
        "phase_history": [{"to_phase": "KERNEL_AGENT", "ts_unix": 0.0}],
    }
    attribution = collect_attribution(state, [], [], [])
    geak_invocations = [
        {
            "attempt_id": "geak-failed",
            "kernel_id": "k0",
            "backend": "geak",
            "decision": "FAILED",
            "ts": "1970-01-01T00:00:05+00:00",
            "duration_sec": 3.0,
            "error_class": "CompileError",
            "error": "compile failed",
        }
    ]
    forge_invocations = [
        {
            "attempt_id": "forge-kept",
            "kernel_id": "k1",
            "backend": "forge",
            "decision": "KEEP",
            "ts": "1970-01-01T00:00:15+00:00",
            "elapsed_sec": 7.5,
        }
    ]

    result = collect_optimizations(
        state,
        attribution,
        geak_invocations,
        forge_invocations,
        [],
        gemm_tuning={
            "runs": [
                {
                    "engine": "forge",
                    "status": "succeeded",
                    "decision": "KEEP",
                    "adopted": True,
                }
            ]
        },
    )

    assert result["entries"][0]["backend"] == "geak"
    assert result["entries"][0]["execution_mode"] == "whole_pipeline"
    assert result["entries"][1]["backend"] == "forge"
    assert result["entries"][1]["execution_mode"] == "per_kernel"
    assert result["entries"][1]["adopted_attempt_id"] == "forge-kept"
    assert result["entries"][1]["source_phase"] == "KERNEL_AGENT"
    assert result["entries"][1]["gain_method"] == "legacy_ledger_derived"
    assert result["entries"][1]["accepted_heads"] == ["mla"]
    assert result["entries"][1]["extra_server_args_is_invariant"] is True
    assert result["entries"][1]["candidate_flags"] == {"fast_math": True}
    assert result["backend_attempts"] == [
        {
            "attempt_id": "geak-failed",
            "run_id": "",
            "kernel_id": "k0",
            "backend": "geak",
            "decision": "FAILED",
            "ts": "1970-01-01T00:00:05+00:00",
            "duration_sec": 3.0,
            "micro_speedup": None,
            "compile_passed": None,
            "correctness_passed": None,
            "error_class": "CompileError",
            "error": "compile failed",
            "result_path": None,
            "verification_path": None,
            "sequence": 1,
        },
        {
            "attempt_id": "forge-kept",
            "run_id": "",
            "kernel_id": "k1",
            "backend": "forge",
            "decision": "KEEP",
            "ts": "1970-01-01T00:00:15+00:00",
            "duration_sec": 7.5,
            "micro_speedup": None,
            "compile_passed": None,
            "correctness_passed": None,
            "error_class": None,
            "error": None,
            "result_path": None,
            "verification_path": None,
            "sequence": 1,
        },
    ]
    by_backend = result["summary_by_source"]["kernel_agent"]["by_backend"]
    assert by_backend["geak"] == {"keeps": 1, "total_gain_pct": 10.0}
    assert by_backend["forge"] == {"keeps": 1, "total_gain_pct": 10.0}
    assert result["summary_by_kind"]["gemm_tuning"] == {
        "keeps": 1,
        "total_gain_pct": 10.0,
        "by_backend": {
            "geak": {"keeps": 0, "total_gain_pct": 0.0},
            "forge": {"keeps": 1, "total_gain_pct": 10.0},
            "unattributed": {"keeps": 0, "total_gain_pct": 0.0},
        },
    }
    assert result["validation"]["method"] == "reconstructed"
    assert result["validation"]["validated_total_gain_pct"] == 20.0
    assert result["validation"]["attribution_gap_pct"] == 0.0
    assert result["validation"]["validated_at_stack_len"] == 2
    assert result["gemm_tuning_runs"][0]["engine"] == "forge"


def test_collect_optimizations_excludes_non_optimization_stack_anchors():
    state = {
        "session_id": "anchors",
        "optimization_stack": [
            {"action": "baseline", "tput": 100.0},
            {"action": "sweep", "tput": 110.0},
            {"action": "validate_stack", "tput": 115.0},
            {"action": "explore", "variant_name": "kept", "tput": 120.0},
        ],
        "gain_per_stack_entry": [0.0, 10.0, 15.0, 20.0],
        "cumulative_gain_validated": 20.0,
        "cumulative_gain_validated_stack_len": 4,
        "phase_history": [{"to_phase": "EXPLORE", "ts_unix": 0.0}],
    }
    attribution = collect_attribution(state, [], [], [])

    result = collect_optimizations(state, attribution, [], [], [])

    assert [entry["name"] for entry in result["entries"]] == ["kept"]
    assert result["validation"]["source_breakdown"]["sweep_gain_pct"] == 10.0


def test_collect_optimizations_generates_stable_attempt_ids_and_rejects_ambiguous_keep():
    state = {
        "session_id": "stable",
        "baseline_tput": 100.0,
        "cumulative_gain_validated": 10.0,
        "cumulative_gain_validated_stack_len": 1,
        "optimization_stack": [
            {
                "action": "kernel_optimize",
                "kernel_id": "k1",
                "backend": "forge",
                "tput": 110.0,
            }
        ],
        "gain_per_stack_entry": [
            {
                "delta_pct": 10.0,
                "cum_gain_after": 10.0,
            }
        ],
    }
    attribution = collect_attribution(state, [], [], [])

    single = collect_optimizations(
        state,
        attribution,
        [],
        [{"kernel_id": "k1", "backend": "forge", "decision": "KEEP"}],
        [],
    )

    generated = "stable:kernel-attempt:k1:forge:1"
    assert single["backend_attempts"][0]["attempt_id"] == generated
    assert single["entries"][0]["adopted_attempt_id"] == generated

    warnings: list[str] = []
    ambiguous = collect_optimizations(
        state,
        attribution,
        [],
        [
            {
                "attempt_id": "keep-1",
                "kernel_id": "k1",
                "backend": "forge",
                "decision": "KEEP",
            },
            {
                "attempt_id": "keep-2",
                "kernel_id": "k1",
                "backend": "forge",
                "decision": "KEEP",
            },
        ],
        warnings,
    )

    assert ambiguous["entries"][0]["adopted_attempt_id"] is None
    assert any("multiple KEEP attempts" in warning for warning in warnings)


def test_collect_optimizations_keeps_unknown_source_honest():
    state = {
        "session_id": "legacy",
        "optimization_stack": [{"action": "integrate_patch", "tput": 10.0}],
        "gain_per_stack_entry": [3.0],
        "cumulative_gain_validated_stack_len": 1,
    }
    attribution = collect_attribution(state, [], [], [])

    result = collect_optimizations(state, attribution, [], [], [])

    assert result["entries"][0]["source"] == "unattributed"
    assert result["entries"][0]["source_method"] == "unknown"
    assert result["summary_by_source"]["unattributed"]["keeps"] == 1


def test_exporter_emits_canonical_optimizations(tmp_path):
    state = {
        "session_id": "export",
        "baseline_tput": 100.0,
        "cumulative_gain_validated": 10.0,
        "cumulative_gain_validated_stack_len": 1,
        "optimization_stack": [
            {
                "action": "replay_warm_recipe",
                "variant_name": "warm",
                "tput": 110.0,
                "ts": "2026-01-01T00:00:00+00:00",
            }
        ],
        "gain_per_stack_entry": [10.0],
        "gemm_tuning_attempts": [
            {
                "engine": "geak",
                "status": "failed",
                "decision": "FAILED",
                "duration_sec": 4.5,
                "error_class": "TuningError",
                "error": "no valid candidate",
                "parameters": {"libtype": "ck"},
                "candidates": [{"name": "candidate-1", "status": "eliminated"}],
            }
        ],
    }
    (tmp_path / "state.json").write_text(json.dumps(state), encoding="utf-8")
    (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")

    result = exporter.build(tmp_path)

    assert result["schema_version"] == "hyperloom.session_breakdown.v5.0"
    assert result["optimizations"]["schema_version"] == 3
    # No recorder parts in this fixture, so the state.json read model is the
    # documented fallback and must say so rather than pass itself off as
    # author-time evidence.
    assert result["optimizations"]["source_of_truth"] == "state"
    assert result["optimizations"]["attempts"] == []
    assert result["optimizations"]["entries"][0]["source"] == "warm_replay"
    assert result["optimizations"]["validation"]["validated_total_gain_pct"] == 10.0
    assert result["optimizations"]["gemm_tuning_runs"][0]["duration_sec"] == 4.5
    assert result["optimizations"]["gemm_tuning_runs"][0]["error_class"] == "TuningError"
    assert result["optimizations"]["gemm_tuning_runs"][0]["parameters"] == {
        "libtype": "ck"
    }
    assert result["optimizations"]["gemm_tuning_runs"][0]["candidates"] == [
        {"name": "candidate-1", "status": "eliminated"}
    ]
    assert "optimization_stack" not in result
    assert "attribution" not in result
    assert "geak_invocations" not in result
    assert "forge_invocations" not in result
    assert "geak" not in result
    assert "gemm_tuning" not in result


def test_v4_optimizations_derive_from_validated_adoptions():
    run = {"session_id": "v4"}
    operations = [
        {
            "operation_id": "op1",
            "kind": "kernel_optimization",
            "name": "k1",
            "phase": "KERNEL_AGENT",
            "strategy": "kernel_agent_forge",
            "subject": {"kind": "kernel", "name": "k1"},
        }
    ]
    measurements = [
        {
            "measurement_id": "m1",
            "name": "final_throughput",
            "value": 125.0,
        }
    ]
    adoptions = [
        {
            "adoption_id": "a1",
            "operation_id": "op1",
            "decision": "KEEP",
            "validated": True,
            "gain_pct": 25.0,
            "measurement_ids": ["m1"],
            "artifact_ids": ["p1"],
            "subject": {"kind": "kernel", "name": "k1"},
        }
    ]
    artifacts = [
        {
            "artifact_id": "p1",
            "kind": "patch",
            "path": "/tmp/k1.patch",
        }
    ]

    result = collect_v4_optimizations(
        run,
        operations,
        measurements,
        adoptions,
        artifacts,
        [],
    )

    entry = result["entries"][0]
    assert entry["source"] == "kernel_agent"
    assert entry["backend"] == "forge"
    assert entry["kernel_id"] == "k1"
    assert entry["artifacts"] == [{"kind": "patch", "path": "/tmp/k1.patch"}]


def _recorded_fixture():
    """One kept kernel patch, one rejected kernel patch, one kept enablement."""
    operations = [
        {
            "operation_id": "op-k001",
            "kind": "kernel_optimization",
            "name": "k001",
            "agent": "kernel_agent",
            "producer": "kernel-agent",
            "phase": "KERNEL_AGENT",
            "strategy": "forge",
            "status": "succeeded",
            "started_at": "2026-01-01T00:00:00+00:00",
            "ended_at": "2026-01-01T01:00:00+00:00",
            "subject": {"subject_type": "kernel", "name": "k001"},
            "measurement_refs": ["m-k001-after"],
            "artifact_refs": ["art-k001"],
            "attempts": [
                {
                    "attempt_id": "forge-1",
                    "backend": "forge",
                    "status": "succeeded",
                    "started_at": "2026-01-01T00:10:00+00:00",
                    "ended_at": "2026-01-01T00:20:00+00:00",
                    "outputs": {"compile_passed": True, "correctness_passed": True},
                }
            ],
        },
        {
            "operation_id": "op-k002",
            "kind": "kernel_optimization",
            "name": "k002",
            "agent": "kernel_agent",
            "producer": "kernel-agent",
            "phase": "KERNEL_AGENT",
            "strategy": "forge",
            "status": "needs_review",
            "started_at": "2026-01-01T02:00:00+00:00",
            "ended_at": "2026-01-01T03:00:00+00:00",
            "subject": {"subject_type": "kernel", "name": "k002"},
            "measurement_refs": ["m-k002-after", "m-k002-gain"],
            "gates": [
                {
                    "kind": "keep_threshold",
                    "name": "keep_threshold",
                    "status": "failed",
                    "decision": "deny",
                    "reason": "throughput delta +0.55% < keep_threshold 3.00%",
                    "inputs": {"keep_threshold_pct": 3.0},
                }
            ],
            "decisions": [
                {
                    "verdict": "REJECTED",
                    "reason": "throughput delta +0.55% < keep_threshold 3.00%",
                }
            ],
        },
        {
            "operation_id": "op-enable",
            "kind": "integrate_patch",
            "name": "integrate_patch",
            "agent": "framework_agent",
            "producer": "coordinator",
            "phase": "KERNEL_AGENT",
            "status": "succeeded",
            "started_at": "2026-01-01T04:00:00+00:00",
            "ended_at": "2026-01-01T05:00:00+00:00",
            "subject": {"subject_type": "integrate_patch", "name": "prelude"},
        },
    ]
    measurements = [
        {"measurement_id": "m-k001-after", "name": "final_throughput", "value": 125.0},
        {"measurement_id": "m-k002-after", "name": "final_throughput", "value": 100.55},
        {"measurement_id": "m-k002-gain", "name": "e2e_gain_pct", "value": 0.55},
    ]
    adoptions = [
        {
            "adoption_id": "ad-k001",
            "operation_id": "op-k001",
            "decision": "KEEP",
            "validated": True,
            "agent": "kernel_agent",
            "attribution_eligible": True,
            "gain_pct": 25.0,
            "reason": "integrate_e2e_passed",
            "adopted_at": "2026-01-01T01:00:00+00:00",
            "measurement_ids": ["m-k001-after"],
        },
        {
            "adoption_id": "ad-enable",
            "operation_id": "op-enable",
            "decision": "KEEP",
            "validated": True,
            "agent": "framework_agent",
            "attribution_eligible": False,
            "gain_pct": 0.4,
            "reason": "enablement patch applied before baseline",
            "adopted_at": "2026-01-01T05:00:00+00:00",
        },
    ]
    artifacts = [
        {"artifact_id": "art-k001", "kind": "patch", "path": "/tmp/k001.patch"},
    ]
    return operations, measurements, adoptions, artifacts


def test_recorded_optimizations_keep_rejected_attempts_with_their_reason():
    operations, measurements, adoptions, artifacts = _recorded_fixture()
    warnings: list[str] = []

    result = collect_recorded_optimizations(
        "s1", operations, measurements, adoptions, artifacts, [], [], warnings
    )

    assert result["source_of_truth"] == "recorder"
    rejected = next(row for row in result["attempts"] if row["name"] == "k002")
    assert rejected["adopted"] is False
    assert rejected["agent"] == "kernel_agent"
    assert rejected["decision"] == "REJECTED"
    assert rejected["local_gain_pct"] == 0.55
    assert rejected["throughput_after"] == 100.55
    assert rejected["keep_threshold_pct"] == 3.0
    assert "keep_threshold 3.00%" in rejected["decision_reason"]


def test_recorded_optimizations_bucket_attempts_by_recorded_agent():
    operations, measurements, adoptions, artifacts = _recorded_fixture()

    result = collect_recorded_optimizations(
        "s1", operations, measurements, adoptions, artifacts, [], [], []
    )

    by_agent = result["summary_by_agent"]
    assert by_agent["kernel_agent"]["attempts"] == 2
    assert by_agent["kernel_agent"]["keeps"] == 1
    assert by_agent["kernel_agent"]["reverts"] == 1
    assert by_agent["kernel_agent"]["attributable_gain_pct"] == 25.0
    # The enablement patch is a real adoption that must never be sold as gain.
    assert by_agent["framework_agent"]["keeps"] == 1
    assert by_agent["framework_agent"]["non_attributable_keeps"] == 1
    assert by_agent["framework_agent"]["attributable_gain_pct"] == 0.0
    # Every attempt is owned by whoever recorded it, never inferred.
    assert {row["agent"] for row in result["attempts"]} == {
        "kernel_agent",
        "framework_agent",
    }


def test_recorded_optimizations_exclude_ineligible_keeps_from_entries():
    operations, measurements, adoptions, artifacts = _recorded_fixture()

    result = collect_recorded_optimizations(
        "s1", operations, measurements, adoptions, artifacts, [], [], []
    )

    assert [entry["name"] for entry in result["entries"]] == ["k001"]
    assert result["entries"][0]["source_method"] == "recorded"
    assert result["validation"]["validated_total_gain_pct"] == 25.0
    assert result["validation"]["attempt_count"] == 3
    assert result["validation"]["non_attributable_keep_count"] == 1


def test_entries_are_a_gain_ledger_that_points_back_at_its_attempt():
    """``entries`` must not restate what its attempt already says.

    Descriptive detail lives on the attempt and is reached through
    ``adopted_attempt_id``. A field present in both places has to carry the
    same value in both, so that reading either one gives the same answer.
    """
    operations, measurements, adoptions, artifacts = _recorded_fixture()

    result = collect_recorded_optimizations(
        "s1", operations, measurements, adoptions, artifacts, [], [], []
    )

    entry = result["entries"][0]
    attempt = next(
        row
        for row in result["attempts"]
        if row["attempt_id"] == entry["adopted_attempt_id"]
    )
    for restated in (
        "kernel_id",
        "source_phase",
        "keep_threshold_pct",
        "decision_reason",
        "artifacts",
        "throughput_before",
    ):
        assert restated not in entry
    for shared in ("name", "backend", "adoption_id", "throughput_after", "local_gain_pct"):
        assert entry[shared] == attempt[shared]


def test_attempt_gain_is_never_named_like_the_baseline_relative_one():
    """The two gains are different numbers and must not share a field name."""
    operations, measurements, adoptions, artifacts = _recorded_fixture()

    result = collect_recorded_optimizations(
        "s1", operations, measurements, adoptions, artifacts, [], [], []
    )

    for attempt in result["attempts"]:
        assert "gain_pct" not in attempt
        assert "local_gain_pct" in attempt


def test_recorded_optimizations_report_gain_against_the_session_baseline():
    """Two adoptions from the gemma session, with its real measured numbers.

    Each executor measures against whatever it started from, so the two local
    gains (7.09% and 10.95%) cannot simply be added. Reported gain is measured
    against the session baseline, which is itself a recorded measurement.
    """
    operations = [
        {
            "operation_id": "op-baseline",
            "kind": "composite",
            "name": "baseline",
            "agent": "coordinator",
            "measurement_refs": ["m-baseline"],
        },
        {
            "operation_id": "op-gemm",
            "kind": "kernel_optimization",
            "name": "gemm_tune_vllm_moe_triton",
            "agent": "kernel_agent",
            "ended_at": "2026-08-08T06:56:21+00:00",
            "subject": {"subject_type": "kernel", "name": "gemm_tune_vllm_moe_triton"},
        },
        {
            "operation_id": "op-patch",
            "kind": "integrate_patch",
            "name": "integrate_patch",
            "agent": "framework_agent",
            "ended_at": "2026-08-09T04:22:15+00:00",
        },
    ]
    measurements = [
        {"measurement_id": "m-baseline", "name": "throughput", "value": 4726.9446478},
    ]
    adoptions = [
        {
            "adoption_id": "ad-gemm",
            "operation_id": "op-gemm",
            "decision": "KEEP",
            "validated": True,
            "agent": "kernel_agent",
            "gain_pct": 7.0904327726706935,
            # An earlier PRELUDE patch had already moved the workload off the
            # baseline before this kernel started, which is why the kernel's
            # own starting point is not 4726.94.
            "throughput_before": 4744.5975753,
            "throughput_after": 5081.0100767,
            "adopted_at": "2026-08-08T06:56:21+00:00",
        },
        {
            "adoption_id": "ad-patch",
            "operation_id": "op-patch",
            "decision": "KEEP",
            "validated": True,
            "agent": "framework_agent",
            "gain_pct": 10.949641325767333,
            "throughput_before": 5081.0100767,
            "throughput_after": 5637.3624559,
            "adopted_at": "2026-08-09T04:22:15+00:00",
        },
    ]

    warnings: list[str] = []
    result = collect_recorded_optimizations(
        "gemma", operations, measurements, adoptions, [], [], [], warnings
    )

    first, second = result["entries"]
    assert first["gain_method"] == "baseline_chain"
    assert first["gain_pct"] == 7.116912
    assert first["local_gain_pct"] == 7.090433
    assert second["gain_pct"] == 11.769809
    assert second["local_gain_pct"] == 10.949641
    assert second["cumulative_gain_pct"] == 19.260175
    # Per-agent totals add up to what the attempts claim, not to the session's
    # end-to-end move; the difference is stated instead of being handed to the
    # kernel that happened to run next.
    assert result["summary_by_agent"]["kernel_agent"]["attributable_gain_pct"] == 7.116912
    assert (
        result["summary_by_agent"]["framework_agent"]["attributable_gain_pct"] == 11.769809
    )
    validation = result["validation"]
    assert validation["validated_total_gain_pct"] == 19.260175
    assert validation["attributed_total_gain_pct"] == 18.886722
    assert validation["unattributed_gain_pct"] == 0.373453
    assert validation["attribution_gap_pct"] == 0.373453
    # The audit identity that has to survive rounding: what the session moved
    # is what the attempts claim plus what nobody claims.
    assert (
        validation["attributed_total_gain_pct"] + validation["unattributed_gain_pct"]
        == validation["validated_total_gain_pct"]
    )
    assert any("belongs to no attempt" in warning for warning in warnings)


def test_gain_before_the_first_adopted_step_is_not_handed_to_it():
    """The 0.37pp that started the leaderboard argument, in isolation.

    A patch moves the workload off the baseline and is never adopted. The
    kernel that runs next must report what it itself added, not what it
    inherited.
    """
    operations = [
        {
            "operation_id": "op-baseline",
            "kind": "composite",
            "name": "baseline",
            "measurement_refs": ["m-baseline"],
        },
        {
            "operation_id": "op-k1",
            "kind": "kernel_optimization",
            "name": "k1",
            "agent": "kernel_agent",
            "ended_at": "2026-01-01T02:00:00+00:00",
        },
    ]
    measurements = [{"measurement_id": "m-baseline", "name": "throughput", "value": 1000.0}]
    adoptions = [
        {
            "adoption_id": "ad-k1",
            "operation_id": "op-k1",
            "decision": "KEEP",
            "validated": True,
            "agent": "kernel_agent",
            "throughput_before": 1100.0,
            "throughput_after": 1210.0,
        }
    ]
    warnings: list[str] = []

    result = collect_recorded_optimizations(
        "s1", operations, measurements, adoptions, [], [], [], warnings
    )

    entry = result["entries"][0]
    # It started at 1100 and left at 1210, so it added 11pp of the baseline —
    # not the 21pp it would inherit by being measured from the baseline.
    assert entry["gain_pct"] == 11.0
    assert entry["cumulative_gain_pct"] == 21.0
    assert result["validation"]["unattributed_gain_pct"] == 10.0
    assert any("belongs to no attempt" in warning for warning in warnings)


def test_adoption_throughput_outranks_overwritten_measurements():
    """A retry on the same kernel overwrites the measurements it referenced."""
    operations = [
        {
            "operation_id": "op-k1",
            "kind": "kernel_optimization",
            "name": "k1",
            "agent": "kernel_agent",
            "measurement_refs": ["m-final"],
            "ended_at": "2026-01-01T00:00:00+00:00",
        },
    ]
    # What the later retry left behind, not what the adoption was decided on.
    measurements = [{"measurement_id": "m-final", "name": "final_throughput", "value": 5100.76}]
    adoptions = [
        {
            "adoption_id": "ad-k1",
            "operation_id": "op-k1",
            "decision": "KEEP",
            "validated": True,
            "agent": "kernel_agent",
            "gain_pct": 7.09,
            "throughput_after": 5081.01,
        }
    ]

    result = collect_recorded_optimizations(
        "s1", operations, measurements, adoptions, [], [], [], []
    )

    assert result["attempts"][0]["throughput_after"] == 5081.01


def test_an_adoption_citing_overwritten_evidence_says_so():
    """Archives predating per-occurrence ids cannot be repaired, only labelled.

    The frozen values still stand, but the readings the adoption points at were
    written over by a later re-measure. Presenting the two side by side without
    a word is what made this look like the numbers had been edited after the
    fact.
    """
    operations = [
        {
            "operation_id": "op-k1",
            "kind": "kernel_optimization",
            "name": "k1",
            "agent": "kernel_agent",
            "measurement_refs": ["m-before", "m-after"],
        },
    ]
    measurements = [
        {"measurement_id": "m-before", "name": "baseline_throughput", "value": 5081.01},
        {"measurement_id": "m-after", "name": "final_throughput", "value": 5100.76},
    ]
    adoptions = [
        {
            "adoption_id": "ad-k1",
            "operation_id": "op-k1",
            "decision": "KEEP",
            "validated": True,
            "agent": "kernel_agent",
            "measurement_ids": ["m-before", "m-after"],
            "throughput_before": 4744.6,
            "throughput_after": 5081.01,
        }
    ]

    attempt = collect_recorded_optimizations(
        "s1", operations, measurements, adoptions, [], [], [], []
    )["attempts"][0]

    assert attempt["measurement_source"] == "adoption_pinned_stale"
    assert attempt["throughput_after"] == 5081.01


def test_intact_pinned_evidence_is_not_called_stale():
    operations = [
        {
            "operation_id": "op-k1",
            "kind": "kernel_optimization",
            "name": "k1",
            "agent": "kernel_agent",
            "measurement_refs": ["m-before", "m-after"],
        },
    ]
    measurements = [
        {"measurement_id": "m-before", "name": "baseline_throughput", "value": 4744.6},
        {"measurement_id": "m-after", "name": "final_throughput", "value": 5081.01},
    ]
    adoptions = [
        {
            "adoption_id": "ad-k1",
            "operation_id": "op-k1",
            "decision": "KEEP",
            "validated": True,
            "agent": "kernel_agent",
            "measurement_ids": ["m-before", "m-after"],
            "throughput_before": 4744.6,
            "throughput_after": 5081.01,
        }
    ]

    attempt = collect_recorded_optimizations(
        "s1", operations, measurements, adoptions, [], [], [], []
    )["attempts"][0]

    assert attempt["measurement_source"] == "adoption_pinned"


def test_repeated_readings_of_a_metric_are_numbered_oldest_first():
    """Recorded ids are unreadable by necessity, so the ordinal is added here.

    An id has to be reproducible from the record being written, since several
    producers replay their records after a resume, which rules out numbering
    them as they arrive. The plain ordinal a reader wants is therefore assigned
    on the way out, where every reading is in hand at once.
    """
    operations = [
        {
            "operation_id": "op-k1",
            "kind": "kernel_optimization",
            "name": "k1",
            "agent": "kernel_agent",
            "measurement_refs": ["m-after-late", "m-before", "m-after-early"],
        },
    ]
    measurements = [
        {
            "measurement_id": "m-after-late",
            "name": "final_throughput",
            "value": 5100.76,
            "measured_at": "2026-01-01T18:00:00+00:00",
        },
        {
            "measurement_id": "m-before",
            "name": "baseline_throughput",
            "value": 4744.6,
            "measured_at": "2026-01-01T06:00:00+00:00",
        },
        {
            "measurement_id": "m-after-early",
            "name": "final_throughput",
            "value": 5081.01,
            "measured_at": "2026-01-01T06:00:00+00:00",
        },
    ]
    adoptions = [
        {
            "adoption_id": "ad-k1",
            "operation_id": "op-k1",
            "decision": "KEEP",
            "validated": True,
            "agent": "kernel_agent",
            "measurement_ids": ["m-before", "m-after-early", "m-after-late"],
            "throughput_before": 4744.6,
            "throughput_after": 5081.01,
        }
    ]

    attempt = collect_recorded_optimizations(
        "s1", operations, measurements, adoptions, [], [], [], []
    )["attempts"][0]
    numbered = {row["value"]: row for row in attempt["measurements"]}

    # The reading the decision was made on is the first of its name, even
    # though the operation happens to reference the later one first.
    assert numbered[5081.01]["occurrence"] == 0
    assert numbered[5100.76]["occurrence"] == 1
    assert numbered[5081.01]["occurrences_of_name"] == 2
    # A name measured once is numbered too, rather than left to be guessed at.
    assert numbered[4744.6]["occurrence"] == 0
    assert numbered[4744.6]["occurrences_of_name"] == 1

