# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json

from hyperloom.inference_optimizer.breakdown import exporter
from hyperloom.inference_optimizer.breakdown.collectors import (
    collect_attribution,
    collect_optimizations,
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
    assert result["optimizations"]["schema_version"] == 2
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

