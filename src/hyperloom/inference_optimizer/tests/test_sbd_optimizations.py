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
                "variant_name": "framework-patch",
                "tput": 160.0,
                "ts": "1970-01-01T00:00:20+00:00",
                "patch_path": "/tmp/framework.patch",
            },
            {
                "action": "integrate_patch",
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
    assert "action" not in result["entries"][1]
    assert result["entries"][1]["optimization_kind"] == "framework_patch"
    assert result["entries"][2]["optimization_kind"] == "env"
    summary = result["summary_by_source"]
    assert summary["warm_replay"] == {"keeps": 1, "total_gain_pct": 50.0}
    assert summary["framework_agent"] == {"keeps": 1, "total_gain_pct": 10.0}
    assert summary["explore"] == {"keeps": 1, "total_gain_pct": 10.0}


def test_collect_optimizations_splits_kernel_backend():
    state = {
        "session_id": "s2",
        "baseline_tput": 100.0,
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
                "ts": "1970-01-01T00:00:20+00:00",
            },
        ],
        "gain_per_stack_entry": [10.0, 20.0],
        "phase_history": [{"to_phase": "KERNEL_AGENT", "ts_unix": 0.0}],
    }
    attribution = collect_attribution(state, [], [], [])

    result = collect_optimizations(state, attribution, [], [], [])

    assert result["entries"][0]["backend"] == "geak"
    assert result["entries"][0]["execution_mode"] == "whole_pipeline"
    assert result["entries"][1]["backend"] == "forge"
    assert result["entries"][1]["execution_mode"] == "per_kernel"
    by_backend = result["summary_by_source"]["kernel_agent"]["by_backend"]
    assert by_backend["geak"] == {"keeps": 1, "total_gain_pct": 10.0}
    assert by_backend["forge"] == {"keeps": 1, "total_gain_pct": 10.0}


def test_collect_optimizations_excludes_non_optimization_stack_anchors():
    state = {
        "session_id": "anchors",
        "optimization_stack": [
            {"action": "baseline", "tput": 100.0},
            {"action": "validate_stack", "tput": 110.0},
            {"action": "explore", "variant_name": "kept", "tput": 120.0},
        ],
        "gain_per_stack_entry": [0.0, 10.0, 20.0],
        "cumulative_gain_validated_stack_len": 3,
        "phase_history": [{"to_phase": "EXPLORE", "ts_unix": 0.0}],
    }
    attribution = collect_attribution(state, [], [], [])

    result = collect_optimizations(state, attribution, [], [], [])

    assert [entry["name"] for entry in result["entries"]] == ["kept"]


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
    }
    (tmp_path / "state.json").write_text(json.dumps(state), encoding="utf-8")
    (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")

    result = exporter.build(tmp_path)

    assert result["schema_version"] == "hyperloom.session_breakdown.v5.0"
    assert result["optimizations"]["schema_version"] == 1
    assert result["optimizations"]["entries"][0]["source"] == "warm_replay"
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

