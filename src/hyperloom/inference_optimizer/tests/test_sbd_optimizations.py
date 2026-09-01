# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json

import pytest

from hyperloom.inference_optimizer.breakdown import exporter
from hyperloom.inference_optimizer.breakdown.collectors import (
    collect_attribution,
    collect_optimization_stack,
    collect_recorded_optimizations,
)
from hyperloom.inference_optimizer.breakdown.recorder import assemble_parts, instrument


def test_a_promoted_collective_keep_is_credited_to_its_own_family(tmp_path):
    """The collective lane settles its own verdict, outside the integrate queue.

    No kernel recorder fires for it, so without a record of its own the change
    is invisible to the read model: the patch lands, the workload moves, and
    every point it earned reports as belonging to no step.
    """
    instrument.record_collective_promotion(
        tmp_path,
        integration_id="integration-1",
        kernel_id="k007",
        baseline_tput=100.0,
        new_tput=130.0,
        gain_pct=30.0,
        patch_path="/ws/collective.patch",
        collective_op="all_reduce",
        world_size=8,
        ts="2026-01-01T00:00:20+00:00",
    )
    parts = assemble_parts(tmp_path)
    operations = list(parts.get("operations") or [])
    measurements = list(parts.get("measurements") or [])
    operations.append({"operation_id": "op-base", "kind": "baseline", "measurement_refs": ["m-base"]})
    measurements.append({"measurement_id": "m-base", "name": "throughput", "value": 100.0})
    warnings: list[str] = []

    result = collect_recorded_optimizations(
        "s1",
        operations,
        measurements,
        list(parts.get("adoptions") or []),
        list(parts.get("artifacts") or []),
        [],
        [],
        warnings,
    )

    entry = result["entries"][0]
    assert entry["optimization_kind"] == "kernel_collective"
    assert entry["source"] == "kernel_agent"
    assert entry["backend"] == "forge"
    assert entry["gain_method"] == "baseline_chain"
    assert result["summary_by_kind"]["kernel_collective"]["total_gain_pct"] == 30.0
    assert result["validation"]["attributed_total_gain_pct"] == 30.0
    assert result["validation"]["unattributed_gain_pct"] == 0.0
    assert warnings == []


def test_collective_stack_entry_keeps_campaign_evidence():
    state = {
        "cumulative_gain_validated_stack_len": 1,
        "optimization_stack": [
            {
                "action": "collective",
                "variant_name": "forge_collective",
                "engine": "forge_collective",
                "kernel_id": "k007",
                "tput": 130.0,
                "ts": "1970-01-01T00:00:20+00:00",
                "collective_op": "all_reduce",
                "world_size": 8,
                "collective_attempt_id": "attempt-1",
                "integration_id": "integration-1",
            },
        ],
    }

    entry = collect_optimization_stack(state)[0]

    assert entry["collective_op"] == "all_reduce"
    assert entry["world_size"] == 8
    assert entry["collective_attempt_id"] == "attempt-1"
    assert entry["integration_id"] == "integration-1"
    assert entry["validated"] is True


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


def test_a_session_whose_records_never_arrived_says_so(tmp_path):
    """An optimized session with no recorder parts is a gap, not a zero.

    Rebuilding the section from ``state.json`` made a run whose fragments went
    missing look exactly like a run that adopted nothing. The stack in
    ``state.json`` is read only to tell those two apart.
    """
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
    }
    (tmp_path / "state.json").write_text(json.dumps(state), encoding="utf-8")
    (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")

    result = exporter.build(tmp_path)
    optimizations = result["optimizations"]

    assert result["schema_version"] == "hyperloom.session_breakdown.v5.0"
    assert optimizations["schema_version"] == 5
    assert optimizations["source_of_truth"] == "recorder"
    assert optimizations["available"] is False
    assert optimizations["entries"] == []
    assert optimizations["attempts"] == []
    assert any("state.json carries 1 adopted optimization" in warning for warning in result["warnings"])
    assert "optimization_stack" not in result
    assert "attribution" not in result
    assert "geak_invocations" not in result
    assert "forge_invocations" not in result
    assert result["geak"] == {}
    assert "gemm_tuning" not in result


def test_the_key_that_says_records_are_missing_is_there_when_they_are_not(tmp_path):
    """``available`` has to answer on both paths to be worth asking.

    Distinguishing a session whose records never landed from one that adopted
    nothing is what this section is for, and a consumer cannot make that call
    against a key that only appears when the answer is no.
    """
    instrument.record_collective_promotion(
        tmp_path,
        integration_id="integration-1",
        kernel_id="k007",
        baseline_tput=100.0,
        new_tput=130.0,
        gain_pct=30.0,
        ts="2026-01-01T00:00:20+00:00",
    )
    parts = assemble_parts(tmp_path)

    result = collect_recorded_optimizations(
        "s1",
        list(parts.get("operations") or []),
        list(parts.get("measurements") or []),
        list(parts.get("adoptions") or []),
        list(parts.get("artifacts") or []),
        [],
        [],
        [],
    )

    assert result["available"] is True
    assert result["source_of_truth"] == "recorder"
    assert "unavailable_reason" not in result


def test_a_session_that_adopted_nothing_is_not_reported_as_a_gap(tmp_path):
    (tmp_path / "state.json").write_text(
        json.dumps({"session_id": "quiet", "baseline_tput": 100.0}),
        encoding="utf-8",
    )
    (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")

    result = exporter.build(tmp_path)

    assert result["optimizations"]["available"] is False
    assert not any("state.json carries" in warning for warning in result["warnings"])


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

    result = collect_recorded_optimizations("s1", operations, measurements, adoptions, artifacts, [], [], warnings)

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

    result = collect_recorded_optimizations("s1", operations, measurements, adoptions, artifacts, [], [], [])

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

    result = collect_recorded_optimizations("s1", operations, measurements, adoptions, artifacts, [], [], [])

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

    result = collect_recorded_optimizations("s1", operations, measurements, adoptions, artifacts, [], [], [])

    entry = result["entries"][0]
    attempt = next(row for row in result["attempts"] if row["attempt_id"] == entry["adopted_attempt_id"])
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

    result = collect_recorded_optimizations("s1", operations, measurements, adoptions, artifacts, [], [], [])

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
    result = collect_recorded_optimizations("gemma", operations, measurements, adoptions, [], [], [], warnings)

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
    assert result["summary_by_agent"]["framework_agent"]["attributable_gain_pct"] == 11.769809
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

    result = collect_recorded_optimizations("s1", operations, measurements, adoptions, [], [], [], warnings)

    entry = result["entries"][0]
    # It started at 1100 and left at 1210, so it added 11pp of the baseline —
    # not the 21pp it would inherit by being measured from the baseline.
    assert entry["gain_pct"] == 11.0
    assert entry["cumulative_gain_pct"] == 21.0
    assert result["validation"]["unattributed_gain_pct"] == 10.0
    assert any("belongs to no attempt" in warning for warning in warnings)


def test_a_step_that_recorded_only_a_percentage_is_not_counted_twice():
    """A step with no finishing throughput used to be paid for twice.

    Its own figure went into the total, and then the next step's head start —
    which is that same figure — was booked again as drift. The percentage is
    measured against where the step started, so the missing reading can be put
    back and the chain carried on.
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
            "ended_at": "2026-01-01T01:00:00+00:00",
        },
        {
            "operation_id": "op-gemm",
            "kind": "gemm_tuning",
            "name": "gemm",
            "agent": "kernel_agent",
            "ended_at": "2026-01-01T02:00:00+00:00",
        },
        {
            "operation_id": "op-k2",
            "kind": "kernel_optimization",
            "name": "k2",
            "agent": "kernel_agent",
            "ended_at": "2026-01-01T03:00:00+00:00",
        },
    ]
    measurements = [{"measurement_id": "m-baseline", "name": "throughput", "value": 1000.0}]
    adoptions = [
        {
            "adoption_id": "ad-k1",
            "operation_id": "op-k1",
            "decision": "KEEP",
            "validated": True,
            "throughput_before": 1000.0,
            "throughput_after": 1100.0,
        },
        # A GEMM adoption carries the speedup it was decided on and no
        # throughput at all.
        {
            "adoption_id": "ad-gemm",
            "operation_id": "op-gemm",
            "decision": "KEEP",
            "validated": True,
            "gain_pct": 5.0,
        },
        {
            "adoption_id": "ad-k2",
            "operation_id": "op-k2",
            "decision": "KEEP",
            "validated": True,
            "throughput_before": 1155.0,
            "throughput_after": 1250.0,
        },
    ]
    warnings: list[str] = []

    result = collect_recorded_optimizations("s1", operations, measurements, adoptions, [], [], [], warnings)

    first, middle, last = result["entries"]
    assert first["gain_pct"] == 10.0
    # 5% of the 1100 it started from is 5.5 points of the 1000 baseline, not 5.
    assert middle["gain_method"] == "local_gain_projected"
    assert middle["gain_pct"] == 5.5
    assert middle["chain_continuous"] is False
    assert last["gain_pct"] == 9.5
    # The session moved 1000 -> 1250. Booking the middle step's effect once
    # gives exactly that; booking it again as the last step's drift gave 30.
    assert last["cumulative_gain_pct"] == 25.0
    assert result["validation"]["unattributed_gain_pct"] == 0.0
    assert any("recorded no finishing throughput" in warning for warning in warnings)


def test_the_session_total_prefers_what_the_run_measured_over_its_own_sum():
    """A total summed from the ledger can never be found to disagree with it.

    The run promotes an end-to-end figure of its own when it validates. That
    figure is the one the section reports, and the ledger's sum is kept beside
    it so the two can be seen to part company.
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
            "ended_at": "2026-01-01T01:00:00+00:00",
        },
        {
            "operation_id": "op-validated",
            "kind": "session_validation",
            "name": "cumulative_gain_validated",
            "ended_at": "2026-01-01T01:05:00+00:00",
            "outputs": {
                "validated_at_stack_len": 1,
                "source": "integrate_patch",
                "measurement_basis": "e2e_rebench",
                "validated_gain_pct": 14.0,
            },
        },
    ]
    measurements = [{"measurement_id": "m-baseline", "name": "throughput", "value": 1000.0}]
    adoptions = [
        {
            "adoption_id": "ad-k1",
            "operation_id": "op-k1",
            "decision": "KEEP",
            "validated": True,
            "throughput_before": 1000.0,
            "throughput_after": 1100.0,
        }
    ]
    warnings: list[str] = []

    result = collect_recorded_optimizations("s1", operations, measurements, adoptions, [], [], [], warnings)

    validation = result["validation"]
    assert validation["method"] == "recorded_session_validation"
    assert validation["validated_total_gain_pct"] == 14.0
    assert validation["ledger_total_gain_pct"] == 10.0
    assert validation["reconciliation_gap_pct"] == 4.0
    assert validation["validation_basis"] == "e2e_rebench"
    assert validation["validation_source"] == "integrate_patch"
    # The session validation is not an attempt and must not become a ledger row.
    assert len(result["entries"]) == 1
    assert any("but the run promoted" in warning for warning in warnings)


def test_each_promotion_leaves_its_own_checkpoint(tmp_path):
    """Two checkpoints that measure the same number are still two checkpoints.

    Keying on the value would collapse them, which is the trap an earlier fix
    already had to dig the measurement ids out of.
    """
    for stack_len, ts in ((1, "2026-01-01T01:00:00+00:00"), (2, "2026-01-01T02:00:00+00:00")):
        instrument.record_session_validation(
            tmp_path,
            baseline_tput=1000.0,
            validated_tput=1100.0,
            validated_gain_pct=10.0,
            stack_len=stack_len,
            source="integrate_patch",
            measurement_basis="e2e_rebench",
            ts=ts,
        )

    operations = [
        row for row in assemble_parts(tmp_path).get("operations") or [] if row.get("kind") == "session_validation"
    ]

    assert len(operations) == 2
    # The newest checkpoint is the one the export reports.
    result = collect_recorded_optimizations("s1", operations, [], [], [], [], [], [])
    assert result["validation"]["validated_at_stack_len"] == 2


def test_a_session_with_no_promoted_figure_falls_back_to_its_own_sum():
    operations = [
        {
            "operation_id": "op-k1",
            "kind": "kernel_optimization",
            "name": "k1",
            "agent": "kernel_agent",
            "ended_at": "2026-01-01T01:00:00+00:00",
        },
    ]
    adoptions = [
        {
            "adoption_id": "ad-k1",
            "operation_id": "op-k1",
            "decision": "KEEP",
            "validated": True,
            "gain_pct": 5.0,
        }
    ]
    warnings: list[str] = []

    result = collect_recorded_optimizations("s1", operations, [], adoptions, [], [], [], warnings)

    validation = result["validation"]
    assert validation["method"] == "ledger_sum"
    assert validation["validated_total_gain_pct"] == 5.0
    assert validation["reconciliation_gap_pct"] is None


def test_a_keep_no_accuracy_gate_ruled_on_is_counted_as_such():
    operations = [
        {
            "operation_id": "op-fa",
            "kind": "framework_agent",
            "name": "framework_agent",
            "agent": "framework_agent",
            "ended_at": "2026-01-01T01:00:00+00:00",
        },
    ]
    adoptions = [
        {
            "adoption_id": "ad-fa",
            "operation_id": "op-fa",
            "decision": "KEEP",
            "validated": True,
            "gain_pct": 3.0,
            "validation_basis": "keep_verdict_unscored",
        }
    ]

    result = collect_recorded_optimizations("s1", operations, [], adoptions, [], [], [], [])

    assert result["attempts"][0]["validation_basis"] == "keep_verdict_unscored"
    assert result["validation"]["unscored_keep_count"] == 1


def test_both_sides_of_the_record_name_a_patch_author_the_same_way():
    """The rule used to exist twice, and a copy that drifts moves gain.

    The write side stamps an owner when the patch lands; the read side has to
    name one for sessions recorded before it did. They answer with the same
    function or they eventually answer differently.
    """
    from hyperloom.inference_optimizer.breakdown.recorder.instrument import _resolve_agent

    cases = [
        ({"framework_agent_authoring": True, "source_phase": "KERNEL_AGENT"}, "framework_agent"),
        (
            {
                "provenance": "specialist:serving",
                "domain": "serving_specialist",
                "source_phase": "FRAMEWORK_AGENT",
            },
            "framework_agent",
        ),
        ({"provenance": "specialist:latency", "source_phase": "KERNEL_AGENT"}, "unattributed"),
        ({"domain": "attention"}, "unattributed"),
        ({"source_phase": "KERNEL_AGENT"}, "unattributed"),
        ({}, "unattributed"),
    ]
    for evidence, expected in cases:
        write_side = _resolve_agent("integrate_patch", result=evidence)
        read_side = collect_recorded_optimizations(
            "s1",
            [
                {
                    "operation_id": "op-patch",
                    "kind": "integrate_patch",
                    "name": "integrate_patch",
                    "outputs": evidence,
                    "ended_at": "2026-01-01T01:00:00+00:00",
                }
            ],
            [],
            [],
            [],
            [],
            [],
            [],
        )["attempts"][0]["agent"]

        assert write_side == expected, evidence
        assert read_side == expected, evidence


def test_a_value_says_which_of_its_possible_sources_it_came_from():
    """A stated verdict and an inferred status are different claims."""
    operations = [
        {
            "operation_id": "op-a",
            "kind": "kernel_optimization",
            "name": "stated",
            "agent": "kernel_agent",
            "ended_at": "2026-01-01T01:00:00+00:00",
        },
        {
            "operation_id": "op-b",
            "kind": "kernel_optimization",
            "name": "inferred",
            "agent": "kernel_agent",
            "status": "succeeded",
            "ended_at": "2026-01-01T02:00:00+00:00",
        },
    ]
    adoptions = [
        {
            "adoption_id": "ad-a",
            "operation_id": "op-a",
            "decision": "KEEP",
            "validated": True,
            "gain_pct": 4.0,
            "throughput_before": 1000.0,
            "throughput_after": 1040.0,
        }
    ]

    attempts = collect_recorded_optimizations("s1", operations, [], adoptions, [], [], [], [])["attempts"]
    stated = next(row for row in attempts if row["name"] == "stated")
    inferred = next(row for row in attempts if row["name"] == "inferred")

    assert stated["decision_source"] == "adoption.decision"
    assert stated["local_gain_source"] == "adoption.gain_pct"
    assert stated["throughput_before_source"] == "adoption"
    assert stated["throughput_after_source"] == "adoption"
    assert inferred["decision_source"] == "operation.status"
    assert inferred["local_gain_source"] == ""
    assert inferred["throughput_before_source"] == ""
    assert inferred["throughput_after_source"] == ""


def test_an_adoption_whose_operation_was_never_recorded_is_reported():
    """The ledger walks operations, so this adoption's gain simply vanishes."""
    operations = [
        {
            "operation_id": "op-k1",
            "kind": "kernel_optimization",
            "name": "k1",
            "agent": "kernel_agent",
            "ended_at": "2026-01-01T01:00:00+00:00",
        },
    ]
    adoptions = [
        {
            "adoption_id": "ad-k1",
            "operation_id": "op-k1",
            "decision": "KEEP",
            "validated": True,
        },
        {
            "adoption_id": "ad-ghost",
            "operation_id": "op-never-written",
            "decision": "KEEP",
            "validated": True,
            "gain_pct": 4.0,
        },
    ]
    warnings: list[str] = []

    collect_recorded_optimizations("s1", operations, [], adoptions, [], [], [], warnings)

    assert any("never recorded" in warning and "ad-ghost" in warning for warning in warnings)


def test_one_producers_singleton_is_not_dropped_for_anothers_without_a_word(tmp_path):
    """A singleton fragment is named for its producer, so two mean two claims.

    Only the newest survives, and the loser does not merge into it: its whole
    payload goes. Nothing downstream can see that it was ever written.
    """
    for producer, ts in (("coordinator", "2026-01-01T01:00:00+00:00"), ("kernel-agent", "2026-01-01T02:00:00+00:00")):
        instrument.record_run_snapshot(
            tmp_path,
            payload={"run_id": "r1", "recorded_by": producer, "ts": ts},
            producer=producer,
        )
    warnings: list[str] = []

    assemble_parts(tmp_path, warnings=warnings)

    assert any("more than one producer" in warning and "dropped whole" in warning for warning in warnings)


def test_two_producers_disagreeing_on_one_entity_do_not_settle_it_silently(tmp_path):
    """Merging partial updates is the point; disagreeing on a field is not.

    Repeated updates from one producer merge into its own fragment long before
    assembly, so two payloads for one id are two producers, and the later
    timestamp decides the value with nothing said about the one it replaced.
    """
    for producer, decision in (("coordinator", "KEEP"), ("kernel-agent", "REVERT")):
        instrument.record_adoption(
            tmp_path,
            adoption_id="ad-1",
            operation_id="op-1",
            decision=decision,
            producer=producer,
        )
    warnings: list[str] = []

    assemble_parts(tmp_path, warnings=warnings)

    assert any("conflicting values" in warning and "ad-1.decision" in warning for warning in warnings)


def test_a_change_that_landed_with_nobody_claiming_it_is_reported():
    """The mirror of an orphan adoption, and the one that moves a number.

    The step is skipped by the gain walk, but the workload still moved, so the
    next adopted step starts higher than the ledger expects and the difference
    is booked as gain belonging to nobody. Unreported, that reads as ordinary
    drift rather than as a record that never arrived.
    """
    operations = [
        {
            "operation_id": "op-base",
            "kind": "baseline",
            "measurement_refs": ["m-base"],
        },
        {
            "operation_id": "op-lost",
            "kind": "kernel_optimization",
            "name": "k-lost",
            "agent": "kernel_agent",
            "ended_at": "2026-01-01T01:00:00+00:00",
            "outputs": {"integrated": True, "decision": "KEEP"},
        },
        {
            "operation_id": "op-next",
            "kind": "kernel_optimization",
            "name": "k-next",
            "agent": "kernel_agent",
            "ended_at": "2026-01-01T02:00:00+00:00",
        },
    ]
    measurements = [{"measurement_id": "m-base", "name": "throughput", "value": 100.0}]
    adoptions = [
        {
            "adoption_id": "ad-next",
            "operation_id": "op-next",
            "decision": "KEEP",
            "validated": True,
            "throughput_before": 110.0,
            "throughput_after": 120.0,
        },
    ]
    warnings: list[str] = []

    result = collect_recorded_optimizations("s1", operations, measurements, adoptions, [], [], [], warnings)

    validation = result["validation"]
    assert validation["unclaimed_integration_count"] == 1
    # The 10pp the lost step earned is sitting in the unattributed bucket.
    assert validation["unattributed_gain_pct"] == 10.0
    assert any("no adoption crediting it" in warning and "op-lost" in warning for warning in warnings)


def test_a_threshold_says_which_of_its_four_homes_it_came_from():
    """A bar the gate ruled against and one read off a config are not the same."""
    operations = [
        {
            "operation_id": "op-gated",
            "kind": "kernel_optimization",
            "name": "gated",
            "ended_at": "2026-01-01T01:00:00+00:00",
            "gates": [{"inputs": {"keep_threshold_pct": 3.0}}],
            "outputs": {"keep_threshold_pct": 1.0},
        },
        {
            "operation_id": "op-configured",
            "kind": "kernel_optimization",
            "name": "configured",
            "ended_at": "2026-01-01T02:00:00+00:00",
            "outputs": {"keep_threshold_pct": 1.0},
        },
    ]

    attempts = collect_recorded_optimizations("s1", operations, [], [], [], [], [], [])["attempts"]
    gated = next(row for row in attempts if row["name"] == "gated")
    configured = next(row for row in attempts if row["name"] == "configured")

    assert (gated["keep_threshold_pct"], gated["keep_threshold_source"]) == (3.0, "gate.inputs")
    assert (configured["keep_threshold_pct"], configured["keep_threshold_source"]) == (1.0, "outputs")


def test_two_names_for_one_reading_do_not_quietly_pick_one():
    """Both names fill the same role, and they were taken by different producers."""
    operations = [
        {
            "operation_id": "op-k1",
            "kind": "kernel_optimization",
            "name": "k1",
            "ended_at": "2026-01-01T01:00:00+00:00",
            "measurement_refs": ["m-final", "m-plain"],
        },
    ]
    measurements = [
        {
            "measurement_id": "m-final",
            "name": "final_throughput",
            "value": 120.0,
            "measured_at": "2026-01-01T00:30:00+00:00",
        },
        {
            "measurement_id": "m-plain",
            "name": "throughput",
            "value": 118.0,
            "measured_at": "2026-01-01T00:40:00+00:00",
        },
    ]
    warnings: list[str] = []

    attempts = collect_recorded_optimizations("s1", operations, measurements, [], [], [], [], warnings)["attempts"]

    assert attempts[0]["throughput_after"] == 120.0
    assert attempts[0]["throughput_after_source"] == "measurement.final_throughput"
    assert attempts[0]["alias_conflicts"] == ["throughput_after:final_throughput/throughput"]
    assert any("the first read won" in warning for warning in warnings)


def test_an_adoption_on_a_kind_the_ledger_ignores_is_reported():
    """A kind missing from the attempt table is how a new optimizer goes uncounted."""
    operations = [
        {
            "operation_id": "op-profile",
            "kind": "composite",
            "name": "profile",
            "ended_at": "2026-01-01T01:00:00+00:00",
        },
    ]
    adoptions = [
        {
            "adoption_id": "ad-profile",
            "operation_id": "op-profile",
            "decision": "KEEP",
            "validated": True,
            "gain_pct": 3.0,
        },
    ]
    warnings: list[str] = []

    result = collect_recorded_optimizations("s1", operations, [], adoptions, [], [], [], warnings)

    assert result["entries"] == []
    assert any("does not count as attempts" in warning and "profile" in warning for warning in warnings)


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

    result = collect_recorded_optimizations("s1", operations, measurements, adoptions, [], [], [], [])

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

    attempt = collect_recorded_optimizations("s1", operations, measurements, adoptions, [], [], [], [])["attempts"][0]

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

    attempt = collect_recorded_optimizations("s1", operations, measurements, adoptions, [], [], [], [])["attempts"][0]

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

    attempt = collect_recorded_optimizations("s1", operations, measurements, adoptions, [], [], [], [])["attempts"][0]
    numbered = {row["value"]: row for row in attempt["measurements"]}

    # The reading the decision was made on is the first of its name, even
    # though the operation happens to reference the later one first.
    assert numbered[5081.01]["occurrence"] == 0
    assert numbered[5100.76]["occurrence"] == 1
    assert numbered[5081.01]["occurrences_of_name"] == 2
    # A name measured once is numbered too, rather than left to be guessed at.
    assert numbered[4744.6]["occurrence"] == 0
    assert numbered[4744.6]["occurrences_of_name"] == 1


def _ledger_with_baseline(tmp_path, baseline_tput: float):
    """Assemble the recorded parts plus the baseline reading gains are measured against."""
    parts = assemble_parts(tmp_path)
    operations = list(parts.get("operations") or [])
    measurements = list(parts.get("measurements") or [])
    operations.append({"operation_id": "op-base", "kind": "baseline", "measurement_refs": ["m-base"]})
    measurements.append({"measurement_id": "m-base", "name": "throughput", "value": baseline_tput})
    return collect_recorded_optimizations(
        "s1",
        operations,
        measurements,
        list(parts.get("adoptions") or []),
        list(parts.get("artifacts") or []),
        [],
        [],
        [],
    )


def test_a_reproduced_warm_replay_is_an_adopted_step_in_the_ledger(tmp_path):
    """A replay the run promoted has to reach the ledger as an adopted step.

    The keep decision belongs to the promote path, not to the replay executor,
    which settles on ``succeeded`` either way. Mirroring the action before that
    ruling recorded every replay as discarded, so a reproduced one was pushed
    onto the stack and moved ``cumulative_gain_validated`` while the canonical
    streams held no adoption for it: ``entries`` came back empty on a session
    that had measurably gained, and its whole gain read as unattributed.
    """
    instrument.record_action_operation(
        tmp_path,
        action="replay_warm_recipe",
        task_id="warm-1",
        status="kept",
        decision="promoted",
        result={
            "status": "kept",
            "base_tput": 638.08,
            "output_throughput": 1907.49,
            "delta_pct": 198.94,
            "attribution_eligible": True,
            "provenance": "warm_replay",
            "validated": True,
        },
        phase="PRELUDE",
    )

    result = _ledger_with_baseline(tmp_path, 638.08)

    entry = result["entries"][0]
    assert entry["optimization_kind"] == "replay_warm_recipe"
    assert entry["source"] == "warm_replay"
    assert entry["gain_method"] == "baseline_chain"
    assert entry["gain_pct"] == pytest.approx(198.94, abs=0.01)
    # The ledger and the gain the run promoted are the same number, so the
    # session reports no reconciliation gap.
    assert result["validation"]["ledger_total_gain_pct"] == pytest.approx(198.94, abs=0.01)
    assert result["validation"]["unattributed_gain_pct"] == 0.0
    assert result["validation"]["keep_count"] == 1


def test_a_replay_that_did_not_reproduce_stays_out_of_the_ledger(tmp_path):
    """Drift is a measured non-result, and must not be credited as a keep.

    The fix for the discarded-reproduced replay must not reach the other way
    and let a replay that missed the bar claim gain it never earned.
    """
    instrument.record_action_operation(
        tmp_path,
        action="replay_warm_recipe",
        task_id="warm-2",
        status="succeeded",
        decision="discarded",
        result={
            "status": "succeeded",
            "base_tput": 638.08,
            "output_throughput": 600.0,
            "delta_pct": -5.9,
        },
        phase="PRELUDE",
    )

    result = _ledger_with_baseline(tmp_path, 638.08)

    assert result["entries"] == []
    assert result["validation"]["ledger_total_gain_pct"] == 0.0
    # The attempt is still on the record; only the credit is withheld.
    assert result["attempts"][0]["decision"] == "DISCARDED"
    assert result["attempts"][0]["adopted"] is False


def test_a_replay_promoted_without_an_accuracy_verdict_says_so(tmp_path):
    """Adopting on a keep verdict alone is a different record from passing a gate.

    A replay is admitted when its eval could not be scored, so the ledger has to
    carry that it was never checked rather than report it as validated.
    """
    instrument.record_action_operation(
        tmp_path,
        action="replay_warm_recipe",
        task_id="warm-3",
        status="kept",
        decision="promoted",
        result={
            "status": "kept",
            "base_tput": 100.0,
            "output_throughput": 130.0,
            "delta_pct": 30.0,
            "attribution_eligible": True,
            # No accuracy verdict: the eval never produced a score.
            "validated": None,
        },
        phase="PRELUDE",
    )

    result = _ledger_with_baseline(tmp_path, 100.0)

    assert result["entries"][0]["gain_pct"] == pytest.approx(30.0, abs=0.01)
    assert result["attempts"][0]["validation_basis"] == "keep_verdict_unscored"
    assert result["validation"]["unscored_keep_count"] == 1
