# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json

from hyperloom.inference_optimizer.breakdown import exporter
from hyperloom.inference_optimizer.breakdown.collectors import (
    collect_attribution,
    collect_recorded_optimizations,
)
from hyperloom.inference_optimizer.breakdown.recorder import assemble_parts, instrument


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
    assert any(
        "state.json carries 1 adopted optimization" in warning
        for warning in result["warnings"]
    )
    assert "optimization_stack" not in result
    assert "attribution" not in result
    assert "geak_invocations" not in result
    assert "forge_invocations" not in result
    assert "geak" not in result
    assert "gemm_tuning" not in result


def test_a_session_that_adopted_nothing_is_not_reported_as_a_gap(tmp_path):
    (tmp_path / "state.json").write_text(
        json.dumps({"session_id": "quiet", "baseline_tput": 100.0}),
        encoding="utf-8",
    )
    (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")

    result = exporter.build(tmp_path)

    assert result["optimizations"]["available"] is False
    assert not any(
        "state.json carries" in warning for warning in result["warnings"]
    )


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

    result = collect_recorded_optimizations(
        "s1", operations, measurements, adoptions, [], [], [], warnings
    )

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

    result = collect_recorded_optimizations(
        "s1", operations, measurements, adoptions, [], [], [], warnings
    )

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
        row
        for row in assemble_parts(tmp_path).get("operations") or []
        if row.get("kind") == "session_validation"
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

    result = collect_recorded_optimizations(
        "s1", operations, [], adoptions, [], [], [], warnings
    )

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

    result = collect_recorded_optimizations(
        "s1", operations, [], adoptions, [], [], [], warnings
    )

    assert result["entries"] == []
    assert any(
        "does not count as attempts" in warning and "profile" in warning
        for warning in warnings
    )


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

