# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Tests for the additive ``kernel_journey`` breakdown section.

Covers the recorder substreams (discovery / dispatch / backend_result / e2e),
their assembly into the kernel-major view, and the guarantee that the section
stays absent (so historical breakdowns are byte-for-byte unchanged) when no
substream was recorded.
"""

from __future__ import annotations

from pathlib import Path

from inference_optimizer.breakdown.recorder import (
    assemble_parts,
    instrument,
)


def test_kernel_journey_absent_without_substreams(tmp_path: Path) -> None:
    # Only an unrelated section recorded -> kernel_journey must not appear.
    instrument.record_phase_event(
        tmp_path, action="profile",
        entry={"task_id": "t1", "status": "succeeded"},
    )
    out = assemble_parts(tmp_path)
    assert "kernel_journey" not in out


def test_kernel_journey_composes_full_lifecycle(tmp_path: Path) -> None:
    instrument.record_kernel_discovery(
        tmp_path, source="tracelens", status="success",
        hot_kernels=[
            {"kernel_id": "k001", "name": "moe", "gpu_pct": 42.0,
             "bottleneck": "memory", "reusable_native_kernel": True,
             "recommended_backends": ["geak", "oob"]},
            {"kernel_id": "k002", "name": "ln", "gpu_pct": 7.5},
        ],
        scan={"splitter_mode": "auto", "candidates_path": str(tmp_path / "c.json")},
    )
    instrument.record_kernel_dispatch(
        tmp_path, kernel_id="k001", dispatched=True,
        backends=["geak", "claude"], orchestration_commit="abc1234",
    )
    instrument.record_kernel_dispatch(
        tmp_path, kernel_id="k002", dispatched=False,
        skip_reason="non_reusable_kernel",
    )
    instrument.record_kernel_backend_result(tmp_path, {
        "kernel_id": "k001", "run_id": "r1", "attempts": [
            {"attempt_id": "a1", "backend": "geak", "status": "succeeded",
             "decision": "KEEP", "micro_speedup": 1.8, "compile_passed": True,
             "correctness_passed": "pass", "duration_sec": 120.5},
            {"attempt_id": "a2", "backend": "claude", "status": "failed",
             "decision": "FAILED", "error": "compile err"},
        ],
    })
    instrument.record_kernel_e2e(
        tmp_path, kernel_id="k001", integrated=True, e2e_gain_pct=3.2,
        validated=True, decision="KEEP", patch_path="patches/k001.patch",
        target_file="moe.py",
    )

    out = assemble_parts(tmp_path)
    kj = out["kernel_journey"]

    # Raw substreams are popped, never leaking into the envelope.
    for raw in ("kernel_discovery", "kernel_dispatch",
                "kernel_backend_result", "kernel_e2e"):
        assert raw not in out

    assert len(kj["discovery_runs"]) == 1
    assert kj["discovery_runs"][0]["hot_kernel_count"] == 2

    # Sorted by gpu_pct desc.
    assert [k["kernel_id"] for k in kj["kernels"]] == ["k001", "k002"]

    k001 = kj["kernels"][0]
    assert k001["outcome"] == "adopted"
    assert k001["dispatch"]["dispatched"] is True
    assert k001["dispatch"]["orchestration_commit"] == "abc1234"
    assert len(k001["backend_attempts"]) == 2
    assert k001["backend_attempts"][0]["correctness_passed"] is True
    assert k001["backend_attempts"][0]["duration_sec"] == 120.5
    assert k001["e2e"]["e2e_gain_pct"] == 3.2

    k002 = kj["kernels"][1]
    assert k002["outcome"] == "skipped"
    assert k002["dispatch"]["skip_reason"] == "non_reusable_kernel"
    assert k002["backend_attempts"] == []


def test_kernel_backend_result_keeps_retries_across_runs(tmp_path: Path) -> None:
    # Same kernel/backend, two different runs -> two distinct attempts.
    for run in ("r1", "r2"):
        instrument.record_kernel_backend_result(tmp_path, {
            "kernel_id": "k001", "run_id": run, "attempts": [
                {"attempt_id": "", "backend": "geak", "status": "failed",
                 "decision": "FAILED"},
            ],
        })
    out = assemble_parts(tmp_path)
    attempts = out["kernel_journey"]["kernels"][0]["backend_attempts"]
    assert len(attempts) == 2
