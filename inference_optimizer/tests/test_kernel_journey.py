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


def test_kernel_backend_result_records_pre_dispatch_failure(tmp_path: Path) -> None:
    # Backend failed before running any attempt (empty attempts + failed status)
    # -> a synthetic FAILED marker so the failure is visible in kernel_journey.
    instrument.record_kernel_backend_result(tmp_path, {
        "kernel_id": "k001", "run_id": "r1", "attempts": [],
        "status": "failed", "error_class": "non_reusable_kernel",
        "error": "empty kernel shape", "backend": "geak",
    })
    out = assemble_parts(tmp_path)
    attempts = out["kernel_journey"]["kernels"][0]["backend_attempts"]
    assert len(attempts) == 1
    att = attempts[0]
    assert att["decision"] == "FAILED"
    assert att["pre_dispatch_failure"] is True
    assert att["error_class"] == "non_reusable_kernel"
    assert att["backend"] == "geak"
    # With an attempt present the kernel reads as "attempted", not "skipped".
    assert out["kernel_journey"]["kernels"][0]["outcome"] == "attempted"


def test_backend_attempt_maps_kernel_agent_field_names(tmp_path: Path) -> None:
    # kernel-agent emits elapsed_s / created_at / error_type and keeps the
    # achieved speedup at the kernel level in verification (best attempt). The
    # recorder must map those onto the journey attempt + entry.
    instrument.record_kernel_backend_result(tmp_path, {
        "kernel_id": "k001", "run_id": "r1",
        "verification": {"micro_speedup": 1.42, "best_attempt_id": "a1"},
        "attempts": [
            {"attempt_id": "a1", "backend": "geak", "status": "completed",
             "elapsed_s": 87.5, "created_at": "2026-06-12T00:00:00Z"},
            {"attempt_id": "a2", "backend": "claude", "status": "timeout",
             "error_type": "timeout", "elapsed_s": 12.0},
        ],
    })
    out = assemble_parts(tmp_path)
    entry = out["kernel_journey"]["kernels"][0]
    a1, a2 = entry["backend_attempts"]
    assert a1["duration_sec"] == 87.5
    assert a1["ts"] == "2026-06-12T00:00:00Z"
    # kernel-level best speedup stamped onto the adopted attempt.
    assert a1["micro_speedup"] == 1.42
    assert a2["error_class"] == "timeout"
    # Entry exposes the best achieved speedup for the e2e correlation.
    assert entry["micro_speedup"] == 1.42


def test_discovery_run_carries_duration(tmp_path: Path) -> None:
    instrument.record_kernel_discovery(
        tmp_path, source="tracelens", status="success",
        hot_kernels=[{"kernel_id": "k1", "name": "moe", "gpu_pct": 10.0}],
        scan={}, duration_sec=4.2,
    )
    out = assemble_parts(tmp_path)
    assert out["kernel_journey"]["discovery_runs"][0]["duration_sec"] == 4.2


def test_attach_kernel_roofline_enriches_journey() -> None:
    from inference_optimizer.breakdown.exporter import _attach_kernel_roofline

    kernel_journey = {
        "discovery_runs": [],
        "kernels": [{
            "kernel_id": "k001", "name": "moe", "gpu_pct": 42.0,
            "bound_type": "",
            "discovery": {
                "kernel_id": "k001", "bound_type": "",
                "arithmetic_intensity": None, "efficiency_percent": None,
            },
            "backend_attempts": [],
        }],
    }
    kernel_roofline = {
        "kernels": [{
            "kernel_id": "k001", "name": "moe", "bound_type": "memory",
            "arithmetic_intensity": 3.5, "flops_per_byte": 2.1,
            "efficiency_percent": 61.0, "rocprof_roofline": {"foo": "bar"},
        }],
    }
    _attach_kernel_roofline(kernel_journey, kernel_roofline)
    entry = kernel_journey["kernels"][0]
    assert entry["roofline"]["arithmetic_intensity"] == 3.5
    assert entry["roofline"]["rocprof_roofline"] == {"foo": "bar"}
    # Header + discovery numeric fields backfilled from roofline.
    assert entry["bound_type"] == "memory"
    assert entry["discovery"]["bound_type"] == "memory"
    assert entry["discovery"]["arithmetic_intensity"] == 3.5
    assert entry["discovery"]["efficiency_percent"] == 61.0
