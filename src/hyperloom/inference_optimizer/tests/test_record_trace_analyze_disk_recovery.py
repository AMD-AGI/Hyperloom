# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``record_trace_analyze`` recovers a thin TraceLens envelope from the on-disk
kernel-roofline report.

A ``status=ok`` envelope that lost its payload keys in transit used to cache an
empty ``last_trace_analyze``, which silently emptied every downstream consumer
(``roofline_snapshots``, the specialist ROOFLINE EVIDENCE section, the kernel
phase) while the analysis itself sat complete on disk.
"""

from __future__ import annotations

import json
from pathlib import Path

from hyperloom.orchestrator.state.shared_state import SharedState


def _write_roofline_report(
    session_dir: Path,
    *,
    filename: str = "kernel_roofline_current.json",
    analysis_md_path: str = "",
    kernel_candidates_path: str = "",
    kernels: list[dict] | None = None,
) -> Path:
    """Write a session-level kernel-roofline report and return its path."""
    reports = session_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    path = reports / filename
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source": "tracelens_analysis",
                "trace_input": "/trace",
                "trace_input_type": "capture_dir",
                "analysis_md_path": analysis_md_path,
                "kernel_candidates_path": kernel_candidates_path,
                "roofline_json_path": "",
                "kernels": kernels if kernels is not None else [],
            }
        ),
        encoding="utf-8",
    )
    return path


def _attention_kernels() -> list[dict]:
    """Two hot kernels shaped like a real TraceLens report (attention on top)."""
    return [
        {
            "kernel_id": "k001",
            "name": "aten::_flash_attention_forward",
            "gpu_pct": 40.0,
            "duration_us": 681195.0,
            "bottleneck": "compute",
            "bound_type": "compute",
            "source_file": "attention.py(243): sequence_parallel_attention_vision",
            "reusable_native_kernel": True,
            "suggestion": "substitute a vendor attention backend",
            "recommended_actions": ["swap backend"],
        },
        {
            "kernel_id": "k002",
            "name": "aten::cat",
            "gpu_pct": 1.1,
            "duration_us": 18941.0,
            "bottleneck": "memory",
            "bound_type": "memory",
            "source_file": "attention.py(243): sequence_parallel_attention_vision",
            "reusable_native_kernel": False,
        },
    ]


def test_thin_envelope_recovers_hot_kernels_from_disk(tmp_path: Path) -> None:
    """``hot_kernels`` absent from the envelope → recovered from the report."""
    analysis_md = tmp_path / "analysis.md"
    analysis_md.write_text("# Executive Summary\nCompute 51.6%\n", encoding="utf-8")
    _write_roofline_report(
        tmp_path,
        analysis_md_path=str(analysis_md),
        kernel_candidates_path=str(tmp_path / "kernel_candidates.json"),
        kernels=_attention_kernels(),
    )

    state = SharedState.load_or_init(tmp_path)
    state.record_trace_analyze(
        {
            "trace_input": "/trace",
            "roofline_output_name": "kernel_roofline_current.json",
        },
        # The defect: status=ok but every payload key lost in transit.
        {"status": "ok"},
    )

    cached = state.last_trace_analyze
    assert [k["name"] for k in cached["hot_kernels_top15"]] == [
        "aten::_flash_attention_forward",
        "aten::cat",
    ]
    assert cached["hot_kernels_top15"][0]["gpu_pct"] == 40.0
    assert cached["analysis_md_path"] == str(analysis_md)
    assert "Executive Summary" in cached["analysis_md_text"]
    assert cached["candidates_path"] == str(tmp_path / "kernel_candidates.json")


def test_recovered_hot_kernels_are_ordered_by_gpu_pct(tmp_path: Path) -> None:
    """Report rows in arbitrary order → cached top-N is sorted by GPU share."""
    _write_roofline_report(
        tmp_path,
        kernels=[
            {"kernel_id": "k_small", "name": "aten::cat", "gpu_pct": 1.1},
            {"kernel_id": "k_big", "name": "aten::_flash_attention_forward", "gpu_pct": 40.0},
            {"kernel_id": "k_mid", "name": "aten::addmm", "gpu_pct": 2.1},
        ],
    )

    state = SharedState.load_or_init(tmp_path)
    state.record_trace_analyze(
        {"trace_input": "/trace", "roofline_output_name": "kernel_roofline_current.json"},
        {"status": "ok"},
    )

    assert [k["kernel_id"] for k in state.last_trace_analyze["hot_kernels_top15"]] == [
        "k_big",
        "k_mid",
        "k_small",
    ]


def test_envelope_hot_kernels_win_over_disk(tmp_path: Path) -> None:
    """A populated envelope is authoritative; the report is only a fallback."""
    _write_roofline_report(tmp_path, kernels=_attention_kernels())

    state = SharedState.load_or_init(tmp_path)
    state.record_trace_analyze(
        {"trace_input": "/trace", "roofline_output_name": "kernel_roofline_current.json"},
        {
            "status": "ok",
            "hot_kernels": [{"kernel_id": "from_envelope", "name": "aten::mm", "gpu_pct": 9.0}],
        },
    )

    cached = state.last_trace_analyze
    assert [k["kernel_id"] for k in cached["hot_kernels_top15"]] == ["from_envelope"]


def test_recovery_is_silent_when_no_report_on_disk(tmp_path: Path) -> None:
    """No report → previous behaviour (empty summary), no exception."""
    state = SharedState.load_or_init(tmp_path)
    state.record_trace_analyze(
        {"trace_input": "/trace", "roofline_output_name": "kernel_roofline_current.json"},
        {"status": "ok"},
    )

    cached = state.last_trace_analyze
    assert cached["hot_kernels_top15"] == []
    assert cached["analysis_md_path"] == ""


def test_recovery_tolerates_corrupt_report(tmp_path: Path) -> None:
    """A truncated/corrupt report degrades to empty rather than raising."""
    reports = tmp_path / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "kernel_roofline_current.json").write_text("{not json", encoding="utf-8")

    state = SharedState.load_or_init(tmp_path)
    state.record_trace_analyze(
        {"trace_input": "/trace", "roofline_output_name": "kernel_roofline_current.json"},
        {"status": "ok"},
    )

    assert state.last_trace_analyze["hot_kernels_top15"] == []


def test_recovery_falls_back_to_default_report_name(tmp_path: Path) -> None:
    """No ``roofline_output_name`` in the payload → the current-report name is tried."""
    _write_roofline_report(tmp_path, kernels=_attention_kernels())

    state = SharedState.load_or_init(tmp_path)
    state.record_trace_analyze({"trace_input": "/trace"}, {"status": "ok"})

    assert [k["name"] for k in state.last_trace_analyze["hot_kernels_top15"]] == [
        "aten::_flash_attention_forward",
        "aten::cat",
    ]
