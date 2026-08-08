# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""End-to-end: an unresolvable source file and a multi-GPU collective, together.

Reproduces the failure that left ``runs/kernel_opt`` empty across 19 MiniMax-M3
sessions, and the collective lane that shares its root cause.

TraceLens cannot attribute a kernel that never passes the ATen dispatcher: a
hand-written Triton kernel or an aiter collective is filed as a ``(Synthetic Op)``
whose Kernel Path column is a placeholder -- ``Not found`` for Triton,
``AITER (vendor)`` for aiter. Both placeholders used to survive as a truthy
``source_file``, which skipped the grep fallback and made
``classify_patchability`` report the nonsensical "source not under a reusable
framework root: Not found". Every hot kernel was then dropped before any backend
saw it, and the collective lane had nothing to select.

The whole chain is exercised here, from the report through candidate
finalization to the orchestrator's lane gate, because each tier on its own can
pass while the seam between them is broken.
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import pytest

_TOOLS = Path(__file__).resolve().parents[2] / "agents" / "kernel" / "tools"
sys.path.insert(0, str(_TOOLS))

import tracelens_analysis as tl  # noqa: E402
import tracelens_skill_runner as tsr  # noqa: E402

from hyperloom.orchestrator.kernel.request_handlers import (  # noqa: E402
    select_collective_candidate,
)
from hyperloom.orchestrator.phases.kernel import KernelPhase  # noqa: E402


# Paths must sit under a reusable framework root, else patchability rejects them
# for a reason unrelated to what this test covers.
_TRITON_LAUNCHER = "/sgl-workspace/sglang/python/sglang/kernels/ops/moe/fused_moe_e2e.py(124): _grouped_gemm"
_COLLECTIVE_LAUNCHER = "/sgl-workspace/aiter/aiter/dist/device_communicators/custom_all_reduce.py(811): fused_ar_rms"
_TRITON_DEFINITION = "/sgl-workspace/sglang/python/sglang/kernels/ops/moe/mxfp8_moe_amd_gfx95.py"
_COLLECTIVE_DEFINITION = "/sgl-workspace/aiter/csrc/include/custom_all_reduce.cuh"

_TRITON_KERNEL = "_mxfp8_grouped_gemm_kernel"
# The trace carries the full mangled symbol; TraceLens elides it in the report,
# so the fixture uses each form where the real pipeline would.
_COLLECTIVE_KERNEL = "_ZN5aiter33reduce_scatter_cross_device_storeIDF16bLi8EEEvPNS_8RankDataENS_11RankSignalsEiiii"
_COLLECTIVE_KERNEL_TRUNCATED = "_ZN5aiter33reduce_scatter_cross_device_storeIDF16bLi8EEEvPNS_8RankDataENS_1..."

_TID = 7


def _frame(name: str, ts: float, dur: float) -> dict:
    return {"cat": "python_function", "name": name, "tid": _TID, "ts": ts, "dur": dur}


def _launch(corr: int, api: str, ts: float) -> dict:
    return {"cat": "cuda_runtime", "name": api, "tid": _TID, "ts": ts, "dur": 2.0,
            "args": {"correlation": corr}}


def _kernel(name: str, corr: int, ts: float) -> dict:
    return {"cat": "kernel", "name": name, "ts": ts, "dur": 5.0, "args": {"correlation": corr}}


@pytest.fixture()
def trace_file(tmp_path: Path) -> Path:
    """A trace carrying both launch stacks, gzipped like a real capture.

    The Triton stack ends in triton runtime plumbing and the collective stack in
    aiter's JIT dispatch wrapper; both must be walked past to reach the user
    frame that names the source.
    """
    events: list[dict] = []

    # Triton kernel: user frame -> triton runtime -> builtin launch.
    events += [
        _frame("/sgl-workspace/sglang/python/sglang/srt/models/m.py(10): forward", 100.0, 100.0),
        _frame(_TRITON_LAUNCHER, 101.0, 98.0),
        _frame("/venv/triton/runtime/jit.py(708): run", 102.0, 96.0),
        _frame("<built-in function launch>", 103.0, 94.0),
        _launch(1, "hipModuleLaunchKernel", 150.0),
        _kernel(_TRITON_KERNEL, 1, 200.0),
    ]
    # aiter collective: user frame -> JIT dispatch wrapper -> builtin launch.
    events += [
        _frame(_COLLECTIVE_LAUNCHER, 300.0, 100.0),
        _frame("/venv/aiter/jit/core.py(1504): wrapper", 301.0, 98.0),
        _frame("<built-in function launch>", 302.0, 96.0),
        _launch(2, "hipLaunchKernel", 350.0),
        _kernel(_COLLECTIVE_KERNEL, 2, 400.0),
    ]

    path = tmp_path / "e2e.trace.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write(json.dumps({"traceEvents": events}))
    return path


def _row(operation: str, kernel_path: str, kernel_name: str, time_ms: str, pct: str) -> str:
    return (
        f"| {operation} |  | {kernel_path} | {kernel_name} | {time_ms} | {pct} "
        f"| 342 | — | — | — | miscellaneous |"
    )


@pytest.fixture()
def analysis_md(tmp_path: Path) -> Path:
    """A report whose Kernel Path cells carry the two upstream placeholders."""
    header = (
        "| Operation |  Args  | Kernel Path | Kernel Name | Time (ms) | %E2E | Count "
        "|FLOPS/Byte| Efficiency | Bound | Sub-Category |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|"
    )
    blocks = [
        # Triton kernel, placeholder "Not found".
        ("MXFP8 grouped GEMM dominates uncategorized time",
         _row(f"hipModuleLaunchKernel->{_TRITON_KERNEL} (Synthetic Op)", "Not found", _TRITON_KERNEL, "228.74", "20.75")),
        # aiter collective, placeholder "AITER (vendor)" and an elided symbol.
        ("Exposed tensor-parallel collective",
         _row(f"hipLaunchKernel->{_COLLECTIVE_KERNEL_TRUNCATED} (Synthetic Op)",
              "AITER (vendor)", _COLLECTIVE_KERNEL_TRUNCATED, "88.10", "8.00")),
        # A bare runtime API: not a kernel, must never be routed anywhere.
        ("Graph launch aggregate",
         _row("hipGraphLaunch", "", "hipGraphLaunch", "60.00", "5.44")),
    ]

    lines = ["# TraceLens Analysis", ""]
    for index, (title, _row_text) in enumerate(blocks, 1):
        lines += ["<!-- impact-begin kind=p_item category=other low=1.0 mid=2.0 high=3.0 -->",
                  f"**Impact**: P{index} {title}", "<!-- impact-end -->", ""]
    lines += ["", "## Detailed Analysis", ""]
    for index, (title, row_text) in enumerate(blocks, 1):
        # The compute-tier marker is what makes a block a candidate block; the
        # parser skips any heading not preceded by one.
        lines += [f'<a id="detailed-analysis-compute-p{index}"></a>',
                  f"<!-- reasoning-candidate tier=compute rank={index} -->",
                  f"#### 🔴 P{index}: {title}", "", "**Data:**", "", header, row_text, "", "---", ""]

    path = tmp_path / "analysis.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _finalize(analysis_md: Path, trace_file: Path) -> list[dict]:
    """Both stages the real CLI runs, in order.

    ``_finalize_candidates`` resolves sources and patchability;
    ``enrich_candidates_with_runtime_metadata`` then attaches the kernel contract
    that types a candidate as a collective. Running only the first leaves the
    collective lane with nothing to select, so the seam belongs in this test.
    """
    import argparse

    rows = tsr.parse_analysis_md(analysis_md, top_k=20)
    assert rows, "fixture analysis.md did not parse; the report contract changed"
    candidates = tl._finalize_candidates(rows, framework="sglang", trace_files=[trace_file])
    tl.enrich_candidates_with_runtime_metadata(
        candidates,
        argparse.Namespace(
            framework="sglang",
            model_name="MiniMax-M3-MXFP8",
            target_platform="MI355X",
            analysis_mode="inference",
            runtime_env="local",
        ),
    )
    return candidates


def _by_kernel(candidates: list[dict], kernel: str) -> dict:
    for item in candidates:
        if kernel in str(item.get("device_kernel_name") or "") or kernel in str(item.get("name") or ""):
            return item
    raise AssertionError(f"{kernel} missing from finalized candidates")


# --- Scenario 1: the source file TraceLens could not resolve ------------------


def test_triton_kernel_resolves_despite_the_not_found_placeholder(analysis_md, trace_file):
    item = _by_kernel(_finalize(analysis_md, trace_file), _TRITON_KERNEL)

    # The placeholder must not survive as a path...
    assert item["source_file"] != "Not found"
    # ...and grep must identify the definition rather than trusting the launcher.
    assert item["source_file"] == _TRITON_DEFINITION
    assert item.get("source_line") is None
    assert item.get("source_function") in (None, "")
    assert item["source_resolution_method"] == "name_grep"
    assert item["trace_launcher_file"] == _TRITON_LAUNCHER.split("(", 1)[0]
    assert "trace launcher differs from grep source" in item["source_resolution_reason"]


def test_resolved_triton_kernel_reaches_a_backend(analysis_md, trace_file):
    """The regression that mattered: hot kernels were dropped before dispatch."""
    item = _by_kernel(_finalize(analysis_md, trace_file), _TRITON_KERNEL)
    assert item["reusable_native_kernel"] is True, item.get("skip_reason")
    assert "forge" in item["recommended_backends"]


# --- Scenario 2: the multi-GPU collective -------------------------------------


def test_collective_resolves_despite_the_vendor_placeholder(analysis_md, trace_file):
    item = _by_kernel(_finalize(analysis_md, trace_file), "reduce_scatter_cross_device_store")

    assert item["source_file"] != "AITER (vendor)"
    assert item["source_file"] == _COLLECTIVE_DEFINITION
    assert item.get("source_line") is None
    assert item["source_resolution_method"] == "name_grep"
    assert item["trace_launcher_file"] == _COLLECTIVE_LAUNCHER.split("(", 1)[0]
    assert "trace launcher differs from grep source" in item["source_resolution_reason"]


def _state_as_orchestrator_sees_it(candidates: list[dict], tmp_path: Path):
    """Reproduce what shared state actually holds after a trace analysis.

    ``record_trace_analyze`` stores a roofline-oriented projection in
    ``hot_kernels_top15`` -- no ``kernel_contract``, no ``kernel_repo``, no
    shapes -- and keeps the enriched rows on disk at ``candidates_path``. A test
    that feeds the full rows in directly would hide a lane that reads the
    projection.
    """
    path = tmp_path / "kernel_candidates.json"
    path.write_text(json.dumps({"hot_kernels": candidates}, default=str), encoding="utf-8")
    projection = [
        {
            "kernel_id": c.get("kernel_id"),
            "name": c.get("name"),
            "gpu_pct": c.get("gpu_pct"),
            "source_file": c.get("source_file"),
            "reusable_native_kernel": c.get("reusable_native_kernel"),
            "recommended_backends": c.get("recommended_backends") or [],
        }
        for c in candidates
    ]
    assert all("kernel_contract" not in row for row in projection)
    return type(
        "S", (), {"last_trace_analyze": {"hot_kernels_top15": projection, "candidates_path": str(path)}}
    )()


def test_collective_is_typed_and_selectable_by_the_lane(analysis_md, trace_file, tmp_path):
    candidates = _finalize(analysis_md, trace_file)
    item = _by_kernel(candidates, "reduce_scatter_cross_device_store")

    contract = item.get("kernel_contract") or {}
    assert contract.get("kind") == "collective"
    assert contract.get("collective_op") == "reduce_scatter"
    assert item["reusable_native_kernel"] is True, item.get("skip_reason")

    # The lane must reach the enriched rows, not the projection.
    picked = select_collective_candidate(_state_as_orchestrator_sees_it(candidates, tmp_path))
    assert picked is not None
    assert "reduce_scatter_cross_device_store" in str(picked.get("name"))


def test_collective_lane_gate_opens_on_this_analysis(analysis_md, trace_file):
    """TP>1 plus exposed communication is what arms the lane."""
    from types import SimpleNamespace

    fake = SimpleNamespace(
        shared_state=SimpleNamespace(
            tp=8,
            last_collective={},
            current_comm_pct=lambda: 8.0,
            # Selection reads the analysis, so the gate requires one first.
            last_trace_analyze={"hot_kernels_top15": []},
        ),
        COLLECTIVE_COMM_PCT_FLOOR=KernelPhase.COLLECTIVE_COMM_PCT_FLOOR,
    )
    assert KernelPhase._collective_required_before_kernel_opt(fake) is True


# --- Guard: the two scenarios must not resurrect the false positive -----------


def test_bare_runtime_api_is_never_routed(analysis_md, trace_file):
    """hipGraphLaunch is a launch API, not a kernel.

    Name-grepping it used to match aiter's hipify mapping table, which was then
    handed to a backend as if it were rewritable kernel source.
    """
    item = _by_kernel(_finalize(analysis_md, trace_file), "hipGraphLaunch")
    assert item["reusable_native_kernel"] is False
    assert not item["recommended_backends"]
    assert "cuda_to_hip_mappings" not in str(item.get("source_file") or "")


def test_no_candidate_keeps_a_placeholder_as_its_source(analysis_md, trace_file):
    """Whatever the outcome, no sentinel may be left in a path-typed field."""
    for item in _finalize(analysis_md, trace_file):
        source = str(item.get("source_file") or "")
        if source:
            assert tl.looks_like_source_path(source), f"{item.get('name')} -> {source!r}"
