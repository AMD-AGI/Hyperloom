###############################################################################
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# See LICENSE for license information.
###############################################################################

"""Contract tests for the ``analysis.json`` reader and the resolver field map.

The reader (``_analysis_json.load_report_tasks``) is the compute path's Stage A
plus the source-finding half of Stage B: it expands TraceLens' typed
``compute_optimizations[]`` into candidate rows and stamps each one with the
verdict from TraceLens' ``resolve_kernel_source``. These tests pin that seam --
the six ResolveResult fields onto the candidate keys, the new #1057
non-patchable-with-source behavior, and the fail-closed miss -- with the resolver
stubbed so no live source tree is needed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import _analysis_json as aj
import _kernel_source as ks
import tracelens_analysis as tla
from _kernel_partition import partition_kernels
from TraceLens.TraceUtils.kernel_source import ResolveResult, SourceLocation


def _write_report(tmp_path: Path, members: list[dict], operation: str = "aiter::gemm") -> Path:
    report = {
        "compute_optimizations": [
            {
                "priority": 1,
                "operation": operation,
                "identification": "id",
                "reasoning": "why",
                "resolution": "fix",
                "prose_truncated": False,
                "impact": {"mid": 3.5, "low": 1.0, "high": 5.0},
                "members": members,
            }
        ]
    }
    path = tmp_path / "analysis.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


def _member(**overrides) -> dict:
    member = {
        "impact_score": 3.5,
        "kernel_launcher_path": "",
        "library": "AITER",
        "category": "gemm",
        "analysis_md_rank": "P1",
        "kernel_name": ["some_kernel"],
        "args_shapes": ["(64,20,128)"],
        "args_datatypes": ["bf16"],
        "time_ms": 2.0,
        "count": 4,
        "pct_e2e": 12.5,
        "flops_per_byte": 3.3,
        "efficiency_percent": 33.0,
        "efficiency_peak_value": 708.0,
        "efficiency_peak_unit": "TFLOPS",
        "bound": "compute",
    }
    member.update(overrides)
    return member


def test_stage_a_lifts_metrics_and_maps_gpu_pct(tmp_path, monkeypatch):
    """Task scalars + member metrics land on the row; gpu_pct := pct_e2e, no recompute."""
    monkeypatch.setattr(
        ks,
        "resolve_kernel_source",
        lambda *a, **k: ResolveResult(location=None, patchable=False, method="unresolved"),
    )
    rows = aj.load_report_tasks(_write_report(tmp_path, [_member()]))
    assert len(rows) == 1
    row = rows[0]
    assert row["name"] == "aiter::gemm"
    assert row["gpu_pct"] == 12.5
    # percent_of_total is dropped: it was only ever an alias of gpu_pct.
    assert "percent_of_total" not in row
    assert row["duration_us"] == 2000.0
    assert row["call_count"] == 4
    # args_shapes + args_datatypes fold back into the inline "(dims) dtype" form
    # the shape consumers read the dtype out of; input_dtypes is derived downstream.
    assert row["shapes"] == ["(64,20,128) bf16"]
    assert "input_dtypes" not in row
    assert row["device_kernel_name"] == "some_kernel"
    assert row["library"] == "AITER"
    assert row["tracelens_category"] == "gemm"
    assert row["efficiency_peak_unit"] == "TFLOPS"
    assert row["impact_score"] == 3.5


def test_resolve_result_fields_map_onto_the_candidate(tmp_path, monkeypatch):
    """The six ResolveResult fields map to the six HL candidate keys (patchable case)."""
    monkeypatch.setattr(
        ks,
        "resolve_kernel_source",
        lambda *a, **k: ResolveResult(
            location=SourceLocation(source_file="/repo/moe.py", line=42, framework="aiter"),
            patchable=True,
            kind="",
            reason="",
            method="symbol_index",
        ),
    )
    row = aj.load_report_tasks(_write_report(tmp_path, [_member()]))[0]
    assert row["source_file"] == "/repo/moe.py"
    assert row["source_line"] == 42
    assert row["op_to_source_patchable"] is True
    assert row["kernel_kind"] == ""
    assert row["source_resolution_method"] == "symbol_index"


def test_non_patchable_kernel_keeps_its_source(tmp_path, monkeypatch):
    """#1057: a vendor GEMM comes back non-patchable WITH a dispatcher source, not blank."""
    monkeypatch.setattr(
        ks,
        "resolve_kernel_source",
        lambda *a, **k: ResolveResult(
            location=SourceLocation(source_file="/repo/dispatcher.cu", line=None),
            patchable=False,
            kind="aiter_ck",
            reason="Composable Kernel template instantiation",
            method="gate_non_patchable",
        ),
    )
    row = aj.load_report_tasks(_write_report(tmp_path, [_member(kernel_name=["Cijk_ck"])]))[0]
    assert row["source_file"] == "/repo/dispatcher.cu"
    assert row["op_to_source_patchable"] is False
    assert row["kernel_kind"] == "aiter_ck"
    assert row["source_resolution_method"] == "gate_non_patchable"
    assert row["op_to_source_reason"] == "Composable Kernel template instantiation"


def test_fail_closed_symbol_has_no_source(tmp_path, monkeypatch):
    """An unresolvable symbol: location=None, method=unresolved, blank source_file."""
    monkeypatch.setattr(
        ks,
        "resolve_kernel_source",
        lambda *a, **k: ResolveResult(location=None, patchable=False, method="unresolved", reason="no live match"),
    )
    row = aj.load_report_tasks(_write_report(tmp_path, [_member()]))[0]
    assert row["source_file"] == ""
    assert row["source_line"] is None
    assert row["source_resolution_method"] == "unresolved"
    assert row["op_to_source_patchable"] is False


def test_triton_symbol_routes_with_kernel_file(tmp_path, monkeypatch):
    """A Triton SYMBOL takes the Triton route with its launcher; op_name rides along."""
    seen: dict = {}

    def _spy(kernel_name="", *, kernel_file="", is_triton=False, op_name="", search_paths=None):
        seen.update(kernel_file=kernel_file, is_triton=is_triton, op_name=op_name)
        return ResolveResult(location=None, patchable=False, method="unresolved")

    monkeypatch.setattr(ks, "resolve_kernel_source", _spy)
    member = _member(kernel_name=["triton_poi_fused_add"], kernel_launcher_path="moe.py(10): fwd")
    aj.load_report_tasks(_write_report(tmp_path, [member]))
    assert seen["is_triton"] is True
    assert seen["kernel_file"] == "moe.py(10): fwd"
    assert seen["op_name"] == "aiter::gemm"


def test_triton_launcher_path_routes_even_when_symbol_is_clean(tmp_path, monkeypatch):
    """An aiter Gluon Triton kernel names triton only in its ``.../triton/...`` launcher."""
    seen: dict = {}

    def _spy(kernel_name="", *, kernel_file="", is_triton=False, op_name="", search_paths=None):
        seen.update(kernel_file=kernel_file, is_triton=is_triton)
        return ResolveResult(location=None, patchable=False, method="unresolved")

    monkeypatch.setattr(ks, "resolve_kernel_source", _spy)
    member = _member(
        kernel_name=["paged_attention_decode_sliding_window"],
        library="AITER",
        kernel_launcher_path="aiter/ops/triton/gluon/pa_decode_gluon.py(5194): pa_decode_gluon",
    )
    aj.load_report_tasks(_write_report(tmp_path, [member]))
    assert seen["is_triton"] is True
    assert seen["kernel_file"] == "aiter/ops/triton/gluon/pa_decode_gluon.py(5194): pa_decode_gluon"


def test_native_symbol_with_py_launcher_stays_native(tmp_path, monkeypatch):
    """A native kernel dispatched through a .py must NOT take the Triton route.

    Forcing Triton on a Tensile/CK symbol would bypass TraceLens' native gate and
    mislabel a precompiled kernel as patchable.
    """
    seen: dict = {}

    def _spy(kernel_name="", *, kernel_file="", is_triton=False, op_name="", search_paths=None):
        seen.update(kernel_file=kernel_file, is_triton=is_triton)
        return ResolveResult(location=None, patchable=False, method="unresolved")

    monkeypatch.setattr(ks, "resolve_kernel_source", _spy)
    member = _member(kernel_name=["Cijk_Ailk_Bljk"], library="aiter", kernel_launcher_path="tuned_gemm.py(9): g")
    aj.load_report_tasks(_write_report(tmp_path, [member]))
    assert seen["is_triton"] is False
    assert seen["kernel_file"] == ""


def test_non_patchable_with_source_partitions_to_skipped_with_a_source(tmp_path, monkeypatch):
    """The #1057 behavior end to end: a resolved-but-non-patchable kernel is skipped, not blank."""
    monkeypatch.setattr(
        ks,
        "resolve_kernel_source",
        lambda *a, **k: ResolveResult(
            location=SourceLocation(source_file="/repo/dispatcher.cu", line=None),
            patchable=False,
            kind="aiter_ck",
            reason="Composable Kernel template instantiation",
            method="gate_non_patchable",
        ),
    )
    rows = aj.load_report_tasks(_write_report(tmp_path, [_member(kernel_name=["Cijk_ck"])]))
    for row in rows:
        reusable, skip_reason = tla.classify_patchability(row)
        row["reusable_native_kernel"] = reusable
        row["skip_reason"] = skip_reason
    routable, skipped = partition_kernels(rows, lambda c: c.get("reusable_native_kernel") is True)
    assert routable == []
    assert len(skipped) == 1
    # It carries its dispatcher source for display/handoff rather than landing blank.
    assert skipped[0]["source_file"] == "/repo/dispatcher.cu"
    assert "Composable Kernel" in skipped[0]["skip_reason"]


def test_fail_closed_symbol_partitions_to_skipped_without_a_source(tmp_path, monkeypatch):
    """A genuine miss is skipped with a blank source_file, never corrupting the routable set."""
    monkeypatch.setattr(
        ks,
        "resolve_kernel_source",
        lambda *a, **k: ResolveResult(location=None, patchable=False, method="unresolved", reason="no live match"),
    )
    rows = aj.load_report_tasks(_write_report(tmp_path, [_member()]))
    for row in rows:
        reusable, skip_reason = tla.classify_patchability(row)
        row["reusable_native_kernel"] = reusable
        row["skip_reason"] = skip_reason
    routable, skipped = partition_kernels(rows, lambda c: c.get("reusable_native_kernel") is True)
    assert routable == []
    assert len(skipped) == 1
    assert skipped[0]["source_file"] == ""


def test_missing_or_empty_file_returns_no_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(
        ks,
        "resolve_kernel_source",
        lambda *a, **k: ResolveResult(location=None, patchable=False, method="unresolved"),
    )
    assert aj.load_report_tasks(tmp_path / "nope.json") == []
    empty = tmp_path / "empty.json"
    empty.write_text("{}", encoding="utf-8")
    assert aj.load_report_tasks(empty) == []
