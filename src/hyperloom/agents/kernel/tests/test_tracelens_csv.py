# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Regression tests for tracelens_analysis candidate extraction and routing."""

from __future__ import annotations

import gzip
import json
import os
import sys
from pathlib import Path

import pytest

# tools/ is not a package — stick its dir on sys.path so we can import.
_TOOL_DIR = Path(__file__).resolve().parent.parent / "tools"
if str(_TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOL_DIR))

import tracelens_analysis as tla  # noqa: E402
import _idle_gate as idle_gate  # noqa: E402
import _task_group_contract as task_group_contract  # noqa: E402
import tracelens_skill_runner as tlr  # noqa: E402


def test_default_top_k_uses_large_pool_by_default(monkeypatch):
    """The candidate-build pool defaults to a large value, not 10."""
    monkeypatch.delenv("HYPERLOOM_KERNEL_CANDIDATES_TOP_K", raising=False)
    assert tla._default_top_k() == tla._DEFAULT_KERNEL_CANDIDATES_TOP_K
    assert tla._default_top_k() > 10


def test_default_top_k_env_override(monkeypatch):
    """HYPERLOOM_KERNEL_CANDIDATES_TOP_K overrides the pool size."""
    monkeypatch.setenv("HYPERLOOM_KERNEL_CANDIDATES_TOP_K", "25")
    assert tla._default_top_k() == 25


def test_default_top_k_zero_means_unbounded(monkeypatch):
    """Zero or negative disables the build-time cap (huge internal cap)."""
    monkeypatch.setenv("HYPERLOOM_KERNEL_CANDIDATES_TOP_K", "0")
    assert tla._default_top_k() >= 1_000_000
    monkeypatch.setenv("HYPERLOOM_KERNEL_CANDIDATES_TOP_K", "-5")
    assert tla._default_top_k() >= 1_000_000


def test_default_top_k_invalid_falls_back(monkeypatch):
    """A non-integer env value falls back to the default pool."""
    monkeypatch.setenv("HYPERLOOM_KERNEL_CANDIDATES_TOP_K", "not-an-int")
    assert tla._default_top_k() == tla._DEFAULT_KERNEL_CANDIDATES_TOP_K


def test_agent_dry_run_initializes_route_and_writes_resolution_artifact(
    monkeypatch,
    tmp_path,
    capsys,
):
    """Dry-run must not read a route variable initialized only in live mode."""
    import json as _json

    trace = tmp_path / "trace.json"
    trace.write_text('{"traceEvents": []}', encoding="utf-8")
    source = tmp_path / "kernel.py"
    source.write_text("def kernel():\n    pass\n", encoding="utf-8")
    report = tmp_path / "trace_report.json"
    monkeypatch.setattr(
        tla,
        "analyze_trace_files",
        lambda *_args, **_kwargs: [
            {
                "kernel_id": "k001",
                "name": "kernel",
                "gpu_pct": 100.0,
                "duration_us": 1.0,
                "source_file": str(source),
                "source_type": "python",
                "source_resolution_method": "name_grep",
            }
        ],
    )
    monkeypatch.setattr(
        tla,
        "write_reports",
        lambda *_args, **_kwargs: {"trace_report_path": str(report)},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tracelens_analysis.py",
            "--trace-input",
            str(trace),
            "--workspace-path",
            str(tmp_path / "workspace"),
            "--dry-run",
        ],
    )

    assert tla.main() == 0
    result = _json.loads(capsys.readouterr().out)
    resolution_path = Path(result["artifact_paths"]["kernel_source_resolution"])
    assert resolution_path.is_file()
    assert _json.loads(resolution_path.read_text(encoding="utf-8"))["entries"][0]["source_file"] == str(source)


def test_agent_dry_run_does_not_spend_a_candidate_review_session(monkeypatch, tmp_path, capsys):
    """A dry run plans the analysis; it must not run the review agent.

    The stage costs an agent session, waits out a 900-second bound when the
    stream stalls, and reads the framework tree -- all to audit a table a dry
    run publishes for inspection and never dispatches from.
    """
    import json as _json

    trace = tmp_path / "trace.json"
    trace.write_text('{"traceEvents": []}', encoding="utf-8")
    source = tmp_path / "kernel.py"
    source.write_text("def kernel():\n    pass\n", encoding="utf-8")
    monkeypatch.setattr(
        tla,
        "analyze_trace_files",
        lambda *_args, **_kwargs: [
            {
                "kernel_id": "k001",
                "name": "kernel",
                "gpu_pct": 100.0,
                "duration_us": 1.0,
                "source_file": str(source),
                "source_type": "python",
                "source_resolution_method": "name_grep",
            }
        ],
    )
    monkeypatch.setattr(
        tla,
        "write_reports",
        lambda *_a, **_kw: {"trace_report_path": str(tmp_path / "trace_report.json")},
    )

    ran: list[str] = []
    monkeypatch.setattr(
        tla,
        "run_candidate_review_stage",
        lambda *_a, **_kw: ran.append("called") or {},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tracelens_analysis.py",
            "--trace-input",
            str(trace),
            "--workspace-path",
            str(tmp_path / "workspace"),
            "--dry-run",
        ],
    )

    assert tla.main() == 0
    assert ran == []
    result = _json.loads(capsys.readouterr().out)
    assert "kernel_candidates_raw" not in result["artifact_paths"]


# A path — is_kernel_event strict cat == 'kernel'
def test_a_filters_python_function_synchronize():
    """The exact event that ranked first in the buggy resume trace."""
    sync_event = {
        "name": "torch/cuda/streams.py(222): synchronize",
        "cat": "python_function",
        "dur": 88673.57,
    }
    assert tla.is_kernel_event(sync_event) is False


def test_a_filters_cuda_runtime_hipdevicesynchronize():
    """``hipDeviceSynchronize`` is a HIP runtime API call, not a GPU kernel."""
    runtime_event = {
        "name": "hipDeviceSynchronize",
        "cat": "cuda_runtime",
        "dur": 92.225,
    }
    assert tla.is_kernel_event(runtime_event) is False


def test_a_filters_cpu_op():
    cpu_op = {"name": "aten::matmul", "cat": "cpu_op", "dur": 50.0}
    assert tla.is_kernel_event(cpu_op) is False


def test_a_accepts_real_kernel_event():
    real = {
        "name": ("_ZN5aiter24add_rmsnorm_quant_kernelIDF16bDF16bLi256ELi16ELb1ELb0ELb1ELi1EEEvPT0_"),
        "cat": "kernel",
        "dur": 6.48,
    }
    assert tla.is_kernel_event(real) is True


def test_a_accepts_kernel_with_runtime_lookalike_name_but_kernel_cat():
    # Defensive: pathological name + correct cat → still a kernel
    weird = {"name": "void synchronize_kernel<...>", "cat": "kernel", "dur": 1.0}
    assert tla.is_kernel_event(weird) is True


def test_a_rejects_kernel_cat_when_name_is_runtime_api():
    # Belt-and-braces: even with cat=kernel, names listed in
    # RUNTIME_API_NAMES (caught by mis-tagged traces) are rejected.
    weird = {"name": "hipDeviceSynchronize", "cat": "kernel", "dur": 1.0}
    assert tla.is_kernel_event(weird) is False


def test_a_top_kernels_no_sync_events_in_real_trace_shape():
    """Build a synthetic trace mirroring the resume4 shape and confirm
    is_kernel_event rejects the sync events before they can reach top-K."""
    events = [
        # 5 host-side sync events, big durations (the buggy ones)
        {"name": "torch/cuda/streams.py(222): synchronize", "cat": "python_function", "dur": 88673.0},
        {"name": "torch/cuda/__init__.py(1073): synchronize", "cat": "python_function", "dur": 10414.0},
        {"name": "<built-in function _cuda_synchronize>", "cat": "python_function", "dur": 10368.0},
        {"name": "hipDeviceSynchronize", "cat": "cuda_runtime", "dur": 10364.0},
        {"name": "<built-in method synchronize of Event object at 0x123>", "cat": "python_function", "dur": 88671.0},
        # 3 real kernels, smaller duration
        {"name": "aiter::add_rmsnorm_quant_kernel<...>", "cat": "kernel", "dur": 466.0},
        {"name": "sgl_hip::activation::act_and_mul_kernel<...>", "cat": "kernel", "dur": 380.0},
        {"name": "void paged_attention_ll4mi_QKV_mfma16_kernel<...>", "cat": "kernel", "dur": 889.0},
    ]
    # Direct call paths used by analyze_trace_files
    kept = [e for e in events if tla.is_kernel_event(e)]
    assert len(kept) == 3
    kept_names = {e["name"] for e in kept}
    assert "aiter::add_rmsnorm_quant_kernel<...>" in kept_names
    assert "sgl_hip::activation::act_and_mul_kernel<...>" in kept_names
    assert "void paged_attention_ll4mi_QKV_mfma16_kernel<...>" in kept_names
    # No sync poison left
    for n in kept_names:
        assert "synchronize" not in n.lower()


# The torch.profiler Chrome-trace category for a GPU kernel is literally
# "kernel". Pin the torch convention so a rename cannot break GPU-kernel detection.
def test_issue_769_kernel_event_uses_torch_cat_kernel():
    """A real GPU kernel uses cat=='kernel'; the renamed 'kernel_agent' is not a trace category."""
    real_kernel = {"name": "void some_gemm_kernel<...>", "cat": "kernel", "dur": 5.0}
    assert tla.is_kernel_event(real_kernel) is True
    # The component-name string 'kernel_agent' must never be treated as a GPU
    # kernel trace category.
    not_a_kernel = {"name": "void some_gemm_kernel<...>", "cat": "kernel_agent", "dur": 5.0}
    assert tla.is_kernel_event(not_a_kernel) is False


def test_issue_769_count_gpu_kernel_events_nonzero_on_healthy_trace(tmp_path):
    """count_gpu_kernel_events must count cat=='kernel' events in a real torch trace."""
    trace = {
        "traceEvents": [
            {"name": "python_function frame", "cat": "python_function", "dur": 100.0},
            {"name": "aten::matmul", "cat": "cpu_op", "dur": 50.0},
            {"name": "hipDeviceSynchronize", "cat": "cuda_runtime", "dur": 92.0},
            {"name": "void gemm_kernel<...>", "cat": "kernel", "dur": 6.0},
            {"name": "void attn_kernel<...>", "cat": "kernel", "dur": 11.0},
            {"name": "void rmsnorm_kernel<...>", "cat": "kernel", "dur": 3.0},
        ]
    }
    trace_path = tmp_path / "healthy.trace.json.gz"
    with gzip.open(trace_path, "wt", encoding="utf-8") as fh:
        json.dump(trace, fh)
    assert tla.count_gpu_kernel_events(trace_path) == 3


# Native-only kernel-opt targeting
def test_compile_generated_kernel_is_not_reusable_native():
    candidate = {
        "name": "triton_poi_fused_add_mul_0",
        "source_file": "/tmp/torchinductor_root/ab/cdef.py",
        "source_type": tla.source_type_for(
            "triton_poi_fused_add_mul_0",
            "/tmp/torchinductor_root/ab/cdef.py",
        ),
    }
    assert candidate["source_type"] == "runtime_generated"
    assert (
        tla.is_runtime_generated_kernel(
            candidate["name"],
            candidate["source_file"],
        )
        is True
    )
    assert tla.classify_patchability(candidate)[0] is False
    assert tla.recommend_backends(candidate) == []
    assert "not reusable" in tla.build_notes(candidate)


def test_stable_framework_triton_source_is_reusable_native(monkeypatch):
    candidate = {
        "name": "triton_attention_decode_kernel",
        "source_file": "/sgl-workspace/sglang/python/sglang/srt/layers/attention/triton_ops.py",
        "source_type": tla.source_type_for(
            "triton_attention_decode_kernel",
            "/sgl-workspace/sglang/python/sglang/srt/layers/attention/triton_ops.py",
        ),
    }
    assert candidate["source_type"] == "triton"
    assert (
        tla.is_runtime_generated_kernel(
            candidate["name"],
            candidate["source_file"],
        )
        is False
    )
    assert tla.classify_patchability(candidate)[0] is True
    assert tla.recommend_backends(candidate) == ["forge"]


def test_recommend_backends_for_python_source():
    """forge must be recommended for ``python`` source_type too."""
    candidate = {
        "name": "some_python_dispatcher",
        "source_file": "/sgl-workspace/sglang/python/sglang/srt/layers/dispatcher.py",
        "source_type": "python",
        "reusable_native_kernel": True,
    }
    assert tla.recommend_backends(candidate) == ["forge"]


def test_recommend_backends_for_unknown_source():
    """Unknown source_type: forge is still recommended (don't pre-filter by extension)."""
    candidate = {
        "name": "some_unrecognised_kernel",
        "source_file": "/some/path/kernel.xyz",
        "source_type": "unknown",
        "reusable_native_kernel": True,
    }
    assert tla.recommend_backends(candidate) == ["forge"]


def test_recommend_backends_is_forge_only():
    """Invariant: forge is the sole per-kernel backend in the ladder."""
    candidate = {
        "name": "some_kernel",
        "source_file": "/sgl-workspace/sglang/python/sglang/srt/layers/x.py",
        "source_type": "triton",
        "reusable_native_kernel": True,
    }
    ladder = tla.recommend_backends(candidate)
    assert ladder == ["forge"], f"forge must be the sole per-kernel backend, got {ladder}"


def test_unknown_source_root_is_not_reusable_native():
    candidate = {
        "name": "my_custom_kernel",
        "source_file": "/tmp/random/my_custom_kernel.cu",
        "source_type": tla.source_type_for(
            "my_custom_kernel",
            "/tmp/random/my_custom_kernel.cu",
        ),
    }
    assert candidate["source_type"] == "hip_cpp"
    assert tla.classify_patchability(candidate)[0] is False
    assert tla.recommend_backends(candidate) == []


def test_known_rmsnorm_harness_is_registered_without_repo_root(monkeypatch, tmp_path):
    """A curated harness is found from the kernel name alone, with no repo root.

    The hint is checkout-relative, so it is resolved against the search roots
    rather than a pinned ``/sgl-workspace`` path, and only a file that is really
    there is reported: a harness list naming paths nobody can open reads
    downstream as a runnable harness.
    """
    harness = tmp_path / "aiter" / "op_tests" / "test_rmsnorm2d.py"
    harness.parent.mkdir(parents=True)
    harness.write_text("def test_rmsnorm2d(): pass\n", encoding="utf-8")
    monkeypatch.setattr(tla, "kernel_search_roots", lambda: (str(tmp_path / "aiter"),))
    tla._harness_search_bases.cache_clear()

    files = tla.find_benchmark_files(
        "_ZN5aiter24add_rmsnorm_quant_kernelIDF16bDF16bLi256EEEv",
        "",
        "/sgl-workspace/aiter/csrc/kernels/rmsnorm_quant_kernels.cu",
    )
    tla._harness_search_bases.cache_clear()

    assert files == [str(harness)]


def test_absent_curated_harness_is_not_reported(monkeypatch, tmp_path):
    """A hint that resolves nowhere yields nothing, not an unopenable path."""
    monkeypatch.setattr(tla, "kernel_search_roots", lambda: (str(tmp_path / "aiter"),))
    tla._harness_search_bases.cache_clear()
    files = tla.find_benchmark_files("kernel_paged_attention_2d", "", "/pkg/attention.py")
    tla._harness_search_bases.cache_clear()
    assert files == []


def test_125_finalize_adds_kernel_category_for_attention():
    cand = {"name": "paged_attention_ll4mi_QKV_mfma16_kernel<bf16>"}
    assert tla.derive_kernel_category(cand) == "SDPA"


def test_125_derive_category_explicit_wins_over_heuristic():
    cand = {"name": "Cijk_Alik_GEMM_x", "tracelens_category": "MoE"}
    assert tla.derive_kernel_category(cand) == "MoE"


def test_125_derive_category_unknown_for_opaque_name():
    cand = {"name": "ZZZ_some_opaque_thunk_42"}
    assert tla.derive_kernel_category(cand) == "unknown"


def test_125_derive_category_normalizations():
    cases = [
        ("rocblas_sgemm_kernel", "GEMM"),
        ("flash_fmha_decode_kernel", "SDPA"),
        ("rmsnorm_kernel<bf16>", "LayerNorm"),
        ("act_and_mul_kernel<bf16>", "Activation"),
        ("moe_dispatch_kernel", "MoE"),
        ("softmax_kernel_v2", "Softmax"),
        ("all_reduce_xgmi_kernel", "Communication"),
        # PyTorch GEMM op-name variants the heuristic must also resolve (not just the CSV path).
        ("aten::mm", "GEMM"),
        ("aten::addmm", "GEMM"),
        ("aten::bmm", "GEMM"),
    ]
    for name, expected in cases:
        assert tla.derive_kernel_category({"name": name}) == expected, name


def test_125_finalize_outputs_source_path_field():
    """_finalize_candidates exposes the source_path mirror of source_file."""
    candidates = [
        {
            "name": "rmsnorm_kernel",
            "duration_us": 100.0,
            "call_count": 10,
            "source_file": "/path/to/rmsnorm.cu",
            "source_type": "hip_cpp",
            "shapes": [[16, 1024]],
        }
    ]
    out = tla._finalize_candidates(candidates, total_dur=100.0)
    assert out[0]["source_path"] == "/path/to/rmsnorm.cu"
    assert out[0]["kernel_category"] == "LayerNorm"


def test_finalize_uses_csv_op_category_for_aten_mm(tmp_path):
    """Layer-2 fix: aten::mm classifies as GEMM via the TraceLens CSV op-category lookup, not the name heuristic."""
    csv_dir = tmp_path / "perf_report_csvs"
    csv_dir.mkdir()
    (csv_dir / "unified_perf_summary.csv").write_text(
        "name,op category,extra\naten::mm,GEMM,ignored\n",
        encoding="utf-8",
    )
    candidates = [
        {
            "name": "aten::mm",
            "duration_us": 100.0,
            "call_count": 1,
            "source_file": "",
            "source_type": "unknown",
            "shapes": [[15360, 2048]],
        }
    ]
    out = tla._finalize_candidates(
        candidates,
        total_dur=100.0,
        perf_report_csv_dir=csv_dir,
    )
    assert out[0]["tracelens_category"] == "GEMM"
    assert out[0]["kernel_category"] == "GEMM"


def test_load_op_category_map_parses_unified_perf_summary(tmp_path):
    """load_op_category_map: {name -> raw TraceLens op category}, first non-empty per name."""
    csv_dir = tmp_path / "perf_report_csvs"
    csv_dir.mkdir()
    (csv_dir / "unified_perf_summary.csv").write_text(
        "name,op category,extra\n"
        "aten::mm,GEMM,row1\n"
        "aten::mm,GEMM,row2\n"
        "aiter::ck_moe_stage1,MoE_unfused,row3\n"
        "aten::copy_,elementwise,row4\n"
        "noise_op,,row5\n",
        encoding="utf-8",
    )
    m = tla.load_op_category_map(csv_dir)
    assert m == {
        "aten::mm": "GEMM",
        "aiter::ck_moe_stage1": "MoE_unfused",
        "aten::copy_": "elementwise",
    }


def test_load_op_category_map_missing_returns_empty(tmp_path):
    """csv absent / wrong path => {} so callers degrade to the name heuristic."""
    assert tla.load_op_category_map(tmp_path / "nonexistent") == {}
    (tmp_path / "perf_report_csvs").mkdir()
    assert tla.load_op_category_map(tmp_path / "perf_report_csvs") == {}


# ── #727 companion: fused-MoE trace-anchored shape capture ────────────────────

# The two operand-tuple rows TraceLens writes for the fused-MoE expert kernel in
# ``ops_unique_args.csv`` (gate/up GEMM then down GEMM), as captured for the
# Qwen3-30B-A3B MoE decode workload (conc 64, ISL/OSL 1024).
_FUSED_MOE_OPS_UNIQUE_ARGS = (
    "name,op category,Input Dims,Input type\n"
    "sglang_profiler::fused_moe_triton_kernels_invoke_fused_moe_kernel_427,MoE_fused,"
    '"((15360, 2048), (128, 1536, 2048), (), (122880, 1536), (), (), (), '
    '(15360, 8), (15360, 8), (131007,), (2047,), (1,), ())",'
    "\"('c10::BFloat16', 'c10::BFloat16', '', 'c10::BFloat16', '', '', '', "
    "'float', 'int', 'int', 'int', 'int', '')\"\n"
    "sglang_profiler::fused_moe_triton_kernels_invoke_fused_moe_kernel_427,MoE_fused,"
    '"((122880, 768), (128, 2048, 768), (), (15360, 8, 2048), (), (), (), '
    '(15360, 8), (15360, 8), (131007,), (2047,), (1,), ())",'
    "\"('c10::BFloat16', 'c10::BFloat16', '', 'c10::BFloat16', '', '', '', "
    "'float', 'int', 'int', 'int', 'int', '')\"\n"
)


def _write_fused_moe_ops_unique_args(tmp_path):
    csv_dir = tmp_path / "perf_report_csvs"
    csv_dir.mkdir()
    (csv_dir / "ops_unique_args.csv").write_text(_FUSED_MOE_OPS_UNIQUE_ARGS, encoding="utf-8")
    return csv_dir


def test_resolve_fused_moe_shapes_matches_manual_harness(tmp_path):
    """Operand dims recovered from ops_unique_args.csv match the manual GEAK harness shapes."""
    csv_dir = _write_fused_moe_ops_unique_args(tmp_path)
    shapes = tla.resolve_fused_moe_shapes_from_csv(csv_dir)
    # gate/up GEMM: A(num_tokens,H) x w1(E,2I,H) -> C(T,2I)
    assert "(15360,2048) bf16" in shapes  # A
    assert "(128,1536,2048) bf16" in shapes  # w1 (2*I=1536)
    assert "(122880,1536) bf16" in shapes  # C (T=num_tokens*topk)
    # down GEMM: A(T,I) x w2(E,H,I) -> C(num_tokens,topk,H)
    assert "(122880,768) bf16" in shapes  # A (I=768)
    assert "(128,2048,768) bf16" in shapes  # w2
    assert "(15360,8,2048) bf16" in shapes  # C
    # 1-tuples keep the trailing comma; ints map to i32, floats to f32.
    assert "(131007,) i32" in shapes
    assert "(15360,8) f32" in shapes
    # No empty/scalar operand leaks through.
    assert all(s and s != "()" for s in shapes)


def test_resolve_fused_moe_preserves_invocation_boundaries(tmp_path):
    csv_dir = _write_fused_moe_ops_unique_args(tmp_path)

    cases = tla.resolve_fused_moe_invocation_cases_from_csv(csv_dir)

    assert len(cases) == 2
    assert "(15360,2048) bf16" in cases[0]["input_shapes"][0]["shape"]
    assert "(122880,768) bf16" in cases[1]["input_shapes"][0]["shape"]
    assert cases[0]["raw_arg_spec"]["input_dims"] != cases[1]["raw_arg_spec"]["input_dims"]


def test_resolve_fused_moe_shapes_missing_csv_returns_empty(tmp_path):
    """Absent sidecar / no fused-MoE rows => [] so the candidate's empty shapes stay untouched."""
    assert tla.resolve_fused_moe_shapes_from_csv(tmp_path / "nope") == []
    csv_dir = tmp_path / "perf_report_csvs"
    csv_dir.mkdir()
    (csv_dir / "ops_unique_args.csv").write_text(
        'name,op category,Input Dims,Input type\naten::mm,GEMM,"((15360, 2048),)","(\'c10::BFloat16\',)"\n',
        encoding="utf-8",
    )
    assert tla.resolve_fused_moe_shapes_from_csv(csv_dir) == []


def test_finalize_grafts_fused_moe_shapes_onto_empty_candidate(tmp_path):
    """The empty-shaped fused-MoE candidate is back-filled with trace-anchored shapes + provenance."""
    csv_dir = _write_fused_moe_ops_unique_args(tmp_path)
    candidates = [
        {
            "name": "invoke_fused_moe_kernel",
            "duration_us": 302429.0,
            "call_count": 96,
            "source_file": ("/sgl-workspace/sglang/python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe.py"),
            "source_type": "python",
            "shapes": [],
            "tracelens_category": "moe_fused",
        }
    ]
    out = tla._finalize_candidates(
        candidates,
        total_dur=302429.0,
        perf_report_csv_dir=csv_dir,
    )
    assert out[0]["shapes"], "fused-MoE candidate must carry non-empty shapes"
    assert "(15360,2048) bf16" in out[0]["shapes"]
    assert out[0]["shape_provenance"] == "torch_trace"
    assert len(out[0]["invocation_cases"]) == 2
    assert out[0]["raw_arg_spec"] == out[0]["invocation_cases"][0]["raw_arg_spec"]


def test_finalize_grafts_csv_shapes_for_other_bucket_attention_candidate(tmp_path):
    """other_bucket fallback candidates use exact op-name CSV shapes."""
    csv_dir = tmp_path / "perf_report_csvs"
    csv_dir.mkdir()
    (csv_dir / "ops_unique_args.csv").write_text(
        (
            "name,op category,Input Dims,Input type\n"
            "sglang_profiler::attention_paged_attention_ragged_100,other,"
            '"((64, 32, 128), (2181038080,), (64, 32, 128), '
            '(1177709, 1, 8, 128), (), (1,), ())",'
            "\"('c10::BFloat16', 'unsigned char', 'c10::BFloat16', "
            "'c10::BFloat16', '', 'float', '')\"\n"
        ),
        encoding="utf-8",
    )
    candidates = [
        {
            "name": "sglang_profiler::attention_paged_attention_ragged_100",
            "duration_us": 170086.617,
            "call_count": 0,
            "candidate_source": "other_bucket_fallback",
            "source_file": "/sgl-workspace/aiter/csrc/cpp_itfs/pa/pa_ragged.py",
            "source_type": "python",
            "shapes": [],
            "tracelens_category": "other",
        }
    ]

    out = tla._finalize_candidates(
        candidates,
        total_dur=170086.617,
        perf_report_csv_dir=csv_dir,
    )

    assert out[0]["shapes"]
    assert "(64,32,128) bf16" in out[0]["shapes"]
    assert "(1177709,1,8,128) bf16" in out[0]["shapes"]
    assert "(1,) f32" in out[0]["shapes"]
    assert out[0]["shape_provenance"] == "torch_trace"


def test_finalize_does_not_touch_non_moe_or_already_shaped(tmp_path):
    """Unmatched non-MoE ops and already-shaped MoE candidates are left as-is."""
    csv_dir = _write_fused_moe_ops_unique_args(tmp_path)
    candidates = [
        {  # non-MoE op: must NOT be back-filled from the fused-MoE sidecar
            "name": "aten::mm",
            "duration_us": 100.0,
            "shapes": [],
            "source_file": "",
            "source_type": "unknown",
        },
        {  # MoE op that already carries shapes: keep them verbatim
            "name": "invoke_fused_moe_kernel",
            "duration_us": 200.0,
            "shapes": ["(99,99) bf16"],
            "source_file": "",
            "source_type": "python",
            "tracelens_category": "moe_fused",
        },
    ]
    out = tla._finalize_candidates(
        candidates,
        total_dur=300.0,
        perf_report_csv_dir=csv_dir,
    )
    assert out[0]["shapes"] == []
    assert out[1]["shapes"] == ["(99,99) bf16"]
    assert out[1]["invocation_cases"][0]["input_shapes"] == ["(99,99) bf16"]


def test_finalize_falls_back_to_heuristic_when_csv_missing(tmp_path):
    """Backward compatibility: no csv => the name heuristic still tags aiter::ck_moe_* as MoE (csv path is additive)."""
    candidates = [
        {
            "name": "aiter::ck_moe_stage1",
            "duration_us": 100.0,
            "call_count": 1,
            "source_file": "",
            "source_type": "unknown",
            "shapes": [[1, 1]],
        }
    ]
    out = tla._finalize_candidates(
        candidates,
        total_dur=100.0,
        perf_report_csv_dir=tmp_path / "does_not_exist",
    )
    assert out[0].get("tracelens_category", "") == ""
    assert out[0]["kernel_category"] == "MoE"


def test_normalize_upstream_category_handles_moe_aux_and_collective():
    """New mappings: MoE_aux -> MoE and CustomCollective -> Communication normalize cleanly."""
    normalize_upstream_category = tlr.normalize_upstream_category

    assert normalize_upstream_category("MoE_aux") == "MoE"
    assert normalize_upstream_category("moe_aux") == "MoE"
    assert normalize_upstream_category("CustomCollective") == "Communication"
    assert normalize_upstream_category("customcollective") == "Communication"


def test_write_reports_enriches_candidates_with_runtime_metadata(tmp_path):
    import json as _json
    from argparse import Namespace

    trace = tmp_path / "trace.json"
    trace.write_text("{}", encoding="utf-8")
    # write_reports requires the upstream analysis.md ; provide a stub.
    analysis_md = tmp_path / "run" / "tracelens" / "analysis.md"
    analysis_md.parent.mkdir(parents=True, exist_ok=True)
    analysis_md.write_text("# TraceLens stub\n", encoding="utf-8")
    candidate = {
        "kernel_id": "k001",
        "name": "paged_attention",
        "duration_us": 100.0,
        "call_count": 2,
        "gpu_pct": 10.0,
        "source_file": "/sgl-workspace/aiter/paged_attention.py",
        "shapes": [[1, 32, 128]],
        "is_multigpu": False,
        "num_gpus_recommended": 1,
        # Set the routable marker explicitly since this test bypasses _finalize_candidates.
        "reusable_native_kernel": True,
    }
    args = Namespace(
        trace_input=str(trace),
        model_name="llama",
        framework="sglang",
        target_platform="MI300X",
        analysis_mode="inference",
        runtime_env="local",
        dry_run=False,
    )

    artifacts = tla.write_reports(
        tmp_path / "run",
        trace_input_type="file",
        trace_files=[trace],
        candidates=[candidate],
        args=args,
        existing_report_path=analysis_md,
    )
    payload = _json.loads(Path(artifacts["kernel_candidates"]).read_text(encoding="utf-8"))
    enriched = payload["hot_kernels"][0]

    assert enriched["framework"] == "sglang"
    assert enriched["backend"] == "sglang"
    assert enriched["input_shapes"] == [{"call_num": 2, "shape": [1, 32, 128]}]
    assert enriched["output_shapes"] == []
    assert enriched["input_dtypes"] == []
    assert enriched["output_dtypes"] == []
    assert enriched["runtime_args"] == {}
    assert enriched["env_vars"] == {}
    assert enriched["kernel_params"] == {}
    assert enriched["runtime_flags"]["analysis_mode"] == "inference"
    assert enriched["runtime_flags"]["runtime_env"] == "local"
    assert enriched["runtime_flags"]["target_platform"] == "MI300X"
    assert enriched["runtime_flags"]["is_multigpu"] is False
    assert enriched["runtime_flags"]["num_gpus_recommended"] == 1


def test_load_model_kernel_params_reads_head_dim(tmp_path):
    import json as _json

    model_dir = tmp_path / "Qwen-Qwen3-8B"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        _json.dumps(
            {
                "head_dim": 128,
                "hidden_size": 4096,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
            }
        ),
        encoding="utf-8",
    )

    params = tla.load_model_kernel_params(str(model_dir))

    assert params["HEAD_SIZE"] == 128
    assert params["NUM_ATTENTION_HEADS"] == 32
    assert params["NUM_KEY_VALUE_HEADS"] == 8
    assert params["MODEL_CONFIG_PATH"] == str(model_dir / "config.json")


def test_load_model_kernel_params_derives_head_size_from_hidden_size(tmp_path):
    import json as _json

    model_dir = tmp_path / "meta-llama-Llama-3.1-8B"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        _json.dumps(
            {
                "hidden_size": 4096,
                "num_attention_heads": 32,
            }
        ),
        encoding="utf-8",
    )

    params = tla.load_model_kernel_params(str(model_dir))

    assert params["HEAD_SIZE"] == 128
    assert params["HIDDEN_SIZE"] == 4096
    assert params["NUM_ATTENTION_HEADS"] == 32


def test_load_model_kernel_params_preserves_mla_head_dims(tmp_path):
    import json as _json

    model_dir = tmp_path / "DeepSeek-R1-0528"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        _json.dumps(
            {
                "hidden_size": 7168,
                "num_attention_heads": 128,
                "qk_nope_head_dim": 128,
                "qk_rope_head_dim": 64,
                "v_head_dim": 128,
                "kv_lora_rank": 512,
            }
        ),
        encoding="utf-8",
    )

    params = tla.load_model_kernel_params(str(model_dir))

    assert "HEAD_SIZE" not in params
    assert params["QK_NOPE_HEAD_DIM"] == 128
    assert params["QK_ROPE_HEAD_DIM"] == 64
    assert params["V_HEAD_DIM"] == 128
    assert params["KV_LORA_RANK"] == 512


def test_write_reports_enriches_head_size_from_model_config(tmp_path):
    import json as _json
    from argparse import Namespace

    trace = tmp_path / "trace.json"
    trace.write_text("{}", encoding="utf-8")
    model_dir = tmp_path / "Qwen-Qwen3-8B"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        _json.dumps({"head_dim": 128, "num_attention_heads": 32}),
        encoding="utf-8",
    )
    analysis_md = tmp_path / "run" / "tracelens" / "analysis.md"
    analysis_md.parent.mkdir(parents=True, exist_ok=True)
    analysis_md.write_text("# TraceLens stub\n", encoding="utf-8")
    candidate = {
        "kernel_id": "k001",
        "name": "paged_attention",
        "duration_us": 100.0,
        "call_count": 2,
        "gpu_pct": 10.0,
        "source_file": "/sgl-workspace/aiter/paged_attention.py",
        "shapes": [[1, 32, 128]],
        "reusable_native_kernel": True,
    }
    args = Namespace(
        trace_input=str(trace),
        model_name=str(model_dir),
        framework="sglang",
        target_platform="MI300X",
        analysis_mode="inference",
        runtime_env="local",
        dry_run=False,
    )

    artifacts = tla.write_reports(
        tmp_path / "run",
        trace_input_type="file",
        trace_files=[trace],
        candidates=[candidate],
        args=args,
        existing_report_path=analysis_md,
    )
    payload = _json.loads(Path(artifacts["kernel_candidates"]).read_text(encoding="utf-8"))

    assert payload["hot_kernels"][0]["kernel_params"]["HEAD_SIZE"] == 128


# write_reports surfaces the upstream analysis.md as-is (no copies/aliases/fabrication)
def _make_write_reports_args(trace_path):
    from argparse import Namespace

    return Namespace(
        trace_input=str(trace_path),
        model_name="qwen3-30b-a3b",
        framework="sglang",
        target_platform="MI300X",
        analysis_mode="inference",
        runtime_env="local",
        dry_run=False,
    )


def test_write_reports_raises_when_analysis_md_missing(tmp_path):
    """write_reports refuses to fabricate a Markdown when analysis.md is missing."""
    import pytest

    trace = tmp_path / "trace.json"
    trace.write_text("{}", encoding="utf-8")
    args = _make_write_reports_args(trace)

    with pytest.raises(RuntimeError, match="did not produce analysis.md"):
        tla.write_reports(
            tmp_path / "run",
            trace_input_type="file",
            trace_files=[trace],
            candidates=[],
            args=args,
        )


def test_write_reports_raises_when_existing_report_does_not_exist(tmp_path):
    """A passed-but-nonexistent report path is treated as orchestrator failure, not a cue to fabricate."""
    import pytest

    trace = tmp_path / "trace.json"
    trace.write_text("{}", encoding="utf-8")
    args = _make_write_reports_args(trace)

    with pytest.raises(RuntimeError, match="did not produce analysis.md"):
        tla.write_reports(
            tmp_path / "run",
            trace_input_type="file",
            trace_files=[trace],
            candidates=[],
            args=args,
            existing_report_path=tmp_path / "does-not-exist" / "analysis.md",
        )


def test_write_reports_does_not_create_filename_aliases(tmp_path):
    """``analysis.md`` is the single exit; legacy aliases must not be written."""
    trace = tmp_path / "trace.json"
    trace.write_text("{}", encoding="utf-8")
    run_dir = tmp_path / "run"
    tracelens_dir = run_dir / "tracelens"
    tracelens_dir.mkdir(parents=True, exist_ok=True)
    analysis_md = tracelens_dir / "analysis.md"
    analysis_md.write_text("# TraceLens upstream report\n", encoding="utf-8")
    args = _make_write_reports_args(trace)

    artifacts = tla.write_reports(
        run_dir,
        trace_input_type="file",
        trace_files=[trace],
        candidates=[],
        args=args,
        existing_report_path=analysis_md,
    )

    # The returned trace_report_path must point at the upstream file,
    # not at a Hyperloom-owned copy.
    assert artifacts["trace_report_path"] == str(analysis_md)
    # And the legacy aliases must NOT exist on disk.
    assert not (tracelens_dir / "standalone_analysis.md").exists()
    assert not (tracelens_dir / "tracelens_report.md").exists()


def test_write_reports_does_not_mutate_upstream_analysis_md(tmp_path):
    """Hyperloom must not rewrite the upstream report's contents (byte-identity check)."""
    trace = tmp_path / "trace.json"
    trace.write_text("{}", encoding="utf-8")
    run_dir = tmp_path / "run"
    tracelens_dir = run_dir / "tracelens"
    tracelens_dir.mkdir(parents=True, exist_ok=True)
    analysis_md = tracelens_dir / "analysis.md"
    upstream_body = "# TraceLens upstream report\n\n## Detailed Analysis\n"
    analysis_md.write_text(upstream_body, encoding="utf-8")
    args = _make_write_reports_args(trace)

    tla.write_reports(
        run_dir,
        trace_input_type="file",
        trace_files=[trace],
        candidates=[],
        args=args,
        existing_report_path=analysis_md,
    )

    assert analysis_md.read_text(encoding="utf-8") == upstream_body


# ``kernel_candidates.json`` exposes ``hot_kernels`` as the FULL ranked hotspot
# set (routable + non-routable) while ``routable_kernels`` / ``skipped_kernels``
# carry the reusable / non-reusable subsets.
def _contract_candidates():
    return [
        {
            "kernel_id": "k001",
            "name": "fused_moe",
            "duration_us": 300.0,
            "gpu_pct": 30.0,
            "source_file": "/repo/aiter/moe.py",
            "reusable_native_kernel": True,
        },
        {
            "kernel_id": "k002",
            "name": "aten::mm",
            "duration_us": 200.0,
            "gpu_pct": 20.0,
            "source_file": "",
            "reusable_native_kernel": False,
        },
        # No ``reusable_native_kernel`` key at all -> non-routable (``is not True``).
        {"kernel_id": "k003", "name": "Cijk_vendor_gemm", "duration_us": 150.0, "gpu_pct": 15.0, "source_file": ""},
        {
            "kernel_id": "k004",
            "name": "aiter::rmsnorm",
            "duration_us": 100.0,
            "gpu_pct": 10.0,
            "source_file": "/repo/aiter/rmsnorm.py",
            "reusable_native_kernel": True,
        },
    ]


def test_write_reports_kernel_candidates_contract_full_hot_with_routable_skipped_subsets(tmp_path):
    """kernel_candidates.json: hot_kernels is full; routable/skipped are disjoint subsets whose union == hot."""
    trace = tmp_path / "trace.json"
    trace.write_text("{}", encoding="utf-8")
    analysis_md = tmp_path / "run" / "tracelens" / "analysis.md"
    analysis_md.parent.mkdir(parents=True, exist_ok=True)
    analysis_md.write_text("# TraceLens stub\n", encoding="utf-8")
    args = _make_write_reports_args(trace)

    artifacts = tla.write_reports(
        tmp_path / "run",
        trace_input_type="file",
        trace_files=[trace],
        candidates=_contract_candidates(),
        args=args,
        existing_report_path=analysis_md,
    )
    payload = json.loads(Path(artifacts["kernel_candidates"]).read_text(encoding="utf-8"))

    hot_ids = [c["kernel_id"] for c in payload["hot_kernels"]]
    routable_ids = [c["kernel_id"] for c in payload["routable_kernels"]]
    skipped_ids = [c["kernel_id"] for c in payload["skipped_kernels"]]

    # hot_kernels is the FULL ranked set, order preserved.
    assert hot_ids == ["k001", "k002", "k003", "k004"]
    # routable == exactly the ``reusable_native_kernel is True`` rows.
    assert routable_ids == ["k001", "k004"]
    assert all(c.get("reusable_native_kernel") is True for c in payload["routable_kernels"])
    # skipped == exactly the non-True rows (False or absent).
    assert skipped_ids == ["k002", "k003"]
    assert all(c.get("reusable_native_kernel") is not True for c in payload["skipped_kernels"])
    # Contract invariants: partition of hot with no overlap and no leakage.
    assert set(routable_ids) | set(skipped_ids) == set(hot_ids)
    assert set(routable_ids) & set(skipped_ids) == set()
    assert set(skipped_ids) <= set(hot_ids)
    assert len(payload["hot_kernels"]) == len(payload["routable_kernels"]) + len(payload["skipped_kernels"])


def test_write_reports_tracelens_report_hot_kernels_is_full_set(tmp_path):
    """tracelens_report.json mirrors the FULL hot_kernels set (no routable-only truncation)."""
    trace = tmp_path / "trace.json"
    trace.write_text("{}", encoding="utf-8")
    analysis_md = tmp_path / "run" / "tracelens" / "analysis.md"
    analysis_md.parent.mkdir(parents=True, exist_ok=True)
    analysis_md.write_text("# TraceLens stub\n", encoding="utf-8")
    args = _make_write_reports_args(trace)

    artifacts = tla.write_reports(
        tmp_path / "run",
        trace_input_type="file",
        trace_files=[trace],
        candidates=_contract_candidates(),
        args=args,
        existing_report_path=analysis_md,
    )
    report = json.loads(Path(artifacts["tracelens_report_json"]).read_text(encoding="utf-8"))
    assert [c["kernel_id"] for c in report["hot_kernels"]] == ["k001", "k002", "k003", "k004"]


# SDK runner for TraceLens analysis-orchestrator skill
def test_124_build_orchestrator_prompt_supplies_step0_inputs(tmp_path):
    skill = tmp_path / "SKILL.md"
    trace = tmp_path / "mixed_steady_state_0_trace.json.gz"
    out = tmp_path / "tracelens"
    root = tmp_path / "TraceLens-internal"
    capture = tmp_path / "capture_traces"

    prompt = tlr.build_orchestrator_prompt(
        skill_path=skill,
        trace_path=trace,
        output_dir=out,
        tracelens_root=root,
        tracelens_internal_root=tmp_path / "TraceLens-internal",
        platform="MI300X",
        framework="vllm",
        analysis_mode="default",
        capture_folder=capture,
    )

    assert str(skill) in prompt
    assert str(trace) in prompt
    assert str(out) in prompt
    assert "Analysis mode: inference" in prompt
    assert "Inference execution mode: graph_capture" in prompt
    assert "Comparison scope: standalone" in prompt
    assert "Do not ask the user" in prompt


def test_count_gpu_kernel_events_distinguishes_cpu_only_and_real_traces(tmp_path):
    import gzip
    import json as _json

    cpu_only = tmp_path / "cpu_only.json.gz"
    with gzip.open(cpu_only, "wt") as f:
        _json.dump(
            {
                "traceEvents": [
                    {"cat": "cpu_op", "name": "aten::add", "dur": 1.0},
                    {"cat": "python_function", "name": "wrapper", "dur": 2.0},
                    {"cat": "cuda_runtime", "name": "hipDeviceSynchronize", "dur": 3.0},
                ]
            },
            f,
        )
    assert tla.count_gpu_kernel_events(cpu_only) == 0

    real = tmp_path / "real.json.gz"
    with gzip.open(real, "wt") as f:
        _json.dump(
            {
                "traceEvents": [
                    {"cat": "cpu_op", "name": "aten::add", "dur": 1.0},
                    {"cat": "kernel", "name": "void some_gemm_kernel<...>", "dur": 7.0},
                    {"cat": "kernel", "name": "void some_attn_kernel<...>", "dur": 11.0},
                    {"cat": "cuda_runtime", "name": "hipLaunchKernel", "dur": 0.5},
                ]
            },
            f,
        )
    assert tla.count_gpu_kernel_events(real) == 2


def test_124_tracelens_analysis_fails_fast_on_cpu_only_trace(tmp_path):
    """A CPU-only trace must fail loudly before TraceLens install / split / SDK runs."""
    import gzip
    import json as _json
    from unittest.mock import patch

    tl_root = tmp_path / "TraceLens-internal"
    skill_dir = tl_root / "TraceLens" / "Agent" / "Analysis" / "skills" / "analysis-orchestrator"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("stub")
    workspace = tmp_path / "ws"
    workspace.mkdir()

    trace = tmp_path / "cpu_only_trace.json.gz"
    with gzip.open(trace, "wt") as f:
        _json.dump(
            {
                "traceEvents": [
                    {"cat": "cpu_op", "name": "aten::add", "dur": 1.0},
                    {"cat": "python_function", "name": "wrapper", "dur": 2.0},
                ]
            },
            f,
        )

    captured: list[list[str]] = []

    class _Result:
        def __init__(self):
            self.returncode = 0
            self.stdout = "ok"

    def fake_run(cmd, *args, **kwargs):
        captured.append(list(cmd))
        return _Result()

    argv = [
        "tracelens_analysis.py",
        "--trace-input",
        str(trace),
        "--workspace-path",
        str(workspace),
        "--tracelens-root",
        str(tl_root),
        "--target-platform",
        "MI300X",
        "--budget-minutes",
        "1",
        "--no-llm-orchestrator",
    ]
    import os as _os

    env_backup = dict(_os.environ)
    try:
        with patch.object(tla.subprocess, "run", side_effect=fake_run), patch.object(tla.sys, "argv", argv):
            try:
                rc = tla.main()
            except SystemExit as exc:
                rc = int(exc.code or 0)
    finally:
        _os.environ.clear()
        _os.environ.update(env_backup)

    assert rc != 0, "fail-fast on CPU-only trace must return non-zero"
    assert all(
        "TraceLens.TraceUtils.split_inference_trace_annotation" not in str(p) for cmd in captured for p in cmd
    ), f"splitter must not run on CPU-only trace; captured={captured}"
    assert all(
        "TraceLens_generate_perf_report_pytorch_inference" not in str(c[0]) or "--help" in c for c in captured if c
    ), f"perf-report CLI must not be invoked for CPU-only trace; captured={captured}"


def test_124_run_tracelens_skill_uses_sdk_and_artifacts(tmp_path):
    import asyncio
    from dataclasses import dataclass
    from typing import Any

    @dataclass
    class _TextBlock:
        text: str

    @dataclass
    class _Message:
        content: list[Any]

    class _FakeOptions:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    output_dir = tmp_path / "out"
    captured: dict[str, Any] = {}

    async def _fake_query(*, prompt, options):
        captured["prompt"] = prompt
        captured["options"] = options.kwargs
        output_dir.mkdir(parents=True, exist_ok=True)
        # TraceLens contract: orchestrator writes ``analysis.md``.
        (output_dir / "analysis.md").write_text("# report\n", encoding="utf-8")
        yield _Message(content=[_TextBlock("done")])

    res = asyncio.run(
        tlr.run_tracelens_skill(
            skill_path=tmp_path / "skill.md",
            trace_path=tmp_path / "trace.json.gz",
            output_dir=output_dir,
            tracelens_root=tmp_path,
            tracelens_internal_root=tmp_path / "TraceLens-internal",
            platform="MI300X",
            framework="sglang",
            analysis_mode="default",
            capture_folder=None,
            budget_minutes=1,
            sdk_query_factory=_fake_query,
            sdk_options_cls=_FakeOptions,
        )
    )

    assert res.report_path.exists()
    assert res.artifact_paths["tracelens_agent_report"] == str(res.report_path)
    assert "analysis-orchestrator" in captured["prompt"] or "skill.md" in captured["prompt"]
    assert "Bash" in captured["options"]["allowed_tools"]
    assert "Task" in captured["options"]["allowed_tools"]
    assert res.runner == "claude_agent_sdk"


def test_run_tracelens_skill_uses_hermetic_claude_env(tmp_path, monkeypatch):
    """TraceLens SDK runner must not inherit stale global Claude settings.

    Passing ``env`` and ``setting_sources=[]`` keeps the SDK child tied to the
    active run contract rather than a stale ``~/.claude/settings.json`` token.
    """
    import asyncio
    from dataclasses import dataclass
    from typing import Any

    @dataclass
    class _TextBlock:
        text: str

    @dataclass
    class _Message:
        content: list[Any]

    class _FakeOptions:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
    monkeypatch.setenv("_".join(("ANTHROPIC", "API", "KEY")), "anthropic-token-active")
    monkeypatch.delenv("_".join(("ANTHROPIC", "AUTH", "TOKEN")), raising=False)
    monkeypatch.delenv("_".join(("OPENAI", "API", "KEY")), raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    output_dir = tmp_path / "out"
    captured: dict[str, Any] = {}

    async def _fake_query(*, prompt, options):
        captured["options"] = options.kwargs
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "analysis.md").write_text("# report\n", encoding="utf-8")
        yield _Message(content=[_TextBlock("done")])

    asyncio.run(
        tlr.run_tracelens_skill(
            skill_path=tmp_path / "skill.md",
            trace_path=tmp_path / "trace.json.gz",
            output_dir=output_dir,
            tracelens_root=tmp_path,
            tracelens_internal_root=tmp_path / "TraceLens-internal",
            platform="MI355X",
            framework="vllm",
            analysis_mode="inference",
            capture_folder=None,
            budget_minutes=1,
            model="claude-sonnet-4-5-20250929",
            sdk_query_factory=_fake_query,
            sdk_options_cls=_FakeOptions,
        )
    )

    opts = captured["options"]
    assert opts["setting_sources"] == []
    child_env = opts["env"]
    assert child_env["ANTHROPIC_BASE_URL"] == "https://api.anthropic.com"
    assert child_env["_".join(("ANTHROPIC", "API", "KEY"))] == "anthropic-token-active"
    assert child_env["_".join(("ANTHROPIC", "AUTH", "TOKEN"))] == "anthropic-token-active"
    assert child_env["ANTHROPIC_MODEL"] == "claude-sonnet-4-5-20250929"
    assert child_env["ANTHROPIC_SMALL_FAST_MODEL"] == "claude-sonnet-4-5-20250929"


def _use_openai_only_env(monkeypatch) -> None:
    """Pin the process env to the OpenAI-only credential shape."""
    monkeypatch.setenv("_".join(("OPENAI", "API", "KEY")), "openai-token")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.delenv("_".join(("ANTHROPIC", "API", "KEY")), raising=False)
    monkeypatch.delenv("_".join(("ANTHROPIC", "AUTH", "TOKEN")), raising=False)
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    monkeypatch.delenv("_".join(("DEEPSEEK", "API", "KEY")), raising=False)


def test_run_tracelens_skill_openai_only_uses_codex_tool_runner(tmp_path, monkeypatch):
    """OpenAI-only deployments must run TraceLens on the Codex Agent SDK.

    Hyperloom makes no bare LLM API calls, so this path must go through an
    agent runtime whose tools, sandbox and turn management come from the SDK.
    """
    import asyncio

    from hyperloom.common.codex_session import CodexSessionResult

    output_dir = tmp_path / "out"
    calls: list[dict] = []

    async def _fake_codex_turn(**kwargs):
        calls.append(kwargs)
        (output_dir / "analysis.md").write_text("# Codex TraceLens report\n", encoding="utf-8")
        return CodexSessionResult(
            text="wrote analysis.md",
            usage={"input_tokens": 11, "output_tokens": 7, "reasoning_output_tokens": 900},
            thread_id="thread-123",
        )

    def _no_claude_query(**_kwargs):
        raise AssertionError("OpenAI-only TraceLens path must not use Claude SDK")

    _use_openai_only_env(monkeypatch)

    res = asyncio.run(
        tlr.run_tracelens_skill(
            skill_path=tmp_path / "skill.md",
            trace_path=tmp_path / "trace.json.gz",
            output_dir=output_dir,
            tracelens_root=tmp_path,
            tracelens_internal_root=None,
            platform="MI355X",
            framework="vllm",
            analysis_mode="inference",
            capture_folder=None,
            budget_minutes=30,
            model="gpt-5.5",
            sdk_query_factory=_no_claude_query,
            sdk_options_cls=object,
            codex_turn_runner=_fake_codex_turn,
        )
    )

    assert res.report_path == output_dir / "analysis.md"
    assert res.report_path.read_text(encoding="utf-8") == "# Codex TraceLens report\n"
    # The result carries the runner that actually ran, so the caller reports the
    # real provider instead of hardcoding one.
    assert res.runner == "codex"
    assert res.raw_text == "wrote analysis.md"
    assert "tracelens_agent_report" in res.artifact_paths
    assert res.artifact_paths["tracelens_cmd_prefix"] == str(output_dir / "cache" / "cmd_prefix.txt")
    # A clean turn records no SDK error.
    assert "tracelens_agent_sdk_error" not in res.artifact_paths

    assert len(calls) == 1
    call = calls[0]
    assert call["model"] == "gpt-5.5"
    # The session works out of the TraceLens root so the skill's command-prefix
    # cache and relative paths resolve as they do on the Claude path.
    assert call["cwd"] == tmp_path
    # The output dir is the only extra writable root: nothing else is mutable.
    assert call["writable_roots"] == (output_dir,)
    assert call["timeout_sec"] == 30 * 60.0
    assert "TraceLens analysis runner" in call["developer_instructions"]
    assert str(tmp_path / "skill.md") in call["prompt"]


def test_run_tracelens_skill_codex_floors_the_turn_timeout(tmp_path, monkeypatch):
    """A sub-minute budget must not translate into an instant Codex timeout."""
    import asyncio

    from hyperloom.common.codex_session import CodexSessionResult

    output_dir = tmp_path / "out"
    calls: list[dict] = []

    async def _fake_codex_turn(**kwargs):
        calls.append(kwargs)
        (output_dir / "analysis.md").write_text("# report\n", encoding="utf-8")
        return CodexSessionResult(text="done")

    _use_openai_only_env(monkeypatch)

    asyncio.run(
        tlr.run_tracelens_skill(
            skill_path=tmp_path / "skill.md",
            trace_path=tmp_path / "trace.json.gz",
            output_dir=output_dir,
            tracelens_root=tmp_path,
            tracelens_internal_root=None,
            platform="MI355X",
            framework="vllm",
            analysis_mode="inference",
            capture_folder=None,
            budget_minutes=0.1,
            codex_turn_runner=_fake_codex_turn,
        )
    )

    assert calls[0]["timeout_sec"] == 60.0


def test_run_tracelens_skill_codex_raises_when_report_missing(tmp_path, monkeypatch):
    """A Codex failure with no report must surface the SDK error, not a bare miss."""
    import asyncio

    from hyperloom.common.codex_session import CodexSessionTimeoutError

    output_dir = tmp_path / "out"

    async def _failing_codex_turn(**_kwargs):
        raise CodexSessionTimeoutError("Codex turn timed out after 300s")

    _use_openai_only_env(monkeypatch)

    with pytest.raises(RuntimeError) as excinfo:
        asyncio.run(
            tlr.run_tracelens_skill(
                skill_path=tmp_path / "skill.md",
                trace_path=tmp_path / "trace.json.gz",
                output_dir=output_dir,
                tracelens_root=tmp_path,
                tracelens_internal_root=None,
                platform="MI355X",
                framework="vllm",
                analysis_mode="inference",
                capture_folder=None,
                budget_minutes=5,
                codex_turn_runner=_failing_codex_turn,
            )
        )

    assert "timed out after 300s" in str(excinfo.value)
    assert str(output_dir / "analysis.md") in str(excinfo.value)


def test_run_tracelens_skill_codex_keeps_report_written_before_failure(tmp_path, monkeypatch):
    """analysis.md is the source of truth: a late SDK error is metadata."""
    import asyncio

    from hyperloom.common.codex_session import CodexSessionError

    output_dir = tmp_path / "out"

    async def _late_failure(**_kwargs):
        (output_dir / "analysis.md").write_text("# report\n", encoding="utf-8")
        raise CodexSessionError("app-server transport closed")

    _use_openai_only_env(monkeypatch)

    res = asyncio.run(
        tlr.run_tracelens_skill(
            skill_path=tmp_path / "skill.md",
            trace_path=tmp_path / "trace.json.gz",
            output_dir=output_dir,
            tracelens_root=tmp_path,
            tracelens_internal_root=None,
            platform="MI355X",
            framework="vllm",
            analysis_mode="inference",
            capture_folder=None,
            budget_minutes=5,
            codex_turn_runner=_late_failure,
        )
    )

    assert res.runner == "codex"
    assert "app-server transport closed" in res.artifact_paths["tracelens_agent_sdk_error"]


def test_run_tracelens_skill_codex_reports_in_band_turn_error(tmp_path, monkeypatch):
    """A turn that completes carrying an SDK error must not look successful."""
    import asyncio

    from hyperloom.common.codex_session import CodexSessionResult

    output_dir = tmp_path / "out"

    async def _in_band_error(**_kwargs):
        (output_dir / "analysis.md").write_text("# report\n", encoding="utf-8")
        return CodexSessionResult(text="", error="rate limit exceeded")

    _use_openai_only_env(monkeypatch)

    res = asyncio.run(
        tlr.run_tracelens_skill(
            skill_path=tmp_path / "skill.md",
            trace_path=tmp_path / "trace.json.gz",
            output_dir=output_dir,
            tracelens_root=tmp_path,
            tracelens_internal_root=None,
            platform="MI355X",
            framework="vllm",
            analysis_mode="inference",
            capture_folder=None,
            budget_minutes=5,
            codex_turn_runner=_in_band_error,
        )
    )

    assert res.artifact_paths["tracelens_agent_sdk_error"] == "rate limit exceeded"


def test_run_tracelens_skill_aborts_on_stream_idle_timeout(tmp_path, monkeypatch):
    """A gateway stream that goes silent mid-response must abort on the
    per-message idle timeout instead of blocking forever. The runner records
    the idle-timeout error and, since analysis.md was already written, still
    returns it as the report."""
    import asyncio
    from dataclasses import dataclass
    from typing import Any

    @dataclass
    class _TextBlock:
        text: str

    @dataclass
    class _Message:
        content: list[Any]

    class _FakeOptions:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    output_dir = tmp_path / "out"

    # Drive the idle timeout fast (the resolver floors real values at 30s).
    monkeypatch.setattr(tlr, "_resolve_stream_idle_timeout_sec", lambda: 0.5)

    async def _stalling_query(*, prompt, options):
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "analysis.md").write_text("# partial report\n", encoding="utf-8")
        # One chunk arrives, then the stream goes silent (partial response,
        # stop_reason=None) — emulate by sleeping far past the idle timeout.
        yield _Message(content=[_TextBlock("chunk-1")])
        await asyncio.sleep(60)
        yield _Message(content=[_TextBlock("never-reached")])

    started = asyncio.run(_run_and_time(tlr, _stalling_query, _FakeOptions, tmp_path, output_dir))
    res, elapsed = started

    # Aborted quickly (well under the 60s stall), not hung.
    assert elapsed < 30.0
    assert res.report_path.exists()
    sdk_error = res.artifact_paths.get("tracelens_agent_sdk_error", "")
    assert "idle timeout" in sdk_error


async def _run_and_time(tlr_mod, query, options_cls, tmp_path, output_dir):
    import time as _time

    t0 = _time.monotonic()
    res = await tlr_mod.run_tracelens_skill(
        skill_path=tmp_path / "skill.md",
        trace_path=tmp_path / "trace.json.gz",
        output_dir=output_dir,
        tracelens_root=tmp_path,
        tracelens_internal_root=tmp_path / "TraceLens-internal",
        platform="MI300X",
        framework="sglang",
        analysis_mode="default",
        capture_folder=None,
        budget_minutes=1,
        sdk_query_factory=query,
        sdk_options_cls=options_cls,
    )
    return res, _time.monotonic() - t0


# ===========================================================================
# analysis.md is the only contracted TraceLens output.
def test_t2_run_tracelens_skill_ignores_intermediate_sidecars(tmp_path):
    """SDK orchestrator sidecars must not be surfaced as Hyperloom inputs."""
    import asyncio
    import json as _json
    from dataclasses import dataclass
    from typing import Any

    @dataclass
    class _TextBlock:
        text: str

    @dataclass
    class _Message:
        content: list[Any]

    class _FakeOptions:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    output_dir = tmp_path / "out"

    async def _fake_query(*, prompt, options):
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "analysis.md").write_text("# report\n", encoding="utf-8")
        (output_dir / "priority_data.json").write_text(
            _json.dumps({"findings": []}),
            encoding="utf-8",
        )
        (output_dir / "category_data").mkdir(parents=True, exist_ok=True)
        (output_dir / "category_data" / "category_manifest.json").write_text(
            _json.dumps({"categories": []}),
            encoding="utf-8",
        )
        yield _Message(content=[_TextBlock("done — sidecars ignored")])

    res = asyncio.run(
        tlr.run_tracelens_skill(
            skill_path=tmp_path / "skill.md",
            trace_path=tmp_path / "trace.json.gz",
            output_dir=output_dir,
            tracelens_root=tmp_path,
            tracelens_internal_root=tmp_path / "TraceLens-internal",
            platform="MI300X",
            framework="sglang",
            analysis_mode="default",
            capture_folder=None,
            budget_minutes=1,
            sdk_query_factory=_fake_query,
            sdk_options_cls=_FakeOptions,
        )
    )

    assert res.report_path.exists(), "analysis.md is the single source of truth and must exist"
    assert "tracelens_agent_report" in res.artifact_paths
    assert "tracelens_priority_data" not in res.artifact_paths
    assert "tracelens_category_manifest" not in res.artifact_paths


def test_t2_missing_analysis_md_still_raises(tmp_path):
    """Negative control: a missing ``analysis.md`` is still a hard error (T2 only relaxes sidecars)."""
    import asyncio
    from dataclasses import dataclass
    from typing import Any

    @dataclass
    class _TextBlock:
        text: str

    @dataclass
    class _Message:
        content: list[Any]

    class _FakeOptions:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    output_dir = tmp_path / "out"

    async def _fake_query(*, prompt, options):
        output_dir.mkdir(parents=True, exist_ok=True)
        # Write ONLY a sidecar (no analysis.md); the wrapper must still fail loudly.
        (output_dir / "priority_data.json").write_text("{}", encoding="utf-8")
        yield _Message(content=[_TextBlock("done")])

    with pytest.raises(RuntimeError, match="analysis.md"):
        asyncio.run(
            tlr.run_tracelens_skill(
                skill_path=tmp_path / "skill.md",
                trace_path=tmp_path / "trace.json.gz",
                output_dir=output_dir,
                tracelens_root=tmp_path,
                tracelens_internal_root=tmp_path / "TraceLens-internal",
                platform="MI300X",
                framework="sglang",
                analysis_mode="default",
                capture_folder=None,
                budget_minutes=1,
                sdk_query_factory=_fake_query,
                sdk_options_cls=_FakeOptions,
            )
        )


# splitter CLI must match the real split_inference_trace_annotation interface
# (positional trace_path, -o, --find-steady-state); the old --input/--platform form failed.
def test_discover_trace_inputs_prefers_merged_trace_over_tp0_decode(tmp_path):
    trace_dir = tmp_path / "torch_trace"
    trace_dir.mkdir()
    tp0_decode = trace_dir / "177-TP-0-DECODE.trace.json.gz"
    tp0_decode.write_text("{}", encoding="utf-8")
    merged = trace_dir / "merged-177.trace.json.gz"
    merged.write_text("{}", encoding="utf-8")
    tp1_extend = trace_dir / "177-TP-1-EXTEND.trace.json.gz"
    tp1_extend.write_text("{}", encoding="utf-8")

    kind, traces = tla.discover_trace_inputs(trace_dir)
    assert kind == "capture_dir"
    assert traces[0] == merged
    assert traces[-1] == tp0_decode


def _xdit_roofline_capture(trace_dir: Path) -> Path:
    """Recreate an xDiT roofline capture directory as production writes it.

    Names and sizes are taken from a real failing run: a 910 KB rank capture
    with 30k kernel events, beside a trace_split/ directory of ~900-byte
    per-phase fragments and annotation sidecars.
    """
    trace_dir.mkdir(parents=True, exist_ok=True)
    raw = trace_dir / "rank_0.trace.json.gz"
    with gzip.open(raw, "wt") as fh:
        json.dump({"traceEvents": [{"cat": "kernel", "name": "void flash_fwd<...>", "dur": 40} for _ in range(64)]}, fh)

    split = trace_dir / "trace_split"
    split.mkdir()
    for name in (
        "decode_only_steady_state_prefill_0_prefilldecode_0_decode_1_bs1_conc1_rank_0.trace.json.gz",
        "mixed_steady_state_prefill_0_prefilldecode_0_decode_1_bs1_conc1_rank_0.trace.json.gz",
    ):
        with gzip.open(split / name, "wt") as fh:
            json.dump({"traceEvents": []}, fh)
    for i in range(3):
        with gzip.open(split / f"rank_0.trace_annotation_iteration_{i}.json.gz", "wt") as fh:
            json.dump({"traceEvents": []}, fh)
    (split / "execution_details.json").write_text('{"steps": 1}', encoding="utf-8")
    return raw


def test_discover_trace_inputs_prefers_raw_capture_over_split_fragments(tmp_path):
    """The raw capture must lead, whatever the fragments are named.

    Every file in this layout used to land in the same default bucket, so
    alphabetical order decided -- and `decode_only_...` sorts ahead of
    `rank_0.trace.json.gz`. The preflight then read a 900-byte fragment, found
    no GPU kernels, and reported the whole capture as CPU-only.
    """
    trace_dir = tmp_path / "torch_trace"
    raw = _xdit_roofline_capture(trace_dir)

    kind, traces = tla.discover_trace_inputs(trace_dir)

    assert kind == "capture_dir"
    assert traces[0] == raw, "the raw capture must be the first candidate"
    # Nothing derived may outrank it.
    assert all("trace_split" in p.parts for p in traces[1:])


def test_the_leading_candidate_is_the_one_with_the_kernels(tmp_path):
    """Ordering is only useful if it puts a probe-able trace first.

    Ties the two halves of the fix together: whichever file discovery leads
    with is the file the CPU-only preflight opens, so that file has to be the
    one carrying GPU kernel events.
    """
    trace_dir = tmp_path / "torch_trace"
    _xdit_roofline_capture(trace_dir)

    _kind, traces = tla.discover_trace_inputs(trace_dir)

    assert tla.count_gpu_kernel_events(traces[0]) == 64
    # The fragment that used to be probed first really does look CPU-only, so
    # the old ordering failed for a real reason and not a test artefact.
    fragment = next(p for p in traces if p.name.startswith("decode_only_"))
    assert tla.count_gpu_kernel_events(fragment) == 0


def test_annotation_sidecars_sort_after_a_capture_even_outside_trace_split(tmp_path):
    """Annotation sidecars are per-iteration slivers wherever they sit."""
    trace_dir = tmp_path / "torch_trace"
    trace_dir.mkdir()
    raw = trace_dir / "rank_0.trace.json.gz"
    raw.write_text("{}", encoding="utf-8")
    sidecar = trace_dir / "aaa.trace_annotation_iteration_0.json.gz"
    sidecar.write_text("{}", encoding="utf-8")

    _kind, traces = tla.discover_trace_inputs(trace_dir)

    assert traces[0] == raw
    assert traces[-1] == sidecar


def test_fragments_flat_beside_the_capture_are_still_demoted(tmp_path):
    """The demotion must not depend on the splitter nesting its output.

    Production nests fragments under trace_split/ today, but keying only on the
    directory would leave a flat layout exactly as broken as before: the phase
    names sort ahead of rank files on the first letter.
    """
    trace_dir = tmp_path / "torch_trace"
    trace_dir.mkdir()
    raw = _rank_trace(trace_dir / "rank_0.trace.json.gz", kernels=40)
    flat_fragment = _rank_trace(
        trace_dir / "decode_only_steady_state_prefill_0_decode_1_bs1_conc1_rank_0.trace.json.gz", kernels=0
    )

    _kind, traces = tla.discover_trace_inputs(trace_dir)

    assert traces[0] == raw
    assert traces[-1] == flat_fragment


def test_an_eight_rank_flat_capture_does_not_exhaust_the_probe_budget(tmp_path):
    """xDiT runs at TP=8, so a flat layout could present eight fragments first.

    With the fragments still in the default bucket they would consume the whole
    probe budget before any rank file was opened, and the run would fail with the
    error this change exists to remove.
    """
    trace_dir = tmp_path / "torch_trace"
    trace_dir.mkdir()
    for rank in range(8):
        _rank_trace(trace_dir / f"decode_only_steady_state_x_rank_{rank}.trace.json.gz", kernels=0)
        _rank_trace(trace_dir / f"mixed_steady_state_x_rank_{rank}.trace.json.gz", kernels=0)
    captures = [_rank_trace(trace_dir / f"rank_{r}.trace.json.gz", kernels=25) for r in range(8)]

    _kind, traces = tla.discover_trace_inputs(trace_dir)

    leading = traces[: tla._KERNEL_PROBE_LIMIT]
    assert any(p in captures for p in leading), "a real capture must be reachable inside the probe budget"
    assert tla.count_gpu_kernel_events(traces[0]) > 0


def test_a_non_trace_sidecar_does_not_lead_discovery(tmp_path):
    """execution_details.json is swept in by the *.json glob but is not a trace.

    It sorts ahead of rank_0.trace.json.gz alphabetically, so before size
    ordering it was trace_files[0] in every healthy nested capture: one wasted
    probe, and a promotion logged on every run, which made the log line
    meaningless exactly when it should have meant something.
    """
    trace_dir = tmp_path / "torch_trace"
    raw = _xdit_roofline_capture(trace_dir)
    sidecar = trace_dir / "trace_split" / "execution_details.json"
    assert sidecar.exists()

    _kind, traces = tla.discover_trace_inputs(trace_dir)

    assert traces[0] == raw
    assert traces.index(sidecar) > 0


def test_the_larger_trace_leads_within_a_bucket(tmp_path):
    """Size is the part of the ordering that needs no name recognition."""
    trace_dir = tmp_path / "torch_trace"
    trace_dir.mkdir()
    small = _rank_trace(trace_dir / "aaa_unknown_shape.trace.json.gz", kernels=1)
    large = _rank_trace(trace_dir / "zzz_unknown_shape.trace.json.gz", kernels=400)
    assert large.stat().st_size > small.stat().st_size

    _kind, traces = tla.discover_trace_inputs(trace_dir)

    assert traces[0] == large


def test_merged_trace_still_wins_over_a_raw_rank_capture(tmp_path):
    """The pre-existing preference is unchanged: merged first when present."""
    trace_dir = tmp_path / "torch_trace"
    trace_dir.mkdir()
    (trace_dir / "rank_0.trace.json.gz").write_text("{}", encoding="utf-8")
    merged = trace_dir / "merged-42.trace.json.gz"
    merged.write_text("{}", encoding="utf-8")

    _kind, traces = tla.discover_trace_inputs(trace_dir)

    assert traces[0] == merged


def test_127_splitter_cli_uses_positional_trace_path_and_find_steady_state(
    tmp_path,
    capsys,
):
    """The end-to-end split path must call the real splitter interface, not the broken --input/--platform form."""
    import gzip
    import json as _json
    from unittest.mock import patch

    # Pretend TraceLens root is present so the run reaches the splitter step.
    tl_root = tmp_path / "TraceLens-internal"
    skill_dir = tl_root / "TraceLens" / "Agent" / "Analysis" / "skills" / "analysis-orchestrator"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("stub")
    workspace = tmp_path / "ws"
    workspace.mkdir()
    capture = tmp_path / "capture_traces"
    capture.mkdir()
    trace = tmp_path / "trace.json.gz"
    with gzip.open(trace, "wt") as f:
        _json.dump(
            {
                "traceEvents": [
                    # At least one real GPU kernel event so the new fail-fast
                    # validation lets the run continue into the splitter step.
                    {"cat": "kernel", "name": "void some_real_kernel<...>", "dur": 5.0},
                ]
            },
            f,
        )

    captured: list[list[str]] = []

    class _Result:
        def __init__(self, returncode=0, stdout=""):
            self.returncode = returncode
            self.stdout = stdout

    def fake_run(cmd, *args, **kwargs):
        captured.append(list(cmd))
        # Make pip install and splitter invocations succeed.
        return _Result(returncode=0, stdout="ok")

    argv = [
        "tracelens_analysis.py",
        "--trace-input",
        str(trace),
        "--workspace-path",
        str(workspace),
        "--tracelens-root",
        str(tl_root),
        "--target-platform",
        "MI300X",
        "--budget-minutes",
        "1",
        "--no-llm-orchestrator",
        "--capture-folder",
        str(capture),
        "--split-conc",
        "8",
        "--split-osl",
        "1024",
    ]
    import os as _os

    env_backup = dict(_os.environ)
    try:
        with patch.object(tla.subprocess, "run", side_effect=fake_run), patch.object(tla.sys, "argv", argv):
            try:
                tla.main()
            except SystemExit as exc:
                # tla.main() may CLI-exit because the mocked run does not
                # produce analysis.md. The test asserts the splitter command
                # shape below, not the program's overall exit status.
                _ = exc
    finally:
        _os.environ.clear()
        _os.environ.update(env_backup)

    splitter_cmd = next(
        (c for c in captured if any("split_inference_trace_annotation" in str(p) for p in c)),
        None,
    )
    assert splitter_cmd is not None, f"splitter never invoked; cmds={captured}"
    # Real CLI: positional trace_path, -o output, --find-steady-state.
    assert "--input" not in splitter_cmd, f"--input must be removed: {splitter_cmd}"
    assert "--platform" not in splitter_cmd, f"--platform is not a valid splitter flag: {splitter_cmd}"
    assert "--find-steady-state" in splitter_cmd, splitter_cmd
    assert "-o" in splitter_cmd, splitter_cmd
    assert str(trace) in splitter_cmd, f"trace_path must be passed positionally: {splitter_cmd}"
    # CONC / OSL passthroughs.
    assert "--CONC" in splitter_cmd and "8" in splitter_cmd, splitter_cmd
    assert "--OSL" in splitter_cmd and "1024" in splitter_cmd, splitter_cmd

    assert all(
        not (c and "TraceLens_generate_perf_report_pytorch_inference" in str(c[0]) and "--profile_json_path" in c)
        for c in captured
    ), f"perf-report CSV fallback must not run; analysis.md is the single source of truth. cmds={captured}"
    out = capsys.readouterr().out
    result = _json.loads(out)
    assert result["status"] == "failed"
    assert "trace_split_no_steady_state" in result["error"]


# Splitter must receive --R (from --split-r or $RANDOM_RANGE_RATIO) so mixed-window
# selection uses the analytic PD ratio instead of an empirical heuristic.
def _drive_main_capturing_subprocess(tmp_path, extra_argv, env_overrides=None, trace_factory=None):
    """Helper: stage a TraceLens-ish tree, stub subprocess.run, drive tla.main() once, return captured argvs.

    ``trace_factory`` builds the input trace from the staging directory when the
    default single-kernel stub is not enough (for example a step-annotated trace
    the pretrimmer can act on); it returns the path it wrote.
    """
    import gzip
    import json as _json
    import os as _os
    from unittest.mock import patch

    tl_root = tmp_path / "TraceLens-internal"
    skill_dir = tl_root / "TraceLens" / "Agent" / "Analysis" / "skills" / "analysis-orchestrator"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("stub")
    workspace = tmp_path / "ws"
    workspace.mkdir()
    capture = tmp_path / "capture_traces"
    capture.mkdir()
    if trace_factory is not None:
        trace = trace_factory(tmp_path)
    else:
        trace = tmp_path / "trace.json.gz"
        with gzip.open(trace, "wt") as f:
            _json.dump(
                {
                    "traceEvents": [
                        {"cat": "kernel", "name": "void some_real_kernel<...>", "dur": 5.0},
                    ]
                },
                f,
            )

    captured: list[list[str]] = []

    class _Result:
        def __init__(self, returncode=0, stdout=""):
            self.returncode = returncode
            self.stdout = stdout

    def fake_run(cmd, *_a, **_kw):
        captured.append(list(cmd))
        return _Result(returncode=0, stdout="ok")

    argv = [
        "tracelens_analysis.py",
        "--trace-input",
        str(trace),
        "--workspace-path",
        str(workspace),
        "--tracelens-root",
        str(tl_root),
        "--target-platform",
        "MI300X",
        "--budget-minutes",
        "1",
        "--no-llm-orchestrator",
        "--capture-folder",
        str(capture),
        *extra_argv,
    ]

    env_backup = dict(_os.environ)
    try:
        for k, v in (env_overrides or {}).items():
            if v is None:
                _os.environ.pop(k, None)
            else:
                _os.environ[k] = v
        with patch.object(tla.subprocess, "run", side_effect=fake_run), patch.object(tla.sys, "argv", argv):
            try:
                tla.main()
            except SystemExit:
                # main() exits via SystemExit on success; swallow it in-test.
                pass
    finally:
        _os.environ.clear()
        _os.environ.update(env_backup)

    return captured, trace


def _find_splitter_cmd(captured):
    return next(
        (c for c in captured if any("split_inference_trace_annotation" in str(p) for p in c)),
        None,
    )


def _drive_main_over_capture_dir(tmp_path, trace_dir, extra_argv=None):
    """Drive tla.main() with a capture *directory* and capture subprocess argvs.

    A sibling of :func:`_drive_main_capturing_subprocess`, which always passes a
    single file. Multi-rank selection only shows up when discovery has more than
    one candidate to choose from.

    ``extra_argv`` appends CLI flags, which is how the ``--skip-split`` route
    that scriptable workloads actually take gets exercised.
    """
    import os as _os
    from unittest.mock import patch

    tl_root = tmp_path / "TraceLens-internal"
    skill_dir = tl_root / "TraceLens" / "Agent" / "Analysis" / "skills" / "analysis-orchestrator"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("stub")
    workspace = tmp_path / "ws"
    workspace.mkdir()

    captured: list[list[str]] = []

    class _Result:
        def __init__(self, returncode=0, stdout=""):
            self.returncode = returncode
            self.stdout = stdout

    def fake_run(cmd, *_a, **_kw):
        captured.append(list(cmd))
        return _Result(returncode=0, stdout="ok")

    argv = [
        "tracelens_analysis.py",
        "--trace-input",
        str(trace_dir),
        "--workspace-path",
        str(workspace),
        "--tracelens-root",
        str(tl_root),
        "--target-platform",
        "MI300X",
        "--budget-minutes",
        "1",
        "--no-llm-orchestrator",
    ]
    argv += list(extra_argv or [])

    env_backup = dict(_os.environ)
    try:
        with patch.object(tla.subprocess, "run", side_effect=fake_run), patch.object(tla.sys, "argv", argv):
            try:
                tla.main()
            except SystemExit:
                pass
    finally:
        _os.environ.clear()
        _os.environ.update(env_backup)
    return captured


def _rank_trace(path: Path, kernels: int, cpu_events: int = 0) -> Path:
    """Write a rank trace with the given number of GPU kernels.

    ``cpu_events`` pads with host-side events, which is what a CPU-only capture
    actually looks like: a large file with no kernels in it.
    """
    events = [{"cat": "kernel", "name": "void real_kernel<...>", "dur": 5.0} for _ in range(kernels)]
    events += [{"cat": "cpu_op", "name": f"aten::some_host_op_{i}", "dur": 1.0} for i in range(cpu_events)]
    with gzip.open(path, "wt") as fh:
        json.dump({"traceEvents": events}, fh)
    return path


def test_analysis_input_follows_the_candidate_that_passed_the_preflight(tmp_path):
    """A CPU-only leading rank must not be what gets analysed.

    The preflight probes several candidates, so it can pass on rank_1 while
    rank_0 leads discovery. If the analysis kept using the first candidate, the
    check would clear a capture on one rank's evidence and then hand TraceLens
    the empty one -- quieter than the failure it replaced, and worse.
    """
    trace_dir = tmp_path / "torch_trace"
    trace_dir.mkdir()
    # A CPU-only rank is a big file with no kernels in it, so size ordering puts
    # it first on merit. Ordering cannot help here; only the promotion can.
    empty = _rank_trace(trace_dir / "rank_0.trace.json.gz", kernels=0, cpu_events=400)
    populated = _rank_trace(trace_dir / "rank_1.trace.json.gz", kernels=12)
    assert empty.stat().st_size > populated.stat().st_size

    _kind, traces = tla.discover_trace_inputs(trace_dir)
    assert traces[0] == empty

    captured = _drive_main_over_capture_dir(tmp_path, trace_dir)

    splitter = _find_splitter_cmd(captured)
    assert splitter is not None, "the splitter should have been invoked"
    args = [str(p) for p in splitter]
    assert str(populated) in args
    assert str(empty) not in args


def test_analysis_input_is_left_alone_when_the_first_candidate_has_kernels(tmp_path):
    """No promotion when the leading candidate is already the right one."""
    trace_dir = tmp_path / "torch_trace"
    trace_dir.mkdir()
    first = _rank_trace(trace_dir / "rank_0.trace.json.gz", kernels=9)
    _rank_trace(trace_dir / "rank_1.trace.json.gz", kernels=9)

    captured = _drive_main_over_capture_dir(tmp_path, trace_dir)

    splitter = _find_splitter_cmd(captured)
    assert splitter is not None
    assert str(first) in [str(p) for p in splitter]


def _capture_sidecar(path: Path, kernels: int = 2) -> Path:
    """Write a CUDA-graph capture sidecar in its production shape.

    ``kernels`` defaults to 2 on purpose. A capture records the graph being
    built, so a couple of launches still reach the device while the rest of the
    file is host-side call tree — the run this guards against had 2 kernels in
    1.49M events. A sidecar with *zero* kernels would already be stopped by the
    CPU-only preflight; two is the count that gets through it.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    return _rank_trace(path, kernels=kernels, cpu_events=200)


def test_capture_only_input_is_rejected_before_the_splitter(tmp_path, capsys):
    """A directory holding nothing but graph-capture sidecars must not analyse.

    The sidecars carry kernels, so the CPU-only preflight passes them and the
    splitter is handed a file with no iteration loop in it. It then reports
    ``trace_split_no_steady_state``, which reads as "the profiled window was too
    short" and sends the next person to lengthen a capture that was never a
    workload timeline. The rejection has to name the real cause instead.
    """
    trace_dir = tmp_path / "torch_trace"
    capture = trace_dir / "capture_traces"
    for bs in (2, 4, 8):
        for rank in (0, 1):
            _capture_sidecar(capture / f"bs_{bs}_rank{rank}.json.gz")

    # Precondition: this is exactly the shape the old check waved through.
    _kind, traces = tla.discover_trace_inputs(trace_dir)
    assert len(traces) == 6
    assert tla._count_kernels_if_readable(traces[0])[1] > 0

    captured = _drive_main_over_capture_dir(tmp_path, trace_dir)

    assert _find_splitter_cmd(captured) is None, "the splitter must never be invoked"
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "failed"
    assert "trace_input_capture_only" in result["error"]
    # The message has to point at the profile, not at the splitter or the window.
    assert "re-profile" in result["error"]


def test_capture_sidecars_beside_a_real_trace_still_analyse(tmp_path):
    """The healthy layout must be unaffected.

    A normal profile writes its annotated trace *beside* the capture sidecars,
    which is why the rejection tests ``all`` and not ``any``. Getting this
    backwards would disable roofline for every well-formed profile.
    """
    trace_dir = tmp_path / "torch_trace"
    trace_dir.mkdir()
    real = _rank_trace(trace_dir / "rank_0.trace.json.gz", kernels=12)
    _capture_sidecar(trace_dir / "capture_traces" / "bs_2_rank0.json.gz")

    captured = _drive_main_over_capture_dir(tmp_path, trace_dir)

    splitter = _find_splitter_cmd(captured)
    assert splitter is not None, "a real trace beside sidecars must still analyse"
    assert str(real) in [str(p) for p in splitter]


def test_lone_capture_sidecar_file_is_rejected_by_name(tmp_path, capsys):
    """``--trace-input`` pointed straight at one sidecar is the same mistake."""
    sidecar = _capture_sidecar(tmp_path / "torch_trace" / "bs_16_rank3.json.gz")

    captured = _drive_main_over_capture_dir(tmp_path, sidecar)

    assert _find_splitter_cmd(captured) is None
    assert "trace_input_capture_only" in json.loads(capsys.readouterr().out)["error"]


def test_capture_classification_ignores_an_unrelated_ancestor_dir(tmp_path):
    """An ancestor named ``capture_traces`` must not condemn a real trace.

    Paths arrive absolute, so an unbounded component test would reject every
    candidate whenever the session happened to live under a directory of that
    name — pointing ``--trace-input`` inside a previous capture, say.
    """
    root = tmp_path / "capture_traces" / "torch_trace"
    root.mkdir(parents=True)
    real = _rank_trace(root / "rank_0.trace.json.gz", kernels=7)

    assert not tla._is_capture_fragment(real, tla._capture_classification_root(root))
    # A genuine sidecar below the root is still caught.
    nested = _capture_sidecar(root / "capture_traces" / "bs_2_rank0.json.gz")
    assert tla._is_capture_fragment(nested, tla._capture_classification_root(root))


def test_bare_bs_prefix_is_not_enough_to_condemn_a_trace(tmp_path):
    """``bs_`` without a batch number must not classify as a sidecar.

    The classifier decides whether an input is rejected outright, not just how
    it sorts, so matching three characters of a filename is too cheap a reason
    to throw a real trace away. The sidecar shapes carry a batch number.
    """
    root = tmp_path / "torch_trace"
    root.mkdir()
    classify_root = tla._capture_classification_root(root)

    assert tla._is_capture_fragment(root / "bs_2_rank0.json.gz", classify_root)
    assert tla._is_capture_fragment(root / "bs_512.json.gz", classify_root)
    assert tla._is_capture_fragment(root / "graph_capture_rank_0.pt.trace.json.gz", classify_root)
    assert not tla._is_capture_fragment(root / "bs_baseline.trace.json.gz", classify_root)
    assert not tla._is_capture_fragment(root / "rank_0.trace.json.gz", classify_root)


def test_capture_sidecars_sort_behind_a_real_trace(tmp_path):
    """Discovery ordering must keep sidecars behind the annotated trace.

    The sort key and the preflight now share one classifier, so this pins the
    ordering half: a sidecar that is *larger* than the real trace still sorts
    behind it.
    """
    trace_dir = tmp_path / "torch_trace"
    trace_dir.mkdir()
    real = _rank_trace(trace_dir / "rank_0.trace.json.gz", kernels=4)
    big_sidecar = _capture_sidecar(trace_dir / "capture_traces" / "bs_2_rank0.json.gz", kernels=2)
    assert big_sidecar.stat().st_size > real.stat().st_size

    _kind, traces = tla.discover_trace_inputs(trace_dir)
    assert traces[0] == real


@pytest.mark.parametrize(
    "relpath, expected",
    [
        # Hyperloom-patched SGLang.
        ("capture_traces/bs_2_rank0.json.gz", True),
        ("capture_traces/bs_64_rank7.json.gz", True),
        # Unpatched SGLang: neither the directory nor the filename matches the
        # patched shape, and the ``cuda_`` prefix defeats a start-anchored test.
        ("graph_capture_profile/cuda_graph_capture-DecodeCudaGraphRunner-TP-3.json.gz", True),
        # vLLM.
        ("graph_capture_rank0.json.gz", True),
        # Workload traces must survive all of the above.
        ("1786734684.9990146-TP-0.trace.json.gz", False),
        ("rank_0.trace.json.gz", False),
        ("merged-annotated.trace.json.gz", False),
    ],
)
def test_capture_classifier_covers_every_observed_profile_layout(tmp_path, relpath, expected):
    """The classifier keys on shape, so a new layout does not slip through.

    Each entry is a layout a production profile actually wrote. An exact-name
    whitelist passed the first two and missed the SGLang-without-patch one.
    """
    path = tmp_path / relpath
    assert tla._is_capture_fragment(path, tmp_path) is expected


def test_capture_dir_match_is_anchored_so_a_descriptive_name_is_safe(tmp_path):
    """``graph_capture`` anchors in a directory name, unlike in a filename.

    A directory is named for what it holds, so an unanchored token would also
    condemn ``torch_profiler_with_graph_capture/`` -- and the capture-only
    preflight is an ``all(...)``, so one false positive rejects the whole input.
    """
    safe = tmp_path / "torch_profiler_with_graph_capture" / "rank_0.trace.json.gz"
    assert tla._is_capture_fragment(safe, tmp_path) is False
    for capture_dir in ("graph_capture", "graph_capture_profile", "capture_traces"):
        assert tla._is_capture_fragment(tmp_path / capture_dir / "rank_0.trace.json.gz", tmp_path) is True


def test_discover_capture_folder_finds_the_unpatched_sglang_layout(tmp_path):
    """The capture folder must be locatable, not merely demoted during ranking.

    Ranking keeps the sidecars out of the analysis input; discovery is what
    hands them to TraceLens as ``--capture_folder``. Two hard-coded names meant
    a run could pick the right workload trace and still lose its graph-capture
    input.
    """
    trace_dir = tmp_path / "torch_trace"
    (trace_dir / "graph_capture_profile").mkdir(parents=True)
    real = _rank_trace(trace_dir / "1786735404.4274018-TP-0.trace.json.gz", kernels=4)
    assert tlr.discover_capture_folder(trace_dir, [real]) == trace_dir / "graph_capture_profile"


def test_discover_capture_folder_ignores_a_descriptive_sibling(tmp_path):
    trace_dir = tmp_path / "torch_trace"
    (trace_dir / "torch_profiler_with_graph_capture").mkdir(parents=True)
    real = _rank_trace(trace_dir / "rank_0.trace.json.gz", kernels=4)
    assert tlr.discover_capture_folder(trace_dir, [real]) is None


def test_unpatched_sglang_capture_sorts_behind_the_workload_trace(tmp_path):
    """GLM-5.2 regression: a 103 MB capture must not outrank a 20 MB trace.

    ``graph_capture_profile/`` matched no known capture shape, so its files
    shared the default bucket with the workload traces, where the tie-break is
    descending size. The capture is the larger file, so it led discovery, the
    probe stopped on it, and the splitter cut a bs=1/conc=1 graph-capture window
    that carried zero GPU events. The whole kernel phase was lost to it.
    """
    trace_dir = tmp_path / "torch_trace"
    trace_dir.mkdir()
    real = _rank_trace(trace_dir / "1786734684.9990146-TP-0.trace.json.gz", kernels=8)
    big_capture = _capture_sidecar(
        trace_dir / "graph_capture_profile" / "cuda_graph_capture-DecodeCudaGraphRunner-TP-3.json.gz",
        kernels=2,
    )
    assert big_capture.stat().st_size > real.stat().st_size

    _kind, traces = tla.discover_trace_inputs(trace_dir)
    assert traces[0] == real


def test_capture_under_an_ancestor_named_trace_split_still_orders_correctly(tmp_path):
    """An ancestor directory name must not flatten the whole ranking.

    ``--trace-input`` is resolved to an absolute path, so a ``trace_split``
    component is checked against every ancestor unless the test is anchored at
    the capture root. Pointing at a capture that happens to sit below such a
    directory would otherwise demote every candidate into the same bucket, at
    which point ordering falls back to filename and the original bug is exactly
    reproduced -- from nothing but a coincidence of naming.
    """
    trace_dir = tmp_path / "trace_split" / "run" / "torch_trace"
    trace_dir.mkdir(parents=True)
    fragment = _rank_trace(trace_dir / "aaa_trace_annotation_iteration_1.json.gz", kernels=0)
    capture = _rank_trace(trace_dir / "zzz_rank_0.trace.json.gz", kernels=30)

    _kind, traces = tla.discover_trace_inputs(trace_dir)

    assert traces[0] == capture, (
        "the real capture must lead despite the ancestor directory name "
        f"and despite sorting last alphabetically; got {[p.name for p in traces]}"
    )
    assert traces.index(fragment) > traces.index(capture)


def test_unreadable_leading_candidate_is_not_reported_as_cpu_only(tmp_path):
    """A corrupt trace and a CPU-only trace must not read as the same finding.

    Both counted as zero kernels before, so a truncated rank_0 was described as
    having "no GPU kernel events" -- sending the next reader to the profiler for
    a file problem. That is the same misdirection this change was written to
    remove, so it should not be reintroduced by the promotion that fixes it.
    """
    trace_dir = tmp_path / "torch_trace"
    trace_dir.mkdir()
    corrupt = trace_dir / "rank_0.trace.json.gz"
    corrupt.write_bytes(b"\x1f\x8b" + b"\x00" * 4000)  # gzip magic, truncated body
    _rank_trace(trace_dir / "rank_1.trace.json.gz", kernels=7)

    readable, count = tla._count_kernels_if_readable(corrupt)

    assert readable is False, "a truncated gzip must report as unreadable"
    assert count == 0
    populated_readable, populated_count = tla._count_kernels_if_readable(trace_dir / "rank_1.trace.json.gz")
    assert populated_readable is True
    assert populated_count == 7


def test_promotion_is_recorded_as_a_trace_health_warning(tmp_path, capsys):
    """A run that switched its own input has to be explicable afterwards.

    A CLI log line is not a contract: session breakdown and roofline snapshot
    read ``trace_health_warnings`` and the artifact map, and neither recorded
    which of the discovered traces was actually analysed. Without this, a
    silently promoted run is indistinguishable downstream from one that used the
    file discovery reported.
    """
    trace_dir = tmp_path / "torch_trace"
    trace_dir.mkdir()
    _rank_trace(trace_dir / "rank_0.trace.json.gz", kernels=0, cpu_events=400)
    populated = _rank_trace(trace_dir / "rank_1.trace.json.gz", kernels=12)

    _drive_main_over_capture_dir(tmp_path, trace_dir)

    result = json.loads(capsys.readouterr().out)
    promoted = [
        w for w in (result.get("trace_health_warnings") or []) if w.get("code") == "trace_analysis_input_promoted"
    ]
    assert promoted, (
        f"promotion was not surfaced in trace_health_warnings; warnings seen: {result.get('trace_health_warnings')}"
    )
    assert promoted[0]["analysed"] == populated.name
    assert promoted[0]["leading_candidate"] == "rank_0.trace.json.gz"
    # The probe record is what distinguishes "rank_0 had no kernels" from
    # "rank_0 could not be read", which are different problems.
    assert "rank_0.trace.json.gz=0" in promoted[0]["probed"]


def test_no_promotion_warning_when_the_leading_candidate_is_used(tmp_path, capsys):
    """The warning must stay absent on the ordinary path.

    An informational warning that fires on every healthy run is noise, and noise
    in ``trace_health_warnings`` costs the Coordinator the signal.
    """
    trace_dir = tmp_path / "torch_trace"
    trace_dir.mkdir()
    _rank_trace(trace_dir / "rank_0.trace.json.gz", kernels=9)
    _rank_trace(trace_dir / "rank_1.trace.json.gz", kernels=9)

    _drive_main_over_capture_dir(tmp_path, trace_dir)

    result = json.loads(capsys.readouterr().out)
    codes = [w.get("code") for w in (result.get("trace_health_warnings") or [])]
    assert "trace_analysis_input_promoted" not in codes


def test_194_3_splitter_receives_R_from_cli_arg(tmp_path):
    """`--split-r 0.5` must produce `--R 0.5` on the splitter argv (fractional ratios survive verbatim)."""
    captured, _ = _drive_main_capturing_subprocess(
        tmp_path,
        extra_argv=[
            "--split-conc",
            "32",
            "--split-osl",
            "1024",
            "--split-r",
            "0.5",
        ],
        env_overrides={"RANDOM_RANGE_RATIO": None},
    )
    splitter_cmd = _find_splitter_cmd(captured)
    assert splitter_cmd is not None, f"splitter never invoked; cmds={captured}"
    assert "--R" in splitter_cmd, splitter_cmd
    # The value must immediately follow --R.
    assert splitter_cmd[splitter_cmd.index("--R") + 1] == "0.5", splitter_cmd


def test_194_3_splitter_receives_R_from_random_range_ratio_env(tmp_path):
    """Without --split-r, the wrapper falls back to RANDOM_RANGE_RATIO
    env — the same variable Hyperloom propagates from the YAML config
    into every Magpie subprocess. Locks down the env→splitter seam."""
    captured, _ = _drive_main_capturing_subprocess(
        tmp_path,
        extra_argv=["--split-conc", "32", "--split-osl", "1024"],
        env_overrides={"RANDOM_RANGE_RATIO": "0.8"},
    )
    splitter_cmd = _find_splitter_cmd(captured)
    assert splitter_cmd is not None, f"splitter never invoked; cmds={captured}"
    assert "--R" in splitter_cmd, splitter_cmd
    assert splitter_cmd[splitter_cmd.index("--R") + 1] == "0.8", splitter_cmd


def test_194_3_splitter_omits_R_when_unset(tmp_path):
    """No --split-r and no RANDOM_RANGE_RATIO env → the splitter must
    not see --R. The splitter's built-in default (`R=None`) keeps the
    old heuristic path live for legacy traces that pre-date the
    skill-aligned formulas."""
    captured, _ = _drive_main_capturing_subprocess(
        tmp_path,
        extra_argv=["--split-conc", "32", "--split-osl", "1024"],
        env_overrides={"RANDOM_RANGE_RATIO": None, "TRACELENS_SPLIT_R": None},
    )
    splitter_cmd = _find_splitter_cmd(captured)
    assert splitter_cmd is not None, f"splitter never invoked; cmds={captured}"
    assert "--R" not in splitter_cmd, splitter_cmd


def test_194_3_splitter_ignores_non_numeric_R(tmp_path):
    """A malformed env --R value must be dropped (not propagated, which would argparse.error the splitter)."""
    captured, _ = _drive_main_capturing_subprocess(
        tmp_path,
        extra_argv=["--split-conc", "32", "--split-osl", "1024"],
        env_overrides={"RANDOM_RANGE_RATIO": "not-a-float"},
    )
    splitter_cmd = _find_splitter_cmd(captured)
    assert splitter_cmd is not None, f"splitter never invoked; cmds={captured}"
    assert "--R" not in splitter_cmd, splitter_cmd


# parse_analysis_md — TraceLens final-report contract
_FIXTURE_LLAMA70B_ANALYSIS_MD = (
    Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "tracelens_v03_llama70b_analysis.md"
)


def test_parse_analysis_md_llama70b_fixture_yields_21_compute_candidates():
    """Round-trip the official Llama-3 70B golden analysis.md fixture into 21 compute candidates."""
    cands = tlr.parse_analysis_md(_FIXTURE_LLAMA70B_ANALYSIS_MD, top_k=50)
    assert len(cands) == 21, (
        f"expected 21 candidates (18 GEMM + 2 SDPA_fwd + 1 SDPA_bwd) from the fixture; got {len(cands)}"
    )

    by_cat = {}
    for c in cands:
        by_cat.setdefault(c["tracelens_category"], []).append(c)
    assert len(by_cat["gemm"]) == 18
    assert len(by_cat["sdpa_fwd"]) == 2
    assert len(by_cat["sdpa_bwd"]) == 1

    p1_first = cands[0]
    assert p1_first["name"] == "aten::mm"
    assert p1_first["tracelens_category"] == "gemm"
    assert p1_first["tracelens_pitem_rank"] == 1
    assert p1_first["library"] == "Tensile"
    assert p1_first["bound_type"] == "compute-bound"
    # Time (ms) -> duration_us; first row of P1 = 7607.463 ms.
    assert abs(p1_first["duration_us"] - 7607463.0) < 1.0
    assert p1_first["call_count"] == 320
    assert abs(p1_first["percent_of_total"] - 13.42) < 0.001
    assert abs(p1_first["efficiency_percent"] - 68.74) < 0.001
    assert p1_first["efficiency_peak_value"] == 708.0
    assert "TFLOPS" in p1_first["efficiency_peak_unit"]
    assert p1_first["impact_score"] == 15.12  # mid value from p_item marker
    # Args is "<br>"-joined upstream; parser must normalise to a list of
    # whitespace-trimmed shape strings without losing entries.
    assert p1_first["shapes"] == [
        "(24576,8192) bf16",
        "(8192,28672) bf16",
        "(24576,28672) bf16",
    ]
    # Kernel Path is "—" for every row in this fixture; parser must keep the
    # field as empty string (not the dash) so downstream "no source path"
    # checks remain truthy.
    assert p1_first["source_file"] == ""

    # Last candidate is the lone SDPA_bwd row (P3 in the report).
    p3_only = cands[-1]
    assert p3_only["name"] == "flash_attn::_flash_attn_backward"
    assert p3_only["tracelens_category"] == "sdpa_bwd"
    assert p3_only["tracelens_pitem_rank"] == 3
    assert p3_only["library"] == "CK"
    assert p3_only["call_count"] == 160


def test_parse_analysis_md_returns_empty_when_no_detailed_analysis(tmp_path):
    """Empty Detailed Analysis -> 0 candidates, so caller can fall back."""
    md = tmp_path / "analysis.md"
    md.write_text(
        "# Stub\n\n## Compute Kernel Optimizations\n\n"
        "✅ No actionable per-category compute-kernel bottlenecks were promoted.\n\n"
        "## Detailed Analysis\n\n### Compute Kernel Insights\n\n"
        "_No compute-kernel reasoning candidates were promoted._\n",
        encoding="utf-8",
    )
    assert tlr.parse_analysis_md(md, top_k=10) == []


def test_parse_analysis_md_missing_file_returns_empty(tmp_path):
    """Non-existent report -> 0 candidates (callers fall back, never raise)."""
    assert tlr.parse_analysis_md(tmp_path / "nope.md", top_k=10) == []


def test_parse_analysis_md_top_k_caps_total_rows(tmp_path):
    """top_k caps the per-row total across all P-items, not per category."""
    cands = tlr.parse_analysis_md(_FIXTURE_LLAMA70B_ANALYSIS_MD, top_k=5)
    assert len(cands) == 5
    # First 5 rows of the fixture are all P1 GEMMs.
    assert all(c["tracelens_pitem_rank"] == 1 for c in cands)


# Filter for GEAK based on budget (Higher P-item, Lower Efficiency)
def _write_two_pitem_analysis_md(md: Path) -> None:
    md.write_text(
        "<!-- impact-begin kind=p_item category=gemm mid=4.0 low=2.0 high=8.0 -->\n"
        "<!-- impact-begin kind=p_item category=sdpa_fwd mid=1.5 low=0.5 high=3.0 -->\n"
        "\n"
        "## Detailed Analysis\n\n### Compute Kernel Insights\n\n"
        "<!-- reasoning-candidate tier=compute rank=1 -->\n"
        "#### 🔴 P1: GEMM cluster (Tensile)\n\n"
        "**Identification:** stub identification\n"
        "**Data:**\n"
        "| Operation | Args | Kernel Path | Time (ms) | %E2E | Count | "
        "FLOPS/Byte | Efficiency | Bound |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
        "| high_eff_gemm | (1,2) bf16 | — | 1.0 | 5 | 10 | 1000 | "
        "80% of 708 TFLOPS | compute-bound |\n"
        "| low_eff_gemm | (1,2) bf16 | — | 1.0 | 5 | 10 | 1000 | "
        "5% of 708 TFLOPS | compute-bound |\n"
        "| unknown_eff_gemm | (1,2) bf16 | — | 1.0 | 5 | 10 | 1000 | "
        " | compute-bound |\n"
        "| mid_eff_gemm | (1,2) bf16 | — | 1.0 | 5 | 10 | 1000 | "
        "40% of 708 TFLOPS | compute-bound |\n"
        "**Reasoning for Slowdown:** stub reasoning\n"
        "**Resolution:** stub resolution\n"
        "**Impact estimate:**\n"
        "Low end: 1.0 ms savings (0.1% E2E)\n"
        "High end: 2.0 ms savings (0.2% E2E)\n"
        "\n"
        "<!-- reasoning-candidate tier=compute rank=2 -->\n"
        "#### 🟡 P2: SDPA (CK)\n\n"
        "**Identification:** stub identification\n"
        "**Data:**\n"
        "| Operation | Args | Kernel Path | Time (ms) | %E2E | Count | "
        "FLOPS/Byte | Efficiency | Bound |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
        "| p2_sdpa | (1,2) bf16 | — | 1.0 | 5 | 10 | 1000 | "
        "10% of 708 TFLOPS | compute-bound |\n"
        "**Reasoning for Slowdown:** stub reasoning\n"
        "**Resolution:** stub resolution\n"
        "**Impact estimate:**\n"
        "Low end: 1.0 ms savings (0.1% E2E)\n"
        "High end: 2.0 ms savings (0.2% E2E)\n",
        encoding="utf-8",
    )


def test_parse_analysis_md_sorts_within_pitem_by_lower_efficiency(tmp_path):
    """Within a P-item, lower-efficiency rows sort first (survive top_k); P1 before P2 across items."""
    md = tmp_path / "analysis.md"
    _write_two_pitem_analysis_md(md)

    cands = tlr.parse_analysis_md(md, top_k=10)
    names = [c["name"] for c in cands]
    assert names == [
        # P1 rows sorted ascending by efficiency:
        "low_eff_gemm",
        "mid_eff_gemm",
        "high_eff_gemm",
        # Unknown / 0.0 efficiency lands last within the P-item:
        "unknown_eff_gemm",
        # P2 still after every P1 row regardless of efficiency:
        "p2_sdpa",
    ]


def test_parse_analysis_md_efficiency_sort_respects_top_k_budget(tmp_path):
    """Budget cap: top_k keeps the lowest-efficiency rows within a P-item."""
    md = tmp_path / "analysis.md"
    _write_two_pitem_analysis_md(md)

    cands = tlr.parse_analysis_md(md, top_k=2)
    names = [c["name"] for c in cands]
    assert names == ["low_eff_gemm", "mid_eff_gemm"], (
        "top_k=2 must keep the two lowest-efficiency P1 rows; the "
        "high-efficiency / unknown rows must be dropped before any P2 row"
    )


# normalize_upstream_category — TraceLens orchestrator_prepare.py enum
@pytest.mark.parametrize(
    "raw,expected",
    [
        ("gemm", "GEMM"),
        ("groupedgemm_fwd", "GEMM"),
        ("groupedgemm_bwd", "GEMM"),
        ("moe_fused", "MoE"),
        ("moe_unfused", "MoE"),
        ("sdpa_fwd", "SDPA"),
        ("sdpa_bwd", "SDPA"),
        ("inferenceattention", "SDPA"),
        ("rmsnorm", "LayerNorm"),
        ("norm_fwd", "LayerNorm"),
        ("norm_bwd", "LayerNorm"),
        ("convolution", "Convolution"),
        ("conv_fwd", "Convolution"),
        ("conv_bwd", "Convolution"),
        ("triton", "Triton"),
        ("elementwise", "Elementwise"),
        ("reduce", "Reduction"),
        ("cpu_idle", "Other"),
        ("other", "Other"),
        # Mixed case + whitespace + alt separators must normalise the same way.
        ("  GEMM  ", "GEMM"),
        ("Sdpa-Fwd", "SDPA"),
        ("MoE/Fused", "MoE"),
    ],
)
def test_normalize_upstream_category_matches_orchestrator_prepare_enum(raw, expected):
    """Mirror TraceLens-internal CATEGORY_SKILL_MAP keys exactly."""
    assert tlr.normalize_upstream_category(raw) == expected


def test_normalize_upstream_category_passes_through_unknown():
    """Unknown categories are surfaced verbatim — never silently coerced."""
    assert tlr.normalize_upstream_category("brand_new_skill") == "brand_new_skill"


def test_normalize_upstream_category_empty_returns_unknown():
    assert tlr.normalize_upstream_category("") == "unknown"


def test_derive_kernel_category_uses_upstream_enum_when_present():
    """When TraceLens tags a candidate, GEAK label must come from upstream map."""
    for raw, expected in [
        ("gemm", "GEMM"),
        ("groupedgemm_fwd", "GEMM"),
        ("inferenceattention", "SDPA"),
        ("moe_fused", "MoE"),
        ("rmsnorm", "LayerNorm"),
    ]:
        cand = {"tracelens_category": raw, "name": "ignored_when_cat_present"}
        assert tla.derive_kernel_category(cand) == expected, raw


def test_derive_kernel_category_falls_back_to_name_heuristic():
    """Raw-trace fallback path has no tracelens_category; heuristics still apply."""
    assert tla.derive_kernel_category({"name": "rocblas_gemm_kernel"}) == "GEMM"
    assert tla.derive_kernel_category({"name": "fmha_fwd_kernel"}) == "SDPA"
    assert tla.derive_kernel_category({"name": "rmsnorm_fused"}) == "LayerNorm"
    assert tla.derive_kernel_category({"name": "totally_unknown_op"}) == "unknown"


# _extract_pitem_prose extracts Reasoning / Resolution / Impact.
_SYNTHETIC_PITEM_BODY = """\
#### 🔴 P1: RMSNorm fused with quantization (Triton)

**Identification:** Four `aiter::rmsnorm_quant` operations were flagged as memory-bound with efficiencies of 0.88%-4.31% against peak HBM bandwidth of 5.3 TB/s. (source: `rmsnorm_metrics.json` → `operations[].efficiency.efficiency_percent`)

**Data:**

| Operation | Args | Kernel Path | Time (ms) | %E2E | Count | FLOPS/Byte | Efficiency | Bound |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rmsnorm_quant | (8,4096) bf16 | aiter/ops/rmsnorm.py(76): rmsnorm | 123.4 | 4.2 | 64 | 0.5 | 30% of 5.3 TB/s | memory-bound |

**Reasoning for Slowdown:**

Memory-bound elementwise kernel; HBM bandwidth saturated by the bf16 load + fp8 quant store pair.

**Resolution:**

Fuse RMSNorm with the immediately-following GEMM to amortize global loads, or rewrite as a single-pass Triton kernel with `tl.store(..., mask=)`.

**Impact estimate:**

Low end (baseline shapes): 12.5 ms savings (3.2% E2E). High end (peak decode batch): 40.0 ms savings (10.4% E2E).
"""


def test_extract_pitem_prose_pulls_all_sections():
    prose = tlr._extract_pitem_prose(_SYNTHETIC_PITEM_BODY)
    assert "Four `aiter::rmsnorm_quant`" in prose["identification"]
    assert "rmsnorm_metrics.json" in prose["identification"]
    assert "Memory-bound elementwise kernel" in prose["reasoning_for_slowdown"]
    assert "HBM bandwidth saturated" in prose["reasoning_for_slowdown"]
    assert "Fuse RMSNorm" in prose["resolution"]
    assert "amortize global loads" in prose["resolution"]
    assert prose["impact_low_ms"] == 12.5
    assert prose["impact_low_e2e_pct"] == 3.2
    assert prose["impact_high_ms"] == 40.0
    assert prose["impact_high_e2e_pct"] == 10.4


def test_extract_pitem_prose_identification_stops_at_data_marker():
    """Identification ends at ``**Data:**`` — must not leak the 9-column table into the field."""
    body = (
        "**Identification:** Three ops flagged at 0.5% efficiency. "
        "(source: gemm_metrics.json)\n\n"
        "**Data:**\n\n| Op | Args | ... |\n\n"
        "**Reasoning for Slowdown:**\nMemory-bound.\n"
    )
    prose = tlr._extract_pitem_prose(body)
    assert prose["identification"].startswith("Three ops flagged")
    assert "gemm_metrics.json" in prose["identification"]
    assert "| Op |" not in prose["identification"], (
        "Identification leaked into the Data table — end-marker order is wrong"
    )
    assert "Memory-bound" not in prose["identification"]


def test_extract_pitem_prose_returns_empty_strings_when_markers_absent():
    """Bodies without the four labels still return the full dict shape (key presence guaranteed)."""
    prose = tlr._extract_pitem_prose("**Data:**\n| ... | ... |\n")
    assert prose["identification"] == ""
    assert prose["reasoning_for_slowdown"] == ""
    assert prose["resolution"] == ""
    assert prose["impact_low_ms"] == 0.0
    assert prose["impact_low_e2e_pct"] == 0.0
    assert prose["impact_high_ms"] == 0.0
    assert prose["impact_high_e2e_pct"] == 0.0


def test_extract_pitem_prose_reasoning_stops_at_resolution_marker():
    """Reasoning must not leak into Resolution when both are present."""
    body = (
        "**Reasoning for Slowdown:**\nFirst paragraph.\n\n"
        "**Resolution:**\nSecond paragraph.\n\n"
        "**Impact estimate:**\nLow end: 1.0 ms savings (0.5% E2E).\n"
        "High end: 2.0 ms savings (1.0% E2E).\n"
    )
    prose = tlr._extract_pitem_prose(body)
    assert prose["reasoning_for_slowdown"] == "First paragraph."
    assert prose["resolution"] == "Second paragraph."
    assert prose["impact_low_ms"] == 1.0
    assert prose["impact_high_ms"] == 2.0


def test_extract_between_returns_empty_when_start_marker_missing():
    """Defensive guard: missing start marker → empty, never raises."""
    assert tlr._extract_between("body", "**Missing:**", ("**End:**",)) == ""


def test_parse_analysis_md_attaches_prose_from_fixture():
    """Every parsed LLama70B fixture candidate carries non-empty prose fields from its parent P-item block."""
    cands = tlr.parse_analysis_md(_FIXTURE_LLAMA70B_ANALYSIS_MD, top_k=50)
    assert cands, "fixture must produce at least one candidate"
    # All 21 fixture candidates share P-item prose with their group.
    for c in cands:
        assert "identification" in c
        assert "reasoning_for_slowdown" in c
        assert "resolution" in c
        assert "impact_low_ms" in c
        assert "impact_high_ms" in c
        # The fixture's P-items all have non-empty prose; require it.
        assert c["reasoning_for_slowdown"], (
            f"empty reasoning_for_slowdown on candidate {c.get('name')!r} (rank P{c.get('tracelens_pitem_rank')})"
        )
        assert c["resolution"], f"empty resolution on candidate {c.get('name')!r}"

    # P1 prose mentions "Tile / wave-occupancy tuning" per the fixture.
    p1_rows = [c for c in cands if c["tracelens_pitem_rank"] == 1]
    assert any("wave-occupancy" in c["resolution"] for c in p1_rows), (
        "P1 resolution should mention wave-occupancy tuning (from fixture)"
    )


# parse_analysis_md — spec allows trailing category-specific extra columns after the 9 canonical
# ones (attention appends 3, generic-op appends Sub-Category); the parser must accept them, not skip.
_FIXTURE_QWEN3_ATTENTION_ANALYSIS_MD = (
    Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "tracelens_v03_qwen3_moe_attention_analysis.md"
)


def test_parse_analysis_md_tolerates_attention_12_column_table_per_spec():
    """A 12-column attention ``**Data:**`` table (9 canonical + 3 spec-allowed extras) must parse using the first 9 cells."""
    cands = tlr.parse_analysis_md(_FIXTURE_QWEN3_ATTENTION_ANALYSIS_MD, top_k=10)
    assert len(cands) == 1, f"expected 1 attention candidate from the 12-column fixture; got {len(cands)}"
    c = cands[0]
    assert c["name"] == "vllm::unified_attention_with_output"
    assert c["tracelens_category"] == "inferenceattention"
    assert c["tracelens_pitem_rank"] == 1
    assert c["bound_type"] == "memory-bound"
    # Time (ms) -> duration_us; row is 45.862 ms.
    assert abs(c["duration_us"] - 45862.0) < 1.0
    assert c["call_count"] == 48
    assert abs(c["percent_of_total"] - 2.61) < 0.001
    assert abs(c["efficiency_percent"] - 3.69) < 0.001
    assert c["efficiency_peak_value"] == 8.0
    assert "TB/s" in c["efficiency_peak_unit"]
    # impact_score is the mid value carried by the p_item marker.
    assert c["impact_score"] == 2.2
    # Kernel Path is a real launcher string (not "—"), so source_file
    # must round-trip the relative path (resolution happens downstream).
    assert "qwen3_moe.py" in c["source_file"]
    # The three trailing extra cells are spec-allowed extras, preserved under tracelens_extra_columns.
    extras = c.get("tracelens_extra_columns")
    assert extras is not None, "tracelens_extra_columns missing for 12-col row"
    assert extras.get("dominant kernel") == "`_fwd_kernel` (93.61%)"
    assert extras.get("workload") == "unknown"
    assert extras.get("attention pattern") == "GQA (8:1)"
    # Canonical fields must NOT leak into extras.
    for canonical_key in (
        "operation",
        "args",
        "kernel path",
        "time (ms)",
        "%e2e",
        "count",
        "flops/byte",
        "efficiency",
        "bound",
    ):
        assert canonical_key not in extras


def test_unified_attention_fixture_emits_semantic_workload_selectors():
    candidates = tlr.parse_analysis_md(
        _FIXTURE_QWEN3_ATTENTION_ANALYSIS_MD,
        top_k=10,
    )
    candidate = candidates[0]
    candidate["kernel_id"] = "k001"
    group = {
        "primary_kernel_id": "k001",
        "rows": [candidate],
    }

    cases = task_group_contract.build_task_group_shape_cases(group)
    assert len(cases) == 1
    selector = cases[0]["selector"]
    assert selector == {
        "CASE_ID": "case_001",
        "QTOKENS": 1087,
        "QHEADS": 32,
        "KVHEADS": 4,
        "HEADSIZE": 128,
    }
    assert {key: value for key, value in selector.items() if key != "CASE_ID"} == {
        "QTOKENS": 1087,
        "QHEADS": 32,
        "KVHEADS": 4,
        "HEADSIZE": 128,
    }
    canonical_workload = "_".join(f"{key}{selector[key]}" for key in sorted(selector) if key != "CASE_ID")
    assert canonical_workload == ("HEADSIZE128_KVHEADS4_QHEADS32_QTOKENS1087")

    grouped_candidate = dict(candidate)
    grouped_candidate["task_group"] = {
        **group,
        "shape_cases": cases,
    }
    shapes = task_group_contract.forge_shapes_from_candidate(grouped_candidate)
    assert shapes["primary"] == selector
    assert shapes["minimal"] == selector
    assert shapes["validation"] == [selector]


def test_parse_analysis_md_tolerates_subcategory_10_column_table_per_spec(tmp_path):
    """A 10-column table with a trailing ``Sub-Category`` must parse using the first 9 cells."""
    md = tmp_path / "analysis.md"
    md.write_text(
        "<!-- impact-begin kind=p_item category=other mid=4.0 low=2.0 high=8.0 -->\n"
        "\n## Detailed Analysis\n\n### Compute Kernel Insights\n\n"
        "<!-- reasoning-candidate tier=compute rank=1 -->\n"
        "#### 🔴 P1: Generic op cluster (Triton)\n\n"
        "**Identification:** stub identification\n"
        "**Data:**\n"
        "| Operation | Args | Kernel Path | Time (ms) | %E2E | Count | "
        "FLOPS/Byte | Efficiency | Bound | Sub-Category |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
        "| custom_op | (1,2) bf16 | — | 1.0 | 5 | 10 | 1000 | "
        "40% of 708 TFLOPS | compute-bound | scatter_gather |\n"
        "**Reasoning for Slowdown:** stub reasoning\n"
        "**Resolution:** stub resolution\n",
        encoding="utf-8",
    )
    cands = tlr.parse_analysis_md(md, top_k=10)
    assert len(cands) == 1
    c = cands[0]
    assert c["name"] == "custom_op"
    assert c["tracelens_category"] == "other"
    assert c["bound_type"] == "compute-bound"
    assert c["call_count"] == 10
    # ``Sub-Category`` is preserved in extras, never the candidate top-level.
    extras = c.get("tracelens_extra_columns")
    assert extras is not None
    assert extras.get("sub-category") == "scatter_gather"
    assert "sub-category" not in c


def test_parse_analysis_md_canonical_9_column_table_has_no_extras_key():
    """Canonical 9-column candidates must NOT carry a ``tracelens_extra_columns`` key."""
    cands = tlr.parse_analysis_md(_FIXTURE_LLAMA70B_ANALYSIS_MD, top_k=50)
    assert cands, "Llama70B fixture must produce candidates"
    for c in cands:
        assert "tracelens_extra_columns" not in c, (
            f"canonical 9-col candidate {c.get('name')!r} unexpectedly "
            f"carries tracelens_extra_columns={c.get('tracelens_extra_columns')!r}"
        )


def test_parse_analysis_md_rejects_fewer_than_canonical_columns(tmp_path):
    """A table missing a canonical column (here ``Bound``, 8 cols) must be skipped, not mis-mapped."""
    md = tmp_path / "analysis.md"
    md.write_text(
        "<!-- impact-begin kind=p_item category=gemm mid=4.0 low=2.0 high=8.0 -->\n"
        "\n## Detailed Analysis\n\n### Compute Kernel Insights\n\n"
        "<!-- reasoning-candidate tier=compute rank=1 -->\n"
        "#### 🔴 P1: Missing column (Tensile)\n\n"
        "**Identification:** stub identification\n"
        "**Data:**\n"
        "| Operation | Args | Kernel Path | Time (ms) | %E2E | Count | "
        "FLOPS/Byte | Efficiency |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- |\n"
        "| stub_op | (1,2) bf16 | — | 1.0 | 5 | 10 | 1000 | "
        "40% of 708 TFLOPS |\n"
        "**Reasoning for Slowdown:** stub reasoning\n"
        "**Resolution:** stub resolution\n",
        encoding="utf-8",
    )
    assert tlr.parse_analysis_md(md, top_k=10) == []


def test_parse_analysis_md_rejects_reordered_canonical_columns(tmp_path):
    """Reordered canonical columns (Bound/Efficiency swapped) must be skipped, not mis-mapped."""
    md = tmp_path / "analysis.md"
    md.write_text(
        "<!-- impact-begin kind=p_item category=gemm mid=4.0 low=2.0 high=8.0 -->\n"
        "\n## Detailed Analysis\n\n### Compute Kernel Insights\n\n"
        "<!-- reasoning-candidate tier=compute rank=1 -->\n"
        "#### 🔴 P1: Reordered columns (Tensile)\n\n"
        "**Identification:** stub identification\n"
        "**Data:**\n"
        "| Operation | Args | Kernel Path | Time (ms) | %E2E | Count | "
        "FLOPS/Byte | Bound | Efficiency |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
        "| stub_op | (1,2) bf16 | — | 1.0 | 5 | 10 | 1000 | "
        "compute-bound | 40% of 708 TFLOPS |\n"
        "**Reasoning for Slowdown:** stub reasoning\n"
        "**Resolution:** stub resolution\n",
        encoding="utf-8",
    )
    assert tlr.parse_analysis_md(md, top_k=10) == []


# classify_patchability gate + skip_reason audit field.
def test_classify_patchability_accepts_stable_triton_source():
    """A stable Triton source is reusable; skip_reason is empty."""
    cand = {
        "name": "triton_attention_decode_kernel",
        "source_file": "/sgl-workspace/sglang/python/sglang/srt/layers/attn.py",
        "source_type": "triton",
    }
    reusable, reason = tla.classify_patchability(cand)
    assert reusable is True
    assert reason == ""


def test_classify_patchability_rejects_missing_source_file():
    reusable, reason = tla.classify_patchability(
        {"name": "rms_norm", "source_type": "triton"},
    )
    assert reusable is False
    assert "source file not resolved" in reason


def test_classify_patchability_rejects_cpp_itfs_py_host_launcher(monkeypatch):
    """A csrc/cpp_itfs/*.py host launcher (device code is in a sibling
    .cuh/.cpp.jinja) must be skipped, not edited."""
    src = "/path/aiter/csrc/cpp_itfs/pa/pa_ragged.py"
    # Make the reusable-root gate pass deterministically regardless of host env.
    monkeypatch.setattr(tla, "_reusable_roots", lambda: ("/path/aiter/",))
    reusable, reason = tla.classify_patchability(
        {"name": "paged_attention_ragged", "source_file": src, "source_type": "python"},
    )
    assert reusable is False
    assert "cpp_itfs host launcher" in reason


def test_library_token_pairing():
    """Library detection keeps kernel<->benchmark same-lib."""
    assert tla._library_token("/sgl-workspace/aiter/op_tests/test_activation.py") == "aiter"
    # sgl-kernel / sgl_kernel normalize to sglang.
    assert tla._library_token("/sgl-workspace/sglang/sgl-kernel/include/hip/x.cuh") == "sglang"
    assert tla._library_token("/sgl-workspace/sglang/python/sglang/srt/x.py") == "sglang"
    assert tla._library_token("/random/path/foo.py") == ""
    # A sglang kernel and an aiter test are different libraries -> must not pair.
    src = "/sgl-workspace/sglang/sgl-kernel/include/hip/hip_act_and_mul.cuh"
    test = "/sgl-workspace/aiter/op_tests/test_activation.py"
    assert tla._library_token(src) != tla._library_token(test)


def test_classify_patchability_allows_aiter_device_source_unknown_type(monkeypatch):
    """aiter .cu/.cuh device sources are patchable even when source_type is
    'unknown' (classifier ran before source_file resolved). Enables forge to
    optimize aiter::mha_batch_prefill etc."""
    src = "/sgl-workspace/aiter/csrc/py_itfs_ck/mha_batch_prefill_kernels.cu"
    monkeypatch.setattr(tla, "_reusable_roots", lambda: ("/sgl-workspace/aiter/",))
    reusable, reason = tla.classify_patchability(
        {"name": "aiter::mha_batch_prefill", "source_file": src, "source_type": "unknown"},
    )
    assert reusable is True, reason


def test_classify_patchability_still_rejects_aiter_py_dispatcher(monkeypatch):
    """aten::mm -> aiter tuned_gemm.py is a dispatcher (real GEMM is a compiled
    CK/hipBLASLt lib); editing the .py does nothing, so it stays non-patchable."""
    src = "/sgl-workspace/aiter/aiter/tuned_gemm.py"
    monkeypatch.setattr(tla, "_reusable_roots", lambda: ("/sgl-workspace/aiter/",))
    reusable, reason = tla.classify_patchability(
        {"name": "aten::mm", "source_file": src, "source_type": "python", "library": "pytorch native"},
    )
    assert reusable is False


def test_classify_patchability_keeps_cpp_itfs_device_source(monkeypatch):
    """The real device source (.cuh) under cpp_itfs stays reusable."""
    src = "/path/aiter/csrc/cpp_itfs/pa/pa_kernels.cuh"
    monkeypatch.setattr(tla, "_reusable_roots", lambda: ("/path/aiter/",))
    reusable, reason = tla.classify_patchability(
        {"name": "paged_attention", "source_file": src, "source_type": "hip_cpp"},
    )
    assert reusable is True
    assert reason == ""


def test_classify_patchability_rejects_vendor_blas_name_markers():
    """Vendor BLAS / collective name markers are rejected even under a reusable framework root."""
    for marker_name in (
        "rocblas_sgemm_kernel",
        "hipblas_gemm_strided",
        "tensile_gemm_NN_bf16",
        "rccl_AllReduce_sum",
        "nccl_kernel",
        "aten::copy_",
    ):
        reusable, reason = tla.classify_patchability(
            {
                "name": marker_name,
                "source_file": "/sgl-workspace/aiter/foo.py",
                "source_type": "python",
            }
        )
        assert reusable is False, marker_name
        assert "non-patchable" in reason or "PyTorch native" in reason, marker_name


def test_classify_patchability_rejects_aten_without_library():
    """aten::* without a library hint is treated as Tensile / native backend."""
    reusable, reason = tla.classify_patchability(
        {
            "name": "aten::mm",
            "source_file": "/sgl-workspace/aiter/foo.py",
            "source_type": "python",
            "library": "",
        }
    )
    assert reusable is False
    assert "Tensile" in reason or "vendor" in reason


def test_classify_patchability_rejects_aten_tensile_library():
    """Explicit library == 'Tensile' is the most common reject path."""
    reusable, reason = tla.classify_patchability(
        {
            "name": "aten::mm",
            "source_file": "/sgl-workspace/aiter/foo.py",
            "source_type": "python",
            "library": "Tensile",
        }
    )
    assert reusable is False
    assert "Tensile" in reason


def test_classify_patchability_rejects_runtime_generated_kernel():
    reusable, reason = tla.classify_patchability(
        {
            "name": "triton_poi_fused_add_0",
            "source_file": "/tmp/torchinductor_root/ab/cdef.py",
            "source_type": "runtime_generated",
        }
    )
    assert reusable is False
    assert "runtime-generated" in reason


def test_classify_patchability_rejects_unreusable_source_root():
    reusable, reason = tla.classify_patchability(
        {
            "name": "my_custom_kernel",
            "source_file": "/tmp/random/my_custom_kernel.cu",
            "source_type": "hip_cpp",
        }
    )
    assert reusable is False
    assert "reusable framework root" in reason


def test_build_audit_summary_splits_tasks_and_skipped():
    """``build_audit_summary`` surfaces kernel name + skip_reason for every dropped candidate."""
    finalized = [
        {
            "kernel_id": "k001",
            "name": "good_triton_kernel",
            "source_file": "/sgl-workspace/aiter/x.py",
            "source_type": "triton",
            "reusable_native_kernel": True,
            "skip_reason": "",
            "gpu_pct": 12.5,
            "tracelens_pitem_rank": 1,
            "recommended_backends": ["forge"],
        },
        {
            "kernel_id": "k002",
            "name": "rocblas_sgemm",
            "source_file": "/sgl-workspace/aiter/x.py",
            "source_type": "python",
            "reusable_native_kernel": False,
            "skip_reason": "non-patchable kernel name marker 'rocblas' in 'rocblas_sgemm'",
            "gpu_pct": 5.2,
        },
        {
            "kernel_id": "k003",
            "name": "aten::mm",
            "source_file": "",
            "source_type": "tracelens_report",
            "reusable_native_kernel": False,
            "skip_reason": "source file not resolved",
            "gpu_pct": 30.0,
        },
    ]
    summary = tla.build_audit_summary(
        finalized,
        trace_input="/tmp/trace.json.gz",
        framework="sglang",
        target_platform="mi300x",
    )
    assert summary["task_count"] == 1
    assert summary["skipped_count"] == 2
    assert summary["trace_input"] == "/tmp/trace.json.gz"
    assert summary["framework"] == "sglang"
    assert summary["target_platform"] == "mi300x"

    task_names = [t["name"] for t in summary["tasks"]]
    skipped_names = [s["name"] for s in summary["skipped"]]
    assert task_names == ["good_triton_kernel"]
    assert set(skipped_names) == {"rocblas_sgemm", "aten::mm"}

    rocblas_entry = next(s for s in summary["skipped"] if s["name"] == "rocblas_sgemm")
    assert "rocblas" in rocblas_entry["skip_reason"]
    aten_entry = next(s for s in summary["skipped"] if s["name"] == "aten::mm")
    assert "source file" in aten_entry["skip_reason"]
    # Reusable tasks carry recommended_backends so the audit shows routing without reloading candidates.
    assert summary["tasks"][0]["recommended_backends"] == ["forge"]


def test_build_audit_summary_handles_empty_input():
    summary = tla.build_audit_summary([], trace_input="/tmp/x.json.gz")
    assert summary["task_count"] == 0
    assert summary["skipped_count"] == 0
    assert summary["tasks"] == []
    assert summary["skipped"] == []


# Source-function aggregation.
def test_parse_launcher_path_extracts_python_frame():
    """``<path>(<line>): <fn>`` is the canonical TraceLens shape."""
    path, line, func = tlr._parse_launcher_path(
        "aiter/ops/rmsnorm.py(76): rmsnorm",
    )
    assert path == "aiter/ops/rmsnorm.py"
    assert line == 76
    assert func == "rmsnorm"


def test_parse_launcher_path_handles_hash_l_form():
    """Bare file refs / ``<path>#L<line>`` are accepted as fallback shapes."""
    path, line, func = tlr._parse_launcher_path(
        "/sgl-workspace/aiter/csrc/foo.cu#L42",
    )
    assert path == "/sgl-workspace/aiter/csrc/foo.cu"
    assert line == 42
    assert func is None


def test_parse_launcher_path_returns_none_for_empty_and_garbage():
    """Empty / placeholder Kernel Path values collapse to ``("", None, None)``; bare paths pass through."""
    assert tlr._parse_launcher_path("") == ("", None, None)
    assert tlr._parse_launcher_path("—") == ("", None, None)
    assert tlr._parse_launcher_path("-") == ("", None, None)
    assert tlr._parse_launcher_path("N/A") == ("", None, None)
    # Bare path: line/fn stay None (stem fallback happens downstream).
    path, line, func = tlr._parse_launcher_path("just/a/path.py")
    assert path == "just/a/path.py"
    assert line is None
    assert func is None


# ---------------------------------------------------------------------------
# _resolve_launcher_to_abs_source — TraceLens launcher path → absolute file.
# Pins the three resolution paths (importlib spec, env override, hardcoded fallback) plus no-op cases.


def _seed_pkg(tmp_path, pkg: str, relpath: str, funcs: tuple[str, ...] = ()) -> Path:
    """Create ``<tmp_path>/<pkg>/<relpath>`` (with optional top-level ``def`` names) and return the absolute file."""
    target = tmp_path / pkg / relpath
    target.parent.mkdir(parents=True, exist_ok=True)
    body = ["# stub for resolver tests\n"]
    for fn in funcs:
        body.append(f"def {fn}(*args, **kwargs):\n    pass\n")
    target.write_text("".join(body), encoding="utf-8")
    return target


def test_resolve_launcher_via_env_override(tmp_path, monkeypatch):
    """``$HYPERLOOM_FRAMEWORK_SOURCE_ROOTS`` is the highest-priority resolver source (no import needed)."""
    target = _seed_pkg(
        tmp_path,
        "aiter",
        "ops/rmsnorm.py",
        funcs=("rmsnorm2d_fwd",),
    )
    monkeypatch.setenv(
        tlr._FRAMEWORK_SOURCE_ROOTS_ENV,
        f"aiter={tmp_path}",
    )

    resolved = tlr._resolve_launcher_to_abs_source(
        "aiter/ops/rmsnorm.py(62): rmsnorm2d_fwd",
    )
    assert resolved is not None
    abs_path, line, func = resolved
    assert abs_path == str(target)
    assert line == 62
    assert func == "rmsnorm2d_fwd"


def test_resolve_launcher_via_importlib_spec(monkeypatch):
    """With no env override, ``find_spec`` resolves the absolute origin (tested via stdlib ``unittest/case.py``)."""
    monkeypatch.delenv(tlr._FRAMEWORK_SOURCE_ROOTS_ENV, raising=False)
    resolved = tlr._resolve_launcher_to_abs_source(
        "unittest/case.py(1): expectedFailure",
    )
    assert resolved is not None
    abs_path, line, func = resolved
    assert os.path.isabs(abs_path) and abs_path.endswith("unittest/case.py")
    assert line == 1
    assert func == "expectedFailure"


def test_resolve_launcher_via_hardcoded_fallback(tmp_path, monkeypatch):
    """When the package isn't importable but a hardcoded fallback root holds the file, the resolver still succeeds."""
    monkeypatch.delenv(tlr._FRAMEWORK_SOURCE_ROOTS_ENV, raising=False)

    # Seed a fake aiter checkout and point the fallback table at it (package not importable).
    _seed_pkg(
        tmp_path,
        "aiter_pinned_xfx",
        "ops/rmsnorm.py",
        funcs=("rmsnorm2d_fwd_with_add",),
    )
    monkeypatch.setattr(
        tlr,
        "_FRAMEWORK_PKG_FALLBACK_ROOTS",
        {"aiter_pinned_xfx": (str(tmp_path),)},
    )
    # Force find_spec to miss so the fallback table is exercised.
    monkeypatch.setattr(tlr, "_package_root_parent", lambda pkg: None)

    resolved = tlr._resolve_launcher_to_abs_source(
        "aiter_pinned_xfx/ops/rmsnorm.py(76): rmsnorm2d_fwd_with_add",
    )
    assert resolved is not None
    abs_path, line, func = resolved
    assert abs_path == str(tmp_path / "aiter_pinned_xfx" / "ops" / "rmsnorm.py")
    assert line == 76
    assert func == "rmsnorm2d_fwd_with_add"


def test_resolve_launcher_splits_an_absolute_annotated_path():
    """Absolute launchers keep their path while separating line and function."""
    assert tlr._resolve_launcher_to_abs_source(
        "/sgl-workspace/aiter/aiter/ops/rmsnorm.py(62): fn",
    ) == (
        "/sgl-workspace/aiter/aiter/ops/rmsnorm.py",
        62,
        "fn",
    )


def test_resolve_launcher_returns_none_for_placeholders_and_misses():
    """Placeholders, empty strings, and unresolvable packages collapse to None (no fabricated path)."""
    assert tlr._resolve_launcher_to_abs_source("") is None
    assert tlr._resolve_launcher_to_abs_source("—") is None
    # Unresolvable package → None; caller falls back to the verbatim string.
    assert (
        tlr._resolve_launcher_to_abs_source(
            "definitely_not_a_real_pkg_8x9z/foo.py(1): fn",
        )
        is None
    )


def test_resolve_launcher_rejects_when_function_not_in_file(tmp_path, monkeypatch):
    """AST guard: when the resolved ``.py`` exists but lacks the launcher's function, the resolver refuses it."""
    # Seed a .py file that does NOT define rmsnorm2d_fwd.
    target = tmp_path / "aiter_shadowed_xyz" / "ops" / "rmsnorm.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "def some_other_function():\n    pass\n",
        encoding="utf-8",
    )
    monkeypatch.setenv(
        tlr._FRAMEWORK_SOURCE_ROOTS_ENV,
        f"aiter_shadowed_xyz={tmp_path}",
    )
    # No fallback paths so the bad-symbol path is rejected outright.
    monkeypatch.setattr(tlr, "_package_root_parent", lambda pkg: None)
    monkeypatch.setattr(tlr, "_FRAMEWORK_PKG_FALLBACK_ROOTS", {})

    assert (
        tlr._resolve_launcher_to_abs_source(
            "aiter_shadowed_xyz/ops/rmsnorm.py(62): rmsnorm2d_fwd",
        )
        is None
    )


def test_resolve_launcher_ast_check_falls_through_to_next_root(tmp_path, monkeypatch):
    """When the first candidate root holds a stub that fails AST
    validation, the resolver MUST keep walking the candidate list
    instead of short-circuiting — otherwise a single bad spec
    (shadowed pkg / stale wheel) permanently masks the real source on
    the fallback path."""
    bad_root = tmp_path / "bad"
    good_root = tmp_path / "good"
    bad_target = bad_root / "aiter_pinned_qrs" / "ops" / "rmsnorm.py"
    bad_target.parent.mkdir(parents=True, exist_ok=True)
    bad_target.write_text("def not_it():\n    pass\n", encoding="utf-8")
    good_target = good_root / "aiter_pinned_qrs" / "ops" / "rmsnorm.py"
    good_target.parent.mkdir(parents=True, exist_ok=True)
    good_target.write_text(
        "def rmsnorm2d_fwd(x):\n    return x\n",
        encoding="utf-8",
    )

    monkeypatch.delenv(tlr._FRAMEWORK_SOURCE_ROOTS_ENV, raising=False)
    monkeypatch.setattr(tlr, "_package_root_parent", lambda pkg: None)
    monkeypatch.setattr(
        tlr,
        "_FRAMEWORK_PKG_FALLBACK_ROOTS",
        {"aiter_pinned_qrs": (str(bad_root), str(good_root))},
    )

    resolved = tlr._resolve_launcher_to_abs_source(
        "aiter_pinned_qrs/ops/rmsnorm.py(76): rmsnorm2d_fwd",
    )
    assert resolved is not None
    abs_path, line, func = resolved
    assert abs_path == str(good_target)
    assert line == 76
    assert func == "rmsnorm2d_fwd"


def test_resolve_launcher_skips_ast_check_for_non_py_sources(tmp_path, monkeypatch):
    """AST validation only applies to Python sources; HIP/CUDA refs pass existence-only validation."""
    target = tmp_path / "aiter_hipxyz" / "csrc" / "rms_hip.cu"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("// device code\n", encoding="utf-8")
    monkeypatch.setenv(
        tlr._FRAMEWORK_SOURCE_ROOTS_ENV,
        f"aiter_hipxyz={tmp_path}",
    )

    resolved = tlr._resolve_launcher_to_abs_source(
        "aiter_hipxyz/csrc/rms_hip.cu#L42",
    )
    assert resolved is not None
    abs_path, line, func = resolved
    assert abs_path == str(target)
    assert line == 42
    assert func is None


def test_resolve_launcher_skips_unparseable_env_entries(tmp_path, monkeypatch):
    """Malformed env entries are silently skipped; the one valid entry still wins."""
    target = _seed_pkg(
        tmp_path,
        "vllm",
        "model_executor/models/qwen.py",
        funcs=("forward",),
    )
    monkeypatch.setenv(
        tlr._FRAMEWORK_SOURCE_ROOTS_ENV,
        # First two entries are malformed (skipped); the third is the valid one.
        f"junk_without_equals,=/just/value,vllm={tmp_path}",
    )

    resolved = tlr._resolve_launcher_to_abs_source(
        "vllm/model_executor/models/qwen.py(10): forward",
    )
    assert resolved is not None
    abs_path, _, _ = resolved
    assert abs_path == str(target)


def test_function_line_from_ast_finds_def_lineno(tmp_path):
    src = tmp_path / "kernel.py"
    src.write_text(
        "import torch\n\n\ndef other():\n    pass\n\n\ndef rms_norm(x):\n    return x\n",
        encoding="utf-8",
    )
    # The ``def rms_norm`` line is at line 8 (1-indexed).
    assert tlr._function_line_from_ast(src, "rms_norm") == 8
    assert tlr._function_line_from_ast(src, "missing") is None


def test_function_line_from_ast_returns_none_on_invalid_source(tmp_path):
    """Unreadable / non-Python files don't raise — caller falls back."""
    src = tmp_path / "broken.py"
    src.write_text("this is not valid python ::: !!!", encoding="utf-8")
    assert tlr._function_line_from_ast(src, "anything") is None
    assert tlr._function_line_from_ast(tmp_path / "does_not_exist.py", "x") is None


def test_aggregate_by_source_function_groups_same_function_calls(tmp_path):
    """Candidates sharing Operation name + source function group together; a different function stays separate."""
    src = tmp_path / "rmsnorm.py"
    src.write_text(
        "def rms_norm(x):\n    return x\n\n\ndef other_fn(x):\n    return x\n",
        encoding="utf-8",
    )
    cands = [
        {
            "kernel_id": "k001",
            "name": "aiter::rms_norm",
            "duration_us": 100.0,
            "call_count": 64,
            "gpu_pct": 5.0,
            "tracelens_launcher_path": f"{src}(2): rms_norm",
        },
        {
            "kernel_id": "k002",
            "name": "aiter::rms_norm",  # same op, different shape
            "duration_us": 50.0,
            "call_count": 32,
            "gpu_pct": 2.5,
            "tracelens_launcher_path": f"{src}(2): rms_norm",
        },
        {
            "kernel_id": "k003",
            "name": "aiter::other_fn_kernel",
            "duration_us": 30.0,
            "call_count": 16,
            "gpu_pct": 1.5,
            "tracelens_launcher_path": f"{src}(5): other_fn",
        },
    ]
    groups = tlr.aggregate_by_source_function(cands)
    assert len(groups) == 2
    g0, g1 = groups
    assert g0["function_name"] == "rms_norm"
    assert g0["operation"] == "aiter::rms_norm"
    assert g0["task_group_id"] == "tg001"
    assert set(g0["kernel_ids"]) == {"k001", "k002"}
    assert g0["primary_kernel_id"] == "k001"
    assert g0["aggregate_duration_us"] == 150.0
    assert g0["aggregate_call_count"] == 96
    assert g0["aggregate_gpu_pct"] == 7.5
    assert g0["definition_line"] == 1
    assert g0["ast_resolved"] is True

    assert g1["function_name"] == "other_fn"
    assert g1["operation"] == "aiter::other_fn_kernel"
    assert g1["task_group_id"] == "tg002"
    assert g1["kernel_ids"] == ["k003"]
    assert g1["definition_line"] == 5


def test_python_task_group_key_is_stable_across_definition_line_changes(tmp_path):
    src = tmp_path / "operator.py"
    src.write_text("def forward(x):\n    return x\n", encoding="utf-8")
    candidate = {
        "kernel_id": "k001",
        "name": "fused_operator",
        "duration_us": 100.0,
        "tracelens_launcher_path": f"{src}(1): forward",
    }
    first_key = tlr.aggregate_by_source_function([candidate])[0]["task_group_key"]

    src.write_text("\n\n\ndef forward(x):\n    return x\n", encoding="utf-8")
    candidate["tracelens_launcher_path"] = f"{src}(4): forward"
    second_key = tlr.aggregate_by_source_function([candidate])[0]["task_group_key"]

    assert first_key == second_key


def test_aggregate_does_not_merge_different_operations_sharing_wrapper(tmp_path):
    """Q1 invariant: distinct operations sharing one Python wrapper stay in separate task_groups (operation is part of the key)."""
    src = tmp_path / "gpt_oss.py"
    src.write_text(
        "def x():\n    pass\n\n\ndef forward(x):\n    return x\n",
        encoding="utf-8",
    )
    launcher = f"{src}(5): forward"
    cands = [
        {
            "kernel_id": "k001",
            "name": "vllm::rocm_unquantized_gemm",
            "duration_us": 12704.0,
            "call_count": 360,
            "tracelens_launcher_path": launcher,
        },
        {
            "kernel_id": "k002",
            "name": "vllm::rocm_aiter_triton_add_rmsnorm_pad",
            "duration_us": 9870.0,
            "call_count": 360,
            "tracelens_launcher_path": launcher,
        },
        # Same op as k001 at a different shape MUST still merge with k001.
        {
            "kernel_id": "k003",
            "name": "vllm::rocm_unquantized_gemm",
            "duration_us": 1260.0,
            "call_count": 36,
            "tracelens_launcher_path": launcher,
        },
    ]
    groups = tlr.aggregate_by_source_function(cands)
    assert len(groups) == 2, (
        f"expected 2 groups (one per Operation); got {len(groups)} — "
        "k001 and k002 likely merged on shared wrapper, the bug"
    )
    by_op = {g["operation"]: g for g in groups}
    assert "vllm::rocm_unquantized_gemm" in by_op
    assert "vllm::rocm_aiter_triton_add_rmsnorm_pad" in by_op
    gemm_group = by_op["vllm::rocm_unquantized_gemm"]
    assert set(gemm_group["kernel_ids"]) == {"k001", "k003"}, (
        "same-Operation rows at different shapes must collapse into one group"
    )
    rms_group = by_op["vllm::rocm_aiter_triton_add_rmsnorm_pad"]
    assert rms_group["kernel_ids"] == ["k002"]


def test_aggregate_keeps_same_operation_in_different_functions_separate(
    tmp_path,
):
    src = tmp_path / "operator.py"
    src.write_text(
        "def first(x):\n    return x\n\ndef second(x):\n    return x\n",
        encoding="utf-8",
    )
    candidates = [
        {
            "kernel_id": "k001",
            "name": "shared_operation",
            "duration_us": 100.0,
            "tracelens_launcher_path": f"{src}(1): first",
        },
        {
            "kernel_id": "k002",
            "name": "shared_operation",
            "duration_us": 90.0,
            "tracelens_launcher_path": f"{src}(4): second",
        },
    ]

    groups = tlr.aggregate_by_source_function(candidates)

    assert len(groups) == 2
    assert {group["function_name"] for group in groups} == {
        "first",
        "second",
    }


def test_aggregate_collects_distinct_pitem_prose_when_function_spans_pitems(tmp_path):
    """Q2 invariant: a function spanning multiple P-items collects every P-item's prose on ``all_pitem_prose`` (deduped by (rank,title), sorted by rank)."""
    src = tmp_path / "rmsnorm.py"
    src.write_text("def rms_norm(x):\n    return x\n", encoding="utf-8")
    launcher = f"{src}(1): rms_norm"
    cands = [
        {
            "kernel_id": "k001",
            "name": "aiter::rms_norm",
            "duration_us": 200.0,
            "call_count": 100,
            "tracelens_launcher_path": launcher,
            "tracelens_pitem_rank": 2,
            "tracelens_pitem_title": "Memory-Bound at decode shapes",
            "identification": "Decode-shape Identification.",
            "reasoning_for_slowdown": "Decode-shape Reasoning.",
            "resolution": "Decode-shape Resolution.",
            "impact_low_ms": 5.0,
            "impact_high_ms": 10.0,
        },
        {
            "kernel_id": "k002",
            "name": "aiter::rms_norm",
            "duration_us": 80.0,
            "call_count": 40,
            "tracelens_launcher_path": launcher,
            "tracelens_pitem_rank": 5,
            "tracelens_pitem_title": "Compute-Bound at prefill shapes",
            "identification": "Prefill-shape Identification.",
            "reasoning_for_slowdown": "Prefill-shape Reasoning.",
            "resolution": "Prefill-shape Resolution.",
            "impact_low_ms": 1.0,
            "impact_high_ms": 3.0,
        },
        # Same P2 again — must dedupe (only one entry retained).
        {
            "kernel_id": "k003",
            "name": "aiter::rms_norm",
            "duration_us": 50.0,
            "call_count": 25,
            "tracelens_launcher_path": launcher,
            "tracelens_pitem_rank": 2,
            "tracelens_pitem_title": "Memory-Bound at decode shapes",
            "identification": "Decode-shape Identification.",
            "reasoning_for_slowdown": "Decode-shape Reasoning.",
            "resolution": "Decode-shape Resolution.",
            "impact_low_ms": 5.0,
            "impact_high_ms": 10.0,
        },
    ]
    groups = tlr.aggregate_by_source_function(cands)
    assert len(groups) == 1
    g = groups[0]
    prose = g["all_pitem_prose"]
    assert len(prose) == 2, f"expected 2 distinct (rank,title) prose entries; got {len(prose)}"
    # Sorted by rank ascending → P2 first, P5 second.
    assert prose[0]["rank"] == 2
    assert "decode" in prose[0]["title"].lower()
    assert prose[0]["reasoning_for_slowdown"] == "Decode-shape Reasoning."
    assert prose[1]["rank"] == 5
    assert "prefill" in prose[1]["title"].lower()
    assert prose[1]["resolution"] == "Prefill-shape Resolution."
    # Set-typed bookkeeping must not leak into the returned dict (breaks JSON).
    assert "_pitem_prose_seen" not in g


def test_same_kernel_different_shapes_yields_one_task_with_all_shapes_as_cases(
    tmp_path,
):
    """End-to-end: same op + same source function + different shapes per row collapse to ONE task_group; each row's shapes are preserved verbatim and rendered as a distinct ``Case N:``."""
    src = tmp_path / "model_executor.py"
    src.write_text(
        "def x(): pass\n\n\ndef forward(x):\n    return x\n",
        encoding="utf-8",
    )
    launcher = f"{src}(5): forward"
    cands = [
        {
            "kernel_id": "k001",
            "name": "vllm::rocm_unquantized_gemm",
            "shapes": ["(64,2880) bf16", "(128,2880) bf16", "(128,) bf16"],
            "duration_us": 12704.0,
            "call_count": 360,
            "bound_type": "memory-bound",
            "tracelens_launcher_path": launcher,
        },
        {
            "kernel_id": "k002",
            "name": "vllm::rocm_unquantized_gemm",
            "shapes": ["(64,2880) bf16", "(640,2880) bf16", "(640,) bf16"],
            "duration_us": 10992.0,
            "call_count": 360,
            "bound_type": "memory-bound",
            "tracelens_launcher_path": launcher,
        },
        {
            "kernel_id": "k003",
            "name": "vllm::rocm_unquantized_gemm",
            "shapes": ["(64,512) bf16", "(2880,512) bf16", "(2880,) bf16"],
            "duration_us": 9291.0,
            "call_count": 360,
            "bound_type": "memory-bound",
            "tracelens_launcher_path": launcher,
        },
        {
            "kernel_id": "k004",
            "name": "vllm::rocm_unquantized_gemm",
            "shapes": ["(2048,2880) bf16", "(128,2880) bf16", "(128,) bf16"],
            "duration_us": 1260.0,
            "call_count": 36,
            "bound_type": "memory-bound",
            "tracelens_launcher_path": launcher,
        },
    ]
    groups = tlr.aggregate_by_source_function(cands)
    assert len(groups) == 1, f"expected 1 task_group (same op + same source); got {len(groups)}"
    g = groups[0]
    assert g["operation"] == "vllm::rocm_unquantized_gemm"
    assert set(g["kernel_ids"]) == {"k001", "k002", "k003", "k004"}
    assert g["primary_kernel_id"] == "k001"  # heaviest (12704 us)
    assert len(g["rows"]) == 4
    assert [case["case_id"] for case in g["shape_cases"]] == [
        "case_001",
        "case_002",
        "case_003",
        "case_004",
    ]
    # Each row preserves its own shape list verbatim — no merging,
    # no de-duplication. Order is duration-desc post-aggregation so
    # row[0]=k001, row[3]=k004.
    assert g["rows"][0]["shapes"] == ["(64,2880) bf16", "(128,2880) bf16", "(128,) bf16"]
    assert g["rows"][3]["shapes"] == ["(2048,2880) bf16", "(128,2880) bf16", "(128,) bf16"]
    # Cross-row distinctness: the "(640,2880)" shape only appears in
    # k002's row, never bleeds into k001's or k003's row.
    assert "(640,2880) bf16" in g["rows"][1]["shapes"]
    assert "(640,2880) bf16" not in g["rows"][0]["shapes"]
    assert "(640,2880) bf16" not in g["rows"][2]["shapes"]

    # Now render the benchmark cases block from the primary candidate
    # carrying the task_group — this is what the kernel_optimization
    # subprocess sees in build_prompt.
    import importlib

    ko = importlib.import_module("kernel_optimization")
    primary = dict(g["rows"][0])
    primary["task_group"] = g
    block = ko._build_benchmark_cases_block(primary)
    assert "## Benchmark cases" in block
    # Every row produces a distinct ``Case N:`` line, in
    # aggregate-time-descending order.
    assert "Case 1: operation=vllm::rocm_unquantized_gemm" in block
    assert "Case 2: operation=vllm::rocm_unquantized_gemm" in block
    assert "Case 3: operation=vllm::rocm_unquantized_gemm" in block
    assert "Case 4: operation=vllm::rocm_unquantized_gemm" in block
    # Each row's distinct Args appear in its own Case line. The
    # ``(640,2880) bf16`` shape only exists in k002's row, so it must
    # appear in exactly one Case (the second, since k002 is the
    # second-heaviest at 10992 us).
    assert block.count("(640,2880) bf16") == 1
    case2_segment = block.split("Case 2:")[1].split("Case 3:")[0]
    assert "(640,2880) bf16" in case2_segment, (
        "k002's unique shape must land in Case 2 — confirms shape preservation per-row, not cross-row merging"
    )
    # Same for k003's unique ``(2880,512)`` shape → Case 3.
    case3_segment = block.split("Case 3:")[1].split("Case 4:")[0]
    assert "(2880,512) bf16" in case3_segment
    # And k004's unique ``(2048,2880)`` shape → Case 4.
    case4_segment = block.split("Case 4:")[1]
    assert "(2048,2880) bf16" in case4_segment


def test_aggregate_drops_empty_prose_entries(tmp_path):
    """rank=0 / no-prose candidates contribute no entry to ``all_pitem_prose`` (dropped post-process)."""
    src = tmp_path / "rmsnorm.py"
    src.write_text("def rms_norm(x):\n    return x\n", encoding="utf-8")
    cands = [
        {
            "kernel_id": "k001",
            "name": "aiter::rms_norm",
            "duration_us": 100.0,
            "tracelens_launcher_path": f"{src}(1): rms_norm",
            # No P-item rank, no prose — raw-trace fallback shape.
        },
    ]
    groups = tlr.aggregate_by_source_function(cands)
    assert len(groups) == 1
    assert groups[0]["all_pitem_prose"] == []


def test_aggregate_by_source_function_skips_unparseable_launcher_paths():
    """Candidates with empty / em-dash Kernel Path (LLama70B fixture
    shape) produce zero groups — caller falls back to per-kernel."""
    cands = [
        {"kernel_id": "k001", "name": "x", "tracelens_launcher_path": ""},
        {"kernel_id": "k002", "name": "y", "tracelens_launcher_path": "—"},
        # No tracelens_launcher_path field AND no source_file: skipped.
        {"kernel_id": "k003", "name": "z"},
    ]
    assert tlr.aggregate_by_source_function(cands) == []


def test_aggregate_falls_back_to_source_file_when_no_launcher_path():
    """Candidates from raw-trace / csv fallback paths lack
    ``tracelens_launcher_path`` but may carry a Python-shaped path in
    ``source_file``; we still parse those when possible."""
    cands = [
        {
            "kernel_id": "k001",
            "name": "rms_norm",
            "duration_us": 100.0,
            "call_count": 10,
            "source_file": "aiter/rmsnorm.py(42): rms_norm",
        },
    ]
    groups = tlr.aggregate_by_source_function(cands)
    assert len(groups) == 1
    assert groups[0]["function_name"] == "rms_norm"


# ===========================================================================
# task-group over-splitting: native (.cu/.hip/.cpp) kernels have no Python AST
# def-line (TraceLens reports the per-call ``#L`` line), and C++ template/dtype
# mangling varies the name. Native sources key on ``(normalized_op,
# canonical_path)`` only, preserving the invariant that distinct base-name
# kernels sharing a wrapper never merge.
# ===========================================================================
def test_normalize_operation_key_strips_templates():
    """Template/dtype args are dropped; distinct base names stay distinct;
    nested templates are handled; an all-template name falls back to the
    original so a group key never collapses to empty."""
    normalize_operation_key = task_group_contract.normalize_operation_key

    assert normalize_operation_key("rmsnorm_kernel<bf16>") == "rmsnorm_kernel"
    assert normalize_operation_key("rmsnorm_kernel<fp16>") == "rmsnorm_kernel"
    assert normalize_operation_key("foo<bar<baz>>") == "foo"
    # Distinct base names must remain distinct after normalization.
    assert normalize_operation_key("a<x>") != normalize_operation_key("b<x>")
    # No templates: returned verbatim (Q1 base names never change).
    assert normalize_operation_key("vllm::rocm_unquantized_gemm") == "vllm::rocm_unquantized_gemm"
    # Degenerate all-template name: keep original rather than an empty key.
    assert normalize_operation_key("<all>") == "<all>"


def test_operation_key_drops_launch_decoration_for_both_source_kinds():
    """One kernel reached through two launch APIs is one operator.

    The launch API, the C return type and the synthetic-op suffix describe how a
    trace saw the dispatch, so keeping them splits a single kernel into one task
    group per launch path and ports its source once per group.
    """
    normalize_operation_key = task_group_contract.normalize_operation_key
    native_operation_key = task_group_contract.native_operation_key

    graph = "hipGraphLaunch->_mxfp8_linear_kernel (Synthetic Op)"
    module = "hipModuleLaunchKernel->_mxfp8_linear_kernel (Synthetic Op)"
    for key in (normalize_operation_key, native_operation_key):
        assert key(graph) == "_mxfp8_linear_kernel"
        assert key(module) == key(graph)
    # Distinct kernels behind the same launch API stay distinct.
    assert normalize_operation_key("hipGraphLaunch->_first_kernel (Synthetic Op)") != normalize_operation_key(
        "hipGraphLaunch->_second_kernel (Synthetic Op)"
    )
    # Demangling stays native-only and still runs after the shared cleanup.
    assert native_operation_key("void _ZN4vllm11some_kernelEv (Synthetic Op)") == "vllm::some_kernel"


def test_py_task_group_merges_launch_paths_of_one_kernel():
    """Two launch paths of one Triton kernel aggregate into a single group."""
    source = str(Path(__file__).resolve())
    rows = [
        {
            "kernel_id": "k001",
            "name": "hipGraphLaunch->_mxfp8_linear_kernel (Synthetic Op)",
            "source_file": source,
            "reusable_native_kernel": True,
            "gpu_pct": 6.637,
            "duration_us": 84817.0,
            "call_count": 7440,
        },
        {
            "kernel_id": "k002",
            "name": "hipModuleLaunchKernel->_mxfp8_linear_kernel (Synthetic Op)",
            "source_file": source,
            "reusable_native_kernel": True,
            "gpu_pct": 2.12,
            "duration_us": 27152.0,
            "call_count": 126,
        },
    ]
    groups = tla.build_task_groups(rows)
    assert len(groups) == 1
    assert groups[0]["kernel_ids"] == ["k001", "k002"]
    assert groups[0]["operation_key"] == "_mxfp8_linear_kernel"
    assert groups[0]["aggregate_call_count"] == 7566


def test_is_native_source_detects_device_extensions():
    """Native C/C++/HIP/CUDA suffixes are recognized (case-insensitive);
    Python and unrelated files are not."""
    for p in ("kern.cu", "a/b/kern.cuh", "x.hip", "y.cpp", "Z.CU", "k.cc"):
        assert tlr._is_native_source(p), p
    for p in ("model.py", "wrapper.pyi", "notes.txt", ""):
        assert not tlr._is_native_source(p), p


def test_grep_for_keyword_treats_dash_prefixed_keyword_as_literal(tmp_path):
    """Profiler-derived names can begin with ``-``; grep must not treat them
    as command-line options."""
    src = tmp_path / "kernel.py"
    src.write_text("def uses_dash_prefixed_name():\n    return '--danger'\n", encoding="utf-8")

    tla._GREP_CACHE.clear()
    assert tla._grep_for_keyword("--danger", tmp_path) == [src]


def test_aggregate_merges_native_kernel_across_call_site_lines(tmp_path):
    """A native .cu kernel invoked from two call sites reports two different
    ``#L`` lines (no Python AST def-line exists). Native sources must key on
    ``(op, path)`` only and collapse to one task_group."""
    src = tmp_path / "rmsnorm.cu"
    src.write_text(
        "__global__ void rmsnorm_kernel(float* x) { /* ... */ }\n",
        encoding="utf-8",
    )
    cands = [
        {
            "kernel_id": "k001",
            "name": "rmsnorm_kernel",
            "duration_us": 100.0,
            "call_count": 64,
            "gpu_pct": 5.0,
            "tracelens_launcher_path": f"{src}(120): rmsnorm_kernel",
        },
        {
            "kernel_id": "k002",
            "name": "rmsnorm_kernel",  # same kernel, different call site
            "duration_us": 50.0,
            "call_count": 32,
            "gpu_pct": 2.5,
            "tracelens_launcher_path": f"{src}(456): rmsnorm_kernel",
        },
    ]
    groups = tlr.aggregate_by_source_function(cands)
    assert len(groups) == 1, f"native kernel split across {len(groups)} groups by call-site line — #420 regression"
    g = groups[0]
    assert set(g["kernel_ids"]) == {"k001", "k002"}
    assert g["aggregate_duration_us"] == 150.0
    assert g["aggregate_call_count"] == 96


def test_aggregate_merges_native_template_instances_by_source(tmp_path):
    """Three instantiations of ONE ``__global__`` template
    (``add_rmsnorm_quant_kernel``) in ONE .cu, named with DIFFERENT
    Itanium-mangled symbols and autoresolving to the SAME bare .cu path, must
    collapse into ONE task_group: the mangled operation and per-call line are
    NOT part of the native key."""
    src = tmp_path / "rmsnorm_quant_kernels.cu"
    src.write_text(
        "template <typename DTYPE_I, typename DTYPE_O, int BlockSize,\n"
        "          bool ADD_RESIDUAL, bool FUSE_QUANT>\n"
        "__global__ void add_rmsnorm_quant_kernel() {}\n",
        encoding="utf-8",
    )
    bare = str(src)  # autoresolved .cu carries no "(line): func" suffix
    cands = [
        {  # rmsnorm (mode 1), shape A
            "kernel_id": "k005",
            "name": "_ZN5aiter24add_rmsnorm_quant_kernelIDF16bDF16bLi256ELi16ELb0ELb0EEEvPKT_",
            "duration_us": 300.0,
            "call_count": 30,
            "tracelens_launcher_path": bare,
        },
        {  # rmsnorm (mode 1), shape B -> different BlockSize template arg
            "kernel_id": "k006",
            "name": "_ZN5aiter24add_rmsnorm_quant_kernelIDF16bDF16bLi512ELi16ELb0ELb0EEEvPKT_",
            "duration_us": 200.0,
            "call_count": 20,
            "tracelens_launcher_path": bare,
        },
        {  # add_rmsnorm (mode 2) -> ADD_RESIDUAL=true template arg
            "kernel_id": "k007",
            "name": "_ZN5aiter24add_rmsnorm_quant_kernelIDF16bDF16bLi256ELi16ELb1ELb0EEEvPKT_",
            "duration_us": 100.0,
            "call_count": 10,
            "tracelens_launcher_path": bare,
        },
    ]
    groups = tlr.aggregate_by_source_function(cands)
    assert len(groups) == 1, (
        f"3 instantiations of one __global__ split into {len(groups)} "
        "groups — #420 native grouping must ignore the mangled operation"
    )
    g = groups[0]
    assert set(g["kernel_ids"]) == {"k005", "k006", "k007"}
    assert g["aggregate_duration_us"] == 600.0
    assert g["aggregate_call_count"] == 60
    assert g["source_path"].endswith("rmsnorm_quant_kernels.cu")


def test_aggregate_splits_distinct_native_operators_in_one_source(tmp_path):
    src = tmp_path / "quant_kernels.cu"
    src.write_text(
        "__global__ void quantize_kernel() {}\n__global__ void dequantize_kernel() {}\n",
        encoding="utf-8",
    )
    cands = [
        {
            "kernel_id": "k001",
            "name": "quantize_kernel",
            "duration_us": 100.0,
            "tracelens_launcher_path": str(src),
        },
        {
            "kernel_id": "k002",
            "name": "dequantize_kernel",
            "duration_us": 80.0,
            "tracelens_launcher_path": str(src),
        },
    ]

    groups = tlr.aggregate_by_source_function(cands)

    assert len(groups) == 2
    assert {group["operation_key"] for group in groups} == {
        "quantize_kernel",
        "dequantize_kernel",
    }


def test_aggregate_normalizes_template_dtype_on_python_track(tmp_path):
    """Operation normalization applies to the Python track too: two
    candidates sharing one wrapper whose names differ only by dtype
    template args merge. Path/line/fn are identical here, so this
    isolates normalization from the native call-site-line rule."""
    src = tmp_path / "layer.py"
    src.write_text("def forward(x):\n    return x\n", encoding="utf-8")
    launcher = f"{src}(1): forward"
    cands = [
        {
            "kernel_id": "k001",
            "name": "fused_moe_kernel<bf16>",
            "duration_us": 70.0,
            "call_count": 7,
            "tracelens_launcher_path": launcher,
        },
        {
            "kernel_id": "k002",
            "name": "fused_moe_kernel<fp16>",
            "duration_us": 30.0,
            "call_count": 3,
            "tracelens_launcher_path": launcher,
        },
    ]
    groups = tlr.aggregate_by_source_function(cands)
    assert len(groups) == 1
    assert set(groups[0]["kernel_ids"]) == {"k001", "k002"}


def test_aggregate_canonicalizes_native_source_path():
    """The same .cu file reached via a non-normalized path
    (``sub/../rmsnorm.cu``) and a clean path must canonicalize to one
    group rather than splitting on the literal path string."""
    cands = [
        {
            "kernel_id": "k001",
            "name": "rmsnorm_kernel",
            "duration_us": 40.0,
            "call_count": 4,
            "tracelens_launcher_path": "csrc/sub/../rmsnorm.cu(12): rmsnorm_kernel",
        },
        {
            "kernel_id": "k002",
            "name": "rmsnorm_kernel",
            "duration_us": 10.0,
            "call_count": 1,
            "tracelens_launcher_path": "csrc/rmsnorm.cu(99): rmsnorm_kernel",
        },
    ]
    groups = tlr.aggregate_by_source_function(cands)
    assert len(groups) == 1, "path spellings of one .cu must canonicalize"
    assert set(groups[0]["kernel_ids"]) == {"k001", "k002"}


# build_task_groups (tracelens_analysis.py wrapper)
# ===========================================================================
def test_build_task_groups_filters_non_reusable():
    """build_task_groups skips candidates with reusable_native_kernel=False
    so vendor / aten:: / runtime-generated kernels never appear in a
    group's kernel_ids."""
    cands = [
        {
            "kernel_id": "k001",
            "name": "rms_norm",
            "duration_us": 50.0,
            "call_count": 4,
            "tracelens_launcher_path": "aiter/rmsnorm.py(1): rms_norm",
            "reusable_native_kernel": True,
        },
        {
            "kernel_id": "k002",
            "name": "rocblas_sgemm",
            "duration_us": 80.0,
            "call_count": 2,
            "tracelens_launcher_path": "aiter/rmsnorm.py(1): rms_norm",
            "reusable_native_kernel": False,  # filtered out
        },
    ]
    groups = tla.build_task_groups(cands)
    assert len(groups) == 1
    assert groups[0]["kernel_ids"] == ["k001"]
    assert "k002" not in groups[0]["kernel_ids"]


# summary.json carries task_groups[] view.
def test_build_audit_summary_includes_task_groups():
    summary = tla.build_audit_summary(
        candidates=[],
        trace_input="/tmp/x.json.gz",
        task_groups=[
            {
                "task_group_id": "tg001",
                "source_path": "/foo/x.py",
                "definition_line": 10,
                "function_name": "rms",
                "primary_kernel_id": "k001",
                "kernel_ids": ["k001", "k002"],
                "rows": [{"_": "row1"}, {"_": "row2"}],
                "aggregate_duration_us": 123.4,
                "aggregate_call_count": 96,
                "aggregate_gpu_pct": 7.5,
            },
        ],
    )
    assert summary["task_group_count"] == 1


# _default_workspace_path — USER_DATA_PATH rollout for TraceLens ; locks the fallback precedence.
def test_default_workspace_path_prefers_user_data_path(monkeypatch):
    """USER_DATA_PATH wins over both WORKSPACE_PATH and the hard-coded default."""
    monkeypatch.setenv("USER_DATA_PATH", "/some/user/data")
    monkeypatch.setenv("WORKSPACE_PATH", "/some/legacy/workspace")
    assert tla._default_workspace_path() == "/some/user/data"


def test_default_workspace_path_falls_back_to_workspace_path(monkeypatch):
    """When USER_DATA_PATH is unset, WORKSPACE_PATH is honoured (backwards compat)."""
    monkeypatch.delenv("USER_DATA_PATH", raising=False)
    monkeypatch.setenv("WORKSPACE_PATH", "/legacy/workspace")
    assert tla._default_workspace_path() == "/legacy/workspace"


def test_default_workspace_path_final_fallback_to_hyperloom_default(monkeypatch):
    """No envs set → delegates to _paths, which adapts to the host."""
    from hyperloom.agents.kernel.tools import _paths

    monkeypatch.delenv("USER_DATA_PATH", raising=False)
    monkeypatch.delenv("WORKSPACE_PATH", raising=False)
    assert tla._default_workspace_path() == _paths.default_workspace_root()


def test_default_workspace_path_treats_empty_user_data_path_as_unset(monkeypatch):
    """An empty USER_DATA_PATH must not shadow a real WORKSPACE_PATH; ``or`` semantics."""
    monkeypatch.setenv("USER_DATA_PATH", "")
    monkeypatch.setenv("WORKSPACE_PATH", "/legacy/workspace")
    assert tla._default_workspace_path() == "/legacy/workspace"


# Idle-% sanity gate on the Executive Summary. High idle => pivot to params;
# default threshold 80% (overridable via HYPERLOOM_TRACELENS_IDLE_PCT_THRESHOLD).

_EXEC_SUMMARY_LOW_IDLE = """\
# Workload Analysis

## Executive Summary

A single-rank trace.

| Metric | Value |
|--------|-------|
| Total Time | 1234.5 ms |
| Compute % | 99.30% |
| Idle % | 0.25% |
| Exposed Communication % | 0.42% |
"""

_EXEC_SUMMARY_HIGH_IDLE = """\
# Workload Analysis

## Executive Summary

A single-rank trace that's mostly waiting on the host.

| Metric | Value |
|--------|-------|
| Total Time | 9999.9 ms |
| Compute % | 30.00% |
| Idle % | 60.50% |
| Exposed Communication % | 9.50% |
"""

_EXEC_SUMMARY_NO_IDLE_ROW = """\
# Workload Analysis

## Executive Summary

| Metric | Value |
|--------|-------|
| Total Time | 1234.5 ms |
| Compute % | 99.30% |
"""


def test_extract_idle_pct_parses_low_idle_row(tmp_path):
    md = tmp_path / "analysis.md"
    md.write_text(_EXEC_SUMMARY_LOW_IDLE, encoding="utf-8")
    assert tlr.extract_idle_pct_from_analysis_md(md) == pytest.approx(0.25)


def test_extract_idle_pct_parses_high_idle_row(tmp_path):
    md = tmp_path / "analysis.md"
    md.write_text(_EXEC_SUMMARY_HIGH_IDLE, encoding="utf-8")
    assert tlr.extract_idle_pct_from_analysis_md(md) == pytest.approx(60.5)


def test_extract_idle_pct_returns_none_when_no_idle_row(tmp_path):
    """Reports without an Idle % row degrade to ``None`` so the gate skips rather than fails."""
    md = tmp_path / "analysis.md"
    md.write_text(_EXEC_SUMMARY_NO_IDLE_ROW, encoding="utf-8")
    assert tlr.extract_idle_pct_from_analysis_md(md) is None


def test_extract_idle_pct_returns_none_when_file_missing(tmp_path):
    assert tlr.extract_idle_pct_from_analysis_md(tmp_path / "nope.md") is None


def test_extract_idle_pct_against_llama70b_fixture():
    """Llama 3 70B fixture has Idle % = 0.25%; pins the regex against drift."""
    fixture = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "tracelens_v03_llama70b_analysis.md"
    assert fixture.exists(), f"fixture must be present: {fixture}"
    assert tlr.extract_idle_pct_from_analysis_md(fixture) == pytest.approx(0.25)


def test_resolve_idle_pct_threshold_uses_default_when_env_unset(monkeypatch):
    monkeypatch.delenv(idle_gate.HIGH_IDLE_PCT_THRESHOLD_ENV, raising=False)
    assert idle_gate.resolve_idle_pct_threshold() == idle_gate.HIGH_IDLE_PCT_THRESHOLD_DEFAULT


def test_resolve_idle_pct_threshold_honours_env_override(monkeypatch):
    monkeypatch.setenv(idle_gate.HIGH_IDLE_PCT_THRESHOLD_ENV, "35.5")
    assert idle_gate.resolve_idle_pct_threshold() == pytest.approx(35.5)


def test_resolve_idle_pct_threshold_rejects_nonsense_env_value(monkeypatch):
    """Garbage / negative / empty env values fall back to the default, not a crash."""
    monkeypatch.setenv(idle_gate.HIGH_IDLE_PCT_THRESHOLD_ENV, "not-a-float")
    assert idle_gate.resolve_idle_pct_threshold() == idle_gate.HIGH_IDLE_PCT_THRESHOLD_DEFAULT
    monkeypatch.setenv(idle_gate.HIGH_IDLE_PCT_THRESHOLD_ENV, "-5")
    assert idle_gate.resolve_idle_pct_threshold() == idle_gate.HIGH_IDLE_PCT_THRESHOLD_DEFAULT
    monkeypatch.setenv(idle_gate.HIGH_IDLE_PCT_THRESHOLD_ENV, "")
    assert idle_gate.resolve_idle_pct_threshold() == idle_gate.HIGH_IDLE_PCT_THRESHOLD_DEFAULT


def test_build_high_idle_warning_shape(tmp_path):
    """Pin the high-idle warning shape: code/severity/idle_pct/threshold_pct (rounded)/source/message."""
    report = tmp_path / "analysis.md"
    report.write_text("# noop\n", encoding="utf-8")
    w = tla._build_high_idle_warning(
        idle_pct=42.567,
        threshold_pct=20.0,
        report_path=report,
    )
    assert w["code"] == "high_gpu_idle_pct"
    assert w["severity"] == "warning"
    assert w["idle_pct"] == pytest.approx(42.57)
    assert w["threshold_pct"] == pytest.approx(20.0)
    assert w["source"] == str(report)
    # Message formats both numbers via :.2f.
    assert "42.57%" in w["message"]
    assert "20.00%" in w["message"]
    assert "parameter optimization" in w["message"]


def test_build_audit_summary_propagates_trace_health_warnings():
    """``summary.json`` surfaces the same trace_health_warnings as the live result."""
    warnings = [
        {
            "code": "high_gpu_idle_pct",
            "severity": "warning",
            "idle_pct": 35.0,
            "threshold_pct": 20.0,
            "source": "/tmp/x/analysis.md",
            "message": "test",
        }
    ]
    summary = tla.build_audit_summary(
        [],
        trace_input="/tmp/trace.json.gz",
        framework="sglang",
        target_platform="MI300X",
        task_groups=[],
        trace_health_warnings=warnings,
    )
    assert summary["trace_health_warnings"] == warnings
    assert summary["task_count"] == 0
    assert summary["skipped_count"] == 0


def test_build_audit_summary_defaults_trace_health_warnings_to_empty_list():
    """No findings is the empty list, never ``None`` (consumers iterate without a guard)."""
    summary = tla.build_audit_summary(
        [],
        trace_input="/tmp/trace.json.gz",
        framework="sglang",
        target_platform="MI300X",
        task_groups=[],
    )
    assert summary["trace_health_warnings"] == []


# atom maps to ``inference`` analysis mode
def test_infer_analysis_mode_atom_returns_inference():
    """atom shares inference-mode kernel grouping with sglang/vllm (else falls to ``default``)."""
    assert tlr.infer_analysis_mode("atom", "") == "inference"
    assert tlr.infer_analysis_mode("ATOM", "default") == "inference"
    assert tlr.infer_analysis_mode("  atom  ", "default") == "inference"


def test_infer_analysis_mode_sglang_vllm_unchanged():
    """Regression guard: sglang / vllm remain ``inference``."""
    assert tlr.infer_analysis_mode("sglang", "") == "inference"
    assert tlr.infer_analysis_mode("vllm", "default") == "inference"


def test_infer_analysis_mode_explicit_request_wins():
    """Caller-supplied non-default mode bypasses the framework default for every framework."""
    assert tlr.infer_analysis_mode("atom", "training") == "training"
    assert tlr.infer_analysis_mode("sglang", "training") == "training"
    assert tlr.infer_analysis_mode("unknown", "training") == "training"


def test_infer_analysis_mode_unknown_framework_stays_default():
    """Frameworks outside the canonical set fall back to ``default``."""
    assert tlr.infer_analysis_mode("trtllm", "") == "default"
    assert tlr.infer_analysis_mode("", "") == "default"


# atom entries present in _FRAMEWORK_PKG_FALLBACK_ROOTS
def test_framework_pkg_fallback_roots_has_atom_entry():
    """The atom fallback roots must include ``/app/ATOM`` plus a site-packages variant."""
    table = tlr._FRAMEWORK_PKG_FALLBACK_ROOTS
    assert "atom" in table, "atom missing from _FRAMEWORK_PKG_FALLBACK_ROOTS"
    roots = table["atom"]
    assert "/app/ATOM" in roots
    assert any("site-packages" in r or "dist-packages" in r for r in roots), (
        f"atom fallback roots lack a site-packages entry: {roots!r}"
    )


def test_resolve_launcher_via_atom_fallback_root(tmp_path, monkeypatch):
    """End-to-end: a relative atom path resolves against the atom fallback root when import atom doesn't fire."""
    monkeypatch.delenv(tlr._FRAMEWORK_SOURCE_ROOTS_ENV, raising=False)
    _seed_pkg(
        tmp_path,
        "atom",
        "model_engine/model_runner.py",
        funcs=("forward",),
    )
    monkeypatch.setattr(
        tlr,
        "_FRAMEWORK_PKG_FALLBACK_ROOTS",
        {"atom": (str(tmp_path),)},
    )
    # Force find_spec to miss so the test exercises the fallback path.
    monkeypatch.setattr(tlr, "_package_root_parent", lambda pkg: None)

    resolved = tlr._resolve_launcher_to_abs_source(
        "atom/model_engine/model_runner.py(125): forward",
    )
    assert resolved is not None
    abs_path, line, func = resolved
    assert abs_path == str(tmp_path / "atom" / "model_engine" / "model_runner.py")
    assert line == 125
    assert func == "forward"


# ---------------------------------------------------------------------------
# _extract_total_time_us_from_gpu_timeline
# ---------------------------------------------------------------------------


def test_extract_total_time_us_from_gpu_timeline(tmp_path):
    csv_dir = tmp_path / "perf_report_csvs"
    csv_dir.mkdir()
    (csv_dir / "gpu_timeline.csv").write_text(
        "type,time ms,percent\ncompute_time,100.5,80.0\ntotal_time,125.0,100.0\nidle_time,24.5,20.0\n",
        encoding="utf-8",
    )
    result = tla._extract_total_time_us_from_gpu_timeline(tmp_path)
    assert result == 125000.0


def test_extract_total_time_us_returns_none_when_missing(tmp_path):
    assert tla._extract_total_time_us_from_gpu_timeline(tmp_path) is None


# ---------------------------------------------------------------------------
# Low-compute gate: gpu_timeline readers + gate evaluation
# ---------------------------------------------------------------------------


def _write_gpu_timeline(tmp_path, body: str):
    csv_dir = tmp_path / "perf_report_csvs"
    csv_dir.mkdir(exist_ok=True)
    (csv_dir / "gpu_timeline.csv").write_text(body, encoding="utf-8")


def test_total_time_is_none_when_its_cell_is_unreadable(tmp_path):
    """Same fail-open contract for the window total the gpu_pct basis needs."""
    _write_gpu_timeline(tmp_path, "type,time ms,percent\ntotal_time,,100.0\n")
    assert tla._extract_total_time_us_from_gpu_timeline(tmp_path) is None


def test_low_compute_gate_fires_on_spin_wait_window(monkeypatch, tmp_path):
    """GLM-5.2 regression: 3.99% compute / 0.02% idle must not pass as healthy.

    A collective that spin-waits on peer ranks is charged as GPU-busy, so the
    idle gate sees 0.02% and lets the window through. The compute share is what
    exposes it.
    """
    monkeypatch.delenv(idle_gate.LOW_COMPUTE_PCT_THRESHOLD_ENV, raising=False)
    threshold, warning = tla._evaluate_low_compute_gate(3.99, 95.99, tmp_path / "analysis.md")
    assert threshold == 10.0
    assert warning is not None
    assert warning["code"] == "low_gpu_compute_pct"
    assert warning["compute_pct"] == 3.99
    assert warning["exposed_comm_pct"] == 95.99


def test_low_compute_gate_passes_healthy_window(monkeypatch, tmp_path):
    monkeypatch.delenv(idle_gate.LOW_COMPUTE_PCT_THRESHOLD_ENV, raising=False)
    _, warning = tla._evaluate_low_compute_gate(99.3, 0.42, tmp_path / "analysis.md")
    assert warning is None


def test_low_compute_gate_skipped_when_pct_unknown(monkeypatch, tmp_path):
    """No compute row in the report: fail open rather than suppress candidates."""
    monkeypatch.delenv(idle_gate.LOW_COMPUTE_PCT_THRESHOLD_ENV, raising=False)
    _, warning = tla._evaluate_low_compute_gate(None, None, tmp_path / "analysis.md")
    assert warning is None


def test_high_idle_gate_skips_when_idle_pct_unknown(monkeypatch, tmp_path):
    """A fallback report has no Idle % row, so ``extract_idle_pct_from_analysis_md``
    returns None; the gate must then return no warning (the skip path that the
    deleted raw-trace guard used to defend). Candidates are NOT blanked."""
    monkeypatch.delenv(idle_gate.HIGH_IDLE_PCT_THRESHOLD_ENV, raising=False)
    threshold, warning = tla._evaluate_high_idle_gate(None, tmp_path / "analysis.md")
    assert threshold == 80.0
    assert warning is None


def test_high_idle_gate_fires_above_threshold(monkeypatch, tmp_path):
    """A healthy report with a high Idle % still suppresses: B1 did not disarm
    the gate on the case it is actually for."""
    monkeypatch.delenv(idle_gate.HIGH_IDLE_PCT_THRESHOLD_ENV, raising=False)
    threshold, warning = tla._evaluate_high_idle_gate(95.0, tmp_path / "analysis.md")
    assert threshold == 80.0
    assert warning is not None
    assert warning["code"] == "high_gpu_idle_pct"


def test_extract_compute_and_comm_pct_from_analysis_md(tmp_path):
    md = tmp_path / "analysis.md"
    md.write_text(
        "# Analysis\n\n"
        "| Metric | Value |\n"
        "|--------|-------|\n"
        "| Total Time | 18186.60 ms |\n"
        "| Compute % | 3.99% |\n"
        "| Idle % | 0.02% |\n"
        "| Exposed Communication % | 95.99% |\n",
        encoding="utf-8",
    )
    assert tlr.extract_compute_pct_from_analysis_md(md) == 3.99
    assert tlr.extract_exposed_comm_pct_from_analysis_md(md) == 95.99
    assert tlr.extract_idle_pct_from_analysis_md(md) == 0.02


def test_extract_compute_pct_from_analysis_md_missing_row(tmp_path):
    md = tmp_path / "analysis.md"
    md.write_text("# Analysis\n\n| Idle % | 1.0% |\n", encoding="utf-8")
    assert tlr.extract_compute_pct_from_analysis_md(md) is None
    assert tlr.extract_exposed_comm_pct_from_analysis_md(md) is None


# ---------------------------------------------------------------------------
# _normalize_profiler_op_name / graph-captured keyword recovery
# ---------------------------------------------------------------------------


def test_normalize_profiler_op_name_strips_graph_wrappers():
    n = tla._normalize_profiler_op_name
    # Launch wrapper + return type + template are peeled to the bare symbol.
    assert n("hipGraphLaunch->void ck::foo_kernel<int>") == "ck::foo_kernel<int>"
    # Plain symbol with no namespace keeps the identifier after stripping void.
    assert (
        n("hipGraphLaunch->void paged_attention_ll4mi_QKV_mfma16_kernel<x>")
        == "paged_attention_ll4mi_QKV_mfma16_kernel<x>"
    )
    # Embedded Itanium-mangled symbol is surfaced for the _Z demangle path.
    assert (
        n("hipGraphLaunch->_ZN5aiter37dynamic_per_group_scaled_quant_fp8E")
        == "_ZN5aiter37dynamic_per_group_scaled_quant_fp8E"
    )
    # .kd suffix and "(Synthetic Op)" annotation are removed.
    assert n("hipGraphLaunch->triton_poi_fused_2.kd (Synthetic Op)") == "triton_poi_fused_2"
    # Already-clean names pass through unchanged (regression guard).
    assert n("aten::mm") == "aten::mm"
    assert (
        n("sglang_profiler::fused_moe_triton_kernels_invoke_fused_moe_kernel_427")
        == "sglang_profiler::fused_moe_triton_kernels_invoke_fused_moe_kernel_427"
    )


def test_candidate_keywords_recovers_graph_captured_symbols():
    # Before normalization these kept the "hipGraphLaunch->void " prefix and
    # greped to nothing; now they yield the real kernel identifier.
    kws = tla._candidate_keywords("hipGraphLaunch->void paged_attention_ll4mi_QKV_mfma16_kernel<x>")
    assert "paged_attention_ll4mi_QKV_mfma16_kernel" in kws

    kws = tla._candidate_keywords("hipGraphLaunch->_ZN5aiter37dynamic_per_group_scaled_quant_fp8E")
    assert "dynamic_per_group_scaled_quant_fp8E" in kws

    # Clean profiler symbol still resolves to the same keyword as before.
    kws = tla._candidate_keywords("sglang_profiler::fused_moe_triton_kernels_invoke_fused_moe_kernel_427")
    assert kws == ["fused_moe_triton_kernels_invoke_fused_moe_kernel_427"]


# --- pretrim_startup_transient (#profiler-start transient) -------------------


def _write_trace(
    path: Path,
    step_durs,
    *,
    step_us_gap=0.0,
    host_lead_us=30_000.0,
    phases=None,
    unique_names=True,
    omit_host_steps=(),
    omit_device_steps=(),
    extra_events=None,
):
    """Build a minimal torch-profiler trace with GPU step annotations.

    ``step_durs`` are microsecond durations laid end to end; each becomes one
    ``step[<phase> ...]`` gpu_user_annotation, its host-side twin, and a kernel
    inside it, so the trimmer has both something to measure and something to
    drop. ``phases`` supplies a per-step phase token (default ``DECODE`` for
    every step).

    ``unique_names=False`` omits the per-step ``g_sk`` field, reproducing the
    framework builds whose step annotations repeat verbatim; the two timelines
    then cannot be paired by name. ``omit_host_steps`` and ``omit_device_steps``
    drop one side's annotation for the given step indices, which is how real
    captures arrive: most measured rank traces carry every host ``step[...]``
    against only a handful of device ones.
    """
    # torch stamps metadata with the profiler-open ts, i.e. always before any
    # cut point -- reproduce that, it is what made a naive ts filter drop it.
    events = [
        {"ph": "M", "name": "process_name", "ts": 999_999.0, "pid": 1, "args": {"name": "python"}},
        {"ph": "M", "name": "thread_name", "ts": 999_999.0, "pid": 1, "tid": 7, "args": {"name": "t"}},
    ]
    ts = 1_000_000.0
    for i, dur in enumerate(step_durs):
        phase = (phases or ["DECODE"] * len(step_durs))[i]
        name = f"step[{phase} bs=64 g_sk={i}]" if unique_names else f"step[{phase} bs=64]"
        # The host runs a step ahead: step N's launches are issued while step
        # N-1 still owns the GPU, so the lead is bounded by the *previous*
        # step's duration. Reproduce that offset -- it is what makes a
        # single-timestamp cut unable to separate the two.
        host_ts = ts - min(host_lead_us, step_durs[i - 1]) if i else ts - 1.0
        if i not in omit_host_steps:
            events.append(
                {"ph": "X", "cat": "user_annotation", "name": name, "pid": 1, "tid": 1, "ts": host_ts, "dur": 3.0}
            )
        events.append(
            {"ph": "X", "cat": "cpu_op", "name": f"launch_{i}", "pid": 1, "tid": 1, "ts": host_ts + 0.1, "dur": 1.0}
        )
        if i not in omit_device_steps:
            events.append(
                {
                    "ph": "X",
                    "cat": "gpu_user_annotation",
                    "name": name,
                    "pid": 5,
                    "tid": 7,
                    "ts": ts,
                    "dur": dur,
                }
            )
        events.append(
            {
                "ph": "X",
                "cat": "kernel",
                "name": f"some_kernel_{i}",
                "pid": 5,
                "tid": 7,
                "ts": ts + 1.0,
                "dur": max(1.0, dur / 2),
            }
        )
        ts += dur + step_us_gap
    events.extend(extra_events or [])
    payload = {"schemaVersion": 1, "baseTimeNanoseconds": 12345, "traceEvents": events}
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        json.dump(payload, fh)
    return path


def test_pretrim_drops_leading_barrier_step(tmp_path):
    """A 479x leading step is cut; the steady tail and metadata survive."""
    src = _write_trace(tmp_path / "r.trace.json.gz", [15_781_320.0] + [32_944.0] * 127)
    dst = tmp_path / "r.pretrimmed.trace.json.gz"

    trimmed, report = tla.pretrim_startup_transient(src, dst)

    assert trimmed is True
    assert report["reason"] == "trimmed"
    assert report["dropped_steps"] == 1
    assert report["remaining_steps"] == 127
    assert report["outlier_ratio"] > 400

    out = tla.open_json(dst)
    spans = tla._step_annotation_spans(out["traceEvents"])
    assert len(spans) == 127
    assert max(dur for _, dur, _ in spans) < 40_000.0  # the barrier step is gone
    # ph:"M" metadata has no ts and must be preserved, else the chunk loses its
    # process/thread names and TraceLens can't attribute anything.
    assert sum(1 for ev in out["traceEvents"] if ev.get("ph") == "M") == 2
    assert out["baseTimeNanoseconds"] == 12345


def test_pretrim_noop_on_clean_trace(tmp_path):
    """Uniform steps are left alone and no output file is written."""
    src = _write_trace(tmp_path / "r.trace.json.gz", [32_900.0, 33_100.0] * 32)
    dst = tmp_path / "r.pretrimmed.trace.json.gz"

    trimmed, report = tla.pretrim_startup_transient(src, dst)

    assert trimmed is False
    assert report["reason"] == "no_leading_outlier"
    assert not dst.exists()


def test_pretrim_keeps_midstream_outlier(tmp_path):
    """Only a *leading* run is cut; a slow step in the middle is real behaviour."""
    durs = [32_944.0] * 10 + [5_000_000.0] + [32_944.0] * 10
    src = _write_trace(tmp_path / "r.trace.json.gz", durs)
    dst = tmp_path / "r.pretrimmed.trace.json.gz"

    trimmed, report = tla.pretrim_startup_transient(src, dst)

    assert trimmed is False
    assert report["reason"] == "no_leading_outlier"


def test_pretrim_refuses_when_too_many_leading_outliers(tmp_path):
    """A run that never reaches steady state is surfaced, not trimmed away."""
    durs = [5_000_000.0] * 6 + [32_944.0] * 20
    src = _write_trace(tmp_path / "r.trace.json.gz", durs)
    dst = tmp_path / "r.pretrimmed.trace.json.gz"

    trimmed, report = tla.pretrim_startup_transient(src, dst)

    assert trimmed is False
    assert report["reason"] == "too_many_leading_outliers"
    assert report["leading_outliers"] == 6
    assert not dst.exists()


def test_pretrim_skips_short_traces(tmp_path):
    """Below the step floor the median is not worth trusting."""
    src = _write_trace(tmp_path / "r.trace.json.gz", [9_000_000.0] + [32_944.0] * 3)
    dst = tmp_path / "r.pretrimmed.trace.json.gz"

    trimmed, report = tla.pretrim_startup_transient(src, dst)

    assert trimmed is False
    assert report["reason"] == "too_few_steps"


def test_pretrim_degrades_on_unreadable_trace(tmp_path):
    """An unreadable trace must fall through to the raw path, not raise."""
    bad = tmp_path / "bad.trace.json.gz"
    bad.write_bytes(b"not a gzip")

    trimmed, report = tla.pretrim_startup_transient(bad, tmp_path / "out.trace.json.gz")

    assert trimmed is False
    assert report["reason"] == "unreadable_trace"


def test_pretrim_ignores_traces_without_step_annotations(tmp_path):
    """No step annotations -> nothing to measure against."""
    payload = {"traceEvents": [{"ph": "X", "cat": "kernel", "name": "k", "ts": 1.0, "dur": 5.0} for _ in range(50)]}
    src = tmp_path / "r.trace.json.gz"
    with gzip.open(src, "wt", encoding="utf-8") as fh:
        json.dump(payload, fh)

    trimmed, report = tla.pretrim_startup_transient(src, tmp_path / "out.trace.json.gz")

    assert trimmed is False
    assert report["reason"] == "too_few_steps"
    assert report["steps"] == 0


def test_pretrim_threshold_is_the_module_default(tmp_path):
    """A step under the default multiple is left alone; over it is cut."""
    dst = tmp_path / "r.pretrimmed.trace.json.gz"
    # ~6x the median -- real jitter can reach this, so it must survive.
    under = _write_trace(tmp_path / "under.trace.json.gz", [200_000.0] + [32_944.0] * 20)
    assert tla.pretrim_startup_transient(under, dst)[0] is False
    # ~15x -- past the default, and nowhere near the 479x the real barrier hits.
    over = _write_trace(tmp_path / "over.trace.json.gz", [500_000.0] + [32_944.0] * 20)
    trimmed, report = tla.pretrim_startup_transient(over, dst)
    assert trimmed is True
    assert report["dropped_steps"] == 1
    assert report["outlier_factor"] == tla._PRETRIM_OUTLIER_FACTOR


def test_pretrim_keeps_host_side_of_first_surviving_step(tmp_path):
    """The kept step's host ops survive even though they start inside the
    dropped step's device span.

    Host launches run a step ahead of the device, so a single-timestamp cut on
    the device boundary would strip them -- and with them the shape-carrying
    frames TraceLens attributes MoE kernels through.
    """
    src = _write_trace(
        tmp_path / "r.trace.json.gz",
        [15_781_320.0] + [32_944.0] * 40,
        host_lead_us=839_000.0,
    )
    dst = tmp_path / "r.pretrimmed.trace.json.gz"

    trimmed, report = tla.pretrim_startup_transient(src, dst)
    assert trimmed is True
    # Two cuts, host earlier -- and by the full lead, not a value clamped down
    # to the kept step's own duration.
    assert report["gpu_cut_ts"] - report["cpu_cut_ts"] == pytest.approx(839_000.0)

    ev = tla.open_json(dst)["traceEvents"]
    kept_host = {e["name"] for e in ev if e.get("cat") == "user_annotation"}
    kept_dev = {n for _, _, n in tla._step_annotation_spans(ev)}
    # Same set on both timelines: no step is half-present.
    assert kept_host == kept_dev
    assert "step[DECODE bs=64 g_sk=0]" not in kept_host  # dropped step, both sides
    assert "step[DECODE bs=64 g_sk=1]" in kept_host  # kept step, host side survived
    assert len(kept_host) == 40


def test_pretrim_drops_dropped_step_device_work_after_the_host_cut(tmp_path):
    """Kernels belonging to the dropped step do not leak past the host cut.

    The dropped step's device span extends beyond the kept step's host start, so
    a host-boundary cut alone would leave its trailing kernels behind as orphans.
    """
    src = _write_trace(
        tmp_path / "r.trace.json.gz",
        [15_781_320.0] + [32_944.0] * 40,
        host_lead_us=839_000.0,
    )
    dst = tmp_path / "r.pretrimmed.trace.json.gz"
    tla.pretrim_startup_transient(src, dst)

    ev = tla.open_json(dst)["traceEvents"]
    kernels = {e["name"] for e in ev if e.get("cat") == "kernel"}
    assert "some_kernel_0" not in kernels
    assert "some_kernel_1" in kernels
    assert len(kernels) == 40


def test_pretrim_leaves_enough_steps_for_the_splitter(tmp_path):
    """Step count downstream is set by --num-steps, and the trim keeps well
    clear of it: the splitter still gets its full window, only shifted."""
    src = _write_trace(tmp_path / "r.trace.json.gz", [15_781_320.0] + [32_944.0] * 127)
    dst = tmp_path / "r.pretrimmed.trace.json.gz"

    _, report = tla.pretrim_startup_transient(src, dst)

    assert report["remaining_steps"] == 127
    assert report["remaining_steps"] >= 32  # the default --split-num-steps


def test_pretrim_pairs_timelines_by_position_not_by_name(tmp_path):
    """Repeated step names must still cut both timelines at the right place.

    Framework builds that omit the cumulative-sequence-length fields emit the
    same ``step[DECODE bs=64]`` for every step. Pairing the timelines by name
    resolves the surviving step to the *first* occurrence, which puts the host
    cut at the head of the capture and leaves the dropped step's entire host
    side in the trimmed trace.
    """
    src = _write_trace(
        tmp_path / "r.trace.json.gz",
        [15_781_320.0] + [32_944.0] * 40,
        host_lead_us=839_000.0,
        unique_names=False,
    )
    dst = tmp_path / "r.pretrimmed.trace.json.gz"

    trimmed, report = tla.pretrim_startup_transient(src, dst)
    assert trimmed is True
    assert report["gpu_cut_ts"] - report["cpu_cut_ts"] == pytest.approx(839_000.0)

    ev = tla.open_json(dst)["traceEvents"]
    # One host annotation per surviving device step, and no more: the dropped
    # step's host side went with its device side.
    assert len(tla._step_annotation_spans(ev)) == 40
    assert sum(1 for e in ev if e.get("cat") == "user_annotation") == 40
    # launch_0 is the dropped step's host op; it must not survive on the
    # strength of a name it shares with every other step.
    launches = {e["name"] for e in ev if e.get("cat") == "cpu_op"}
    assert "launch_0" not in launches
    assert "launch_1" in launches


def test_pretrim_refuses_when_the_timelines_hold_different_step_counts(tmp_path):
    """Unequal step counts make positional pairing meaningless, either way round.

    Falling back to a single cut is the outcome the split-cut design exists to
    avoid, so the trim is refused and the counts recorded instead.
    """
    src = _write_trace(
        tmp_path / "fewer_host.trace.json.gz",
        [15_781_320.0] + [32_944.0] * 20,
        omit_host_steps=(1, 2),
    )
    dst = tmp_path / "r.pretrimmed.trace.json.gz"

    trimmed, report = tla.pretrim_startup_transient(src, dst)

    assert trimmed is False
    assert report["reason"] == "timeline_step_count_mismatch"
    assert (report["steps"], report["host_steps"]) == (21, 19)
    assert not dst.exists()


def test_pretrim_refuses_when_device_step_annotations_are_missing(tmp_path):
    """More host steps than device steps is the shape real captures arrive in.

    Measured rank traces carry every host ``step[...]`` against a fraction of
    the device ones, and the gaps are not at the tail. Indexing the host list by
    the device position then resolves to some earlier step, putting the host cut
    before the transient and leaving its host side in the window -- and a guard
    that only rejects *too few* host entries lets it through, because there are
    more of them, not fewer.
    """
    src = _write_trace(
        tmp_path / "fewer_device.trace.json.gz",
        [15_781_320.0] + [32_944.0] * 20,
        omit_device_steps=(5, 9, 13),
    )
    dst = tmp_path / "r.pretrimmed.trace.json.gz"

    trimmed, report = tla.pretrim_startup_transient(src, dst)

    assert trimmed is False
    assert report["reason"] == "timeline_step_count_mismatch"
    assert (report["steps"], report["host_steps"]) == (18, 21)
    assert not dst.exists()


def test_pretrim_keeps_leading_prefill_steps(tmp_path):
    """Prefill steps are not transients just because decode dominates the trace.

    A mixed capture's EXTEND steps run one to two orders of magnitude longer
    than its DECODE steps. Measured against a whole-trace median they all read
    as outliers, and the leading ones -- the window
    ``--steady-state-mode=prefilldecode`` exists to analyse -- get trimmed away.
    """
    durs = [1_200_000.0, 880_000.0, 1_530_000.0] + [32_944.0] * 30
    phases = ["EXTEND"] * 3 + ["DECODE"] * 30
    src = _write_trace(tmp_path / "r.trace.json.gz", durs, phases=phases)
    dst = tmp_path / "r.pretrimmed.trace.json.gz"

    trimmed, report = tla.pretrim_startup_transient(src, dst)

    assert trimmed is False
    assert report["reason"] == "no_leading_outlier"
    # Each phase gets its own baseline rather than one median for the trace.
    assert set(report["phase_medians_ms"]) == {"EXTEND", "DECODE"}
    assert report["first_step_phase"] == "EXTEND"
    assert report["first_step_ratio"] < tla._PRETRIM_OUTLIER_FACTOR
    assert not dst.exists()


def test_pretrim_drops_a_prefill_transient_against_its_own_phase(tmp_path):
    """The guard still fires when the transient lands on a prefill step.

    Per-phase baselines must not amount to exempting prefill: a barrier that
    opens on an EXTEND step is as fatal to the window as one on a DECODE step.
    """
    durs = [15_781_320.0, 1_200_000.0, 1_150_000.0, 1_180_000.0] + [32_944.0] * 30
    phases = ["EXTEND"] * 4 + ["DECODE"] * 30
    src = _write_trace(tmp_path / "r.trace.json.gz", durs, phases=phases)
    dst = tmp_path / "r.pretrimmed.trace.json.gz"

    trimmed, report = tla.pretrim_startup_transient(src, dst)

    assert trimmed is True
    assert report["dropped_steps"] == 1
    assert report["dropped_phase"] == "EXTEND"
    # Measured against EXTEND's 1.2 s median (13x) rather than decode's 33 ms,
    # which would have made it 479x and swept the healthy prefill steps with it.
    assert report["outlier_ratio"] == pytest.approx(15_781_320.0 / 1_200_000.0, rel=1e-3)
    assert report["phase_medians_ms"]["EXTEND"] == pytest.approx(1_200.0)
    assert report["remaining_steps"] == 33


def test_pretrim_leaves_a_phase_without_a_baseline_alone(tmp_path):
    """A leading step whose phase is too rare to have a median is not dropped.

    With no population of its own to compare against there is no evidence the
    step is abnormal, so it stays and the window keeps it.
    """
    durs = [9_000_000.0] + [32_944.0] * 30
    phases = ["EXTEND"] + ["DECODE"] * 30
    src = _write_trace(tmp_path / "r.trace.json.gz", durs, phases=phases)
    dst = tmp_path / "r.pretrimmed.trace.json.gz"

    trimmed, report = tla.pretrim_startup_transient(src, dst)

    assert trimmed is False
    assert report["reason"] == "no_leading_outlier"
    assert "EXTEND" not in report["phase_medians_ms"]
    assert report["first_step_ratio"] is None


def test_pretrim_reports_the_worst_ratio_across_dropped_steps(tmp_path):
    """``outlier_ratio`` covers the whole dropped run, not just its first step."""
    durs = [500_000.0, 5_000_000.0] + [32_944.0] * 20
    src = _write_trace(tmp_path / "r.trace.json.gz", durs)
    dst = tmp_path / "r.pretrimmed.trace.json.gz"

    trimmed, report = tla.pretrim_startup_transient(src, dst)

    assert trimmed is True
    assert report["dropped_steps"] == 2
    assert report["outlier_ratio"] == pytest.approx(5_000_000.0 / 32_944.0, rel=1e-3)


def test_pretrim_write_failure_is_reported_as_a_failure(tmp_path, monkeypatch):
    """A failed write must not be reported with the ``trimmed`` reason.

    The caller keys its log line and the pretrim artifact off ``reason``; a
    report that still says ``trimmed`` after the write failed makes a full-disk
    run look like a successful trim that simply was not applied.
    """
    src = _write_trace(tmp_path / "r.trace.json.gz", [15_781_320.0] + [32_944.0] * 40)
    dst = tmp_path / "out" / "r.pretrimmed.trace.json.gz"

    def _boom(*_args, **_kwargs):
        raise OSError("No space left on device")

    # Fails after the output file is opened, so a partial one is on disk.
    monkeypatch.setattr(tla.json, "dump", _boom)
    trimmed, report = tla.pretrim_startup_transient(src, dst)

    assert trimmed is False
    assert report["reason"] == "write_failed"
    assert "No space left on device" in report["error"]
    # The step accounting is kept for diagnosis, but no partial file is left
    # behind for a later step to mistake for a usable trace.
    assert report["dropped_steps"] == 1
    assert not dst.exists()


def test_pretrim_redirects_only_the_splitter_input(tmp_path):
    """The splitter reads the trimmed copy; nothing else is repointed at it.

    ``analysis_trace_path`` is what capture-folder discovery and the split
    warnings resolve against, so reassigning it to the derived file under
    ``trace_split/`` moves both onto a directory that holds neither the capture
    nor its graph-capture sidecar.
    """
    captured, trace = _drive_main_capturing_subprocess(
        tmp_path,
        [],
        trace_factory=lambda root: _write_trace(root / "trace.trace.json.gz", [15_781_320.0] + [32_944.0] * 40),
    )

    splitter_cmd = _find_splitter_cmd(captured)
    assert splitter_cmd is not None, f"splitter never invoked; cmds={captured}"
    positional = splitter_cmd[3]
    assert positional.endswith(".pretrimmed.trace.json.gz"), splitter_cmd
    assert str(trace) not in splitter_cmd, splitter_cmd

    # The trim ran, and the raw capture is still on disk and unmodified.
    pretrim_files = list((tmp_path / "ws").rglob("pretrim.json"))
    assert len(pretrim_files) == 1, pretrim_files
    pretrim = json.loads(pretrim_files[0].read_text())
    assert pretrim["applied"] is True
    assert pretrim["source"] == str(trace)
    assert len(tla._step_annotation_spans(tla.open_json(trace)["traceEvents"])) == 41


def test_capture_folder_is_not_discoverable_from_the_split_directory(tmp_path):
    """Why the splitter's input has to stay separate from the analysed path.

    Discovery searches the trace file's own directory and its parent. A trimmed
    copy lives in ``tracelens/trace_split/``, whose neighbourhood is the run's
    output tree -- the capture's sidecar is not in it.
    """
    capture_dir = tmp_path / "torch_trace"
    (capture_dir / "graph_capture_profile").mkdir(parents=True)
    raw = capture_dir / "r.trace.json.gz"
    raw.write_bytes(b"")
    split_dir = tmp_path / "tracelens" / "trace_split"
    split_dir.mkdir(parents=True)
    trimmed = split_dir / "r.pretrimmed.trace.json.gz"
    trimmed.write_bytes(b"")

    assert tlr.discover_capture_folder(raw, [raw]) == capture_dir / "graph_capture_profile"
    assert tlr.discover_capture_folder(raw, [trimmed]) is None
