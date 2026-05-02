"""Regression tests for the kernel-agent tracelens_analysis filter fixes.

Locks two related fixes uncovered by the resume3/resume4 1h validation:

* **A path** ``is_kernel_event`` previously did fuzzy substring matching
  on `KERNEL_HINTS=("kernel","triton","hip","cuda",...)` against both the
  event name AND category. That accidentally promoted `cat='python_function'`
  rows like ``torch/cuda/streams.py(222): synchronize`` (a CPU wait that
  accumulates the entire wrapped GPU duration) to be the #1 hot kernel —
  88ms attributed to a CPU sync. Fix: require ``cat == 'kernel'`` strictly.

* **B path** the wrapper now reads TraceLens's own ``kernel_summary.csv``
  output first; falls back to the (fixed) raw parser only when the csv
  is missing.  TraceLens already filters host-side events correctly via
  ``Parent cpu_op == hipGraphLaunch`` joins, so reading its csv is both
  more accurate and self-tracking when TraceLens schema improves.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# tools/ is not a package — stick its dir on sys.path so we can import.
_TOOL_DIR = Path(__file__).resolve().parent
if str(_TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOL_DIR))

import tracelens_analysis as tla  # noqa: E402


# ===========================================================================
# A path — is_kernel_event strict cat == 'kernel'
# ===========================================================================
def test_a_filters_python_function_synchronize():
    """The exact event that ranked #1 in the buggy resume3/4 trace."""
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
        "name": ("_ZN5aiter24add_rmsnorm_quant_kernel"
                 "IDF16bDF16bLi256ELi16ELb1ELb0ELb1ELi1EEEvPT0_"),
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
    the buggy events drop out of top-K."""
    events = [
        # 5 host-side sync events, big durations (the buggy ones)
        {"name": "torch/cuda/streams.py(222): synchronize",
         "cat": "python_function", "dur": 88673.0},
        {"name": "torch/cuda/__init__.py(1073): synchronize",
         "cat": "python_function", "dur": 10414.0},
        {"name": "<built-in function _cuda_synchronize>",
         "cat": "python_function", "dur": 10368.0},
        {"name": "hipDeviceSynchronize", "cat": "cuda_runtime", "dur": 10364.0},
        {"name": "<built-in method synchronize of Event object at 0x123>",
         "cat": "python_function", "dur": 88671.0},
        # 3 real kernels, smaller duration
        {"name": "aiter::add_rmsnorm_quant_kernel<...>", "cat": "kernel", "dur": 466.0},
        {"name": "sgl_hip::activation::act_and_mul_kernel<...>",
         "cat": "kernel", "dur": 380.0},
        {"name": "void paged_attention_ll4mi_QKV_mfma16_kernel<...>",
         "cat": "kernel", "dur": 889.0},
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


# ===========================================================================
# B path — parse_tracelens_kernel_summary
# ===========================================================================
@pytest.fixture
def tl_csv(tmp_path: Path) -> Path:
    """Write a kernel_summary.csv shaped like real TraceLens output."""
    csv_path = tmp_path / "kernel_summary.csv"
    # \u00b5 == µ (micro sign as TraceLens emits)
    csv_path.write_text(
        "Parent op category,Parent cpu_op,Kernel name,Kernel stream,"
        "Kernel duration (\u00b5s)_sum,Kernel duration (\u00b5s)_count,"
        "Kernel duration (\u00b5s)_mean,Kernel duration (\u00b5s)_min\n"
        "graph,hipGraphLaunch,Cijk_Alik_Bljk_*MT256x16x64,3,5077.217,37,137.22,121.4\n"
        "graph,hipGraphLaunch,Cijk_Alik_Bljk_*MT16x16x512,3,2084.107,72,28.94,16.6\n"
        "graph,hipGraphLaunch,_ZN5aiter24add_rmsnorm_quant_kernel<bf16>,3,466.586,72,6.48,5.9\n"
        "graph,hipGraphLaunch,_ZN5aiter24add_rmsnorm_quant_kernel<bf16>,5,442.089,72,6.14,5.1\n"
        "graph,hipGraphLaunch,paged_attention_ll4mi_QKV_mfma16_kernel<...>,3,889.104,36,24.69,21.6\n"
        "graph,hipGraphLaunch,_ZN7sgl_hip10activation18act_and_mul_kernel<bf16>,3,380.188,36,10.56,9.8\n",
        encoding="utf-8",
    )
    return csv_path


def test_b_parses_tracelens_csv(tl_csv):
    out = tla.parse_tracelens_kernel_summary(tl_csv, top_k=10)
    assert out is not None
    assert len(out) == 5  # 6 csv rows but 2 add_rmsnorm rows aggregate by name
    # Top-1 should be the GEMM with biggest sum-duration
    top = out[0]
    assert top["name"].startswith("Cijk_Alik_Bljk_*MT256x16x64")
    assert top["duration_us"] == pytest.approx(5077.217, rel=1e-3)
    assert top["call_count"] == 37
    assert top["kernel_id"] == "k001"
    # Check aggregation: add_rmsnorm appears once with summed duration
    add_rmsnorm = [k for k in out if "add_rmsnorm_quant" in k["name"]]
    assert len(add_rmsnorm) == 1
    assert add_rmsnorm[0]["duration_us"] == pytest.approx(466.586 + 442.089, rel=1e-3)
    assert add_rmsnorm[0]["call_count"] == 144
    # gpu_pct is computed against total
    total = sum(k["duration_us"] for k in out)
    assert sum(k["gpu_pct"] for k in out) == pytest.approx(100.0, abs=0.5)


def test_b_returns_none_when_csv_missing(tmp_path):
    out = tla.parse_tracelens_kernel_summary(tmp_path / "nonexistent.csv", top_k=10)
    assert out is None


def test_b_returns_none_when_csv_missing_required_columns(tmp_path):
    bad = tmp_path / "kernel_summary.csv"
    bad.write_text("foo,bar\n1,2\n", encoding="utf-8")
    assert tla.parse_tracelens_kernel_summary(bad, top_k=10) is None


def test_b_returns_none_when_csv_has_zero_kernels(tmp_path):
    empty = tmp_path / "kernel_summary.csv"
    empty.write_text(
        "Kernel name,Kernel duration (\u00b5s)_sum,Kernel duration (\u00b5s)_count\n",
        encoding="utf-8",
    )
    assert tla.parse_tracelens_kernel_summary(empty, top_k=10) is None


# ===========================================================================
# Native-only kernel-opt targeting
# ===========================================================================
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
    assert tla.is_runtime_generated_kernel(
        candidate["name"], candidate["source_file"],
    ) is True
    assert tla.is_reusable_native_kernel(candidate) is False
    assert tla.recommend_backends(candidate) == []
    assert "not reusable" in tla.build_notes(candidate)


def test_stable_framework_triton_source_is_reusable_native():
    candidate = {
        "name": "triton_attention_decode_kernel",
        "source_file": "/sgl-workspace/sglang/python/sglang/srt/layers/attention/triton_ops.py",
        "source_type": tla.source_type_for(
            "triton_attention_decode_kernel",
            "/sgl-workspace/sglang/python/sglang/srt/layers/attention/triton_ops.py",
        ),
    }
    assert candidate["source_type"] == "triton"
    assert tla.is_runtime_generated_kernel(
        candidate["name"], candidate["source_file"],
    ) is False
    assert tla.is_reusable_native_kernel(candidate) is True
    assert tla.recommend_backends(candidate) == ["geak", "claude", "codex"]


def test_unknown_source_root_is_not_reusable_native():
    candidate = {
        "name": "my_custom_kernel",
        "source_file": "/tmp/random/my_custom_kernel.cu",
        "source_type": tla.source_type_for(
            "my_custom_kernel", "/tmp/random/my_custom_kernel.cu",
        ),
    }
    assert candidate["source_type"] == "hip_cpp"
    assert tla.is_reusable_native_kernel(candidate) is False
    assert tla.recommend_backends(candidate) == []


def test_known_rmsnorm_harness_is_registered_without_repo_root():
    files = tla.find_benchmark_files(
        "_ZN5aiter24add_rmsnorm_quant_kernelIDF16bDF16bLi256EEEv",
        "",
        "/sgl-workspace/aiter/csrc/kernels/rmsnorm_quant_kernels.cu",
    )
    assert any("rmsnorm" in path.lower() for path in files)
