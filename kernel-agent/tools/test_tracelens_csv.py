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


# ===========================================================================
# #125 — TraceLens structured output consumption
# (kernel_category / shape / source_path triple for GEAK)
# ===========================================================================


def test_125_csv_extracts_parent_op_category(tl_csv):
    """csv parser surfaces 'Parent op category' as tracelens_category."""
    out = tla.parse_tracelens_kernel_summary(tl_csv, top_k=10)
    assert out is not None
    for cand in out:
        assert cand.get("tracelens_category") == "graph", (
            f"missing/wrong tracelens_category on {cand['name']}: {cand}"
        )


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
    ]
    for name, expected in cases:
        assert tla.derive_kernel_category({"name": name}) == expected, name


@pytest.fixture
def tl_category_data(tmp_path: Path) -> Path:
    """Mock TraceLens orchestrator category_data/ directory."""
    cd = tmp_path / "category_data"
    cd.mkdir()
    (cd / "GEMM.json").write_text(
        '{"category": "GEMM", "kernels": ['
        '{"name": "Cijk_Alik_Bljk_MT256x16x64", "duration_us": 5077.2, '
        '"call_count": 37, "shape": [16, 64, 64], "source_path": '
        '"/sgl-workspace/aiter/csrc/kernels/gemm.cu"},'
        '{"name": "rocblas_dgemm_kernel", "duration_us": 1200.0, '
        '"call_count": 10, "shape": [16, 16, 512]}'
        ']}'
    )
    (cd / "SDPA.json").write_text(
        '{"name": "SDPA", "items": ['
        '{"name": "paged_attention_ll4mi_QKV_mfma16_kernel<bf16>", '
        '"duration_us": 889.1, "call_count": 36, '
        '"input_shape": [1, 2048, 64], "path": '
        '"/sgl-workspace/sglang/csrc/attention.cu"}'
        ']}'
    )
    return cd


def test_125_parses_category_data_with_full_triple(tl_category_data):
    out = tla.parse_tracelens_category_data(tl_category_data, top_k=10)
    assert out is not None
    assert len(out) == 3, f"expected 3 kernels, got {len(out)}"
    by_name = {c["name"]: c for c in out}

    gemm = by_name["Cijk_Alik_Bljk_MT256x16x64"]
    assert gemm["kernel_category"] == "GEMM"
    assert gemm["shapes"]
    assert gemm["source_file"] == "/sgl-workspace/aiter/csrc/kernels/gemm.cu"
    assert gemm["source_path"] == gemm["source_file"]
    assert gemm["duration_us"] == pytest.approx(5077.2, rel=1e-3)

    sdpa = by_name["paged_attention_ll4mi_QKV_mfma16_kernel<bf16>"]
    assert sdpa["kernel_category"] == "SDPA"
    assert sdpa["source_file"] == "/sgl-workspace/sglang/csrc/attention.cu"


def test_125_category_data_returns_none_when_dir_missing(tmp_path):
    out = tla.parse_tracelens_category_data(tmp_path / "nope", top_k=10)
    assert out is None


def test_125_category_data_returns_none_when_dir_empty(tmp_path):
    cd = tmp_path / "category_data"
    cd.mkdir()
    out = tla.parse_tracelens_category_data(cd, top_k=10)
    assert out is None


def test_125_category_data_handles_flat_dict_layout(tmp_path):
    cd = tmp_path / "category_data"
    cd.mkdir()
    (cd / "all.json").write_text(
        '{"GEMM": [{"name": "k1", "duration_us": 10, "shape": [1,2]}],'
        ' "SDPA": [{"name": "k2", "duration_us": 20}]}'
    )
    out = tla.parse_tracelens_category_data(cd, top_k=10)
    assert out is not None
    assert len(out) == 2
    by_name = {c["name"]: c for c in out}
    assert by_name["k1"]["kernel_category"] == "GEMM"
    assert by_name["k2"]["kernel_category"] == "SDPA"


def test_125_category_data_skips_invalid_json(tmp_path):
    cd = tmp_path / "category_data"
    cd.mkdir()
    (cd / "bad.json").write_text("not valid json{")
    (cd / "good.json").write_text(
        '{"category": "GEMM", "kernels": [{"name": "k1", "duration_us": 10}]}'
    )
    out = tla.parse_tracelens_category_data(cd, top_k=10)
    assert out is not None
    assert len(out) == 1
    assert out[0]["name"] == "k1"


def test_125_csv_carries_category_through_to_finalize(tl_csv):
    out = tla.parse_tracelens_kernel_summary(tl_csv, top_k=10)
    assert out is not None
    for cand in out:
        assert cand["tracelens_category"] == "graph"
        assert cand["kernel_category"]


def test_125_augment_csv_with_raw_shapes(tmp_path: Path, tl_csv):
    """augment_csv_candidates_with_raw_shapes pulls shape from raw trace."""
    import gzip
    import json as _json

    raw = tmp_path / "raw.json.gz"
    with gzip.open(raw, "wt") as f:
        _json.dump({
            "traceEvents": [
                {
                    "name": "Cijk_Alik_Bljk_*MT256x16x64",
                    "cat": "kernel",
                    "dur": 137.22,
                    "ts": 1,
                    "args": {"shape": [[256, 16, 64]]},
                },
            ]
        }, f)

    candidates = tla.parse_tracelens_kernel_summary(tl_csv, top_k=10)
    gemm = next(c for c in candidates if c["name"].startswith("Cijk_Alik_Bljk_*MT256x16x64"))
    assert gemm.get("shapes") == [], "csv path should produce empty shapes pre-augment"

    tla.augment_csv_candidates_with_raw_shapes(candidates, [raw])

    gemm = next(c for c in candidates if c["name"].startswith("Cijk_Alik_Bljk_*MT256x16x64"))
    assert gemm["shapes"], f"expected shape backfilled, got {gemm['shapes']}"


def test_125_csv_no_parent_op_category_column_degrades_gracefully(tmp_path):
    """Older TraceLens builds lack 'Parent op category' — must not crash."""
    csv = tmp_path / "kernel_summary.csv"
    csv.write_text(
        "Kernel name,Kernel duration (\u00b5s)_sum,Kernel duration (\u00b5s)_count\n"
        "my_kernel,100.0,5\n",
        encoding="utf-8",
    )
    out = tla.parse_tracelens_kernel_summary(csv, top_k=10)
    assert out is not None
    assert len(out) == 1
    assert out[0].get("tracelens_category", "") == ""  # field absent or empty
    assert "kernel_category" in out[0]


def test_125_extract_category_kernels_layouts():
    """_extract_category_kernels handles all three observed layouts."""
    # Layout 1: kernels array
    out1 = tla._extract_category_kernels(
        {"category": "GEMM", "kernels": [{"name": "k1"}]}
    )
    assert len(out1) == 1 and out1[0]["category"] == "GEMM"

    # Layout 2: items array (alt key)
    out2 = tla._extract_category_kernels(
        {"name": "SDPA", "items": [{"name": "k2"}]}
    )
    assert len(out2) == 1 and out2[0]["category"] == "SDPA"

    # Layout 3: flat dict-of-lists
    out3 = tla._extract_category_kernels(
        {"GEMM": [{"name": "k1"}], "SDPA": [{"name": "k2"}]}
    )
    assert {e["category"] for e in out3} == {"GEMM", "SDPA"}

    # Layout 4: bare list
    out4 = tla._extract_category_kernels([{"name": "k", "category": "X"}])
    assert len(out4) == 1


def test_125_finalize_outputs_source_path_field():
    """_finalize_candidates exposes source_path mirror of source_file (#125)."""
    candidates = [{
        "name": "rmsnorm_kernel",
        "duration_us": 100.0,
        "call_count": 10,
        "source_file": "/path/to/rmsnorm.cu",
        "source_type": "hip_cpp",
        "shapes": [[16, 1024]],
    }]
    out = tla._finalize_candidates(candidates, total_dur=100.0)
    assert out[0]["source_path"] == "/path/to/rmsnorm.cu"
    assert out[0]["kernel_category"] == "LayerNorm"


# ===========================================================================
# #127 — TraceLens splitter CLI must match the real
# split_inference_trace_annotation interface (positional trace_path,
# -o/--output-dir, --find-steady-state). The previous --input/--platform
# form failed at runtime against a real Magpie/SGLang trace.
# ===========================================================================
def test_127_splitter_cli_uses_positional_trace_path_and_find_steady_state(tmp_path):
    """The end-to-end split path must call the real splitter interface,
    not the broken --input/--platform form. Drives a mock subprocess.run
    and asserts argv shape."""
    import gzip
    import json as _json
    from unittest.mock import patch

    # Pretend TraceLens root + perf-report CLI are present so the run
    # reaches the splitter step.
    tl_root = tmp_path / "TraceLens-internal"
    skill_dir = tl_root / "TraceLens" / "AgenticMode" / "Standalone" / ".cursor" / "skills"
    skill_dir.mkdir(parents=True)
    (skill_dir / "standalone-analysis-orchestrator.md").write_text("stub")
    workspace = tmp_path / "ws"
    workspace.mkdir()
    trace = tmp_path / "trace.json.gz"
    with gzip.open(trace, "wt") as f:
        _json.dump({"traceEvents": []}, f)

    captured: list[list[str]] = []

    class _Result:
        def __init__(self, returncode=0, stdout=""):
            self.returncode = returncode
            self.stdout = stdout

    def fake_run(cmd, *args, **kwargs):
        captured.append(list(cmd))
        # Make the perf-report CLI probe + actual run + pip install all succeed.
        return _Result(returncode=0, stdout="ok")

    def fake_which(name):
        return f"/usr/bin/{name}"

    argv = [
        "tracelens_analysis.py",
        "--trace-input", str(trace),
        "--workspace-path", str(workspace),
        "--tracelens-root", str(tl_root),
        "--target-platform", "MI300X",
        "--top-k", "5",
        "--budget-minutes", "1",
        "--split-conc", "8",
        "--split-osl", "1024",
    ]
    import os as _os
    env_backup = dict(_os.environ)
    try:
        with patch.object(tla.subprocess, "run", side_effect=fake_run), \
             patch.object(tla.shutil, "which", side_effect=fake_which), \
             patch.object(tla.sys, "argv", argv):
            try:
                tla.main()
            except SystemExit:
                pass
    finally:
        _os.environ.clear()
        _os.environ.update(env_backup)

    splitter_cmd = next(
        (c for c in captured
         if any("split_inference_trace_annotation" in str(p) for p in c)),
        None,
    )
    assert splitter_cmd is not None, f"splitter never invoked; cmds={captured}"
    # Real CLI: positional trace_path, -o output, --find-steady-state.
    assert "--input" not in splitter_cmd, (
        f"--input must be removed: {splitter_cmd}"
    )
    assert "--platform" not in splitter_cmd, (
        f"--platform is not a valid splitter flag: {splitter_cmd}"
    )
    assert "--find-steady-state" in splitter_cmd, splitter_cmd
    assert "-o" in splitter_cmd, splitter_cmd
    assert str(trace) in splitter_cmd, (
        f"trace_path must be passed positionally: {splitter_cmd}"
    )
    # CONC / OSL passthroughs.
    assert "--CONC" in splitter_cmd and "8" in splitter_cmd, splitter_cmd
    assert "--OSL" in splitter_cmd and "1024" in splitter_cmd, splitter_cmd
