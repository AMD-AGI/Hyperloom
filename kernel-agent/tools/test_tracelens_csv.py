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
import tracelens_skill_runner as tlr  # noqa: E402


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


def test_125_augment_csv_with_unified_perf_summary_shape_dtype(tmp_path: Path, tl_csv):
    """unified_perf_summary carries TraceLens Input Dims/Input type."""
    unified = tmp_path / "unified_perf_summary.csv"
    kernel_name = "Cijk_Alik_Bljk_*MT256x16x64"
    unclear_kernel_name = "unclear_elementwise_kernel"
    unified.write_text(
        "name,Input Dims,Input type,kernel_details_summary,perf_params\n"
        "aten::mm,\"((24576,8192), (8192,28672), (24576,28672))\","
        "\"('c10::BFloat16', 'c10::BFloat16', 'c10::BFloat16')\","
        f"\"[{{'name': '{kernel_name}', 'stream': 3, 'count': 4}}]\","
        "\"{'shape_out': (24576, 28672), 'output_dtype': 'c10::BFloat16'}\"\n"
        "aten::mm,\"((24576,8192), (8192,28672), (24576,28672))\","
        "\"('c10::BFloat16', 'c10::BFloat16', 'c10::BFloat16')\","
        f"\"[{{'name': '{kernel_name}', 'stream': 4, 'count': 3}}]\","
        "\"{'shape_out': (24576, 28672), 'output_dtype': 'c10::BFloat16'}\"\n"
        "aten::add,\"((1,), (), ())\","
        "\"('long int', 'long int', 'Scalar')\","
        f"\"[{{'name': '{unclear_kernel_name}', 'stream': 3}}]\","
        "\"{'shape_in1': (1,), 'shape_in2': (), "
        "'dtype_in1_in2_out': ('long int', 'long int', None), "
        "'stride_output': None}\"\n",
        encoding="utf-8",
    )
    candidates = tla.parse_tracelens_kernel_summary(tl_csv, top_k=10)
    candidates.append({"name": unclear_kernel_name})
    gemm = next(c for c in candidates if c["name"] == kernel_name)
    assert "input_shapes" not in gemm or gemm["input_shapes"] == []

    tla.augment_csv_candidates_with_unified_perf_summary(candidates, unified)

    gemm = next(c for c in candidates if c["name"] == kernel_name)
    assert gemm["input_shapes"] == [
        {"call_num": 7, "shape": [24576, 8192]},
        {"call_num": 7, "shape": [8192, 28672]},
        {"call_num": 7, "shape": [24576, 28672]},
    ]
    assert gemm["input_dtypes"] == ["c10::BFloat16"]
    assert gemm["output_shapes"] == [[24576, 28672]]
    assert gemm["output_dtypes"] == ["c10::BFloat16"]
    assert gemm["runtime_args"]["tracelens_args"] == [
        {
            "op": "aten::mm",
            "input_dims": "((24576,8192), (8192,28672), (24576,28672))",
            "input_types": "('c10::BFloat16', 'c10::BFloat16', 'c10::BFloat16')",
        }
    ]
    unclear = next(c for c in candidates if c["name"] == unclear_kernel_name)
    assert unclear.get("output_shapes", []) == []
    assert unclear.get("output_dtypes", []) == []


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


def test_write_reports_enriches_candidates_with_runtime_metadata(tmp_path):
    import json as _json
    from argparse import Namespace

    trace = tmp_path / "trace.json"
    trace.write_text("{}", encoding="utf-8")
    candidate = {
        "kernel_id": "k001",
        "name": "paged_attention",
        "duration_us": 100.0,
        "call_count": 2,
        "gpu_pct": 10.0,
        "source_file": "/tmp/paged_attention.py",
        "shapes": [[1, 32, 128]],
        "is_multigpu": False,
        "num_gpus_recommended": 1,
    }
    args = Namespace(
        trace_input=str(trace),
        model_name="llama",
        framework="sglang",
        target_platform="MI300X",
        analysis_mode="inference",
        runtime_env="local",
        dry_run=False,
        compat_report_path="",
    )

    artifacts = tla.write_reports(
        tmp_path / "run",
        trace_input_type="file",
        trace_files=[trace],
        candidates=[candidate],
        args=args,
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
        _json.dumps({
            "head_dim": 128,
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
        }),
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
        _json.dumps({
            "hidden_size": 4096,
            "num_attention_heads": 32,
        }),
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
        _json.dumps({
            "hidden_size": 7168,
            "num_attention_heads": 128,
            "qk_nope_head_dim": 128,
            "qk_rope_head_dim": 64,
            "v_head_dim": 128,
            "kv_lora_rank": 512,
        }),
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
    candidate = {
        "kernel_id": "k001",
        "name": "paged_attention",
        "duration_us": 100.0,
        "call_count": 2,
        "gpu_pct": 10.0,
        "source_file": "/tmp/paged_attention.py",
        "shapes": [[1, 32, 128]],
    }
    args = Namespace(
        trace_input=str(trace),
        model_name=str(model_dir),
        framework="sglang",
        target_platform="MI300X",
        analysis_mode="inference",
        runtime_env="local",
        dry_run=False,
        compat_report_path="",
    )

    artifacts = tla.write_reports(
        tmp_path / "run",
        trace_input_type="file",
        trace_files=[trace],
        candidates=[candidate],
        args=args,
    )
    payload = _json.loads(Path(artifacts["kernel_candidates"]).read_text(encoding="utf-8"))

    assert payload["hot_kernels"][0]["kernel_params"]["HEAD_SIZE"] == 128


# ===========================================================================
# #124 — SDK runner for TraceLens analysis-orchestrator skill
# ===========================================================================
def test_124_build_orchestrator_prompt_supplies_step0_inputs(tmp_path):
    skill = tmp_path / "analysis-orchestrator.md"
    trace = tmp_path / "mixed_steady_state_0_trace.json.gz"
    out = tmp_path / "tracelens"
    root = tmp_path / "TraceLens-internal"
    capture = tmp_path / "capture_traces"

    prompt = tlr.build_orchestrator_prompt(
        skill_path=skill,
        trace_path=trace,
        output_dir=out,
        tracelens_root=root,
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
    assert "Do not ask the user" in prompt


def test_124_priority_data_members_convert_to_raw_candidates(tmp_path):
    import json as _json

    priority = tmp_path / "priority_data.json"
    priority.write_text(_json.dumps({
        "findings": [{
            "category": "gemm",
            "impact_score": 4.2,
            "library": "hipBLASLt",
            "members": [{
                "operation": "Cijk_Alik_Bljk_MT256",
                "time_ms": 5.5,
                "impact_score": 3.1,
                "library": "hipBLASLt",
                "bound_type": "compute",
            }],
        }],
    }), encoding="utf-8")

    rows = tlr.raw_candidates_from_priority_data(priority, top_k=10)
    assert rows == [{
        "name": "Cijk_Alik_Bljk_MT256",
        "duration_us": 5500.0,
        "call_count": 1,
        "source_file": "",
        "source_type": "unknown",
        "shapes": [],
        "tracelens_category": "gemm",
        "impact_score": 3.1,
        "impact_score_low": 0.0,
        "impact_score_high": 0.0,
        "library": "hipBLASLt",
        "bound_type": "compute",
    }]


def test_count_gpu_kernel_events_distinguishes_cpu_only_and_real_traces(tmp_path):
    import gzip, json as _json
    cpu_only = tmp_path / "cpu_only.json.gz"
    with gzip.open(cpu_only, "wt") as f:
        _json.dump({"traceEvents": [
            {"cat": "cpu_op", "name": "aten::add", "dur": 1.0},
            {"cat": "python_function", "name": "wrapper", "dur": 2.0},
            {"cat": "cuda_runtime", "name": "hipDeviceSynchronize", "dur": 3.0},
        ]}, f)
    assert tla.count_gpu_kernel_events(cpu_only) == 0

    real = tmp_path / "real.json.gz"
    with gzip.open(real, "wt") as f:
        _json.dump({"traceEvents": [
            {"cat": "cpu_op", "name": "aten::add", "dur": 1.0},
            {"cat": "kernel", "name": "void some_gemm_kernel<...>", "dur": 7.0},
            {"cat": "kernel", "name": "void some_attn_kernel<...>", "dur": 11.0},
            {"cat": "cuda_runtime", "name": "hipLaunchKernel", "dur": 0.5},
        ]}, f)
    assert tla.count_gpu_kernel_events(real) == 2


def test_124_tracelens_analysis_fails_fast_on_cpu_only_trace(tmp_path):
    """Issue #126/#124 regression: when the upstream profile run produces
    a CPU-only trace (e.g. PMC LD_PRELOAD steals the rocprofiler-sdk slot
    from torch.profiler), tracelens_analysis must fail loudly *before*
    spending time on TraceLens install / split / SDK orchestrator runs."""
    import gzip
    import json as _json
    from unittest.mock import patch

    tl_root = tmp_path / "TraceLens-internal"
    skill_dir = tl_root / "TraceLens" / "Agent" / "Analysis" / ".cursor" / "skills"
    skill_dir.mkdir(parents=True)
    (skill_dir / "analysis-orchestrator.md").write_text("stub")
    workspace = tmp_path / "ws"
    workspace.mkdir()

    trace = tmp_path / "cpu_only_trace.json.gz"
    with gzip.open(trace, "wt") as f:
        _json.dump({"traceEvents": [
            {"cat": "cpu_op", "name": "aten::add", "dur": 1.0},
            {"cat": "python_function", "name": "wrapper", "dur": 2.0},
        ]}, f)

    captured: list[list[str]] = []

    class _Result:
        def __init__(self):
            self.returncode = 0
            self.stdout = "ok"

    def fake_run(cmd, *args, **kwargs):
        captured.append(list(cmd))
        return _Result()

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
        "--no-llm-orchestrator",
    ]
    import os as _os
    env_backup = dict(_os.environ)
    try:
        with patch.object(tla.subprocess, "run", side_effect=fake_run), \
             patch.object(tla.shutil, "which", side_effect=fake_which), \
             patch.object(tla.sys, "argv", argv):
            try:
                rc = tla.main()
            except SystemExit as exc:
                rc = int(exc.code or 0)
    finally:
        _os.environ.clear()
        _os.environ.update(env_backup)

    assert rc != 0, "fail-fast on CPU-only trace must return non-zero"
    assert all(
        "TraceLens.TraceUtils.split_inference_trace_annotation" not in str(p)
        for cmd in captured for p in cmd
    ), f"splitter must not run on CPU-only trace; captured={captured}"
    assert all(
        "TraceLens_generate_perf_report_pytorch_inference" not in str(c[0])
        or "--help" in c
        for c in captured if c
    ), f"perf-report CLI must not be invoked for CPU-only trace; captured={captured}"


def test_124_run_tracelens_skill_uses_sdk_and_artifacts(tmp_path):
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
    captured: dict[str, Any] = {}

    async def _fake_query(*, prompt, options):
        captured["prompt"] = prompt
        captured["options"] = options.kwargs
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "standalone_analysis.md").write_text("# report\n", encoding="utf-8")
        (output_dir / "priority_data.json").write_text(
            _json.dumps({"findings": [], "priorities": []}),
            encoding="utf-8",
        )
        yield _Message(content=[_TextBlock("done")])

    res = asyncio.run(tlr.run_tracelens_skill(
        skill_path=tmp_path / "skill.md",
        trace_path=tmp_path / "trace.json.gz",
        output_dir=output_dir,
        tracelens_root=tmp_path,
        platform="MI300X",
        framework="sglang",
        analysis_mode="default",
        capture_folder=None,
        budget_minutes=1,
        sdk_query_factory=_fake_query,
        sdk_options_cls=_FakeOptions,
    ))

    assert res.report_path.exists()
    assert res.priority_data_path.exists()
    assert "analysis-orchestrator" in captured["prompt"] or "skill.md" in captured["prompt"]
    assert "Bash" in captured["options"]["allowed_tools"]
    assert "Task" in captured["options"]["allowed_tools"]


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
    skill_dir = tl_root / "TraceLens" / "Agent" / "Analysis" / ".cursor" / "skills"
    skill_dir.mkdir(parents=True)
    (skill_dir / "analysis-orchestrator.md").write_text("stub")
    workspace = tmp_path / "ws"
    workspace.mkdir()
    capture = tmp_path / "capture_traces"
    capture.mkdir()
    trace = tmp_path / "trace.json.gz"
    with gzip.open(trace, "wt") as f:
        _json.dump({"traceEvents": [
            # At least one real GPU kernel event so the new fail-fast
            # validation lets the run continue into the splitter step.
            {"cat": "kernel", "name": "void some_real_kernel<...>", "dur": 5.0},
        ]}, f)

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
        "--no-llm-orchestrator",
        "--capture-folder", str(capture),
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
            except SystemExit as exc:
                # tla.main() may CLI-exit via sys.exit() depending on the
                # mocked perf-report CLI's rc. Either a clean exit code (0 /
                # None) or a soft-fallback non-zero is acceptable here — the
                # test asserts the splitter command shape further down, not
                # the program's overall exit status.
                _ = exc
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

    perf_cmd = next(
        (c for c in captured
         if c
         and "TraceLens_generate_perf_report_pytorch_inference" in str(c[0])
         and "--profile_json_path" in c),
        None,
    )
    assert perf_cmd is not None, f"perf-report CLI never invoked; cmds={captured}"
    assert "--group_by_parent_module" in perf_cmd, perf_cmd
    assert "--enable_pseudo_ops" in perf_cmd, perf_cmd
    assert "--group_by_num_kernels" in perf_cmd, perf_cmd
    assert "--gpu_arch_json_path" in perf_cmd, perf_cmd
    assert "--capture_folder" in perf_cmd and str(capture) in perf_cmd, perf_cmd


# ===========================================================================
# #194 §3 — splitter must receive --R so mixed-window selection uses the
# analytic PD ratio. Source: tracelens_analysis must pass --R when given
# either via --split-r CLI arg or via the RANDOM_RANGE_RATIO env var (the
# same env Hyperloom propagates from the YAML config to the benchmark
# subprocess). Without --R the splitter falls back to an empirical
# heuristic, drifting from the benchmark-contract ratio the skill aligns
# the rest of the pipeline to.
# ===========================================================================
def _drive_main_capturing_subprocess(tmp_path, extra_argv, env_overrides=None):
    """Helper: stage a TraceLens-ish tree, stub subprocess.run + which,
    drive tla.main() once, and return the list of captured argvs."""
    import gzip
    import json as _json
    import os as _os
    from unittest.mock import patch

    tl_root = tmp_path / "TraceLens-internal"
    skill_dir = (
        tl_root / "TraceLens" / "Agent" / "Analysis" / ".cursor" / "skills"
    )
    skill_dir.mkdir(parents=True)
    (skill_dir / "analysis-orchestrator.md").write_text("stub")
    workspace = tmp_path / "ws"
    workspace.mkdir()
    capture = tmp_path / "capture_traces"
    capture.mkdir()
    trace = tmp_path / "trace.json.gz"
    with gzip.open(trace, "wt") as f:
        _json.dump({"traceEvents": [
            {"cat": "kernel", "name": "void some_real_kernel<...>", "dur": 5.0},
        ]}, f)

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
        "--trace-input", str(trace),
        "--workspace-path", str(workspace),
        "--tracelens-root", str(tl_root),
        "--target-platform", "MI300X",
        "--top-k", "5",
        "--budget-minutes", "1",
        "--no-llm-orchestrator",
        "--capture-folder", str(capture),
        *extra_argv,
    ]

    env_backup = dict(_os.environ)
    try:
        for k, v in (env_overrides or {}).items():
            if v is None:
                _os.environ.pop(k, None)
            else:
                _os.environ[k] = v
        with patch.object(tla.subprocess, "run", side_effect=fake_run), \
             patch.object(tla.shutil, "which", side_effect=lambda n: f"/usr/bin/{n}"), \
             patch.object(tla.sys, "argv", argv):
            try:
                tla.main()
            except SystemExit:
                pass
    finally:
        _os.environ.clear()
        _os.environ.update(env_backup)

    return captured, trace


def _find_splitter_cmd(captured):
    return next(
        (c for c in captured
         if any("split_inference_trace_annotation" in str(p) for p in c)),
        None,
    )


def test_194_3_splitter_receives_R_from_cli_arg(tmp_path):
    """`--split-r 0.5` on the wrapper must produce `--R 0.5` on the
    splitter argv. Floating-point ratios must survive verbatim — the
    splitter declares `type=float` and any string coercion to int
    would silently truncate fractional R."""
    captured, _ = _drive_main_capturing_subprocess(
        tmp_path,
        extra_argv=[
            "--split-conc", "32",
            "--split-osl", "1024",
            "--split-r", "0.5",
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
    """A malformed env value must NOT be propagated to the splitter —
    the splitter would `argparse.error` and abort the whole pipeline
    on a value error. Silently dropping (with a log line) is the
    least-bad option; it falls back to the splitter's default heuristic
    which is exactly the pre-#194-§3 behaviour."""
    captured, _ = _drive_main_capturing_subprocess(
        tmp_path,
        extra_argv=["--split-conc", "32", "--split-osl", "1024"],
        env_overrides={"RANDOM_RANGE_RATIO": "not-a-float"},
    )
    splitter_cmd = _find_splitter_cmd(captured)
    assert splitter_cmd is not None, f"splitter never invoked; cmds={captured}"
    assert "--R" not in splitter_cmd, splitter_cmd


# ===========================================================================
# parse_analysis_md — TraceLens v0.3 final-report contract (#155 review)
# ===========================================================================
_FIXTURE_LLAMA70B_ANALYSIS_MD = (
    Path(__file__).resolve().parents[1]
    / "tests" / "fixtures" / "tracelens_v03_llama70b_analysis.md"
)


def test_parse_analysis_md_llama70b_fixture_yields_21_compute_candidates():
    """Round-trip the TraceLens v0.3 reference fixture for Llama-3 70B.

    The fixture (TraceLens-internal ``evals/analysis_tests/e2e_tests/
    llama_70b/analysis_output_ref/analysis.md``) is the official golden
    output, so its 9-column Detailed Analysis tables are the contract our
    parser must round-trip without loss.
    """
    cands = tlr.parse_analysis_md(_FIXTURE_LLAMA70B_ANALYSIS_MD, top_k=50)
    assert len(cands) == 21, (
        f"expected 21 candidates (18 GEMM + 2 SDPA_fwd + 1 SDPA_bwd) from "
        f"the fixture; got {len(cands)}"
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


# ===========================================================================
# normalize_upstream_category — TraceLens orchestrator_prepare.py enum (#155 #4)
# ===========================================================================
@pytest.mark.parametrize("raw,expected", [
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
])
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

# ===========================================================================
# PR-A §1: _extract_pitem_prose extracts Reasoning / Resolution / Impact
# ===========================================================================
_SYNTHETIC_PITEM_BODY = """\
#### 🔴 P1: RMSNorm fused with quantization (Triton)

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


def test_extract_pitem_prose_pulls_three_sections():
    prose = tlr._extract_pitem_prose(_SYNTHETIC_PITEM_BODY)
    assert "Memory-bound elementwise kernel" in prose["reasoning_for_slowdown"]
    assert "HBM bandwidth saturated" in prose["reasoning_for_slowdown"]
    assert "Fuse RMSNorm" in prose["resolution"]
    assert "amortize global loads" in prose["resolution"]
    assert prose["impact_low_ms"] == 12.5
    assert prose["impact_low_e2e_pct"] == 3.2
    assert prose["impact_high_ms"] == 40.0
    assert prose["impact_high_e2e_pct"] == 10.4


def test_extract_pitem_prose_returns_empty_strings_when_markers_absent():
    """Bodies without the three labels must still return the full dict
    shape so downstream consumers can rely on key presence."""
    prose = tlr._extract_pitem_prose("**Data:**\n| ... | ... |\n")
    assert prose["reasoning_for_slowdown"] == ""
    assert prose["resolution"] == ""
    assert prose["impact_low_ms"] == 0.0
    assert prose["impact_low_e2e_pct"] == 0.0
    assert prose["impact_high_ms"] == 0.0
    assert prose["impact_high_e2e_pct"] == 0.0


def test_extract_pitem_prose_reasoning_stops_at_resolution_marker():
    """Reasoning should not leak into Resolution when both are present —
    the end-marker ordering is what guarantees a clean split."""
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
    """Round-trip the LLama70B fixture and verify every parsed candidate
    carries the new prose fields populated from its parent P-item block.
    The fixture has 4 Detailed Analysis blocks (P1 GEMM, P2 SDPA_fwd, etc.),
    each with all three sections, so every candidate must end up with
    non-empty reasoning / resolution / both impact halves."""
    cands = tlr.parse_analysis_md(_FIXTURE_LLAMA70B_ANALYSIS_MD, top_k=50)
    assert cands, "fixture must produce at least one candidate"
    # All 21 fixture candidates share P-item prose with their group.
    for c in cands:
        assert "reasoning_for_slowdown" in c
        assert "resolution" in c
        assert "impact_low_ms" in c
        assert "impact_high_ms" in c
        # The fixture's P-items all have non-empty prose; require it.
        assert c["reasoning_for_slowdown"], (
            f"empty reasoning_for_slowdown on candidate {c.get('name')!r} "
            f"(rank P{c.get('tracelens_pitem_rank')})"
        )
        assert c["resolution"], (
            f"empty resolution on candidate {c.get('name')!r}"
        )

    # P1 prose mentions "Tile / wave-occupancy tuning" per the fixture.
    p1_rows = [c for c in cands if c["tracelens_pitem_rank"] == 1]
    assert any(
        "wave-occupancy" in c["resolution"] for c in p1_rows
    ), "P1 resolution should mention wave-occupancy tuning (from fixture)"


# ===========================================================================
# PR-A §2: classify_patchability gate + skip_reason audit field
# ===========================================================================
def test_classify_patchability_accepts_stable_triton_source():
    """Previously-reusable candidate stays reusable; skip_reason is empty."""
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


def test_classify_patchability_rejects_vendor_blas_name_markers():
    """Folded from feature branch _NON_PATCHABLE_MARKERS: rocblas/hipblas/etc.
    rejected even when source_file resolves under a reusable framework root."""
    for marker_name in (
        "rocblas_sgemm_kernel",
        "hipblas_gemm_strided",
        "tensile_gemm_NN_bf16",
        "rccl_AllReduce_sum",
        "nccl_kernel",
        "aten::copy_",
    ):
        reusable, reason = tla.classify_patchability({
            "name": marker_name,
            "source_file": "/sgl-workspace/aiter/foo.py",
            "source_type": "python",
        })
        assert reusable is False, marker_name
        assert "non-patchable" in reason or "PyTorch native" in reason, marker_name


def test_classify_patchability_rejects_aten_without_library():
    """aten::* without a library hint is treated as Tensile / native backend."""
    reusable, reason = tla.classify_patchability({
        "name": "aten::mm",
        "source_file": "/sgl-workspace/aiter/foo.py",
        "source_type": "python",
        "library": "",
    })
    assert reusable is False
    assert "Tensile" in reason or "vendor" in reason


def test_classify_patchability_rejects_aten_tensile_library():
    """Explicit library == 'Tensile' is the most common reject path."""
    reusable, reason = tla.classify_patchability({
        "name": "aten::mm",
        "source_file": "/sgl-workspace/aiter/foo.py",
        "source_type": "python",
        "library": "Tensile",
    })
    assert reusable is False
    assert "Tensile" in reason


def test_classify_patchability_rejects_runtime_generated_kernel():
    reusable, reason = tla.classify_patchability({
        "name": "triton_poi_fused_add_0",
        "source_file": "/tmp/torchinductor_root/ab/cdef.py",
        "source_type": "runtime_generated",
    })
    assert reusable is False
    assert "runtime-generated" in reason


def test_classify_patchability_rejects_unreusable_source_root():
    reusable, reason = tla.classify_patchability({
        "name": "my_custom_kernel",
        "source_file": "/tmp/random/my_custom_kernel.cu",
        "source_type": "hip_cpp",
    })
    assert reusable is False
    assert "reusable framework root" in reason


def test_is_reusable_native_kernel_delegates_to_classify():
    """The bool wrapper must stay in lockstep with classify_patchability."""
    samples = [
        {"name": "rms_norm", "source_file": "", "source_type": "triton"},
        {"name": "rocblas_sgemm", "source_file": "/sgl-workspace/aiter/x.py", "source_type": "python"},
        {
            "name": "triton_attn",
            "source_file": "/sgl-workspace/sglang/python/sglang/x.py",
            "source_type": "triton",
        },
    ]
    for cand in samples:
        assert tla.is_reusable_native_kernel(cand) == tla.classify_patchability(cand)[0]


def test_build_audit_summary_splits_tasks_and_skipped():
    """``build_audit_summary`` must surface kernel name + skip_reason for
    every dropped candidate so operators can answer 'why didn't GEAK see
    kernel X?' from the sidecar alone."""
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
            "recommended_backends": ["geak", "claude", "codex"],
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
    # Reusable tasks must carry recommended_backends so an operator can
    # see which backend each task is routed to without reloading
    # kernel_candidates.json.
    assert summary["tasks"][0]["recommended_backends"] == ["geak", "claude", "codex"]


def test_build_audit_summary_handles_empty_input():
    summary = tla.build_audit_summary([], trace_input="/tmp/x.json.gz")
    assert summary["task_count"] == 0
    assert summary["skipped_count"] == 0
    assert summary["tasks"] == []
    assert summary["skipped"] == []
# ===========================================================================
# PR-B §1: source-function aggregation
# ===========================================================================
def test_parse_launcher_path_extracts_python_frame():
    """``<path>(<line>): <fn>`` is the canonical TraceLens v0.3 shape."""
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
    """Empty / placeholder Kernel Path values must collapse to
    ``("", None, None)`` so source-function aggregation skips the row
    instead of grouping every placeholder under a bogus ``Path("—")``.
    Real bare paths still pass through (caller may resolve them at the
    AST layer)."""
    assert tlr._parse_launcher_path("") == ("", None, None)
    assert tlr._parse_launcher_path("—") == ("", None, None)
    assert tlr._parse_launcher_path("-") == ("", None, None)
    assert tlr._parse_launcher_path("N/A") == ("", None, None)
    # Bare path with no line / fn: function_name resolution falls back
    # to file stem at the _resolve_source_target layer, but the parser
    # itself should leave both fields None.
    path, line, func = tlr._parse_launcher_path("just/a/path.py")
    assert path == "just/a/path.py"
    assert line is None
    assert func is None


def test_function_line_from_ast_finds_def_lineno(tmp_path):
    src = tmp_path / "kernel.py"
    src.write_text(
        "import torch\n\n\ndef other():\n    pass\n\n\ndef rms_norm(x):\n"
        "    return x\n",
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
    """Two candidates that resolve to the same function become one group;
    a third candidate at a different function stays separate."""
    # Create a real Python file so the AST resolver can run.
    src = tmp_path / "rmsnorm.py"
    src.write_text(
        "def rms_norm(x):\n    return x\n\n\ndef other_fn(x):\n    return x\n",
        encoding="utf-8",
    )
    cands = [
        {
            "kernel_id": "k001",
            "name": "rms_norm_call_1",
            "duration_us": 100.0,
            "call_count": 64,
            "gpu_pct": 5.0,
            "tracelens_launcher_path": f"{src}(2): rms_norm",
        },
        {
            "kernel_id": "k002",
            "name": "rms_norm_call_2",
            "duration_us": 50.0,
            "call_count": 32,
            "gpu_pct": 2.5,
            "tracelens_launcher_path": f"{src}(2): rms_norm",
        },
        {
            "kernel_id": "k003",
            "name": "other_fn_call",
            "duration_us": 30.0,
            "call_count": 16,
            "gpu_pct": 1.5,
            "tracelens_launcher_path": f"{src}(5): other_fn",
        },
    ]
    groups = tlr.aggregate_by_source_function(cands)
    assert len(groups) == 2
    # Heaviest group (rms_norm: 150 us aggregate) comes first.
    g0, g1 = groups
    assert g0["function_name"] == "rms_norm"
    assert g0["task_group_id"] == "tg001"
    assert set(g0["kernel_ids"]) == {"k001", "k002"}
    assert g0["primary_kernel_id"] == "k001"  # highest duration_us
    assert g0["aggregate_duration_us"] == 150.0
    assert g0["aggregate_call_count"] == 96
    assert g0["aggregate_gpu_pct"] == 7.5
    # AST resolved the launcher line=2 → AST FunctionDef lineno=1.
    assert g0["definition_line"] == 1
    assert g0["ast_resolved"] is True

    assert g1["function_name"] == "other_fn"
    assert g1["task_group_id"] == "tg002"
    assert g1["kernel_ids"] == ["k003"]
    # Fixture: line 1 ``def rms_norm``, blank lines 3-4, ``def other_fn``
    # on line 5; AST resolves the launcher's reported line=5 to itself.
    assert g1["definition_line"] == 5


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


# PR-B §1 + §2: build_task_groups (tracelens_analysis.py wrapper)
# ===========================================================================
def test_build_task_groups_filters_non_reusable():
    """build_task_groups skips candidates with reusable_native_kernel=False
    so vendor / aten:: / runtime-generated kernels never appear in a
    group's kernel_ids."""
    cands = [
        {
            "kernel_id": "k001", "name": "rms_norm",
            "duration_us": 50.0, "call_count": 4,
            "tracelens_launcher_path": "aiter/rmsnorm.py(1): rms_norm",
            "reusable_native_kernel": True,
        },
        {
            "kernel_id": "k002", "name": "rocblas_sgemm",
            "duration_us": 80.0, "call_count": 2,
            "tracelens_launcher_path": "aiter/rmsnorm.py(1): rms_norm",
            "reusable_native_kernel": False,  # filtered out
        },
    ]
    groups = tla.build_task_groups(cands)
    assert len(groups) == 1
    assert groups[0]["kernel_ids"] == ["k001"]
    assert "k002" not in groups[0]["kernel_ids"]


# ===========================================================================
# PR-B §1: summary.json carries task_groups[] view
# ===========================================================================
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
