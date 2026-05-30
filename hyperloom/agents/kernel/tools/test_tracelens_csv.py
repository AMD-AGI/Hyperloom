"""Regression tests for the kernel-agent tracelens_analysis filter fixes.

Locks the raw trace filtering fix uncovered by the resume3/resume4 1h
validation:

* **A path** ``is_kernel_event`` previously did fuzzy substring matching
  on `KERNEL_HINTS=("kernel","triton","hip","cuda",...)` against both the
  event name AND category. That accidentally promoted `cat='python_function'`
  rows like ``torch/cuda/streams.py(222): synchronize`` (a CPU wait that
  accumulates the entire wrapped GPU duration) to be the #1 hot kernel —
  88ms attributed to a CPU sync. Fix: require ``cat == 'kernel'`` strictly.

The production TraceLens interface now consumes only ``analysis.md``. Legacy
``priority_data`` / ``category_data`` / CSV fallbacks are intentionally gone.
"""

from __future__ import annotations

import os
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
    assert tla.is_runtime_generated_kernel(
        candidate["name"], candidate["source_file"],
    ) is False
    assert tla.is_reusable_native_kernel(candidate) is True
    # Without CURSOR_API_KEY: recommendation excludes cursor (auto-skip).
    monkeypatch.delenv("CURSOR_API_KEY", raising=False)
    assert tla.recommend_backends(candidate) == ["geak", "claude", "codex"]
    # With CURSOR_API_KEY: cursor is appended to the recommendation tail.
    monkeypatch.setenv("CURSOR_API_KEY", "crsr_test_dummy")
    assert tla.recommend_backends(candidate) == ["geak", "claude", "codex", "cursor"]


def test_recommend_backends_includes_geak_for_python_source():
    """Policy: every kernel Claude/Codex can rewrite, GEAK can rewrite
    too. Pre-fix, ``python`` source_type returned ``["claude", "codex"]``
    and dropped GEAK — that excluded e.g. the hottest kernel on a
    Qwen3-30B-A3B run (`fused_moe_kernel` mis-resolved to a Python
    benchmark harness) from ever reaching GEAK. Post-fix GEAK is in
    the ladder for python too; the AST resolver (PR-B.1) addresses
    the underlying misclassification but the policy change is the
    safety net for cases the resolver can't disambiguate."""
    candidate = {
        "name": "some_python_dispatcher",
        "source_file": "/sgl-workspace/sglang/python/sglang/srt/layers/dispatcher.py",
        "source_type": "python",
        "reusable_native_kernel": True,
    }
    assert tla.recommend_backends(candidate) == ["geak", "claude", "codex"]


def test_recommend_backends_includes_geak_for_unknown_source():
    """Fallback / unknown source_type path: GEAK must still be in the
    ladder. The capability differences are GEAK-side, not
    Hyperloom-side — let GEAK decide what to handle rather than
    pre-filtering by extension."""
    candidate = {
        "name": "some_unrecognised_kernel",
        "source_file": "/some/path/kernel.xyz",
        "source_type": "unknown",
        "reusable_native_kernel": True,
    }
    assert tla.recommend_backends(candidate) == ["geak", "claude", "codex"]


def test_recommend_backends_geak_is_first_in_ladder():
    """Invariant: when GEAK is in the ladder, it is FIRST. High-priority
    handoff means GEAK gets the swing before Claude/Codex; the
    fallback order matters at runtime if GEAK times out or rejects."""
    candidate = {
        "name": "some_kernel",
        "source_file": "/sgl-workspace/sglang/python/sglang/srt/layers/x.py",
        "source_type": "triton",
        "reusable_native_kernel": True,
    }
    ladder = tla.recommend_backends(candidate)
    assert ladder and ladder[0] == "geak", (
        f"GEAK must be first in the ladder, got {ladder}"
    )


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
    # write_reports now requires the upstream TraceLens v0.3 analysis.md
    # (see #203). Provide a stub so the function reaches the JSON-writing
    # branch we are exercising here.
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
        # Per AMD-AGI/Hyperloom#314, ``kernel_candidates.json::hot_kernels``
        # now only carries candidates that ``classify_patchability`` marked
        # routable. In production this field is set by
        # ``_finalize_candidates`` before ``write_reports`` runs; this
        # unit test bypasses ``_finalize_candidates`` and passes a raw
        # candidate straight into ``write_reports``, so set the routing
        # marker explicitly to mirror the production fixture and keep
        # the downstream assertions on ``hot_kernels[0]`` valid.
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
        # Per AMD-AGI/Hyperloom#314, see twin fixture above.
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


# ===========================================================================
# #203 — write_reports must surface the upstream analysis.md as-is
# (no copies, no aliases, no inline fabricated fallback)
# ===========================================================================
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
    """#203: write_reports refuses to fabricate a Markdown when the
    TraceLens v0.3 SDK orchestrator failed to produce analysis.md.
    The legacy inline bullet-list fallback silently masked upstream
    failures (see #144 mis-resolution chain) and is gone.
    """
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
    """#203: even if a path is passed, the file must actually exist —
    a non-existent path is treated as orchestrator failure, not as a
    cue to fabricate a stand-in.
    """
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
    """#203: ``analysis.md`` is the single contracted exit. The legacy
    ``standalone_analysis.md`` / ``tracelens_report.md`` aliases were
    removed because they wrote byte-identical copies of the same file
    under different names. This test pins that hygiene fix.
    """
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
    """#203: Hyperloom must not rewrite the upstream report's contents.
    Verifying byte-identity here prevents a future refactor from
    sneaking a re-render in.
    """
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
        # TraceLens v0.3 contract: orchestrator writes ``analysis.md``.
        # The legacy ``standalone_analysis.md`` fallback was dropped in #203.
        (output_dir / "analysis.md").write_text("# report\n", encoding="utf-8")
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
    assert res.artifact_paths["tracelens_agent_report"] == str(res.report_path)
    assert "analysis-orchestrator" in captured["prompt"] or "skill.md" in captured["prompt"]
    assert "Bash" in captured["options"]["allowed_tools"]
    assert "Task" in captured["options"]["allowed_tools"]


# ===========================================================================
# T2 — analysis.md is the only contracted TraceLens output.
# ===========================================================================
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
            _json.dumps({"findings": []}), encoding="utf-8",
        )
        (output_dir / "category_data").mkdir(parents=True, exist_ok=True)
        (output_dir / "category_data" / "category_manifest.json").write_text(
            _json.dumps({"categories": []}), encoding="utf-8",
        )
        yield _Message(content=[_TextBlock("done — sidecars ignored")])

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

    assert res.report_path.exists(), "analysis.md is the single source of truth and must exist"
    assert "tracelens_agent_report" in res.artifact_paths
    assert "tracelens_priority_data" not in res.artifact_paths
    assert "tracelens_category_manifest" not in res.artifact_paths


def test_t2_missing_analysis_md_still_raises(tmp_path):
    """Negative control: ``analysis.md`` itself is still the contracted
    single source of truth, so its absence is still a hard error. T2
    only relaxes the sidecars, not the report."""
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
        # Deliberately write ONLY a sidecar (no analysis.md). The wrapper
        # must still fail loudly because analysis.md is the contracted
        # report (docx §2 "single source of truth").
        (output_dir / "priority_data.json").write_text("{}", encoding="utf-8")
        yield _Message(content=[_TextBlock("done")])

    with pytest.raises(RuntimeError, match="analysis.md"):
        asyncio.run(tlr.run_tracelens_skill(
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


# ===========================================================================
# #127 — TraceLens splitter CLI must match the real
# split_inference_trace_annotation interface (positional trace_path,
# -o/--output-dir, --find-steady-state). The previous --input/--platform
# form failed at runtime against a real Magpie/SGLang trace.
# ===========================================================================
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


def test_127_splitter_cli_uses_positional_trace_path_and_find_steady_state(
    tmp_path, capsys,
):
    """The end-to-end split path must call the real splitter interface,
    not the broken --input/--platform form. Drives a mock subprocess.run
    and asserts argv shape."""
    import gzip
    import json as _json
    from unittest.mock import patch

    # Pretend TraceLens root is present so the run reaches the splitter step.
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
        # Make pip install and splitter invocations succeed.
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
        "--split-conc", "8",
        "--split-osl", "1024",
    ]
    import os as _os
    env_backup = dict(_os.environ)
    try:
        with patch.object(tla.subprocess, "run", side_effect=fake_run), \
             patch.object(tla.sys, "argv", argv):
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

    assert all(
        not (
            c
            and "TraceLens_generate_perf_report_pytorch_inference" in str(c[0])
            and "--profile_json_path" in c
        )
        for c in captured
    ), (
        "perf-report CSV fallback must not run; analysis.md is the single "
        f"source of truth. cmds={captured}"
    )
    out = capsys.readouterr().out
    result = _json.loads(out)
    assert result["status"] == "failed"
    assert "trace_split_no_steady_state" in result["error"]


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
# docx §2 Recommended Interfacing Approach — Filter for GEAK based on
# budget (Higher P-item, Lower Efficiency)
# ===========================================================================
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
    """docx §2 Recommended Interfacing Approach: ``Filter for GEAK based
    on budget (Higher P-item, Lower Efficiency)``. Within a P-item, rows
    with lower efficiency must come first so they survive the ``top_k``
    budget cap. Cross-P-item order is still rank-based (P1 before P2)."""
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
    """docx §2 budget cap: after sorting by efficiency within a P-item,
    the ``top_k`` slice must keep the lowest-efficiency rows. Without
    the sort, a budget of 2 would drop the kernel with the most
    headroom — exactly the regression docx §2 calls out."""
    md = tmp_path / "analysis.md"
    _write_two_pitem_analysis_md(md)

    cands = tlr.parse_analysis_md(md, top_k=2)
    names = [c["name"] for c in cands]
    assert names == ["low_eff_gemm", "mid_eff_gemm"], (
        "top_k=2 must keep the two lowest-efficiency P1 rows; the "
        "high-efficiency / unknown rows must be dropped before any P2 row"
    )


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
    """Identification ends at ``**Data:**`` — must NOT leak the
    9-column table or any subsequent prose into the identification
    field. Without the Data end-marker the identification would
    swallow everything up to ``**Reasoning for Slowdown:**``."""
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
    """Bodies without the four labels must still return the full dict
    shape so downstream consumers can rely on key presence."""
    prose = tlr._extract_pitem_prose("**Data:**\n| ... | ... |\n")
    assert prose["identification"] == ""
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
        assert "identification" in c
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
# parse_analysis_md — TraceLens v0.3 spec § Operations Table Schema
# tolerance for trailing category-specific extra columns.
#
# The spec at TraceLens-internal/.../utils/templates/sub_agent_spec.md
# (Operations Table Schema, compute tier) explicitly allows sub-agents to
# "append extra columns at the end when needed (e.g. Sub-Category in the
# generic-op analyzer)" as long as the first 9 canonical columns are
# present in order. Real TraceLens runs on attention-bound models append
# 3 extras (Dominant Kernel / Workload / Attention Pattern) for the
# inferenceattention category; generic-op-analyzer appends 1 extra
# (Sub-Category). The parser must accept those rows verbatim, not skip
# them silently.
# ===========================================================================
_FIXTURE_QWEN3_ATTENTION_ANALYSIS_MD = (
    Path(__file__).resolve().parents[1]
    / "tests"
    / "fixtures"
    / "tracelens_v03_qwen3_moe_attention_analysis.md"
)


def test_parse_analysis_md_tolerates_attention_12_column_table_per_spec():
    """Real-world reproducer: Qwen3-30B MoE inferenceattention P-item.

    TraceLens emitted a 12-column ``**Data:**`` table for the attention
    category (canonical 9 + ``Dominant Kernel`` + ``Workload`` +
    ``Attention Pattern``). The previous strict-equality header check
    skipped the block entirely, returning ``hot_kernels=[]`` and
    starving a 1.5h KERNEL phase of any candidate to optimize. Per
    ``sub_agent_spec.md`` (Operations Table Schema, compute tier),
    appending extra columns at the end is spec-compliant, so the parser
    must consume the row using only the first 9 cells.
    """
    cands = tlr.parse_analysis_md(_FIXTURE_QWEN3_ATTENTION_ANALYSIS_MD, top_k=10)
    assert len(cands) == 1, (
        f"expected 1 attention candidate from the 12-column fixture; got "
        f"{len(cands)}"
    )
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
    # The three trailing extra cells (Dominant Kernel / Workload /
    # Attention Pattern) are SPEC-allowed extras (sub_agent_spec.md
    # § Operations Table Schema: "Agents may append extra columns at
    # the end when needed"). The parser preserves them verbatim under
    # ``tracelens_extra_columns`` so downstream consumers (GEAK / OOB)
    # have programmatic access to category-specific metadata without
    # re-parsing analysis.md.
    extras = c.get("tracelens_extra_columns")
    assert extras is not None, "tracelens_extra_columns missing for 12-col row"
    assert extras.get("dominant kernel") == "`_fwd_kernel` (93.61%)"
    assert extras.get("workload") == "unknown"
    assert extras.get("attention pattern") == "GQA (8:1)"
    # Canonical fields must NOT leak into extras (they belong on the
    # candidate top-level as typed fields).
    for canonical_key in (
        "operation", "args", "kernel path", "time (ms)", "%e2e",
        "count", "flops/byte", "efficiency", "bound",
    ):
        assert canonical_key not in extras


def test_parse_analysis_md_tolerates_subcategory_10_column_table_per_spec(tmp_path):
    """The generic-op analyzer appends a ``Sub-Category`` column for
    uncategorized ops (`other` category). The parser must accept the
    row using only the first 9 cells, dropping the Sub-Category cell
    as expected.
    """
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
    # ``Sub-Category`` is preserved verbatim for downstream GEAK / OOB
    # access; it never leaks into the candidate top-level.
    extras = c.get("tracelens_extra_columns")
    assert extras is not None
    assert extras.get("sub-category") == "scatter_gather"
    assert "sub-category" not in c


def test_parse_analysis_md_canonical_9_column_table_has_no_extras_key():
    """Regression guard: candidates parsed from a canonical 9-column
    table (the Llama70B fixture) must NOT carry the
    ``tracelens_extra_columns`` field. Adding the key with an empty
    dict would force downstream consumers to special-case "extras
    present but empty" vs "extras absent".
    """
    cands = tlr.parse_analysis_md(_FIXTURE_LLAMA70B_ANALYSIS_MD, top_k=50)
    assert cands, "Llama70B fixture must produce candidates"
    for c in cands:
        assert "tracelens_extra_columns" not in c, (
            f"canonical 9-col candidate {c.get('name')!r} unexpectedly "
            f"carries tracelens_extra_columns={c.get('tracelens_extra_columns')!r}"
        )


def test_parse_analysis_md_rejects_fewer_than_canonical_columns(tmp_path):
    """Regression guard: a table missing any of the 9 canonical columns
    must still be skipped — silent wrong-mapping would be worse than a
    missed candidate. Here ``Bound`` is dropped (8 columns total).
    """
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
    """Regression guard: spec requires the 9 canonical columns ``in this
    order``. Swapping any two breaks the row→field mapping silently, so
    the parser must skip the block instead of producing rows whose
    ``efficiency`` / ``bound`` are misread.
    """
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


# ---------------------------------------------------------------------------
# _resolve_launcher_to_abs_source — TraceLens launcher path → absolute file.
#
# torch.profiler strips ``sys.path`` prefixes from ``__file__`` before
# writing Python frame names, so TraceLens forwards strings like
# ``aiter/ops/rmsnorm.py(62): rmsnorm2d_fwd``. Without resolution the
# patchability gate rejects every row as ``source not under a reusable
# framework root``. These tests pin the three production-image
# resolution paths (importlib spec, env override, hardcoded fallback)
# plus the placeholder and absolute-path no-op cases.
# ---------------------------------------------------------------------------


def _seed_pkg(tmp_path, pkg: str, relpath: str, funcs: tuple[str, ...] = ()) -> Path:
    """Create ``<tmp_path>/<pkg>/<relpath>`` and return the absolute file.

    ``funcs`` lets a test declare top-level ``def`` names that resolver
    AST validation expects to find. Empty by default for non-Python
    fixtures (``.cu`` etc.) where AST is intentionally skipped.
    """
    target = tmp_path / pkg / relpath
    target.parent.mkdir(parents=True, exist_ok=True)
    body = ["# stub for resolver tests\n"]
    for fn in funcs:
        body.append(f"def {fn}(*args, **kwargs):\n    pass\n")
    target.write_text("".join(body), encoding="utf-8")
    return target


def test_resolve_launcher_via_env_override(tmp_path, monkeypatch):
    """``$HYPERLOOM_FRAMEWORK_SOURCE_ROOTS`` is the highest-priority
    resolver source. Operator can pre-stage any framework root layout
    without needing the package to be importable in the current
    interpreter."""
    target = _seed_pkg(
        tmp_path, "aiter", "ops/rmsnorm.py", funcs=("rmsnorm2d_fwd",),
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
    """When no env override is set, ``importlib.util.find_spec`` walks
    the live ``sys.path`` and returns the absolute origin. This is the
    path that should fire on a regular production pod (aiter / sglang
    are editable installs at /sgl-workspace, vllm sits under
    dist-packages).

    ``unittest`` is shipped with every CPython interpreter (no
    production deps needed), and ``unittest/case.py`` always defines
    the ``expectedFailure`` decorator — perfect for pinning the
    find_spec branch AND the AST-symbol guard without depending on
    aiter/sglang/vllm being installed in the test env."""
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
    """When the package isn't importable but a hardcoded fallback root
    holds the file on disk, the resolver still succeeds. This is the
    safety net for static-analysis paths that parse CSVs without
    actually importing ``aiter`` / ``sglang`` / ``vllm``."""
    monkeypatch.delenv(tlr._FRAMEWORK_SOURCE_ROOTS_ENV, raising=False)

    # Seed a fake aiter checkout under tmp_path and force the fallback
    # table to point at it. The package is NOT made importable, so the
    # ``find_spec`` branch must miss and the resolver must fall through
    # to the hardcoded table.
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
    # Force find_spec to return None so the test exercises the
    # fallback table even if some operator pre-installed a real
    # ``aiter_pinned_xfx`` package.
    monkeypatch.setattr(tlr, "_package_root_parent", lambda pkg: None)

    resolved = tlr._resolve_launcher_to_abs_source(
        "aiter_pinned_xfx/ops/rmsnorm.py(76): rmsnorm2d_fwd_with_add",
    )
    assert resolved is not None
    abs_path, line, func = resolved
    assert abs_path == str(tmp_path / "aiter_pinned_xfx" / "ops" / "rmsnorm.py")
    assert line == 76
    assert func == "rmsnorm2d_fwd_with_add"


def test_resolve_launcher_returns_none_for_absolute_path():
    """Already-absolute launcher paths are returned as None so the
    caller preserves the original string verbatim — there's nothing
    to rewrite, and pretending we resolved would silently swallow
    paths the operator deliberately set to a non-package location."""
    assert (
        tlr._resolve_launcher_to_abs_source(
            "/sgl-workspace/aiter/aiter/ops/rmsnorm.py(62): fn",
        )
        is None
    )


def test_resolve_launcher_returns_none_for_placeholders_and_misses():
    """Placeholders, empty strings, and unresolvable packages must
    collapse to None so the patchability gate emits its normal
    ``source file not resolved`` / ``source not under a reusable
    framework root`` rejection (vs. a fabricated path)."""
    assert tlr._resolve_launcher_to_abs_source("") is None
    assert tlr._resolve_launcher_to_abs_source("—") is None
    # Package doesn't exist anywhere → resolver gives up; caller falls
    # back to the verbatim launcher string.
    assert (
        tlr._resolve_launcher_to_abs_source(
            "definitely_not_a_real_pkg_8x9z/foo.py(1): fn",
        )
        is None
    )


def test_resolve_launcher_rejects_when_function_not_in_file(tmp_path, monkeypatch):
    """AST validation guard: when the resolved ``.py`` exists but does
    NOT define the launcher's function, the resolver MUST refuse the
    path. This catches two real failure modes — ``sys.path`` shadowing
    (find_spec returns the wrong same-named package) and operator
    misconfiguration of ``$HYPERLOOM_FRAMEWORK_SOURCE_ROOTS`` pointing
    at a stub/snapshot tree. Without this check GEAK would receive a
    real-on-disk source path that doesn't host the kernel."""
    # Seed a real .py file under tmp_path but DO NOT define
    # rmsnorm2d_fwd in it.
    target = tmp_path / "aiter_shadowed_xyz" / "ops" / "rmsnorm.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "def some_other_function():\n    pass\n", encoding="utf-8",
    )
    monkeypatch.setenv(
        tlr._FRAMEWORK_SOURCE_ROOTS_ENV,
        f"aiter_shadowed_xyz={tmp_path}",
    )
    # No fallback paths should rescue this — we want to assert the
    # bad-symbol path is rejected outright.
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
        "def rmsnorm2d_fwd(x):\n    return x\n", encoding="utf-8",
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
    """AST validation only applies to Python sources. HIP/CUDA refs
    (``<path>#L<line>`` shape) and bare ``.cu`` paths must pass
    existence-only validation since ``ast.parse`` cannot walk them.
    Pin this so we don't accidentally regress HIP kernel resolution
    when the gate hardens."""
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
    """Malformed ``$HYPERLOOM_FRAMEWORK_SOURCE_ROOTS`` entries are
    silently skipped — a single bad export must not poison the whole
    resolver."""
    target = _seed_pkg(
        tmp_path,
        "vllm",
        "model_executor/models/qwen.py",
        funcs=("forward",),
    )
    monkeypatch.setenv(
        tlr._FRAMEWORK_SOURCE_ROOTS_ENV,
        # First entry has no '=' (skipped); second is malformed (no key);
        # third is the only valid one and must win.
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
    """Two candidates that share the same Operation name AND resolve to
    the same source function become one group; a third candidate at a
    different function stays separate.

    Uses the SAME ``name`` for the two grouped candidates because that
    matches TraceLens's real-world contract: rows in a Detailed Analysis
    Data table that share a Kernel Path also share the Operation name
    by construction (see standalone_analysis.md examples). The
    grouping key is ``(operation, source_path, line, function)`` —
    different operations sharing a Python wrapper would NOT merge."""
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


def test_aggregate_does_not_merge_different_operations_sharing_wrapper(tmp_path):
    """Q1 invariant: two semantically-distinct kernel operations that
    happen to share the same Python wrapper (same Kernel Path) MUST
    stay in separate task_groups. This is the real-world hazard the
    user surfaced — P1 ``vllm::rocm_unquantized_gemm`` and P2
    ``vllm::rocm_aiter_triton_add_rmsnorm_pad`` both have Kernel Path
    ``vllm/model_executor/models/gpt_oss.py(283): forward`` because
    that's the calling Python frame, not the kernel implementation.
    Keying on source function alone would merge them into one
    meaningless ``rewrite forward`` task; including operation_name
    keeps each kernel identity intact."""
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


def test_aggregate_collects_distinct_pitem_prose_when_function_spans_pitems(tmp_path):
    """Q2 invariant: when the same operation+source-function legitimately
    appears in MULTIPLE TraceLens P-items (e.g. the same kernel
    classified once at decode shapes and again at prefill shapes,
    yielding two distinct prose tuples), every P-item's prose is
    collected on the task_group's ``all_pitem_prose`` list, deduped
    by ``(rank, title)`` and sorted by rank ascending so P1 reads
    first."""
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
    assert len(prose) == 2, (
        f"expected 2 distinct (rank,title) prose entries; got {len(prose)}"
    )
    # Sorted by rank ascending → P2 first, P5 second.
    assert prose[0]["rank"] == 2
    assert "decode" in prose[0]["title"].lower()
    assert prose[0]["reasoning_for_slowdown"] == "Decode-shape Reasoning."
    assert prose[1]["rank"] == 5
    assert "prefill" in prose[1]["title"].lower()
    assert prose[1]["resolution"] == "Prefill-shape Resolution."
    # Set-typed bookkeeping must not leak into the returned dict
    # (would break JSON serialization in summary.json / kernel_candidates.json).
    assert "_pitem_prose_seen" not in g


def test_same_kernel_different_shapes_yields_one_task_with_all_shapes_as_cases(
    tmp_path,
):
    """End-to-end pin for the most common P-item shape: same Operation
    name + same source function + DIFFERENT Args (shapes) per row.

    This is exactly the P1 ``vllm::rocm_unquantized_gemm`` pattern from
    the user screenshot — 4 rows of one kernel at 4 distinct shape
    tuples. The contract end-to-end:

    1. ``aggregate_by_source_function`` collapses all rows into ONE
       ``task_group`` (Q1 invariant: same operation+source key).
    2. The group's ``rows[]`` preserves each candidate's own ``shapes``
       list verbatim — no shape de-duplication, no cross-row mixing.
    3. ``primary_kernel_id`` is the heaviest (max ``duration_us``).
    4. ``_build_benchmark_cases_block`` renders one ``Case N:`` line
       per row, each with that row's own Args / time / count / bound /
       efficiency. GEAK sees every shape variant as its own benchmark
       case it can target individually.
    """
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
    assert len(groups) == 1, (
        f"expected 1 task_group (same op + same source); got {len(groups)}"
    )
    g = groups[0]
    assert g["operation"] == "vllm::rocm_unquantized_gemm"
    assert set(g["kernel_ids"]) == {"k001", "k002", "k003", "k004"}
    assert g["primary_kernel_id"] == "k001"  # heaviest (12704 us)
    assert len(g["rows"]) == 4
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
        "k002's unique shape must land in Case 2 — confirms shape "
        "preservation per-row, not cross-row merging"
    )
    # Same for k003's unique ``(2880,512)`` shape → Case 3.
    case3_segment = block.split("Case 3:")[1].split("Case 4:")[0]
    assert "(2880,512) bf16" in case3_segment
    # And k004's unique ``(2048,2880)`` shape → Case 4.
    case4_segment = block.split("Case 4:")[1]
    assert "(2048,2880) bf16" in case4_segment


def test_aggregate_drops_empty_prose_entries(tmp_path):
    """Candidates from non-Detailed-Analysis paths (raw-trace fallback)
    have rank=0 and no prose. They contribute exactly one bookkeeping
    entry to ``all_pitem_prose`` during aggregation, but the
    post-process step drops it so JSON consumers see an empty list,
    not a noise entry."""
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


# ===========================================================================
# _default_workspace_path — USER_DATA_PATH rollout for TraceLens (#203)
#
# Locks the fallback chain so a regression that flips precedence (e.g.
# putting WORKSPACE_PATH first) would fail loudly. GEAK / OOB / install.sh
# intentionally still default to $WORKSPACE_PATH; only TraceLens migrated
# in this PR.
# ===========================================================================
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
    """No envs set → hard-coded default matches inference_optimizer/paths.DEFAULT_SESSION_DIR."""
    monkeypatch.delenv("USER_DATA_PATH", raising=False)
    monkeypatch.delenv("WORKSPACE_PATH", raising=False)
    assert tla._default_workspace_path() == "/workspace/hyperloom"


def test_default_workspace_path_treats_empty_user_data_path_as_unset(monkeypatch):
    """An empty USER_DATA_PATH must not shadow a real WORKSPACE_PATH; ``or`` semantics."""
    monkeypatch.setenv("USER_DATA_PATH", "")
    monkeypatch.setenv("WORKSPACE_PATH", "/legacy/workspace")
    assert tla._default_workspace_path() == "/legacy/workspace"


# ===========================================================================
# T3 — Idle-% sanity gate on the Executive Summary
# ===========================================================================
# Per Report_Interfacing.docx §1 (Executive Summary schema) and §2
# (idle-gate sanity check), the Executive Summary table reports
# ``Idle %`` (e.g. ``| Idle % | 0.25% |``). When idle time dominates wall
# clock, kernel-level rewriting cannot improve end-to-end latency — the
# operator should pivot to parameter optimization (batch size, KV cache
# shape, prefill/decode split). The default threshold is 80% (raised
# from the docx-suggested 20% after observing every production
# Qwen3-32B trace land in the 48–60% band; see
# ``HIGH_IDLE_PCT_THRESHOLD_DEFAULT`` docstring), overridable via
# ``HYPERLOOM_TRACELENS_IDLE_PCT_THRESHOLD``.

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
    """Older / partial reports without an Idle % row degrade gracefully
    to ``None`` so the runtime gate skips rather than failing the run."""
    md = tmp_path / "analysis.md"
    md.write_text(_EXEC_SUMMARY_NO_IDLE_ROW, encoding="utf-8")
    assert tlr.extract_idle_pct_from_analysis_md(md) is None


def test_extract_idle_pct_returns_none_when_file_missing(tmp_path):
    assert tlr.extract_idle_pct_from_analysis_md(tmp_path / "nope.md") is None


def test_extract_idle_pct_against_llama70b_fixture():
    """Real TraceLens v0.3 fixture: Llama 3 70B has Idle % = 0.25% in
    its Executive Summary — pin this against drift in the regex."""
    fixture = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "tracelens_v03_llama70b_analysis.md"
    assert fixture.exists(), f"fixture must be present: {fixture}"
    assert tlr.extract_idle_pct_from_analysis_md(fixture) == pytest.approx(0.25)


def test_resolve_idle_pct_threshold_uses_default_when_env_unset(monkeypatch):
    monkeypatch.delenv(tla.HIGH_IDLE_PCT_THRESHOLD_ENV, raising=False)
    assert tla._resolve_idle_pct_threshold() == tla.HIGH_IDLE_PCT_THRESHOLD_DEFAULT


def test_resolve_idle_pct_threshold_honours_env_override(monkeypatch):
    monkeypatch.setenv(tla.HIGH_IDLE_PCT_THRESHOLD_ENV, "35.5")
    assert tla._resolve_idle_pct_threshold() == pytest.approx(35.5)


def test_resolve_idle_pct_threshold_rejects_nonsense_env_value(monkeypatch):
    """Operators who paste garbage into the env var should get the default,
    not a crash. The shape of this code defends against silent failure
    by validating ``float()`` and the non-negative guard."""
    monkeypatch.setenv(tla.HIGH_IDLE_PCT_THRESHOLD_ENV, "not-a-float")
    assert tla._resolve_idle_pct_threshold() == tla.HIGH_IDLE_PCT_THRESHOLD_DEFAULT
    monkeypatch.setenv(tla.HIGH_IDLE_PCT_THRESHOLD_ENV, "-5")
    assert tla._resolve_idle_pct_threshold() == tla.HIGH_IDLE_PCT_THRESHOLD_DEFAULT
    monkeypatch.setenv(tla.HIGH_IDLE_PCT_THRESHOLD_ENV, "")
    assert tla._resolve_idle_pct_threshold() == tla.HIGH_IDLE_PCT_THRESHOLD_DEFAULT


def test_build_high_idle_warning_shape(tmp_path):
    """The structured warning is the contract between
    ``tracelens_analysis`` and ``trace_analyze_handler`` (T4). Pin the
    shape: code, severity, idle_pct (rounded), threshold_pct (rounded),
    source path, and a human-readable message that names both numbers."""
    report = tmp_path / "analysis.md"
    report.write_text("# noop\n", encoding="utf-8")
    # 42.567 → round-to-2 = 42.57 (unambiguous, avoids banker's-rounding
    # ties that bite e.g. 42.345 → 42.34 on Python's round()).
    w = tla._build_high_idle_warning(
        idle_pct=42.567, threshold_pct=20.0, report_path=report,
    )
    assert w["code"] == "high_gpu_idle_pct"
    assert w["severity"] == "warning"
    assert w["idle_pct"] == pytest.approx(42.57)
    assert w["threshold_pct"] == pytest.approx(20.0)
    assert w["source"] == str(report)
    # The pre-rounded value (3 d.p.) shows up in the message via :.2f
    # formatting → "42.57%", and the threshold uses the same formatter.
    assert "42.57%" in w["message"]
    assert "20.00%" in w["message"]
    assert "parameter optimization" in w["message"]


def test_build_audit_summary_propagates_trace_health_warnings():
    """``summary.json`` (the audit sidecar) must surface the same
    structured warnings as the JSON-RPC ``result`` so an operator
    inspecting the on-disk artefact and the live response see the same
    findings."""
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
    """Steady-state (no findings) is the empty list — never ``None`` —
    so downstream consumers can ``for w in summary[...]`` without a
    ``None`` guard."""
    summary = tla.build_audit_summary(
        [],
        trace_input="/tmp/trace.json.gz",
        framework="sglang",
        target_platform="MI300X",
        task_groups=[],
    )
    assert summary["trace_health_warnings"] == []
