from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest

import kernel_optimization as ko


def _args(**overrides):
    base = {
        "micro_speedup": None,
        "e2e_gain_pct": None,
        "accuracy_passed": None,
        "correctness_passed": None,
        "dry_run": False,
        "source_file": "/tmp/source.hip",
    }
    base.update(overrides)
    return Namespace(**base)


def _attempt(report: Path | None = None, artifact: Path | None = None):
    paths = {}
    if report is not None:
        paths["report"] = str(report)
        if artifact is None:
            artifact = report.parent / "optimized.hip"
    if artifact is not None:
        if not artifact.exists():
            artifact.write_text(
                "#include <hip/hip_runtime.h>\nextern \"C\" void optimized_kernel() {}\n",
                encoding="utf-8",
            )
        paths["partial_latest_optimized"] = str(artifact)
    return {
        "status": "completed",
        "attempt_id": "a1",
        "backend": "claude",
        "optimized_path": str(artifact or "/tmp/optimized.hip"),
        "backend_paths": paths,
    }


def test_benchmark_available_alone_does_not_pass_correctness(tmp_path):
    verification = ko.build_verification(
        _args(micro_speedup=1.3),
        [_attempt()],
        benchmark_available=True,
    )
    assert verification["compile_passed"] is True
    assert verification["correctness_passed"] is False
    assert verification["correctness_source"] == "missing"
    proposal = ko.make_proposal(verification)
    assert proposal["decision"] == "NEEDS_REVIEW"
    assert "correctness evidence missing or failed" in proposal["reasons"]


def test_report_correctness_passes_when_explicit(tmp_path):
    report = tmp_path / "optimization_report.md"
    report.write_text(
        "Correctness passed\nSpeedup: 1.32x\n",
        encoding="utf-8",
    )
    verification = ko.build_verification(
        _args(e2e_gain_pct=1.0, accuracy_passed=True),
        [_attempt(report)],
        benchmark_available=True,
    )
    assert verification["correctness_passed"] is True
    assert verification["correctness_source"] == "report_scan"
    assert verification["micro_speedup"] == 1.32
    assert ko.make_proposal(verification)["decision"] == "KEEP"


def test_report_correctness_passes_with_machine_marker(tmp_path):
    report = tmp_path / "optimization_report.md"
    report.write_text(
        "Compared with the baseline.\n[CORRECTNESS] PASS\n[MICRO_SPEEDUP] 1.28x\n",
        encoding="utf-8",
    )
    verification = ko.build_verification(
        _args(e2e_gain_pct=1.0, accuracy_passed=True),
        [_attempt(report)],
        benchmark_available=True,
    )
    assert verification["correctness_passed"] is True
    assert verification["correctness_source"] == "report_scan"
    assert verification["micro_speedup"] == 1.28


def test_report_correctness_passes_with_reference_language(tmp_path):
    report = tmp_path / "optimization_report.md"
    report.write_text(
        "The optimized implementation matches reference outputs for all test shapes.\n"
        "Speedup: 1.41x\n",
        encoding="utf-8",
    )
    verification = ko.build_verification(
        _args(e2e_gain_pct=1.0, accuracy_passed=True),
        [_attempt(report)],
        benchmark_available=True,
    )
    assert verification["correctness_passed"] is True
    assert verification["correctness_source"] == "report_scan"


def test_extracts_complete_source_from_text_artifact(tmp_path):
    artifact = tmp_path / "optimized.txt"
    artifact.write_text(
        "Final code:\n```hip\n#include <hip/hip_runtime.h>\n"
        "extern \"C\" void optimized_kernel() {}\n```\n",
        encoding="utf-8",
    )
    report = tmp_path / "optimization_report.md"
    report.write_text(
        "[CORRECTNESS] PASS\n[MICRO_SPEEDUP] 1.25x\n",
        encoding="utf-8",
    )
    verification = ko.build_verification(
        _args(e2e_gain_pct=1.0, accuracy_passed=True),
        [_attempt(report, artifact=artifact)],
        benchmark_available=True,
    )
    assert verification["artifact_valid"] is True
    assert verification["artifact_source"] == "extracted_code_block"
    assert verification["best_artifact_path"].endswith("_extracted.hip")
    assert ko.make_proposal(verification)["decision"] == "KEEP"


def test_complete_kernel_artifact_can_integrate_without_e2e_yet(tmp_path):
    report = tmp_path / "optimization_report.md"
    report.write_text(
        "[CORRECTNESS] PASS\n[MICRO_SPEEDUP] 1.30x\n",
        encoding="utf-8",
    )
    verification = ko.build_verification(
        _args(),
        [_attempt(report)],
        benchmark_available=True,
    )
    proposal = ko.make_proposal(verification)
    assert verification["artifact_valid"] is True
    assert verification["e2e_gain_pct"] is None
    assert verification["accuracy_passed"] is None
    assert proposal["decision"] == "KEEP"
    assert "deferred to integrate" in proposal["reasons"][0]


def test_report_correctness_failure_blocks_keep(tmp_path):
    report = tmp_path / "optimization_report.md"
    report.write_text(
        "Correctness failed: assert_close failed\nSpeedup: 2.0x\n",
        encoding="utf-8",
    )
    verification = ko.build_verification(
        _args(e2e_gain_pct=1.0, accuracy_passed=True),
        [_attempt(report)],
        benchmark_available=True,
    )
    assert verification["correctness_passed"] is False
    assert verification["correctness_source"] == "report_scan"
    assert ko.make_proposal(verification)["decision"] == "NEEDS_REVIEW"


def test_cli_correctness_override(tmp_path):
    artifact = tmp_path / "optimized.hip"
    verification = ko.build_verification(
        _args(correctness_passed=True, micro_speedup=1.25,
              e2e_gain_pct=0.5, accuracy_passed=True),
        [_attempt(artifact=artifact)],
        benchmark_available=False,
    )
    assert verification["correctness_passed"] is True
    assert verification["correctness_source"] == "cli_override"
    assert ko.make_proposal(verification)["decision"] == "KEEP"


def test_benchmark_files_list_counts_as_benchmark(tmp_path):
    bench = tmp_path / "bench.py"
    bench.write_text("print('ok')\n", encoding="utf-8")
    args = _args()
    args.benchmark_file = ""
    args.test_harness_path = ""
    assert ko.has_benchmark(args, {"benchmark_files": [str(bench)]}) is True


def _metadata_from_prompt(prompt: str) -> dict:
    marker = "Kernel runtime metadata"
    start = prompt.index("```json", prompt.index(marker)) + len("```json")
    end = prompt.index("```", start)
    return json.loads(prompt[start:end])


def _prompt_args(target_platform: str):
    args = _args(source_file="", target_platform=target_platform)
    args.kernel_id = "platform_kernel"
    args.num_gpus = 0
    args.budget_minutes = 60
    return args


@pytest.mark.parametrize(
    ("target_platform", "expected_name", "expected_arch", "expected_flag"),
    [
        ("mi300x", "AMD Instinct MI300X", "gfx942", "--offload-arch=gfx942"),
        ("mi325x", "AMD Instinct MI325X", "gfx942", "--offload-arch=gfx942"),
        ("mi355x", "AMD Instinct MI355X", "gfx950", "--offload-arch=gfx950"),
    ],
)
def test_build_prompt_uses_target_platform_hardware_notes(
    target_platform, expected_name, expected_arch, expected_flag,
):
    prompt = ko.build_prompt(
        {"name": "platform_kernel", "source_type": "hip"},
        _prompt_args(target_platform),
    )

    assert expected_name in prompt
    assert expected_arch in prompt
    assert expected_flag in prompt
    assert "DO NOT use gfx950/MI355X-only features" not in prompt
    if target_platform == "mi355x":
        assert "--offload-arch=gfx942" not in prompt


def test_build_prompt_unknown_target_platform_uses_runtime_inspection():
    prompt = ko.build_prompt(
        {"name": "platform_kernel", "source_type": "hip"},
        _prompt_args("future_gpu"),
    )

    assert "query the runtime environment" in prompt
    assert "ROCR_VISIBLE_DEVICES" in prompt
    assert "choose --offload-arch=<arch>" in prompt
    assert "AMD Instinct MI300X (gfx942, CDNA3)" not in prompt


def test_build_prompt_env_fallback_prefers_target_gpu_type(monkeypatch):
    monkeypatch.setenv("TARGET_GPU_TYPE", "mi325x")
    monkeypatch.setenv("GPU_TYPE", "mi300x")
    args = _args(source_file="")
    args.kernel_id = "platform_kernel"
    args.num_gpus = 0
    args.budget_minutes = 60

    prompt = ko.build_prompt(
        {"name": "platform_kernel", "source_type": "hip"},
        args,
    )

    assert "AMD Instinct MI325X" in prompt
    assert "target platform: `mi325x`" in prompt


def test_build_prompt_includes_geak_runtime_metadata():
    args = _args(source_file="")
    args.kernel_id = "k001"
    args.num_gpus = 0
    args.budget_minutes = 60
    candidate = {
        "name": "paged_attention",
        "source_file": "/tmp/paged_attention.py",
        "source_type": "triton",
        "kernel_repo": "/tmp/repo",
        "gpu_pct": 12.5,
        "input_shapes": [{"call_num": 5, "shape": [1, 32, 128]}],
        "output_shapes": [[1, 32, 128]],
        "input_dtypes": ["fp16"],
        "output_dtypes": ["fp16"],
        "framework": "sglang",
        "runtime_args": {"batch_size": 1},
        "runtime_flags": {"decode": True},
        "env_vars": {"SGLANG_USE_TRITON": "1"},
        "kernel_params": {
            "KV_DTYPE": "fp8",
            "BLOCK_SIZE": 16,
            "HEAD_SIZE": 128,
        },
    }

    metadata = _metadata_from_prompt(ko.build_prompt(candidate, args))

    assert metadata["kernel_name"] == "paged_attention"
    assert metadata["kernel_path"] == "/tmp/paged_attention.py"
    assert metadata["backend"] == "sglang"
    assert metadata["input_shapes"] == [{"call_num": 5, "shape": [1, 32, 128]}]
    assert metadata["output_shapes"] == [[1, 32, 128]]
    assert metadata["input_dtypes"] == ["fp16"]
    assert metadata["output_dtypes"] == ["fp16"]
    assert metadata["runtime_args"] == {"batch_size": 1}
    assert metadata["runtime_flags"]["decode"] is True
    assert metadata["env_vars"] == {"SGLANG_USE_TRITON": "1"}
    assert metadata["kernel_params"]["KV_DTYPE"] == "fp8"
    assert metadata["kernel_params"]["BLOCK_SIZE"] == 16
    assert metadata["kernel_params"]["HEAD_SIZE"] == 128


def test_build_prompt_metadata_is_backward_compatible():
    args = _args(source_file="")
    args.kernel_id = "legacy"
    args.num_gpus = 0
    args.budget_minutes = 60
    candidate = {
        "name": "legacy_kernel",
        "source_file": "/tmp/legacy.py",
        "source_type": "python",
        "shapes": [[4, 8]],
        "call_count": 3,
    }

    metadata = _metadata_from_prompt(ko.build_prompt(candidate, args))

    assert metadata["kernel_name"] == "legacy_kernel"
    assert metadata["kernel_path"] == "/tmp/legacy.py"
    assert metadata["input_shapes"] == [{"call_num": 3, "shape": [4, 8]}]
    assert metadata["output_shapes"] == []
    assert metadata["input_dtypes"] == []
    assert metadata["output_dtypes"] == []
    assert metadata["runtime_args"] == {}
    assert metadata["env_vars"] == {}
    assert metadata["kernel_params"] == {
        "BLOCK_SIZE": None,
        "HEAD_SIZE": None,
        "KV_DTYPE": None,
    }


def test_build_prompt_metadata_extracts_extra_sglang_args():
    args = _args(
        source_file="",
        extra_sglang_args=(
            "--kv-cache-dtype fp8 --page-size 16 --attention-backend aiter "
            "--decode-attention-backend aiter --disable-cuda-graph "
            "--cuda-graph-max-bs 128 --num-continuous-decode-steps 4"
        ),
    )
    args.kernel_id = "paged"
    args.num_gpus = 0
    args.budget_minutes = 60
    candidate = {
        "name": "paged_attention",
        "source_file": "/tmp/paged_attention.py",
        "source_type": "triton",
    }

    metadata = _metadata_from_prompt(ko.build_prompt(candidate, args))

    assert metadata["runtime_args"]["kv_cache_dtype"] == "fp8"
    assert metadata["runtime_args"]["page_size"] == 16
    assert metadata["runtime_args"]["cuda_graph_max_bs"] == 128
    assert metadata["runtime_args"]["num_continuous_decode_steps"] == 4
    assert metadata["runtime_flags"]["attention_backend"] == "aiter"
    assert metadata["runtime_flags"]["decode_attention_backend"] == "aiter"
    assert metadata["runtime_flags"]["disable_cuda_graph"] is True
    assert metadata["kernel_params"]["KV_DTYPE"] == "fp8"
    assert metadata["kernel_params"]["BLOCK_SIZE"] == 16


# ============================================================================
# PR-A §4: TraceLens hypothesis block in build_prompt
# ============================================================================
def test_build_hypothesis_block_returns_empty_when_no_prose_fields():
    """Candidates from non-TraceLens-v0.3 paths (raw trace, csv fallback,
    legacy priority_data) lack the prose fields. The block must be a no-op
    in that case so the prompt body is byte-identical to pre-PR."""
    block = ko._build_hypothesis_block(
        {"name": "kernel_no_prose", "source_type": "triton"},
    )
    assert block == ""


def test_build_hypothesis_block_renders_reasoning_and_resolution():
    block = ko._build_hypothesis_block({
        "name": "rms_norm",
        "reasoning_for_slowdown": "Memory-bound kernel saturating HBM bandwidth.",
        "resolution": "Fuse RMSNorm with the following GEMM to amortize loads.",
        "impact_low_ms": 0.0,
        "impact_low_e2e_pct": 0.0,
        "impact_high_ms": 0.0,
        "impact_high_e2e_pct": 0.0,
    })
    assert "## TraceLens Hypothesis [validate before acting]" in block
    assert "Memory-bound kernel saturating HBM bandwidth." in block
    assert "Fuse RMSNorm with the following GEMM" in block
    # Hypothesis framing must always be present so GEAK doesn't take
    # TraceLens's guess as ground truth.
    assert "verify the reasoning" in block
    assert "(hypothesis)" in block
    # Empty impact range must not be rendered.
    assert "Estimated impact range" not in block


def test_build_hypothesis_block_renders_impact_range_when_set():
    block = ko._build_hypothesis_block({
        "name": "fused_moe",
        "reasoning_for_slowdown": "",
        "resolution": "",
        "impact_low_ms": 12.5,
        "impact_low_e2e_pct": 3.2,
        "impact_high_ms": 40.0,
        "impact_high_e2e_pct": 10.4,
    })
    assert "Estimated impact range" in block
    assert "12.50 ms" in block
    assert "3.20% E2E" in block
    assert "40.00 ms" in block
    assert "10.40% E2E" in block
    # Numbers are TraceLens roofline estimates — the framing must say so
    # so GEAK doesn't treat them as measured speedups.
    assert "roofline" in block
    assert "Reasoning for slowdown" not in block
    assert "Recommended direction" not in block


def test_build_hypothesis_block_renders_identification_when_present():
    """The Identification line carries per-rank context + the source
    metrics-file reference (e.g. ``gemm_metrics.json → operations[].
    efficiency.efficiency_percent``). Surfacing it lets GEAK trace any
    hypothesis back to the raw TraceLens data when it needs to
    disagree. Must be labelled distinctly from Reasoning so the agent
    doesn't conflate "what was flagged" with "why it's slow"."""
    block = ko._build_hypothesis_block({
        "name": "rms_norm",
        "identification": (
            "Four `aiter::rmsnorm_quant` operations flagged as memory-bound. "
            "(source: rmsnorm_metrics.json -> operations[].efficiency.efficiency_percent)"
        ),
        "reasoning_for_slowdown": "Memory-bound kernel saturating HBM bandwidth.",
        "resolution": "Fuse RMSNorm with the following GEMM.",
    })
    assert "Identification (TraceLens context):" in block
    assert "Four `aiter::rmsnorm_quant`" in block
    assert "rmsnorm_metrics.json" in block
    # Identification appears BEFORE Reasoning so the agent reads the
    # "what" before the "why" — matches the template's section order.
    id_pos = block.index("Identification (TraceLens context):")
    reason_pos = block.index("Reasoning for slowdown (hypothesis):")
    assert id_pos < reason_pos


def test_build_hypothesis_block_renders_when_only_identification_present():
    """A P-item with only Identification (no Reasoning/Resolution/Impact)
    must still produce a block — GEAK needs the source pointer even
    when the analysis-orchestrator didn't synthesise downstream prose."""
    block = ko._build_hypothesis_block({
        "name": "kernel",
        "identification": "Three ops flagged. (source: gemm_metrics.json)",
    })
    assert block != ""
    assert "Identification (TraceLens context):" in block


def test_build_prompt_omits_hypothesis_block_when_no_prose():
    """Backward compat: candidates without prose fields produce the same
    prompt shape as before — no surprise section, no extra blank lines
    that change downstream token counts."""
    prompt = ko.build_prompt(
        {"name": "legacy_kernel", "source_type": "triton"},
        _prompt_args("mi300x"),
    )
    assert "TraceLens Hypothesis" not in prompt


def test_build_prompt_includes_hypothesis_block_when_prose_present():
    prompt = ko.build_prompt(
        {
            "name": "rms_norm",
            "source_type": "triton",
            "reasoning_for_slowdown": "Memory-bound; HBM bandwidth saturated.",
            "resolution": "Fuse with subsequent GEMM to halve global loads.",
            "impact_low_ms": 5.0,
            "impact_low_e2e_pct": 1.2,
            "impact_high_ms": 20.0,
            "impact_high_e2e_pct": 5.0,
        },
        _prompt_args("mi300x"),
    )
    assert "## TraceLens Hypothesis [validate before acting]" in prompt
    assert "Memory-bound; HBM bandwidth saturated." in prompt
    assert "Fuse with subsequent GEMM" in prompt
    assert "5.00 ms" in prompt
    assert "20.00 ms" in prompt


# ============================================================================
# PR-B §2: benchmark-cases block in build_prompt
# ============================================================================
def test_build_benchmark_cases_block_returns_empty_without_task_group():
    """Legacy per-kernel dispatch (no task_group attached) must produce
    byte-identical output to PR-A."""
    block = ko._build_benchmark_cases_block(
        {"name": "rms_norm", "source_type": "triton"},
    )
    assert block == ""


def test_build_benchmark_cases_block_renders_single_row():
    block = ko._build_benchmark_cases_block({
        "name": "rms_norm",
        "task_group": {
            "function_name": "rms_norm",
            "source_path": "/sgl-workspace/aiter/rmsnorm.py",
            "definition_line": 42,
            "ast_resolved": True,
            "rows": [{
                "name": "rms_norm",
                "shapes": ["(8,4096) bf16"],
                "duration_us": 100_000.0,  # 100 ms aggregate
                "call_count": 100,
                "percent_of_total": 4.2,
                "flops_per_byte": 0.5,
                "bound_type": "memory-bound",
                "efficiency_percent": 30.0,
                "efficiency_peak_value": 5.3,
                "efficiency_peak_unit": "TB/s",
            }],
        },
    })
    assert "## Benchmark cases" in block
    assert "single TraceLens row" in block
    assert "rms_norm" in block
    assert "/sgl-workspace/aiter/rmsnorm.py:42" in block
    assert "Case 1: operation=rms_norm" in block
    # 100 ms / 100 calls = 1.000000 ms per call.
    assert "per_call_ms=1.000000" in block
    assert "bound=memory-bound" in block
    assert "30.00% of 5.3 TB/s" in block


def test_build_benchmark_cases_block_renders_multiple_rows_sorted_by_time():
    """Multi-row groups must explicitly say 'optimize once, applies to all'
    and render rows in aggregate-time-descending order from build_prompt's
    perspective (the test_group_rows arrive pre-sorted from aggregate)."""
    block = ko._build_benchmark_cases_block({
        "name": "rms_norm",
        "task_group": {
            "function_name": "rms_norm",
            "source_path": "/foo/x.py",
            "definition_line": 10,
            "rows": [
                {
                    "name": "rms_norm_prefill",
                    "shapes": ["(64,4096) bf16"],
                    "duration_us": 500_000.0,
                    "call_count": 8,
                    "bound_type": "compute-bound",
                },
                {
                    "name": "rms_norm_decode",
                    "shapes": ["(8,4096) bf16"],
                    "duration_us": 50_000.0,
                    "call_count": 100,
                    "bound_type": "memory-bound",
                },
            ],
        },
    })
    assert "across 2 TraceLens rows" in block
    assert "Optimize the source function once" in block
    case_1_idx = block.index("Case 1: operation=rms_norm_prefill")
    case_2_idx = block.index("Case 2: operation=rms_norm_decode")
    assert case_1_idx < case_2_idx


def test_build_prompt_includes_benchmark_cases_when_task_group_present():
    """End-to-end: build_prompt threads the new block in when the
    candidate carries a task_group."""
    prompt = ko.build_prompt(
        {
            "name": "rms_norm",
            "source_type": "triton",
            "task_group": {
                "function_name": "rms_norm",
                "source_path": "/foo/x.py",
                "definition_line": 10,
                "rows": [{
                    "name": "rms_norm",
                    "shapes": ["(8,4096) bf16"],
                    "duration_us": 100_000.0,
                    "call_count": 100,
                    "bound_type": "memory-bound",
                }],
            },
        },
        _prompt_args("mi300x"),
    )
    assert "## Benchmark cases" in prompt
    assert "operation=rms_norm" in prompt


def test_build_prompt_omits_benchmark_cases_for_legacy_candidates():
    prompt = ko.build_prompt(
        {"name": "legacy_kernel", "source_type": "triton"},
        _prompt_args("mi300x"),
    )
    assert "## Benchmark cases" not in prompt


# ============================================================================
# PR-B §3: bound-keyed optimization priority block in build_prompt
# ============================================================================
def test_build_priority_block_empty_when_no_bound_info():
    block = ko._build_priority_block({"name": "kernel", "source_type": "triton"})
    assert block == ""


def test_build_priority_block_memory_bound_leads_with_memory_traffic():
    block = ko._build_priority_block({
        "name": "rms_norm",
        "bound_type": "memory-bound",
    })
    assert "Optimization priorities" in block
    assert "memory-bound" in block
    # Lever 1 must be memory traffic; lever 2 must be shape-aware.
    lev1 = block.index("1. **Memory traffic reduction**")
    lev2 = block.index("2. **Shape-aware tuning**")
    assert lev1 < lev2


def test_build_priority_block_compute_bound_leads_with_compute_utilization():
    block = ko._build_priority_block({
        "name": "gemm_kernel",
        "bound_type": "compute-bound",
    })
    assert "1. **Compute utilization**" in block
    assert "primary lever for compute-bound" in block


def test_build_priority_block_unknown_bound_uses_default_order():
    block = ko._build_priority_block({
        "name": "kernel",
        "bound_type": "mixed",
    })
    # mixed → unknown bucket → structural simplification first.
    assert "1. **Structural simplification**" in block


def test_build_priority_block_reads_bound_from_task_group_primary_row():
    """When candidate has no top-level bound_type, fall back to the
    first task_group row's bound_type."""
    block = ko._build_priority_block({
        "name": "rms_norm",
        "task_group": {
            "rows": [{"name": "rms_norm", "bound_type": "memory-bound"}],
        },
    })
    assert "1. **Memory traffic reduction**" in block


def test_build_prompt_includes_priority_block_when_bound_present():
    prompt = ko.build_prompt(
        {"name": "gemm", "source_type": "triton", "bound_type": "compute-bound"},
        _prompt_args("mi300x"),
    )
    assert "## Optimization priorities" in prompt
    assert "1. **Compute utilization**" in prompt


def test_build_prompt_omits_priority_block_for_legacy_candidates():
    prompt = ko.build_prompt(
        {"name": "legacy", "source_type": "triton"},
        _prompt_args("mi300x"),
    )
    assert "## Optimization priorities" not in prompt
