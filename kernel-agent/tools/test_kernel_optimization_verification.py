from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

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
