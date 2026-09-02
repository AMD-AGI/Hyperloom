# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path

import pytest

_TOOLS_DIR = Path(__file__).resolve().parent.parent / "tools"
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

import kernel_optimization as ko  # noqa: E402


def _args(**overrides):
    base = {
        "backends": "",
        "benchmark_file": "",
        "micro_speedup": None,
        "e2e_gain_pct": None,
        "accuracy_passed": None,
        "correctness_passed": None,
        "dry_run": False,
        "source_file": "/tmp/source.hip",
        "extra_server_args": "",
        "kernel_id": "k001",
    }
    base.update(overrides)
    return Namespace(**base)


def _candidate(**overrides):
    base = {
        "kernel_id": "k001",
        "name": "kernel",
        "source_type": "triton",
        "source_file": "/tmp/kernel.py",
    }
    base.update(overrides)
    return base


def test_choose_backends_default_does_not_select_forge(monkeypatch):
    monkeypatch.delenv("KERNEL_OPT_BACKEND_ORDER", raising=False)

    selected, notes = ko.choose_backends(_args(), _candidate())

    assert selected == []
    assert notes["user_specified_backends"] is False


def test_choose_backends_geak_env_does_not_select_per_kernel_backend(monkeypatch):
    monkeypatch.setenv("KERNEL_OPT_BACKEND_ORDER", "geak")

    selected, notes = ko.choose_backends(_args(), _candidate())

    assert selected == []
    assert notes["user_specified_backends"] is False


def test_choose_backends_forge_env_is_explicit_opt_in(monkeypatch):
    monkeypatch.setenv("KERNEL_OPT_BACKEND_ORDER", "forge")

    selected, notes = ko.choose_backends(_args(), _candidate())

    assert selected == ["forge"]
    assert notes["user_specified_backends"] is True


def test_choose_backends_forge_cli_does_not_enable_without_env(monkeypatch):
    monkeypatch.delenv("KERNEL_OPT_BACKEND_ORDER", raising=False)

    selected, notes = ko.choose_backends(_args(backends="forge"), _candidate())

    assert selected == []
    assert notes["user_specified_backends"] is False


def _attempt(
    report: Path | None = None,
    artifact: Path | None = None,
    *,
    backend: str = "generic",
):
    paths = {}
    if report is not None:
        paths["report"] = str(report)
        if artifact is None:
            artifact = report.parent / "optimized.hip"
    if artifact is not None:
        if not artifact.exists():
            artifact.write_text(
                '#include <hip/hip_runtime.h>\nextern "C" void optimized_kernel() {}\n',
                encoding="utf-8",
            )
        paths["partial_latest_optimized"] = str(artifact)
    return {
        "status": "completed",
        "attempt_id": "a1",
        "backend": backend,
        "optimized_path": str(artifact or "/tmp/optimized.hip"),
        "backend_paths": paths,
    }


def test_structured_shape_cases_parse_moe_args():
    shape = "(15360,8,768) bf16<br>(128,1536,2048) bf16<br>(139256,) int<br>(139256,) fp32"
    candidate = {
        "name": "aiter::ck_moe_stage2",
        "input_shapes": [{"call_num": 48, "shape": shape}],
    }

    cases = ko._structured_benchmark_shape_cases(candidate)

    primary = cases["primary_shape"]
    assert primary["call_count"] == 48
    assert primary["args"][0] == {
        "index": 0,
        "raw": "(15360,8,768) bf16",
        "shape": [15360, 8, 768],
        "dtype": "bf16",
    }
    assert primary["args"][2]["shape"] == [139256]
    assert primary["args"][2]["dtype"] == "int"
    assert primary["args"][3]["dtype"] == "fp32"
    assert cases["supplementary_shapes"] == []


def test_captured_shapes_block_claims_measurement_only_for_a_measurement():
    """Measured shapes are identified and passed through without weakening the instruction."""
    measured = ko._build_captured_shapes_block({"shapes": ["(8192,6144) bf16"], "shape_provenance": "torch_trace"})
    assert "TraceLens-captured" in measured
    assert "do NOT invent" in measured


def test_captured_shapes_block_is_empty_without_dims():
    """No block means the backend picks its own shapes -- the state this whole
    path exists to get out of.
    """
    assert ko._build_captured_shapes_block({"shapes": []}) == ""


def test_structured_shape_cases_prefer_input_shapes():
    candidate = {
        "name": "aiter::ck_moe_stage2",
        "input_shapes": [
            {"call_num": 7, "shape": "(1,2,3) bf16<BR/>(4,) int"},
        ],
        # Prose field must not win over input_shapes.
        "shapes": [{"call_num": 99, "shape": "(999,) fp32"}],
    }

    cases = ko._structured_benchmark_shape_cases(candidate)

    primary = cases["primary_shape"]
    assert primary["source"] == "input_shapes"
    assert primary["call_count"] == 7
    assert primary["args"][0]["shape"] == [1, 2, 3]
    assert primary["args"][0]["dtype"] == "bf16"
    assert primary["args"][1]["shape"] == [4]
    assert primary["args"][1]["dtype"] == "int"


def test_structured_shape_cases_tolerates_bad_task_group_duration():
    candidate = {
        "task_group": {
            "rows": [
                {
                    "name": "op",
                    "shapes": ["(8,16) fp32"],
                    "call_count": 3,
                    "duration_us": "N/A",
                }
            ],
        },
    }

    cases = ko._structured_benchmark_shape_cases(candidate)

    primary = cases["primary_shape"]
    assert primary["source"] == "task_group"
    assert primary["aggregate_time_ms"] == 0.0
    assert primary["args"][0]["shape"] == [8, 16]


def test_structured_shape_cases_include_task_group_supplementary_shapes():
    candidate = {
        "input_shapes": [
            {"call_num": 5, "shape": "(1,2) fp16"},
        ],
        "task_group": {
            "rows": [
                {
                    "name": "op",
                    "shapes": ["(1,2) fp16"],
                    "call_count": 5,
                    "duration_us": 1000,
                },
                {
                    "name": "op",
                    "shapes": ["(4,8) bf16"],
                    "call_count": 3,
                    "duration_us": 500,
                },
            ],
        },
    }

    cases = ko._structured_benchmark_shape_cases(candidate)

    assert cases["primary_shape"]["args"][0]["shape"] == [1, 2]
    supplementary = cases["supplementary_shapes"]
    assert len(supplementary) == 1
    assert supplementary[0]["source"] == "task_group"
    assert supplementary[0]["args"][0]["shape"] == [4, 8]
    assert supplementary[0]["args"][0]["dtype"] == "bf16"


def test_structured_shape_cases_falls_back_to_input_shapes_when_group_rows_empty():
    candidate = {
        "input_shapes": [
            {"call_num": 5, "shape": "(1,2) fp16"},
        ],
        "task_group": {
            "rows": [
                {"name": "op", "shapes": [], "call_count": 5},
            ],
        },
    }

    cases = ko._structured_benchmark_shape_cases(candidate)

    primary = cases["primary_shape"]
    assert primary["source"] == "input_shapes"
    assert primary["call_count"] == 5
    assert primary["args"][0]["shape"] == [1, 2]


def test_build_prompt_includes_structured_shape_contract():
    """The runtime-metadata block promotes ``input_shapes`` into a shape contract.

    Rendered through the full prompt: the metadata block is GEAK's, and forge is
    handed the same operands through its invocation spec instead.
    """
    shape = "(15360,8,768) bf16<br>(128,1536,2048) bf16"
    candidate = {
        "name": "aiter::ck_moe_stage2",
        "source_file": "/tmp/gemm_moe_ck2stages.cu",
        "source_type": "hip_cpp",
        "kernel_repo": "/tmp/aiter",
        "gpu_pct": 24.3,
        "input_shapes": [{"call_num": 48, "shape": shape}],
    }

    prompt = ko.build_prompt(candidate, _args())

    assert "when `benchmark_shape_cases` is present" in prompt
    assert '"benchmark_shape_cases"' in prompt
    assert '"primary_shape"' in prompt
    metadata_json = prompt.split("```json\n", 1)[1].split("\n```", 1)[0]
    metadata = json.loads(metadata_json)
    primary = metadata["benchmark_shape_cases"]["primary_shape"]
    assert primary["args"][0]["shape"] == [15360, 8, 768]
    assert primary["args"][1]["shape"] == [128, 1536, 2048]


def test_build_prompt_omits_structured_shape_cases_without_program_output():
    candidate = {
        "name": "aiter::ck_moe_stage2",
        "source_file": "/tmp/gemm_moe_ck2stages.cu",
        "source_type": "hip_cpp",
        "kernel_repo": "/tmp/aiter",
        # Legacy prose-only shapes remain available through the old Shapes:
        # line and Benchmark shapes block, but they do not create the new
        # structured metadata.
        "shapes": [{"call_num": 48, "shape": "(15360,8,768) bf16"}],
    }

    prompt = ko.build_prompt(candidate, _args())
    metadata_json = prompt.split("```json\n", 1)[1].split("\n```", 1)[0]
    metadata = json.loads(metadata_json)

    assert "benchmark_shape_cases" not in metadata
    assert "Shapes:" in prompt
    assert "when `benchmark_shape_cases` is present" not in prompt


def test_build_prompt_forge_keeps_the_target_arch_and_the_harness_paths(tmp_path):
    """Three blocks left the forge shape by omission, not by design.

    The comment above ``forge_sections`` lists what it removes and why, and none
    of these were on it. ``hardware_notes`` carries the ROCm arch, so without it
    an agent can emit a gfx942 intrinsic for a gfx950 host and the rewrite does
    not compile; ``bench_block`` names the harnesses the trace resolved.
    """
    bench = tmp_path / "test_gemm.py"
    bench.write_text("def test_gemm(): pass\n", encoding="utf-8")
    candidate = {
        "name": "gemm_afp4wfp4",
        "source_file": "/tmp/gemm_afp4wfp4.py",
        "source_type": "python",
        "benchmark_files": [str(bench)],
        "target_platform": "mi355x",
    }

    prompt = ko.build_prompt(candidate, _args(), backend="forge")

    assert "Hardware notes" in prompt
    assert str(bench) in prompt
    # Still without the sections that fight forge's own workspace guard.
    assert "optimization_report.md" not in prompt
    assert "optimized_versions/" not in prompt


def test_build_prompt_forge_falls_back_to_the_full_analysis(tmp_path):
    """The forge shape keeps "the trace evidence forge cannot derive".

    The hypothesis block is built from TraceLens p-items, so a kernel the trace
    ranked without one renders it empty and ``analysis.md`` is the only place
    that evidence exists. Building the fallback after the forge early return
    dropped it for exactly those rows -- the ones with the least other context.
    """
    report = tmp_path / "analysis.md"
    report.write_text("# Analysis\n\nMemory bound on the sparse path.\n", encoding="utf-8")
    candidate = _candidate(trace_report_path=str(report))
    candidate.pop("tracelens_hypothesis", None)

    forge = ko.build_prompt(candidate, _args(), backend="forge")
    geak = ko.build_prompt(candidate, _args())

    assert "Memory bound on the sparse path." in forge
    assert "TraceLens Context" in forge
    assert "Memory bound on the sparse path." in geak


def test_build_prompt_forge_multinode_states_the_constraint_without_the_geak_recipe(monkeypatch):
    """The GPU-less constraint is shared; the procedure under it is not.

    The multi-node block routes measurements through ``kernel-bench`` and names
    ``optimized_versions/`` and ``optimization_report.md`` -- the two paths
    forge's workspace guard refuses, and the reason the deliverable contract was
    dropped. Handing forge the whole block put them straight back, so a
    multi-node forge run lost its first iteration exactly as before.
    """
    monkeypatch.setenv("INFERENCE_OPTIMIZER_NODES", "2")
    prompt = ko.build_prompt(_candidate(), _args(), backend="forge")

    assert "this node has no GPU" in prompt
    assert "optimization_report.md" not in prompt
    assert "optimized_versions/" not in prompt
    assert "kernel-bench" not in prompt


def test_build_prompt_geak_multinode_keeps_its_dispatch_recipe(monkeypatch):
    """GEAK brings no benchmark of its own, so it still needs the procedure."""
    monkeypatch.setenv("INFERENCE_OPTIMIZER_NODES", "2")
    prompt = ko.build_prompt(_candidate(), _args())

    assert "kernel-bench" in prompt
    assert "optimization_report.md" in prompt


def test_build_prompt_forge_drops_the_geak_harness_and_keeps_trace_evidence():
    """Forge is handed trace evidence only, not the harness it does not run.

    The dropped sections are not merely redundant there. The deliverable files
    land as new untracked paths that forge's workspace guard refuses, costing the
    iteration that wrote them, and the A/B recipes stand up a second benchmark
    beside the driver its own gate scores. Asserting their absence is what keeps
    a later edit from reintroducing either through the shared prompt.
    """
    candidate = {
        "name": "aiter::ck_moe_stage2",
        "source_file": "/tmp/gemm_moe_ck2stages.cu",
        "source_type": "hip_cpp",
        "kernel_repo": "/tmp/aiter",
        "gpu_pct": 24.3,
        "device_kernel_name": "ck_moe_stage2_kernel",
        "source_resolution_method": "op_to_source",
        "input_shapes": [{"call_num": 48, "shape": "(15360,8,768) bf16"}],
    }

    forge = ko.build_prompt(candidate, _args(), backend="forge")
    full = ko.build_prompt(candidate, _args())

    # Absent from forge, still present for the backend that needs them.
    for token in (
        "optimization_report.md",
        "optimized_versions/",
        "mini-swe-agent step",
        "cpp_extension.load",
        "structured context for GEAK",
        "IMPORTANT — sandbox rules",
    ):
        assert token not in forge, token
        assert token in full, token

    # Kept: what forge cannot derive from the kernel or its invocation spec.
    assert "DEVICE KERNEL FOCUS" in forge
    assert "ck_moe_stage2_kernel" in forge
    assert "Preserve function name" in forge
    assert forge.startswith("# TASK: Optimize the `aiter::ck_moe_stage2` kernel")
    # A skipped section must not leave a run of blank lines behind.
    assert "\n\n\n" not in forge


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
    """Report-scan correctness lights up on its own (no `accuracy_passed`)."""
    report = tmp_path / "optimization_report.md"
    report.write_text(
        "Correctness passed\nSpeedup: 1.32x\n",
        encoding="utf-8",
    )
    verification = ko.build_verification(
        _args(e2e_gain_pct=1.0),
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
        _args(e2e_gain_pct=1.0),
        [_attempt(report)],
        benchmark_available=True,
    )
    assert verification["correctness_passed"] is True
    assert verification["correctness_source"] == "report_scan"
    assert verification["micro_speedup"] == 1.28


def test_forge_policy_uses_total_pristine_improvement_not_incremental(tmp_path):
    report = tmp_path / "optimization_report.md"
    report.write_text("[CORRECTNESS] PASS\n", encoding="utf-8")
    attempt = _attempt(report, backend="forge")
    attempt.update(
        {
            "pristine_baseline_ms": 1.0,
            "search_start_ms": 0.8,
            "best_ms": 1.2,
            "mean_case_speedup": 1.25,
            "search_start_mean_case_speedup": 1.25,
            "total_improved": True,
            "incremental_improved": False,
            "improved": True,
            "improved_during_search": False,
        }
    )

    verification = ko.build_verification(
        _args(),
        [attempt],
        benchmark_available=True,
    )

    assert verification["micro_speedup"] == 1.25
    assert verification["micro_speedup_source"] == "forge_mean_case_result"
    assert ko.make_proposal(verification)["decision"] == "KEEP"


def test_forge_rejects_ambiguous_report_without_structured_score(tmp_path):
    report = tmp_path / "optimization_report.md"
    report.write_text(
        "[CORRECTNESS] PASS\n[MICRO_SPEEDUP] 1.30x\n",
        encoding="utf-8",
    )

    verification = ko.build_verification(
        _args(),
        [_attempt(report, backend="forge")],
        benchmark_available=True,
    )

    assert verification["micro_speedup"] == 1.0
    assert verification["micro_speedup_source"] == "default_unmeasured"
    assert ko.make_proposal(verification)["decision"] == "PARTIAL"


def test_non_forge_preserves_structured_timing_fallback(tmp_path):
    report = tmp_path / "optimization_report.md"
    report.write_text("[CORRECTNESS] PASS\n", encoding="utf-8")
    attempt = _attempt(report, backend="generic")
    attempt.update(
        {
            "pristine_baseline_ms": 2.0,
            "best_ms": 1.0,
            "improved": True,
        }
    )

    verification = ko.build_verification(
        _args(),
        [attempt],
        benchmark_available=True,
    )

    assert verification["micro_speedup"] == 2.0
    assert verification["micro_speedup_source"] == "structured_timing_result"


def test_report_correctness_passes_with_reference_language(tmp_path):
    report = tmp_path / "optimization_report.md"
    report.write_text(
        "The optimized implementation matches reference outputs for all test shapes.\nSpeedup: 1.41x\n",
        encoding="utf-8",
    )
    verification = ko.build_verification(
        _args(e2e_gain_pct=1.0),
        [_attempt(report)],
        benchmark_available=True,
    )
    assert verification["correctness_passed"] is True
    assert verification["correctness_source"] == "report_scan"


def test_extracts_complete_source_from_text_artifact(tmp_path):
    artifact = tmp_path / "optimized.txt"
    artifact.write_text(
        'Final code:\n```hip\n#include <hip/hip_runtime.h>\nextern "C" void optimized_kernel() {}\n```\n',
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
    """Explicit "Correctness failed" in the report must block KEEP (no `accuracy_passed`)."""
    report = tmp_path / "optimization_report.md"
    report.write_text(
        "Correctness failed: assert_close failed\nSpeedup: 2.0x\n",
        encoding="utf-8",
    )
    verification = ko.build_verification(
        _args(e2e_gain_pct=1.0),
        [_attempt(report)],
        benchmark_available=True,
    )
    assert verification["correctness_passed"] is False
    assert verification["correctness_source"] == "report_scan"
    assert ko.make_proposal(verification)["decision"] == "NEEDS_REVIEW"


def test_cli_correctness_override(tmp_path):
    artifact = tmp_path / "optimized.hip"
    verification = ko.build_verification(
        _args(correctness_passed=True, micro_speedup=1.25, e2e_gain_pct=0.5, accuracy_passed=True),
        [_attempt(artifact=artifact)],
        benchmark_available=False,
    )
    assert verification["correctness_passed"] is True
    assert verification["correctness_source"] == "cli_override"
    assert ko.make_proposal(verification)["decision"] == "KEEP"


def test_speedup_just_above_gate_keeps(tmp_path):
    """A 1.12x speedup clears the 1.10x KEEP gate."""
    artifact = tmp_path / "optimized.hip"
    verification = ko.build_verification(
        _args(correctness_passed=True, micro_speedup=1.12, e2e_gain_pct=0.5, accuracy_passed=True),
        [_attempt(artifact=artifact)],
        benchmark_available=False,
    )
    assert ko.make_proposal(verification)["decision"] == "KEEP"


def test_speedup_at_gate_boundary_keeps(tmp_path):
    artifact = tmp_path / "optimized.hip"
    verification = ko.build_verification(
        _args(
            correctness_passed=True,
            micro_speedup=1.10,
            e2e_gain_pct=0.5,
            accuracy_passed=True,
        ),
        [_attempt(artifact=artifact)],
        benchmark_available=False,
    )

    assert ko.make_proposal(verification)["decision"] == "KEEP"


def test_speedup_below_gate_needs_review(tmp_path):
    """A 1.07x speedup (improvement but under the 1.10x gate) routes to NEEDS_REVIEW, not KEEP."""
    artifact = tmp_path / "optimized.hip"
    verification = ko.build_verification(
        _args(correctness_passed=True, micro_speedup=1.07, e2e_gain_pct=0.5, accuracy_passed=True),
        [_attempt(artifact=artifact)],
        benchmark_available=False,
    )
    proposal = ko.make_proposal(verification)
    assert proposal["decision"] == "NEEDS_REVIEW"
    assert any("below KEEP" in r for r in proposal["reasons"])


def test_benchmark_files_list_counts_as_benchmark(tmp_path):
    bench = tmp_path / "bench.py"
    bench.write_text("print('ok')\n", encoding="utf-8")
    args = _args()
    args.benchmark_file = ""
    assert ko.has_benchmark(args, {"benchmark_files": [str(bench)]}) is True


# Backend stdout must never be promoted to a `source_file` artifact; only the
# fenced-block extraction path may surface it.


def test_geak_stdout_log_must_not_false_positive_as_source_file(tmp_path):
    """Stdout-log-only artifact (no patch, no .cu, no code fence) → artifact_source == "missing"."""
    log_path = tmp_path / "geak-deadbeef_stdout.log"
    log_path.write_text(
        "minisweagent.agents.parallel_agent: INFO: [running 12.0min] Sub-agents working\n"
        "2 total patches (task_0: 2)\n"
        # Embeds void/int/float markers that must not be accepted as CUDA source.
        "Trajectory note: convert the int loop, drop the void wrapper, use float.\n"
        "Best patch: patch_1 (agent 0)\n",
        encoding="utf-8",
    )
    attempt = {
        "status": "completed",
        "attempt_id": "geak-deadbeef",
        "backend": "forge",
        "optimized_path": str(log_path),
        "backend_paths": {},
    }

    artifact_path, source, error = ko._select_source_artifact(
        attempt,
        target_file="/tmp/source.cu",
        run_dir=tmp_path,
    )

    assert artifact_path == ""
    assert source == "missing"
    assert ".cu source artifact found" in error


def test_geak_stdout_log_with_fenced_cuda_block_is_extracted(tmp_path):
    """Fenced CU in stdout log is surfaced via the `.log` route, labelled ``extracted_code_block`` (not ``source_file``)."""
    log_path = tmp_path / "forge-c0ffee_stdout.log"
    log_path.write_text(
        "Here is the final optimized kernel:\n"
        "```cuda\n"
        "#include <hip/hip_runtime.h>\n"
        'extern "C" __global__ void optimized_kernel(float* out, const float* in) {\n'
        "  int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
        "  out[idx] = in[idx] * 2.0f;\n"
        "}\n"
        "```\n"
        "Done.\n",
        encoding="utf-8",
    )
    attempt = {
        "status": "completed",
        "attempt_id": "forge-c0ffee",
        "backend": "forge",
        "optimized_path": str(log_path),
        "backend_paths": {},
    }

    artifact_path, source, error = ko._select_source_artifact(
        attempt,
        target_file="/tmp/source.cu",
        run_dir=tmp_path,
    )

    assert error == ""
    assert source == "extracted_code_block"
    assert artifact_path.endswith("_extracted.cu")
    body = Path(artifact_path).read_text(encoding="utf-8")
    assert 'extern "C" __global__ void optimized_kernel' in body
    assert "```" not in body
    assert "Here is the final" not in body


def test_geak_patch_is_preferred_over_stdout_log(tmp_path):
    """Patch wins over stdout log; with no readable original, a bare diff yields `missing` rather than promoting the marker-noise log."""
    patch_path = tmp_path / "patch_1.patch"
    patch_path.write_text(
        "--- a/kernel.cu\n+++ b/kernel.cu\n@@ -1,1 +1,1 @@\n-old\n+new\n",
        encoding="utf-8",
    )
    log_path = tmp_path / "geak-cafebabe_stdout.log"
    log_path.write_text(
        "void int float extern __global__ #include\n",  # marker-rich noise
        encoding="utf-8",
    )
    attempt = {
        "status": "completed",
        "attempt_id": "geak-cafebabe",
        "backend": "forge",
        "optimized_path": str(log_path),
        "backend_paths": {
            "partial_latest_optimized": str(patch_path),
        },
    }

    artifact_path, source, error = ko._select_source_artifact(
        attempt,
        target_file="/tmp/source.cu",
        run_dir=tmp_path,
    )

    # The marker-noise log is NOT silently promoted to source_file.
    assert source == "missing"
    assert artifact_path == ""


def test_patch_only_winner_reconstructs_full_source(tmp_path):
    """A backend whose best artifact is a unified diff (no full-source .py, no
    fenced block) must reconstruct the complete optimized source by applying the
    patch to the original kernel — not defer with artifact_source='missing'.
    """
    original = tmp_path / "fused_moe.py"
    original.write_text(
        "import triton\n\n\ndef helper():\n    return 1\n\n\ndef kernel():\n    return helper()\n",
        encoding="utf-8",
    )
    # Unified diff that adds a cache helper.
    patch_path = tmp_path / "asm-kernel-rewrite" / "patch_0.patch"
    patch_path.parent.mkdir(parents=True)
    patch_path.write_text(
        "--- a/fused_moe.py\n"
        "+++ b/fused_moe.py\n"
        "@@ -1,5 +1,8 @@\n"
        " import triton\n"
        " \n"
        " \n"
        "+_CACHE = {}\n"
        "+\n"
        "+\n"
        " def helper():\n"
        "     return 1\n",
        encoding="utf-8",
    )
    attempt = {
        "status": "completed",
        "attempt_id": "geak-asm",
        "backend": "forge",
        "backend_paths": {"partial_latest_optimized": str(patch_path)},
    }

    artifact_path, source, error = ko._select_source_artifact(
        attempt,
        target_file=str(original),
        run_dir=tmp_path,
    )

    assert source == "reconstructed_from_patch", error
    assert artifact_path
    text = Path(artifact_path).read_text(encoding="utf-8")
    # Reconstruction = original + patch (complete file, not a diff).
    assert "_CACHE = {}" in text
    assert "def kernel():" in text  # untouched original content preserved
    assert "@@" not in text  # not a diff
    # Nothing was written outside run_dir.
    assert Path(artifact_path).resolve().is_relative_to(tmp_path.resolve())


def test_patch_with_absolute_path_header_is_rejected(tmp_path):
    """A backend diff whose header targets an ABSOLUTE path must NOT reconstruct
    (and must not write outside the work dir). Backend patches are untrusted."""
    original = tmp_path / "fused_moe.py"
    original.write_text("import triton\n\n\ndef kernel():\n    return 1\n", encoding="utf-8")
    evil = tmp_path / "evil.patch"
    evil.write_text(
        "--- a/fused_moe.py\n+++ /etc/cron.d/pwned\n@@ -1,1 +1,2 @@\n import triton\n+# owned\n",
        encoding="utf-8",
    )
    attempt = {
        "status": "completed",
        "attempt_id": "geak-evil",
        "backend": "forge",
        "backend_paths": {"partial_latest_optimized": str(evil)},
    }
    artifact_path, source, _ = ko._select_source_artifact(
        attempt,
        target_file=str(original),
        run_dir=tmp_path,
    )
    assert source == "missing"
    assert artifact_path == ""


def test_patch_with_parent_traversal_header_is_rejected(tmp_path):
    """A backend diff header using ``..`` traversal must NOT reconstruct."""
    original = tmp_path / "fused_moe.py"
    original.write_text("import triton\n\n\ndef kernel():\n    return 1\n", encoding="utf-8")
    evil = tmp_path / "evil.patch"
    evil.write_text(
        "--- a/fused_moe.py\n+++ b/../../../../tmp/pwned.py\n@@ -1,1 +1,2 @@\n import triton\n+# owned\n",
        encoding="utf-8",
    )
    attempt = {
        "status": "completed",
        "attempt_id": "geak-evil2",
        "backend": "forge",
        "backend_paths": {"partial_latest_optimized": str(evil)},
    }
    artifact_path, source, _ = ko._select_source_artifact(
        attempt,
        target_file=str(original),
        run_dir=tmp_path,
    )
    assert source == "missing"
    assert artifact_path == ""


def test_patch_targeting_other_basename_is_rejected(tmp_path):
    """A diff that edits a different file than the target kernel is rejected."""
    original = tmp_path / "fused_moe.py"
    original.write_text("import triton\n\n\ndef kernel():\n    return 1\n", encoding="utf-8")
    other = tmp_path / "other.patch"
    other.write_text(
        "--- a/setup.py\n+++ b/setup.py\n@@ -1,1 +1,2 @@\n import triton\n+# x\n",
        encoding="utf-8",
    )
    attempt = {
        "status": "completed",
        "attempt_id": "geak-other",
        "backend": "forge",
        "backend_paths": {"partial_latest_optimized": str(other)},
    }
    artifact_path, source, _ = ko._select_source_artifact(
        attempt,
        target_file=str(original),
        run_dir=tmp_path,
    )
    assert source == "missing"
    assert artifact_path == ""


_RECON_ORIGINAL = "import triton\n\n\ndef helper():\n    return 1\n\n\ndef kernel():\n    return helper()\n"
_RECON_KERNEL_SECTION = (
    "diff --git a/fused_moe.py b/fused_moe.py\n"
    "--- a/fused_moe.py\n"
    "+++ b/fused_moe.py\n"
    "@@ -1,5 +1,8 @@\n"
    " import triton\n"
    " \n"
    " \n"
    "+_CACHE = {}\n"
    "+\n"
    "+\n"
    " def helper():\n"
    "     return 1\n"
)


def _reconstruct(tmp_path, patch_text, *, original_name="fused_moe.py", original_text=_RECON_ORIGINAL):
    """Run reconstruction for ``patch_text`` against a written original kernel."""
    original = tmp_path / original_name
    original.parent.mkdir(parents=True, exist_ok=True)
    original.write_text(original_text, encoding="utf-8")
    patch_path = tmp_path / "winner.patch"
    patch_path.write_text(patch_text, encoding="utf-8")
    out = tmp_path / "reconstructed.py"
    return ko._reconstruct_source_from_patch(patch_path, str(original), out)


def test_reconstruct_git_style_diff_with_index_header(tmp_path):
    """A real-shape ``diff --git`` + ``index`` multi-component path reconstructs
    via the contained ``git apply`` path (not silently falling back)."""
    patch = (
        "diff --git a/aiter/ops/fused_moe.py b/aiter/ops/fused_moe.py\n"
        "index 1d75520b5..0159e0442 100644\n"
        "--- a/aiter/ops/fused_moe.py\n"
        "+++ b/aiter/ops/fused_moe.py\n"
        "@@ -1,5 +1,8 @@\n"
        " import triton\n"
        " \n"
        " \n"
        "+_CACHE = {}\n"
        "+\n"
        "+\n"
        " def helper():\n"
        "     return 1\n"
    )
    out = _reconstruct(tmp_path, patch, original_name="aiter/ops/fused_moe.py")
    assert out
    text = Path(out).read_text(encoding="utf-8")
    assert "_CACHE = {}" in text and "def kernel():" in text and "@@" not in text


def test_reconstruct_multi_file_patch_slices_matched_target(tmp_path):
    """Multi-file patch where only the kernel original exists: the kernel section
    is sliced and applied (a whole-patch apply would fail on the absent files)."""
    patch = (
        "diff --git a/bench.py b/bench.py\n--- a/bench.py\n+++ b/bench.py\n"
        "@@ -1 +1 @@\n-old\n+new\n"
        + _RECON_KERNEL_SECTION
        + "diff --git a/other.py b/other.py\n--- a/other.py\n+++ b/other.py\n"
        "@@ -1 +1 @@\n-a\n+b\n"
    )
    out = _reconstruct(tmp_path, patch)
    assert out
    assert "_CACHE = {}" in Path(out).read_text(encoding="utf-8")


def test_reconstruct_prefers_real_target_over_orig_sibling(tmp_path):
    """A ``.orig`` sibling with the same stem must not be chosen over the real
    kernel file."""
    patch = (
        "diff --git a/fused_moe.py.orig b/fused_moe.py.orig\n"
        "--- a/fused_moe.py.orig\n+++ b/fused_moe.py.orig\n"
        "@@ -1 +1 @@\n-junk\n+junk2\n" + _RECON_KERNEL_SECTION
    )
    out = _reconstruct(tmp_path, patch)
    assert out
    assert "_CACHE = {}" in Path(out).read_text(encoding="utf-8")


def test_reconstruct_ignores_companion_new_file_dev_null(tmp_path):
    """A ``/dev/null`` new-file companion entry is ignored, not fatal, as long as
    the matched kernel section has real hunks."""
    patch = (
        "diff --git a/new_helper.py b/new_helper.py\n"
        "new file mode 100644\n--- /dev/null\n+++ b/new_helper.py\n"
        "@@ -0,0 +1 @@\n+HELPER = 1\n" + _RECON_KERNEL_SECTION
    )
    out = _reconstruct(tmp_path, patch)
    assert out
    assert "_CACHE = {}" in Path(out).read_text(encoding="utf-8")


def test_reconstruct_rejects_empty_patch(tmp_path):
    """An empty / 0-byte patch reconstructs nothing."""
    assert _reconstruct(tmp_path, "") == ""
    assert _reconstruct(tmp_path, "   \n  \n") == ""


def test_reconstruct_rejects_hunkless_patch(tmp_path):
    """Headers present but no ``@@`` hunk (rename/mode-only shape) → nothing."""
    patch = (
        "diff --git a/fused_moe.py b/fused_moe.py\n"
        "old mode 100644\nnew mode 100755\n"
        "--- a/fused_moe.py\n+++ b/fused_moe.py\n"
    )
    assert _reconstruct(tmp_path, patch) == ""


def test_reconstruct_rejects_absolute_path_patch(tmp_path):
    """An absolute-path diff header is refused (no write outside)."""
    victim = tmp_path / "victim.py"
    victim.write_text("safe\n", encoding="utf-8")
    patch = f"--- a/fused_moe.py\n+++ {victim}\n@@ -1 +1 @@\n-safe\n+PWNED\n"
    assert _reconstruct(tmp_path, patch) == ""
    assert victim.read_text(encoding="utf-8") == "safe\n"


def test_reconstruct_rejects_parent_traversal_patch(tmp_path):
    """A ``..`` traversal header is refused and writes nothing outside the dir."""
    outside = tmp_path / "outside.py"
    outside.write_text("safe\n", encoding="utf-8")
    work = tmp_path / "work"
    work.mkdir()
    original = work / "fused_moe.py"
    original.write_text(_RECON_ORIGINAL, encoding="utf-8")
    patch_path = work / "winner.patch"
    patch_path.write_text(
        "--- a/fused_moe.py\n+++ b/../outside.py\n@@ -1 +1 @@\n-safe\n+PWNED\n",
        encoding="utf-8",
    )
    out = ko._reconstruct_source_from_patch(patch_path, str(original), work / "out.py")
    assert out == ""
    assert outside.read_text(encoding="utf-8") == "safe\n"


def test_reconstruct_no_matching_target(tmp_path):
    """A patch that touches only unrelated files yields nothing."""
    patch = "diff --git a/foo.py b/foo.py\n--- a/foo.py\n+++ b/foo.py\n@@ -1 +1 @@\n-x\n+y\n"
    assert _reconstruct(tmp_path, patch) == ""


def test_reconstruct_double_slash_header_cannot_escape_sandbox(tmp_path):
    """A ``b//abs/path`` header must not write outside the sandbox: the empty
    ``//`` component is dropped and a containment guard keeps the apply inside the
    temp dir, never creating the outside victim path."""
    victim_dir = tmp_path / "OUTSIDE"
    work = tmp_path / "work"
    work.mkdir()
    original = work / "fused_moe.py"
    original.write_text(_RECON_ORIGINAL, encoding="utf-8")
    escape = f"b/{victim_dir}/fused_moe.py"  # -> b//tmp/.../OUTSIDE/fused_moe.py
    patch_path = work / "winner.patch"
    patch_path.write_text(
        f"diff --git a/fused_moe.py {escape}\n"
        f"--- a/fused_moe.py\n+++ {escape}\n"
        "@@ -1,5 +1,8 @@\n import triton\n \n \n+_CACHE = {}\n+\n+\n def helper():\n     return 1\n",
        encoding="utf-8",
    )
    ko._reconstruct_source_from_patch(patch_path, str(original), work / "out.py")
    # Nothing written outside the dir.
    assert not victim_dir.exists()


def test_build_patch_snapshot_sources_worktree_and_base(tmp_path):
    """Snapshot staging materialises byte-exact content for every write path:
    from the worktree when present, else reconstructed from base + patch."""
    base = tmp_path / "base" / "aiter" / "ops"
    base.mkdir(parents=True)
    (base / "k.py").write_text("import triton\nOLD\n")
    worktree = tmp_path / "wt"
    (worktree / "aiter" / "ops").mkdir(parents=True)
    (worktree / "aiter" / "ops" / "k.py").write_text("import triton\nNEW\n")
    (worktree / "aiter" / "ops" / "helper.py").write_text("HELP\n")
    patch = tmp_path / "p.patch"
    patch.write_text(
        "diff --git a/aiter/ops/k.py b/aiter/ops/k.py\n--- a/aiter/ops/k.py\n+++ b/aiter/ops/k.py\n"
        "@@ -1,2 +1,2 @@\n import triton\n-OLD\n+NEW\n"
        "diff --git a/aiter/ops/helper.py b/aiter/ops/helper.py\nnew file mode 100644\n"
        "--- /dev/null\n+++ b/aiter/ops/helper.py\n@@ -0,0 +1 @@\n+HELP\n"
    )
    res = ko.build_patch_snapshot(
        str(patch),
        worktree=worktree,
        kernel_repo=str(tmp_path / "base"),
        clean_base=str(tmp_path / "base"),
        out_dir=tmp_path / "snap",
    )
    assert res is not None
    snap = Path(res["snapshot_dir"])
    assert (snap / "aiter/ops/k.py").read_text() == "import triton\nNEW\n"
    assert (snap / "aiter/ops/helper.py").read_text() == "HELP\n"
    assert len(res["descriptors"]) == 2


def test_build_patch_snapshot_uses_exported_files_without_kernel_repo(
    tmp_path,
):
    files_root = tmp_path / "files"
    optimized = files_root / "vllm" / "ops" / "kernel.py"
    optimized.parent.mkdir(parents=True)
    optimized.write_text("VALUE = 2\n")
    patch = tmp_path / "forge.patch"
    patch.write_text(
        "diff --git a/vllm/ops/kernel.py b/vllm/ops/kernel.py\n"
        "--- a/vllm/ops/kernel.py\n"
        "+++ b/vllm/ops/kernel.py\n"
        "@@ -1 +1 @@\n"
        "-VALUE = 1\n"
        "+VALUE = 2\n"
    )

    res = ko.build_patch_snapshot(
        str(patch),
        worktree=files_root,
        kernel_repo="",
        clean_base="",
        out_dir=tmp_path / "snapshot",
    )

    assert res is not None
    assert (Path(res["snapshot_dir"]) / "vllm" / "ops" / "kernel.py").read_text() == "VALUE = 2\n"


def test_prepare_deploy_patch_drops_python_cache_entries(tmp_path):
    patch = tmp_path / "forge.patch"
    patch.write_text(
        "diff --git a/pkg/__pycache__/kernel.pyc "
        "b/pkg/__pycache__/kernel.pyc\n"
        "Binary files a/pkg/__pycache__/kernel.pyc "
        "and b/pkg/__pycache__/kernel.pyc differ\n"
        "diff --git a/pkg/kernel.py b/pkg/kernel.py\n"
        "--- a/pkg/kernel.py\n"
        "+++ b/pkg/kernel.py\n"
        "@@ -1 +1 @@\n"
        "-VALUE = 1\n"
        "+VALUE = 2\n"
    )

    deploy_patch = ko.prepare_deploy_patch(
        str(patch),
        output_path=tmp_path / "deploy.patch",
    )
    text = Path(deploy_patch).read_text()

    assert "__pycache__" not in text
    assert "pkg/kernel.py" in text


def test_resolve_deploy_repo_root_from_absolute_installed_source(tmp_path):
    deploy_root = tmp_path / "site-packages"
    source = deploy_root / "vllm" / "model_executor" / "attention.py"
    changed = deploy_root / "vllm" / "v1" / "attention" / "ops" / "triton_unified_attention.py"
    source.parent.mkdir(parents=True)
    changed.parent.mkdir(parents=True)
    source.write_text("def wrapper():\n    return True\n")
    changed.write_text("def kernel():\n    return 1\n")
    descriptors = [
        {
            "op": "write",
            "path": ("vllm/v1/attention/ops/triton_unified_attention.py"),
            "is_new": False,
        }
    ]

    assert ko.resolve_deploy_repo_root(
        str(source),
        descriptors,
    ) == str(deploy_root)


def test_resolve_deploy_repo_root_anchors_on_the_traced_source(tmp_path):
    """A tie between two ancestors is broken by the source the trace resolved.

    aiter ships the same relative path under both ``ops/triton`` and
    ``ops/triton/_triton_kernels``, holding two different files, and forge roots
    its worktree at the deeper one -- so the exported entry ``gemm/basic/k.py``
    exists under two ancestors of the traced source. The walk alone finds two
    matches and refuses, reporting a correct rewrite as an unresolvable
    artifact. Which of the two defines the kernel is not a guess: it is what
    ``target_file`` says.
    """
    pkg = tmp_path / "site-packages" / "aiter" / "ops" / "triton"
    shallow = pkg / "gemm" / "basic" / "k.py"
    deep = pkg / "_triton_kernels" / "gemm" / "basic" / "k.py"
    for path, body in ((shallow, "shallow\n"), (deep, "deep\n")):
        path.parent.mkdir(parents=True)
        path.write_text(body)
    descriptors = [
        {"op": "write", "path": "gemm/basic/k.py", "is_new": False},
        {"op": "write", "path": "graph_harness.py", "is_new": True},
    ]

    # Both ancestors carry the relative path, so the walk cannot decide.
    assert shallow.exists() and deep.exists()
    assert ko.resolve_deploy_repo_root(str(deep), descriptors) == str(pkg / "_triton_kernels")
    # The shallower copy anchors on its own root, not on whichever comes first.
    assert ko.resolve_deploy_repo_root(str(shallow), descriptors) == str(pkg)


def test_resolve_deploy_repo_root_anchor_still_requires_the_preimage(tmp_path):
    """The anchor concludes nothing the preimage check would not confirm.

    Stripping a descriptor off the traced path is only a candidate root. A
    descriptor set that does not belong to that tree has to fall through, or the
    anchor would hand a backend a root it never verified.
    """
    deploy_root = tmp_path / "site-packages"
    source = deploy_root / "pkg" / "ops" / "k.py"
    source.parent.mkdir(parents=True)
    source.write_text("kernel\n")

    # Tail matches the traced path, but the sibling preimage is absent, so the
    # derived root fails the check and no ancestor satisfies it either.
    descriptors = [
        {"op": "write", "path": "ops/k.py", "is_new": False},
        {"op": "write", "path": "ops/missing_sibling.py", "is_new": False},
    ]
    assert ko.resolve_deploy_repo_root(str(source), descriptors) == ""

    # A traversal entry is refused rather than resolved by subtraction.
    assert (
        ko.resolve_deploy_repo_root(
            str(source),
            [{"op": "write", "path": "../etc/passwd", "is_new": False}],
        )
        == ""
    )


def test_build_patch_snapshot_returns_none_when_content_unavailable(tmp_path):
    """If any write path can't be made byte-exact (no worktree file, no base to
    reconstruct from), the attempt is non-deployable -> None (hard fail)."""
    base = tmp_path / "base"
    base.mkdir()
    worktree = tmp_path / "wt"
    worktree.mkdir()
    patch = tmp_path / "p.patch"
    patch.write_text(
        "diff --git a/helper.py b/helper.py\nnew file mode 100644\n"
        "--- /dev/null\n+++ b/helper.py\n@@ -0,0 +1 @@\n+HELP\n"
    )
    res = ko.build_patch_snapshot(
        str(patch),
        worktree=worktree,
        kernel_repo=str(base),
        clean_base=str(base),
        out_dir=tmp_path / "snap",
    )
    assert res is None


def test_verification_uses_forge_patch_for_multifile_bundle(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    (base / "kernel.py").write_text("def kernel():\n    return 1\n")
    (base / "helper.py").write_text("HELPER = 1\n")
    patch = tmp_path / "forge.patch"
    patch.write_text(
        "diff --git a/kernel.py b/kernel.py\n"
        "--- a/kernel.py\n"
        "+++ b/kernel.py\n"
        "@@ -1,2 +1,2 @@\n"
        " def kernel():\n"
        "-    return 1\n"
        "+    return 2\n"
        "diff --git a/helper.py b/helper.py\n"
        "--- a/helper.py\n"
        "+++ b/helper.py\n"
        "@@ -1 +1 @@\n"
        "-HELPER = 1\n"
        "+HELPER = 2\n"
    )
    primary = tmp_path / "v1_forge.py"
    primary.write_text("def kernel():\n    return 2\n")
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    (worktree / "kernel.py").write_text(primary.read_text())
    (worktree / "helper.py").write_text("HELPER = 2\n")
    attempt = {
        "status": "completed",
        "attempt_id": "forge-multifile",
        "backend": "forge",
        "optimized_path": str(primary),
        "backend_paths": {
            "forge_patch": str(patch),
            "forge_workspace": str(worktree),
        },
    }

    verification = ko.build_verification(
        _args(
            source_file=str(base / "kernel.py"),
            kernel_repo=str(base),
            correctness_passed=True,
            micro_speedup=1.2,
            accuracy_passed=True,
        ),
        [attempt],
        benchmark_available=True,
    )

    bundle = verification["best_artifact_bundle"]
    assert bundle["type"] == "patch_snapshot"
    assert set(bundle["write_paths"]) == {"kernel.py", "helper.py"}


def test_verification_deploys_sibling_forge_change_without_kernel_repo(
    tmp_path,
):
    deploy_root = tmp_path / "site-packages"
    source = deploy_root / "vllm" / "model_executor" / "attention.py"
    changed = deploy_root / "vllm" / "v1" / "attention" / "ops" / "triton_unified_attention.py"
    source.parent.mkdir(parents=True)
    changed.parent.mkdir(parents=True)
    source.write_text("def wrapper():\n    return True\n")
    changed.write_text("def kernel():\n    return 1\n")

    exported = tmp_path / "canonical-files"
    exported_changed = exported / "vllm" / "v1" / "attention" / "ops" / "triton_unified_attention.py"
    exported_changed.parent.mkdir(parents=True)
    exported_changed.write_text("def kernel():\n    return 2\n")
    patch = tmp_path / "forge.patch"
    patch.write_text(
        "diff --git "
        "a/vllm/v1/attention/ops/triton_unified_attention.py "
        "b/vllm/v1/attention/ops/triton_unified_attention.py\n"
        "--- a/vllm/v1/attention/ops/triton_unified_attention.py\n"
        "+++ b/vllm/v1/attention/ops/triton_unified_attention.py\n"
        "@@ -1,2 +1,2 @@\n"
        " def kernel():\n"
        "-    return 1\n"
        "+    return 2\n"
    )
    primary = tmp_path / "v1_forge.py"
    primary.write_text(source.read_text())
    attempt = {
        "status": "completed",
        "attempt_id": "forge-sibling",
        "backend": "forge",
        "optimized_path": str(primary),
        "pristine_baseline_ms": 2.0,
        "best_ms": 1.0,
        "improved": True,
        "backend_paths": {
            "forge_patch": str(patch),
            "forge_canonical_files_root": str(exported),
            "forge_best_manifest": str(tmp_path / "manifest.json"),
        },
    }

    verification = ko.build_verification(
        _args(
            source_file=str(source),
            kernel_repo="",
            correctness_passed=True,
        ),
        [attempt],
        benchmark_available=True,
    )

    bundle = verification["best_artifact_bundle"]
    assert verification["artifact_valid"] is True
    assert bundle["type"] == "patch_snapshot"
    assert bundle["repo_root"] == str(deploy_root)
    assert bundle["write_paths"] == ["vllm/v1/attention/ops/triton_unified_attention.py"]
    assert (
        Path(bundle["snapshot_dir"]) / "vllm" / "v1" / "attention" / "ops" / "triton_unified_attention.py"
    ).read_text() == "def kernel():\n    return 2\n"


def test_verification_does_not_fall_back_when_forge_snapshot_fails(
    tmp_path,
):
    source = tmp_path / "attention.py"
    source.write_text("def wrapper():\n    return True\n")
    primary = tmp_path / "v1_forge.py"
    primary.write_text(source.read_text())
    patch = tmp_path / "forge.patch"
    patch.write_text(
        "diff --git a/vllm/ops/missing.py b/vllm/ops/missing.py\n"
        "--- a/vllm/ops/missing.py\n"
        "+++ b/vllm/ops/missing.py\n"
        "@@ -1 +1 @@\n"
        "-VALUE = 1\n"
        "+VALUE = 2\n"
    )
    attempt = {
        "status": "completed",
        "attempt_id": "forge-nondeployable",
        "backend": "forge",
        "optimized_path": str(primary),
        "pristine_baseline_ms": 2.0,
        "best_ms": 1.0,
        "improved": True,
        "backend_paths": {
            "forge_patch": str(patch),
        },
    }

    verification = ko.build_verification(
        _args(
            source_file=str(source),
            kernel_repo="",
            correctness_passed=True,
        ),
        [attempt],
        benchmark_available=True,
    )

    assert verification["artifact_valid"] is False
    assert verification["best_artifact_bundle"] == {}
    assert "snapshot" in verification["artifact_error"].lower()
    assert ko.make_proposal(verification)["decision"] != "KEEP"


def test_forge_patch_builds_multifile_deploy_bundle(tmp_path):
    repo = tmp_path / "aiter"
    mirror = repo / "csrc" / "kernels" / "attention_ragged.cu"
    runtime = repo / "csrc" / "cpp_itfs" / "pa" / "pa_kernels.cuh"
    mirror.parent.mkdir(parents=True)
    runtime.parent.mkdir(parents=True)
    old_mirror = '#include <hip/hip_runtime.h>\nextern "C" void original_kernel() {}\n'
    new_mirror = '#include <hip/hip_runtime.h>\nextern "C" void optimized_kernel() {}\n'
    mirror.write_text(old_mirror, encoding="utf-8")
    runtime.write_text("OLD_RUNTIME\n", encoding="utf-8")
    artifact = tmp_path / "v1_forge.cu"
    artifact.write_text(new_mirror, encoding="utf-8")
    report = tmp_path / "optimization_report.md"
    report.write_text(
        "[CORRECTNESS] PASS\n[MICRO_SPEEDUP] 1.10x\n",
        encoding="utf-8",
    )
    files_root = tmp_path / "optimized_versions" / "files"
    final_mirror = files_root / "csrc" / "kernels" / "attention_ragged.cu"
    final_runtime = files_root / "csrc" / "cpp_itfs" / "pa" / "pa_kernels.cuh"
    final_mirror.parent.mkdir(parents=True)
    final_runtime.parent.mkdir(parents=True)
    final_mirror.write_text(new_mirror, encoding="utf-8")
    final_runtime.write_text("NEW_RUNTIME\n", encoding="utf-8")
    patch = tmp_path / "forge.patch"
    patch.write_text(
        "diff --git a/csrc/kernels/attention_ragged.cu "
        "b/csrc/kernels/attention_ragged.cu\n"
        "--- a/csrc/kernels/attention_ragged.cu\n"
        "+++ b/csrc/kernels/attention_ragged.cu\n"
        "@@ -1,2 +1,2 @@\n"
        " #include <hip/hip_runtime.h>\n"
        '-extern "C" void original_kernel() {}\n'
        '+extern "C" void optimized_kernel() {}\n'
        "diff --git a/csrc/cpp_itfs/pa/pa_kernels.cuh "
        "b/csrc/cpp_itfs/pa/pa_kernels.cuh\n"
        "--- a/csrc/cpp_itfs/pa/pa_kernels.cuh\n"
        "+++ b/csrc/cpp_itfs/pa/pa_kernels.cuh\n"
        "@@ -1 +1 @@\n-OLD_RUNTIME\n+NEW_RUNTIME\n",
        encoding="utf-8",
    )
    attempt = {
        "attempt_id": "forge-attention",
        "backend": "forge",
        "status": "completed",
        "optimized_path": str(artifact),
        "backend_paths": {
            "partial_latest_optimized": str(artifact),
            "partial_report": str(report),
            "forge_patch": str(patch),
            "output_dir": str(tmp_path),
        },
    }

    verification = ko.build_verification(
        _args(source_file=str(mirror), kernel_repo=str(repo)),
        [attempt],
        benchmark_available=True,
    )

    bundle = verification["best_artifact_bundle"]
    assert bundle["type"] == "patch_snapshot"
    assert set(bundle["write_paths"]) == {
        "csrc/kernels/attention_ragged.cu",
        "csrc/cpp_itfs/pa/pa_kernels.cuh",
    }


# Downstream-consumer contract: breakdown collector's `glob("{attempt_id}*")` must
# match both the `_optimized.<suffix>` and `_stdout.log` names.


def test_optimized_dir_glob_picks_up_both_legacy_and_new_attempt_files(tmp_path):
    """Lock the `glob("{attempt_id}*")` contract: both names surface for the same attempt id."""
    opt_dir = tmp_path / "optimized"
    opt_dir.mkdir()

    legacy = opt_dir / "geak-deadbeef_optimized.cu"
    legacy.write_text("// historical dry-run / pre-2026-05 layout\n", encoding="utf-8")

    new = opt_dir / "geak-deadbeef_stdout.log"
    new.write_text("real backend stdout transcript\n", encoding="utf-8")

    unrelated = opt_dir / "geak-cafebabe_stdout.log"
    unrelated.write_text("different attempt; must not leak in\n", encoding="utf-8")

    matched = sorted(opt_dir.glob("geak-deadbeef*"))

    assert {p.name for p in matched} == {
        "geak-deadbeef_optimized.cu",
        "geak-deadbeef_stdout.log",
    }, (
        'breakdown/collectors/kernels.py uses `glob(f"{attempt_id}*")` to discover '
        "per-attempt artefacts — both the legacy `_optimized.<suffix>` and "
        "the post-2026-05 `_stdout.log` names must remain discoverable so "
        "older session dirs and new ones render identically in the breakdown."
    )


def test_run_attempt_dry_run_emits_optimized_suffix_file(tmp_path):
    """Dry-run keeps the historical `<attempt_id>_optimized<source_suffix>` filename for smoke-test back-compat."""
    import argparse

    run_dir = tmp_path / "runs" / "sess001"
    args = argparse.Namespace(
        dry_run=True,
        source_file="/tmp/k.cu",
        session_id="sess001",
        budget_minutes=60,
        num_gpus=1,
        target_platform="",
        kernel_id="k001",
    )
    log_path = tmp_path / "run.log"
    log_path.write_text("", encoding="utf-8")

    result = ko.run_attempt(
        "forge",
        args=args,
        candidate={"kernel_id": "k001", "name": "k", "source_file": "/tmp/k.cu"},
        run_dir=run_dir,
        log_path=log_path,
    )

    optimized_path = Path(result["optimized_path"])
    assert optimized_path.exists(), "dry-run must materialise the placeholder"
    assert optimized_path.name.endswith("_optimized.cu"), (
        "dry-run filename must remain `<attempt_id>_optimized<source_suffix>` "
        "for smoke-test back-compat; got " + optimized_path.name
    )
    assert optimized_path.parent.name == "optimized"


def test_unrecoverable_forge_timeout_is_not_promoted_to_partial(
    tmp_path,
    monkeypatch,
):
    run_dir = tmp_path / "runs" / "sess001"
    output_dir = tmp_path / "forge-output"
    output_dir.mkdir()
    (output_dir / "optimization_report.md").write_text(
        "micro_speedup: N/A (no validated improvement kept)\n[correctness] fail\n"
    )
    source = tmp_path / "kernel.py"
    source.write_text("def kernel(x):\n    return x\n")
    args = _args(source_file=str(source), target_platform="MI300X")
    args.session_id = "sess001"
    args.budget_minutes = 60
    args.num_gpus = 1
    log_path = tmp_path / "run.log"
    log_path.write_text("")

    monkeypatch.setattr(
        ko,
        "invoke_backend",
        lambda *_args, **_kwargs: {
            "returncode": 1,
            "stdout": "",
            "stdout_tail": "",
            "stderr_tail": "timeout",
            "output_dir": str(output_dir),
            "timed_out": True,
            "salvaged": False,
        },
    )

    attempt = ko.run_attempt(
        "forge",
        args=args,
        candidate={
            "kernel_id": "k001",
            "name": "kernel",
            "source_file": str(source),
        },
        run_dir=run_dir,
        log_path=log_path,
    )

    assert attempt["status"] == "failed"
    assert attempt["timed_out"] is True
    assert attempt["salvaged"] is False


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
    target_platform,
    expected_name,
    expected_arch,
    expected_flag,
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


def test_build_prompt_includes_budget_protocol_warning():
    args = _args(source_file="")
    args.kernel_id = "budget_kernel"
    args.num_gpus = 0
    args.budget_minutes = 60

    prompt = ko.build_prompt(
        {"name": "budget_kernel", "source_type": "hip"},
        args,
    )

    assert "BUDGET PROTOCOL" in prompt
    assert "--cost-limit 0.0" in prompt
    assert "TELEMETRY" in prompt
    assert prompt.index("BUDGET PROTOCOL") < prompt.index("kernel_name:")


def test_build_prompt_budget_protocol_precedes_source_attribution():
    args = _args(source_file="/tmp/device.cu")
    args.kernel_id = "promoted_kernel"
    args.num_gpus = 0
    args.budget_minutes = 60
    candidate = {
        "name": "promoted_kernel",
        "source_file": "/tmp/device.cu",
        "source_type": "hip",
        "source_promoted_from_launcher": True,
        "launcher_source_file": "/tmp/wrapper.py",
    }

    prompt = ko.build_prompt(candidate, args)

    assert "BUDGET PROTOCOL" in prompt
    assert "SOURCE ATTRIBUTION NOTE" in prompt
    assert prompt.index("BUDGET PROTOCOL") < prompt.index("SOURCE ATTRIBUTION NOTE")


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


def test_build_prompt_metadata_extracts_extra_server_args():
    args = _args(
        source_file="",
        extra_server_args=(
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


def test_load_candidates_backfills_current_tracelens_report_path(tmp_path):
    report = tmp_path / "analysis.md"
    report.write_text("# TraceLens Analysis\n", encoding="utf-8")
    candidates_path = tmp_path / "kernel_candidates.json"
    candidates_path.write_text(
        json.dumps(
            {
                "trace_report_path": str(report),
                "hot_kernels": [{"kernel_id": "k1", "name": "paged_attention"}],
            }
        ),
        encoding="utf-8",
    )

    candidate = ko.load_candidates(candidates_path)[0]

    assert candidate["trace_report_path"] == str(report)


def test_build_prompt_includes_tracelens_context_from_trace_report_path(tmp_path):
    report = tmp_path / "analysis.md"
    report.write_text(
        "# TraceLens Analysis\n\n## Detailed Analysis\nP1: paged attention\n",
        encoding="utf-8",
    )
    args = _args(source_file="")
    args.kernel_id = "paged"
    args.num_gpus = 0
    args.budget_minutes = 60
    candidate = {
        "name": "paged_attention",
        "source_file": "/tmp/paged_attention.py",
        "source_type": "triton",
        "trace_report_path": str(report),
    }

    prompt = ko.build_prompt(candidate, args)

    assert "## TraceLens Context" in prompt
    assert "P1: paged attention" in prompt


def test_build_prompt_strips_base64_images_from_tracelens_context(tmp_path):
    big_b64 = "A" * 5000
    report = tmp_path / "analysis.md"
    report.write_text(
        "# TraceLens Analysis\n\n"
        f"![Performance Improvement](data:image/png;base64,{big_b64})\n\n"
        "## Detailed Analysis\nP2: rmsnorm tuning\n",
        encoding="utf-8",
    )
    args = _args(source_file="")
    args.kernel_id = "k009"
    args.num_gpus = 0
    args.budget_minutes = 60
    candidate = {
        "name": "aiter::rmsnorm",
        "source_file": "/tmp/rmsnorm.py",
        "source_type": "python",
        "trace_report_path": str(report),
    }

    prompt = ko.build_prompt(candidate, args)

    assert "data:image/png;base64" not in prompt
    assert big_b64 not in prompt
    assert "<<stripped: base64 image — Performance Improvement>>" in prompt
    assert "P2: rmsnorm tuning" in prompt


# TraceLens hypothesis block in build_prompt.
def test_build_hypothesis_block_returns_empty_when_no_prose_fields():
    """Candidates lacking prose fields → no-op block."""
    block = ko._build_hypothesis_block(
        {"name": "kernel_no_prose", "source_type": "triton"},
    )
    assert block == ""


def test_build_hypothesis_block_renders_reasoning_and_resolution():
    block = ko._build_hypothesis_block(
        {
            "name": "rms_norm",
            "reasoning_for_slowdown": "Memory-bound kernel saturating HBM bandwidth.",
            "resolution": "Fuse RMSNorm with the following GEMM to amortize loads.",
            "impact_low_ms": 0.0,
            "impact_low_e2e_pct": 0.0,
            "impact_high_ms": 0.0,
            "impact_high_e2e_pct": 0.0,
        }
    )
    assert "## TraceLens Hypothesis [validate before acting]" in block
    assert "Memory-bound kernel saturating HBM bandwidth." in block
    assert "Fuse RMSNorm with the following GEMM" in block
    # Hypothesis framing always present.
    assert "verify the reasoning" in block
    assert "(hypothesis)" in block
    assert "Estimated impact range" not in block


def test_build_hypothesis_block_renders_impact_range_when_set():
    block = ko._build_hypothesis_block(
        {
            "name": "fused_moe",
            "reasoning_for_slowdown": "",
            "resolution": "",
            "impact_low_ms": 12.5,
            "impact_low_e2e_pct": 3.2,
            "impact_high_ms": 40.0,
            "impact_high_e2e_pct": 10.4,
        }
    )
    assert "Estimated impact range" in block
    assert "12.50 ms" in block
    assert "3.20% E2E" in block
    assert "40.00 ms" in block
    assert "10.40% E2E" in block
    # Numbers are framed as TraceLens roofline estimates.
    assert "roofline" in block
    assert "Reasoning for slowdown" not in block
    assert "Recommended direction" not in block


def test_build_hypothesis_block_renders_identification_when_present():
    """Identification line carries per-rank context + source metrics-file ref, labelled distinctly from Reasoning."""
    block = ko._build_hypothesis_block(
        {
            "name": "rms_norm",
            "identification": (
                "Four `aiter::rmsnorm_quant` operations flagged as memory-bound. "
                "(source: rmsnorm_metrics.json -> operations[].efficiency.efficiency_percent)"
            ),
            "reasoning_for_slowdown": "Memory-bound kernel saturating HBM bandwidth.",
            "resolution": "Fuse RMSNorm with the following GEMM.",
        }
    )
    assert "Identification (TraceLens context):" in block
    assert "Four `aiter::rmsnorm_quant`" in block
    assert "rmsnorm_metrics.json" in block
    # Identification appears before Reasoning.
    id_pos = block.index("Identification (TraceLens context):")
    reason_pos = block.index("Reasoning for slowdown (hypothesis):")
    assert id_pos < reason_pos


def test_build_hypothesis_block_renders_when_only_identification_present():
    """A P-item with only Identification still produces a block (GEAK needs the source pointer)."""
    block = ko._build_hypothesis_block(
        {
            "name": "kernel_agent",
            "identification": "Three ops flagged. (source: gemm_metrics.json)",
        }
    )
    assert block != ""
    assert "Identification (TraceLens context):" in block


def test_build_hypothesis_block_renders_all_pitem_prose_when_function_spans_pitems():
    """Multi-entry ``task_group.all_pitem_prose`` renders every P-item with a ``### P{rank}`` header, rank-sorted."""
    candidate = {
        "name": "aiter::rms_norm",
        # Flat prose diverges to confirm the renderer reads from all_pitem_prose.
        "identification": "<should not appear in multi-pitem render>",
        "reasoning_for_slowdown": "<should not appear>",
        "task_group": {
            "all_pitem_prose": [
                {
                    "rank": 2,
                    "title": "Memory-Bound at decode shapes",
                    "identification": "Decode rows: 2.0% of HBM peak. (source: rmsnorm_metrics.json)",
                    "reasoning_for_slowdown": "Small batch → low arithmetic intensity → HBM-bound.",
                    "resolution": "Increase batch upstream OR fuse with adjacent elementwise.",
                    "impact_low_ms": 5.0,
                    "impact_low_e2e_pct": 1.0,
                    "impact_high_ms": 10.0,
                    "impact_high_e2e_pct": 2.0,
                },
                {
                    "rank": 5,
                    "title": "Compute-Bound at prefill shapes",
                    "identification": "Prefill rows: 95% of compute peak. (source: rmsnorm_metrics.json)",
                    "reasoning_for_slowdown": "Large batch saturates MFMA pipelines.",
                    "resolution": "Tile-size tuning; compute-side levers only.",
                    "impact_low_ms": 1.0,
                    "impact_low_e2e_pct": 0.2,
                    "impact_high_ms": 3.0,
                    "impact_high_e2e_pct": 0.6,
                },
            ],
        },
    }
    block = ko._build_hypothesis_block(candidate)
    assert "appears across MULTIPLE TraceLens P-items" in block
    assert "### P2 — Memory-Bound at decode shapes" in block
    assert "### P5 — Compute-Bound at prefill shapes" in block
    assert "Decode rows: 2.0% of HBM peak" in block
    assert "Prefill rows: 95% of compute peak" in block
    assert "Increase batch upstream" in block
    assert "Tile-size tuning" in block
    # P2 before P5.
    p2_pos = block.index("### P2")
    p5_pos = block.index("### P5")
    assert p2_pos < p5_pos
    assert "5.00 ms" in block and "10.00 ms" in block
    assert "1.00 ms" in block and "3.00 ms" in block
    # Flat prose must not leak into the multi-pitem render.
    assert "<should not appear>" not in block


def test_build_hypothesis_block_falls_back_to_flat_prose_for_single_pitem():
    """Single-entry ``all_pitem_prose`` → legacy flat layout (common case, avoids header noise)."""
    candidate = {
        "name": "kernel_agent",
        "reasoning_for_slowdown": "Memory-bound.",
        "resolution": "Fuse with neighbour.",
        "task_group": {
            "all_pitem_prose": [
                {
                    "rank": 1,
                    "title": "Memory-Bound GEMM",
                    "reasoning_for_slowdown": "Memory-bound.",
                    "resolution": "Fuse with neighbour.",
                },
            ],
        },
    }
    block = ko._build_hypothesis_block(candidate)
    assert "appears across MULTIPLE" not in block
    assert "**Reasoning for slowdown (hypothesis):**" in block
    assert "Memory-bound." in block


def test_build_prompt_omits_hypothesis_block_when_no_prose():
    """Backward compat: candidates without prose fields produce the same prompt shape (no extra section/blank lines)."""
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


# Benchmark-cases block in build_prompt.
def test_build_benchmark_cases_block_returns_empty_without_task_group():
    """Legacy dispatch (no task_group) emits no benchmark-cases block."""
    block = ko._build_benchmark_cases_block(
        {"name": "rms_norm", "source_type": "triton"},
    )
    assert block == ""


def test_build_benchmark_cases_block_renders_single_row():
    block = ko._build_benchmark_cases_block(
        {
            "name": "rms_norm",
            "task_group": {
                "function_name": "rms_norm",
                "source_path": "/sgl-workspace/aiter/rmsnorm.py",
                "definition_line": 42,
                "ast_resolved": True,
                "rows": [
                    {
                        "name": "rms_norm",
                        "shapes": ["(8,4096) bf16"],
                        "duration_us": 100_000.0,
                        "call_count": 100,
                        "percent_of_total": 4.2,
                        "flops_per_byte": 0.5,
                        "bound_type": "memory-bound",
                        "efficiency_percent": 30.0,
                        "efficiency_peak_value": 5.3,
                        "efficiency_peak_unit": "TB/s",
                    }
                ],
            },
        }
    )
    assert "## Benchmark cases" in block
    assert "single TraceLens row" in block
    assert "rms_norm" in block
    assert "/sgl-workspace/aiter/rmsnorm.py:42" in block
    assert "Case 1: operation=rms_norm" in block
    assert "per_call_ms=1.000000" in block
    assert "bound=memory-bound" in block
    assert "30.00% of 5.3 TB/s" in block


def test_build_benchmark_cases_block_renders_multiple_rows_sorted_by_time():
    """Multi-row groups render rows aggregate-time-descending and say 'optimize once, applies to all'."""
    block = ko._build_benchmark_cases_block(
        {
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
        }
    )
    assert "across 2 TraceLens rows" in block
    assert "Optimize the source function once" in block
    case_1_idx = block.index("Case 1: operation=rms_norm_prefill")
    case_2_idx = block.index("Case 2: operation=rms_norm_decode")
    assert case_1_idx < case_2_idx


def test_build_prompt_includes_benchmark_cases_when_task_group_present():
    """End-to-end: build_prompt threads the block in when the candidate carries a task_group."""
    prompt = ko.build_prompt(
        {
            "name": "rms_norm",
            "source_type": "triton",
            "task_group": {
                "function_name": "rms_norm",
                "source_path": "/foo/x.py",
                "definition_line": 10,
                "rows": [
                    {
                        "name": "rms_norm",
                        "shapes": ["(8,4096) bf16"],
                        "duration_us": 100_000.0,
                        "call_count": 100,
                        "bound_type": "memory-bound",
                    }
                ],
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


# Bound-keyed optimization priority block in build_prompt.
def test_build_priority_block_empty_when_no_bound_info():
    block = ko._build_priority_block({"name": "kernel_agent", "source_type": "triton"})
    assert block == ""


def test_build_priority_block_memory_bound_leads_with_memory_traffic():
    block = ko._build_priority_block(
        {
            "name": "rms_norm",
            "bound_type": "memory-bound",
        }
    )
    assert "Optimization priorities" in block
    assert "memory-bound" in block
    lev1 = block.index("1. **Memory traffic reduction**")
    lev2 = block.index("2. **Shape-aware tuning**")
    assert lev1 < lev2


def test_build_priority_block_compute_bound_leads_with_compute_utilization():
    block = ko._build_priority_block(
        {
            "name": "gemm_kernel",
            "bound_type": "compute-bound",
        }
    )
    assert "1. **Compute utilization**" in block
    assert "primary lever for compute-bound" in block


def test_build_priority_block_unknown_bound_uses_default_order():
    block = ko._build_priority_block(
        {
            "name": "kernel_agent",
            "bound_type": "mixed",
        }
    )
    # mixed → unknown bucket → structural simplification first.
    assert "1. **Structural simplification**" in block


def test_build_priority_block_reads_bound_from_task_group_primary_row():
    """No top-level bound_type → fall back to the first task_group row's bound_type."""
    block = ko._build_priority_block(
        {
            "name": "rms_norm",
            "task_group": {
                "rows": [{"name": "rms_norm", "bound_type": "memory-bound"}],
            },
        }
    )
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


# make_proposal must surface ``artifact_error`` (not "compile failed") when zero
# backend attempts produced a usable result.
def test_make_proposal_surfaces_backend_dispatch_failure():
    """All dispatch failed (``best`` None) → REVERT reason names the real cause, not compile."""
    verification = {
        "compile_passed": False,
        "correctness_passed": False,
        "artifact_valid": False,
        "artifact_error": "no usable backend attempt",
        "best_attempt_id": "",
        "best_backend": "",
        "best_artifact_path": "",
        "micro_speedup": 0.0,
        "micro_speedup_source": "default_unmeasured",
    }
    proposal = ko.make_proposal(verification)
    assert proposal["decision"] == "REVERT"
    assert len(proposal["reasons"]) == 1
    assert "backend dispatch failed" in proposal["reasons"][0]
    assert "no usable backend attempt" in proposal["reasons"][0]
    # The generic string must NOT appear when we know the real cause.
    assert "compile failed" not in proposal["reasons"][0]


def test_make_proposal_keeps_legacy_compile_failed_when_artifact_lookup_failed():
    """Attempt produced output but artifact resolution failed → REVERT with 'compile failed'."""
    verification = {
        "compile_passed": False,
        "correctness_passed": False,
        "artifact_valid": False,
        "artifact_error": "no complete .hip source artifact found; tried: x.hip",
        "best_attempt_id": "geak-abc",
        "best_backend": "geak",
        "best_artifact_path": "",
        "micro_speedup": 0.0,
        "micro_speedup_source": "default_unmeasured",
    }
    proposal = ko.make_proposal(verification)
    assert proposal["decision"] == "REVERT"
    assert proposal["reasons"] == ["compile failed"]


def test_make_proposal_empty_artifact_error_falls_back_to_compile_failed():
    """compile_passed=False with empty artifact_error must not crash and keeps the fallback reason."""
    verification = {
        "compile_passed": False,
        "correctness_passed": False,
        "artifact_valid": False,
        "artifact_error": "",
        "best_attempt_id": "",
        "best_backend": "",
        "best_artifact_path": "",
        "micro_speedup": 0.0,
        "micro_speedup_source": "default_unmeasured",
    }
    proposal = ko.make_proposal(verification)
    assert proposal["decision"] == "REVERT"
    assert proposal["reasons"] == ["compile failed"]


def _applyback(**overrides):
    payload = {
        "artifact_kind": "framework_applyback",
        "artifact_schema_version": 2,
        "validation_scope": "reference",
        "reference_correctness_passed": True,
        "reference_snr_db": 48.5,
        "integration_validation_required": True,
        "integration_validation_status": "pending",
        "best_commit": "a" * 40,
        "commit_ref": "refs/hyperloom/applyback/attempt-1",
        "builder_symbol": "build_fused_gemm_module",
        "changed_files": ["flydsl_kernel.py", "kernel.py"],
    }
    payload.update(overrides)
    return payload


def _applyback_attempt(tmp_path, *, report_text="", **overrides):
    report = tmp_path / "optimization_report.md"
    report.write_text(report_text or "[correctness] pass\n[integration_validation] pending\n")
    artifact = tmp_path / "optimized.py"
    artifact.write_text(
        "import torch\n\n\ndef flydsl_kernel(x):\n    return x + 1\n",
        encoding="utf-8",
    )
    attempt = _attempt(report, artifact, backend="forge")
    attempt["mean_case_speedup"] = 2.0
    attempt["flydsl_applyback"] = _applyback(**overrides)
    return attempt


def test_reference_verified_applyback_answers_the_micro_correctness_gate(tmp_path):
    verification = ko.build_verification(
        _args(source_file=str(tmp_path / "kernel.py"), micro_speedup=1.6),
        [_applyback_attempt(tmp_path)],
        benchmark_available=True,
    )

    assert verification["correctness_passed"] is True
    assert verification["correctness_source"] == "forge_rewrite_reference"
    assert verification["integration_validation_status"] == "pending"


def test_applyback_reference_outranks_the_report_scan(tmp_path):
    """The validated manifest is the authority; report prose cannot override it."""
    verification = ko.build_verification(
        _args(source_file=str(tmp_path / "kernel.py"), micro_speedup=1.6),
        [_applyback_attempt(tmp_path, report_text="correctness failed\n")],
        benchmark_available=True,
    )

    assert verification["correctness_source"] == "forge_rewrite_reference"
    assert verification["correctness_passed"] is True


def test_cli_correctness_override_still_outranks_the_applyback(tmp_path):
    verification = ko.build_verification(
        _args(
            source_file=str(tmp_path / "kernel.py"),
            micro_speedup=1.6,
            correctness_passed=False,
        ),
        [_applyback_attempt(tmp_path)],
        benchmark_available=True,
    )

    assert verification["correctness_source"] == "cli_override"
    assert verification["correctness_passed"] is False


def test_verification_propagates_the_applyback_provenance(tmp_path):
    verification = ko.build_verification(
        _args(source_file=str(tmp_path / "kernel.py"), micro_speedup=1.6),
        [_applyback_attempt(tmp_path)],
        benchmark_available=True,
    )

    evidence = verification["framework_applyback"]
    assert evidence["artifact_kind"] == "framework_applyback"
    assert evidence["artifact_schema_version"] == 2
    assert evidence["validation_scope"] == "reference"
    assert evidence["reference_correctness_passed"] is True
    assert evidence["reference_snr_db"] == 48.5
    assert evidence["integration_validation_required"] is True
    assert evidence["integration_validation_status"] == "pending"
    assert evidence["commit"] == "a" * 40
    assert evidence["commit_ref"] == "refs/hyperloom/applyback/attempt-1"
    assert evidence["builder_symbol"] == "build_fused_gemm_module"
    assert evidence["changed_files"] == ["flydsl_kernel.py", "kernel.py"]


def test_verification_without_an_applyback_carries_no_evidence(tmp_path):
    report = tmp_path / "optimization_report.md"
    report.write_text("[correctness] pass\n")

    verification = ko.build_verification(
        _args(source_file=str(tmp_path / "kernel.py"), micro_speedup=1.6),
        [_attempt(report, backend="geak")],
        benchmark_available=True,
    )

    assert verification["framework_applyback"] == {}
    assert verification["integration_validation_status"] == ""
    assert verification["correctness_source"] == "report_scan"


def test_pending_integration_keeps_the_micro_proposal_and_names_the_deferral(tmp_path):
    """A pending framework verdict must not block the patch from reaching E2E."""
    verification = ko.build_verification(
        _args(source_file=str(tmp_path / "kernel.py"), micro_speedup=1.6),
        [_applyback_attempt(tmp_path)],
        benchmark_available=True,
    )

    proposal = ko.make_proposal(verification)

    assert proposal["decision"] == "KEEP"
    assert proposal["reasons"] == [
        "framework apply-back reference-verified; framework E2E/accuracy deferred to integrate"
    ]


def _git(repo, *args: str) -> str:
    import subprocess

    proc = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True)
    return proc.stdout.strip()


def _rewrite_backend_result(tmp_path):
    """Reproduce what the rewrite backend leaves behind for one attempt."""
    workspace = tmp_path / "worktree"
    workspace.mkdir()
    _git(workspace, "init", "-q")
    _git(workspace, "config", "user.email", "forge@test")
    _git(workspace, "config", "user.name", "forge")
    base_kernel = "def kernel(x):\n    return x\n"
    (workspace / "kernel.py").write_text(base_kernel)
    _git(workspace, "add", "-A")
    _git(workspace, "commit", "-qm", "base")
    base_commit = _git(workspace, "rev-parse", "HEAD")

    patched_kernel = "def kernel(x):\n    return flydsl_kernel(x)\n"
    flydsl_module = "def flydsl_kernel(x):\n    return x\n"
    (workspace / "kernel.py").write_text(patched_kernel)
    (workspace / "flydsl_kernel.py").write_text(flydsl_module)
    _git(workspace, "add", "-A")
    _git(workspace, "commit", "-qm", "apply-back")
    best_commit = _git(workspace, "rev-parse", "HEAD")
    patch_text = _git(workspace, "diff", "--binary", f"{base_commit}..{best_commit}")

    canonical = workspace / "forge_experiments" / "rewrite"
    (canonical / "files").mkdir(parents=True)
    (canonical / "files" / "kernel.py").write_text(patched_kernel)
    (canonical / "files" / "flydsl_kernel.py").write_text(flydsl_module)
    (canonical / "forge.patch").write_text(patch_text + "\n")
    (canonical / "manifest.json").write_text("{}", encoding="utf-8")

    output_dir = tmp_path / "attempt"
    exported = output_dir / "optimized_versions" / "files"
    exported.mkdir(parents=True)
    # The primary compatibility artifact is deliberately the unchanged base:
    # only the multi-file bundle carries the real change.
    (output_dir / "optimized_versions" / "v1_forge.py").write_text(base_kernel)
    (exported / "kernel.py").write_text(patched_kernel)
    (exported / "flydsl_kernel.py").write_text(flydsl_module)
    (output_dir / "optimized_versions" / "forge.patch").write_text(patch_text + "\n")
    (output_dir / "optimization_report.md").write_text(
        "# Forge optimization report\n\n"
        "[micro_speedup] 2.0000x\nmean_case_speedup=2.000000\n"
        "[correctness] pass\n[integration_validation] pending\n"
    )

    return {
        "returncode": 0,
        "stdout": "forge rewrite done (cli)",
        "cli_workspace": str(output_dir),
        "output_dir": str(output_dir),
        "best_commit": best_commit,
        "best_ms": 1.0,
        "pristine_baseline_ms": 2.0,
        "mean_case_speedup": 2.0,
        "search_start_mean_case_speedup": 1.0,
        "improved": True,
        "total_improved": True,
        "incremental_improved": True,
        "best_manifest": str(canonical / "manifest.json"),
        "canonical_patch_path": str(canonical / "forge.patch"),
        "canonical_files_root": str(canonical / "files"),
        "changed_files": ["flydsl_kernel.py", "kernel.py"],
        "forge_workspace": str(workspace),
        "artifacts": [str(canonical / "forge.patch")],
        "flydsl_applyback": _applyback(best_commit=best_commit),
        "target_functions": ["kernel"],
    }, workspace


def test_rewrite_backend_result_reaches_a_reference_verified_keep(tmp_path, monkeypatch):
    """The backend's result keys must line up with what verification reads."""
    result, workspace = _rewrite_backend_result(tmp_path)
    monkeypatch.setattr(ko, "invoke_backend", lambda *a, **k: result)

    run_dir = tmp_path / "runs" / "sess001"
    log_path = tmp_path / "run.log"
    log_path.write_text("", encoding="utf-8")
    args = _args(
        source_file=str(workspace / "kernel.py"),
        kernel_repo=str(workspace),
        session_id="sess001",
        budget_minutes=60,
        num_gpus=1,
        target_platform="",
    )
    attempt = ko.run_attempt(
        "forge",
        args=args,
        candidate=_candidate(source_file=str(workspace / "kernel.py"), source_type="triton"),
        run_dir=run_dir,
        log_path=log_path,
    )

    assert attempt["flydsl_applyback"]["artifact_kind"] == "framework_applyback"
    assert attempt["backend_paths"]["forge_canonical_files_root"].endswith("rewrite/files")

    verification = ko.build_verification(args, [attempt], benchmark_available=True)

    assert verification["correctness_source"] == "forge_rewrite_reference"
    assert verification["correctness_passed"] is True
    assert verification["integration_validation_status"] == "pending"
    assert verification["framework_applyback"]["changed_files"] == [
        "flydsl_kernel.py",
        "kernel.py",
    ]
    # The unchanged primary artifact must not stand in for the real change.
    bundle = verification["best_artifact_bundle"]
    assert verification["artifact_valid"] is True
    assert bundle["type"] == "patch_snapshot"
    assert sorted(bundle["write_paths"]) == ["flydsl_kernel.py", "kernel.py"]
    snapshot = Path(bundle["snapshot_dir"])
    assert (snapshot / "flydsl_kernel.py").is_file()
    assert (snapshot / "kernel.py").read_text() == "def kernel(x):\n    return flydsl_kernel(x)\n"

    proposal = ko.make_proposal(verification)
    assert proposal["decision"] == "KEEP"
    assert proposal["reasons"] == [
        "framework apply-back reference-verified; framework E2E/accuracy deferred to integrate"
    ]


def test_pending_integration_does_not_rescue_a_below_threshold_speedup(tmp_path):
    verification = ko.build_verification(
        _args(source_file=str(tmp_path / "kernel.py"), micro_speedup=1.02),
        [_applyback_attempt(tmp_path)],
        benchmark_available=True,
    )

    proposal = ko.make_proposal(verification)

    assert proposal["decision"] == "NEEDS_REVIEW"
    assert any("below KEEP threshold" in reason for reason in proposal["reasons"])


def test_build_prompt_out_of_root_source_file_not_embedded(tmp_path):
    # A planted directory carrying a root substring is outside every real root.
    planted = tmp_path / "sgl-workspace" / "aiter"
    planted.mkdir(parents=True)
    planted_file = planted / "kernel.py"
    planted_file.write_text("PLANTED_CONTENT", encoding="utf-8")

    prompt = ko.build_prompt(_candidate(), _args(source_file=str(planted_file)))

    assert "PLANTED_CONTENT" not in prompt


def test_build_prompt_in_root_source_file_embedded(tmp_path, monkeypatch):
    src = tmp_path / "kernel.py"
    src.write_text("def in_root_kernel(): pass\n", encoding="utf-8")

    monkeypatch.setattr(
        "hyperloom.orchestrator.framework.paths.resolve_patch_target_roots",
        lambda: (str(tmp_path) + "/",),
    )

    prompt = ko.build_prompt(_candidate(), _args(source_file=str(src)))

    assert "in_root_kernel" in prompt
