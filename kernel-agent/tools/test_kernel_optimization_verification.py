# Copyright Advanced Micro Devices, Inc. All rights reserved.

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
    """Report-scan correctness lights up on its own (no `accuracy_passed`, which would mask the report scanner via accuracy_override)."""
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


def _geak_attempt(tmp_path: Path, *, status: str = "complete", speedup: float = 1.3):
    """Build a GEAK-shaped attempt with a final_report.json on disk."""
    final = tmp_path / "geak_final_report.json"
    final.write_text(json.dumps({
        "status": status,
        "best_patch": str(tmp_path / "patch_1.patch"),
        "best_speedup": speedup,
        "summary": "import-only harness, no kernel exercised",
    }), encoding="utf-8")
    artifact = tmp_path / "worktree" / "moe_op.py"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text("import torch\ndef ck_moe_stage1_fwd(*a, **k): pass\n", encoding="utf-8")
    return {
        "status": "completed",
        "attempt_id": "geak-aaa",
        "backend": "geak",
        "optimized_path": str(artifact),
        "backend_paths": {
            "geak_final_report": str(final),
            "geak_per_task_best_speedup": str(speedup),
            "geak_per_task_best_patch": str(tmp_path / "patch_1.patch"),
            "geak_per_task_best_worktree": str(artifact.parent.parent),
        },
    }


def test_geak_correctness_trusted_by_default(tmp_path):
    """PR-E default ON: GEAK status=complete + measured-speedup auto-promotes to KEEP (Qwen3-30B-A3B-Base ran GEAK 0/4 KEEP without it)."""
    verification = ko.build_verification(
        _args(source_file="/tmp/moe_op.py"),
        [_geak_attempt(tmp_path, status="complete", speedup=1.3)],
        benchmark_available=True,
    )
    assert verification["micro_speedup"] == 1.3
    assert verification["correctness_passed"] is True
    assert verification["correctness_source"] == "geak_assumed_pass"
    proposal = ko.make_proposal(verification)
    assert proposal["decision"] == "KEEP", proposal


def test_geak_correctness_can_be_disabled_via_env(tmp_path, monkeypatch):
    """HYPERLOOM_TRUST_GEAK_CORRECTNESS=0 restores pre-PR-E conservative NEEDS_REVIEW behaviour."""
    monkeypatch.setenv("HYPERLOOM_TRUST_GEAK_CORRECTNESS", "0")
    verification = ko.build_verification(
        _args(source_file="/tmp/moe_op.py"),
        [_geak_attempt(tmp_path, status="complete", speedup=1.3)],
        benchmark_available=True,
    )
    assert verification["micro_speedup"] == 1.3
    assert verification["correctness_passed"] is False
    assert verification["correctness_source"] == "missing"
    proposal = ko.make_proposal(verification)
    assert proposal["decision"] == "NEEDS_REVIEW"


def test_geak_correctness_trust_requires_nonzero_speedup(tmp_path):
    """Trust gate requires geak_per_task_best_speedup > 0, else a no-op patch would be silently KEPT."""
    attempt = _geak_attempt(tmp_path, status="complete", speedup=0.0)
    attempt["backend_paths"]["geak_per_task_best_speedup"] = "0"
    verification = ko.build_verification(
        _args(),
        [attempt],
        benchmark_available=True,
    )
    assert verification["correctness_passed"] is False
    assert verification["correctness_source"] == "missing"


def test_geak_correctness_trust_requires_complete_status(tmp_path):
    """Trust default must not promote status='complete_no_patch' (select_patch found nothing) to KEEP."""
    verification = ko.build_verification(
        _args(),
        [_geak_attempt(tmp_path, status="complete_no_patch", speedup=1.3)],
        benchmark_available=True,
    )
    assert verification["correctness_passed"] is False
    assert verification["correctness_source"] == "missing"


def test_report_correctness_passes_with_reference_language(tmp_path):
    report = tmp_path / "optimization_report.md"
    report.write_text(
        "The optimized implementation matches reference outputs for all test shapes.\n"
        "Speedup: 1.41x\n",
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
    """Explicit "Correctness failed" in the report must block KEEP (no `accuracy_passed`, which would mask it via accuracy_override)."""
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
        _args(correctness_passed=True, micro_speedup=1.25,
              e2e_gain_pct=0.5, accuracy_passed=True),
        [_attempt(artifact=artifact)],
        benchmark_available=False,
    )
    assert verification["correctness_passed"] is True
    assert verification["correctness_source"] == "cli_override"
    assert ko.make_proposal(verification)["decision"] == "KEEP"


def test_speedup_just_above_gate_keeps(tmp_path):
    """A 1.07x speedup clears the 1.05x KEEP gate (issue #442: was rejected by the old higher gate)."""
    artifact = tmp_path / "optimized.hip"
    verification = ko.build_verification(
        _args(correctness_passed=True, micro_speedup=1.07,
              e2e_gain_pct=0.5, accuracy_passed=True),
        [_attempt(artifact=artifact)],
        benchmark_available=False,
    )
    assert ko.make_proposal(verification)["decision"] == "KEEP"


def test_speedup_below_gate_needs_review(tmp_path):
    """A 1.03x speedup (improvement but under the 1.05x gate) routes to NEEDS_REVIEW, not KEEP."""
    artifact = tmp_path / "optimized.hip"
    verification = ko.build_verification(
        _args(correctness_passed=True, micro_speedup=1.03,
              e2e_gain_pct=0.5, accuracy_passed=True),
        [_attempt(artifact=artifact)],
        benchmark_available=False,
    )
    proposal = ko.make_proposal(verification)
    assert proposal["decision"] == "NEEDS_REVIEW"
    assert any("below KEEP" in r for r in proposal["reasons"])


# GEAK worktree artifact recovery: rewritten source lives in the worktree slot dir, not the binary-laden .patch.


def test_geak_best_worktree_maps_parallel_slot(tmp_path):
    """``parallel_<M>/patch.patch`` resolves to ``worktrees/slot_<M>/`` under the same round dir."""
    round_dir = tmp_path / "results" / "round_1"
    slot_dir = round_dir / "worktrees" / "slot_3"
    slot_dir.mkdir(parents=True)
    patch = round_dir / "parallel_3" / "patch_1.patch"
    patch.parent.mkdir(parents=True)
    patch.write_text("diff --git a/x b/x\n", encoding="utf-8")

    assert ko._geak_best_worktree(str(patch)) == slot_dir


def test_geak_best_worktree_returns_none_for_unexpected_layout(tmp_path):
    """Non-``parallel_<M>`` layouts fail soft so the caller falls back to ``.patch``-based recovery."""
    other = tmp_path / "results" / "round_1" / "weird_dir" / "patch.patch"
    other.parent.mkdir(parents=True)
    other.write_text("", encoding="utf-8")
    assert ko._geak_best_worktree(str(other)) is None
    assert ko._geak_best_worktree("") is None


def test_geak_best_worktree_returns_none_when_slot_dir_missing(tmp_path):
    """Missing ``worktrees/slot_<M>/`` on disk → helper refuses to return a non-existent path."""
    parallel = tmp_path / "results" / "round_1" / "parallel_0"
    parallel.mkdir(parents=True)
    patch = parallel / "patch_0.patch"
    patch.write_text("", encoding="utf-8")
    assert ko._geak_best_worktree(str(patch)) is None


def test_worktree_source_paths_prefers_repo_relative_join(tmp_path):
    """With ``kernel_repo`` set, resolve via ``source_file - kernel_repo`` first (before basename rglob) to avoid same-named stub collisions."""
    repo = tmp_path / "repo"
    src = repo / "aiter" / "ops" / "rmsnorm.py"
    src.parent.mkdir(parents=True)
    src.write_text("def f(): pass\n", encoding="utf-8")

    worktree = tmp_path / "worktree"
    canonical = worktree / "aiter" / "ops" / "rmsnorm.py"
    decoy = worktree / "aiter" / "ops" / "triton" / "normalization" / "rmsnorm.py"
    canonical.parent.mkdir(parents=True)
    decoy.parent.mkdir(parents=True)
    canonical.write_text("def f(): return 'canonical'\n", encoding="utf-8")
    decoy.write_text("def f(): return 'decoy'\n", encoding="utf-8")

    paths = ko._worktree_source_paths(
        worktree, source_file=str(src), kernel_repo=str(repo),
    )
    assert paths
    # Canonical mapping fires first; decoy collected but never primary.
    assert paths[0] == canonical


def test_worktree_source_paths_falls_back_to_basename_when_no_repo(tmp_path):
    """Empty ``kernel_repo`` → fall back to bounded ``worktree.rglob(basename)`` for legacy CSV-only fixtures."""
    worktree = tmp_path / "worktree"
    hit = worktree / "deep" / "nested" / "rmsnorm.py"
    hit.parent.mkdir(parents=True)
    hit.write_text("def f(): pass\n", encoding="utf-8")

    paths = ko._worktree_source_paths(
        worktree, source_file="/anywhere/rmsnorm.py", kernel_repo="",
    )
    assert paths == [hit]


def test_candidate_artifact_paths_prefers_worktree_over_patch(tmp_path):
    """Worktree ``.py`` must precede ``.patch`` so suffix check picks it up before fence extraction on a binary-laden diff."""
    repo = tmp_path / "repo"
    src = repo / "aiter" / "ops" / "rmsnorm.py"
    src.parent.mkdir(parents=True)
    src.write_text("def f(): pass\n", encoding="utf-8")

    worktree = tmp_path / "results" / "round_1" / "worktrees" / "slot_0"
    wt_file = worktree / "aiter" / "ops" / "rmsnorm.py"
    wt_file.parent.mkdir(parents=True)
    wt_file.write_text("def f(): return 'patched'\n", encoding="utf-8")

    patch = tmp_path / "results" / "round_1" / "parallel_0" / "patch_0.patch"
    patch.parent.mkdir(parents=True)
    patch.write_text("diff --git a/x b/x\n", encoding="utf-8")

    attempt = {
        "status": "completed",
        "attempt_id": "geak0",
        "backend": "geak",
        "optimized_path": str(patch),
        "backend_paths": {
            "geak_per_task_best_patch": str(patch),
            "geak_per_task_best_worktree": str(worktree),
        },
    }
    paths = ko._candidate_artifact_paths(
        attempt, ".py",
        source_file=str(src),
        kernel_repo=str(repo),
    )
    assert paths
    assert paths[0] == wt_file
    # Patch stays in the list as a fallback, just not first.
    assert any(p == patch for p in paths)


def test_build_verification_recovers_py_from_worktree(tmp_path):
    """End-to-end: GEAK ``.patch`` + worktree ``.py`` yields artifact_valid=True, artifact_source='source_file' (not 'missing')."""
    repo = tmp_path / "repo"
    src = repo / "aiter" / "ops" / "rmsnorm.py"
    src.parent.mkdir(parents=True)
    src.write_text("def rmsnorm2d_fwd():\n    pass\n", encoding="utf-8")

    worktree = tmp_path / "results" / "round_1" / "worktrees" / "slot_0"
    wt_file = worktree / "aiter" / "ops" / "rmsnorm.py"
    wt_file.parent.mkdir(parents=True)
    wt_file.write_text(
        "def rmsnorm2d_fwd():\n    return 'optimized'\n", encoding="utf-8",
    )

    patch = tmp_path / "results" / "round_1" / "parallel_0" / "patch_1.patch"
    patch.parent.mkdir(parents=True)
    patch.write_text(
        "diff --git a/aiter/ops/rmsnorm.py b/aiter/ops/rmsnorm.py\n",
        encoding="utf-8",
    )

    attempt = {
        "status": "completed",
        "attempt_id": "geak0",
        "backend": "geak",
        "optimized_path": str(patch),
        "backend_paths": {
            "geak_per_task_best_patch": str(patch),
            "geak_per_task_best_worktree": str(worktree),
        },
    }
    verification = ko.build_verification(
        _args(source_file=str(src), kernel_repo=str(repo)),
        [attempt],
        benchmark_available=True,
    )
    assert verification["artifact_valid"] is True
    assert verification["artifact_source"] == "source_file"
    assert verification["best_artifact_path"] == str(wt_file)


# GEAK prompt yaml patcher: rewrites a misleading task_runner.py example to placeholders; pin idempotency + fail-soft.


def _seed_yaml(tmp_path):
    """Minimal copy of the relevant upstream YAML block (works without minisweagent installed)."""
    yaml = tmp_path / "config" / "mini_kernel_strategy_list.yaml"
    yaml.parent.mkdir(parents=True)
    yaml.write_text(
        "system_prompt: |\n"
        "    profile_kernel:\n"
        "    - Forbidden in `command`: `&&`, `cd`\n"
        '    - Good example: `command="python3 scripts/task_runner.py performance",'
        ' workdir="/path/to/project"`\n'
        '    - Also ok: `command="python3 /absolute/path/to/scripts/task_runner.py'
        ' performance"`\n'
        '    - Bad example: `command="cd /path && python3 scripts/task_runner.py'
        ' performance"`\n'
        "    other_tool:\n"
        "    - keep\n",
        encoding="utf-8",
    )
    return yaml


def test_geak_prompt_patcher_replaces_misleading_example(tmp_path, monkeypatch):
    import geak_prompt_patcher as gpp

    yaml = _seed_yaml(tmp_path)
    monkeypatch.setenv("HYPERLOOM_GEAK_PROMPT_YAML", str(yaml))

    ok, msg = gpp.ensure_geak_prompt_patched()
    assert ok is True
    assert "patched" in msg
    text = yaml.read_text(encoding="utf-8")
    assert "task_runner.py performance" not in text
    assert "<your_benchmark.py>" in text
    assert "Forbidden in `command`" in text
    assert "other_tool" in text


def test_geak_prompt_patcher_idempotent(tmp_path, monkeypatch):
    import geak_prompt_patcher as gpp

    yaml = _seed_yaml(tmp_path)
    monkeypatch.setenv("HYPERLOOM_GEAK_PROMPT_YAML", str(yaml))

    ok1, _ = gpp.ensure_geak_prompt_patched()
    text1 = yaml.read_text(encoding="utf-8")
    assert ok1 is True

    ok2, msg2 = gpp.ensure_geak_prompt_patched()
    text2 = yaml.read_text(encoding="utf-8")
    assert ok2 is True
    assert "already patched" in msg2
    assert text1 == text2


def test_geak_prompt_patcher_fails_soft_when_yaml_missing(tmp_path, monkeypatch):
    """Missing YAML → ``ensure_geak_prompt_patched`` returns ``(False, …)`` (UX hardening, not a correctness gate)."""
    import geak_prompt_patcher as gpp

    monkeypatch.setenv(
        "HYPERLOOM_GEAK_PROMPT_YAML", str(tmp_path / "does_not_exist.yaml"),
    )
    ok, msg = gpp.ensure_geak_prompt_patched()
    assert ok is False
    assert "minisweagent not installed" in msg


def test_geak_prompt_patcher_refuses_to_guess_on_drift(tmp_path, monkeypatch):
    """Upstream wording drift → patcher refuses rather than half-patching the YAML."""
    import geak_prompt_patcher as gpp

    yaml = tmp_path / "drifted.yaml"
    yaml.write_text(
        "system_prompt: |\n"
        "    profile_kernel:\n"
        '    - Good example: `command="python3 different_runner.py", workdir="/x"`\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("HYPERLOOM_GEAK_PROMPT_YAML", str(yaml))

    ok, msg = gpp.ensure_geak_prompt_patched()
    assert ok is False
    assert "upstream example block changed" in msg
    assert "different_runner.py" in yaml.read_text(encoding="utf-8")


def test_geak_prompt_patcher_recognises_upstream_already_fixed(tmp_path, monkeypatch):
    """When upstream YAML already uses generic <your-test-command> placeholders
    (GEAK ec61bdb+), the patcher should return ok without modifying the file."""
    import geak_prompt_patcher as gpp

    yaml = tmp_path / "config" / "mini_kernel_strategy_list.yaml"
    yaml.parent.mkdir(parents=True)
    yaml.write_text(
        "system_prompt: |\n"
        "    profile_kernel:\n"
        "    - Forbidden in `command`: `&&`, `cd`\n"
        '    - Good example: `command="<your-test-command>", workdir="/path/to/project"`\n'
        '    - Bad example: `command="cd /path/to/project && <your-test-command>"`\n'
        "    other_tool:\n"
        "    - keep\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HYPERLOOM_GEAK_PROMPT_YAML", str(yaml))

    ok, msg = gpp.ensure_geak_prompt_patched()
    assert ok is True
    assert "upstream" in msg.lower() or "already" in msg.lower()
    # File must not be modified.
    assert "<your-test-command>" in yaml.read_text(encoding="utf-8")


def test_benchmark_files_list_counts_as_benchmark(tmp_path):
    bench = tmp_path / "bench.py"
    bench.write_text("print('ok')\n", encoding="utf-8")
    args = _args()
    args.benchmark_file = ""
    args.test_harness_path = ""
    assert ko.has_benchmark(args, {"benchmark_files": [str(bench)]}) is True


# Regression: backend stdout must never be promoted to `source_file` artifact (Qwen3-8B k007 2026-05-20); stdout now goes to `_stdout.log` and only the fenced-block extraction path may surface it.


def test_geak_stdout_log_must_not_false_positive_as_source_file(tmp_path):
    """Stdout-log-only artifact (no patch, no .cu, no code fence) → artifact_source == "missing" (Qwen3-8B k007 2026-05-20)."""
    log_path = tmp_path / "geak-deadbeef_stdout.log"
    log_path.write_text(
        "minisweagent.agents.parallel_agent: INFO: [running 12.0min] Sub-agents working\n"
        "2 total patches (task_0: 2)\n"
        # Embeds void/int/float markers the pre-fix heuristic accepted as CUDA source.
        "Trajectory note: convert the int loop, drop the void wrapper, use float.\n"
        "Best patch: patch_1 (agent 0)\n",
        encoding="utf-8",
    )
    attempt = {
        "status": "completed",
        "attempt_id": "geak-deadbeef",
        "backend": "geak",
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
    log_path = tmp_path / "claude-c0ffee_stdout.log"
    log_path.write_text(
        "Here is the final optimized kernel:\n"
        "```cuda\n"
        "#include <hip/hip_runtime.h>\n"
        "extern \"C\" __global__ void optimized_kernel(float* out, const float* in) {\n"
        "  int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
        "  out[idx] = in[idx] * 2.0f;\n"
        "}\n"
        "```\n"
        "Done.\n",
        encoding="utf-8",
    )
    attempt = {
        "status": "completed",
        "attempt_id": "claude-c0ffee",
        "backend": "claude",
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
    assert "extern \"C\" __global__ void optimized_kernel" in body
    assert "```" not in body
    assert "Here is the final" not in body


def test_geak_patch_is_preferred_over_stdout_log(tmp_path):
    """Patch wins over stdout log; a diff has no fenced block so we return `missing` rather than promoting the log (`_candidate_artifact_paths` precedence)."""
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
        "backend": "geak",
        "optimized_path": str(log_path),
        "backend_paths": {
            "geak_per_task_best_patch": str(patch_path),
            "geak_latest_patch": str(patch_path),
        },
    }

    artifact_path, source, error = ko._select_source_artifact(
        attempt,
        target_file="/tmp/source.cu",
        run_dir=tmp_path,
    )

    # Crucial: the marker-noise log is NOT silently promoted to source_file (the pre-fix bug).
    assert source == "missing"
    assert artifact_path == ""


# Downstream-consumer contract: breakdown collector's `glob("{attempt_id}*")` must keep matching both legacy `_optimized.<suffix>` and new `_stdout.log` names (kernel-agent/SKILL.md § Per-attempt stdout file naming).


def test_optimized_dir_glob_picks_up_both_legacy_and_new_attempt_files(tmp_path):
    """Lock the `glob("{attempt_id}*")` contract: both names surface for the same attempt id across old + new sessions."""
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
        "breakdown/collectors.py uses `glob(f\"{attempt_id}*\")` to discover "
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
        geak_budget_min=120,
        budget_minutes=60,
        disable_rag=False,
        disable_xs_memory=False,
        num_gpus=1,
        target_platform="",
        test_command="",
        kernel_id="k001",
    )
    log_path = tmp_path / "run.log"
    log_path.write_text("", encoding="utf-8")

    result = ko.run_attempt(
        "claude",
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
        json.dumps({
            "trace_report_path": str(report),
            "hot_kernels": [{"kernel_id": "k1", "name": "paged_attention"}],
        }),
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


# PR-A §4: TraceLens hypothesis block in build_prompt
def test_build_hypothesis_block_returns_empty_when_no_prose_fields():
    """Candidates lacking prose fields → no-op block (prompt byte-identical to pre-PR)."""
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
    # Hypothesis framing always present so GEAK doesn't treat the guess as ground truth.
    assert "verify the reasoning" in block
    assert "(hypothesis)" in block
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
    # Numbers are TraceLens roofline estimates, framed as such so GEAK doesn't treat them as measured.
    assert "roofline" in block
    assert "Reasoning for slowdown" not in block
    assert "Recommended direction" not in block


def test_build_hypothesis_block_renders_identification_when_present():
    """Identification line carries per-rank context + source metrics-file ref, labelled distinctly from Reasoning."""
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
    # Identification appears before Reasoning (what before why).
    id_pos = block.index("Identification (TraceLens context):")
    reason_pos = block.index("Reasoning for slowdown (hypothesis):")
    assert id_pos < reason_pos


def test_build_hypothesis_block_renders_when_only_identification_present():
    """A P-item with only Identification still produces a block (GEAK needs the source pointer)."""
    block = ko._build_hypothesis_block({
        "name": "kernel",
        "identification": "Three ops flagged. (source: gemm_metrics.json)",
    })
    assert block != ""
    assert "Identification (TraceLens context):" in block


def test_build_hypothesis_block_renders_all_pitem_prose_when_function_spans_pitems():
    """Q2: multi-entry ``task_group.all_pitem_prose`` renders every P-item with a ``### P{rank}`` header, rank-sorted."""
    candidate = {
        "name": "aiter::rms_norm",
        # Primary's flat prose intentionally divergent so the test confirms the renderer reads from all_pitem_prose.
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
    # P2 before P5 (rank-ascending).
    p2_pos = block.index("### P2")
    p5_pos = block.index("### P5")
    assert p2_pos < p5_pos
    assert "5.00 ms" in block and "10.00 ms" in block
    assert "1.00 ms" in block and "3.00 ms" in block
    # Candidate's flat prose must not leak into the multi-pitem render.
    assert "<should not appear>" not in block


def test_build_hypothesis_block_falls_back_to_flat_prose_for_single_pitem():
    """Single-entry ``all_pitem_prose`` → legacy flat layout (common case, avoids header noise)."""
    candidate = {
        "name": "kernel",
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


# PR-B §2: benchmark-cases block in build_prompt
def test_build_benchmark_cases_block_returns_empty_without_task_group():
    """Legacy dispatch (no task_group) → byte-identical output to PR-A."""
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
                "duration_us": 100_000.0,
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
    assert "per_call_ms=1.000000" in block
    assert "bound=memory-bound" in block
    assert "30.00% of 5.3 TB/s" in block


def test_build_benchmark_cases_block_renders_multiple_rows_sorted_by_time():
    """Multi-row groups render rows aggregate-time-descending and say 'optimize once, applies to all'."""
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
    """End-to-end: build_prompt threads the block in when the candidate carries a task_group."""
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


# PR-B §3: bound-keyed optimization priority block in build_prompt
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
    """No top-level bound_type → fall back to the first task_group row's bound_type."""
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


# Defect 1 regression: make_proposal must surface ``artifact_error`` (not "compile failed") when zero backend attempts produced a usable result (geak_dispatch_audit.md Defect 1).
def test_make_proposal_surfaces_backend_dispatch_failure():
    """All dispatch failed (``best`` None) → REVERT reason names the real cause (no usable backend attempt), not compile."""
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
    # The misleading legacy string must NOT appear when we know the real cause.
    assert "compile failed" not in proposal["reasons"][0]


def test_make_proposal_keeps_legacy_compile_failed_when_artifact_lookup_failed():
    """Compile-side regression: attempt produced output but artifact resolution failed → REVERT with legacy 'compile failed'."""
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
    """Belt-and-braces: compile_passed=False with empty artifact_error must not crash and keeps the legacy reason."""
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
