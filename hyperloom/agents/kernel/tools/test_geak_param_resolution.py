"""GEAK parameter resolution — TraceLens-first + profiling timeout.

Pins the contract for three latent bugs surfaced on DeepSeek-R1 +
AITER ``fmha_v3_varlen_fwd`` (Step-5 stall, wrong source override,
PA-benchmark for an MHA kernel):

1. ``_resolve_source_file`` — TraceLens (candidate) wins over the
   Orchestration LLM's ``--source-file`` payload. The LLM occasionally
   confuses kernel IDs (e.g. fmoe k001 vs fmha k003) and supplies a
   path that no longer matches the kernel being optimized; defending
   against this at the kernel-agent boundary keeps the upstream prompt
   surface unchanged.

2. ``_match_benchmark_for_kernel`` — kernel-name-aware reorder of
   ``candidate.benchmark_files`` so semantically-matching tests (e.g.
   ``test_mha*`` for an fmha kernel) head the list. TraceLens lists
   every benchmark known under the repo; without semantic match,
   ``invoke_backend`` picks ``[0]`` and may run a benchmark that
   doesn't even exercise the kernel.

3. ``_profile_timeout_sec`` — bounds each GEAK profiling/benchmark
   subprocess via a ``timeout <N>`` prefix on the rendered
   ``test_command``. Without this, an aiter ``test_pa.py`` no-arg run
   spawns 90 configs × 3 Metrix replays and stalls Step 5 for hours,
   burning the entire GEAK budget before any patch is attempted.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kernel_optimization as ko  # noqa: E402


# ---------------------------------------------------------------------------
# 1. source_file consistency
# ---------------------------------------------------------------------------


def test_source_file_candidate_wins_when_llm_disagrees(tmp_path):
    """LLM passed a *different* path than TraceLens — candidate must win."""
    log_path = tmp_path / "ko.log"
    candidate = {"source_file": "/sgl-workspace/aiter/aiter/ops/mha.py"}

    resolved = ko._resolve_source_file(
        llm_source="/sgl-workspace/aiter/aiter/fused_moe.py",
        candidate=candidate,
        kernel_id="k003",
        log_path=log_path,
    )

    assert resolved == "/sgl-workspace/aiter/aiter/ops/mha.py"
    log_text = log_path.read_text()
    assert "[source-override]" in log_text
    assert "k003" in log_text
    assert "fused_moe.py" in log_text
    assert "mha.py" in log_text


def test_source_file_matching_paths_no_warning(tmp_path):
    """LLM and candidate agree — silent passthrough, no warning."""
    log_path = tmp_path / "ko.log"
    candidate = {"source_file": "/sgl-workspace/aiter/aiter/ops/mha.py"}

    resolved = ko._resolve_source_file(
        llm_source="/sgl-workspace/aiter/aiter/ops/mha.py",
        candidate=candidate,
        kernel_id="k003",
        log_path=log_path,
    )

    assert resolved == "/sgl-workspace/aiter/aiter/ops/mha.py"
    assert not log_path.exists() or "[source-override]" not in log_path.read_text()


def test_source_file_empty_candidate_falls_back_to_llm(tmp_path):
    """Candidate has no source_file (legacy / synthetic) → use LLM's value."""
    log_path = tmp_path / "ko.log"
    candidate: dict = {}

    resolved = ko._resolve_source_file(
        llm_source="/some/llm/path.py",
        candidate=candidate,
        kernel_id="k003",
        log_path=log_path,
    )

    assert resolved == "/some/llm/path.py"
    assert not log_path.exists() or "[source-override]" not in log_path.read_text()


def test_source_file_both_empty_returns_empty(tmp_path):
    log_path = tmp_path / "ko.log"
    resolved = ko._resolve_source_file(
        llm_source="",
        candidate={},
        kernel_id="k003",
        log_path=log_path,
    )
    assert resolved == ""


def test_source_file_log_path_optional():
    """No log_path supplied — must not crash."""
    candidate = {"source_file": "/a/b/mha.py"}
    resolved = ko._resolve_source_file(
        llm_source="/a/b/fused_moe.py",
        candidate=candidate,
        kernel_id="k003",
        log_path=None,
    )
    assert resolved == "/a/b/mha.py"


# ---------------------------------------------------------------------------
# 2. benchmark semantic match
# ---------------------------------------------------------------------------


def test_benchmark_match_fmha_prefers_mha_over_pa():
    """fmha kernel must NOT pick test_pa.py first."""
    bench = [
        "/sgl-workspace/aiter/op_tests/test_pa.py",
        "/sgl-workspace/aiter/op_tests/op_benchmarks/triton/bench_pa_decode.py",
        "/sgl-workspace/aiter/op_tests/test_batch_prefill.py",
        "/sgl-workspace/aiter/op_tests/test_mha_fp8.py",
        "/sgl-workspace/aiter/op_tests/test_mha.py",
    ]
    ordered = ko._match_benchmark_for_kernel(
        "aiter::fmha_v3_varlen_fwd", bench
    )
    assert ordered[0].endswith("test_mha_fp8.py") or ordered[0].endswith("test_mha.py")
    # PA tests must come after MHA tests
    pa_first_idx = next(
        (i for i, p in enumerate(ordered) if Path(p).name.startswith("test_pa")), -1
    )
    mha_first_idx = next(
        (i for i, p in enumerate(ordered) if "mha" in Path(p).name.lower()), -1
    )
    assert mha_first_idx < pa_first_idx, ordered


def test_benchmark_match_paged_attention_prefers_pa():
    """paged_attention kernel should pick test_pa first."""
    bench = [
        "/sgl-workspace/aiter/op_tests/test_mha.py",
        "/sgl-workspace/aiter/op_tests/test_pa.py",
    ]
    ordered = ko._match_benchmark_for_kernel(
        "aiter::paged_attention_v1", bench
    )
    assert ordered[0].endswith("test_pa.py")


def test_benchmark_match_moe_prefers_moe_bench():
    bench = [
        "/repo/test_gemm.py",
        "/repo/test_fused_moe.py",
        "/repo/test_attn.py",
    ]
    ordered = ko._match_benchmark_for_kernel("aiter::fmoe_kernel", bench)
    assert ordered[0].endswith("test_fused_moe.py")


def test_benchmark_match_unknown_kernel_preserves_order():
    """No pattern matches → original order returned."""
    bench = ["/a.py", "/b.py", "/c.py"]
    ordered = ko._match_benchmark_for_kernel("aiter::nothing_known", bench)
    assert ordered == bench


def test_benchmark_match_empty_list_returns_empty():
    assert ko._match_benchmark_for_kernel("anything", []) == []


def test_benchmark_match_filters_non_strings():
    bench = ["/a.py", None, "", "/b.py"]
    ordered = ko._match_benchmark_for_kernel("anything", bench)
    assert ordered == ["/a.py", "/b.py"]


def test_benchmark_match_gemm_prefers_gemm_bench():
    bench = [
        "/repo/test_attn.py",
        "/repo/test_gemm_a8w8.py",
        "/repo/bench_matmul.py",
    ]
    ordered = ko._match_benchmark_for_kernel("gemm_a8w8_blockscale_kernel", bench)
    assert ordered[0].endswith("test_gemm_a8w8.py")


def test_benchmark_match_rmsnorm_prefers_norm_bench():
    bench = [
        "/repo/test_attn.py",
        "/repo/test_rmsnorm.py",
        "/repo/test_layernorm.py",
    ]
    ordered = ko._match_benchmark_for_kernel("aiter::add_rmsnorm", bench)
    assert "norm" in Path(ordered[0]).name.lower()


# ---------------------------------------------------------------------------
# 3. profile timeout
# ---------------------------------------------------------------------------


def test_profile_timeout_default(monkeypatch):
    monkeypatch.delenv("KERNEL_OPT_PROFILE_TIMEOUT_SEC", raising=False)
    assert ko._profile_timeout_sec() == 600


def test_profile_timeout_env_override(monkeypatch):
    monkeypatch.setenv("KERNEL_OPT_PROFILE_TIMEOUT_SEC", "1200")
    assert ko._profile_timeout_sec() == 1200


def test_profile_timeout_malformed_env_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("KERNEL_OPT_PROFILE_TIMEOUT_SEC", "not-a-number")
    assert ko._profile_timeout_sec() == 600


def test_profile_timeout_floor_at_one(monkeypatch):
    monkeypatch.setenv("KERNEL_OPT_PROFILE_TIMEOUT_SEC", "0")
    assert ko._profile_timeout_sec() == 1


def test_render_test_command_singlegpu_wraps_in_timeout(monkeypatch, tmp_path):
    """Single-GPU test_command must be prefixed with ``timeout <N>``."""
    bench = tmp_path / "test_mha.py"
    bench.write_text("# fake")

    cmd = ko._render_geak_test_command(
        kernel_name="aiter::fmha_v3_varlen_fwd",
        bench_files=[str(bench)],
        is_multigpu=False,
        num_gpus=1,
        timeout_sec=300,
    )
    assert cmd == f"timeout 300 python {bench}"


def test_render_test_command_multigpu_wraps_in_timeout(tmp_path):
    bench = tmp_path / "test_collective.py"
    bench.write_text("# fake")

    cmd = ko._render_geak_test_command(
        kernel_name="all_reduce_kernel",
        bench_files=[str(bench)],
        is_multigpu=True,
        num_gpus=4,
        timeout_sec=900,
    )
    assert cmd == f"timeout 900 torchrun --nproc_per_node=4 {bench}"


def test_render_test_command_picks_semantic_match(tmp_path):
    """When multiple benches exist, prefer semantic match."""
    pa = tmp_path / "test_pa.py"
    mha = tmp_path / "test_mha.py"
    pa.write_text("# fake")
    mha.write_text("# fake")

    cmd = ko._render_geak_test_command(
        kernel_name="aiter::fmha_v3_varlen_fwd",
        bench_files=[str(pa), str(mha)],
        is_multigpu=False,
        num_gpus=1,
        timeout_sec=600,
    )
    assert str(mha) in cmd
    assert str(pa) not in cmd


def test_render_test_command_empty_returns_empty_string(tmp_path):
    """No benchmark file at all → empty string (caller treats as no test)."""
    cmd = ko._render_geak_test_command(
        kernel_name="anything",
        bench_files=[],
        is_multigpu=False,
        num_gpus=1,
        timeout_sec=600,
    )
    assert cmd == ""


def test_render_test_command_skips_missing_files(tmp_path):
    bench = tmp_path / "test_mha.py"  # not created
    cmd = ko._render_geak_test_command(
        kernel_name="aiter::fmha",
        bench_files=[str(bench)],
        is_multigpu=False,
        num_gpus=1,
        timeout_sec=600,
    )
    assert cmd == ""


def test_render_test_command_skips_non_test_files(tmp_path):
    helper = tmp_path / "helper.py"
    helper.write_text("# fake")
    cmd = ko._render_geak_test_command(
        kernel_name="aiter::fmha",
        bench_files=[str(helper)],
        is_multigpu=False,
        num_gpus=1,
        timeout_sec=600,
    )
    assert cmd == ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
