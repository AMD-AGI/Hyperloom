# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A producer's write must be found by a consumer's read of the same kernel.

This is the capstone for cross-repo reuse: the two sides run the real identity
resolution, differ in workspace layout and kernel path, and must still land on
one address. Only the store is local; nothing about the identity is mocked.

A miss must therefore mean the kernel really is a different one -- a different
architecture, a different framework -- and never merely a different checkout.
"""

from __future__ import annotations

import pytest

from kernelforge.config import Config
from kernelforge.knowledge.experience_reader import read_top_solutions
from kernelforge.knowledge.experience_sink import write_run_experience
from kernelforge.knowledge.experience_store import KnowledgeConfig

TRITON_SRC = "import triton\n@triton.jit\ndef fused_moe_kernel(x):\n    return x\n"
DIFF = """diff --git a/vllm/model_executor/fused_moe.py b/vllm/model_executor/fused_moe.py
--- a/vllm/model_executor/fused_moe.py
+++ b/vllm/model_executor/fused_moe.py
@@ -1 +1 @@
-old
+new
"""
SUMMARY = {
    "category": "moe",
    "strategy": "tile the K loop",
    "recipe": "Use larger BLOCK_K.",
    "lessons": "Watch occupancy.",
}


@pytest.fixture()
def knowledge_root(tmp_path):
    """One store both sides address, standing in for the shared deployment."""
    return tmp_path / "knowledge"


def _config(workspace, knowledge_root, gpu_type="mi355x"):
    knowledge = KnowledgeConfig.from_env({}, mode="local", local_root=knowledge_root)
    return Config.from_env(
        workspace=str(workspace),
        gpu_target="gfx950",
        gpu_type=gpu_type,
        knowledge_config=knowledge,
        agent_precheck=False,
    )


def _write_source(root, relative, source=TRITON_SRC):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    return path


def _producer_write(tmp_path, knowledge_root, *, experiment_id="producer-0731", best_wall_ms=5.0, gpu_type="mi355x"):
    workspace = tmp_path / "producer"
    kernel = _write_source(workspace, "vllm/model_executor/fused_moe.py")
    return write_run_experience(
        config=_config(workspace, knowledge_root, gpu_type),
        workspace=str(workspace),
        kernel_path=str(kernel),
        kernel_source=TRITON_SRC,
        kernel_backend="triton",
        gpu_target="gfx950",
        experiment_id=experiment_id,
        baseline_wall_ms=10.0,
        best_wall_ms=best_wall_ms,
        mean_case_speedup=10.0 / best_wall_ms,
        cumulative_diff=DIFF.replace("+new", f"+{experiment_id}"),
        digest="digest",
        framework="vllm",
        summary_override=SUMMARY,
    )


def _consumer_read(tmp_path, knowledge_root, *, top_k=3, gpu_type="mi355x"):
    """A different workspace layout for the same framework file."""
    workspace = tmp_path / "consumer" / "worktree"
    kernel = _write_source(workspace, "vllm/model_executor/fused_moe.py")
    return read_top_solutions(
        config=_config(workspace, knowledge_root, gpu_type),
        kernel_path=str(kernel),
        kernel_source=TRITON_SRC,
        kernel_backend="triton",
        framework="vllm",
        top_k=top_k,
    )


def test_write_and_read_resolve_the_same_address_across_workspaces(tmp_path, knowledge_root):
    status = _producer_write(tmp_path, knowledge_root)

    assert status["written"] is True
    # Framework-explicit, and the operator drops its ``_kernel`` suffix so a
    # source symbol and a trace name converge on one address.
    assert status["kernel"].startswith("kernel:forge-loop:fused_moe:vllm:")
    assert status["kernel"].endswith(":triton:mi355x")

    solutions = _consumer_read(tmp_path, knowledge_root)

    assert solutions, "the consumer read must find the producer's record"
    assert solutions[0]["kernel_slug"] == status["kernel"]
    assert solutions[0]["patch_content"] == DIFF.replace("+new", "+producer-0731")
    assert solutions[0]["strategy"] == "tile the K loop"


def test_a_different_gpu_model_is_a_real_mismatch(tmp_path, knowledge_root):
    """Not transferable, so it must not be offered -- and not merely filtered.

    Both models here build for the same target, so an address keyed by the
    compilation target would hand one card's recipe to the other.
    """
    _producer_write(tmp_path, knowledge_root, gpu_type="mi355x")

    assert _consumer_read(tmp_path, knowledge_root, gpu_type="mi300x") == []


def test_framework_follows_the_defining_file_across_packages(tmp_path, knowledge_root):
    """The anchor only calls the kernel; the owner is where it is defined.

    Both sides must agree on that, or a vLLM entry point calling an aiter kernel
    would be filed under one framework and looked up under another.
    """
    workspace = tmp_path / "shared"
    aiter_file = _write_source(
        workspace,
        "aiter/ops/triton/unified.py",
        TRITON_SRC.replace("fused_moe_kernel", "unified_attention_kernel"),
    )
    entry_src = "def unified_attention(x):\n    return call_aiter(x)\n"
    vllm_entry = _write_source(workspace, "vllm/attention/entry.py", entry_src)
    config = _config(workspace, knowledge_root)

    status = write_run_experience(
        config=config,
        workspace=str(workspace),
        kernel_path=str(vllm_entry),
        kernel_source=entry_src,
        kernel_backend="triton",
        gpu_target="gfx950",
        experiment_id="producer-x",
        baseline_wall_ms=10.0,
        best_wall_ms=5.0,
        mean_case_speedup=2.0,
        cumulative_diff=DIFF,
        digest="d",
        source_files=[str(aiter_file)],
        target_functions=["unified_attention_kernel"],
        summary_override=SUMMARY,
    )

    assert status["written"] is True
    assert status["kernel"].startswith("kernel:forge-loop:unified_attention:aiter:")

    solutions = read_top_solutions(
        config=config,
        kernel_path=str(vllm_entry),
        kernel_source=entry_src,
        kernel_backend="triton",
        target_functions=["unified_attention_kernel"],
        source_files=[str(aiter_file)],
        top_k=3,
    )

    assert solutions, "the defining file must lead both sides to one address"
    assert solutions[0]["kernel_slug"] == status["kernel"]


def test_several_producer_runs_come_back_ranked_by_speedup(tmp_path, knowledge_root):
    _producer_write(tmp_path, knowledge_root, experiment_id="slow", best_wall_ms=8.0)
    _producer_write(tmp_path, knowledge_root, experiment_id="fast", best_wall_ms=2.0)
    _producer_write(tmp_path, knowledge_root, experiment_id="mid", best_wall_ms=5.0)

    solutions = _consumer_read(tmp_path, knowledge_root, top_k=3)

    assert [round(s["speedup"], 3) for s in solutions] == [5.0, 2.0, 1.25]
    assert len({s["solution_slug"] for s in solutions}) == 3
