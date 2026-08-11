# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Per-round staging of the kernel agent's KB sub-columns (write side)."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from hyperloom.orchestrator.knowledge.agent_kb import KernelAgentKB
from hyperloom.orchestrator.knowledge.remote_recipe.values import (
    build_kernel_agent_knowledge,
    kernel_agent_canonical_id,
)
from hyperloom.orchestrator.knowledge.kernel_kb_columns import stage_kernel_columns
from hyperloom.orchestrator.knowledge.remote_recipe._vendor.kb_store_client import (
    KnowledgeSections,
)
from hyperloom.orchestrator.knowledge.remote_recipe.values import build_remote_knowledge


def _kb(tmp_path: Path) -> KernelAgentKB:
    return KernelAgentKB(KnowledgeSections(tmp_path / "draft"))


def _staged(kb: KernelAgentKB) -> dict:
    content = kb._sections.staged("kernel")
    return content.knowledge if content is not None else {}


def test_inactive_kb_is_a_no_op(monkeypatch) -> None:
    monkeypatch.delenv("KB_DRAFT_DIR", raising=False)
    monkeypatch.delenv("KB_WARM_START_DIR", raising=False)
    summary = stage_kernel_columns(SimpleNamespace(optimization_stack=[]))
    assert summary == {"active": False}


def test_gemm_column_stages_params_and_tuned_file(tmp_path: Path) -> None:
    tuned = tmp_path / "tuned.json"
    tuned.write_text("{}", encoding="utf-8")
    state = SimpleNamespace(
        optimization_stack=[
            {"action": "gemm_tuning", "tuned_file": str(tuned), "decision": "KEEP"}
        ],
        last_gemm_tuning={"decision": "KEEP", "tuned_file": str(tuned)},
    )
    kb = _kb(tmp_path)

    stage_kernel_columns(state, kb=kb)

    gemm = _staged(kb)["gemm"]
    assert gemm["optimizations"][0]["tuned_file"] == "kernel/gemm/tuned.json"
    assert gemm["files"] == ["kernel/gemm/tuned.json"]
    assert (kb._sections.files_dir / "kernel/gemm/tuned.json").is_file()


def test_rewrite_column_carries_speedup_gain_and_files(tmp_path: Path) -> None:
    patch = tmp_path / "k.diff"
    patch.write_text("diff", encoding="utf-8")
    source = tmp_path / "k.py"
    source.write_text("print(1)", encoding="utf-8")
    state = SimpleNamespace(
        optimization_stack=[
            {
                "action": "integrate",
                "integration_id": "i1",
                "kernel_id": "k1",
                "gain_pct": 5.0,
                "tput": 100.0,
                "patch_path": str(patch),
                "target_file": str(source),
            }
        ],
        kernel_opt_task_attempts={"k1": {"last_micro_speedup": 1.3, "kernel_name": "k1"}},
    )
    kb = _kb(tmp_path)

    stage_kernel_columns(state, kb=kb)

    item = _staged(kb)["rewrite"]["items"][0]
    assert item["speedup"] == 1.3
    assert item["e2e_gain_pct"] == 5.0
    assert item["optimized_throughput"] == 100.0
    assert item["patch"] == "kernel/rewrite/k.diff"
    assert item["source_files"] == ["kernel/rewrite/k.py"]


def test_fusion_column_requires_e2e_keep(tmp_path: Path) -> None:
    patch = tmp_path / "f.diff"
    patch.write_text("diff", encoding="utf-8")
    target = tmp_path / "f.py"
    target.write_text("x=1", encoding="utf-8")
    base = dict(
        optimization_stack=[
            {"action": "fusion", "patch_path": str(patch), "target_file": str(target)}
        ],
        last_fusion={"patch": str(patch), "source_file": str(target)},
    )
    # Not KEEP -> nothing staged.
    kb_revert = _kb(tmp_path / "revert")
    stage_kernel_columns(
        SimpleNamespace(**base, last_fusion_integrate={"decision": "REVERT"}),
        kb=kb_revert,
    )
    assert "fusion" not in _staged(kb_revert)

    kb_keep = _kb(tmp_path / "keep")
    stage_kernel_columns(
        SimpleNamespace(**base, last_fusion_integrate={"decision": "KEEP", "gain_pct": 3.0}),
        kb=kb_keep,
    )
    item = _staged(kb_keep)["fusion"]["items"][0]
    assert item["patch"] == "kernel/fusion/f.diff"
    assert item["source_file"] == "kernel/fusion/f.py"
    assert item["e2e"]["decision"] == "KEEP"


def test_missing_files_skip_the_item_without_raising(tmp_path: Path) -> None:
    state = SimpleNamespace(
        optimization_stack=[
            {
                "action": "integrate",
                "integration_id": "i1",
                "kernel_id": "k1",
                "patch_path": str(tmp_path / "absent.diff"),
                "target_file": str(tmp_path / "absent.py"),
            }
        ],
        kernel_opt_task_attempts={},
    )
    kb = _kb(tmp_path)
    stage_kernel_columns(state, kb=kb)
    assert "rewrite" not in _staged(kb)


def test_staged_columns_overlay_into_close_document(tmp_path: Path) -> None:
    patch = tmp_path / "k.diff"
    patch.write_text("diff", encoding="utf-8")
    source = tmp_path / "k.py"
    source.write_text("print(1)", encoding="utf-8")
    state = SimpleNamespace(
        optimization_stack=[
            {
                "action": "integrate",
                "integration_id": "i1",
                "kernel_id": "k1",
                "gain_pct": 5.0,
                "tput": 130.0,
                "patch_path": str(patch),
                "target_file": str(source),
            }
        ],
        kernel_opt_task_attempts={"k1": {"last_micro_speedup": 1.4, "kernel_name": "k1"}},
        current_best={"tput": 130.0},
        cumulative_gain_validated=30.0,
        session_id="s1",
        recipe_kb_session_id="s1",
    )
    kb = _kb(tmp_path)
    stage_kernel_columns(state, kb=kb)

    bundle = build_remote_knowledge(state, tmp_path / "files", sections=kb._sections)

    rewrite = bundle.knowledge["value"]["kernel"]["rewrite"]["items"][0]
    assert rewrite["speedup"] == 1.4
    assert rewrite["patch"] == "kernel/rewrite/k.diff"
    assert bundle.knowledge["provenance"]["staged_sections"] == ["kernel"]
    assert "kernel/rewrite/k.diff" in {artifact.path for artifact in bundle.artifacts}


def test_kernel_agent_canonical_id_swaps_the_scheme() -> None:
    assert (
        kernel_agent_canonical_id("inference:m:h:vllm:mt:a:0.1:fp8")
        == "kernel:m:h:vllm:mt:a:0.1:fp8"
    )
    # Idempotent, and a bare slug still lands under the kernel scheme.
    assert kernel_agent_canonical_id("kernel:m:h") == "kernel:m:h"
    assert kernel_agent_canonical_id("m:h") == "kernel:m:h"


def test_build_kernel_agent_knowledge_carries_gemm_and_scores_it(tmp_path):
    """The standalone record holds the kernel columns and a kernel-gain score."""
    tuned = tmp_path / "tuned.csv"
    tuned.write_text("M,N,K\n16,512,7168\n", encoding="utf-8")
    state = SimpleNamespace(
        optimization_stack=[
            {
                "action": "gemm_tuning",
                "tuned_file": str(tuned),
                "gain_pct": 15.2,
                "tput": 6638.7,
                "variant_name": "forge_a8w8_blockscale",
            }
        ],
        last_gemm_tuning={"decision": "KEEP", "e2e_gain_pct": 15.2},
        kernel_opt_task_attempts={},
        current_best={"tput": 6638.7},
        cumulative_gain_validated=84.9,
        session_id="s1",
        recipe_kb_session_id="s1",
    )

    bundle, score = build_kernel_agent_knowledge(state, tmp_path / "ka_files")

    value = bundle.knowledge["value"]
    # Columns sit flat on value (not nested under a "kernel" key) so the record
    # is self-describing on its own identity.
    assert set(value) == {"gemm", "fusion", "rewrite"}
    assert len(value["gemm"]["optimizations"]) == 1
    # Scored by kernel gain, not by serving throughput.
    assert score == 15.2
    assert bundle.knowledge["optimized_throughput"] == 15.2
    assert bundle.knowledge["provenance"]["producer"] == "hyperloom-kernel-agent"
    assert {artifact.path for artifact in bundle.artifacts} == {
        "kernel/gemm/artifacts/tuned.csv"
    }
