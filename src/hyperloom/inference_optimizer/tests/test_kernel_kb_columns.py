# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Per-round staging of the kernel agent's KB sub-columns (write side)."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from hyperloom.orchestrator.knowledge.agent_kb import KernelAgentKB
from hyperloom.orchestrator.knowledge.remote_recipe.values import (
    KERNEL_AGENT_METRIC,
    build_kernel_agent_knowledge,
    kernel_agent_canonical_id,
    merge_kernel_columns,
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


def test_kernel_agent_canonical_id_namespaces_the_producer() -> None:
    # KernelForge publishes its own kernel: records; the producer prefix keeps
    # the two from overwriting each other under a shared slug.
    assert (
        kernel_agent_canonical_id("inference:m:h:vllm:mt:a:0.1:fp8")
        == "kernel:hyperloom-m:h:vllm:mt:a:0.1:fp8"
    )
    assert kernel_agent_canonical_id("m:h") == "kernel:hyperloom-m:h"
    # Idempotent: re-deriving from an already-namespaced id changes nothing.
    once = kernel_agent_canonical_id("inference:m:h")
    assert kernel_agent_canonical_id(once) == once


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
    # Scored by kernel gain, under a metric name that says so.
    assert score == 15.2
    assert bundle.knowledge[KERNEL_AGENT_METRIC] == 15.2
    assert "optimized_throughput" not in bundle.knowledge
    assert bundle.knowledge["provenance"]["producer"] == "hyperloom-kernel-agent"
    assert {artifact.path for artifact in bundle.artifacts} == {
        "kernel/gemm/artifacts/tuned.csv"
    }


def test_stage_gives_a_same_named_artifact_its_own_ref(tmp_path):
    """Two different files sharing a basename must not share a ref.

    The draft only keeps ``{section}/{kind}/{name}`` for the first bytes to
    land; a same-named neighbour gets a digest-suffixed name. Handing the
    second file the first one's ref would republish the wrong artifact.
    """
    kb = _kb(tmp_path)
    first = tmp_path / "a" / "tuned.csv"
    second = tmp_path / "b" / "tuned.csv"
    first.parent.mkdir(parents=True, exist_ok=True)
    second.parent.mkdir(parents=True, exist_ok=True)
    first.write_text("first\n", encoding="utf-8")
    second.write_text("second\n", encoding="utf-8")

    first_ref = kb.write_gemm({"optimizations": []}, files=[first])[0]
    second_ref = kb.write_gemm({"optimizations": []}, files=[second])[0]

    assert first_ref == "kernel/gemm/tuned.csv"
    assert second_ref != first_ref
    files_dir = kb._sections.files_dir
    assert (files_dir / first_ref).read_text(encoding="utf-8") == "first\n"
    assert (files_dir / second_ref).read_text(encoding="utf-8") == "second\n"


def test_restaging_identical_bytes_reuses_the_same_ref(tmp_path):
    kb = _kb(tmp_path)
    artifact = tmp_path / "tuned.csv"
    artifact.write_text("same\n", encoding="utf-8")

    first = kb.write_gemm({"optimizations": []}, files=[artifact])[0]
    again = kb.write_gemm({"optimizations": []}, files=[artifact])[0]

    assert first == again == "kernel/gemm/tuned.csv"


def test_kernel_agent_canonical_id_refuses_an_unknown_workload() -> None:
    # "kernel:" alone is a shared junk identity; an unknown workload publishes
    # nowhere rather than there.
    assert kernel_agent_canonical_id("") == ""
    assert kernel_agent_canonical_id("   ") == ""


def test_restaging_a_displaced_artifact_keeps_its_own_ref(tmp_path):
    """Re-staging the file that lost the plain name must not return that name.

    First bytes win ``kernel/gemm/tuned.csv``; the same-named neighbour is
    stored under a digest name. Re-staging that neighbour adds no new file, and
    guessing "no growth means the plain name" handed it the first file's ref —
    republishing the wrong artifact under its metadata.
    """
    kb = _kb(tmp_path)
    winner = tmp_path / "a" / "tuned.csv"
    displaced = tmp_path / "b" / "tuned.csv"
    winner.parent.mkdir(parents=True, exist_ok=True)
    displaced.parent.mkdir(parents=True, exist_ok=True)
    winner.write_text("winner\n", encoding="utf-8")
    displaced.write_text("displaced\n", encoding="utf-8")

    kb.write_gemm({"optimizations": []}, files=[winner])
    displaced_ref = kb.write_gemm({"optimizations": []}, files=[displaced])[0]
    restaged_ref = kb.write_gemm({"optimizations": []}, files=[displaced])[0]

    assert restaged_ref == displaced_ref
    assert restaged_ref != "kernel/gemm/tuned.csv"
    files_dir = kb._sections.files_dir
    assert (files_dir / restaged_ref).read_text(encoding="utf-8") == "displaced\n"


def _cols(gemm=(), fusion=(), rewrite=()):
    return {
        "gemm": {"optimizations": list(gemm)},
        "fusion": {"items": list(fusion)},
        "rewrite": {"items": list(rewrite)},
    }


def test_merge_keeps_a_column_the_incoming_session_never_touched():
    published = _cols(rewrite=[{"kernel_name": "k1", "e2e_gain_pct": 20.0}])
    incoming = _cols(gemm=[{"variant_name": "v1", "e2e_gain_pct": 30.0}])

    merged, carried = merge_kernel_columns(published, incoming)

    assert [i["kernel_name"] for i in merged["rewrite"]["items"]] == ["k1"]
    assert [o["variant_name"] for o in merged["gemm"]["optimizations"]] == ["v1"]
    assert carried == []  # that rewrite record referenced no artifacts


def test_merge_prefers_the_better_recording_of_the_same_optimization():
    published = _cols(rewrite=[{"kernel_name": "k1", "e2e_gain_pct": 20.0, "id": "old"}])
    incoming = _cols(rewrite=[{"kernel_name": "k1", "e2e_gain_pct": 25.0, "id": "new"}])

    merged, _ = merge_kernel_columns(published, incoming)

    assert [i["id"] for i in merged["rewrite"]["items"]] == ["new"]


def test_merge_declines_a_worse_recording_of_the_same_optimization():
    published = _cols(rewrite=[{"kernel_name": "k1", "e2e_gain_pct": 20.0, "id": "old"}])
    incoming = _cols(rewrite=[{"kernel_name": "k1", "e2e_gain_pct": 5.0, "id": "new"}])

    merged, _ = merge_kernel_columns(published, incoming)

    assert [i["id"] for i in merged["rewrite"]["items"]] == ["old"]
    # Nothing improved, so the caller can tell the write is pointless.
    assert merged == published


def test_merge_reports_the_artifacts_an_inherited_record_still_needs():
    published = _cols(
        rewrite=[
            {
                "kernel_name": "k1",
                "e2e_gain_pct": 20.0,
                "patch": "kernel/rewrite/k1.diff",
                "source_files": ["kernel/rewrite/k1.py"],
            }
        ]
    )
    incoming = _cols(gemm=[{"variant_name": "v1", "e2e_gain_pct": 30.0}])

    _, carried = merge_kernel_columns(published, incoming)

    assert carried == ["kernel/rewrite/k1.diff", "kernel/rewrite/k1.py"]


def test_merge_treats_different_kernels_as_different_slots():
    published = _cols(rewrite=[{"kernel_name": "k1", "e2e_gain_pct": 20.0}])
    incoming = _cols(rewrite=[{"kernel_name": "k2", "e2e_gain_pct": 5.0}])

    merged, _ = merge_kernel_columns(published, incoming)

    assert sorted(i["kernel_name"] for i in merged["rewrite"]["items"]) == ["k1", "k2"]
