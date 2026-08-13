# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Per-round staging of the kernel agent's KB sub-columns (write side)."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from hyperloom.orchestrator.knowledge.agent_kb import KernelAgentKB
from hyperloom.orchestrator.knowledge.kernel_kb_columns import stage_kernel_columns
from hyperloom.orchestrator.knowledge.remote_recipe._vendor.kb_store_client import (
    KnowledgeSections,
)
from hyperloom.orchestrator.knowledge.remote_recipe.values import (
    _merge_kernel_section,
    build_remote_knowledge,
)


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


def test_replayed_kernel_artifact_stays_on_same_recipe_page(tmp_path: Path) -> None:
    ref = "kernel/rewrite/warm.py"
    row = {
        "kernel_name": "warm",
        "source_files": [ref],
        "e2e_gain_pct": 5.0,
    }
    warm = tmp_path / "warm"
    source = warm / "files" / ref
    source.parent.mkdir(parents=True)
    source.write_text("optimized", encoding="utf-8")
    (warm / "recipe.json").write_text(
        json.dumps(
            {
                "value": {
                    "kernel": {
                        "rewrite": {"items": [row]},
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    state = SimpleNamespace(
        optimization_stack=[
            {
                "action": "replay_warm_recipe",
                "kernel_replay": {
                    "validation": "combined_recipe_kernel",
                    "count": 1,
                    "columns": ["rewrite"],
                },
            }
        ],
        warm_replay_outcome={
            "status": "reproduced",
            "kernel": {
                "status": "kept",
                "kept": 1,
                "validation": "combined_recipe_kernel",
            },
        },
        warm_kernel_kb_plan=[
            {
                "column": "rewrite",
                "recipe_row": row,
            }
        ],
        current_best={
            "tput": 105.0,
            "extra_server_args": "--warm",
            "extra_envs": {},
        },
        cumulative_gain_validated=5.0,
        kernel_opt_task_attempts={},
        last_gemm_tuning={},
        session_id="same-page",
    )
    sections = KnowledgeSections(
        tmp_path / "draft",
        warm_start_dir=warm,
    )

    bundle = build_remote_knowledge(
        state,
        tmp_path / "published-files",
        sections=sections,
    )

    rewrite = bundle.knowledge["value"]["kernel"]["rewrite"]
    assert rewrite["items"] == [row]
    assert {artifact.path for artifact in bundle.artifacts} == {ref}
    assert (tmp_path / "published-files" / ref).read_text() == "optimized"
    assert bundle.knowledge["optimized_throughput"] == 105.0
    assert "kernel_gain_pct" not in bundle.knowledge


@pytest.mark.parametrize(
    ("column", "list_key", "prior_ref", "new_ref"),
    [
        ("gemm", "optimizations", "kernel/gemm/prior.csv", "kernel/gemm/new.csv"),
        (
            "fusion",
            "items",
            "kernel/fusion/prior.patch",
            "kernel/fusion/new.patch",
        ),
    ],
)
def test_kernel_columns_merge_replayed_and_staged_rows(
    column: str,
    list_key: str,
    prior_ref: str,
    new_ref: str,
) -> None:
    ref_key = "tuned_file" if column == "gemm" else "patch"
    merged = _merge_kernel_section(
        {
            column: {
                list_key: [{"id": "prior", ref_key: prior_ref}],
                "files": [prior_ref],
            }
        },
        {
            column: {
                list_key: [{"id": "new", ref_key: new_ref}],
                "files": [new_ref],
            }
        },
    )

    assert [row["id"] for row in merged[column][list_key]] == ["prior", "new"]
    assert merged[column]["files"] == [new_ref, prior_ref]


def test_new_kernel_row_replaces_same_stable_id_and_prunes_old_ref() -> None:
    merged = _merge_kernel_section(
        {
            "rewrite": {
                "items": [
                    {
                        "id": "same",
                        "patch": "kernel/rewrite/prior.patch",
                    }
                ]
            }
        },
        {
            "rewrite": {
                "items": [
                    {
                        "id": "same",
                        "patch": "kernel/rewrite/new.patch",
                    }
                ]
            }
        },
    )

    assert merged["rewrite"]["items"] == [
        {"id": "same", "patch": "kernel/rewrite/new.patch"}
    ]
    assert merged["rewrite"]["files"] == ["kernel/rewrite/new.patch"]


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

    merged, inherited = merge_kernel_columns(published, incoming)

    assert [i["kernel_name"] for i in merged["rewrite"]["items"]] == ["k1"]
    assert [o["variant_name"] for o in merged["gemm"]["optimizations"]] == ["v1"]
    assert kernel_record_refs(inherited[0]) == set()  # this one names no files


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


def test_merge_hands_back_the_inherited_records_so_their_files_follow():
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

    merged, inherited = merge_kernel_columns(published, incoming)

    # The caller re-uploads and may re-ref these, so they must be the very
    # objects inside the merged document, not copies of them.
    assert inherited == [published["rewrite"]["items"][0]]
    assert inherited[0] is merged["rewrite"]["items"][0]
    assert kernel_record_refs(inherited[0]) == {
        "kernel/rewrite/k1.diff",
        "kernel/rewrite/k1.py",
    }


def test_merge_treats_different_kernels_as_different_slots():
    published = _cols(rewrite=[{"kernel_name": "k1", "e2e_gain_pct": 20.0}])
    incoming = _cols(rewrite=[{"kernel_name": "k2", "e2e_gain_pct": 5.0}])

    merged, _ = merge_kernel_columns(published, incoming)

    assert sorted(i["kernel_name"] for i in merged["rewrite"]["items"]) == ["k1", "k2"]
