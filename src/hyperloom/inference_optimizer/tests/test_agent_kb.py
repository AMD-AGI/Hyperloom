# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The kernel agent's sub-columns: staging, reading back, and reaching CLOSE."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from hyperloom.orchestrator.knowledge.agent_kb import KernelAgentKB
from hyperloom.orchestrator.knowledge.remote_recipe._vendor.kb_store_client import (
    KnowledgeSections,
)
from hyperloom.orchestrator.knowledge.remote_recipe.values import build_remote_knowledge


def _kb(tmp_path: Path) -> KernelAgentKB:
    return KernelAgentKB(
        KnowledgeSections(tmp_path / "draft", warm_start_dir=tmp_path / "warm")
    )


def _warm_record(tmp_path: Path, value: dict) -> None:
    warm = tmp_path / "warm"
    warm.mkdir(parents=True, exist_ok=True)
    (warm / "recipe.json").write_text(json.dumps({"value": value}), encoding="utf-8")


def _staged(kb: KernelAgentKB) -> dict:
    content = kb._sections.staged("kernel")
    return content.knowledge if content is not None else {}


def test_a_run_without_a_draft_turns_every_call_into_a_no_op(monkeypatch) -> None:
    monkeypatch.delenv("KB_DRAFT_DIR", raising=False)
    monkeypatch.delenv("KB_WARM_START_DIR", raising=False)

    kb = KernelAgentKB.open()

    assert kb.active is False
    assert kb.read_rewrite() == {}
    assert kb.write_rewrite({"items": []}) == []
    assert kb.prior_file("kernel/rewrite/x.diff") is None


def test_each_backend_reads_back_its_own_sub_column(tmp_path: Path) -> None:
    _warm_record(
        tmp_path,
        {
            "kernel": {
                "rewrite": {"items": [{"patch": "kernel/rewrite/a.diff"}]},
                "gemm": {"items": ["tuned"]},
            },
            "explore": {"extra_server_args": "--page-size 32"},
        },
    )
    kb = _kb(tmp_path)

    assert kb.read_rewrite() == {"items": [{"patch": "kernel/rewrite/a.diff"}]}
    assert kb.read_gemm() == {"items": ["tuned"]}
    assert kb.read_fusion() == {}


def test_a_record_without_the_kernel_column_reads_as_a_cold_start(tmp_path: Path) -> None:
    _warm_record(tmp_path, {"explore": {"extra_server_args": "--page-size 32"}})
    kb = _kb(tmp_path)

    assert kb.read_gemm() == {}
    assert kb.read_fusion() == {}
    assert kb.read_rewrite() == {}


def test_writing_one_sub_column_leaves_its_siblings_alone(tmp_path: Path) -> None:
    kb = _kb(tmp_path)

    kb.write_gemm({"items": [1]})
    kb.write_fusion({"items": [2]})
    kb.write_rewrite({"items": [3]})

    assert _staged(kb) == {
        "gemm": {"items": [1]},
        "fusion": {"items": [2]},
        "rewrite": {"items": [3]},
    }


def test_a_write_is_the_whole_picture_of_that_sub_column(tmp_path: Path) -> None:
    kb = _kb(tmp_path)
    kb.write_rewrite({"items": [1], "notes": "first pass"})

    kb.write_rewrite({"items": [2]})

    assert _staged(kb)["rewrite"] == {"items": [2]}


def test_staged_files_come_back_as_refs_in_the_order_they_were_passed(
    tmp_path: Path,
) -> None:
    first = tmp_path / "a.diff"
    first.write_text("diff a", encoding="utf-8")
    second = tmp_path / "b.diff"
    second.write_text("diff b", encoding="utf-8")
    kb = _kb(tmp_path)

    refs = kb.write_rewrite({"items": []}, files=[first, second])

    assert refs == ["kernel/rewrite/a.diff", "kernel/rewrite/b.diff"]
    assert _staged(kb)["rewrite"]["files"] == refs
    assert (kb._sections.files_dir / refs[0]).read_text(encoding="utf-8") == "diff a"


def test_a_ref_can_be_folded_into_the_payload_by_writing_twice(tmp_path: Path) -> None:
    patch = tmp_path / "fix.diff"
    patch.write_text("diff", encoding="utf-8")
    kb = _kb(tmp_path)

    refs = kb.write_rewrite({"items": []}, files=[patch])
    kb.write_rewrite({"items": [{"patch": refs[0], "outcome": "KEEP"}]})

    assert _staged(kb)["rewrite"] == {
        "items": [{"patch": "kernel/rewrite/fix.diff", "outcome": "KEEP"}],
        "files": ["kernel/rewrite/fix.diff"],
    }
    assert (kb._sections.files_dir / refs[0]).is_file()


def test_replacing_knowledge_without_new_files_preserves_existing_refs(
    tmp_path: Path,
) -> None:
    patch = tmp_path / "a.patch"
    patch.write_text("patch", encoding="utf-8")
    kb = _kb(tmp_path)

    kb.write_gemm({"optimizations": [{"id": "g1"}]}, files=[patch])
    kb.write_gemm({"optimizations": [{"id": "g1"}, {"id": "g2"}]})

    assert _staged(kb)["gemm"] == {
        "optimizations": [{"id": "g1"}, {"id": "g2"}],
        "files": ["kernel/gemm/a.patch"],
    }


def test_restaging_the_same_artifact_keeps_one_ref(tmp_path: Path) -> None:
    patch = tmp_path / "fix.diff"
    patch.write_text("diff", encoding="utf-8")
    kb = _kb(tmp_path)

    first = kb.write_rewrite({"items": []}, files=[patch])
    second = kb.write_rewrite({"items": []}, files=[patch])

    assert first == second == ["kernel/rewrite/fix.diff"]
    assert kb._sections.staged("kernel").files == [kb._sections.files_dir / first[0]]


def test_an_unreadable_artifact_yields_an_empty_slot_instead_of_raising(
    tmp_path: Path,
) -> None:
    # Positional: callers fold these back by index, so a failure holds its slot
    # rather than shifting every later artifact one place up.
    good = tmp_path / "good.diff"
    good.write_text("diff", encoding="utf-8")
    kb = _kb(tmp_path)

    refs = kb.write_rewrite({"items": []}, files=[tmp_path / "absent.diff", good])

    assert refs == ["", "kernel/rewrite/good.diff"]


def test_a_prior_ref_resolves_to_its_downloaded_file(tmp_path: Path) -> None:
    downloaded = tmp_path / "warm" / "files" / "kernel" / "rewrite" / "a.diff"
    downloaded.parent.mkdir(parents=True, exist_ok=True)
    downloaded.write_text("prior", encoding="utf-8")
    kb = _kb(tmp_path)

    assert kb.prior_file("kernel/rewrite/a.diff") == downloaded
    assert kb.prior_file("kernel/rewrite/missing.diff") is None


def test_a_ref_may_not_escape_the_downloaded_tree(tmp_path: Path) -> None:
    kb = _kb(tmp_path)

    assert kb.prior_file("../../etc/passwd") is None
    assert kb.prior_file("/etc/passwd") is None
    assert kb.prior_file("") is None


def test_what_the_kernel_agent_records_reaches_the_published_document(
    tmp_path: Path,
) -> None:
    """The wiring that matters: a sub-column and its file survive CLOSE."""
    patch = tmp_path / "rewrite.diff"
    patch.write_text("diff --git a b", encoding="utf-8")
    kb = _kb(tmp_path)
    refs = kb.write_rewrite({"items": []}, files=[patch])
    kb.write_rewrite(
        {"items": [{"patch": refs[0], "outcome": "KEEP", "gain_pct": 12.4}]},
    )
    state = SimpleNamespace(
        optimization_stack=[],
        current_best={"tput": 130.0},
        cumulative_gain_validated=30.0,
        session_id="session-1",
        recipe_kb_session_id="session-1",
    )

    bundle = build_remote_knowledge(state, tmp_path / "files", sections=kb._sections)

    kernel = bundle.knowledge["value"]["kernel"]
    assert kernel["rewrite"]["items"] == [
        {"patch": "kernel/rewrite/rewrite.diff", "outcome": "KEEP", "gain_pct": 12.4}
    ]
    assert set(kernel) == {"gemm", "fusion", "rewrite"}
    assert bundle.knowledge["provenance"]["staged_sections"] == ["kernel"]
    assert refs[0] in {artifact.path for artifact in bundle.artifacts}
