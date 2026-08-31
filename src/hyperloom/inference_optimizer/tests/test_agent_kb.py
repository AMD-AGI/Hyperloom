# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Kernel AgentKB prior-column reads."""

from __future__ import annotations

import json
from pathlib import Path

from hyperloom.orchestrator.knowledge.agent_kb import KernelAgentKB
from hyperloom.orchestrator.knowledge.remote_recipe._vendor.kb_store_client import (
    KnowledgeSections,
)


def _kb(tmp_path: Path) -> KernelAgentKB:
    return KernelAgentKB(KnowledgeSections(tmp_path / "draft", warm_start_dir=tmp_path / "warm"))


def _warm_record(tmp_path: Path, value: dict) -> None:
    warm = tmp_path / "warm"
    warm.mkdir(parents=True, exist_ok=True)
    (warm / "recipe.json").write_text(json.dumps({"value": value}), encoding="utf-8")


def test_a_run_without_a_draft_turns_every_call_into_a_no_op(monkeypatch) -> None:
    monkeypatch.delenv("KB_DRAFT_DIR", raising=False)
    monkeypatch.delenv("KB_WARM_START_DIR", raising=False)

    kb = KernelAgentKB.open()

    assert kb.active is False
    assert kb.read_rewrite() == {}
    assert kb.prior_file("kernel/rewrite/x.diff") is None


def test_each_backend_reads_back_its_own_sub_column(tmp_path: Path) -> None:
    _warm_record(
        tmp_path,
        {
            "kernel": {
                "rewrite": {"items": [{"patch": "kernel/rewrite/a.diff"}]},
                "gemm": {"items": ["tuned"]},
            },
            "config": {"extra_server_args": "--page-size 32"},
        },
    )
    kb = _kb(tmp_path)

    assert kb.read_rewrite() == {"items": [{"patch": "kernel/rewrite/a.diff"}]}
    assert kb.read_gemm() == {"items": ["tuned"]}
    assert kb.read_fusion() == {}


def test_a_record_without_the_kernel_column_reads_as_a_cold_start(tmp_path: Path) -> None:
    _warm_record(tmp_path, {"config": {"extra_server_args": "--page-size 32"}})
    kb = _kb(tmp_path)

    assert kb.read_gemm() == {}
    assert kb.read_fusion() == {}
    assert kb.read_rewrite() == {}


def test_a_prior_ref_resolves_to_its_downloaded_file(tmp_path: Path) -> None:
    downloaded = tmp_path / "warm" / "files" / "kernel" / "rewrite" / "a.diff"
    downloaded.parent.mkdir(parents=True, exist_ok=True)
    downloaded.write_text("prior", encoding="utf-8")
    kb = _kb(tmp_path)

    assert kb.prior_file("kernel/rewrite/a.diff") == downloaded
    assert kb.prior_file("kernel/rewrite/missing.diff") is None
