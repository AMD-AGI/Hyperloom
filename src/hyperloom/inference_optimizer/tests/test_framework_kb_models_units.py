# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for framework-agent pure helpers in ``kb`` and ``models``."""

from __future__ import annotations

from pathlib import Path
import pytest

from hyperloom.agents.framework.kb import (
    read_pr_ledger,
)
from hyperloom.agents.framework.models import Candidate


# kb.py
def test_read_pr_ledger_tolerates_malformed_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INFERENCE_OPTIMIZER_FA_KB_PATH", str(tmp_path))
    part = tmp_path / "framework_optimization"
    part.mkdir()
    (part / "lessons.jsonl").write_text(
        '{"a": 1}\n'  # valid
        "\n"  # blank -> skipped
        "   \n"  # whitespace -> skipped
        "not json\n"  # malformed -> skipped
        "[1, 2]\n"  # valid json but not a dict -> skipped
        '{"b": 2}\n',
        encoding="utf-8",
    )
    assert read_pr_ledger() == [{"a": 1}, {"b": 2}]
    # Missing file -> empty (cold start).
    monkeypatch.setenv("INFERENCE_OPTIMIZER_FA_KB_PATH", str(tmp_path / "nope"))
    assert read_pr_ledger() == []


@pytest.mark.parametrize("env_name", ["USER_DATA_PATH", "INFERENCE_OPTIMIZER_FA_KB_PATH"])
async def test_lessons_writer_and_reader_resolve_the_same_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    env_name: str,
) -> None:
    """The PR ledger must be one file, whatever the deployment sets."""
    from hyperloom.orchestrator.knowledge import kb_writeback

    for name in ("USER_DATA_PATH", "INFERENCE_OPTIMIZER_FA_KB_PATH", "FRAMEWORK_AGENT_ROOT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(env_name, str(tmp_path))

    await kb_writeback.write_framework_record(
        kb_writeback.FrameworkOutcome("PR-1", "sha", "p.patch", kb_writeback.OUTCOME_INTEGRATED, 1.0, "sess")
    )

    assert [(row["pr_url"], row["outcome"]) for row in read_pr_ledger()] == [("PR-1", "integrated")]


def test_framework_kb_does_not_share_a_root_with_the_recipe_kb(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The framework KB must not land on a directory a recipe store owns."""
    from hyperloom.agents.framework import kb as fa_kb
    from hyperloom.orchestrator.knowledge.config import KnowledgeConfig

    for name in (
        "INFERENCE_OPTIMIZER_FA_KB_PATH",
        "KNOWLEDGE_LOCAL_ROOT",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path / "workspace"))

    framework_root = fa_kb.mutable_kb_root()
    recipe_root = Path(KnowledgeConfig.from_env().local_root)

    assert framework_root != recipe_root
    assert recipe_root not in framework_root.parents
    assert framework_root not in recipe_root.parents


# models.py
def test_candidate_slug() -> None:
    assert Candidate(ref="feature/Foo@1").slug == "feature-foo-1"
    assert Candidate(ref="PR:123").slug == "pr-123"
    assert Candidate(ref="release/v0.8.x").slug == "release-v0.8.x"
    assert Candidate(ref="!!!").slug == "candidate"
