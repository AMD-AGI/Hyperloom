# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for framework-agent pure helpers in ``kb`` and ``models``."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
import pytest

from hyperloom.agents.framework.kb import (
    read_pr_ledger,
)
from hyperloom.agents.framework.models import Candidate


# kb.py
def test_read_pr_ledger_tolerates_malformed_rows(tmp_path: Path) -> None:
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
    assert read_pr_ledger(kb_root=tmp_path) == [{"a": 1}, {"b": 2}]
    # Missing file -> empty (cold start).
    assert read_pr_ledger(kb_root=tmp_path / "nope") == []


@pytest.mark.parametrize(
    "env",
    [
        pytest.param({}, id="workspace-default"),
        pytest.param({"USER_DATA_PATH": "/tmp/hl-user-data"}, id="user-data-path"),
        pytest.param({"INFERENCE_OPTIMIZER_FA_KB_PATH": "/tmp/hl-explicit-kb"}, id="explicit-override"),
    ],
)
def test_lessons_writer_and_reader_resolve_the_same_file(
    monkeypatch: pytest.MonkeyPatch,
    env: dict[str, str],
) -> None:
    """The PR ledger must be one file, whatever the deployment sets."""
    from hyperloom.agents.framework import kb as fa_kb
    from hyperloom.orchestrator.knowledge import kb_writeback

    for name in ("USER_DATA_PATH", "INFERENCE_OPTIMIZER_FA_KB_PATH", "FRAMEWORK_AGENT_KB_DIR", "FRAMEWORK_AGENT_ROOT"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)

    writer = kb_writeback._default_kb_root() / kb_writeback.LESSONS_FILE
    reader = fa_kb.framework_optimization_root() / kb_writeback.LESSONS_FILE

    assert writer == reader


def test_framework_kb_does_not_share_a_root_with_the_recipe_kb(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The framework KB must not land on a directory a recipe store owns."""
    from hyperloom.agents.framework import kb as fa_kb
    from hyperloom.inference_optimizer.cli.kb import _legacy_recipe_root, _resolve_local_kb_root

    for name in (
        "INFERENCE_OPTIMIZER_FA_KB_PATH",
        "FRAMEWORK_AGENT_KB_DIR",
        "HYPERLOOM_LOCAL_KB_ROOT",
        "KNOWLEDGE_LOCAL_ROOT",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path / "workspace"))

    framework_root = fa_kb.mutable_kb_root()
    recipe_roots = [
        _legacy_recipe_root(os.environ),
        _resolve_local_kb_root(SimpleNamespace(local_kb_root=None)),
    ]

    for recipe_root in recipe_roots:
        assert framework_root != recipe_root
        assert recipe_root not in framework_root.parents
        assert framework_root not in recipe_root.parents


def test_legacy_kb_dirname_agrees_with_the_recipe_side(monkeypatch, tmp_path: Path) -> None:
    """The framework package hardcodes the legacy root's leaf; it must stay in sync."""
    from hyperloom.agents.framework import kb as fa_kb
    from hyperloom.inference_optimizer.cli.kb import _legacy_recipe_root

    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path / "workspace"))

    assert _legacy_recipe_root(os.environ).name == fa_kb._LEGACY_WORKSPACE_KB_DIRNAME


# models.py
def test_candidate_slug() -> None:
    assert Candidate(ref="feature/Foo@1", repo="r").slug == "feature-foo-1"
    assert Candidate(ref="PR:123", repo="r").slug == "pr-123"
    assert Candidate(ref="release/v0.8.x", repo="r").slug == "release-v0.8.x"
    assert Candidate(ref="!!!", repo="r").slug == "candidate"
