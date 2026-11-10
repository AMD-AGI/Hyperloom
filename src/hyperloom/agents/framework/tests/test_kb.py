# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for hyperloom.agents.framework.kb. Hermetic - redirects the KB root via INFERENCE_OPTIMIZER_FA_KB_PATH."""

from __future__ import annotations

from pathlib import Path

import pytest

import hyperloom.agents.framework.kb as kb


# hyperloom.agents.framework.kb module


class TestResolveKbRoot:
    """mutable_kb_root, which has exactly one override and no fallback chain."""

    def test_io_override_matches_writeback_env(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """The fa reader honours the KB override so write/read paths match."""
        monkeypatch.setenv("INFERENCE_OPTIMIZER_FA_KB_PATH", str(tmp_path / "io-kb"))
        assert kb.mutable_kb_root() == tmp_path / "io-kb"

    def test_defaults_to_its_own_workspace_subdirectory(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """With no override the KB lives in its own directory under the workspace."""
        monkeypatch.delenv("INFERENCE_OPTIMIZER_FA_KB_PATH", raising=False)
        monkeypatch.setenv("USER_DATA_PATH", str(tmp_path / "workspace"))
        assert kb.mutable_kb_root() == tmp_path / "workspace" / "framework-kb"

    def test_framework_agent_root_is_not_a_kb_override(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """That variable means "where the skill is installed" and must not move the KB."""
        monkeypatch.delenv("INFERENCE_OPTIMIZER_FA_KB_PATH", raising=False)
        monkeypatch.setenv("FRAMEWORK_AGENT_ROOT", str(tmp_path / "skill"))
        monkeypatch.setenv("USER_DATA_PATH", str(tmp_path / "workspace"))
        assert kb.mutable_kb_root() == tmp_path / "workspace" / "framework-kb"


def test_read_pr_ledger_reads_the_partition_the_writer_uses(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INFERENCE_OPTIMIZER_FA_KB_PATH", str(tmp_path))
    ledger = kb.framework_optimization_root() / kb.LESSONS_FILE
    ledger.parent.mkdir(parents=True)
    ledger.write_text('{"pr_url": "PR-1"}\n', encoding="utf-8")
    assert kb.read_pr_ledger() == [{"pr_url": "PR-1"}]
