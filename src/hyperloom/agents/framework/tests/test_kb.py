# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for hyperloom.agents.framework.kb. Hermetic - redirects the KB root via INFERENCE_OPTIMIZER_FA_KB_PATH."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import hyperloom.agents.framework.kb as kb


@pytest.fixture
def kb_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point KB resolution at a clean tmp_path for every test."""
    monkeypatch.setenv("INFERENCE_OPTIMIZER_FA_KB_PATH", str(tmp_path))
    monkeypatch.delenv("FRAMEWORK_AGENT_KB_DIR", raising=False)
    return tmp_path


# hyperloom.agents.framework.kb module


class TestResolveKbRoot:
    """_resolve_kb_root, which has exactly one override and no fallback chain."""

    def test_io_override_matches_writeback_env(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """The fa reader honours the KB override so write/read paths match."""
        monkeypatch.delenv("FRAMEWORK_AGENT_KB_DIR", raising=False)
        monkeypatch.setenv("INFERENCE_OPTIMIZER_FA_KB_PATH", str(tmp_path / "io-kb"))
        assert kb._resolve_kb_root() == tmp_path / "io-kb"

    def test_defaults_to_its_own_workspace_subdirectory(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """With no override the KB lives in its own directory under the workspace."""
        monkeypatch.delenv("FRAMEWORK_AGENT_KB_DIR", raising=False)
        monkeypatch.delenv("INFERENCE_OPTIMIZER_FA_KB_PATH", raising=False)
        monkeypatch.setenv("USER_DATA_PATH", str(tmp_path / "workspace"))
        assert kb._resolve_kb_root() == tmp_path / "workspace" / "framework-kb"

    def test_withdrawn_override_is_ignored_not_raised(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Resolution stays total; rejecting the withdrawn override is start-up's job."""
        monkeypatch.setenv("FRAMEWORK_AGENT_KB_DIR", str(tmp_path / "legacy"))
        monkeypatch.setenv("INFERENCE_OPTIMIZER_FA_KB_PATH", str(tmp_path / "io-kb"))
        assert kb._resolve_kb_root() == tmp_path / "io-kb"

    def test_framework_agent_root_is_not_a_kb_override(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """That variable means "where the skill is installed" and must not raise."""
        monkeypatch.delenv("FRAMEWORK_AGENT_KB_DIR", raising=False)
        monkeypatch.delenv("INFERENCE_OPTIMIZER_FA_KB_PATH", raising=False)
        monkeypatch.setenv("FRAMEWORK_AGENT_ROOT", str(tmp_path / "skill"))
        monkeypatch.setenv("USER_DATA_PATH", str(tmp_path / "workspace"))
        assert kb._resolve_kb_root() == tmp_path / "workspace" / "framework-kb"


class TestCheckKbConfiguration:
    """Start-up reporting for a KB variable this build no longer reads."""

    def test_withdrawn_override_is_reported_not_enforced(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A deployment still exporting it is told, and still runs."""
        monkeypatch.setenv("FRAMEWORK_AGENT_KB_DIR", str(tmp_path / "legacy"))
        monkeypatch.setenv("USER_DATA_PATH", str(tmp_path / "workspace"))

        with caplog.at_level("WARNING"):
            kb.check_kb_configuration()

        # Names the replacement and where the KB actually resolved, so the operator can tell whether their intent was
        # met.
        assert "INFERENCE_OPTIMIZER_FA_KB_PATH" in caplog.text
        assert str(tmp_path / "workspace" / "framework-kb") in caplog.text

    def test_blank_value_says_nothing(self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
        """An exported-but-empty variable expresses no intent, so it is not news."""
        monkeypatch.setenv("FRAMEWORK_AGENT_KB_DIR", "   ")

        with caplog.at_level("WARNING"):
            kb.check_kb_configuration()

        assert caplog.text == ""

    def test_silent_when_unset(self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
        monkeypatch.delenv("FRAMEWORK_AGENT_KB_DIR", raising=False)

        with caplog.at_level("WARNING"):
            kb.check_kb_configuration()

        assert caplog.text == ""

    def test_start_up_never_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The whole hook is total, not merely each step as written today."""

        def _explode() -> None:
            raise RuntimeError("a step nobody expected to fail")

        monkeypatch.setattr(kb, "check_kb_configuration", _explode)
        monkeypatch.setattr(kb, "migrate_legacy_partition_once", _explode)

        kb.prepare_kb_environment()


class TestMigrateLegacyPartition:
    """The one-time carry-over from ``<workspace>/kb`` to ``<workspace>/framework-kb``."""

    @pytest.fixture(autouse=True)
    def _clean_env(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        monkeypatch.delenv("FRAMEWORK_AGENT_KB_DIR", raising=False)
        monkeypatch.delenv("INFERENCE_OPTIMIZER_FA_KB_PATH", raising=False)
        monkeypatch.setenv("USER_DATA_PATH", str(tmp_path / "workspace"))
        return tmp_path

    @staticmethod
    def _seed_legacy(tmp_path: Path, body: str = '{"pr_url": "PR-1"}') -> Path:
        legacy = tmp_path / "workspace" / "kb" / "framework_optimization"
        legacy.mkdir(parents=True)
        (legacy / "lessons.jsonl").write_text(body, encoding="utf-8")
        return legacy

    def test_carries_the_ledger_to_the_new_root(self, tmp_path: Path) -> None:
        """The writer was already working, so a real deployment has data to move."""
        self._seed_legacy(tmp_path)

        destination = kb.migrate_legacy_partition_once()

        assert destination == tmp_path / "workspace" / "framework-kb" / "framework_optimization"
        assert (destination / "lessons.jsonl").read_text(encoding="utf-8") == '{"pr_url": "PR-1"}'
        assert kb.read_pr_ledger() == [{"pr_url": "PR-1"}]

    def test_adds_nothing_of_its_own_to_the_partition(self, tmp_path: Path) -> None:
        """The partition is a served KB domain, so the migration may not litter it."""
        self._seed_legacy(tmp_path)

        destination = kb.migrate_legacy_partition_once()

        assert destination is not None
        assert sorted(p.name for p in destination.iterdir()) == ["lessons.jsonl"]
        assert sorted(p.name for p in destination.iterdir()) == ["lessons.jsonl"]

    def test_copies_links_as_links(self, tmp_path: Path) -> None:
        """Following a link would pull outside content into a directory the KB serves."""
        legacy = self._seed_legacy(tmp_path)
        outside = tmp_path / "outside.md"
        outside.write_text("not mine", encoding="utf-8")
        try:
            (legacy / "link.md").symlink_to(outside)
        except (OSError, NotImplementedError):
            pytest.skip("symlinks unavailable on this platform/account")

        destination = kb.migrate_legacy_partition_once()

        assert destination is not None
        assert (destination / "link.md").is_symlink()

    def test_leaves_the_source_in_place(self, tmp_path: Path) -> None:
        """A copy, not a move: the operator decides when the old root goes."""
        legacy = self._seed_legacy(tmp_path)

        kb.migrate_legacy_partition_once()

        assert (legacy / "lessons.jsonl").exists()

    def test_is_idempotent(self, tmp_path: Path) -> None:
        self._seed_legacy(tmp_path)

        assert kb.migrate_legacy_partition_once() is not None
        assert kb.migrate_legacy_partition_once() is None

    def test_never_overwrites_a_live_partition(self, tmp_path: Path) -> None:
        """A destination already in use wins; the migration is not a repair tool."""
        self._seed_legacy(tmp_path, '{"pr_url": "OLD"}')
        live = tmp_path / "workspace" / "framework-kb" / "framework_optimization"
        live.mkdir(parents=True)
        (live / "lessons.jsonl").write_text('{"pr_url": "LIVE"}', encoding="utf-8")

        assert kb.migrate_legacy_partition_once() is None
        assert kb.read_pr_ledger() == [{"pr_url": "LIVE"}]

    def test_skipped_when_the_operator_named_a_root(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """The legacy default was never the operator's location to inherit."""
        self._seed_legacy(tmp_path)
        monkeypatch.setenv("INFERENCE_OPTIMIZER_FA_KB_PATH", str(tmp_path / "chosen"))

        assert kb.migrate_legacy_partition_once() is None

    def test_no_legacy_data_is_not_an_error(self, tmp_path: Path) -> None:
        assert kb.migrate_legacy_partition_once() is None


class TestMigrationCannotStopTheRun:
    """A convenience copy must never be able to take a session down with it."""

    @pytest.fixture(autouse=True)
    def _legacy_data(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("FRAMEWORK_AGENT_KB_DIR", raising=False)
        monkeypatch.delenv("INFERENCE_OPTIMIZER_FA_KB_PATH", raising=False)
        monkeypatch.setenv("USER_DATA_PATH", str(tmp_path / "workspace"))
        legacy = tmp_path / "workspace" / "kb" / "framework_optimization"
        legacy.mkdir(parents=True)
        (legacy / "lessons.jsonl").write_text('{"pr_url": "PR-1"}', encoding="utf-8")

    @staticmethod
    def _break_copy(monkeypatch: pytest.MonkeyPatch, exc: BaseException) -> None:
        def _raise(*_a: object, **_k: object) -> None:
            raise exc

        monkeypatch.setattr(kb.shutil, "copytree", _raise)

    def test_io_failure_warns_and_returns_none(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        self._break_copy(monkeypatch, OSError(28, "No space left on device"))

        with caplog.at_level("WARNING"):
            assert kb.migrate_legacy_partition_once() is None

        assert "could not carry the legacy partition over" in caplog.text

    def test_permission_failure_warns_and_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._break_copy(monkeypatch, PermissionError(13, "Permission denied"))

        assert kb.migrate_legacy_partition_once() is None

    def test_unexpected_failure_still_does_not_escape(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Not just OSError: nothing this helper can hit is worth the session."""
        self._break_copy(monkeypatch, ValueError("something nobody predicted"))

        assert kb.migrate_legacy_partition_once() is None

    def test_start_up_still_completes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The whole start-up sequence, not just the helper, survives it."""
        self._break_copy(monkeypatch, OSError(28, "No space left on device"))

        kb.prepare_kb_environment()

    def test_a_failed_copy_leaves_nothing_behind(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """No staging dir survives a failed atomic copy."""
        self._break_copy(monkeypatch, OSError(28, "No space left on device"))

        kb.migrate_legacy_partition_once()

        workspace = tmp_path / "workspace"
        assert not [p for p in workspace.glob("*.migrating*")]
        kb_root = kb.mutable_kb_root()
        assert not kb_root.is_dir() or not any(kb_root.iterdir())

    def test_staging_is_unique_and_outside_the_kb_root(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Concurrent start-ups must not stage onto one another's directory."""
        seen: list[Path] = []
        real_copytree = kb.shutil.copytree

        def _record(src: object, dst: object, *a: object, **k: object) -> object:
            seen.append(Path(str(dst)))
            return real_copytree(src, dst, *a, **k)  # type: ignore[arg-type]

        monkeypatch.setattr(kb.shutil, "copytree", _record)

        kb.migrate_legacy_partition_once()

        assert len(seen) == 1
        staging = seen[0]
        assert str(os.getpid()) in staging.name
        assert kb.mutable_kb_root() not in staging.parents
        assert staging.parent == tmp_path / "workspace"

    def test_the_loser_of_a_race_gives_up_quietly(self, tmp_path: Path) -> None:
        """Second writer finds the destination populated and stands down."""
        assert kb.migrate_legacy_partition_once() is not None
        assert kb.migrate_legacy_partition_once() is None
        assert kb.read_pr_ledger() == [{"pr_url": "PR-1"}]

