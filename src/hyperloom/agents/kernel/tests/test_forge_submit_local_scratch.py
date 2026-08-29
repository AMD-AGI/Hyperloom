"""Regression tests for local scratch placement, sweeping and the copy filter."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_BACKENDS_DIR = Path(__file__).resolve().parent.parent / "tools" / "backends"
sys.path.insert(0, str(_BACKENDS_DIR))
import forge_submit  # noqa: E402


def _durable_attempt(tmp_path: Path, session: str, attempt: str) -> Path:
    output_dir = tmp_path / "sessions" / "forge" / session / attempt
    output_dir.mkdir(parents=True)
    return output_dir


def _on_network_fs(monkeypatch: pytest.MonkeyPatch, value: bool) -> None:
    monkeypatch.setattr(forge_submit, "_is_on_network_fs", lambda _path: value)


def _local_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    root = tmp_path / "local"
    monkeypatch.setenv("FORGE_LOCAL_SCRATCH_ROOT", str(root))
    return root


def test_local_scratch_stays_beside_the_archive_off_a_network_mount(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _on_network_fs(monkeypatch, False)
    output_dir = _durable_attempt(tmp_path, "session-a", "matmul")

    assert forge_submit._local_scratch_dir(output_dir) == output_dir / "worktree"


def test_two_sessions_on_one_kernel_get_separate_local_scratch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _on_network_fs(monkeypatch, True)
    local_root = _local_root(monkeypatch, tmp_path)
    first = forge_submit._local_scratch_dir(_durable_attempt(tmp_path, "session-a", "matmul"))
    second = forge_submit._local_scratch_dir(_durable_attempt(tmp_path, "session-b", "matmul"))

    assert first != second
    assert first == local_root / "session-a" / "matmul" / "worktree"
    assert second == local_root / "session-b" / "matmul" / "worktree"


def test_the_sweep_spares_a_session_that_is_still_running(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _on_network_fs(monkeypatch, True)
    local_root = _local_root(monkeypatch, tmp_path)
    live = _durable_attempt(tmp_path, "session-live", "matmul")
    live_scratch = forge_submit._local_scratch_dir(live)
    live_scratch.mkdir(parents=True)
    (live_scratch / "kernel.py").write_text("x = 1\n")

    forge_submit._local_scratch_dir(_durable_attempt(tmp_path, "session-new", "softmax"))

    assert (live_scratch / "kernel.py").is_file()
    assert local_root / "session-live" in list(local_root.iterdir())


def test_the_sweep_removes_a_session_whose_archive_is_gone(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _on_network_fs(monkeypatch, True)
    local_root = _local_root(monkeypatch, tmp_path)
    dead_scratch = local_root / "session-dead" / "matmul" / "worktree"
    dead_scratch.mkdir(parents=True)

    forge_submit._local_scratch_dir(_durable_attempt(tmp_path, "session-new", "softmax"))

    assert not (local_root / "session-dead").exists()


def test_the_sweep_leaves_unrelated_directories_alone(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _on_network_fs(monkeypatch, True)
    local_root = _local_root(monkeypatch, tmp_path)
    local_root.mkdir(parents=True)
    stranger = local_root / "not-ours"
    stranger.mkdir()

    forge_submit._local_scratch_dir(_durable_attempt(tmp_path, "session-new", "softmax"))

    assert stranger.is_dir()


def test_the_copy_filter_keeps_what_a_package_is_imported_through() -> None:
    directories, suffixes = forge_submit._runtime_artifact_names()

    assert "jit" not in directories
    assert "dist" not in directories
    assert ".so" not in suffixes


def test_the_git_exclude_still_covers_extension_modules() -> None:
    assert "*.so" in forge_submit._runtime_artifact_globs()


def test_the_fallback_lists_answer_the_same_way_the_producer_does() -> None:
    assert "jit" not in forge_submit._FALLBACK_RUNTIME_DIRECTORY_NAMES
    assert "dist" not in forge_submit._FALLBACK_RUNTIME_DIRECTORY_NAMES
    assert ".so" not in forge_submit._FALLBACK_RUNTIME_FILE_SUFFIXES
    assert ".so" in forge_submit._FALLBACK_COMPILED_FILE_SUFFIXES
