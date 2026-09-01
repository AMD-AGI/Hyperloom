"""Regression tests for local scratch placement, sweeping and the scratch filters."""

from __future__ import annotations

import subprocess
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
    # _local_scratch_dir creates the session dir and writes an owner file for
    # the current process, which is alive by definition.
    live = _durable_attempt(tmp_path, "session-live", "matmul")
    live_scratch = forge_submit._local_scratch_dir(live)
    live_scratch.mkdir(parents=True)
    (live_scratch / "kernel.py").write_text("x = 1\n")

    forge_submit._local_scratch_dir(_durable_attempt(tmp_path, "session-new", "softmax"))

    assert (live_scratch / "kernel.py").is_file()
    assert local_root / "session-live" in list(local_root.iterdir())


def test_the_sweep_removes_a_session_with_no_owner_marker(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _on_network_fs(monkeypatch, True)
    local_root = _local_root(monkeypatch, tmp_path)
    # A session whose owner file is absent is treated as dead (e.g. SIGKILL
    # before _write_scratch_owner completed, or left from an old code version).
    dead_scratch = local_root / "session-dead" / "matmul" / "worktree"
    dead_scratch.mkdir(parents=True)

    forge_submit._local_scratch_dir(_durable_attempt(tmp_path, "session-new", "softmax"))

    assert not (local_root / "session-dead").exists()


def test_the_sweep_removes_a_session_with_a_dead_pid(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _on_network_fs(monkeypatch, True)
    local_root = _local_root(monkeypatch, tmp_path)
    dead_dir = local_root / "session-dead"
    dead_scratch = dead_dir / "matmul" / "worktree"
    dead_scratch.mkdir(parents=True)
    # PID 2^22 - 1 is the highest possible Linux pid and is virtually never live.
    forge_submit._scratch_owner_file(dead_dir).write_text("4194303:0", encoding="ascii")

    forge_submit._local_scratch_dir(_durable_attempt(tmp_path, "session-new", "softmax"))

    assert not dead_dir.exists()


def test_the_sweep_spares_a_session_owned_by_another_user(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """EPERM from os.kill proves the pid exists; it must not be read as dead."""
    _on_network_fs(monkeypatch, True)
    local_root = _local_root(monkeypatch, tmp_path)
    foreign = local_root / "session-foreign"
    (foreign / "matmul" / "worktree").mkdir(parents=True)
    forge_submit._scratch_owner_file(foreign).write_text("4242:0", encoding="ascii")

    def _kill(_pid: int, _sig: int) -> None:
        raise PermissionError

    monkeypatch.setattr(forge_submit.os, "kill", _kill)

    forge_submit._local_scratch_dir(_durable_attempt(tmp_path, "session-new", "softmax"))

    assert foreign.is_dir()


def test_the_sweep_leaves_unrelated_directories_alone(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _on_network_fs(monkeypatch, True)
    local_root = _local_root(monkeypatch, tmp_path)
    local_root.mkdir(parents=True)
    stranger = local_root / "not-ours"
    stranger.mkdir()

    forge_submit._local_scratch_dir(_durable_attempt(tmp_path, "session-new", "softmax"))

    assert stranger.is_dir()


def test_the_scratch_copy_keeps_what_a_package_is_imported_through(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The copy shadows the install, so a dropped name has nothing to fall back to."""
    _on_network_fs(monkeypatch, False)
    pkg = tmp_path / "site-packages" / "aiter"
    (pkg / "jit").mkdir(parents=True)
    (pkg / "dist").mkdir()
    (pkg / "__pycache__").mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "jit" / "core.py").write_text("x = 1\n", encoding="utf-8")
    (pkg / "jit" / "extension.so").write_bytes(b"\x7fELF")
    (pkg / "dist" / "shard.py").write_text("y = 2\n", encoding="utf-8")
    (pkg / "stale.pyc").write_bytes(b"\x00")

    prepared = forge_submit._prepare_worktree_nogit(
        str(pkg / "__init__.py"),
        "",
        _durable_attempt(tmp_path, "session-copy", "aiter"),
        "forge/scratch",
    )

    assert prepared is not None
    copied = Path(prepared[0]) / "aiter"
    assert (copied / "jit" / "core.py").is_file()
    assert (copied / "jit" / "extension.so").is_file()
    assert (copied / "dist" / "shard.py").is_file()
    assert not (copied / "__pycache__").exists()
    assert not (copied / "stale.pyc").exists()


def test_the_scratch_exclude_omits_compiled_artefacts(tmp_path: Path) -> None:
    """Compiled artefacts stay in the index so git-revert can restore them."""
    workspace = tmp_path / "scratch"
    workspace.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=workspace, check=True)

    forge_submit._exclude_runtime_artifacts(workspace)

    written = (workspace / ".git" / "info" / "exclude").read_text(encoding="utf-8").split()
    assert "*.so" not in written
    assert "build/" not in written
    assert "__pycache__/" in written
