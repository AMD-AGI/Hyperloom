# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for the JIT-rebuild safety net (loop/jit_rebuild.py)."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from kernelforge.llm.git import GitError
from kernelforge.loop.jit_rebuild import (
    JitRebuildUnavailable,
    force_jit_rebuild,
    force_jit_rebuild_for_changes,
    tracked_source_changes,
)


@pytest.fixture(autouse=True)
def _isolate_aiter_root_dir():
    """Snapshot and restore ``AITER_ROOT_DIR`` around every test in this module."""
    original = os.environ.get("AITER_ROOT_DIR")
    try:
        yield
    finally:
        if original is None:
            os.environ.pop("AITER_ROOT_DIR", None)
        else:
            os.environ["AITER_ROOT_DIR"] = original


def test_aiter_cpp_kernel_selects_source_hash_cache(tmp_path, monkeypatch):
    source = tmp_path / "aiter" / "csrc" / "kernel.cu"
    source.parent.mkdir(parents=True)
    source.write_text("kernel", encoding="utf-8")
    monkeypatch.setenv("FORGE_AITER_CACHE_ROOT", str(tmp_path / "cache"))
    monkeypatch.delenv("AITER_REBUILD", raising=False)
    force_jit_rebuild([str(source)])
    assert "AITER_REBUILD" not in os.environ
    assert "sources" in os.environ["AITER_ROOT_DIR"]


def test_source_hash_cache_removes_legacy_rebuild_flag(tmp_path, monkeypatch):
    source = tmp_path / "aiter" / "csrc" / "kernel.hip"
    source.parent.mkdir(parents=True)
    source.write_text("kernel", encoding="utf-8")
    monkeypatch.setenv("FORGE_AITER_CACHE_ROOT", str(tmp_path / "cache"))
    monkeypatch.setenv("AITER_REBUILD", "0")
    force_jit_rebuild([str(source)])
    assert "AITER_REBUILD" not in os.environ


def test_python_kernel_is_noop(monkeypatch):
    monkeypatch.delenv("AITER_REBUILD", raising=False)
    force_jit_rebuild(["/work/aiter/ops/triton/gemm.py"])
    assert "AITER_REBUILD" not in os.environ


def test_non_aiter_cpp_kernel_is_noop(monkeypatch):
    monkeypatch.delenv("AITER_REBUILD", raising=False)
    force_jit_rebuild(["/work/other/csrc/kernel.cu"])
    assert "AITER_REBUILD" not in os.environ


def test_empty_paths_is_noop(monkeypatch):
    monkeypatch.delenv("AITER_REBUILD", raising=False)
    force_jit_rebuild([])
    force_jit_rebuild(["", None])
    assert "AITER_REBUILD" not in os.environ


def test_various_cpp_extensions_detected(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGE_AITER_CACHE_ROOT", str(tmp_path / "cache"))
    for ext in (".cu", ".cuh", ".hip", ".cpp", ".cc", ".cxx", ".c", ".h", ".hpp"):
        monkeypatch.delenv("AITER_REBUILD", raising=False)
        force_jit_rebuild([f"/work/aiter/csrc/kernel{ext}"])
        assert "sources" in os.environ.get("AITER_ROOT_DIR", ""), ext
        assert "AITER_REBUILD" not in os.environ


def test_a_failed_rebuild_stops_the_caller(monkeypatch):
    """The next thing the caller does is benchmark; a silent skip measures the stale binary."""
    monkeypatch.delenv("AITER_REBUILD", raising=False)

    class Boom:
        def __bool__(self):
            raise TypeError("boom")

    with pytest.raises(TypeError, match="boom"):
        force_jit_rebuild([Boom()])


def test_an_unreadable_source_is_not_keyed_to_the_stale_shard(tmp_path, monkeypatch):
    """Two different contents behind one unreadable path must not select the same compiled artifacts."""
    source = tmp_path / "aiter" / "csrc" / "kernel.cu"
    source.parent.mkdir(parents=True)
    source.write_text("kernel", encoding="utf-8")
    monkeypatch.setenv("FORGE_AITER_CACHE_ROOT", str(tmp_path / "cache"))

    def refuse(self, *args, **kwargs):
        raise PermissionError(f"cannot read {self}")

    monkeypatch.setattr(Path, "read_bytes", refuse)

    with pytest.raises(PermissionError):
        force_jit_rebuild([str(source)])


def test_a_source_that_no_longer_exists_still_selects_a_shard(tmp_path, monkeypatch):
    """A declared path the working tree does not carry is a state the digest can express."""
    monkeypatch.setenv("FORGE_AITER_CACHE_ROOT", str(tmp_path / "cache"))

    force_jit_rebuild([str(tmp_path / "aiter" / "csrc" / "deleted.cu")])

    assert "sources" in os.environ["AITER_ROOT_DIR"]


def test_a_broken_workspace_does_not_read_as_no_source_changes(tmp_path):
    """ "git could not be asked" and "nothing changed" send the rebuild to opposite conclusions."""
    with pytest.raises(GitError):
        tracked_source_changes(tmp_path)


def test_either_unreadable_workspace_raises_one_type(tmp_path, monkeypatch):
    """Callers decide what an unassertable rebuild means to them once, not once per way the workspace can fail."""
    source = tmp_path / "aiter" / "csrc" / "kernel.cu"
    source.parent.mkdir(parents=True)
    source.write_text("kernel", encoding="utf-8")
    monkeypatch.setenv("FORGE_AITER_CACHE_ROOT", str(tmp_path / "cache"))

    with pytest.raises(JitRebuildUnavailable, match="GitError"):
        force_jit_rebuild_for_changes(tmp_path, [str(source)])

    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "--allow-empty", "-m", "base"],
        cwd=tmp_path,
        check=True,
    )

    def refuse(self, *args, **kwargs):
        raise PermissionError(f"cannot read {self}")

    monkeypatch.setattr(Path, "read_bytes", refuse)

    with pytest.raises(JitRebuildUnavailable, match="PermissionError"):
        force_jit_rebuild_for_changes(tmp_path, [str(source)])


def test_tracked_source_changes_include_undeclared_edits(tmp_path: Path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=tmp_path,
        check=True,
    )
    kernel = tmp_path / "aiter" / "csrc" / "kernel.cu"
    helper = tmp_path / "aiter" / "csrc" / "helper.cuh"
    kernel.parent.mkdir(parents=True)
    kernel.write_text("kernel\n")
    helper.write_text("helper\n")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "initial"], cwd=tmp_path, check=True)

    helper.write_text("optimized helper\n")

    assert tracked_source_changes(tmp_path) == [str(helper.resolve())]


def test_jit_cache_includes_actual_undeclared_edit(
    tmp_path: Path,
    monkeypatch,
):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=tmp_path,
        check=True,
    )
    anchor = tmp_path / "aiter" / "csrc" / "kernel.cu"
    helper = tmp_path / "aiter" / "csrc" / "helper.cuh"
    anchor.parent.mkdir(parents=True)
    anchor.write_text("kernel\n")
    helper.write_text("helper\n")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "initial"], cwd=tmp_path, check=True)
    helper.write_text("optimized helper\n")
    captured = []
    monkeypatch.setattr(
        "kernelforge.loop.jit_rebuild.activate_aiter_cache_for_sources",
        lambda paths: captured.extend(paths),
    )

    force_jit_rebuild_for_changes(tmp_path, [str(anchor)])

    assert captured == [str(anchor), str(helper.resolve())]
