# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for the JIT-rebuild safety net (loop/jit_rebuild.py).

monkeypatch.setenv/delenv keeps os.environ mutations from leaking between tests."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from kernelforge.loop.jit_rebuild import (
    force_jit_rebuild,
    force_jit_rebuild_for_changes,
    tracked_source_changes,
)


@pytest.fixture(autouse=True)
def _isolate_aiter_root_dir():
    """Snapshot and restore ``AITER_ROOT_DIR`` around every test in this module.

    ``force_jit_rebuild`` writes ``AITER_ROOT_DIR`` DIRECTLY into ``os.environ``
    (via the aiter-cache isolation helper), not through ``monkeypatch``, so
    monkeypatch's teardown does not undo it. Without this, the value set here
    leaks into later tests (e.g. ``resolve_aiter_root`` in the kernelforge.gemm_tune
    suite reads it and resolves a bogus root).
    """
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


def test_exception_is_swallowed(monkeypatch):
    monkeypatch.delenv("AITER_REBUILD", raising=False)

    class Boom:
        def __bool__(self):
            # __bool__ must raise TypeError (its standard exception) rather than
            # a non-standard one; the test only needs truthiness to raise so the
            # caller's exception handling can be exercised.
            raise TypeError("boom")

    # A non-string, non-empty path whose truthiness raises must be swallowed.
    force_jit_rebuild([Boom()])
    assert "AITER_REBUILD" not in os.environ


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
