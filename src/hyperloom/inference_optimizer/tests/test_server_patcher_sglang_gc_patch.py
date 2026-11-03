# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""sglang_gc_patch lands in a copy of the installed tree without changing shape discovery."""

from __future__ import annotations

import shutil
import sys
import tempfile
import types
from pathlib import Path

import pytest

from hyperloom.orchestrator.actions.executors import PatchedServer, _patch_laying, _server_patcher
from hyperloom.orchestrator.actions.executors._server_patcher import (
    ensure_sglang_patched_for_tracelens,
)

from ._sglang_gc_patch_tree import RUNNER_PATCH, RUNNER_REL, RUNNER_SOURCE


def _editable_install(root: Path) -> Path:
    pkg = root / "python" / "sglang"
    target = root / RUNNER_REL
    target.parent.mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").write_text("__version__ = '0.5.21'\n", encoding="utf-8")
    target.write_text(RUNNER_SOURCE, encoding="utf-8")
    return target


def _gc_tree(tracelens: Path, version_dir: str = "sglang_0_5_21") -> None:
    patches = tracelens / "examples" / "custom_workflows" / "inference_analysis" / "sglang_gc_patch" / version_dir
    patches.mkdir(parents=True)
    (patches / "decode_cuda_graph_runner.patch").write_text(RUNNER_PATCH, encoding="utf-8")


@pytest.fixture(autouse=True)
def _own_tmpdir(tmp_path: Path, monkeypatch) -> None:
    """The patched copies of this test, kept out of the host's temporary directory."""
    (tmp_path / "tmp").mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "tmp"))


def _copies(tmp_path: Path) -> list[Path]:
    root = tmp_path / "tmp" / "hyperloom-tracelens"
    return sorted(root.iterdir()) if root.is_dir() else []


def _fake_sglang(monkeypatch, pkg: Path, version: str) -> None:
    mod = types.ModuleType("sglang")
    mod.__version__ = version
    mod.__file__ = str(pkg / "__init__.py")
    monkeypatch.setitem(sys.modules, "sglang", mod)
    monkeypatch.delenv("HYPERLOOM_SGLANG_VERSION_PIN", raising=False)


def test_gc_patch_applies_and_is_idempotent(tmp_path: Path, monkeypatch):
    repo = tmp_path / "sglang"
    target = _editable_install(repo)
    tracelens = tmp_path / "TraceLens"
    _gc_tree(tracelens)
    _fake_sglang(monkeypatch, repo / "python" / "sglang", "0.5.21")
    monkeypatch.setenv("TRACELENS_ROOT", str(tracelens))

    patched = ensure_sglang_patched_for_tracelens()
    assert patched is not None and patched.site is not None
    text = (patched.site / RUNNER_REL.relative_to("python")).read_text(encoding="utf-8")
    assert "_set_profile_trace_tag" in text
    assert "_profile_runner_name" in text
    assert target.read_text(encoding="utf-8") == RUNNER_SOURCE, "the installed tree is never patched"

    assert ensure_sglang_patched_for_tracelens() == patched
    assert len(_copies(tmp_path)) == 1


def test_gc_patch_uses_the_cached_tracelens_checkout_without_tracelens_root(tmp_path: Path, monkeypatch):
    """An installer-cached ``TraceLens@<sha>`` is the TraceLens root when ``TRACELENS_ROOT`` is unset."""
    repo = tmp_path / "sglang"
    target = _editable_install(repo)
    cache = tmp_path / "cache"
    _gc_tree(cache / "TraceLens@0123abc")
    _fake_sglang(monkeypatch, repo / "python" / "sglang", "0.5.21")
    monkeypatch.delenv("TRACELENS_ROOT", raising=False)
    monkeypatch.setenv("HYPERLOOM_CACHE_DIR", str(cache))

    patched = ensure_sglang_patched_for_tracelens()
    assert patched is not None and patched.site is not None
    assert "_set_profile_trace_tag" in (patched.site / RUNNER_REL.relative_to("python")).read_text(encoding="utf-8")
    assert target.read_text(encoding="utf-8") == RUNNER_SOURCE


def test_newer_sglang_without_exact_gc_set_is_left_alone(tmp_path: Path, monkeypatch):
    repo = tmp_path / "sglang"
    target = _editable_install(repo)
    _gc_tree(tmp_path / "TraceLens", "sglang_0_5_21")
    _fake_sglang(monkeypatch, repo / "python" / "sglang", "0.5.22")
    monkeypatch.setenv("TRACELENS_ROOT", str(tmp_path / "TraceLens"))

    assert ensure_sglang_patched_for_tracelens() is None
    assert target.read_text(encoding="utf-8") == RUNNER_SOURCE


def test_drifted_source_is_not_fuzzy_patched(tmp_path: Path, monkeypatch):
    """A hunk whose context drifted would pass ``patch --fuzz=2``; the gc set must refuse it."""
    repo = tmp_path / "sglang"
    target = _editable_install(repo)
    drifted = RUNNER_SOURCE.replace("    return 1", "    return 2")
    target.write_text(drifted, encoding="utf-8")
    _gc_tree(tmp_path / "TraceLens")
    _fake_sglang(monkeypatch, repo / "python" / "sglang", "0.5.21")
    monkeypatch.setenv("TRACELENS_ROOT", str(tmp_path / "TraceLens"))

    patch_bin = shutil.which("patch")
    if patch_bin:
        patch_file = next((tmp_path / "TraceLens").rglob("*.patch"))
        assert _patch_laying._patch_dry_run(patch_bin, patch_file, repo, 1), "fixture must be fuzz-applicable"

    assert ensure_sglang_patched_for_tracelens() is None
    assert target.read_text(encoding="utf-8") == drifted
    assert _copies(tmp_path) == [], "a copy the patch did not land in is not kept"


def test_older_than_gate_does_not_touch_the_tree(tmp_path: Path, monkeypatch):
    repo = tmp_path / "sglang"
    target = _editable_install(repo)
    _fake_sglang(monkeypatch, repo / "python" / "sglang", "0.5.10")
    monkeypatch.delenv("TRACELENS_ROOT", raising=False)

    assert ensure_sglang_patched_for_tracelens() is None
    assert target.read_text(encoding="utf-8") == RUNNER_SOURCE


def test_missing_gc_tree_fails_soft(tmp_path: Path, monkeypatch):
    repo = tmp_path / "sglang"
    _editable_install(repo)
    _fake_sglang(monkeypatch, repo / "python" / "sglang", "0.5.21")
    monkeypatch.setenv("TRACELENS_ROOT", str(tmp_path / "empty"))
    (tmp_path / "empty").mkdir()

    assert ensure_sglang_patched_for_tracelens() is None


def test_multi_node_controller_without_sglang_defers(monkeypatch):
    monkeypatch.setenv("HYPERLOOM_SGLANG_VERSION_PIN", "0.5.21")
    monkeypatch.setattr(_server_patcher, "_detect_installed_sglang_version", lambda: None)
    monkeypatch.setattr(_server_patcher, "_gc_patch_deferred_to_pods", lambda: True)

    assert ensure_sglang_patched_for_tracelens() == PatchedServer(None)


def test_local_process_without_sglang_reports_failure(monkeypatch):
    monkeypatch.setenv("HYPERLOOM_SGLANG_VERSION_PIN", "0.5.21")
    monkeypatch.setattr(_server_patcher, "_detect_installed_sglang_version", lambda: None)
    monkeypatch.setattr(_server_patcher, "_gc_patch_deferred_to_pods", lambda: False)

    assert ensure_sglang_patched_for_tracelens() is None


def test_node_scope_failure_propagates_instead_of_patching_the_controller(monkeypatch):
    import pytest

    def _boom():
        raise RuntimeError("node scope unreadable")

    monkeypatch.setattr("hyperloom.orchestrator.actions.executors._multi_node_env.is_multi_node", _boom)

    with pytest.raises(RuntimeError, match="node scope unreadable"):
        _server_patcher._gc_patch_deferred_to_pods()
