# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""sglang_gc_patch lands on the installed tree without changing shape discovery."""

from __future__ import annotations

import sys
import types
from pathlib import Path

from hyperloom.orchestrator.actions.executors import _server_patcher
from hyperloom.orchestrator.actions.executors._server_patcher import (
    ensure_sglang_patched_for_tracelens,
)

_RUNNER_REL = Path("python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py")
_SOURCE = """\
def capture():
    return 1


def done():
    return 0
"""
_PATCH = """\
diff --git a/python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py b/python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py
--- a/python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py
+++ b/python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py
@@ -1,2 +1,5 @@
+def _set_profile_trace_tag(self):
+    return "_profile_runner_name"
+
 def capture():
     return 1
"""


def _editable_install(root: Path) -> Path:
    pkg = root / "python" / "sglang"
    target = root / _RUNNER_REL
    target.parent.mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").write_text("__version__ = '0.5.21'\n", encoding="utf-8")
    target.write_text(_SOURCE, encoding="utf-8")
    return target


def _gc_tree(tracelens: Path, version_dir: str = "sglang_0_5_21") -> None:
    patches = tracelens / "examples" / "custom_workflows" / "inference_analysis" / "sglang_gc_patch" / version_dir
    patches.mkdir(parents=True)
    (patches / "decode_cuda_graph_runner.patch").write_text(_PATCH, encoding="utf-8")


def _fake_sglang(monkeypatch, pkg: Path, version: str) -> None:
    mod = types.ModuleType("sglang")
    mod.__version__ = version
    mod.__file__ = str(pkg / "__init__.py")
    monkeypatch.setitem(sys.modules, "sglang", mod)
    monkeypatch.delenv("HYPERLOOM_SGLANG_SHAPE_MODE", raising=False)
    monkeypatch.delenv("HYPERLOOM_SGLANG_VERSION_PIN", raising=False)


def test_gc_patch_applies_and_is_idempotent(tmp_path: Path, monkeypatch):
    repo = tmp_path / "sglang"
    target = _editable_install(repo)
    tracelens = tmp_path / "TraceLens"
    _gc_tree(tracelens)
    _fake_sglang(monkeypatch, repo / "python" / "sglang", "0.5.21")
    monkeypatch.setenv("TRACELENS_ROOT", str(tracelens))

    assert ensure_sglang_patched_for_tracelens() is True
    text = target.read_text(encoding="utf-8")
    assert "_set_profile_trace_tag" in text
    assert "_profile_runner_name" in text

    assert ensure_sglang_patched_for_tracelens() is True


def test_newer_sglang_uses_nearest_not_newer_gc_set(tmp_path: Path, monkeypatch):
    repo = tmp_path / "sglang"
    target = _editable_install(repo)
    _gc_tree(tmp_path / "TraceLens", "sglang_0_5_21")
    _fake_sglang(monkeypatch, repo / "python" / "sglang", "0.5.22")
    monkeypatch.setenv("TRACELENS_ROOT", str(tmp_path / "TraceLens"))

    assert ensure_sglang_patched_for_tracelens() is True
    assert "_set_profile_trace_tag" in target.read_text(encoding="utf-8")


def test_older_than_gate_does_not_touch_the_tree(tmp_path: Path, monkeypatch):
    repo = tmp_path / "sglang"
    target = _editable_install(repo)
    _fake_sglang(monkeypatch, repo / "python" / "sglang", "0.5.10")
    monkeypatch.delenv("TRACELENS_ROOT", raising=False)

    assert ensure_sglang_patched_for_tracelens() is False
    assert target.read_text(encoding="utf-8") == _SOURCE


def test_missing_gc_tree_fails_soft(tmp_path: Path, monkeypatch):
    repo = tmp_path / "sglang"
    _editable_install(repo)
    _fake_sglang(monkeypatch, repo / "python" / "sglang", "0.5.21")
    monkeypatch.setenv("TRACELENS_ROOT", str(tmp_path / "empty"))
    (tmp_path / "empty").mkdir()

    assert ensure_sglang_patched_for_tracelens() is False


def test_multi_node_controller_without_sglang_defers(monkeypatch):
    monkeypatch.setenv("HYPERLOOM_SGLANG_SHAPE_MODE", "sitecustomize")
    monkeypatch.setattr(_server_patcher, "_detect_installed_sglang_version", lambda: None)
    monkeypatch.setattr(_server_patcher, "_gc_patch_deferred_to_pods", lambda: True)

    assert ensure_sglang_patched_for_tracelens() is True


def test_local_process_without_sglang_reports_failure(monkeypatch):
    monkeypatch.setenv("HYPERLOOM_SGLANG_SHAPE_MODE", "sitecustomize")
    monkeypatch.setattr(_server_patcher, "_detect_installed_sglang_version", lambda: None)
    monkeypatch.setattr(_server_patcher, "_gc_patch_deferred_to_pods", lambda: False)

    assert ensure_sglang_patched_for_tracelens() is False
