# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The pod-side patcher picks sglang_gc_patch from the pod's own SGLang version."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

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


@pytest.fixture
def pod_patcher():
    import hyperloom.inference_optimizer as io_pkg

    script = Path(io_pkg.__file__).parent / "multi_node" / "scripts" / "apply_tracelens_patch_multinode.py"
    spec = importlib.util.spec_from_file_location("_tracelens_pod_patcher_under_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _install(tmp_path: Path, monkeypatch, version: str) -> tuple[Path, Path]:
    repo = tmp_path / "sglang"
    target = repo / _RUNNER_REL
    target.parent.mkdir(parents=True)
    target.write_text(_SOURCE, encoding="utf-8")
    pkg = repo / "python" / "sglang"
    (pkg / "__init__.py").write_text("", encoding="utf-8")

    mod = types.ModuleType("sglang")
    mod.__version__ = version
    mod.__file__ = str(pkg / "__init__.py")
    ver = types.ModuleType("sglang.version")
    ver.__version__ = version
    monkeypatch.setitem(sys.modules, "sglang", mod)
    monkeypatch.setitem(sys.modules, "sglang.version", ver)
    monkeypatch.delenv("HYPERLOOM_SGLANG_SHAPE_MODE", raising=False)

    tracelens = tmp_path / "TraceLens"
    gc_dir = tracelens / "examples" / "custom_workflows" / "inference_analysis" / "sglang_gc_patch" / "sglang_0_5_21"
    gc_dir.mkdir(parents=True)
    (gc_dir / "decode_cuda_graph_runner.patch").write_text(_PATCH, encoding="utf-8")
    return target, tracelens


@pytest.mark.parametrize(
    ("version", "override", "expected"),
    [
        ("0.5.17", "", False),
        ("0.5.18", "", True),
        ("0.5.21.dev3+gabc1234", "", True),
        ("0.5.21", "patch", False),
        ("0.5.10", "sitecustomize", True),
        ("", "", False),
    ],
)
def test_gate_follows_version_then_override(pod_patcher, monkeypatch, version, override, expected):
    if override:
        monkeypatch.setenv("HYPERLOOM_SGLANG_SHAPE_MODE", override)
    else:
        monkeypatch.delenv("HYPERLOOM_SGLANG_SHAPE_MODE", raising=False)
    assert pod_patcher._uses_gc_patch(version) is expected


def test_pod_applies_gc_patch_then_skips(pod_patcher, tmp_path, monkeypatch):
    target, tracelens = _install(tmp_path, monkeypatch, "0.5.22")

    first = pod_patcher._apply_on_pod(
        tracelens_root=str(tracelens), tracelens_internal_root="", sglang_version_pin=None
    )
    assert first["status"] == "applied", first
    assert first["patch_set"] == "graph-capture"
    assert "_set_profile_trace_tag" in target.read_text(encoding="utf-8")

    second = pod_patcher._apply_on_pod(
        tracelens_root=str(tracelens), tracelens_internal_root="", sglang_version_pin=None
    )
    assert second["status"] == "skipped", second


def test_pod_below_gate_stays_on_roofline(pod_patcher, tmp_path, monkeypatch):
    target, tracelens = _install(tmp_path, monkeypatch, "0.5.17")

    result = pod_patcher._apply_on_pod(
        tracelens_root=str(tracelens), tracelens_internal_root="", sglang_version_pin=None
    )
    assert result["patch_set"] == "roofline"
    assert target.read_text(encoding="utf-8") == _SOURCE
