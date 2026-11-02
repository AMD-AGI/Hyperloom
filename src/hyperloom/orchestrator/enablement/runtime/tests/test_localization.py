# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the localization module + adapter localization hooks."""

from __future__ import annotations


import pytest


from hyperloom.common.failure_signature import MISSING_MODEL_ARCH, FailureSignature
from hyperloom.orchestrator.enablement.runtime import localization as loc
from hyperloom.orchestrator.enablement.runtime.adapters import get_adapter
from hyperloom.orchestrator.enablement.runtime.stack_actions import EnablementStackAction


def _gap(kind: str = MISSING_MODEL_ARCH) -> FailureSignature:
    return FailureSignature(kind=kind, confidence=0.9)


# classify_closure — compiled-closure gate


def test_closure_empty():
    assert loc.classify_closure([]).kind == loc.EMPTY


def test_closure_python_only():
    v = loc.classify_closure(["vllm/model/deepseek_v4.py", "vllm/config.py"])
    assert v.kind == loc.PYTHON_ONLY
    assert v.is_localizable


@pytest.mark.parametrize("path", ["csrc/kernel.cu", "src/op.cpp", "x.hip", "m.pyx", "CMakeLists.txt", "setup.py"])
def test_closure_compiled_or_build_defers_rung5(path):
    v = loc.classify_closure(["vllm/model.py", path])
    assert v.kind == loc.NEEDS_RUNG5
    assert not v.is_localizable
    assert path in v.reason


def test_a_large_python_closure_is_localizable():
    v = loc.classify_closure([f"vllm/m{i}.py" for i in range(200)])
    assert v.kind == loc.PYTHON_ONLY


# build_localization_diff (injected shims)


def _pr_action() -> EnablementStackAction:
    return EnablementStackAction.from_state(
        {
            "kind": "pr_backport",
            "framework": "vllm",
            "gap_id": "gap.enablement.missing_model_arch",
            "capability": "deepseek_v4",
            "repo_url": "https://github.com/ROCm/vllm.git",
            "pr_number": 1234,
        }
    )


def test_build_pr_backport_python_only():
    diff = "diff --git a/vllm/model.py b/vllm/model.py\n--- a/vllm/model.py\n+++ b/vllm/model.py\n@@ -1 +1 @@\n-a\n+b\n"
    dt, paths, verdict = loc.build_localization_diff(_pr_action(), fetch_pr_patches=lambda s, n: diff, scope="vllm")
    assert verdict.kind == loc.PYTHON_ONLY
    assert paths == ["vllm/model.py"]
    assert dt == diff


def test_build_pr_backport_compiled_rung5():
    diff = "diff --git a/csrc/k.cu b/csrc/k.cu\n--- a/csrc/k.cu\n+++ b/csrc/k.cu\n@@ -1 +1 @@\n-a\n+b\n"
    dt, _, verdict = loc.build_localization_diff(_pr_action(), fetch_pr_patches=lambda s, n: diff, scope="vllm")
    assert verdict.kind == loc.NEEDS_RUNG5
    assert dt == diff  # returned for observability but not localizable


def test_build_pr_backport_with_nothing_in_the_package_is_empty():
    tests = "diff --git a/tests/t.py b/tests/t.py\n--- a/tests/t.py\n+++ b/tests/t.py\n@@ -1 +1 @@\n-a\n+b\n"
    _, _, verdict = loc.build_localization_diff(_pr_action(), fetch_pr_patches=lambda s, n: tests, scope="vllm")
    assert verdict.kind == loc.EMPTY


def test_build_pr_backport_oversize(monkeypatch):
    monkeypatch.setattr(loc, "_MAX_VENDOR_BYTES", 10)
    pkg = "diff --git a/vllm/model.py b/vllm/model.py\n--- a/vllm/model.py\n+++ b/vllm/model.py\n@@ -1 +1 @@\n-a\n+b\n"
    _, _, verdict = loc.build_localization_diff(_pr_action(), fetch_pr_patches=lambda s, n: pkg, scope="vllm")
    assert verdict.kind == loc.OVERSIZE
    assert not verdict.is_localizable


def test_build_pr_backport_fetch_failed_empty():
    _, _, verdict = loc.build_localization_diff(_pr_action(), fetch_pr_patches=lambda s, n: "", scope="vllm")
    assert verdict.kind == loc.EMPTY


# adapter localization hooks


@pytest.mark.parametrize(
    ("framework", "layout", "import_root"),
    [
        ("sglang", "python/sglang", "python"),
        ("vllm", "vllm", ""),
        ("atom", "atom", ""),
    ],
)
def test_the_import_root_is_the_directory_above_the_package(tmp_path, framework, layout, import_root):
    adapter = get_adapter(framework)
    assert adapter.source_import_root(str(tmp_path)) == ""
    (tmp_path / layout).mkdir(parents=True)
    assert adapter.source_import_root(str(tmp_path)) == import_root


def test_vllm_localization_action():
    a = get_adapter("vllm")
    act = a.build_localization_action(_gap(), candidate_ref="PR:1234", repo_url="https://github.com/ROCm/vllm.git")
    assert act is not None and act.kind == "pr_backport" and act.pr_number == 1234


def test_atom_localizes():
    at = get_adapter("atom")
    act = at.build_localization_action(_gap(), candidate_ref="PR:55", repo_url="https://github.com/ROCm/ATOM.git")
    assert act is not None and act.pr_number == 55


def test_xdit_and_unknown_localization_none():
    assert get_adapter("xdit").build_localization_action(_gap(), candidate_ref="PR:1", repo_url="https://x") is None
    assert get_adapter("nope").build_localization_action(_gap(), candidate_ref="PR:1", repo_url="https://x") is None
