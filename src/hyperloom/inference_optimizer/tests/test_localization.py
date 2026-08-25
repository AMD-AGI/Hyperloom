# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the localization module + adapter localization hooks."""

from __future__ import annotations


import pytest

from .conftest import init_git_repo

from hyperloom.agents.framework.enablement import (
    MISSING_MODEL_ARCH,
    RESOURCE_CONSTRAINT,
    CapabilityGap,
    FailureSignature,
)
from hyperloom.orchestrator.framework import localization as loc
from hyperloom.orchestrator.framework.adapters import get_adapter
from hyperloom.orchestrator.framework.stack_actions import EnablementStackAction
from hyperloom.orchestrator.actions.executors._git import _run_git_cp


def _gap(kind: str = MISSING_MODEL_ARCH) -> CapabilityGap:
    return CapabilityGap.from_signature(FailureSignature(kind=kind, confidence=0.9))


# ---------------------------------------------------------------------------
# classify_closure — compiled-closure gate
# ---------------------------------------------------------------------------


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
    assert path in v.compiled_paths or "cap" in v.reason


def test_closure_file_cap_defers_rung5():
    v = loc.classify_closure([f"vllm/m{i}.py" for i in range(65)])
    assert v.kind == loc.NEEDS_RUNG5


# ---------------------------------------------------------------------------
# synthesize_vendor_diff + parse_diff_paths
# ---------------------------------------------------------------------------


def test_synthesize_vendor_add():
    d = loc.synthesize_vendor_diff([("vllm/new.py", "", "def f():\n    return 1\n")])
    assert "diff --git a/vllm/new.py b/vllm/new.py" in d
    assert "new file mode" in d
    assert loc.parse_diff_paths(d) == ["vllm/new.py"]


def test_synthesize_vendor_replace():
    d = loc.synthesize_vendor_diff([("vllm/m.py", "old\n", "new\n")])
    assert "diff --git a/vllm/m.py b/vllm/m.py" in d
    assert "new file mode" not in d
    assert "-old" in d and "+new" in d


def test_synthesize_vendor_skips_identical():
    assert loc.synthesize_vendor_diff([("a.py", "same\n", "same\n")]) == ""


# ---------------------------------------------------------------------------
# build_localization_diff (injected shims)
# ---------------------------------------------------------------------------


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
    dt, paths, verdict = loc.build_localization_diff(
        _pr_action(), fetch_pr_patches=lambda s, n: diff, fetch_raw_file=lambda *a: ""
    )
    assert verdict.kind == loc.PYTHON_ONLY
    assert paths == ["vllm/model.py"]
    assert dt == diff


def test_build_pr_backport_compiled_rung5():
    diff = "diff --git a/csrc/k.cu b/csrc/k.cu\n--- a/csrc/k.cu\n+++ b/csrc/k.cu\n@@ -1 +1 @@\n-a\n+b\n"
    dt, _, verdict = loc.build_localization_diff(
        _pr_action(), fetch_pr_patches=lambda s, n: diff, fetch_raw_file=lambda *a: ""
    )
    assert verdict.kind == loc.NEEDS_RUNG5
    assert dt == diff  # returned for observability but not localizable


def test_build_pr_backport_fetch_failed_empty():
    _, _, verdict = loc.build_localization_diff(
        _pr_action(), fetch_pr_patches=lambda s, n: "", fetch_raw_file=lambda *a: ""
    )
    assert verdict.kind == loc.EMPTY


def test_build_vendor_files_synthesizes():
    action = EnablementStackAction.from_state(
        {
            "kind": "vendor_files",
            "framework": "vllm",
            "gap_id": "g",
            "capability": "c",
            "repo_url": "https://github.com/ROCm/vllm.git",
            "ref": "abc",
            "localized_paths": ["vllm/new.py"],
        }
    )
    dt, paths, verdict = loc.build_localization_diff(
        action, fetch_pr_patches=lambda *a: "", fetch_raw_file=lambda s, r, p: "print(1)\n"
    )
    assert verdict.kind == loc.PYTHON_ONLY
    assert "diff --git a/vllm/new.py b/vllm/new.py" in dt


# ---------------------------------------------------------------------------
# Applies to a temp git tree (end-to-end diff validity)
# ---------------------------------------------------------------------------


def test_synthesized_add_applies_to_git_tree(tmp_path):
    repo = tmp_path / "repo"
    init_git_repo(repo)
    d = loc.synthesize_vendor_diff([("pkg/new_model.py", "", "def build():\n    return 42\n")])
    patch = tmp_path / "loc.patch"
    patch.write_text(d, encoding="utf-8")
    cp = _run_git_cp(["-C", str(repo), "apply", "--check", str(patch)], timeout=30.0)
    assert cp is not None and cp.returncode == 0, getattr(cp, "stderr", "")
    cp2 = _run_git_cp(["-C", str(repo), "apply", str(patch)], timeout=30.0)
    assert cp2 is not None and cp2.returncode == 0
    assert (repo / "pkg" / "new_model.py").read_text().strip().endswith("return 42")


# ---------------------------------------------------------------------------
# adapter localization hooks
# ---------------------------------------------------------------------------


def test_vllm_localization_action_and_refresh(monkeypatch):
    monkeypatch.delenv("HYPERLOOM_ENABLEMENT_ORIGIN_ALLOWLIST", raising=False)
    a = get_adapter("vllm")
    act = a.build_localization_action(
        _gap(), framework="vllm", model="m", candidate_ref="PR:1234", repo_url="https://github.com/ROCm/vllm.git"
    )
    assert act is not None and act.kind == "pr_backport" and act.pr_number == 1234
    argv = a.editable_refresh_argv("/v/bin/python", "/co")
    assert argv == ["/v/bin/python", "-m", "pip", "install", "-e", "/co", "--no-deps"]


def test_vllm_localization_resource_constraint_none():
    a = get_adapter("vllm")
    assert (
        a.build_localization_action(
            _gap(RESOURCE_CONSTRAINT), framework="vllm", model="m", candidate_ref="PR:1", repo_url="https://x"
        )
        is None
    )


def test_atom_localizes_no_refresh(monkeypatch):
    monkeypatch.delenv("HYPERLOOM_ENABLEMENT_ORIGIN_ALLOWLIST", raising=False)
    at = get_adapter("atom")
    act = at.build_localization_action(
        _gap(), framework="atom", model="m", candidate_ref="PR:55", repo_url="https://github.com/ROCm/ATOM.git"
    )
    assert act is not None and act.pr_number == 55
    assert at.editable_refresh_argv("/v/py", "/co") is None


def test_xdit_and_unknown_localization_none():
    assert (
        get_adapter("xdit").build_localization_action(
            _gap(), framework="xdit", model="m", candidate_ref="PR:1", repo_url="https://x"
        )
        is None
    )
    assert (
        get_adapter("nope").build_localization_action(
            _gap(), framework="nope", model="m", candidate_ref="PR:1", repo_url="https://x"
        )
        is None
    )


def test_origin_allowlist_blocks_unlisted(monkeypatch):
    monkeypatch.setenv("HYPERLOOM_ENABLEMENT_ORIGIN_ALLOWLIST", "https://github.com/ROCm")
    a = get_adapter("vllm")
    assert (
        a.build_localization_action(
            _gap(), framework="vllm", model="m", candidate_ref="PR:1", repo_url="https://evil.example/x.git"
        )
        is None
    )
