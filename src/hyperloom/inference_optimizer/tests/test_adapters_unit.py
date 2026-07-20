# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for Rung 3 (M1) enablement adapters (§8.4).

All subprocess calls go through an injected fake ``run`` shim so no ROCm /
network / real venv is required.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from hyperloom.agents.framework.enablement import (
    MISSING_MODEL_ARCH,
    RESOURCE_CONSTRAINT,
    CapabilityGap,
    FailureSignature,
)
from hyperloom.orchestrator.framework import adapters as ad
from hyperloom.orchestrator.framework.adapters import (
    AtomAdapter,
    NullAdapter,
    SglangAdapter,
    VllmRocmAdapter,
    XditAdapter,
    get_adapter,
)


def _gap(kind: str = MISSING_MODEL_ARCH) -> CapabilityGap:
    return CapabilityGap.from_signature(FailureSignature(kind=kind, confidence=0.9))


class _FakeRun:
    """Programmable fake subprocess shim: maps argv-substring -> (rc, out, err)."""

    def __init__(self, rules: list[tuple[str, int, str, str]] | None = None, default_rc: int = 0):
        self.rules = rules or []
        self.default_rc = default_rc
        self.calls: list[list[str]] = []

    def __call__(self, argv, env, cwd):
        self.calls.append(list(argv))
        joined = " ".join(argv)
        for needle, rc, out, err in self.rules:
            if needle in joined:
                return subprocess.CompletedProcess(argv, rc, out, err)
        return subprocess.CompletedProcess(argv, self.default_rc, "", "")


# ---------------------------------------------------------------------------
# supports / registry
# ---------------------------------------------------------------------------

def test_vllm_supports_missing_arch_not_resource_constraint():
    a = VllmRocmAdapter()
    assert a.supports(_gap(MISSING_MODEL_ARCH)) is True
    assert a.supports(_gap(RESOURCE_CONSTRAINT)) is False


def test_sglang_supports_missing_arch_not_resource_constraint():
    a = SglangAdapter()
    assert a.supports(_gap(MISSING_MODEL_ARCH)) is True
    assert a.supports(_gap(RESOURCE_CONSTRAINT)) is False


def test_atom_and_xdit_never_support():
    assert AtomAdapter().supports(_gap(MISSING_MODEL_ARCH)) is False
    assert XditAdapter().supports(_gap(MISSING_MODEL_ARCH)) is False


def test_get_adapter_unknown_returns_null_no_raise():
    a = get_adapter("totally_unknown_fw")
    assert isinstance(a, NullAdapter)
    assert a.supports(_gap()) is False
    assert a.build_stack_action(_gap(), framework="x", model="m") is None


def test_get_adapter_case_insensitive():
    assert isinstance(get_adapter("VLLM"), VllmRocmAdapter)
    assert isinstance(get_adapter("SGLang"), SglangAdapter)


# ---------------------------------------------------------------------------
# build_stack_action: ROCm index gating (never PyPI CUDA fallback)
# ---------------------------------------------------------------------------

def test_vllm_no_rocm_index_returns_none(monkeypatch):
    monkeypatch.delenv("HYPERLOOM_VLLM_ROCM_INDEX_URL", raising=False)
    a = VllmRocmAdapter()
    assert a.build_stack_action(_gap(), framework="vllm", model="m") is None


def test_vllm_with_rocm_index_builds_wheel_action(monkeypatch):
    monkeypatch.setenv("HYPERLOOM_VLLM_ROCM_INDEX_URL", "https://rocm.repo/whl")
    monkeypatch.delenv("HYPERLOOM_ENABLEMENT_INDEX_ALLOWLIST", raising=False)
    a = VllmRocmAdapter()
    action = a.build_stack_action(_gap(), framework="vllm", model="m", gpu_type="mi355x")
    assert action is not None
    assert action.acquisition_method == "wheel"
    assert action.index_url == "https://rocm.repo/whl"
    assert action.packages == ("vllm",)


def test_vllm_index_not_in_allowlist_returns_none(monkeypatch):
    monkeypatch.setenv("HYPERLOOM_VLLM_ROCM_INDEX_URL", "https://evil.repo/whl")
    monkeypatch.setenv("HYPERLOOM_ENABLEMENT_INDEX_ALLOWLIST", "https://rocm.repo")
    a = VllmRocmAdapter()
    assert a.build_stack_action(_gap(), framework="vllm", model="m") is None


def test_vllm_resource_constraint_returns_none(monkeypatch):
    monkeypatch.setenv("HYPERLOOM_VLLM_ROCM_INDEX_URL", "https://rocm.repo/whl")
    a = VllmRocmAdapter()
    assert a.build_stack_action(_gap(RESOURCE_CONSTRAINT), framework="vllm", model="m") is None


def test_sglang_editable_ref_action(monkeypatch):
    monkeypatch.setenv("HYPERLOOM_SGLANG_REPO_URL", "https://github.com/sgl-project/sglang.git")
    monkeypatch.setenv("HYPERLOOM_SGLANG_REF", "v0.4.9")
    monkeypatch.delenv("HYPERLOOM_ENABLEMENT_ORIGIN_ALLOWLIST", raising=False)
    a = SglangAdapter()
    action = a.build_stack_action(_gap(), framework="sglang", model="m")
    assert action is not None
    assert action.acquisition_method == "editable_ref"
    assert action.ref == "v0.4.9"


def test_sglang_no_source_no_index_returns_none(monkeypatch):
    for k in ("HYPERLOOM_SGLANG_REPO_URL", "HYPERLOOM_SGLANG_REF", "HYPERLOOM_SGLANG_INDEX_URL"):
        monkeypatch.delenv(k, raising=False)
    a = SglangAdapter()
    assert a.build_stack_action(_gap(), framework="sglang", model="m") is None


# ---------------------------------------------------------------------------
# provision + ROCm verification (mocked run)
# ---------------------------------------------------------------------------

def _wheel_action() -> "ad.EnablementStackAction":
    from hyperloom.orchestrator.framework.stack_actions import EnablementStackAction

    return EnablementStackAction(
        kind="runtime_candidate",
        framework="vllm",
        gap_id="gap.enablement.missing_model_arch",
        capability="deepseek_v4",
        acquisition_method="wheel",
        index_url="https://rocm.repo/whl",
        packages=("vllm",),
    )


def test_vllm_provision_ok(tmp_path, monkeypatch):
    # venv create ok, pip ok, torch-rocm ok, vllm-rocm ok, versions resolvable.
    run = _FakeRun(
        rules=[
            ("importlib.metadata", 0, "0.21.0", ""),
        ],
        default_rc=0,
    )
    a = VllmRocmAdapter(run=run)
    result = a.provision(_wheel_action(), tmp_path / "attempt")
    assert result.ok is True, result.error
    assert result.runtime.venv_root.endswith("venv")
    assert result.runtime.bin_path.endswith("bin")
    assert result.installed_versions.get("vllm") == "0.21.0"


def test_vllm_provision_pip_fail(tmp_path):
    run = _FakeRun(rules=[("-m pip install", 1, "", "no matching distribution")], default_rc=0)
    a = VllmRocmAdapter(run=run)
    result = a.provision(_wheel_action(), tmp_path / "attempt")
    assert result.ok is False
    assert "pip install failed" in result.error


def test_vllm_provision_rejects_cuda_torch(tmp_path):
    # torch-is-rocm probe fails (rc=1) -> reject a CUDA torch swap-in.
    run = _FakeRun(rules=[("getattr(torch.version,'hip'", 1, "", "")], default_rc=0)
    a = VllmRocmAdapter(run=run)
    result = a.provision(_wheel_action(), tmp_path / "attempt")
    assert result.ok is False
    assert "not a ROCm build" in result.error


def test_vllm_provision_rejects_non_rocm_vllm(tmp_path):
    # torch-rocm ok but vllm platform probe fails.
    def _run(argv, env, cwd):
        joined = " ".join(argv)
        if "getattr(torch.version,'hip'" in joined and "vllm" not in joined:
            return subprocess.CompletedProcess(argv, 0, "", "")  # torch is rocm
        if "import vllm" in joined or "current_platform" in joined:
            return subprocess.CompletedProcess(argv, 1, "", "not rocm")  # vllm not rocm
        return subprocess.CompletedProcess(argv, 0, "", "")

    a = VllmRocmAdapter(run=_run)
    result = a.provision(_wheel_action(), tmp_path / "attempt")
    assert result.ok is False
    assert "ROCm platform" in result.error


def test_vllm_provision_requires_index(tmp_path):
    from hyperloom.orchestrator.framework.stack_actions import EnablementStackAction

    action = EnablementStackAction(
        kind="runtime_candidate", framework="vllm", gap_id="g", capability="c", acquisition_method="wheel"
    )
    a = VllmRocmAdapter(run=_FakeRun())
    result = a.provision(action, tmp_path / "attempt")
    assert result.ok is False
    assert "ROCm wheel index" in result.error


# ---------------------------------------------------------------------------
# verify helpers
# ---------------------------------------------------------------------------

def test_verify_torch_is_rocm_true_false():
    assert ad.verify_torch_is_rocm("/py", run=_FakeRun(default_rc=0)) is True
    assert ad.verify_torch_is_rocm("/py", run=_FakeRun(default_rc=1)) is False


def test_verify_vllm_rocm_true_false():
    assert ad.verify_vllm_rocm("/py", run=_FakeRun(default_rc=0)) is True
    assert ad.verify_vllm_rocm("/py", run=_FakeRun(default_rc=1)) is False
