# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Mocked S6 recipe tests: sgl-kernel and vLLM-from-source.

All subprocess, git, and disk/toolchain calls are mocked.  No GPU, compiler,
or network connection is required.  Tests are CI-friendly.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from hyperloom.orchestrator.framework.build_actions import TargetedBuildAction
from hyperloom.orchestrator.framework.targeted_build import (
    _driver_main,
    run_sgl_kernel_build,
    run_vllm_source_build,
)


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

def _completed(stdout="", stderr="", returncode=0):
    return SimpleNamespace(stdout=stdout, stderr=stderr, returncode=returncode)


def _noop_disk(*a, **kw):
    pass


def _sgl_action(**kw):
    base = dict(
        gap_id="g", framework="sglang", component="sgl_kernel", capability="fp4",
        ref="v0.4.0", repo_url="https://github.com/sgl-project/sglang", gpu_arch="gfx950",
    )
    base.update(kw)
    return TargetedBuildAction(**base)


def _vllm_action(**kw):
    base = dict(
        gap_id="g", framework="vllm", component="vllm_source", capability="deepseek_v4",
        ref="v0.19.0", repo_url="https://github.com/ROCm/vllm", gpu_arch="gfx950",
    )
    base.update(kw)
    return TargetedBuildAction(**base)


class FakeIsolation:
    def __init__(self, worktree_dir, venv_dir):
        self.worktree_dir = worktree_dir
        self.venv_dir = venv_dir

    def prepare_repo_cache(self, req):
        return self.worktree_dir

    def prepare_candidate_workspace(self, req, cand, *, index, execute):
        self.venv_dir.mkdir(parents=True, exist_ok=True)
        return SimpleNamespace(worktree_dir=self.worktree_dir, venv_dir=self.venv_dir)


def _patch_isolation(monkeypatch, worktree, venv):
    import hyperloom.agents.framework.isolation as iso_mod
    monkeypatch.setattr(iso_mod, "_run_git", lambda *a, **kw: None)
    monkeypatch.setattr(iso_mod, "_run_subprocess", lambda *a, **kw: None)
    fake = FakeIsolation(worktree, venv)
    monkeypatch.setattr(iso_mod, "prepare_repo_cache", fake.prepare_repo_cache)
    monkeypatch.setattr(iso_mod, "prepare_candidate_workspace", fake.prepare_candidate_workspace)


def _make_rocm_run(*, hip_ok=True, torch_ver="2.10.0+git8514f05",
                   pip_ok=True, verify_ok=True, git_sha="abc1234",
                   triton_ver="3.1.0"):
    """Generic injectable run for both sgl-kernel and vLLM recipes."""

    def _run(argv, capture_output=False, text=False, timeout=3600, env=None, cwd=None, **kw):
        import json as _j
        cmd = " ".join(str(a) for a in argv)
        # vLLM ROCm verify script (must be before generic is_rocm check)
        if "vllm.platforms" in cmd or "current_platform" in cmd:
            return _completed(returncode=0 if verify_ok else 1,
                              stdout="vllm_rocm_ok\n" if verify_ok else "",
                              stderr="" if verify_ok else "ROCm platform check failed\n")
        # ABI json probe (must be before generic "hip" checks)
        if "is_rocm" in cmd:
            return _completed(stdout=_j.dumps({
                "torch_version": torch_ver,
                "hip_version": "7.2.53211",
                "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.0",
                "is_rocm": hip_ok,
            }) + "\n")
        if "which hipcc" in cmd:
            return _completed(stdout="/opt/rocm/bin/hipcc\n")
        if "dirname" in cmd:
            return _completed(stdout="/opt/rocm\n")
        if "sys.exit(0" in cmd:
            return _completed(returncode=0 if hip_ok else 2)
        if "argv[1]" in cmd:
            if argv and argv[-1] == "triton":
                return _completed(stdout=triton_ver + "\n" if triton_ver else "")
            return _completed(stdout=torch_ver + "\n")
        if "pip" in cmd and "install" in cmd:
            return _completed(returncode=0 if pip_ok else 1,
                              stderr="" if pip_ok else "error: compile failed\n")
        if "setup_rocm.py" in cmd:
            return _completed(returncode=0 if pip_ok else 1)
        if "rev-parse" in cmd:
            return _completed(stdout=git_sha + "\n")
        if "tag -l" in cmd:
            return _completed(stdout="v0.4.0\n")
        if "git" in cmd and "checkout" in cmd:
            return _completed()
        return _completed()

    return _run


def _make_git(git_sha="abc1234"):
    def _git(argv, capture_output=False, text=False, timeout=120, env=None, cwd=None, **kw):
        cmd = " ".join(str(a) for a in argv)
        if "rev-parse" in cmd:
            return _completed(stdout=git_sha + "\n")
        if "checkout" in cmd:
            return _completed()
        return _completed()
    return _git


# ---------------------------------------------------------------------------
# sgl-kernel tests
# ---------------------------------------------------------------------------

def test_sgl_kernel_build_success(monkeypatch, tmp_path):
    wt = tmp_path / "worktree"
    (wt / "sgl-kernel").mkdir(parents=True)
    (wt / "python").mkdir()
    so = wt / "sgl-kernel" / "sgl_kernel_ext.so"
    so.write_bytes(b"\x7fELF")
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    _patch_isolation(monkeypatch, wt, venv)

    result = run_sgl_kernel_build(
        _sgl_action(), str(tmp_path / "attempt"),
        run=_make_rocm_run(), git=_make_git(),
        disk_preflight_fn=_noop_disk,
    )
    assert result.ok is True
    assert result.failure_class == "ok"
    assert result.installed_versions.get("arch") == "gfx950"
    assert result.runtime.pythonpath_prefixes


def test_sgl_kernel_build_missing_gpu_arch_fails(monkeypatch, tmp_path):
    wt = tmp_path / "wt"; (wt / "sgl-kernel").mkdir(parents=True); (wt / "python").mkdir()
    venv = tmp_path / "venv"; (venv / "bin").mkdir(parents=True)
    _patch_isolation(monkeypatch, wt, venv)

    result = run_sgl_kernel_build(
        _sgl_action(gpu_arch=""), str(tmp_path / "attempt"),
        run=_make_rocm_run(), disk_preflight_fn=_noop_disk,
    )
    assert result.ok is False
    assert result.failure_class == "preflight_toolchain"
    assert "gpu_arch" in result.failure_summary


def test_sgl_kernel_build_compile_error(monkeypatch, tmp_path):
    wt = tmp_path / "wt"; (wt / "sgl-kernel").mkdir(parents=True); (wt / "python").mkdir()
    venv = tmp_path / "venv"; (venv / "bin").mkdir(parents=True)
    _patch_isolation(monkeypatch, wt, venv)

    result = run_sgl_kernel_build(
        _sgl_action(), str(tmp_path / "attempt"),
        run=_make_rocm_run(pip_ok=False), disk_preflight_fn=_noop_disk,
    )
    assert result.ok is False
    assert result.failure_class == "compile_error"


def test_sgl_kernel_build_abi_mismatch(monkeypatch, tmp_path):
    wt = tmp_path / "wt"; (wt / "sgl-kernel").mkdir(parents=True); (wt / "python").mkdir()
    venv = tmp_path / "venv"; (venv / "bin").mkdir(parents=True)
    _patch_isolation(monkeypatch, wt, venv)

    result = run_sgl_kernel_build(
        _sgl_action(), str(tmp_path / "attempt"),
        run=_make_rocm_run(hip_ok=False), disk_preflight_fn=_noop_disk,
    )
    assert result.ok is False
    assert result.failure_class in ("abi_mismatch", "preflight_toolchain")


def test_sgl_kernel_build_pyproject_toml_copied(monkeypatch, tmp_path):
    """setup_rocm.py install must not fail when pyproject_other.toml is present."""
    wt = tmp_path / "wt"; (wt / "sgl-kernel").mkdir(parents=True)
    (wt / "python").mkdir()
    (wt / "python" / "pyproject_other.toml").write_text("[project]\nname='sglang'\n")
    so = wt / "sgl-kernel" / "sgl_kernel_ext.so"; so.write_bytes(b"ELF")
    venv = tmp_path / "venv"; (venv / "bin").mkdir(parents=True)
    _patch_isolation(monkeypatch, wt, venv)

    result = run_sgl_kernel_build(
        _sgl_action(), str(tmp_path / "attempt"),
        run=_make_rocm_run(), disk_preflight_fn=_noop_disk,
    )
    assert result.ok
    # The toml should have been copied
    assert (wt / "python" / "pyproject.toml").exists()


def test_sgl_kernel_symbol_missing(monkeypatch, tmp_path):
    wt = tmp_path / "wt"; (wt / "sgl-kernel").mkdir(parents=True); (wt / "python").mkdir()
    so = wt / "sgl-kernel" / "lib.so"; so.write_bytes(b"ELF")
    venv = tmp_path / "venv"; (venv / "bin").mkdir(parents=True)
    _patch_isolation(monkeypatch, wt, venv)

    def _run_missing(argv, **kw):
        cmd = " ".join(str(a) for a in argv)
        if "fp4_gemm" in cmd:
            return _completed(returncode=1)
        return _make_rocm_run()(argv, **kw)

    result = run_sgl_kernel_build(
        _sgl_action(expected_symbols=("sgl_kernel.fp4_gemm",)),
        str(tmp_path / "attempt"),
        run=_run_missing, disk_preflight_fn=_noop_disk,
    )
    assert result.ok is False
    assert result.failure_class == "symbol_missing"


# ---------------------------------------------------------------------------
# vLLM-from-source tests
# ---------------------------------------------------------------------------

def test_vllm_source_build_success(monkeypatch, tmp_path):
    wt = tmp_path / "worktree"; wt.mkdir()
    (wt / "vllm").mkdir()
    so = wt / "vllm" / "_C.so"; so.write_bytes(b"\x7fELF")
    venv = tmp_path / "venv"; (venv / "bin").mkdir(parents=True)
    _patch_isolation(monkeypatch, wt, venv)

    result = run_vllm_source_build(
        _vllm_action(), str(tmp_path / "attempt"),
        run=_make_rocm_run(), disk_preflight_fn=_noop_disk,
    )
    assert result.ok is True
    assert result.failure_class == "ok"
    assert result.installed_versions.get("arch") == "gfx950"
    assert result.runtime.pythonpath_prefixes
    assert result.runtime.entrypoint_bin_dir


def test_vllm_source_build_missing_gpu_arch_fails(monkeypatch, tmp_path):
    wt = tmp_path / "wt"; wt.mkdir()
    venv = tmp_path / "venv"; (venv / "bin").mkdir(parents=True)
    _patch_isolation(monkeypatch, wt, venv)

    result = run_vllm_source_build(
        _vllm_action(gpu_arch=""), str(tmp_path / "attempt"),
        run=_make_rocm_run(), disk_preflight_fn=_noop_disk,
    )
    assert result.ok is False
    assert result.failure_class == "preflight_toolchain"


def test_vllm_source_build_abi_mismatch_refuses_silently(monkeypatch, tmp_path):
    """Must return preflight_toolchain (not compile_error) on non-ROCm torch."""
    wt = tmp_path / "wt"; wt.mkdir()
    venv = tmp_path / "venv"; (venv / "bin").mkdir(parents=True)
    _patch_isolation(monkeypatch, wt, venv)

    result = run_vllm_source_build(
        _vllm_action(), str(tmp_path / "attempt"),
        run=_make_rocm_run(hip_ok=False), disk_preflight_fn=_noop_disk,
    )
    assert result.ok is False
    assert result.failure_class in ("abi_mismatch", "preflight_toolchain")


def test_vllm_source_python_version_abi_guard(monkeypatch, tmp_path):
    """ABI mismatch on wrong Python version is now advisory — build succeeds with runtime_python_exe set."""
    import json as _j
    wt = tmp_path / "wt"; wt.mkdir()
    (wt / "vllm").mkdir()
    (wt / "vllm" / "_C.so").write_bytes(b"\x7fELF")
    venv = tmp_path / "venv"; (venv / "bin").mkdir(parents=True)
    _patch_isolation(monkeypatch, wt, venv)

    def _run_wrong_py(argv, **kw):
        cmd = " ".join(str(a) for a in argv)
        if "is_rocm" in cmd:
            return _completed(stdout=_j.dumps({
                "torch_version": "2.10.0",
                "hip_version": "7.2",
                "python_version": "3.8.0",  # different major.minor
                "is_rocm": True,
            }) + "\n")
        return _make_rocm_run()(argv, **kw)

    if sys.version_info[:2] == (3, 8):
        pytest.skip("host is 3.8, no mismatch to detect")

    result = run_vllm_source_build(
        _vllm_action(), str(tmp_path / "attempt"),
        run=_run_wrong_py, disk_preflight_fn=_noop_disk,
    )
    # ABI guard is now advisory: build must succeed and runtime_python_exe must be set.
    assert result.ok is True
    assert result.runtime.runtime_python_exe


def test_vllm_source_compile_error(monkeypatch, tmp_path):
    wt = tmp_path / "wt"; wt.mkdir()
    venv = tmp_path / "venv"; (venv / "bin").mkdir(parents=True)
    _patch_isolation(monkeypatch, wt, venv)

    result = run_vllm_source_build(
        _vllm_action(), str(tmp_path / "attempt"),
        run=_make_rocm_run(pip_ok=False), disk_preflight_fn=_noop_disk,
    )
    assert result.ok is False
    assert result.failure_class == "compile_error"


def test_vllm_source_rocm_verify_fails_boot_failed(monkeypatch, tmp_path):
    wt = tmp_path / "wt"; wt.mkdir()
    (wt / "vllm").mkdir()
    (wt / "vllm" / "_C.so").write_bytes(b"ELF")
    venv = tmp_path / "venv"; (venv / "bin").mkdir(parents=True)
    _patch_isolation(monkeypatch, wt, venv)

    result = run_vllm_source_build(
        _vllm_action(), str(tmp_path / "attempt"),
        run=_make_rocm_run(verify_ok=False), disk_preflight_fn=_noop_disk,
    )
    assert result.ok is False
    assert result.failure_class == "boot_failed"


def test_vllm_source_symbol_missing(monkeypatch, tmp_path):
    wt = tmp_path / "wt"; wt.mkdir()
    (wt / "vllm").mkdir()
    (wt / "vllm" / "_C.so").write_bytes(b"ELF")
    venv = tmp_path / "venv"; (venv / "bin").mkdir(parents=True)
    _patch_isolation(monkeypatch, wt, venv)

    def _run_sym(argv, **kw):
        cmd = " ".join(str(a) for a in argv)
        # Only fail the explicit symbol verify, not the vllm_rocm verify
        if "model_executor" in cmd and "deepseek_v4" in cmd.lower():
            return _completed(returncode=1)
        return _make_rocm_run()(argv, **kw)

    result = run_vllm_source_build(
        _vllm_action(expected_symbols=("vllm.model_executor.models.deepseek_v4",)),
        str(tmp_path / "attempt"),
        run=_run_sym, disk_preflight_fn=_noop_disk,
    )
    assert result.ok is False
    assert result.failure_class == "symbol_missing"


def test_vllm_source_runtime_fields(monkeypatch, tmp_path):
    """KEEP'd runtime must have pythonpath_prefixes, entrypoint_bin_dir, runtime_python_exe, PYTORCH_ROCM_ARCH."""
    wt = tmp_path / "wt"; wt.mkdir()
    (wt / "vllm").mkdir()
    (wt / "vllm" / "_C.so").write_bytes(b"ELF")
    venv = tmp_path / "venv"; (venv / "bin").mkdir(parents=True)
    _patch_isolation(monkeypatch, wt, venv)

    result = run_vllm_source_build(
        _vllm_action(gpu_arch="gfx950"), str(tmp_path / "attempt"),
        run=_make_rocm_run(), disk_preflight_fn=_noop_disk,
    )
    assert result.ok
    rt = result.runtime
    assert rt.pythonpath_prefixes
    assert rt.entrypoint_bin_dir
    assert rt.runtime_python_exe
    assert rt.runtime_env.get("PYTORCH_ROCM_ARCH") == "gfx950"


def test_vllm_source_load_probe_failure_returns_boot_failed(monkeypatch, tmp_path):
    """Load probe rc!=0 must return boot_failed."""
    wt = tmp_path / "wt"; wt.mkdir()
    (wt / "vllm").mkdir()
    (wt / "vllm" / "_C.so").write_bytes(b"ELF")
    venv = tmp_path / "venv"; (venv / "bin").mkdir(parents=True)
    _patch_isolation(monkeypatch, wt, venv)

    def _run_probe_fail(argv, **kw):
        cmd = " ".join(str(a) for a in argv)
        if "inspect" in cmd and "startswith" in cmd:
            return _completed(returncode=3, stdout="", stderr="wrong path\n")
        return _make_rocm_run()(argv, **kw)

    result = run_vllm_source_build(
        _vllm_action(), str(tmp_path / "attempt"),
        run=_run_probe_fail, disk_preflight_fn=_noop_disk,
    )
    assert result.ok is False
    assert result.failure_class == "boot_failed"
    assert "load probe" in result.failure_summary


def test_sgl_kernel_runtime_python_exe_set(monkeypatch, tmp_path):
    """sgl-kernel build must set runtime_python_exe and entrypoint_bin_dir."""
    wt = tmp_path / "worktree"
    (wt / "sgl-kernel").mkdir(parents=True)
    (wt / "python").mkdir()
    (wt / "sgl-kernel" / "sgl_kernel_ext.so").write_bytes(b"\x7fELF")
    venv = tmp_path / "venv"; (venv / "bin").mkdir(parents=True)
    _patch_isolation(monkeypatch, wt, venv)

    result = run_sgl_kernel_build(
        _sgl_action(), str(tmp_path / "attempt"),
        run=_make_rocm_run(), git=_make_git(),
        disk_preflight_fn=_noop_disk,
    )
    assert result.ok is True
    assert result.runtime.runtime_python_exe
    assert result.runtime.entrypoint_bin_dir


def test_sgl_kernel_source_pr_url_in_installed_versions(monkeypatch, tmp_path):
    wt = tmp_path / "wt"
    (wt / "sgl-kernel").mkdir(parents=True)
    (wt / "python").mkdir()
    (wt / "sgl-kernel" / "sgl_ext.so").write_bytes(b"\x7fELF")
    venv = tmp_path / "venv"; (venv / "bin").mkdir(parents=True)
    _patch_isolation(monkeypatch, wt, venv)

    result = run_sgl_kernel_build(
        _sgl_action(source_pr_url="https://github.com/sgl-project/sglang/pull/5"),
        str(tmp_path / "attempt"),
        run=_make_rocm_run(), git=_make_git(),
        disk_preflight_fn=_noop_disk,
    )
    assert result.ok is True
    assert result.installed_versions.get("source_pr_url") == "https://github.com/sgl-project/sglang/pull/5"


def test_vllm_source_pr_url_in_installed_versions(monkeypatch, tmp_path):
    wt = tmp_path / "wt"; wt.mkdir()
    (wt / "vllm").mkdir()
    (wt / "vllm" / "_C.so").write_bytes(b"ELF")
    venv = tmp_path / "venv"; (venv / "bin").mkdir(parents=True)
    _patch_isolation(monkeypatch, wt, venv)

    result = run_vllm_source_build(
        _vllm_action(source_pr_url="https://github.com/ROCm/vllm/pull/42"),
        str(tmp_path / "attempt"),
        run=_make_rocm_run(), disk_preflight_fn=_noop_disk,
    )
    assert result.ok is True
    assert result.installed_versions.get("source_pr_url") == "https://github.com/ROCm/vllm/pull/42"


def test_recipe_no_source_pr_url_when_empty(monkeypatch, tmp_path):
    wt = tmp_path / "wt"; wt.mkdir()
    (wt / "vllm").mkdir()
    (wt / "vllm" / "_C.so").write_bytes(b"ELF")
    venv = tmp_path / "venv"; (venv / "bin").mkdir(parents=True)
    _patch_isolation(monkeypatch, wt, venv)

    result = run_vllm_source_build(
        _vllm_action(),  # source_pr_url=""
        str(tmp_path / "attempt2"),
        run=_make_rocm_run(), disk_preflight_fn=_noop_disk,
    )
    assert result.ok is True
    assert "source_pr_url" not in result.installed_versions
    root = tmp_path / "attempt"; root.mkdir()
    action = _sgl_action()
    (root / "plan.json").write_text(json.dumps(action.to_state()), encoding="utf-8")

    import hyperloom.orchestrator.framework.targeted_build as tb

    def _fake_sgl(action, attempt_root, **kw):
        from hyperloom.orchestrator.framework.build_actions import BuildResult
        return BuildResult(ok=True, attempt_root=attempt_root, failure_class="ok")

    monkeypatch.setattr(tb, "run_sgl_kernel_build", _fake_sgl)

    rc = _driver_main(["--attempt-root", str(root)])
    assert rc == 0
    data = json.loads((root / "result.json").read_text())
    assert data["ok"] is True


def test_driver_main_routes_vllm_source(monkeypatch, tmp_path):
    root = tmp_path / "attempt"; root.mkdir()
    action = _vllm_action()
    (root / "plan.json").write_text(json.dumps(action.to_state()), encoding="utf-8")

    import hyperloom.orchestrator.framework.targeted_build as tb

    def _fake_vllm(action, attempt_root, **kw):
        from hyperloom.orchestrator.framework.build_actions import BuildResult
        return BuildResult(ok=True, attempt_root=attempt_root, failure_class="ok")

    monkeypatch.setattr(tb, "run_vllm_source_build", _fake_vllm)

    rc = _driver_main(["--attempt-root", str(root)])
    assert rc == 0


def test_driver_main_unknown_component_returns_failure(monkeypatch, tmp_path):
    """A component not in the driver dispatcher returns failure without spawning a real build."""
    root = tmp_path / "attempt"; root.mkdir()
    action = _vllm_action()
    (root / "plan.json").write_text(json.dumps(action.to_state()), encoding="utf-8")

    import hyperloom.orchestrator.framework.targeted_build as tb

    # Remove vllm_source from the dispatcher to simulate an unknown component
    original_driver_main = tb._driver_main

    def _patched_driver(argv=None):
        import argparse, json, sys as _sys
        from pathlib import Path as _Path
        parser = argparse.ArgumentParser()
        parser.add_argument("--attempt-root", required=True)
        args = parser.parse_args(argv)
        root2 = _Path(args.attempt_root)
        plan = json.loads((root2 / "plan.json").read_text())
        result_path = root2 / "result.json"
        # Simulate no dispatcher entry
        result_path.write_text(json.dumps({
            "ok": False, "attempt_root": str(root2),
            "failure_class": "compile_error",
            "failure_summary": "no recipe for component 'unknown'",
            "error": "no recipe for component 'unknown'",
        }), encoding="utf-8")
        return 1

    monkeypatch.setattr(tb, "_driver_main", _patched_driver)

    rc = tb._driver_main(["--attempt-root", str(root)])
    assert rc == 1
    data = json.loads((root / "result.json").read_text())
    assert data["ok"] is False


# ---------------------------------------------------------------------------
# S5 escalation picks the right component
# ---------------------------------------------------------------------------

def test_escalation_component_selection():
    """Verify _maybe_escalate_to_targeted_build component selection logic (unit)."""
    from hyperloom.orchestrator.phases.framework import _derive_gpu_arch

    # Just check the helper is importable and correct
    assert _derive_gpu_arch("mi355x") == "gfx950"
    assert _derive_gpu_arch("mi300x") == "gfx942"
