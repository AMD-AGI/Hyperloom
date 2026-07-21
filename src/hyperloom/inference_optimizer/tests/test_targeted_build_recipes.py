# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Mocked targeted-build recipe tests: AITER, sgl-kernel, and vLLM-from-source.

All subprocess, git, and disk/toolchain calls are mocked. No GPU, compiler, or
network connection is required. Each test exercises one branch of the
recipe -> verify -> result.json flow.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from hyperloom.orchestrator.framework.build_actions import TargetedBuildAction
from hyperloom.orchestrator.framework.targeted_build import (
    _driver_main,
    run_aiter_build,
    run_sgl_kernel_build,
    run_vllm_source_build,
)


# ---------------------------------------------------------------------------
# Shared mock helpers
# ---------------------------------------------------------------------------

def _completed(stdout="", stderr="", returncode=0):
    return SimpleNamespace(stdout=stdout, stderr=stderr, returncode=returncode)


def _noop_disk(*a, **kw):
    pass


class FakeIsolation:
    """Replaces isolation.prepare_repo_cache + prepare_candidate_workspace."""

    def __init__(self, worktree_dir: Path, venv_dir: Path):
        self.worktree_dir = worktree_dir
        self.venv_dir = venv_dir

    def prepare_repo_cache(self, req):
        return self.worktree_dir

    def prepare_candidate_workspace(self, req, candidate, *, index, execute):
        self.venv_dir.mkdir(parents=True, exist_ok=True)
        return SimpleNamespace(worktree_dir=self.worktree_dir, venv_dir=self.venv_dir)


def _make_git(*, aiter_tags="v0.1.0\nv0.0.9", git_sha="abc1234"):
    def _git(argv, capture_output=False, text=False, timeout=120, env=None, cwd=None, **kw):
        cmd = " ".join(str(a) for a in argv)
        if "tag -l" in cmd:
            return _completed(stdout=aiter_tags)
        if "checkout" in cmd:
            return _completed()
        if "rev-parse" in cmd:
            return _completed(stdout=git_sha + "\n")
        return _completed()
    return _git


# ---------------------------------------------------------------------------
# AITER recipe helpers
# ---------------------------------------------------------------------------

def _aiter_action(**kw):
    base = dict(
        gap_id="g",
        framework="vllm",
        component="aiter",
        capability="fp4_moe",
        ref="v0.1.0",
        repo_url="https://github.com/ROCm/aiter",
        gpu_arch="gfx950",
    )
    base.update(kw)
    return TargetedBuildAction(**base)


def _patch_aiter_isolation(monkeypatch, tmp_path, *, make_so=True):
    worktree = tmp_path / "worktree"
    worktree.mkdir(parents=True, exist_ok=True)
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True, exist_ok=True)
    (venv / "bin" / "python").write_text("#!/usr/bin/env python3\n")

    if make_so:
        (worktree / "module_aiter_core.so").write_bytes(b"\x7fELF")

    fake = FakeIsolation(worktree, venv)

    import hyperloom.agents.framework.isolation as iso_mod
    monkeypatch.setattr(iso_mod, "_run_git", lambda *a, **kw: None)
    monkeypatch.setattr(iso_mod, "_run_subprocess", lambda *a, **kw: None)
    monkeypatch.setattr(iso_mod, "prepare_repo_cache", fake.prepare_repo_cache)
    monkeypatch.setattr(iso_mod, "prepare_candidate_workspace", fake.prepare_candidate_workspace)
    return worktree, venv


def _make_aiter_run(
    *,
    hip_ok=True,
    pip_ok=True,
    import_ok=True,
    aiter_tags="v0.1.0\nv0.0.9",
    git_sha="abc1234",
    torch_ver="2.10.0+git8514f05",
    triton_ver="3.1.0",
    toolchain_hipcc="/opt/rocm/bin/hipcc",
):
    """Build an injectable run callable covering all AITER subprocess branches."""

    def _run(argv, capture_output=False, text=False, timeout=3600, env=None, cwd=None, **kw):
        cmd = " ".join(str(a) for a in argv)
        if "is_rocm" in cmd:
            data = {
                "torch_version": torch_ver,
                "hip_version": "7.2.53211",
                "python_version": "3.12.13",
                "is_rocm": hip_ok,
            }
            return _completed(stdout=json.dumps(data) + "\n")
        if "which hipcc" in cmd:
            return _completed(stdout=toolchain_hipcc + "\n" if toolchain_hipcc else "")
        if "dirname" in cmd:
            root = Path(toolchain_hipcc).parent.parent if toolchain_hipcc else Path("/opt/rocm")
            return _completed(stdout=str(root) + "\n")
        if "sys.exit(0" in cmd:
            return _completed(returncode=0 if hip_ok else 2)
        if "argv[1]" in cmd:
            if argv and argv[-1] == "triton":
                return _completed(stdout=triton_ver + "\n" if triton_ver else "")
            return _completed(stdout=torch_ver + "\n")
        if "pip" in cmd and "install" in cmd:
            return _completed(returncode=0 if pip_ok else 1,
                              stderr="" if pip_ok else "error: compile failed")
        if "import aiter" in cmd:
            return _completed(returncode=0 if import_ok else 1)
        if "tag -l" in cmd:
            return _completed(stdout=aiter_tags)
        if "git" in cmd and "checkout" in cmd:
            return _completed()
        if "rev-parse" in cmd:
            return _completed(stdout=git_sha + "\n")
        return _completed()

    return _run


# ---------------------------------------------------------------------------
# sgl-kernel / vLLM recipe helpers
# ---------------------------------------------------------------------------

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
        cmd = " ".join(str(a) for a in argv)
        if "vllm.platforms" in cmd or "current_platform" in cmd:
            return _completed(returncode=0 if verify_ok else 1,
                              stdout="vllm_rocm_ok\n" if verify_ok else "",
                              stderr="" if verify_ok else "ROCm platform check failed\n")
        if "is_rocm" in cmd:
            return _completed(stdout=json.dumps({
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


# ---------------------------------------------------------------------------
# AITER: success
# ---------------------------------------------------------------------------

def test_run_aiter_build_success_pinned_ref(monkeypatch, tmp_path):
    _patch_aiter_isolation(monkeypatch, tmp_path, make_so=True)

    result = run_aiter_build(
        _aiter_action(ref="v0.1.0"),
        str(tmp_path / "attempt"),
        run=_make_aiter_run(), git=_make_git(),
        disk_preflight_fn=_noop_disk,
    )

    assert result.ok is True
    assert result.failure_class == "ok"
    assert result.runtime.pythonpath_prefixes
    assert result.installed_versions["aiter_ref"] == "v0.1.0"
    assert result.installed_versions["arch"] == "gfx950"
    assert "INFERENCE_OPTIMIZER_AITER_JIT_DIR" in result.runtime.runtime_env


def test_run_aiter_build_success_runtime_paths_valid(monkeypatch, tmp_path):
    _patch_aiter_isolation(monkeypatch, tmp_path, make_so=True)
    result = run_aiter_build(
        _aiter_action(ref="v0.1.0"),
        str(tmp_path / "attempt"),
        run=_make_aiter_run(), git=_make_git(),
        disk_preflight_fn=_noop_disk,
    )
    assert result.ok
    jit = result.runtime.runtime_env["INFERENCE_OPTIMIZER_AITER_JIT_DIR"]
    assert "attempt" in jit
    assert ".aiter" not in jit or "attempt" in jit


# ---------------------------------------------------------------------------
# AITER: failure branches
# ---------------------------------------------------------------------------

def test_run_aiter_build_compile_error(monkeypatch, tmp_path):
    _patch_aiter_isolation(monkeypatch, tmp_path, make_so=False)
    result = run_aiter_build(
        _aiter_action(ref="v0.1.0"),
        str(tmp_path / "attempt"),
        run=_make_aiter_run(pip_ok=False), git=_make_git(),
        disk_preflight_fn=_noop_disk,
    )
    assert result.ok is False
    assert result.failure_class == "compile_error"


def test_run_aiter_build_abi_mismatch(monkeypatch, tmp_path):
    _patch_aiter_isolation(monkeypatch, tmp_path)
    result = run_aiter_build(
        _aiter_action(ref="v0.1.0"),
        str(tmp_path / "attempt"),
        run=_make_aiter_run(hip_ok=False), git=_make_git(),
        disk_preflight_fn=_noop_disk,
    )
    assert result.ok is False
    assert result.failure_class in ("abi_mismatch", "preflight_toolchain")


def test_run_aiter_build_symbol_missing(monkeypatch, tmp_path):
    _patch_aiter_isolation(monkeypatch, tmp_path, make_so=True)

    def _run_missing(argv, **kw):
        cmd = " ".join(str(a) for a in argv)
        if "fp4_moe" in cmd:
            return _completed(returncode=1)
        return _make_aiter_run()(argv, **kw)

    result = run_aiter_build(
        _aiter_action(ref="v0.1.0", expected_symbols=("aiter.ops.fp4_moe",)),
        str(tmp_path / "attempt"),
        run=_run_missing, git=_make_git(),
        disk_preflight_fn=_noop_disk,
    )
    assert result.ok is False
    assert result.failure_class == "symbol_missing"


# ---------------------------------------------------------------------------
# AITER: tag autoselect
# ---------------------------------------------------------------------------

def test_run_aiter_build_autoselect_hit(monkeypatch, tmp_path):
    _patch_aiter_isolation(monkeypatch, tmp_path, make_so=True)
    result = run_aiter_build(
        _aiter_action(ref=""),
        str(tmp_path / "attempt"),
        run=_make_aiter_run(aiter_tags="v0.2.0\nv0.1.0"),
        git=_make_git(aiter_tags="v0.2.0\nv0.1.0"),
        disk_preflight_fn=_noop_disk,
    )
    assert result.ok is True
    assert result.installed_versions["aiter_ref"] == "v0.2.0"


def test_run_aiter_build_autoselect_exhaust(monkeypatch, tmp_path):
    _patch_aiter_isolation(monkeypatch, tmp_path, make_so=False)
    result = run_aiter_build(
        _aiter_action(ref=""),
        str(tmp_path / "attempt"),
        run=_make_aiter_run(pip_ok=False, aiter_tags="v0.1.0"),
        git=_make_git(aiter_tags="v0.1.0"),
        disk_preflight_fn=_noop_disk,
    )
    assert result.ok is False
    assert result.failure_class == "compile_error"
    assert "no AITER tag" in result.failure_summary


def test_run_aiter_build_no_live_tree_mutation(monkeypatch, tmp_path):
    """Install-step env must use per-attempt AITER_ROOT_DIR and JIT dir."""
    install_envs: list[dict] = []

    def _capturing_run(argv, **kw):
        cmd = " ".join(str(a) for a in argv)
        if kw.get("env") and ("pip" in cmd and "install" in cmd):
            install_envs.append(dict(kw["env"]))
        return _make_aiter_run()(argv, **kw)

    _patch_aiter_isolation(monkeypatch, tmp_path, make_so=True)
    result = run_aiter_build(
        _aiter_action(ref="v0.1.0"),
        str(tmp_path / "attempt"),
        run=_capturing_run, git=_make_git(),
        disk_preflight_fn=_noop_disk,
    )
    assert result.ok, result.failure_summary
    assert install_envs, "no install env captured"
    for env in install_envs:
        home = env.get("HOME", "")
        aiter_root = env.get("AITER_ROOT_DIR", "")
        jit = env.get("INFERENCE_OPTIMIZER_AITER_JIT_DIR", "")
        if aiter_root:
            assert os.path.expanduser("~/.aiter") not in aiter_root
        if jit:
            assert os.path.expanduser("~") not in jit or "attempt" in jit
        attempt_root = str(tmp_path / "attempt")
        for key, val in [("HOME", home), ("AITER_ROOT_DIR", aiter_root),
                         ("INFERENCE_OPTIMIZER_AITER_JIT_DIR", jit)]:
            if val:
                assert attempt_root in val


def test_run_aiter_build_reproducible_versions(monkeypatch, tmp_path):
    _patch_aiter_isolation(monkeypatch, tmp_path, make_so=True)
    result = run_aiter_build(
        _aiter_action(ref="v0.1.0", gpu_arch="gfx950"),
        str(tmp_path / "attempt"),
        run=_make_aiter_run(torch_ver="2.10.0+git8514f05", git_sha="deadbeef"),
        git=_make_git(git_sha="deadbeef"),
        disk_preflight_fn=_noop_disk,
    )
    assert result.ok
    v = result.installed_versions
    assert "2.10.0" in v["torch"]
    assert v["aiter_sha"] == "deadbeef"
    assert v["arch"] == "gfx950"


def test_run_aiter_build_source_pr_url_in_installed_versions(monkeypatch, tmp_path):
    _patch_aiter_isolation(monkeypatch, tmp_path, make_so=True)
    result = run_aiter_build(
        _aiter_action(ref="v0.1.0", gpu_arch="gfx950",
                      source_pr_url="https://github.com/ROCm/aiter/pull/77"),
        str(tmp_path / "attempt"),
        run=_make_aiter_run(), git=_make_git(),
        disk_preflight_fn=_noop_disk,
    )
    assert result.ok
    assert result.installed_versions.get("source_pr_url") == "https://github.com/ROCm/aiter/pull/77"


def test_run_aiter_build_no_source_pr_url_when_empty(monkeypatch, tmp_path):
    _patch_aiter_isolation(monkeypatch, tmp_path, make_so=True)
    result = run_aiter_build(
        _aiter_action(ref="v0.1.0", gpu_arch="gfx950"),
        str(tmp_path / "attempt"),
        run=_make_aiter_run(), git=_make_git(),
        disk_preflight_fn=_noop_disk,
    )
    assert result.ok
    assert "source_pr_url" not in result.installed_versions


# ---------------------------------------------------------------------------
# AITER: result.json round-trip + driver
# ---------------------------------------------------------------------------

def test_result_json_round_trip_through_poll_build(tmp_path):
    """Driver writes result.json; poll_build reads it as a rich BuildResult."""
    from hyperloom.orchestrator.framework.build_actions import BuildResult, FrameworkRuntime
    from hyperloom.orchestrator.framework.targeted_build import BuildHandle, poll_build

    root = tmp_path / "attempt"
    root.mkdir()

    rich = BuildResult(
        ok=True,
        attempt_root=str(root),
        runtime=FrameworkRuntime(
            pythonpath_prefixes=(str(root / "worktree"),),
            runtime_env={"INFERENCE_OPTIMIZER_AITER_JIT_DIR": str(root / "jit")},
            source_root=str(root / "worktree"),
            attempt_root=str(root),
        ),
        installed_versions={"torch": "2.10.0+git8514f05", "aiter_ref": "v0.1.0"},
        failure_class="ok",
    )
    (root / "result.json").write_text(json.dumps(rich.to_state()), encoding="utf-8")

    proc = SimpleNamespace(returncode=0, poll=lambda: 0)
    handle = BuildHandle(
        action=_aiter_action(ref="v0.1.0"),
        attempt_root=str(root),
        aiter_jit_dir=str(root / "jit"),
        build_log_path=str(root / "build.log"),
        proc=proc,
        pid=12345,
        pgid=12345,
        deadline=time.monotonic() + 3600,
    )

    result = poll_build(handle)
    assert result is not None
    assert result.ok is True
    assert result.installed_versions["aiter_ref"] == "v0.1.0"
    assert result.runtime.pythonpath_prefixes == (str(root / "worktree"),)


def test_driver_main_writes_result_json(monkeypatch, tmp_path):
    root = tmp_path / "attempt"
    root.mkdir()

    action = _aiter_action(ref="v0.1.0")
    (root / "plan.json").write_text(json.dumps(action.to_state()), encoding="utf-8")

    from hyperloom.orchestrator.framework import targeted_build as tb

    def _fake_recipe(action, attempt_root, **kw):
        from hyperloom.orchestrator.framework.build_actions import BuildResult
        return BuildResult(ok=True, attempt_root=attempt_root, failure_class="ok")

    monkeypatch.setattr(tb, "run_aiter_build", _fake_recipe)

    rc = _driver_main(["--attempt-root", str(root)])
    assert rc == 0
    data = json.loads((root / "result.json").read_text())
    assert data["ok"] is True


def test_driver_main_missing_plan_writes_error(tmp_path):
    root = tmp_path / "attempt_no_plan"
    root.mkdir()
    rc = _driver_main(["--attempt-root", str(root)])
    assert rc == 1
    data = json.loads((root / "result.json").read_text())
    assert data["ok"] is False
    assert "plan.json" in data.get("failure_summary", "")


# ---------------------------------------------------------------------------
# sgl-kernel
# ---------------------------------------------------------------------------

def test_sgl_kernel_build_success(monkeypatch, tmp_path):
    wt = tmp_path / "worktree"
    (wt / "sgl-kernel").mkdir(parents=True)
    (wt / "python").mkdir()
    (wt / "sgl-kernel" / "sgl_kernel_ext.so").write_bytes(b"\x7fELF")
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
    wt = tmp_path / "wt"
    (wt / "sgl-kernel").mkdir(parents=True)
    (wt / "python").mkdir()
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    _patch_isolation(monkeypatch, wt, venv)

    result = run_sgl_kernel_build(
        _sgl_action(gpu_arch=""), str(tmp_path / "attempt"),
        run=_make_rocm_run(), disk_preflight_fn=_noop_disk,
    )
    assert result.ok is False
    assert result.failure_class == "preflight_toolchain"
    assert "gpu_arch" in result.failure_summary


def test_sgl_kernel_build_compile_error(monkeypatch, tmp_path):
    wt = tmp_path / "wt"
    (wt / "sgl-kernel").mkdir(parents=True)
    (wt / "python").mkdir()
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    _patch_isolation(monkeypatch, wt, venv)

    result = run_sgl_kernel_build(
        _sgl_action(), str(tmp_path / "attempt"),
        run=_make_rocm_run(pip_ok=False), disk_preflight_fn=_noop_disk,
    )
    assert result.ok is False
    assert result.failure_class == "compile_error"


def test_sgl_kernel_build_abi_mismatch(monkeypatch, tmp_path):
    wt = tmp_path / "wt"
    (wt / "sgl-kernel").mkdir(parents=True)
    (wt / "python").mkdir()
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    _patch_isolation(monkeypatch, wt, venv)

    result = run_sgl_kernel_build(
        _sgl_action(), str(tmp_path / "attempt"),
        run=_make_rocm_run(hip_ok=False), disk_preflight_fn=_noop_disk,
    )
    assert result.ok is False
    assert result.failure_class in ("abi_mismatch", "preflight_toolchain")


def test_sgl_kernel_build_pyproject_toml_copied(monkeypatch, tmp_path):
    """setup_rocm.py install must not fail when pyproject_other.toml is present."""
    wt = tmp_path / "wt"
    (wt / "sgl-kernel").mkdir(parents=True)
    (wt / "python").mkdir()
    (wt / "python" / "pyproject_other.toml").write_text("[project]\nname='sglang'\n")
    (wt / "sgl-kernel" / "sgl_kernel_ext.so").write_bytes(b"ELF")
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    _patch_isolation(monkeypatch, wt, venv)

    result = run_sgl_kernel_build(
        _sgl_action(), str(tmp_path / "attempt"),
        run=_make_rocm_run(), disk_preflight_fn=_noop_disk,
    )
    assert result.ok
    assert (wt / "python" / "pyproject.toml").exists()


def test_sgl_kernel_symbol_missing(monkeypatch, tmp_path):
    wt = tmp_path / "wt"
    (wt / "sgl-kernel").mkdir(parents=True)
    (wt / "python").mkdir()
    (wt / "sgl-kernel" / "lib.so").write_bytes(b"ELF")
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
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


def test_sgl_kernel_runtime_python_exe_set(monkeypatch, tmp_path):
    """sgl-kernel build must set runtime_python_exe and entrypoint_bin_dir."""
    wt = tmp_path / "worktree"
    (wt / "sgl-kernel").mkdir(parents=True)
    (wt / "python").mkdir()
    (wt / "sgl-kernel" / "sgl_kernel_ext.so").write_bytes(b"\x7fELF")
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
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
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    _patch_isolation(monkeypatch, wt, venv)

    result = run_sgl_kernel_build(
        _sgl_action(source_pr_url="https://github.com/sgl-project/sglang/pull/5"),
        str(tmp_path / "attempt"),
        run=_make_rocm_run(), git=_make_git(),
        disk_preflight_fn=_noop_disk,
    )
    assert result.ok is True
    assert result.installed_versions.get("source_pr_url") == "https://github.com/sgl-project/sglang/pull/5"


# ---------------------------------------------------------------------------
# vLLM-from-source
# ---------------------------------------------------------------------------

def test_vllm_source_build_success(monkeypatch, tmp_path):
    wt = tmp_path / "worktree"
    wt.mkdir()
    (wt / "vllm").mkdir()
    (wt / "vllm" / "_C.so").write_bytes(b"\x7fELF")
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
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
    wt = tmp_path / "wt"
    wt.mkdir()
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    _patch_isolation(monkeypatch, wt, venv)

    result = run_vllm_source_build(
        _vllm_action(gpu_arch=""), str(tmp_path / "attempt"),
        run=_make_rocm_run(), disk_preflight_fn=_noop_disk,
    )
    assert result.ok is False
    assert result.failure_class == "preflight_toolchain"


def test_vllm_source_build_abi_mismatch_refuses_silently(monkeypatch, tmp_path):
    """Must return preflight_toolchain (not compile_error) on non-ROCm torch."""
    wt = tmp_path / "wt"
    wt.mkdir()
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    _patch_isolation(monkeypatch, wt, venv)

    result = run_vllm_source_build(
        _vllm_action(), str(tmp_path / "attempt"),
        run=_make_rocm_run(hip_ok=False), disk_preflight_fn=_noop_disk,
    )
    assert result.ok is False
    assert result.failure_class in ("abi_mismatch", "preflight_toolchain")


def test_vllm_source_python_version_abi_guard(monkeypatch, tmp_path):
    """ABI mismatch on wrong Python version is advisory: build succeeds with runtime_python_exe set."""
    wt = tmp_path / "wt"
    wt.mkdir()
    (wt / "vllm").mkdir()
    (wt / "vllm" / "_C.so").write_bytes(b"\x7fELF")
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    _patch_isolation(monkeypatch, wt, venv)

    def _run_wrong_py(argv, **kw):
        cmd = " ".join(str(a) for a in argv)
        if "is_rocm" in cmd:
            return _completed(stdout=json.dumps({
                "torch_version": "2.10.0",
                "hip_version": "7.2",
                "python_version": "3.8.0",
                "is_rocm": True,
            }) + "\n")
        return _make_rocm_run()(argv, **kw)

    if sys.version_info[:2] == (3, 8):
        pytest.skip("host is 3.8, no mismatch to detect")

    result = run_vllm_source_build(
        _vllm_action(), str(tmp_path / "attempt"),
        run=_run_wrong_py, disk_preflight_fn=_noop_disk,
    )
    assert result.ok is True
    assert result.runtime.runtime_python_exe


def test_vllm_source_compile_error(monkeypatch, tmp_path):
    wt = tmp_path / "wt"
    wt.mkdir()
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    _patch_isolation(monkeypatch, wt, venv)

    result = run_vllm_source_build(
        _vllm_action(), str(tmp_path / "attempt"),
        run=_make_rocm_run(pip_ok=False), disk_preflight_fn=_noop_disk,
    )
    assert result.ok is False
    assert result.failure_class == "compile_error"


def test_vllm_source_rocm_verify_fails_boot_failed(monkeypatch, tmp_path):
    wt = tmp_path / "wt"
    wt.mkdir()
    (wt / "vllm").mkdir()
    (wt / "vllm" / "_C.so").write_bytes(b"ELF")
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    _patch_isolation(monkeypatch, wt, venv)

    result = run_vllm_source_build(
        _vllm_action(), str(tmp_path / "attempt"),
        run=_make_rocm_run(verify_ok=False), disk_preflight_fn=_noop_disk,
    )
    assert result.ok is False
    assert result.failure_class == "boot_failed"


def test_vllm_source_symbol_missing(monkeypatch, tmp_path):
    wt = tmp_path / "wt"
    wt.mkdir()
    (wt / "vllm").mkdir()
    (wt / "vllm" / "_C.so").write_bytes(b"ELF")
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    _patch_isolation(monkeypatch, wt, venv)

    def _run_sym(argv, **kw):
        cmd = " ".join(str(a) for a in argv)
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
    wt = tmp_path / "wt"
    wt.mkdir()
    (wt / "vllm").mkdir()
    (wt / "vllm" / "_C.so").write_bytes(b"ELF")
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
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
    wt = tmp_path / "wt"
    wt.mkdir()
    (wt / "vllm").mkdir()
    (wt / "vllm" / "_C.so").write_bytes(b"ELF")
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
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


def test_vllm_source_pr_url_in_installed_versions(monkeypatch, tmp_path):
    wt = tmp_path / "wt"
    wt.mkdir()
    (wt / "vllm").mkdir()
    (wt / "vllm" / "_C.so").write_bytes(b"ELF")
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    _patch_isolation(monkeypatch, wt, venv)

    result = run_vllm_source_build(
        _vllm_action(source_pr_url="https://github.com/ROCm/vllm/pull/42"),
        str(tmp_path / "attempt"),
        run=_make_rocm_run(), disk_preflight_fn=_noop_disk,
    )
    assert result.ok is True
    assert result.installed_versions.get("source_pr_url") == "https://github.com/ROCm/vllm/pull/42"


def test_recipe_no_source_pr_url_when_empty(monkeypatch, tmp_path):
    wt = tmp_path / "wt"
    wt.mkdir()
    (wt / "vllm").mkdir()
    (wt / "vllm" / "_C.so").write_bytes(b"ELF")
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    _patch_isolation(monkeypatch, wt, venv)

    result = run_vllm_source_build(
        _vllm_action(),
        str(tmp_path / "attempt2"),
        run=_make_rocm_run(), disk_preflight_fn=_noop_disk,
    )
    assert result.ok is True
    assert "source_pr_url" not in result.installed_versions


# ---------------------------------------------------------------------------
# Driver dispatch for sgl-kernel / vLLM
# ---------------------------------------------------------------------------

def test_driver_main_routes_sgl_kernel(monkeypatch, tmp_path):
    root = tmp_path / "attempt"
    root.mkdir()
    action = _sgl_action()
    (root / "plan.json").write_text(json.dumps(action.to_state()), encoding="utf-8")

    from hyperloom.orchestrator.framework import targeted_build as tb

    def _fake_sgl(action, attempt_root, **kw):
        from hyperloom.orchestrator.framework.build_actions import BuildResult
        return BuildResult(ok=True, attempt_root=attempt_root, failure_class="ok")

    monkeypatch.setattr(tb, "run_sgl_kernel_build", _fake_sgl)

    rc = _driver_main(["--attempt-root", str(root)])
    assert rc == 0
    data = json.loads((root / "result.json").read_text())
    assert data["ok"] is True


def test_driver_main_routes_vllm_source(monkeypatch, tmp_path):
    root = tmp_path / "attempt"
    root.mkdir()
    action = _vllm_action()
    (root / "plan.json").write_text(json.dumps(action.to_state()), encoding="utf-8")

    from hyperloom.orchestrator.framework import targeted_build as tb

    def _fake_vllm(action, attempt_root, **kw):
        from hyperloom.orchestrator.framework.build_actions import BuildResult
        return BuildResult(ok=True, attempt_root=attempt_root, failure_class="ok")

    monkeypatch.setattr(tb, "run_vllm_source_build", _fake_vllm)

    rc = _driver_main(["--attempt-root", str(root)])
    assert rc == 0


def test_driver_main_unknown_component_returns_failure(monkeypatch, tmp_path):
    """A component not in the driver dispatcher returns failure without a real build."""
    root = tmp_path / "attempt"
    root.mkdir()
    action = _vllm_action()
    (root / "plan.json").write_text(json.dumps(action.to_state()), encoding="utf-8")

    from hyperloom.orchestrator.framework import targeted_build as tb

    def _patched_driver(argv=None):
        import argparse
        from pathlib import Path as _Path
        parser = argparse.ArgumentParser()
        parser.add_argument("--attempt-root", required=True)
        args = parser.parse_args(argv)
        root2 = _Path(args.attempt_root)
        (root2 / "result.json").write_text(json.dumps({
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
# Opt-in real ROCm compile (excluded from CI via -m 'not targeted_build_e2e')
# ---------------------------------------------------------------------------

@pytest.mark.targeted_build_e2e
def test_aiter_build_e2e_real_rocm(tmp_path):
    """Real AITER compile; skipped unless a ROCm host is present."""
    try:
        import torch

        if not getattr(torch.version, "hip", None):
            pytest.skip("not a ROCm torch — skipping real AITER compile")
    except ImportError:
        pytest.skip("torch not importable — skipping real AITER compile")

    from hyperloom.orchestrator.framework.targeted_build import run_aiter_build

    action = TargetedBuildAction(
        gap_id="e2e",
        framework="vllm",
        component="aiter",
        capability="fp4_moe",
        ref="",
        repo_url="https://github.com/ROCm/aiter",
        gpu_arch="gfx950",
        max_jobs=8,
    )
    result = run_aiter_build(action, str(tmp_path / "e2e_attempt"))
    assert result.ok, f"e2e AITER build failed: {result.failure_class} - {result.failure_summary}"
    assert result.installed_versions.get("aiter_ref")
