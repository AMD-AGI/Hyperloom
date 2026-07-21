# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Mocked AITER build matrix tests (S4).

All subprocess, git, and disk/toolchain calls are mocked.  No GPU, compiler,
or network connection is required.  Each test exercises one branch of the
run_aiter_build -> poll_build -> result.json flow.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from hyperloom.orchestrator.framework.build_actions import TargetedBuildAction
from hyperloom.orchestrator.framework.targeted_build import (
    _driver_main,
    _load_result_json,
    run_aiter_build,
)


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

def _completed(stdout="", stderr="", returncode=0):
    return SimpleNamespace(stdout=stdout, stderr=stderr, returncode=returncode)


def _action(**kw):
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


class FakeIsolation:
    """Replaces isolation.prepare_repo_cache + prepare_candidate_workspace."""

    def __init__(self, worktree_dir: Path, venv_dir: Path):
        self.worktree_dir = worktree_dir
        self.venv_dir = venv_dir

    def prepare_repo_cache(self, req):
        return self.worktree_dir

    def prepare_candidate_workspace(self, req, candidate, *, index, execute):
        self.venv_dir.mkdir(parents=True, exist_ok=True)
        return SimpleNamespace(
            worktree_dir=self.worktree_dir,
            venv_dir=self.venv_dir,
        )


def _patch_isolation(monkeypatch, tmp_path, *, make_so=True, so_age=1.0):
    """Monkeypatch isolation functions in targeted_build module namespace."""
    from hyperloom.orchestrator.framework import targeted_build as tb_module
    import hyperloom.orchestrator.framework.targeted_build as tb_module

    worktree = tmp_path / "worktree"
    worktree.mkdir(parents=True, exist_ok=True)
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True, exist_ok=True)
    (venv / "bin" / "python").write_text("#!/usr/bin/env python3\n")

    if make_so:
        so = worktree / "module_aiter_core.so"
        so.write_bytes(b"\x7fELF")

    fake = FakeIsolation(worktree, venv)

    import hyperloom.agents.framework.isolation as iso_mod
    monkeypatch.setattr(iso_mod, "_run_git", lambda *a, **kw: None)
    monkeypatch.setattr(iso_mod, "_run_subprocess", lambda *a, **kw: None)

    # Patch the isolation imports inside run_aiter_build to use our fakes
    monkeypatch.setattr(
        "hyperloom.agents.framework.isolation.prepare_repo_cache",
        fake.prepare_repo_cache,
    )
    monkeypatch.setattr(
        "hyperloom.agents.framework.isolation.prepare_candidate_workspace",
        fake.prepare_candidate_workspace,
    )
    return worktree, venv


def _make_run(
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
    """Build an injectable run callable covering all subprocess branches."""

    def _run(argv, capture_output=False, text=False, timeout=3600,
             env=None, cwd=None, **kw):
        import json as _j
        cmd = " ".join(str(a) for a in argv)
        # probe_torch_abi json probe (must be before generic "hip" check)
        if "is_rocm" in cmd:
            data = {
                "torch_version": torch_ver,
                "hip_version": "7.2.53211",
                "python_version": "3.12.13",
                "is_rocm": hip_ok,
            }
            return _completed(stdout=_j.dumps(data) + "\n")
        # Toolchain probe
        if "which hipcc" in cmd:
            return _completed(stdout=toolchain_hipcc + "\n" if toolchain_hipcc else "")
        if "dirname" in cmd:
            root = Path(toolchain_hipcc).parent.parent if toolchain_hipcc else Path("/opt/rocm")
            return _completed(stdout=str(root) + "\n")
        # ROCm hip assertion probe (write_rocm_torch_constraints)
        if "sys.exit(0" in cmd:
            return _completed(returncode=0 if hip_ok else 2)
        # version probes (write_rocm_torch_constraints)
        if "argv[1]" in cmd:
            if argv and argv[-1] == "triton":
                return _completed(stdout=triton_ver + "\n" if triton_ver else "")
            return _completed(stdout=torch_ver + "\n")
        # pip install
        if "pip" in cmd and "install" in cmd:
            return _completed(returncode=0 if pip_ok else 1,
                              stderr="" if pip_ok else "error: compile failed")
        # import aiter
        if "import aiter" in cmd:
            return _completed(returncode=0 if import_ok else 1)
        # git tags
        if "tag -l" in cmd:
            return _completed(stdout=aiter_tags)
        # git checkout
        if "git" in cmd and "checkout" in cmd:
            return _completed()
        # git rev-parse
        if "rev-parse" in cmd:
            return _completed(stdout=git_sha + "\n")
        return _completed()

    return _run


def _make_git(*, aiter_tags="v0.1.0\nv0.0.9", git_sha="abc1234"):
    """Injectable git runner for run_aiter_build."""
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


def _noop_disk_preflight(path, n, *, per_candidate_gb=1.5):
    pass  # always passes


# ---------------------------------------------------------------------------
# Test: success (pinned ref)
# ---------------------------------------------------------------------------

def test_run_aiter_build_success_pinned_ref(monkeypatch, tmp_path):
    worktree, venv = _patch_isolation(monkeypatch, tmp_path, make_so=True)
    run = _make_run()
    git = _make_git()

    result = run_aiter_build(
        _action(ref="v0.1.0"),
        str(tmp_path / "attempt"),
        run=run, git=git,
        disk_preflight_fn=_noop_disk_preflight,
    )

    assert result.ok is True
    assert result.failure_class == "ok"
    assert result.runtime.pythonpath_prefixes
    assert result.installed_versions["aiter_ref"] == "v0.1.0"
    assert result.installed_versions["arch"] == "gfx950"
    assert "INFERENCE_OPTIMIZER_AITER_JIT_DIR" in result.runtime.runtime_env


def test_run_aiter_build_success_runtime_paths_valid(monkeypatch, tmp_path):
    worktree, venv = _patch_isolation(monkeypatch, tmp_path, make_so=True)
    result = run_aiter_build(
        _action(ref="v0.1.0"),
        str(tmp_path / "attempt"),
        run=_make_run(), git=_make_git(),
        disk_preflight_fn=_noop_disk_preflight,
    )
    assert result.ok
    jit = result.runtime.runtime_env["INFERENCE_OPTIMIZER_AITER_JIT_DIR"]
    assert "attempt" in jit
    # live-tree guard: $HOME/.aiter must NOT be referenced
    assert ".aiter" not in jit or "attempt" in jit


# ---------------------------------------------------------------------------
# Test: compile_error (pip fails)
# ---------------------------------------------------------------------------

def test_run_aiter_build_compile_error(monkeypatch, tmp_path):
    _patch_isolation(monkeypatch, tmp_path, make_so=False)
    result = run_aiter_build(
        _action(ref="v0.1.0"),
        str(tmp_path / "attempt"),
        run=_make_run(pip_ok=False), git=_make_git(),
        disk_preflight_fn=_noop_disk_preflight,
    )
    assert result.ok is False
    assert result.failure_class == "compile_error"


# ---------------------------------------------------------------------------
# Test: abi_mismatch (non-ROCm torch)
# ---------------------------------------------------------------------------

def test_run_aiter_build_abi_mismatch(monkeypatch, tmp_path):
    _patch_isolation(monkeypatch, tmp_path)
    result = run_aiter_build(
        _action(ref="v0.1.0"),
        str(tmp_path / "attempt"),
        run=_make_run(hip_ok=False), git=_make_git(),
        disk_preflight_fn=_noop_disk_preflight,
    )
    assert result.ok is False
    assert result.failure_class in ("abi_mismatch", "preflight_toolchain")


# ---------------------------------------------------------------------------
# Test: symbol_missing
# ---------------------------------------------------------------------------

def test_run_aiter_build_symbol_missing(monkeypatch, tmp_path):
    _patch_isolation(monkeypatch, tmp_path, make_so=True)

    def _run_missing(argv, **kw):
            cmd = " ".join(str(a) for a in argv)
            if "fp4_moe" in cmd:
                return _completed(returncode=1)
            return _make_run()(argv, **kw)

    result = run_aiter_build(
        _action(ref="v0.1.0", expected_symbols=("aiter.ops.fp4_moe",)),
        str(tmp_path / "attempt"),
        run=_run_missing, git=_make_git(),
        disk_preflight_fn=_noop_disk_preflight,
    )
    assert result.ok is False
    assert result.failure_class == "symbol_missing"


# ---------------------------------------------------------------------------
# Test: autoselect hit
# ---------------------------------------------------------------------------

def test_run_aiter_build_autoselect_hit(monkeypatch, tmp_path):
    _patch_isolation(monkeypatch, tmp_path, make_so=True)
    result = run_aiter_build(
        _action(ref=""),  # empty ref -> autoselect
        str(tmp_path / "attempt"),
        run=_make_run(aiter_tags="v0.2.0\nv0.1.0"),
        git=_make_git(aiter_tags="v0.2.0\nv0.1.0"),
        disk_preflight_fn=_noop_disk_preflight,
    )
    assert result.ok is True
    # newest tag selected
    assert result.installed_versions["aiter_ref"] == "v0.2.0"


# ---------------------------------------------------------------------------
# Test: autoselect exhaust (all tags fail)
# ---------------------------------------------------------------------------

def test_run_aiter_build_autoselect_exhaust(monkeypatch, tmp_path):
    _patch_isolation(monkeypatch, tmp_path, make_so=False)
    result = run_aiter_build(
        _action(ref=""),
        str(tmp_path / "attempt"),
        run=_make_run(pip_ok=False, aiter_tags="v0.1.0"),
        git=_make_git(aiter_tags="v0.1.0"),
        disk_preflight_fn=_noop_disk_preflight,
    )
    assert result.ok is False
    assert result.failure_class == "compile_error"
    assert "no AITER tag" in result.failure_summary


# ---------------------------------------------------------------------------
# Test: no live-tree / $HOME mutation
# ---------------------------------------------------------------------------

def test_run_aiter_build_no_live_tree_mutation(monkeypatch, tmp_path):
    """Install-step env must use per-attempt AITER_ROOT_DIR and JIT dir."""
    install_envs: list[dict] = []

    def _capturing_run(argv, **kw):
        cmd = " ".join(str(a) for a in argv)
        if kw.get("env") and ("pip" in cmd and "install" in cmd):
            install_envs.append(dict(kw["env"]))
        return _make_run()(argv, **kw)

    _patch_isolation(monkeypatch, tmp_path, make_so=True)
    result = run_aiter_build(
        _action(ref="v0.1.0"),
        str(tmp_path / "attempt"),
        run=_capturing_run, git=_make_git(),
        disk_preflight_fn=_noop_disk_preflight,
    )
    assert result.ok, result.failure_summary
    assert install_envs, "no install env captured"
    for env in install_envs:
        home = env.get("HOME", "")
        aiter_root = env.get("AITER_ROOT_DIR", "")
        jit = env.get("INFERENCE_OPTIMIZER_AITER_JIT_DIR", "")
        # Per-attempt dirs must not be the global $HOME/.aiter
        if aiter_root:
            assert os.path.expanduser("~/.aiter") not in aiter_root, (
                f"AITER_ROOT_DIR={aiter_root!r} points at global $HOME/.aiter"
            )
        if jit:
            assert os.path.expanduser("~") not in jit or "attempt" in jit, (
                f"JIT dir {jit!r} looks like global $HOME path"
            )
        # Must be under attempt_root
        attempt_root = str(tmp_path / "attempt")
        for key, val in [("HOME", home), ("AITER_ROOT_DIR", aiter_root),
                         ("INFERENCE_OPTIMIZER_AITER_JIT_DIR", jit)]:
            if val:
                assert attempt_root in val, (
                    f"{key}={val!r} is not under attempt_root={attempt_root!r}"
                )


# ---------------------------------------------------------------------------
# Test: reproducible installed_versions + artifact hashes
# ---------------------------------------------------------------------------

def test_run_aiter_build_reproducible_versions(monkeypatch, tmp_path):
    _patch_isolation(monkeypatch, tmp_path, make_so=True)
    result = run_aiter_build(
        _action(ref="v0.1.0", gpu_arch="gfx950"),
        str(tmp_path / "attempt"),
        run=_make_run(torch_ver="2.10.0+git8514f05", git_sha="deadbeef"),
        git=_make_git(git_sha="deadbeef"),
        disk_preflight_fn=_noop_disk_preflight,
    )
    assert result.ok
    v = result.installed_versions
    assert "2.10.0" in v["torch"]
    assert v["aiter_sha"] == "deadbeef"
    assert v["arch"] == "gfx950"


def test_run_aiter_build_source_pr_url_in_installed_versions(monkeypatch, tmp_path):
    _patch_isolation(monkeypatch, tmp_path, make_so=True)
    result = run_aiter_build(
        _action(ref="v0.1.0", gpu_arch="gfx950",
                source_pr_url="https://github.com/ROCm/aiter/pull/77"),
        str(tmp_path / "attempt"),
        run=_make_run(), git=_make_git(),
        disk_preflight_fn=_noop_disk_preflight,
    )
    assert result.ok
    assert result.installed_versions.get("source_pr_url") == "https://github.com/ROCm/aiter/pull/77"


def test_run_aiter_build_no_source_pr_url_when_empty(monkeypatch, tmp_path):
    _patch_isolation(monkeypatch, tmp_path, make_so=True)
    result = run_aiter_build(
        _action(ref="v0.1.0", gpu_arch="gfx950"),
        str(tmp_path / "attempt"),
        run=_make_run(), git=_make_git(),
        disk_preflight_fn=_noop_disk_preflight,
    )
    assert result.ok
    assert "source_pr_url" not in result.installed_versions


# ---------------------------------------------------------------------------
# Test: result.json round-trip through poll_build
# ---------------------------------------------------------------------------

def test_result_json_round_trip_through_poll_build(tmp_path):
    """Driver writes result.json; poll_build reads it as a rich BuildResult."""
    from hyperloom.orchestrator.framework.build_actions import BuildResult, FrameworkRuntime
    from hyperloom.orchestrator.framework.targeted_build import (
        BuildHandle,
        poll_build,
    )

    root = tmp_path / "attempt"
    root.mkdir()

    # Simulate a successful build result
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

    # Build a fake handle whose process has already exited (rc=0)
    proc = SimpleNamespace(returncode=0, poll=lambda: 0)
    handle = BuildHandle(
        action=_action(ref="v0.1.0"),
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


# ---------------------------------------------------------------------------
# Test: _driver_main plan.json -> result.json
# ---------------------------------------------------------------------------

def test_driver_main_writes_result_json(monkeypatch, tmp_path):
    root = tmp_path / "attempt"
    root.mkdir()

    action = _action(ref="v0.1.0")
    (root / "plan.json").write_text(json.dumps(action.to_state()), encoding="utf-8")

    # Patch run_aiter_build inside the driver
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
