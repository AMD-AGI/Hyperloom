# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``_resolve_libexec()`` must cover both ROCm layouts.

The packaged profiling script is executed standalone by the Analysis Agent, so
it is loaded here by path rather than imported as a module.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "local_knowledge"
    / "common_methodology"
    / "profiling"
    / "rocpc_profile.py"
)


@pytest.fixture(scope="module")
def rocpc_profile() -> ModuleType:
    spec = importlib.util.spec_from_file_location("_rocpc_profile_under_test", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_libexec(root: Path) -> Path:
    libexec = root / "libexec" / "rocprofiler-compute"
    libexec.mkdir(parents=True, exist_ok=True)
    (libexec / "rocprof_compute_base.py").write_text("", encoding="utf-8")
    return libexec


def test_resolves_the_classic_rocm_tree(rocpc_profile, tmp_path, monkeypatch) -> None:
    rocm = tmp_path / "opt" / "rocm"
    libexec = _make_libexec(rocm)
    monkeypatch.setenv("ROCM_PATH", str(rocm))

    assert rocpc_profile._resolve_libexec() == str(libexec)


def test_resolves_the_rocm_profiler_wheel(rocpc_profile, tmp_path, monkeypatch) -> None:
    """On TheRock's pip ROCm the profiler is its own `_rocm_profiler` wheel and
    ROCM_PATH points at `_rocm_sdk_devel`, so a ROCM_PATH-only lookup finds
    nothing and profiling silently degrades to the PMC path."""
    site = tmp_path / "site-packages"
    libexec = _make_libexec(site / "_rocm_profiler")
    (site / "_rocm_profiler" / "__init__.py").write_text("", encoding="utf-8")
    sdk_devel = tmp_path / "site-packages" / "_rocm_sdk_devel"
    sdk_devel.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("ROCM_PATH", str(sdk_devel))
    monkeypatch.syspath_prepend(str(site))

    assert rocpc_profile._resolve_libexec() == str(libexec)


def test_returns_none_when_the_tool_is_absent(rocpc_profile, tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ROCM_PATH", str(tmp_path / "nonexistent"))
    monkeypatch.setattr(sys, "path", [p for p in sys.path if "_rocm_profiler" not in p])

    assert rocpc_profile._resolve_libexec() is None


def _make_venv_python(root: Path) -> Path:
    python = root / "bin" / "python"
    python.parent.mkdir(parents=True, exist_ok=True)
    python.write_text("", encoding="utf-8")
    python.chmod(0o755)
    return python


def test_analyze_prefers_the_dedicated_venv(rocpc_profile, tmp_path, monkeypatch) -> None:
    """analyze gates on the exact pins in the tool's own requirements.txt.

    Satisfying them in the serving image would drag numpy and pandas out from
    under torch, so install.sh puts them in a private venv instead.
    """
    venv = tmp_path / "rocpc-venv"
    python = _make_venv_python(venv)
    monkeypatch.setenv("ROCPC_VENV", str(venv))

    assert rocpc_profile._analyze_python("/nonexistent/libexec") == str(python)


def test_analyze_falls_back_when_the_venv_is_absent(rocpc_profile, tmp_path, monkeypatch) -> None:
    """No venv means the old behaviour, not a hard failure."""
    monkeypatch.setenv("ROCPC_VENV", str(tmp_path / "never-created"))
    monkeypatch.setattr(rocpc_profile, "_detect_rocpc_python", lambda libexec: "/usr/bin/python3")

    assert rocpc_profile._analyze_python("/nonexistent/libexec") == "/usr/bin/python3"


def test_analyze_venv_default_location(rocpc_profile, monkeypatch) -> None:
    """install.sh and this script must agree on the path without being told."""
    monkeypatch.delenv("ROCPC_VENV", raising=False)

    assert rocpc_profile._rocpc_venv_python() == "/opt/rocprof-compute-venv/bin/python"


def _make_sdk_pkg(site: Path, name: str, *, profiler: bool = False, amdsmi: bool = False) -> Path:
    root = site / name
    (root / "lib").mkdir(parents=True, exist_ok=True)
    (root / "__init__.py").write_text("", encoding="utf-8")
    if profiler:
        (root / "lib" / "rocprofiler-sdk").mkdir(parents=True, exist_ok=True)
    if amdsmi:
        (root / "share" / "amd_smi" / "amdsmi").mkdir(parents=True, exist_ok=True)
    return root


def test_profiler_env_points_rocm_path_at_the_package_holding_the_sdk(rocpc_profile, tmp_path, monkeypatch) -> None:
    """The runtime-only wheel stack ships no _rocm_sdk_devel, so ROCM_PATH is
    unset and rocprof-compute cannot resolve its rocprofiler-sdk tool at all --
    it aborts before profiling anything."""
    site = tmp_path / "site"
    core = _make_sdk_pkg(site, "_rocm_sdk_core", profiler=True, amdsmi=True)
    monkeypatch.delenv("ROCM_PATH", raising=False)
    monkeypatch.syspath_prepend(str(site))

    env = rocpc_profile._profiler_env()

    assert env["ROCM_PATH"] == str(core)


def test_profiler_env_keeps_a_rocm_path_that_already_has_the_sdk(rocpc_profile, tmp_path, monkeypatch) -> None:
    site = tmp_path / "site"
    _make_sdk_pkg(site, "_rocm_sdk_core", profiler=True)
    devel = tmp_path / "devel"
    (devel / "lib" / "rocprofiler-sdk").mkdir(parents=True)
    monkeypatch.setenv("ROCM_PATH", str(devel))
    monkeypatch.syspath_prepend(str(site))

    assert rocpc_profile._profiler_env()["ROCM_PATH"] == str(devel)


def test_profiler_env_exposes_the_sdk_own_amdsmi(rocpc_profile, tmp_path, monkeypatch) -> None:
    """amdsmi ships inside the SDK wheel, not on sys.path. The PyPI build cannot
    stand in: it is ROCm 7 era and resolves libamd_smi.so under /opt/rocm."""
    site = tmp_path / "site"
    core = _make_sdk_pkg(site, "_rocm_sdk_core", profiler=True, amdsmi=True)
    monkeypatch.delenv("ROCM_PATH", raising=False)
    monkeypatch.delenv("PYTHONPATH", raising=False)
    monkeypatch.syspath_prepend(str(site))

    env = rocpc_profile._profiler_env()

    assert str(core / "share" / "amd_smi") in env["PYTHONPATH"].split(os.pathsep)


def test_profiler_env_is_unchanged_without_a_wheel_stack(rocpc_profile, tmp_path, monkeypatch) -> None:
    """A classic /opt/rocm image already resolves everything itself."""
    monkeypatch.setenv("ROCM_PATH", "/opt/rocm")
    monkeypatch.setattr(rocpc_profile, "_rocm_profiler_runtime_root", lambda: None)

    assert rocpc_profile._profiler_env()["ROCM_PATH"] == "/opt/rocm"


def test_torch_import_preflight_detects_llvm_option_collisions(rocpc_profile, monkeypatch) -> None:
    """Some prebuilt sglang ROCm10 images cannot profile anything that imports
    torch: rocprofv3 aborts in LLVM option registration before the workload even
    reaches GPU code. Detect that quickly instead of hanging inside profile."""
    monkeypatch.setattr(rocpc_profile.shutil, "which", lambda name, path=None: "/opt/venv/bin/rocprofv3")

    def fake_run(*_args, **_kwargs):
        return subprocess.CompletedProcess(
            _args,
            137,
            "",
            "CommandLine Error: Option 'spirv-expand-step' registered more than once!\n"
            "LLVM ERROR: inconsistency in registered CommandLine options\n",
        )

    monkeypatch.setattr(rocpc_profile.subprocess, "run", fake_run)

    ok, msg = rocpc_profile._torch_import_under_rocprofv3("/usr/bin/python3")

    assert not ok
    assert "LLVM option registry" in msg
    assert "spirv-expand-step" in msg


def test_torch_import_preflight_allows_a_working_profiler(rocpc_profile, monkeypatch) -> None:
    monkeypatch.setattr(rocpc_profile.shutil, "which", lambda name, path=None: "/opt/venv/bin/rocprofv3")
    monkeypatch.setattr(
        rocpc_profile.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(_args, 0, "ok\n", ""),
    )

    ok, msg = rocpc_profile._torch_import_under_rocprofv3("/usr/bin/python3")

    assert ok
    assert msg == ""
