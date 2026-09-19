# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``_resolve_libexec()`` must cover both ROCm layouts.

The packaged profiling script is executed standalone by the Analysis Agent, so
it is loaded here by path rather than imported as a module.
"""

from __future__ import annotations

import importlib.util
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
