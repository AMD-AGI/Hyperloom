# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Isolated-venv aiter rebuild strategy and jit/build discovery."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest


_APPLY_TOOL_PATH = Path(__file__).resolve().parent.parent / "tools" / "apply_kernel_patch.py"


@pytest.fixture()
def akp() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("_akp_isolated_aiter_under_test", _APPLY_TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _make_isolated_aiter(tmp_path: Path) -> tuple[Path, Path]:
    venv_root = tmp_path / "vllm-venv"
    site = venv_root / "lib" / "python3.12" / "site-packages"
    aiter_pkg = site / "aiter"
    (aiter_pkg / "jit" / "build").mkdir(parents=True)
    (aiter_pkg / "csrc" / "kernels").mkdir(parents=True)
    (aiter_pkg / "ops" / "triton").mkdir(parents=True)
    return venv_root, aiter_pkg


def test_jit_build_dir_falls_back_to_isolated_venv(akp, tmp_path, monkeypatch):
    venv_root, aiter_pkg = _make_isolated_aiter(tmp_path)
    monkeypatch.setenv("VLLM_VENV_ROOT", str(venv_root))
    # Main process cannot import aiter.
    monkeypatch.setattr(akp.importlib.util, "find_spec", lambda name: None)

    assert akp._aiter_jit_build_dir() == aiter_pkg / "jit" / "build"


def test_jit_build_dir_none_without_isolated_venv(akp, monkeypatch):
    monkeypatch.delenv("VLLM_VENV_ROOT", raising=False)
    monkeypatch.setattr(akp.importlib.util, "find_spec", lambda name: None)

    assert akp._aiter_jit_build_dir() is None


def test_detect_strategy_isolated_aiter_csrc_compiled_no_rebuild_command(akp, tmp_path, monkeypatch):
    venv_root, aiter_pkg = _make_isolated_aiter(tmp_path)
    monkeypatch.setenv("VLLM_VENV_ROOT", str(venv_root))
    akp._CACHED_KNOWN_TARGET_ROOTS = (str(aiter_pkg) + "/",)

    target = aiter_pkg / "csrc" / "kernels" / "foo.cu"
    strat = akp._detect_strategy(target, allow_unknown_target=False)

    assert strat["compiled"] is True
    assert strat["root"] == str(aiter_pkg)
    assert strat["rebuild_command"] == []
    assert strat["artifact_roots"] == [aiter_pkg]


def test_detect_strategy_isolated_aiter_python_target_never_rebuilds(akp, tmp_path, monkeypatch):
    venv_root, aiter_pkg = _make_isolated_aiter(tmp_path)
    monkeypatch.setenv("VLLM_VENV_ROOT", str(venv_root))
    akp._CACHED_KNOWN_TARGET_ROOTS = (str(aiter_pkg) + "/",)

    target = aiter_pkg / "ops" / "triton" / "k.py"
    strat = akp._detect_strategy(target, allow_unknown_target=False)

    assert strat["compiled"] is False
    assert strat["rebuild_command"] == []
    assert strat["artifact_roots"] == []


def test_detect_strategy_sgl_workspace_aiter_unchanged(akp, monkeypatch):
    monkeypatch.setenv("VLLM_VENV_ROOT", "/opt/hyperloom/vllm-venv")
    akp._CACHED_KNOWN_TARGET_ROOTS = ("/sgl-workspace/aiter/",)

    target = Path("/sgl-workspace/aiter/csrc/kernels/foo.cu")
    strat = akp._detect_strategy(target, allow_unknown_target=False)

    assert strat["compiled"] is True
    assert strat["root"] == "/sgl-workspace/aiter"
    assert strat["rebuild_command"] == ["/opt/venv/bin/python", "setup.py", "develop"]
