# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""SGLang in-tree JIT kernels must not trigger a framework editable reinstall."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest


_APPLY_TOOL_PATH = Path(__file__).resolve().parent.parent / "tools" / "apply_kernel_patch.py"

_SGLANG_ROOT = "/sgl-workspace/sglang"
_KDA_CUH = f"{_SGLANG_ROOT}/python/sglang/kernels/jit/csrc/attention/kda_packed_decode.cuh"
_KDA_WRAPPER = f"{_SGLANG_ROOT}/python/sglang/kernels/ops/attention/kda_packed_decode.py"
_EDITABLE_REINSTALL = ["/opt/venv/bin/python", "-m", "pip", "install", "-e", "python"]


@pytest.fixture()
def akp(monkeypatch) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("_akp_sglang_jit_under_test", _APPLY_TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "_CACHED_KNOWN_TARGET_ROOTS", (_SGLANG_ROOT + "/",))
    return module


@pytest.mark.parametrize(
    "relative",
    (
        "python/sglang/kernels/jit/csrc/attention/kda_packed_decode.cuh",
        "python/sglang/kernels/jit/csrc/elementwise/add_constant.cu",
        "python/sglang/kernels/jit/include/sgl_kernel/tensor.h",
    ),
)
def test_sglang_jit_source_never_reinstalls_sglang(akp, relative):
    strategy = akp._detect_strategy(Path(_SGLANG_ROOT) / relative)

    assert strategy["compiled"] is True
    assert strategy["root"] == _SGLANG_ROOT
    assert strategy["rebuild_mode"] == "content_addressed_jit"
    assert strategy["rebuild_command"] == []
    assert strategy["artifact_roots"] == []
    assert strategy["jit_build_dir"] == ""


@pytest.mark.parametrize(
    "relative",
    (
        "python/sglang/kernels/ops/attention/kda_packed_decode.py",
        "python/sglang/kernels/jit/utils/compile.py",
    ),
)
def test_sglang_python_target_stays_source_only(akp, relative):
    strategy = akp._detect_strategy(Path(_SGLANG_ROOT) / relative)

    assert strategy["compiled"] is False
    assert strategy["rebuild_mode"] == "none"
    assert strategy["rebuild_command"] == []


def test_sglang_aot_source_keeps_editable_reinstall(akp):
    target = Path(_SGLANG_ROOT) / "python/sglang/kernels/aot/csrc/elementwise/dsv4_norm_rope.cu"

    strategy = akp._detect_strategy(target)

    assert strategy["rebuild_mode"] == "command"
    assert strategy["rebuild_command"] == _EDITABLE_REINSTALL


def test_kda_patch_set_drives_no_editable_reinstall(akp):
    strategies = akp._multi_root_strategies([Path(_KDA_CUH), Path(_KDA_WRAPPER)])

    assert [strategy["rebuild_mode"] for strategy in strategies] == ["content_addressed_jit"]
    assert all(strategy["rebuild_command"] == [] for strategy in strategies)


def test_sglang_jit_rebuild_defers_to_runtime(akp, tmp_path, monkeypatch):
    def _fail(command, cwd, timeout_sec):
        raise AssertionError(f"unexpected rebuild subprocess: {command}")

    monkeypatch.setattr(akp, "_run_rebuild", _fail)
    strategy = akp._detect_strategy(Path(_KDA_CUH))

    result = akp._run_strategy_rebuild(
        strategy,
        command_override=[],
        fallback_cwd=tmp_path,
        timeout_sec=60,
    )

    assert result["status"] == "deferred"
    assert result["mode"] == "content_addressed_jit"
    assert akp._rebuild_ok_to_proceed(result) is True


def test_sglang_jit_needs_no_jit_cache_invalidation(akp):
    strategy = akp._detect_strategy(Path(_KDA_CUH))
    skipped = {"status": "skipped", "reason": "target is outside aiter csrc"}

    assert akp._runtime_jit_invalidation_error(strategy, skipped) == ""
