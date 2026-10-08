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
_AOT_CU = f"{_SGLANG_ROOT}/python/sglang/kernels/aot/csrc/elementwise/dsv4_norm_rope.cu"
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


@pytest.mark.parametrize("jit_first", (True, False))
def test_jit_and_aot_sources_keep_separate_strategies(akp, jit_first):
    """One root now yields two rebuild modes, so dedup must not drop either."""
    paths = [Path(_KDA_CUH), Path(_AOT_CU)]
    if not jit_first:
        paths.reverse()

    strategies = akp._multi_root_strategies(paths)

    assert {strategy["rebuild_mode"] for strategy in strategies} == {"content_addressed_jit", "command"}
    assert [strategy["rebuild_command"] for strategy in strategies if strategy["rebuild_command"]] == [
        _EDITABLE_REINSTALL
    ]


def test_same_root_and_mode_still_rebuilds_once(akp):
    other_aot = Path(_SGLANG_ROOT) / "python/sglang/kernels/aot/csrc/attention/decode.cu"

    strategies = akp._multi_root_strategies([Path(_AOT_CU), other_aot])

    assert [strategy["rebuild_command"] for strategy in strategies] == [_EDITABLE_REINSTALL]


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


def test_sglang_jit_rebuild_runs_no_import_probe(akp):
    strategy = akp._detect_strategy(Path(_KDA_CUH))

    assert strategy["import_probes"] == []


@pytest.mark.parametrize(
    ("relative", "probes"),
    (
        ("python/sglang/srt/layers/attention/foo.cu", ["sglang", "sglang.srt.server_args"]),
        ("sgl-kernel/csrc/foo.cu", ["sgl_kernel"]),
    ),
)
def test_sglang_editable_rebuild_declares_import_probes(akp, relative, probes):
    strategy = akp._detect_strategy(Path(_SGLANG_ROOT) / relative)

    assert strategy["rebuild_command"] != []
    assert strategy["import_probes"] == probes


def test_rebuild_fails_when_reinstall_breaks_imports(akp, monkeypatch, tmp_path):
    monkeypatch.setattr(akp, "_run_rebuild", lambda command, cwd, timeout_sec: {"status": "ok", "returncode": 0})
    monkeypatch.setattr(
        akp,
        "_verify_rebuild_imports",
        lambda probes, interpreter, **kwargs: {
            "status": "failed",
            "probes": probes,
            "error": "sglang.srt.server_args: no module spec",
        },
    )
    strategy = akp._detect_strategy(Path(_SGLANG_ROOT) / "python/sglang/srt/layers/attention/foo.cu")

    result = akp._run_strategy_rebuild(
        strategy,
        command_override=[],
        fallback_cwd=tmp_path,
        timeout_sec=60,
    )

    assert result["status"] == "failed"
    assert "sglang.srt.server_args" in result["error"]
    assert akp._rebuild_ok_to_proceed(result) is False


def test_rebuild_ok_records_import_metadata(akp, monkeypatch, tmp_path):
    monkeypatch.setattr(akp, "_run_rebuild", lambda command, cwd, timeout_sec: {"status": "ok", "returncode": 0})
    strategy = akp._detect_strategy(Path(_SGLANG_ROOT) / "python/sglang/srt/layers/attention/foo.cu")
    captured = {}

    def _probe(probes, interpreter, **kwargs):
        captured.update(probes=probes, interpreter=interpreter)
        return {"status": "ok", "modules": {name: {"importable": True, "origin": f"/x/{name}.py"} for name in probes}}

    monkeypatch.setattr(akp, "_verify_rebuild_imports", _probe)

    result = akp._run_strategy_rebuild(
        strategy,
        command_override=[],
        fallback_cwd=tmp_path,
        timeout_sec=60,
    )

    assert result["status"] == "ok"
    assert captured == {
        "probes": ["sglang", "sglang.srt.server_args"],
        "interpreter": "/opt/venv/bin/python",
    }
    assert result["import_check"]["modules"]["sglang.srt.server_args"]["importable"] is True


def test_import_probe_detects_missing_module(akp):
    result = akp._verify_rebuild_imports(
        ["json", "hyperloom_definitely_not_installed"],
        sys.executable,
        timeout_sec=120,
    )

    assert result["status"] == "failed"
    assert "hyperloom_definitely_not_installed" in result["error"]
    assert result["modules"]["json"]["importable"] is True
    assert result["sys_path"]


def test_import_probe_passes_for_importable_modules(akp):
    result = akp._verify_rebuild_imports(["json", "importlib.util"], sys.executable, timeout_sec=120)

    assert result["status"] == "ok"
    assert result["modules"]["importlib.util"]["importable"] is True


def test_import_probe_skips_non_interpreter_rebuild(akp):
    result = akp._verify_rebuild_imports(["sglang"], "/usr/bin/make")

    assert result["status"] == "skipped"
    assert "not interpreter-led" in result["reason"]
