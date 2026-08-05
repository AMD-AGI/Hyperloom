# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Isolated-venv aiter rebuild strategy and jit/build discovery."""

from __future__ import annotations

import importlib.util
import json
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
    (site / "aiter_meta" / "csrc" / "kernels").mkdir(parents=True)
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
    site = aiter_pkg.parent
    akp._CACHED_KN_TARGET_ROOTS = (str(site) + "/",)

    target = site / "aiter_meta" / "csrc" / "kernels" / "foo.cu"
    strat = akp._detect_strategy(target, allow_unknown_target=False)

    assert strat["compiled"] is True
    assert strat["root"] == str(site)
    assert strat["rebuild_mode"] == "runtime_jit"
    assert strat["rebuild_command"] == []
    assert strat["artifact_roots"] == [aiter_pkg, site / "aiter_meta"]


def test_detect_strategy_isolated_aiter_python_target_never_rebuilds(akp, tmp_path, monkeypatch):
    venv_root, aiter_pkg = _make_isolated_aiter(tmp_path)
    monkeypatch.setenv("VLLM_VENV_ROOT", str(venv_root))
    akp._CACHED_KNOWN_TARGET_ROOTS = (str(aiter_pkg) + "/",)

    target = aiter_pkg / "ops" / "triton" / "k.py"
    strat = akp._detect_strategy(target, allow_unknown_target=False)

    assert strat["compiled"] is False
    assert strat["rebuild_mode"] == "none"
    assert strat["rebuild_command"] == []
    assert strat["artifact_roots"] == []


def test_detect_strategy_sgl_workspace_aiter_unchanged(akp, monkeypatch):
    monkeypatch.setenv("VLLM_VENV_ROOT", "/opt/hyperloom/vllm-venv")
    akp._CACHED_KNOWN_TARGET_ROOTS = ("/sgl-workspace/aiter/",)

    target = Path("/sgl-workspace/aiter/csrc/kernels/foo.cu")
    strat = akp._detect_strategy(target, allow_unknown_target=False)

    assert strat["compiled"] is True
    assert strat["root"] == "/sgl-workspace/aiter"
    assert strat["rebuild_mode"] == "command"
    assert strat["rebuild_command"] == ["/opt/venv/bin/python", "setup.py", "develop"]


def test_apply_isolated_aiter_meta_csrc_defers_to_runtime_jit(
    akp,
    tmp_path,
    monkeypatch,
):
    _venv_root, aiter_pkg = _make_isolated_aiter(tmp_path)
    monkeypatch.delenv("VLLM_VENV_ROOT", raising=False)
    monkeypatch.setattr(akp.importlib.util, "find_spec", lambda name: None)
    site = aiter_pkg.parent
    akp._CACHED_KNOWN_TARGET_ROOTS = (str(site) + "/",)
    target = site / "aiter_meta" / "csrc" / "kernels" / "foo.cu"
    target.write_text(
        '#include <hip/hip_runtime.h>\nextern "C" void kernel() { int x = 1; }\n',
        encoding="utf-8",
    )
    patch = tmp_path / "v1_forge.cu"
    optimized = (
        '#include <hip/hip_runtime.h>\n'
        'extern "C" void kernel() { int x = 2; }\n'
    )
    patch.write_text(optimized, encoding="utf-8")
    (aiter_pkg / "jit" / "build" / "stale.so").write_text(
        "stale",
        encoding="utf-8",
    )

    result = akp.apply_kernel_patch(
        patch_path=patch,
        target_file=target,
        backup_root=tmp_path / "backups",
        kernel_id="k003",
    )

    assert result["status"] == "ok", result
    assert result["rebuild"]["status"] == "deferred"
    assert result["rebuild"]["mode"] == "runtime_jit"
    assert result["jit_build_backup"]["status"] == "ok"
    assert target.read_text(encoding="utf-8") == optimized
    manifest = json.loads(Path(result["manifest_path"]).read_text())
    assert manifest["status"] == "applied"
    assert manifest["strategy"]["rebuild_mode"] == "runtime_jit"
    assert manifest["strategy"]["jit_build_dir"] == str(
        aiter_pkg / "jit" / "build"
    )

    revert = akp.revert_kernel_patch(result["manifest_path"])

    assert revert["status"] == "ok"
    assert target.read_text(encoding="utf-8").endswith("int x = 1; }\n")
    assert (aiter_pkg / "jit" / "build" / "stale.so").is_file()


def test_apply_isolated_aiter_snapshot_invalidates_jit_for_python_codegen(
    akp,
    tmp_path,
    monkeypatch,
):
    _venv_root, aiter_pkg = _make_isolated_aiter(tmp_path)
    monkeypatch.delenv("VLLM_VENV_ROOT", raising=False)
    monkeypatch.setattr(akp.importlib.util, "find_spec", lambda name: None)
    site = aiter_pkg.parent
    akp._CACHED_KNOWN_TARGET_ROOTS = (str(site) + "/",)
    relative = Path("aiter_meta/csrc/kernels/gen_instances.py")
    target = site / relative
    target.write_text("TILE = 128\n", encoding="utf-8")
    patch = tmp_path / "forge.patch"
    patch.write_text(
        "diff --git a/aiter_meta/csrc/kernels/gen_instances.py "
        "b/aiter_meta/csrc/kernels/gen_instances.py\n"
        "--- a/aiter_meta/csrc/kernels/gen_instances.py\n"
        "+++ b/aiter_meta/csrc/kernels/gen_instances.py\n"
        "@@ -1 +1 @@\n"
        "-TILE = 128\n"
        "+TILE = 64\n",
        encoding="utf-8",
    )
    snapshot = tmp_path / "snapshot"
    snapshot_target = snapshot / relative
    snapshot_target.parent.mkdir(parents=True)
    snapshot_target.write_text("TILE = 64\n", encoding="utf-8")
    (aiter_pkg / "jit" / "build" / "stale.so").write_text(
        "stale",
        encoding="utf-8",
    )

    result = akp.apply_kernel_patch(
        patch_path=patch,
        target_file=target,
        backup_root=tmp_path / "backups",
        snapshot_dir=snapshot,
        repo_root=site,
        kernel_id="k006",
    )

    assert result["status"] == "ok", result
    assert result["rebuild"]["status"] == "deferred"
    assert result["rebuild"]["mode"] == "runtime_jit"
    assert result["jit_build_backup"]["status"] == "ok"
    assert target.read_text(encoding="utf-8") == "TILE = 64\n"


def test_runtime_jit_uses_target_root_not_importable_aiter(
    akp,
    tmp_path,
    monkeypatch,
):
    _venv_root, target_aiter = _make_isolated_aiter(tmp_path / "target")
    import_aiter = (
        tmp_path
        / "imported"
        / "lib"
        / "python3.12"
        / "site-packages"
        / "aiter"
    )
    imported_build = import_aiter / "jit" / "build"
    imported_build.mkdir(parents=True)
    (imported_build / "unrelated.so").write_text("unrelated", encoding="utf-8")
    monkeypatch.delenv("VLLM_VENV_ROOT", raising=False)
    monkeypatch.setattr(
        akp.importlib.util,
        "find_spec",
        lambda name: types.SimpleNamespace(
            submodule_search_locations=[str(import_aiter)]
        ),
    )
    site = target_aiter.parent
    akp._CACHED_KNOWN_TARGET_ROOTS = (str(site) + "/",)
    target = site / "aiter_meta" / "csrc" / "kernels" / "foo.cu"
    original = '#include <hip/hip_runtime.h>\nextern "C" void kernel() {}\n'
    optimized = (
        '#include <hip/hip_runtime.h>\n'
        'extern "C" void kernel() { int x = 2; }\n'
    )
    target.write_text(original, encoding="utf-8")
    patch = tmp_path / "v1_forge.cu"
    patch.write_text(optimized, encoding="utf-8")
    (target_aiter / "jit" / "build" / "target.so").write_text(
        "target",
        encoding="utf-8",
    )

    result = akp.apply_kernel_patch(
        patch_path=patch,
        target_file=target,
        backup_root=tmp_path / "backups",
        kernel_id="k003",
    )

    assert result["status"] == "ok", result
    assert result["jit_build_backup"]["src"] == str(
        target_aiter / "jit" / "build"
    )
    assert not (target_aiter / "jit" / "build").exists()
    assert (imported_build / "unrelated.so").is_file()


def test_runtime_jit_rejects_unverified_cache_invalidation(
    akp,
    tmp_path,
    monkeypatch,
):
    _venv_root, aiter_pkg = _make_isolated_aiter(tmp_path)
    site = aiter_pkg.parent
    akp._CACHED_KNOWN_TARGET_ROOTS = (str(site) + "/",)
    target = site / "aiter_meta" / "csrc" / "kernels" / "foo.cu"
    original = '#include <hip/hip_runtime.h>\nextern "C" void kernel() {}\n'
    target.write_text(original, encoding="utf-8")
    patch = tmp_path / "v1_forge.cu"
    patch.write_text(
        '#include <hip/hip_runtime.h>\n'
        'extern "C" void kernel() { int x = 2; }\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        akp,
        "_invalidate_aiter_jit_build",
        lambda *args, **kwargs: {
            "status": "skipped",
            "reason": "aiter package not importable",
        },
    )

    result = akp.apply_kernel_patch(
        patch_path=patch,
        target_file=target,
        backup_root=tmp_path / "backups",
        kernel_id="k003",
    )

    assert result["status"] == "failed"
    assert result["error_class"] == "aiter_jit_invalidation_failed"
    assert target.read_text(encoding="utf-8") == original
