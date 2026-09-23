# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Isolated-venv aiter rebuild strategy and jit/build discovery."""

from __future__ import annotations

import importlib.util
import json
import os
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
    (aiter_pkg / "__init__.py").write_text("", encoding="utf-8")
    (aiter_pkg / "jit" / "__init__.py").write_text("", encoding="utf-8")
    (site / "aiter_meta" / "csrc" / "kernels").mkdir(parents=True)
    return venv_root, aiter_pkg


def _make_editable_aiter(tmp_path: Path) -> tuple[Path, Path]:
    """Build the ``/sgl-workspace/aiter`` shape: a checkout, not site-packages."""
    checkout = tmp_path / "sgl-workspace" / "aiter"
    aiter_pkg = checkout / "aiter"
    (aiter_pkg / "jit" / "build").mkdir(parents=True)
    (aiter_pkg / "__init__.py").write_text("", encoding="utf-8")
    (aiter_pkg / "jit" / "__init__.py").write_text("", encoding="utf-8")
    (checkout / "csrc" / "kernels").mkdir(parents=True)
    return checkout, aiter_pkg


def test_jit_build_dir_falls_back_to_isolated_venv(akp, tmp_path, monkeypatch):
    venv_root, aiter_pkg = _make_isolated_aiter(tmp_path)
    monkeypatch.setenv("VLLM_VENV_ROOT", str(venv_root))
    # Main process cannot import aiter.
    monkeypatch.setattr(akp.aiter_jit_cache.importlib.util, "find_spec", lambda name: None)

    assert akp._aiter_jit_build_dir() == aiter_pkg / "jit" / "build"


def test_jit_build_dir_none_without_isolated_venv(akp, monkeypatch):
    monkeypatch.delenv("VLLM_VENV_ROOT", raising=False)
    monkeypatch.setattr(akp.aiter_jit_cache.importlib.util, "find_spec", lambda name: None)

    assert akp._aiter_jit_build_dir() is None


def test_runtime_trust_discovers_the_package_once_without_importing(akp, tmp_path, monkeypatch):
    _venv, package = _make_isolated_aiter(tmp_path)
    (package / "__init__.py").write_text("raise AssertionError('AITER must not be imported')\n", encoding="utf-8")
    monkeypatch.delitem(sys.modules, "aiter", raising=False)
    monkeypatch.syspath_prepend(str(package.parent))
    jit = tmp_path / "runtime" / "build"
    monkeypatch.setenv("AITER_JIT_DIR", str(jit.parent))
    find_spec = importlib.util.find_spec
    discovered = []

    def discover_once(name):
        discovered.append(name)
        return find_spec(name) if len(discovered) == 1 else None

    monkeypatch.setattr(importlib.util, "find_spec", discover_once)

    assert akp._trusted_aiter_jit_build_dir(jit) is True
    assert discovered == ["aiter"]
    assert "aiter" not in sys.modules


def test_detect_strategy_isolated_aiter_csrc_compiled_no_rebuild_command(akp, tmp_path, monkeypatch):
    venv_root, aiter_pkg = _make_isolated_aiter(tmp_path)
    monkeypatch.setenv("VLLM_VENV_ROOT", str(venv_root))
    site = aiter_pkg.parent
    monkeypatch.setattr(
        akp,
        "_CACHED_KNOWN_TARGET_ROOTS",
        (str(site) + "/",),
    )

    target = site / "aiter_meta" / "csrc" / "kernels" / "foo.cu"
    strat = akp._detect_strategy(target)

    assert strat["compiled"] is True
    assert strat["root"] == str(site)
    assert strat["rebuild_mode"] == "runtime_jit"
    assert strat["rebuild_command"] == []
    assert strat["artifact_roots"] == []


def test_detect_strategy_isolated_aiter_python_target_never_rebuilds(akp, tmp_path, monkeypatch):
    venv_root, aiter_pkg = _make_isolated_aiter(tmp_path)
    monkeypatch.setenv("VLLM_VENV_ROOT", str(venv_root))
    monkeypatch.setattr(
        akp,
        "_CACHED_KNOWN_TARGET_ROOTS",
        (str(aiter_pkg) + "/",),
    )

    target = aiter_pkg / "ops" / "triton" / "k.py"
    strat = akp._detect_strategy(target)

    assert strat["compiled"] is False
    assert strat["rebuild_mode"] == "none"
    assert strat["rebuild_command"] == []
    assert strat["artifact_roots"] == []


def test_installed_aiter_strategy_preserves_symlinked_site_packages(
    akp,
    tmp_path,
    monkeypatch,
):
    real_site = tmp_path / "real" / "site-packages"
    (real_site / "aiter" / "jit").mkdir(parents=True)
    (real_site / "aiter" / "__init__.py").write_text("", encoding="utf-8")
    (real_site / "aiter" / "jit" / "__init__.py").write_text("", encoding="utf-8")
    (real_site / "aiter_meta" / "csrc" / "kernels").mkdir(parents=True)
    linked_site = tmp_path / "venv" / "lib" / "python3.12" / "site-packages"
    linked_site.parent.mkdir(parents=True)
    linked_site.symlink_to(real_site, target_is_directory=True)
    monkeypatch.setattr(
        akp,
        "_CACHED_KNOWN_TARGET_ROOTS",
        (str(linked_site) + "/",),
    )
    target = linked_site / "aiter_meta" / "csrc" / "kernels" / "foo.cu"

    strategy = akp._detect_strategy(target)

    assert strategy["root"] == str(linked_site.absolute())
    assert strategy["rebuild_mode"] == "runtime_jit"
    assert strategy["jit_build_dir"] == str(linked_site.absolute() / "aiter" / "jit" / "build")


def test_detect_strategy_sgl_workspace_aiter_unchanged(akp, monkeypatch):
    monkeypatch.setenv("VLLM_VENV_ROOT", "/opt/hyperloom/vllm-venv")
    monkeypatch.setattr(
        akp,
        "_CACHED_KNOWN_TARGET_ROOTS",
        ("/sgl-workspace/aiter/",),
    )

    target = Path("/sgl-workspace/aiter/csrc/kernels/foo.cu")
    strat = akp._detect_strategy(target)

    assert strat["compiled"] is True
    assert strat["root"] == "/sgl-workspace/aiter"
    assert strat["rebuild_mode"] == "command"
    assert strat["rebuild_command"] == ["/opt/venv/bin/python", "setup.py", "develop"]
    assert strat["jit_build_dir"] == "/sgl-workspace/aiter/aiter/jit/build"
    assert Path(strat["jit_build_dir"]) == akp._EDITABLE_AITER_ROOT / "aiter" / "jit" / "build"


def test_installed_wheel_jit_build_dir_stays_trusted(akp, tmp_path):
    _venv_root, aiter_pkg = _make_isolated_aiter(tmp_path)

    assert akp._trusted_aiter_jit_build_dir(aiter_pkg / "jit" / "build") is True


def test_jit_build_dir_outside_every_known_root_is_rejected(
    akp,
    tmp_path,
    monkeypatch,
):
    """A forged manifest naming an aiter-shaped tree is still an rmtree target."""
    checkout, _aiter_pkg = _make_editable_aiter(tmp_path)
    monkeypatch.setattr(akp, "_EDITABLE_AITER_ROOT", checkout)
    forged = tmp_path / "attacker" / "aiter"
    (forged / "jit" / "build").mkdir(parents=True)
    (forged / "__init__.py").write_text("", encoding="utf-8")
    (forged / "jit" / "__init__.py").write_text("", encoding="utf-8")

    assert akp._trusted_aiter_jit_build_dir(forged / "jit" / "build") is False


def test_editable_root_without_package_markers_is_rejected(
    akp,
    tmp_path,
    monkeypatch,
):
    """A bare directory tree is not an importable aiter package."""
    checkout = tmp_path / "sgl-workspace" / "aiter"
    (checkout / "aiter" / "jit" / "build").mkdir(parents=True)
    monkeypatch.setattr(akp, "_EDITABLE_AITER_ROOT", checkout)

    assert akp._trusted_aiter_jit_build_dir(checkout / "aiter" / "jit" / "build") is False


def test_symlinked_site_packages_wheel_stays_trusted(akp, tmp_path):
    """site-packages is commonly a symlink; the wheel behind it is still a wheel."""
    _venv_root, aiter_pkg = _make_isolated_aiter(tmp_path)
    linked_site = tmp_path / "linked" / "lib" / "python3.12"
    linked_site.mkdir(parents=True)
    (linked_site / "site-packages").symlink_to(aiter_pkg.parent)

    trusted = akp._trusted_aiter_jit_build_dir(linked_site / "site-packages" / "aiter" / "jit" / "build")

    assert trusted is True


def test_symlink_loop_in_the_manifest_path_is_rejected(akp, tmp_path):
    """An untrusted path that cannot be resolved is not trusted, and does not raise."""
    loop = tmp_path / "loop"
    other = tmp_path / "other"
    loop.symlink_to(other)
    other.symlink_to(loop)

    assert akp._trusted_aiter_jit_build_dir(loop / "aiter" / "jit" / "build") is False


def test_editable_jit_build_survives_invalidate_then_restore(
    akp,
    tmp_path,
    monkeypatch,
):
    """The round trip an editable aiter revert performs, end to end."""
    checkout, aiter_pkg = _make_editable_aiter(tmp_path)
    monkeypatch.setattr(akp, "_EDITABLE_AITER_ROOT", checkout)
    jit_build = aiter_pkg / "jit" / "build"
    (jit_build / "baseline.so").write_text("baseline", encoding="utf-8")
    backup_root = tmp_path / "backups"

    invalidated = akp._invalidate_aiter_jit_build(
        checkout / "csrc" / "kernels" / "foo.cu",
        backup_root,
        jit_build_dir=jit_build,
    )
    assert invalidated["status"] == "ok", invalidated
    assert not jit_build.exists()

    restored = akp._restore_aiter_jit_build(
        invalidated,
        expected_jit_build_dir=str(jit_build),
        backup_root=backup_root,
    )

    assert restored["status"] == "ok", restored
    assert restored["restored_to"] == str(jit_build)
    assert (jit_build / "baseline.so").read_text(encoding="utf-8") == "baseline"


@pytest.mark.parametrize("layout", ("editable", "installed"))
def test_jit_transaction_restores_top_level_modules_and_build_for_known_layouts(akp, tmp_path, monkeypatch, layout):
    if layout == "editable":
        checkout, aiter_pkg = _make_editable_aiter(tmp_path)
        monkeypatch.setattr(akp, "_EDITABLE_AITER_ROOT", checkout)
        target = checkout / "csrc" / "kernels" / "foo.cu"
    else:
        _venv_root, aiter_pkg = _make_isolated_aiter(tmp_path)
        target = aiter_pkg.parent / "aiter_meta" / "csrc" / "kernels" / "foo.cu"
    target.write_text("int tile = 128;\n", encoding="utf-8")
    build = aiter_pkg / "jit" / "build"
    baseline = build / "baseline.o"
    baseline.write_bytes(b"baseline build")
    served = build.parent / "module.so"
    served.write_bytes(b"baseline module")
    backup_root = tmp_path / "backups"

    record = akp._invalidate_aiter_jit_build(target, backup_root, jit_build_dir=build)

    assert not served.exists(), "first use must not import the old native module"
    assert not build.exists()
    assert record["status"] == "ok", record
    assert record["src"] == str(build)
    assert record["build_existed"] is True
    assert record["module_scope"] is None
    assert record["module_names"] == [served.name]
    saved_build = Path(record["backup_path"])
    saved_modules = Path(record["modules_backup_path"])
    assert saved_build.parent == backup_root
    assert saved_modules.parent == backup_root
    assert (saved_build / baseline.name).read_bytes() == b"baseline build"
    assert (saved_modules / served.name).read_bytes() == b"baseline module"

    build.mkdir()
    candidate_build = build / "candidate.o"
    candidate_build.write_bytes(b"candidate build")
    served.write_bytes(b"candidate module")
    candidate_only = build.parent / "candidateonly.so"
    candidate_only.write_bytes(b"candidate-only module")

    restored = akp._restore_aiter_jit_build(record, expected_jit_build_dir=str(build), backup_root=backup_root)

    assert restored["status"] == "ok", restored
    assert restored["restored_to"] == str(build)
    assert served.read_bytes() == b"baseline module"
    assert baseline.read_bytes() == b"baseline build"
    assert not candidate_only.exists()
    assert not candidate_build.exists()


@pytest.mark.parametrize(
    "relative",
    (
        "csrc/kernels/gen_instances.py",
        "csrc/cpp_itfs/mha_fwd.py",
    ),
)
def test_editable_aiter_csrc_python_keeps_source_only_strategy(
    akp,
    monkeypatch,
    relative,
):
    monkeypatch.setattr(
        akp,
        "_CACHED_KNOWN_TARGET_ROOTS",
        ("/sgl-workspace/aiter/",),
    )

    strategy = akp._detect_strategy(Path("/sgl-workspace/aiter") / relative)

    assert strategy["compiled"] is False
    assert strategy["root"] == "/sgl-workspace/aiter"
    assert strategy["rebuild_mode"] == "none"
    assert strategy["rebuild_command"] == []
    assert strategy["artifact_roots"] == []


def test_rebuild_strategy_uses_target_parent_for_legacy_strategy(
    akp,
    tmp_path,
    monkeypatch,
):
    target_parent = tmp_path / "aiter_meta" / "csrc" / "kernels"
    captured = {}

    def _fake_rebuild(command, cwd, timeout_sec):
        captured.update(
            command=command,
            cwd=cwd,
            timeout_sec=timeout_sec,
        )
        return {"status": "ok"}

    monkeypatch.setattr(akp, "_run_rebuild", _fake_rebuild)

    result = akp._run_strategy_rebuild(
        {"rebuild_command": []},
        command_override=["python", "build.py"],
        fallback_cwd=target_parent,
        timeout_sec=123,
    )

    assert result["status"] == "ok"
    assert captured == {
        "command": ["python", "build.py"],
        "cwd": target_parent,
        "timeout_sec": 123,
    }


def test_unknown_snapshot_layout_keeps_fail_fast_root(
    akp,
    tmp_path,
    monkeypatch,
):
    framework_root = tmp_path / "app" / "ATOM" / "atom"
    target = framework_root / "kernels" / "foo.cu"
    target.parent.mkdir(parents=True)
    original = 'extern "C" void kernel() {}\n'
    optimized = 'extern "C" void kernel() { int x = 2; }\n'
    target.write_text(original, encoding="utf-8")
    monkeypatch.setattr(
        akp,
        "_CACHED_KNOWN_TARGET_ROOTS",
        (str(framework_root) + "/",),
    )
    strategy = akp._detect_strategy(target)
    assert strategy["root"] == ""
    assert strategy["deploy_roots"] == []
    patch = tmp_path / "forge.patch"
    patch.write_text(
        "diff --git a/kernels/foo.cu b/kernels/foo.cu\n"
        "--- a/kernels/foo.cu\n"
        "+++ b/kernels/foo.cu\n"
        "@@ -1 +1 @@\n"
        '-extern "C" void kernel() {}\n'
        '+extern "C" void kernel() { int x = 2; }\n',
        encoding="utf-8",
    )
    snapshot_target = tmp_path / "snapshot" / "kernels" / "foo.cu"
    snapshot_target.parent.mkdir(parents=True)
    snapshot_target.write_text(optimized, encoding="utf-8")

    result = akp.apply_kernel_patch(
        patch_path=patch,
        target_file=target,
        backup_root=tmp_path / "backups",
        snapshot_dir=tmp_path / "snapshot",
        kernel_id="k-unknown-layout",
    )

    assert result["status"] == "failed"
    assert "snapshot mode requires a known repo root" in result["error"]
    assert target.read_text(encoding="utf-8") == original
    assert not (target.parent / "kernels" / "foo.cu").exists()


def test_apply_snapshot_keeps_legacy_optional_deploy_roots(
    akp,
    tmp_path,
):
    repo_root = tmp_path / "repo"
    target = repo_root / "kernel.py"
    target.parent.mkdir(parents=True)
    target.write_text("VALUE = 1\n", encoding="utf-8")
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "kernel.py").write_text("VALUE = 2\n", encoding="utf-8")

    result = akp.apply_snapshot(
        descriptors=[
            {
                "op": "write",
                "path": "kernel.py",
                "mode": "",
                "binary": False,
                "is_new": False,
            }
        ],
        snapshot_dir=snapshot,
        repo_root=repo_root,
        backup_dir=tmp_path / "backups",
    )

    assert result["status"] == "ok"
    assert target.read_text(encoding="utf-8") == "VALUE = 2\n"


@pytest.mark.parametrize(
    "relative",
    (
        "aiter/jit/hostcall.cpp",
        "aiter_meta/3rdparty/composable_kernel/include/tile.hpp",
    ),
)
def test_runtime_jit_invalidates_for_compiled_sources_outside_csrc(
    akp,
    tmp_path,
    monkeypatch,
    relative,
):
    _venv_root, aiter_pkg = _make_isolated_aiter(tmp_path)
    site = aiter_pkg.parent
    monkeypatch.setattr(
        akp,
        "_CACHED_KNOWN_TARGET_ROOTS",
        (str(site) + "/",),
    )
    target = site / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    original = "inline int kernel_entry() { return 1; }\n"
    optimized = "inline int kernel_entry() { return 2; }\n"
    target.write_text(original, encoding="utf-8")
    patch = tmp_path / f"v1_forge{target.suffix}"
    patch.write_text(optimized, encoding="utf-8")
    (aiter_pkg / "jit" / "build" / "stale.so").write_text(
        "stale",
        encoding="utf-8",
    )

    result = akp.apply_kernel_patch(
        patch_path=patch,
        target_file=target,
        backup_root=tmp_path / "backups",
        kernel_id="k-outside-csrc",
    )

    assert result["status"] == "ok", result
    assert result["rebuild"]["status"] == "deferred"
    assert result["jit_build_backup"]["status"] == "ok"
    assert not (aiter_pkg / "jit" / "build").exists()
    assert target.read_text(encoding="utf-8") == optimized


def test_apply_isolated_aiter_meta_csrc_defers_to_runtime_jit(
    akp,
    tmp_path,
    monkeypatch,
):
    _venv_root, aiter_pkg = _make_isolated_aiter(tmp_path)
    monkeypatch.delenv("VLLM_VENV_ROOT", raising=False)
    monkeypatch.setattr(akp.aiter_jit_cache.importlib.util, "find_spec", lambda name: None)
    site = aiter_pkg.parent
    monkeypatch.setattr(
        akp,
        "_CACHED_KNOWN_TARGET_ROOTS",
        (str(site) + "/",),
    )
    target = site / "aiter_meta" / "csrc" / "kernels" / "foo.cu"
    target.write_text(
        '#include <hip/hip_runtime.h>\nextern "C" void kernel() { int x = 1; }\n',
        encoding="utf-8",
    )
    patch = tmp_path / "v1_forge.cu"
    optimized = '#include <hip/hip_runtime.h>\nextern "C" void kernel() { int x = 2; }\n'
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
    assert result["artifact_count"] == 0
    assert target.read_text(encoding="utf-8") == optimized
    manifest = json.loads(Path(result["manifest_path"]).read_text())
    assert manifest["status"] == "applied"
    assert manifest["strategy"]["rebuild_modes"] == ["runtime_jit"]
    assert manifest["strategy"]["jit_build_dirs"] == [str(aiter_pkg / "jit" / "build")]

    monkeypatch.setattr(
        akp,
        "_CACHED_KNOWN_TARGET_ROOTS",
        ("/unrelated/static/root/",),
    )
    revert = akp.revert_kernel_patch(result["manifest_path"])

    assert revert["status"] == "ok"
    assert revert["jit_build_restore"]["status"] == "ok"
    assert target.read_text(encoding="utf-8").endswith("int x = 1; }\n")
    assert (aiter_pkg / "jit" / "build" / "stale.so").is_file()


def test_apply_isolated_aiter_snapshot_invalidates_jit_for_python_codegen(
    akp,
    tmp_path,
    monkeypatch,
):
    _venv_root, aiter_pkg = _make_isolated_aiter(tmp_path)
    monkeypatch.delenv("VLLM_VENV_ROOT", raising=False)
    monkeypatch.setattr(akp.aiter_jit_cache.importlib.util, "find_spec", lambda name: None)
    site = aiter_pkg.parent
    monkeypatch.setattr(
        akp,
        "_CACHED_KNOWN_TARGET_ROOTS",
        (str(site) + "/",),
    )
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
        kernel_id="k006",
    )

    assert result["status"] == "ok", result
    assert result["rebuild"]["status"] == "deferred"
    assert result["rebuild"]["mode"] == "runtime_jit"
    assert result["jit_build_backup"]["status"] == "ok"
    assert result["artifact_count"] == 0
    assert target.read_text(encoding="utf-8") == "TILE = 64\n"
    snapshot_manifest = json.loads(Path(result["manifest_path"]).read_text())
    assert snapshot_manifest["strategy"]["root"] == str(site)
    assert snapshot_manifest["strategy"]["deploy_roots"] == [
        str(site / "aiter"),
        str(site / "aiter_meta"),
    ]
    assert snapshot_manifest["strategy"]["rebuild_modes"] == ["runtime_jit"]


def test_revert_exposes_runtime_jit_restore_failure(
    akp,
    tmp_path,
    monkeypatch,
):
    _venv_root, aiter_pkg = _make_isolated_aiter(tmp_path)
    site = aiter_pkg.parent
    monkeypatch.setattr(
        akp,
        "_CACHED_KNOWN_TARGET_ROOTS",
        (str(site) + "/",),
    )
    target = site / "aiter_meta" / "csrc" / "kernels" / "foo.cu"
    original = '#include <hip/hip_runtime.h>\nextern "C" void kernel() {}\n'
    target.write_text(original, encoding="utf-8")
    patch = tmp_path / "v1_forge.cu"
    patch.write_text(
        '#include <hip/hip_runtime.h>\nextern "C" void kernel() { int x = 2; }\n',
        encoding="utf-8",
    )
    (aiter_pkg / "jit" / "build" / "stale.so").write_text(
        "stale",
        encoding="utf-8",
    )
    applied = akp.apply_kernel_patch(
        patch_path=patch,
        target_file=target,
        backup_root=tmp_path / "backups",
        kernel_id="k003",
    )
    assert applied["status"] == "ok", applied
    monkeypatch.setattr(
        akp,
        "_restore_aiter_jit_build",
        lambda *args, **kwargs: {
            "status": "failed",
            "error": "simulated restore failure",
        },
    )

    reverted = akp.revert_kernel_patch(applied["manifest_path"])

    assert reverted["status"] == "partial"
    assert reverted["jit_build_restore"] == {
        "status": "failed",
        "error": "simulated restore failure",
    }
    assert reverted["revert_issues"][0]["kind"] == "jit_build_restore"
    assert target.read_text(encoding="utf-8") == original


def test_finalize_keeps_patch_and_deletes_local_backups(
    akp,
    tmp_path,
    monkeypatch,
):
    _venv_root, aiter_pkg = _make_isolated_aiter(tmp_path)
    site = aiter_pkg.parent
    monkeypatch.setattr(
        akp,
        "_CACHED_KNOWN_TARGET_ROOTS",
        (str(site) + "/",),
    )
    target = site / "aiter_meta" / "csrc" / "kernels" / "foo.cu"
    target.write_text(
        '#include <hip/hip_runtime.h>\nextern "C" void kernel() {}\n',
        encoding="utf-8",
    )
    patch = tmp_path / "v1_forge.cu"
    optimized = '#include <hip/hip_runtime.h>\nextern "C" void kernel() { int x = 2; }\n'
    patch.write_text(optimized, encoding="utf-8")
    (aiter_pkg / "jit" / "build" / "baseline.so").write_text(
        "baseline",
        encoding="utf-8",
    )
    applied = akp.apply_kernel_patch(
        patch_path=patch,
        target_file=target,
        backup_root=tmp_path / "backups",
        kernel_id="k-finalize",
    )
    manifest = json.loads(Path(applied["manifest_path"]).read_text())
    source_backup = Path(manifest["source_backup"]["backup_path"])
    jit_backup = Path(manifest["jit_build_backup"]["backup_path"])

    finalized = akp.finalize_kernel_patch(applied["manifest_path"])

    assert finalized["status"] == "ok"
    assert target.read_text(encoding="utf-8") == optimized
    assert not source_backup.exists()
    assert not jit_backup.exists()


def test_finalize_rejects_reverted_manifest(
    akp,
    tmp_path,
    monkeypatch,
):
    _venv_root, aiter_pkg = _make_isolated_aiter(tmp_path)
    site = aiter_pkg.parent
    monkeypatch.setattr(
        akp,
        "_CACHED_KNOWN_TARGET_ROOTS",
        (str(site) + "/",),
    )
    target = site / "aiter_meta" / "csrc" / "kernels" / "foo.cu"
    target.write_text(
        '#include <hip/hip_runtime.h>\nextern "C" void kernel() {}\n',
        encoding="utf-8",
    )
    patch = tmp_path / "v1_forge.cu"
    patch.write_text(
        '#include <hip/hip_runtime.h>\nextern "C" void kernel() { int x = 2; }\n',
        encoding="utf-8",
    )
    applied = akp.apply_kernel_patch(
        patch_path=patch,
        target_file=target,
        backup_root=tmp_path / "backups",
        kernel_id="k-reverted",
    )
    assert akp.revert_kernel_patch(applied["manifest_path"])["status"] == "ok"

    finalized = akp.finalize_kernel_patch(applied["manifest_path"])

    assert finalized["status"] == "failed"
    assert "reverted" in finalized["error"]


def test_finalize_never_deletes_manifest_directory(akp, tmp_path):
    backup_root = tmp_path / "backup"
    backup_root.mkdir()
    manifest = backup_root / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "status": "applied",
                "source_backup": {
                    "backup_path": str(backup_root),
                },
            }
        ),
        encoding="utf-8",
    )

    result = akp.finalize_kernel_patch(manifest)

    assert result["status"] == "partial"
    assert manifest.is_file()
    assert result["issues"][0]["error"] == "backup path equals manifest directory"


@pytest.mark.parametrize("multinode", (False, True))
def test_installed_snapshot_can_span_aiter_and_vllm(
    akp,
    tmp_path,
    monkeypatch,
    multinode,
):
    _venv_root, aiter_pkg = _make_isolated_aiter(tmp_path)
    monkeypatch.delenv("VLLM_VENV_ROOT", raising=False)
    monkeypatch.setattr(akp.aiter_jit_cache.importlib.util, "find_spec", lambda name: None)
    site = aiter_pkg.parent
    vllm = site / "vllm"
    vllm.mkdir()
    monkeypatch.setattr(
        akp,
        "_CACHED_KNOWN_TARGET_ROOTS",
        (str(site) + "/",),
    )
    aiter_relative = Path("aiter_meta/csrc/kernels/gen_instances.py")
    vllm_relative = Path("vllm/_aiter_ops.py")
    aiter_target = site / aiter_relative
    vllm_target = site / vllm_relative
    aiter_target.write_text("TILE = 128\n", encoding="utf-8")
    vllm_target.write_text("ENABLED = False\n", encoding="utf-8")
    patch = tmp_path / "forge.patch"
    patch.write_text(
        "diff --git a/aiter_meta/csrc/kernels/gen_instances.py "
        "b/aiter_meta/csrc/kernels/gen_instances.py\n"
        "--- a/aiter_meta/csrc/kernels/gen_instances.py\n"
        "+++ b/aiter_meta/csrc/kernels/gen_instances.py\n"
        "@@ -1 +1 @@\n"
        "-TILE = 128\n"
        "+TILE = 64\n"
        "diff --git a/vllm/_aiter_ops.py b/vllm/_aiter_ops.py\n"
        "--- a/vllm/_aiter_ops.py\n"
        "+++ b/vllm/_aiter_ops.py\n"
        "@@ -1 +1 @@\n"
        "-ENABLED = False\n"
        "+ENABLED = True\n",
        encoding="utf-8",
    )
    snapshot = tmp_path / "snapshot"
    for relative, content in (
        (aiter_relative, "TILE = 64\n"),
        (vllm_relative, "ENABLED = True\n"),
    ):
        destination = snapshot / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8")
    (aiter_pkg / "jit" / "build" / "stale.so").write_text(
        "stale",
        encoding="utf-8",
    )
    if multinode:
        monkeypatch.setattr(akp, "_is_multi_node", lambda: True)

        def _fake_dispatch(**kwargs):
            return {
                "status": "ok",
                "per_node": [
                    {
                        "host": "pod-a",
                        "target_path": str(kwargs["target_file"]),
                        "backup_path": (f"/var/kernel_patch_backups/{Path(kwargs['target_file']).name}.bak"),
                        "jit_backup": (
                            {
                                "status": "clean",
                                "src": kwargs["jit_build_dir"],
                            }
                            if kwargs["jit_build_dir"]
                            else {
                                "status": "skipped",
                                "reason": "not requested",
                            }
                        ),
                    }
                ],
            }

        monkeypatch.setattr(
            akp,
            "_dispatch_multinode_apply",
            _fake_dispatch,
        )

    result = akp.apply_kernel_patch(
        patch_path=patch,
        target_file=aiter_target,
        backup_root=tmp_path / "backups",
        snapshot_dir=snapshot,
        kernel_id="k-cross-framework",
    )

    assert result["status"] == "ok", result
    assert aiter_target.read_text(encoding="utf-8") == "TILE = 64\n"
    assert vllm_target.read_text(encoding="utf-8") == "ENABLED = True\n"
    manifest = json.loads(Path(result["manifest_path"]).read_text())
    assert manifest["strategy"]["deploy_roots"] == [
        str(site / "aiter"),
        str(site / "aiter_meta"),
        str(site / "vllm"),
    ]
    if multinode:
        records = result["multinode"]["records_by_host"]["pod-a"]
        assert len(records) == 2
        assert len({record["backup_path"] for record in records}) == 2
        assert sum((record.get("jit_backup") or {}).get("status") == "clean" for record in records) == 1


@pytest.mark.parametrize("explicit_repo_root", (False, True))
def test_installed_snapshot_rejects_other_site_packages(
    akp,
    tmp_path,
    monkeypatch,
    explicit_repo_root,
):
    _venv_root, aiter_pkg = _make_isolated_aiter(tmp_path)
    site = aiter_pkg.parent
    torch = site / "torch"
    torch.mkdir()
    monkeypatch.setattr(
        akp,
        "_CACHED_KNOWN_TARGET_ROOTS",
        (str(site) + "/",),
    )
    aiter_relative = Path("aiter_meta/csrc/kernels/gen_instances.py")
    torch_relative = Path("torch/runtime.py")
    aiter_target = site / aiter_relative
    torch_target = site / torch_relative
    aiter_target.write_text("TILE = 128\n", encoding="utf-8")
    torch_target.write_text("VALUE = 1\n", encoding="utf-8")
    patch = tmp_path / "forge.patch"
    patch.write_text(
        "diff --git a/aiter_meta/csrc/kernels/gen_instances.py "
        "b/aiter_meta/csrc/kernels/gen_instances.py\n"
        "--- a/aiter_meta/csrc/kernels/gen_instances.py\n"
        "+++ b/aiter_meta/csrc/kernels/gen_instances.py\n"
        "@@ -1 +1 @@\n"
        "-TILE = 128\n"
        "+TILE = 64\n"
        "diff --git a/torch/runtime.py b/torch/runtime.py\n"
        "--- a/torch/runtime.py\n"
        "+++ b/torch/runtime.py\n"
        "@@ -1 +1 @@\n"
        "-VALUE = 1\n"
        "+VALUE = 2\n",
        encoding="utf-8",
    )
    snapshot = tmp_path / "snapshot"
    for relative, content in (
        (aiter_relative, "TILE = 64\n"),
        (torch_relative, "VALUE = 2\n"),
    ):
        destination = snapshot / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8")

    result = akp.apply_kernel_patch(
        patch_path=patch,
        target_file=aiter_target,
        backup_root=tmp_path / "backups",
        snapshot_dir=snapshot,
        repo_root=site if explicit_repo_root else None,
        kernel_id="k-forbidden-package",
    )

    assert result["status"] == "failed"
    assert "outside authorized deploy roots" in result["error"]
    assert aiter_target.read_text(encoding="utf-8") == "TILE = 128\n"
    assert torch_target.read_text(encoding="utf-8") == "VALUE = 1\n"


def test_installed_snapshot_rejects_symlink_to_other_package(
    akp,
    tmp_path,
    monkeypatch,
):
    _venv_root, aiter_pkg = _make_isolated_aiter(tmp_path)
    site = aiter_pkg.parent
    (site / "torch").mkdir()
    monkeypatch.setattr(
        akp,
        "_CACHED_KNOWN_TARGET_ROOTS",
        (str(site) + "/",),
    )
    target = site / "aiter_meta" / "csrc" / "kernels" / "gen_instances.py"
    target.write_text("TILE = 128\n", encoding="utf-8")
    relative = Path("aiter_meta/csrc/kernels/runtime.py")
    patch = tmp_path / "forge.patch"
    patch.write_text(
        "diff --git a/aiter_meta/csrc/kernels/runtime.py "
        "b/aiter_meta/csrc/kernels/runtime.py\n"
        "new file mode 120000\n"
        "--- /dev/null\n"
        "+++ b/aiter_meta/csrc/kernels/runtime.py\n"
        "@@ -0,0 +1 @@\n"
        "+../../../torch/runtime.py\n",
        encoding="utf-8",
    )
    snapshot_link = tmp_path / "snapshot" / relative
    snapshot_link.parent.mkdir(parents=True)
    snapshot_link.symlink_to("../../../torch/runtime.py")

    result = akp.apply_kernel_patch(
        patch_path=patch,
        target_file=target,
        backup_root=tmp_path / "backups",
        snapshot_dir=tmp_path / "snapshot",
        kernel_id="k-forbidden-symlink",
    )

    assert result["status"] == "failed"
    assert "symlink target is outside authorized deploy roots" in result["error"]
    assert not (site / relative).exists()


def test_runtime_jit_uses_target_root_not_importable_aiter(
    akp,
    tmp_path,
    monkeypatch,
):
    _venv_root, target_aiter = _make_isolated_aiter(tmp_path / "target")
    import_aiter = tmp_path / "imported" / "lib" / "python3.12" / "site-packages" / "aiter"
    imported_build = import_aiter / "jit" / "build"
    imported_build.mkdir(parents=True)
    (imported_build / "unrelated.so").write_text("unrelated", encoding="utf-8")
    monkeypatch.delenv("VLLM_VENV_ROOT", raising=False)
    monkeypatch.setattr(
        akp.aiter_jit_cache.importlib.util,
        "find_spec",
        lambda name: types.SimpleNamespace(submodule_search_locations=[str(import_aiter)]),
    )
    site = target_aiter.parent
    monkeypatch.setattr(
        akp,
        "_CACHED_KNOWN_TARGET_ROOTS",
        (str(site) + "/",),
    )
    target = site / "aiter_meta" / "csrc" / "kernels" / "foo.cu"
    original = '#include <hip/hip_runtime.h>\nextern "C" void kernel() {}\n'
    optimized = '#include <hip/hip_runtime.h>\nextern "C" void kernel() { int x = 2; }\n'
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
    assert result["jit_build_backup"]["src"] == str(target_aiter / "jit" / "build")
    assert not (target_aiter / "jit" / "build").exists()
    assert (imported_build / "unrelated.so").is_file()


@pytest.mark.parametrize("override", ("", "private-jit"))
def test_installed_runtime_strategy_uses_shared_jit_override(akp, tmp_path, monkeypatch, override):
    _venv, package = _make_isolated_aiter(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AITER_JIT_DIR", override)
    target = package.parent / "aiter_meta" / "csrc" / "kernels" / "foo.cu"

    strategy = akp._detect_strategy(target)

    expected = str(tmp_path / override / "build") if override else ""
    assert strategy["jit_build_dir"] == expected


@pytest.mark.parametrize("snapshot_mode", (False, True))
@pytest.mark.parametrize("discoverable", (False, True))
def test_installed_custom_cache_requires_restore_trust_before_mutation(
    akp, tmp_path, monkeypatch, snapshot_mode, discoverable
):
    venv, package = _make_isolated_aiter(tmp_path)
    monkeypatch.setattr(akp.aiter_jit_cache.importlib.util, "find_spec", lambda name: None)
    monkeypatch.delenv("VLLM_VENV_ROOT", raising=False)
    if discoverable:
        monkeypatch.setenv("VLLM_VENV_ROOT", str(venv))
    monkeypatch.setattr(akp, "_is_multi_node", lambda: False)
    jit = tmp_path / "private-jit"
    jit.mkdir()
    served = jit / "module.so"
    served.write_bytes(b"baseline")
    monkeypatch.setenv("AITER_JIT_DIR", str(jit))
    relative = "aiter_meta/csrc/kernels/foo.cu"
    target = package.parent / relative
    original = "int tile = 128;\n"
    target.write_text(original, encoding="utf-8")
    patch = tmp_path / "optimized.cu"
    patch.write_text("int tile = 64;\n", encoding="utf-8")
    kwargs = {}
    if snapshot_mode:
        snapshot = tmp_path / "snapshot"
        destination = snapshot / relative
        destination.parent.mkdir(parents=True)
        destination.write_bytes(patch.read_bytes())
        patch = tmp_path / "change.patch"
        patch.write_text(
            f"diff --git a/{relative} b/{relative}\n--- a/{relative}\n+++ b/{relative}\n"
            "@@ -1 +1 @@\n-int tile = 128;\n+int tile = 64;\n",
            encoding="utf-8",
        )
        kwargs = {"snapshot_dir": snapshot}
    mutations = []
    real_copy = akp.shutil.copy2

    def observe_copy(src, dst, **options):
        if Path(dst) == target:
            mutations.append(str(src))
        return real_copy(src, dst, **options)

    monkeypatch.setattr(akp.shutil, "copy2", observe_copy)
    result = akp.apply_kernel_patch(patch_path=patch, target_file=target, backup_root=tmp_path / "backups", **kwargs)

    if discoverable:
        assert result["status"] == "ok", result
        assert not served.exists()
        assert akp.revert_kernel_patch(result["manifest_path"])["status"] == "ok"
    else:
        assert result["status"] == "failed", result
        assert "untrusted" in result["error"]
        assert mutations == []
    assert target.read_text(encoding="utf-8") == original
    assert served.read_bytes() == b"baseline"


def test_local_tool_loads_shared_core_outside_repository(tmp_path):
    import subprocess

    script = """
import importlib.util
import sys
from pathlib import Path
spec = importlib.util.spec_from_file_location('standalone_apply', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert module.aiter_jit_cache.__file__ == str(Path(sys.argv[1]).resolve().parents[3] / 'common' / 'aiter_jit_cache.py')
assert 'aiter' not in sys.modules
assert 'torch' not in sys.modules
"""
    proc = subprocess.run(
        [sys.executable, "-I", "-S", "-B", "-c", script, str(_APPLY_TOOL_PATH)],
        cwd=tmp_path,
        env={key: value for key, value in os.environ.items() if key != "PYTHONPATH"},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr


def test_runtime_jit_rejects_unverified_cache_invalidation(
    akp,
    tmp_path,
    monkeypatch,
):
    _venv_root, aiter_pkg = _make_isolated_aiter(tmp_path)
    site = aiter_pkg.parent
    monkeypatch.setattr(
        akp,
        "_CACHED_KNOWN_TARGET_ROOTS",
        (str(site) + "/",),
    )
    target = site / "aiter_meta" / "csrc" / "kernels" / "foo.cu"
    original = '#include <hip/hip_runtime.h>\nextern "C" void kernel() {}\n'
    target.write_text(original, encoding="utf-8")
    patch = tmp_path / "v1_forge.cu"
    patch.write_text(
        '#include <hip/hip_runtime.h>\nextern "C" void kernel() { int x = 2; }\n',
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


def test_multinode_runtime_jit_is_invalidated_on_every_pod(
    akp,
    tmp_path,
    monkeypatch,
):
    _venv_root, aiter_pkg = _make_isolated_aiter(tmp_path)
    site = aiter_pkg.parent
    monkeypatch.setattr(
        akp,
        "_CACHED_KNOWN_TARGET_ROOTS",
        (str(site) + "/",),
    )
    monkeypatch.setattr(akp, "_is_multi_node", lambda: True)
    target = site / "aiter_meta" / "csrc" / "kernels" / "foo.cu"
    target.write_text(
        '#include <hip/hip_runtime.h>\nextern "C" void kernel() {}\n',
        encoding="utf-8",
    )
    patch = tmp_path / "v1_forge.cu"
    patch.write_text(
        '#include <hip/hip_runtime.h>\nextern "C" void kernel() { int x = 2; }\n',
        encoding="utf-8",
    )
    local_stale = aiter_pkg / "jit" / "build" / "local.so"
    local_stale.write_text("local", encoding="utf-8")
    captured = {}

    def _fake_dispatch(**kwargs):
        captured.update(kwargs)
        return {
            "status": "ok",
            "per_node": [
                {
                    "host": host,
                    "target_path": str(target),
                    "backup_path": f"/var/kernel_patch_backups/{host}.bak",
                    "jit_backup": {
                        "status": "clean",
                        "src": kwargs["jit_build_dir"],
                    },
                }
                for host in ("pod-a", "pod-b")
            ],
        }

    monkeypatch.setattr(akp, "_dispatch_multinode_apply", _fake_dispatch)

    result = akp.apply_kernel_patch(
        patch_path=patch,
        target_file=target,
        backup_root=tmp_path / "backups",
        kernel_id="k-multinode",
    )

    assert result["status"] == "ok", result
    assert captured["jit_build_dir"] == str(aiter_pkg / "jit" / "build")
    assert result["jit_build_backup"]["status"] == "remote"
    assert set(result["multinode"]["records_by_host"]) == {"pod-a", "pod-b"}
    assert local_stale.is_file()


def _make_importable_checkout(akp, tmp_path, monkeypatch):
    checkout = tmp_path / "custom-checkout"
    package = checkout / "aiter"
    (package / "jit" / "build").mkdir(parents=True)
    (package / "__init__.py").write_text("raise RuntimeError('must not import aiter')\n", encoding="utf-8")
    (package / "jit" / "__init__.py").write_text("", encoding="utf-8")
    (checkout / "csrc" / "kernels").mkdir(parents=True)
    monkeypatch.setattr(
        akp.aiter_jit_cache.importlib.util,
        "find_spec",
        lambda name: types.SimpleNamespace(submodule_search_locations=[str(package)]),
    )
    monkeypatch.delenv("AITER_JIT_DIR", raising=False)
    monkeypatch.delenv("AITER_META_DIR", raising=False)
    monkeypatch.setattr(akp, "_is_multi_node", lambda: False)
    monkeypatch.setattr(akp, "_clear_python_kernel_caches", lambda target: {"status": "ok"})
    return checkout, package


@pytest.mark.parametrize("relative", ("csrc/kernels/quant_kernels.cu", "csrc/kernels/gen_instances.py"))
@pytest.mark.parametrize("override_jit", (False, True))
def test_importable_checkout_native_snapshot_invalidates_served_modules(
    akp, tmp_path, monkeypatch, relative, override_jit
):
    checkout, package = _make_importable_checkout(akp, tmp_path, monkeypatch)
    jit = tmp_path / "private-jit" if override_jit else package / "jit"
    if override_jit:
        monkeypatch.setenv("AITER_JIT_DIR", str(jit))
    build = jit / "build"
    build.mkdir(parents=True, exist_ok=True)
    (build / "baseline.o").write_bytes(b"old build")
    served = jit / "module_quant.so"
    served.write_bytes(b"old native module")
    target = checkout / relative
    original = "TILE = 128\n" if target.suffix == ".py" else "int tile = 128;\n"
    optimized = original.replace("128", "64")
    target.write_text(original, encoding="utf-8")
    snapshot = tmp_path / "snapshot"
    destination = snapshot / relative
    destination.parent.mkdir(parents=True)
    destination.write_text(optimized, encoding="utf-8")
    patch = tmp_path / "change.patch"
    patch.write_text(
        f"diff --git a/{relative} b/{relative}\n--- a/{relative}\n+++ b/{relative}\n"
        f"@@ -1 +1 @@\n-{original}+{optimized}",
        encoding="utf-8",
    )

    result = akp.apply_kernel_patch(
        patch_path=patch,
        target_file=target,
        backup_root=tmp_path / "backups",
        snapshot_dir=snapshot,
        repo_root=checkout,
        kernel_id="checkout-native",
    )

    assert result["status"] == "ok", result
    assert result["compiled"] is True
    assert result["rebuild"]["status"] == "deferred"
    assert result["rebuild"]["mode"] == "runtime_jit"
    assert not served.exists(), "first use must not import the old native module"
    assert not build.exists()
    assert (package / "jit" / "__init__.py").is_file()
    # Model the runtime's cache miss and first-use compilation without invoking a GPU/compiler.
    build.mkdir()
    (build / "candidate.o").write_bytes(optimized.encode())
    served.write_bytes(optimized.encode())
    (jit / "module_new.so").write_bytes(b"candidate-only module")

    reverted = akp.revert_kernel_patch(result["manifest_path"])

    assert reverted["status"] == "ok", reverted
    assert served.read_bytes() == b"old native module"
    assert (build / "baseline.o").read_bytes() == b"old build"
    assert not (build / "candidate.o").exists()
    assert not (jit / "module_new.so").exists()
    assert target.read_text(encoding="utf-8") == original


@pytest.mark.parametrize("had_build", (False, True))
def test_jit_transaction_invalidates_top_level_so_even_without_build(akp, tmp_path, monkeypatch, had_build):
    _checkout, package = _make_importable_checkout(akp, tmp_path, monkeypatch)
    build = package / "jit" / "build"
    if not had_build:
        build.rmdir()
    served = build.parent / "module_quant.so"
    served.write_bytes(b"baseline")
    backup_root = tmp_path / "backups"
    record = akp._invalidate_aiter_jit_build(package / "dummy.cu", backup_root, jit_build_dir=build)

    assert record["status"] == "ok", record
    assert not served.exists()
    served.write_bytes(b"candidate")
    restored = akp._restore_aiter_jit_build(record, expected_jit_build_dir=build, backup_root=backup_root)
    assert restored["status"] == "ok", restored
    assert served.read_bytes() == b"baseline"
    assert build.exists() is had_build


def test_importable_checkout_does_not_claim_unrelated_source_or_python_ops(akp, tmp_path, monkeypatch):
    checkout, _package = _make_importable_checkout(akp, tmp_path, monkeypatch)
    unknown = akp._detect_strategy(tmp_path / "unrelated" / "csrc" / "kernel.cu")
    assert unknown["rebuild_mode"] == "none"
    assert unknown["root"] == ""
    strategy = akp._detect_strategy(checkout / "aiter" / "ops" / "triton" / "kernel.py")
    assert strategy["compiled"] is False
    assert strategy["rebuild_mode"] == "none"


def test_importable_checkout_honors_source_metadata_override(akp, tmp_path, monkeypatch):
    _checkout, package = _make_importable_checkout(akp, tmp_path, monkeypatch)
    meta = tmp_path / "device-sources"
    (meta / "csrc").mkdir(parents=True)
    monkeypatch.setenv("AITER_META_DIR", str(meta))

    strategy = akp._detect_strategy(meta / "csrc" / "kernel.cu")

    assert strategy["root"] == str(meta)
    assert strategy["rebuild_mode"] == "runtime_jit"
    assert strategy["jit_build_dir"] == str(package / "jit" / "build")
    inactive = akp._detect_strategy(_checkout / "csrc" / "kernels" / "old.cu")
    assert inactive["rebuild_mode"] == "none"
    monkeypatch.delenv("AITER_META_DIR")
    assert akp._detect_strategy(meta / "csrc" / "kernel.cu")["rebuild_mode"] == "none"


def test_clean_checkout_jit_revert_removes_first_use_artifacts(akp, tmp_path, monkeypatch):
    _checkout, package = _make_importable_checkout(akp, tmp_path, monkeypatch)
    build = package / "jit" / "build"
    build.rmdir()
    backup_root = tmp_path / "backups"
    record = akp._invalidate_aiter_jit_build(package / "dummy.cu", backup_root, jit_build_dir=build)
    assert record["status"] == "clean", record
    build.mkdir()
    (build / "candidate.o").write_bytes(b"candidate build")
    served = build.parent / "module_quant.so"
    served.write_bytes(b"candidate module")

    restored = akp._restore_aiter_jit_build(record, expected_jit_build_dir=build, backup_root=backup_root)

    assert restored["status"] == "ok", restored
    assert not served.exists()
    assert not build.exists()


def test_missing_serving_module_backup_preserves_candidate_cache(akp, tmp_path, monkeypatch):
    _checkout, package = _make_importable_checkout(akp, tmp_path, monkeypatch)
    build = package / "jit" / "build"
    served = build.parent / "module_quant.so"
    served.write_bytes(b"baseline")
    backup_root = tmp_path / "backups"
    record = akp._invalidate_aiter_jit_build(package / "dummy.cu", backup_root, jit_build_dir=build)
    saved = Path(record["modules_backup_path"])
    (saved / served.name).unlink()
    saved.rmdir()
    served.write_bytes(b"candidate")

    restored = akp._restore_aiter_jit_build(record, expected_jit_build_dir=build, backup_root=backup_root)

    assert restored["status"] == "failed"
    assert "backup path missing" in restored["error"]
    assert served.read_bytes() == b"candidate"


@pytest.mark.parametrize("failed_rebuild", (False, True))
def test_checkout_native_rebuild_failure_and_finalize(akp, tmp_path, monkeypatch, failed_rebuild):
    checkout, package = _make_importable_checkout(akp, tmp_path, monkeypatch)
    target = checkout / "csrc" / "kernels" / "kernel.cu"
    original = "int tile = 128;\n"
    target.write_text(original, encoding="utf-8")
    patch = tmp_path / "optimized.cu"
    patch.write_text("int tile = 64;\n", encoding="utf-8")
    served = package / "jit" / "module_quant.so"
    served.write_bytes(b"baseline")
    monkeypatch.setattr(akp, "_run_rebuild", lambda *args: {"status": "failed", "returncode": 1})

    result = akp.apply_kernel_patch(
        patch_path=patch,
        target_file=target,
        backup_root=tmp_path / "backups",
        kernel_id="native-outcome",
        rebuild_command=["python", "build.py"] if failed_rebuild else None,
    )

    if failed_rebuild:
        assert result["status"] == "failed", result
        assert result["revert"]["status"] == "ok"
        assert target.read_text(encoding="utf-8") == original
        assert served.read_bytes() == b"baseline"
    else:
        assert result["status"] == "ok", result
        manifest = json.loads(Path(result["manifest_path"]).read_text())
        saved = Path(manifest["jit_build_backup"]["modules_backup_path"])
        assert saved.exists()
        served.write_bytes(b"candidate")
        finalized = akp.finalize_kernel_patch(result["manifest_path"])
        assert finalized["status"] == "ok", finalized
        assert not saved.exists()
        assert served.read_bytes() == b"candidate"


def test_jit_cache_move_failure_restores_already_moved_files(akp, tmp_path, monkeypatch):
    checkout, package = _make_importable_checkout(akp, tmp_path, monkeypatch)
    build = package / "jit" / "build"
    (build / "baseline.o").write_bytes(b"baseline build")
    served = build.parent / "module_quant.so"
    served.write_bytes(b"baseline module")
    real_move = akp.shutil.move

    def fail_module_move(src, dst):
        if Path(src) == served:
            raise OSError("simulated module move failure")
        return real_move(src, dst)

    monkeypatch.setattr(akp.shutil, "move", fail_module_move)
    result = akp._invalidate_aiter_jit_build(checkout / "csrc" / "kernel.cu", tmp_path / "backup", jit_build_dir=build)
    assert result["status"] == "failed"
    assert served.read_bytes() == b"baseline module"
    assert (build / "baseline.o").read_bytes() == b"baseline build"


@pytest.mark.parametrize("alias_src,alias_expected", ((True, False), (True, True), (False, True)))
def test_restore_rejects_build_leaf_alias_without_touching_sibling_modules(
    akp, tmp_path, monkeypatch, alias_src, alias_expected
):
    _checkout, package = _make_importable_checkout(akp, tmp_path, monkeypatch)
    build = package / "jit" / "build"
    (build / "baseline.o").write_bytes(b"baseline build")
    backup_root = tmp_path / "backups"
    record = akp._invalidate_aiter_jit_build(package / "dummy.cu", backup_root, jit_build_dir=build)
    build.mkdir()
    candidate = build.parent / "module_quant.so"
    candidate.write_bytes(b"candidate")
    victim = tmp_path / "victim"
    victim.mkdir()
    preserved = victim / "unrelated.so"
    preserved.write_bytes(b"unrelated")
    alias = victim / "build"
    if os.name == "nt":
        # Exercise the same path-resolution attack without Windows symlink privileges.
        real_resolve, real_is_symlink = Path.resolve, Path.is_symlink
        monkeypatch.setattr(
            Path, "resolve", lambda self, *a, **kw: real_resolve(build if self == alias else self, *a, **kw)
        )
        monkeypatch.setattr(Path, "is_symlink", lambda self: self == alias or real_is_symlink(self))
    else:
        alias.symlink_to(build, target_is_directory=True)
    if alias_src:
        record["src"] = str(alias)

    result = akp._restore_aiter_jit_build(
        record, expected_jit_build_dir=alias if alias_expected else build, backup_root=backup_root
    )

    assert preserved.read_bytes() == b"unrelated"
    assert candidate.read_bytes() == b"candidate"
    assert result["status"] == "failed", result
    assert akp._trusted_aiter_jit_build_dir(alias) is False


@pytest.mark.parametrize("value", ("", " relative cache ", "~/literal-cache"))
def test_jit_override_preserves_explicit_runtime_value(akp, tmp_path, monkeypatch, value):
    _checkout, package = _make_importable_checkout(akp, tmp_path, monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AITER_JIT_DIR", value)
    if not value:
        assert akp._aiter_jit_build_dir() is None
        record = akp._invalidate_aiter_jit_build(package.parent / "csrc" / "kernel.cu", tmp_path / "backups")
        assert record["status"] == "skipped"
        assert (package / "jit" / "build").is_dir()
    else:
        assert akp._aiter_jit_build_dir() == Path(value).absolute() / "build"


def test_checkout_cpp_itfs_invalidates_and_requires_fresh_runtime_build(akp, tmp_path, monkeypatch):
    checkout, _package = _make_importable_checkout(akp, tmp_path, monkeypatch)
    target = checkout / "csrc" / "cpp_itfs" / "mha_fwd.py"
    target.parent.mkdir()
    original = 'MD_NAME = "mha_fwd"\nTILE = 128\n\ndef launch():\n    return TILE\n'
    target.write_text(original, encoding="utf-8")
    patch = tmp_path / "optimized.py"
    patch.write_text(original.replace("128", "64"), encoding="utf-8")
    cache_root = tmp_path / "runtime-cache"
    monkeypatch.setenv("AITER_ROOT_DIR", str(cache_root))
    baseline = cache_root / "build" / "mha_fwd_baseline" / "lib.so"
    _write_lib_so(baseline, mtime=1_700_000_000)

    result = akp.apply_kernel_patch(
        patch_path=patch, target_file=target, backup_root=tmp_path / "backups", kernel_id="checkout-cpp-itfs"
    )

    assert result["status"] == "ok", result
    record = result["cpp_itfs_cache_backup"]
    assert record["is_cpp_itfs"] is True
    assert record["status"] == "ok"
    assert not baseline.exists()
    assert akp.verify_cpp_itfs_rebuilt(record)["verified"] is False
    _write_lib_so(baseline, mtime=int(record["invalidated_unix"]) + 2)
    assert akp.verify_cpp_itfs_rebuilt(record)["verified"] is True
    reverted = akp.revert_kernel_patch(result["manifest_path"])
    assert reverted["status"] == "ok", reverted
    assert target.read_text(encoding="utf-8") == original
    assert int(baseline.stat().st_mtime) == 1_700_000_000


def test_module_restore_copy_failure_keeps_build_backup_retryable(akp, tmp_path, monkeypatch):
    _checkout, package = _make_importable_checkout(akp, tmp_path, monkeypatch)
    build = package / "jit" / "build"
    (build / "baseline.o").write_bytes(b"baseline build")
    served = build.parent / "module_quant.so"
    served.write_bytes(b"baseline module")
    backup_root = tmp_path / "backups"
    record = akp._invalidate_aiter_jit_build(package / "dummy.cu", backup_root, jit_build_dir=build)
    saved_build = Path(record["backup_path"])
    real_copy = akp.shutil.copy2

    def fail_copy(*args, **kwargs):
        raise OSError("simulated restore copy failure")

    monkeypatch.setattr(akp.shutil, "copy2", fail_copy)
    failed = akp._restore_aiter_jit_build(record, expected_jit_build_dir=build, backup_root=backup_root)
    assert failed["status"] == "failed"
    assert saved_build.is_dir(), "a failed module restore must not consume the build backup"
    monkeypatch.setattr(akp.shutil, "copy2", real_copy)

    retried = akp._restore_aiter_jit_build(record, expected_jit_build_dir=build, backup_root=backup_root)

    assert retried["status"] == "ok", retried
    assert served.read_bytes() == b"baseline module"
    assert (build / "baseline.o").read_bytes() == b"baseline build"


@pytest.mark.parametrize("snapshot_mode", (False, True))
def test_uninitialized_home_jit_fallback_refuses_source_patch(akp, tmp_path, monkeypatch, snapshot_mode):
    checkout, package = _make_importable_checkout(akp, tmp_path, monkeypatch)
    target = checkout / "csrc" / "kernels" / "kernel.cu"
    original = "int tile = 128;\n"
    target.write_text(original, encoding="utf-8")
    patch = tmp_path / "optimized.cu"
    patch.write_text("int tile = 64;\n", encoding="utf-8")
    served = package / "jit" / "module_quant.so"
    served.write_bytes(b"baseline module")
    home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    real_access = akp.os.access
    monkeypatch.setattr(
        akp.os, "access", lambda path, mode: False if Path(path) == package / "jit" else real_access(path, mode)
    )

    kwargs = {}
    if snapshot_mode:
        snapshot = tmp_path / "snapshot"
        relative = target.relative_to(checkout).as_posix()
        destination = snapshot / relative
        destination.parent.mkdir(parents=True)
        destination.write_text("int tile = 64;\n", encoding="utf-8")
        patch = tmp_path / "change.patch"
        patch.write_text(
            f"diff --git a/{relative} b/{relative}\n--- a/{relative}\n+++ b/{relative}\n"
            "@@ -1 +1 @@\n-int tile = 128;\n+int tile = 64;\n",
            encoding="utf-8",
        )
        kwargs = {"snapshot_dir": snapshot, "repo_root": checkout}
    result = akp.apply_kernel_patch(
        patch_path=patch, target_file=target, backup_root=tmp_path / "backups", kernel_id="uninitialized-home", **kwargs
    )

    assert result["status"] == "failed", result
    assert "initialize" in result["error"].lower()
    assert target.read_text(encoding="utf-8") == original
    assert served.read_bytes() == b"baseline module"
    assert not (home / ".aiter" / "jit").exists()
    assert akp._aiter_jit_build_dir() is None
    (home / ".aiter" / "jit").mkdir(parents=True)
    assert akp._aiter_jit_build_dir() == home / ".aiter" / "jit" / "build"


def test_legacy_build_only_restore_preserves_serving_modules(akp, tmp_path, monkeypatch):
    _checkout, package = _make_importable_checkout(akp, tmp_path, monkeypatch)
    build = package / "jit" / "build"
    served = build.parent / "module_quant.so"
    served.write_bytes(b"untouched serving module")
    backup_root = tmp_path / "backups"
    saved = backup_root / "jit_build_legacy"
    saved.mkdir(parents=True)
    (saved / "baseline.o").write_bytes(b"baseline build")
    legacy = {"status": "ok", "src": str(build), "backup_path": str(saved)}

    result = akp._restore_aiter_jit_build(legacy, expected_jit_build_dir=build, backup_root=backup_root)

    assert result["status"] == "ok", result
    assert served.read_bytes() == b"untouched serving module"
    assert (build / "baseline.o").read_bytes() == b"baseline build"


@pytest.mark.parametrize("scope", ("legacy", None, [], ["module_quant", "module_missing"]))
def test_manifest_revert_restores_every_clean_scope(akp, tmp_path, monkeypatch, scope):
    _checkout, package = _make_importable_checkout(akp, tmp_path, monkeypatch)
    build = package / "jit/build"
    (build / "candidate.o").write_bytes(b"candidate build")
    for name in ("module_quant", "module_missing", "module_unselected"):
        (build.parent / f"{name}.so").write_bytes(b"candidate")
    record = {"status": "clean", "src": str(build)}
    if scope != "legacy":
        record.update(module_scope=scope, module_names=[], build_existed=False)
    backups = tmp_path / "backups"
    backups.mkdir()
    manifest = backups / "manifest.json"
    manifest.write_text(
        json.dumps({"strategy": {"jit_build_dir": str(build)}, "jit_build_backup": record}), encoding="utf-8"
    )

    result = akp.revert_kernel_patch(manifest)

    assert result["status"] == "ok", result
    assert result["jit_build_restore"]["status"] == "ok"
    assert not build.exists()
    for name in ("module_quant", "module_missing", "module_unselected"):
        removed = scope is None or (isinstance(scope, list) and name in scope)
        assert (build.parent / f"{name}.so").exists() is not removed


@pytest.mark.parametrize("scope", ([], ["module_quant", "module_missing"]))
def test_restore_obeys_persisted_module_scope(akp, tmp_path, monkeypatch, scope):
    _checkout, package = _make_importable_checkout(akp, tmp_path, monkeypatch)
    build = package / "jit" / "build"
    build.rmdir()
    backup_root = tmp_path / "backups"
    record = {
        "status": "clean",
        "src": str(build),
        "build_existed": False,
        "module_scope": scope,
        "module_names": [],
    }
    for name in ("module_quant", "module_missing", "module_unselected"):
        (build.parent / f"{name}.so").write_bytes(b"candidate")
    build.mkdir()
    (build / "candidate.o").write_bytes(b"candidate build")

    result = akp._restore_aiter_jit_build(record, expected_jit_build_dir=build, backup_root=backup_root)

    assert result["status"] == "ok", result
    assert (build.parent / "module_unselected.so").read_bytes() == b"candidate"
    for name in ("module_quant", "module_missing"):
        assert (build.parent / f"{name}.so").exists() is (name not in scope)
    assert not build.exists()


def test_legacy_clean_restore_clears_build_without_touching_modules(akp, tmp_path, monkeypatch):
    _checkout, package = _make_importable_checkout(akp, tmp_path, monkeypatch)
    build = package / "jit" / "build"
    (build / "candidate.o").write_bytes(b"candidate build")
    module = build.parent / "module_quant.so"
    module.write_bytes(b"untouched")

    result = akp._restore_aiter_jit_build(
        {"status": "clean", "src": str(build)},
        expected_jit_build_dir=build,
        backup_root=tmp_path / "backups",
    )

    assert result["status"] == "ok", result
    assert not build.exists()
    assert module.read_bytes() == b"untouched"


def _cpp_itfs_backup(build_dir, *, invalidated_unix=1_700_000_000.0, module_names=None, is_cpp_itfs=True):
    return {
        "is_cpp_itfs": is_cpp_itfs,
        "build_dir": "" if build_dir is None else str(build_dir),
        "invalidated_unix": invalidated_unix,
        "module_names": list(module_names or []),
    }


def _write_lib_so(path: Path, *, mtime: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"so")
    os.utime(path, (mtime, mtime))
    return path


def test_verify_cpp_itfs_rebuilt_skips_non_cpp_itfs_targets(akp, tmp_path):
    got = akp.verify_cpp_itfs_rebuilt(_cpp_itfs_backup(tmp_path, is_cpp_itfs=False))
    assert got == {"verified": True, "status": "skipped", "reason": "non-cpp_itfs target"}
    assert akp.verify_cpp_itfs_rebuilt("not-a-dict")["status"] == "skipped"


def test_verify_cpp_itfs_rebuilt_rejects_absent_and_empty_build_dir(akp, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    unrelated = _write_lib_so(tmp_path / "unrelated" / "lib.so", mtime=1_700_000_000)

    empty = akp.verify_cpp_itfs_rebuilt(_cpp_itfs_backup(None, module_names=[]))
    assert empty["verified"] is False
    assert empty["status"] == "stale"
    assert "absent" in empty["reason"]

    missing = akp.verify_cpp_itfs_rebuilt(_cpp_itfs_backup(tmp_path / "gone"))
    assert missing["verified"] is False
    assert missing["status"] == "stale"
    assert "absent" in missing["reason"]
    assert str(unrelated) not in missing.get("fresh_lib_so", [])


def test_verify_cpp_itfs_rebuilt_mtime_slack_and_module_glob(akp, tmp_path):
    build_dir = tmp_path / "cpp_itfs_build"
    mtime = 1_700_000_000
    scoped = _write_lib_so(build_dir / "attn_abc" / "lib.so", mtime=mtime)
    other = _write_lib_so(build_dir / "other_xyz" / "lib.so", mtime=mtime)

    stale = akp.verify_cpp_itfs_rebuilt(
        _cpp_itfs_backup(build_dir, invalidated_unix=mtime + 2.0, module_names=["attn"])
    )
    assert stale["verified"] is False
    assert stale["status"] == "stale"
    assert "no freshly-built" in stale["reason"]

    slack = akp.verify_cpp_itfs_rebuilt(
        _cpp_itfs_backup(build_dir, invalidated_unix=mtime + 1.0, module_names=["attn"])
    )
    assert slack["verified"] is True
    assert slack["status"] == "ok"
    assert slack["fresh_lib_so"] == [str(scoped)]
    assert str(other) not in slack["fresh_lib_so"]

    fallback = akp.verify_cpp_itfs_rebuilt(_cpp_itfs_backup(build_dir, invalidated_unix=mtime, module_names=[]))
    assert fallback["verified"] is True
    assert set(fallback["fresh_lib_so"]) == {str(scoped), str(other)}


def test_verify_cpp_itfs_rebuilt_uses_invalidation_record(akp, tmp_path):
    target = tmp_path / "aiter" / "csrc" / "cpp_itfs" / "kernel.cu"
    target.parent.mkdir(parents=True)
    target.write_text("kernel", encoding="utf-8")
    build_dir = tmp_path / "cpp_itfs_build"
    backup = tmp_path / "backup"
    backup.mkdir()

    record = akp._invalidate_aiter_cpp_itfs_cache(target, backup, build_dir_override=build_dir)
    assert record["is_cpp_itfs"] is True
    absent = akp.verify_cpp_itfs_rebuilt(record)
    assert absent["verified"] is False
    assert "absent" in absent["reason"]

    _write_lib_so(build_dir / "fresh_mod" / "lib.so", mtime=2_000_000_000)
    record["invalidated_unix"] = 1_700_000_000.0
    record["module_names"] = []
    fresh = akp.verify_cpp_itfs_rebuilt(record)
    assert fresh["verified"] is True
    assert fresh["status"] == "ok"
