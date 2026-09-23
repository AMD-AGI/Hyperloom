# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Filesystem contracts for shared AITER JIT cache transactions."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from hyperloom.common import aiter_jit_cache as cache


@pytest.fixture()
def jit(tmp_path):
    build = tmp_path / "aiter" / "jit" / "build"
    build.mkdir(parents=True)
    (build / "baseline.o").write_bytes(b"baseline build")
    for name in ("module_quant", "module_gemm", "module_attention", "module_other"):
        (build.parent / f"{name}.so").write_bytes(name.encode())
    (build.parent / "__init__.py").write_text("", encoding="utf-8")
    nested = build.parent / "nested"
    nested.mkdir()
    (nested / "untouched.so").write_bytes(b"nested")
    return build


@pytest.mark.parametrize("scope", (None, ("module_gemm", "module_absent"), (), ("module_absent",)))
@pytest.mark.parametrize("had_build", (False, True))
def test_scope_round_trip_preserves_unselected_modules(jit, tmp_path, scope, had_build):
    if not had_build:
        (jit / "baseline.o").unlink()
        jit.rmdir()
    backup_root = tmp_path / "backups"
    record = cache.invalidate_jit_cache(jit, backup_root, modules=scope)
    originals = {"module_quant", "module_gemm", "module_attention", "module_other"}
    selected = originals if scope is None else originals.intersection(scope)

    assert record["status"] == ("ok" if had_build or selected else "clean"), record
    assert record["module_scope"] == (None if scope is None else list(scope))
    assert record["module_names"] == sorted(f"{name}.so" for name in selected)
    assert record["build_existed"] is had_build
    assert not jit.exists()
    for name in originals:
        assert (jit.parent / f"{name}.so").exists() is (name not in selected)
    jit.mkdir()
    (jit / "candidate.o").write_bytes(b"candidate build")
    for name in originals | {"module_absent", "module_new"}:
        (jit.parent / f"{name}.so").write_bytes(b"candidate")

    restored = cache.restore_jit_cache(json.loads(json.dumps(record)), jit, backup_root)

    assert restored == {"status": "ok", "restored_to": str(jit)}
    assert jit.exists() is had_build
    assert not (jit / "candidate.o").exists()
    if had_build:
        assert (jit / "baseline.o").read_bytes() == b"baseline build"
    for name in originals:
        assert (jit.parent / f"{name}.so").read_bytes() == (name.encode() if name in selected else b"candidate")
    for name in ("module_absent", "module_new"):
        assert (jit.parent / f"{name}.so").exists() is (scope is not None and name not in scope)
    assert (jit.parent / "__init__.py").is_file()
    assert (jit.parent / "nested" / "untouched.so").read_bytes() == b"nested"


@pytest.mark.parametrize("scope", (None, (), ("module_gemm",)))
def test_missing_cache_still_records_requested_scope(tmp_path, scope):
    build = tmp_path / "missing" / "build"
    record = cache.invalidate_jit_cache(build, tmp_path / "backups", scope)
    assert record["status"] == "clean"
    assert record["module_names"] == []
    assert record["module_scope"] == (None if scope is None else list(scope))
    assert set(record) == {"src", "status", "build_existed", "module_scope", "module_names", "moved_at"}
    assert not build.parent.exists()


@pytest.mark.parametrize("scope", ("module_gemm", ("../escape",), ("*.so",), ("",), (None,)))
def test_invalid_scope_fails_without_moving_build(jit, tmp_path, scope):
    record = cache.invalidate_jit_cache(jit, tmp_path / "backups", scope)
    assert record["status"] == "failed", record
    assert (jit / "baseline.o").read_bytes() == b"baseline build"
    assert (jit.parent / "module_gemm.so").read_bytes() == b"module_gemm"


def test_legacy_record_without_scope_restores_only_build(jit, tmp_path):
    backup_root = tmp_path / "backups"
    record = cache.invalidate_jit_cache(jit, backup_root, ())
    record.pop("module_scope")
    record.pop("module_names")
    record.pop("build_existed")
    candidate = jit.parent / "candidate.so"
    candidate.write_bytes(b"candidate")
    result = cache.restore_jit_cache(record, jit, backup_root)
    assert result["status"] == "ok", result
    assert candidate.read_bytes() == b"candidate"
    assert (jit.parent / "module_gemm.so").read_bytes() == b"module_gemm"
    assert (jit / "baseline.o").read_bytes() == b"baseline build"


def test_legacy_clean_record_removes_build_only(jit, tmp_path):
    record = {"status": "clean", "src": str(jit)}
    result = cache.restore_jit_cache(record, jit, tmp_path / "backups")
    assert result["status"] == "ok", result
    assert not jit.exists()
    assert (jit.parent / "module_gemm.so").read_bytes() == b"module_gemm"


@pytest.mark.parametrize("missing", ("backup_path", "modules_backup_path", "module_file", "module_names"))
def test_missing_or_incomplete_backup_does_not_delete_candidates(jit, tmp_path, missing):
    backup_root = tmp_path / "backups"
    record = cache.invalidate_jit_cache(jit, backup_root)
    if missing in {"backup_path", "modules_backup_path"}:
        record[missing] = str(backup_root / "absent")
    elif missing == "module_file":
        (Path(record["modules_backup_path"]) / "module_gemm.so").unlink()
    else:
        record["module_names"].append("unrecorded.so")
    jit.mkdir()
    (jit / "candidate.o").write_bytes(b"candidate build")
    candidate = jit.parent / "module_gemm.so"
    candidate.write_bytes(b"candidate")

    result = cache.restore_jit_cache(record, jit, backup_root)

    assert result["status"] == "failed", result
    assert candidate.read_bytes() == b"candidate"
    assert (jit / "candidate.o").read_bytes() == b"candidate build"


@pytest.mark.parametrize(
    "change",
    (
        {"module_scope": "all"},
        {"module_scope": []},
        {"module_scope": ["module_other"]},
        {"module_scope": "absent"},
    ),
)
def test_unknown_or_conflicting_scope_cannot_expand_restore(jit, tmp_path, change):
    backup_root = tmp_path / "backups"
    record = cache.invalidate_jit_cache(jit, backup_root, ("module_gemm",))
    record.update(change)
    if change.get("module_scope") == "absent":
        record.pop("module_scope")
    candidate = jit.parent / "module_gemm.so"
    candidate.write_bytes(b"candidate")

    result = cache.restore_jit_cache(record, jit, backup_root)

    assert result["status"] == "failed", result
    assert candidate.read_bytes() == b"candidate"
    assert (jit.parent / "module_other.so").read_bytes() == b"module_other"


def test_move_failure_rolls_back_and_allows_new_attempt(jit, tmp_path, monkeypatch):
    real_move = cache.shutil.move

    def fail_second_module(src, dst):
        if Path(src) == jit.parent / "module_gemm.so":
            raise OSError("simulated move failure")
        return real_move(src, dst)

    monkeypatch.setattr(cache.shutil, "move", fail_second_module)
    failed = cache.invalidate_jit_cache(jit, tmp_path / "backups")
    assert failed["status"] == "failed"
    assert failed["rollback_errors"] == []
    assert (jit / "baseline.o").read_bytes() == b"baseline build"
    assert (jit.parent / "module_attention.so").read_bytes() == b"module_attention"
    monkeypatch.setattr(cache.shutil, "move", real_move)
    assert cache.invalidate_jit_cache(jit, tmp_path / "backups")["status"] == "ok"


@pytest.mark.parametrize("failure", ("copy", "move"))
def test_restore_failure_keeps_backup_retryable(jit, tmp_path, monkeypatch, failure):
    backup_root = tmp_path / "backups"
    record = cache.invalidate_jit_cache(jit, backup_root)
    saved_build = Path(record["backup_path"])
    target = "copy2" if failure == "copy" else "move"
    real = getattr(cache.shutil, target)

    def fail(*args, **kwargs):
        raise OSError("simulated restore failure")

    monkeypatch.setattr(cache.shutil, target, fail)
    assert cache.restore_jit_cache(record, jit, backup_root)["status"] == "failed"
    assert saved_build.is_dir()
    monkeypatch.setattr(cache.shutil, target, real)
    assert cache.restore_jit_cache(record, jit, backup_root)["status"] == "ok"
    assert (jit / "baseline.o").read_bytes() == b"baseline build"
    assert (jit.parent / "module_gemm.so").read_bytes() == b"module_gemm"


@pytest.mark.parametrize("key", ("src", "backup_path", "modules_backup_path"))
def test_untrusted_record_paths_preserve_candidate(jit, tmp_path, key):
    backup_root = tmp_path / "backups"
    record = cache.invalidate_jit_cache(jit, backup_root)
    outside = tmp_path / "outside" / "build"
    outside.mkdir(parents=True)
    record[key] = str(outside)
    candidate = jit.parent / "module_gemm.so"
    candidate.write_bytes(b"candidate")
    assert cache.restore_jit_cache(record, jit, backup_root)["status"] == "failed"
    assert candidate.read_bytes() == b"candidate"


def test_restore_rejects_backup_inside_candidate_build(jit, tmp_path):
    nested = jit / "backup"
    nested.mkdir()
    record = {"status": "ok", "src": str(jit), "backup_path": str(nested)}
    result = cache.restore_jit_cache(record, jit, tmp_path)
    assert result["status"] == "failed"
    assert (jit / "baseline.o").read_bytes() == b"baseline build"


@pytest.mark.parametrize("value", ("", " missing cache ", "~/literal"))
def test_resolver_preserves_explicit_override(jit, tmp_path, monkeypatch, value):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AITER_JIT_DIR", value)
    assert cache.resolve_jit_build_dir(jit.parent.parent) == (Path(value).absolute() / "build" if value else None)
    assert cache.resolve_jit_build_dir(None) is None


def test_resolver_readonly_home_must_be_initialized(jit, tmp_path, monkeypatch):
    monkeypatch.delenv("AITER_JIT_DIR", raising=False)
    monkeypatch.setattr(cache.os, "access", lambda *args: False)
    home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    assert cache.resolve_jit_build_dir(jit.parent.parent) is None
    fallback = home / ".aiter" / "jit"
    fallback.mkdir(parents=True)
    assert cache.resolve_jit_build_dir(jit.parent.parent) == fallback / "build"


def test_resolver_writable_package_keeps_lexical_path(jit, monkeypatch):
    monkeypatch.delenv("AITER_JIT_DIR", raising=False)
    monkeypatch.setattr(cache.os, "access", lambda *args: True)
    assert cache.resolve_jit_build_dir(jit.parent.parent) == jit


@pytest.fixture
def discoverable_package(tmp_path, monkeypatch):
    package = tmp_path / "site" / "aiter"
    (package / "jit").mkdir(parents=True)
    (package / "__init__.py").write_text("raise AssertionError('AITER must not be imported')\n", encoding="utf-8")
    monkeypatch.delitem(sys.modules, "aiter", raising=False)
    monkeypatch.syspath_prepend(str(package.parent))
    monkeypatch.delenv("AITER_JIT_DIR", raising=False)
    monkeypatch.delenv("VLLM_VENV_ROOT", raising=False)
    return package


def test_package_discovery_prefers_importable_package_without_importing(discoverable_package, tmp_path, monkeypatch):
    venv = tmp_path / "venv"
    (venv / "lib/python3.12/site-packages/aiter").mkdir(parents=True)
    monkeypatch.setenv("VLLM_VENV_ROOT", str(venv))
    assert cache.resolve_package_root() == discoverable_package
    assert "aiter" not in sys.modules


def test_package_discovery_uses_search_locations_not_origin(discoverable_package, monkeypatch):
    spec = importlib.util.spec_from_file_location("aiter", discoverable_package / "__init__.py")
    spec.origin = None
    monkeypatch.setattr(importlib.util, "find_spec", lambda _: spec)
    assert cache.resolve_package_root() == discoverable_package


@pytest.mark.parametrize("package_dir", ["site-packages", "dist-packages"])
@pytest.mark.parametrize("missing", ["none", "import-error", "invalid-spec", "module-only"])
def test_package_discovery_uses_existing_isolated_venv(tmp_path, monkeypatch, package_dir, missing):
    venv = tmp_path / "venv"
    package = venv / "lib" / "python3.12" / package_dir / "aiter"
    package.mkdir(parents=True)
    monkeypatch.setenv("VLLM_VENV_ROOT", f"  {venv}  ")

    def missing_spec(_):
        if missing == "import-error":
            raise ImportError("no aiter")
        if missing == "invalid-spec":
            raise ValueError("no spec")
        if missing == "module-only":
            return importlib.util.spec_from_file_location("aiter", tmp_path / "aiter.py")
        return None

    monkeypatch.setattr(importlib.util, "find_spec", missing_spec)
    assert cache.resolve_package_root() == package
    package.rmdir()
    assert cache.resolve_package_root() is None


@pytest.mark.parametrize("leaf", ["jit", "jit/build"])
def test_serving_context_wrapper_owns_both_paths(discoverable_package, tmp_path, monkeypatch, leaf):
    selected = tmp_path / "manual" / leaf
    selected.mkdir(parents=True)
    monkeypatch.setenv("AITER_JIT_DIR", str(tmp_path / "runtime"))
    assert cache.resolve_serving_context(f"  {selected}  ") == (tmp_path / "manual", tmp_path / "manual/jit/build")
    assert "aiter" not in sys.modules


@pytest.mark.parametrize("runtime", ["runtime-cache", "runtime/build", "~/literal", " runtime-cache ", ""])
def test_serving_context_runtime_override_does_not_move_configs(discoverable_package, tmp_path, monkeypatch, runtime):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AITER_JIT_DIR", runtime)
    expected = (discoverable_package, Path(runtime).absolute() / "build") if runtime else None
    assert cache.resolve_serving_context() == expected


@pytest.mark.parametrize("initialized", [False, True])
def test_serving_context_readonly_home_does_not_move_configs(discoverable_package, tmp_path, monkeypatch, initialized):
    home = tmp_path / "home"
    jit = home / ".aiter/jit"
    if initialized:
        jit.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.setattr(os, "access", lambda *_: False)
    expected = (discoverable_package, jit / "build") if initialized else None
    assert cache.resolve_serving_context() == expected


def test_serving_context_ignores_missing_wrapper(discoverable_package, tmp_path):
    assert cache.resolve_serving_context(tmp_path / "missing") == (
        discoverable_package,
        discoverable_package / "jit/build",
    )


@pytest.mark.parametrize("leaf", ["jit", "jit/build"])
def test_serving_context_retains_caller_probes_only_when_package_missing(tmp_path, monkeypatch, leaf):
    monkeypatch.setattr(importlib.util, "find_spec", lambda _: None)
    monkeypatch.delenv("VLLM_VENV_ROOT", raising=False)
    monkeypatch.delenv("AITER_JIT_DIR", raising=False)
    assert cache.resolve_serving_context() is None
    probe = tmp_path / "manual" / leaf
    probe.mkdir(parents=True)
    assert cache.resolve_serving_context(jit_probe_paths=(str(tmp_path / "missing"), str(probe))) == (
        tmp_path / "manual",
        tmp_path / "manual/jit/build",
    )


@pytest.mark.parametrize("alias_side", ("src", "expected"))
def test_build_leaf_alias_is_never_trusted(jit, tmp_path, monkeypatch, alias_side):
    alias = tmp_path / "victim" / "build"
    real_resolve, real_symlink = Path.resolve, Path.is_symlink
    monkeypatch.setattr(
        Path, "resolve", lambda self, *args, **kwargs: real_resolve(jit if self == alias else self, *args, **kwargs)
    )
    monkeypatch.setattr(Path, "is_symlink", lambda self: self == alias or real_symlink(self))
    src, expected = (alias, jit) if alias_side == "src" else (jit, alias)
    assert not cache.trusted_jit_build_dir(src, expected)
    assert cache.invalidate_jit_cache(alias, tmp_path / "backup")["status"] == "failed"


@pytest.mark.parametrize("path_kind", ("build_file", "module_directory", "nested_backup"))
def test_invalid_cache_layout_fails_before_mutation(jit, tmp_path, path_kind):
    backup = tmp_path / "backups"
    if path_kind == "build_file":
        (jit / "baseline.o").unlink()
        jit.rmdir()
        jit.write_bytes(b"not a directory")
    elif path_kind == "module_directory":
        (jit.parent / "invalid.so").mkdir()
    else:
        backup = jit / "backups"
    result = cache.invalidate_jit_cache(jit, backup)
    assert result["status"] == "failed", result
    assert (jit.parent / "module_gemm.so").read_bytes() == b"module_gemm"
    assert jit.exists()


def test_backup_collision_does_not_replace_existing_data(jit, tmp_path, monkeypatch):
    backup_root = tmp_path / "backups"
    saved = backup_root / "jit_build_123"
    saved.mkdir(parents=True)
    (saved / "original.o").write_bytes(b"existing backup")
    monkeypatch.setattr(cache.time, "time_ns", lambda: 123)

    result = cache.invalidate_jit_cache(jit, backup_root)

    assert result["status"] == "failed", result
    assert (saved / "original.o").read_bytes() == b"existing backup"
    assert (jit / "baseline.o").read_bytes() == b"baseline build"


def test_failed_rollback_reports_recoverable_backup_locations(jit, tmp_path, monkeypatch):
    real_move = cache.shutil.move

    def fail_module_and_rollback(src, dst):
        if Path(src) == jit.parent / "module_attention.so" or Path(dst) == jit:
            raise OSError("simulated rollback failure")
        return real_move(src, dst)

    monkeypatch.setattr(cache.shutil, "move", fail_module_and_rollback)
    result = cache.invalidate_jit_cache(jit, tmp_path / "backups")

    assert result["status"] == "failed", result
    assert result["rollback_errors"] == ["simulated rollback failure"]
    assert (Path(result["backup_path"]) / "baseline.o").read_bytes() == b"baseline build"
    assert (jit.parent / "module_attention.so").read_bytes() == b"module_attention"


@pytest.mark.parametrize("record", (None, {}, {"status": "failed"}, {"status": "ok"}))
def test_restore_skips_without_a_recorded_destination(tmp_path, record):
    assert cache.restore_jit_cache(record, tmp_path / "build", tmp_path)["status"] == "skipped"


@pytest.mark.parametrize(
    "change",
    ({"build_existed": False}, {"backup_path": ""}, {"build_existed": "yes"}, {"module_names": "module.so"}),
)
def test_inconsistent_inventory_preserves_candidates(jit, tmp_path, change):
    backup_root = tmp_path / "backups"
    record = cache.invalidate_jit_cache(jit, backup_root)
    record.update(change)
    candidate = jit.parent / "new.so"
    candidate.write_bytes(b"candidate")
    assert cache.restore_jit_cache(record, jit, backup_root)["status"] == "failed"
    assert candidate.read_bytes() == b"candidate"


def test_candidate_module_directory_fails_before_cleanup(jit, tmp_path):
    backup_root = tmp_path / "backups"
    record = cache.invalidate_jit_cache(jit, backup_root)
    (jit.parent / "invalid.so").mkdir()
    candidate = jit.parent / "candidate.so"
    candidate.write_bytes(b"candidate")
    assert cache.restore_jit_cache(record, jit, backup_root)["status"] == "failed"
    assert candidate.read_bytes() == b"candidate"


def test_trusted_path_resolution_failure_is_false(jit, monkeypatch):
    def fail_resolve(*args, **kwargs):
        raise RuntimeError("symlink loop")

    monkeypatch.setattr(Path, "resolve", fail_resolve)
    assert not cache.trusted_jit_build_dir(jit, jit)


def test_restore_without_backup_root_retains_legacy_api(jit, tmp_path):
    record = cache.invalidate_jit_cache(jit, tmp_path / "backups", ())
    assert cache.restore_jit_cache(record, jit, None)["status"] == "ok"
    assert (jit / "baseline.o").read_bytes() == b"baseline build"


def test_core_loads_as_standalone_stdlib_file(tmp_path):
    source = Path(cache.__file__)
    copied = tmp_path / "aiter_jit_cache.py"
    copied.write_bytes(source.read_bytes())
    script = """
import importlib.util
import sys
from pathlib import Path
spec = importlib.util.spec_from_file_location('standalone_cache', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert not any(name.startswith(('hyperloom', 'aiter', 'torch')) for name in sys.modules)
root = Path(sys.argv[2])
package = root / 'site' / 'aiter'
(package / 'jit').mkdir(parents=True)
(package / '__init__.py').write_text("raise AssertionError('AITER must not be imported')", encoding='utf-8')
sys.path.insert(0, str(package.parent))
assert module.resolve_package_root() == package
assert module.resolve_serving_context() == (package, package / 'jit' / 'build')
assert not any(name.startswith(('hyperloom', 'aiter', 'torch')) for name in sys.modules)
record = module.invalidate_jit_cache(root / 'jit' / 'build', root / 'backups', ())
assert record['status'] == 'clean', record
assert module.restore_jit_cache(record, root / 'jit' / 'build', root / 'backups')['status'] == 'ok'
"""
    proc = subprocess.run(
        [sys.executable, "-I", "-S", "-B", "-c", script, str(copied), str(tmp_path)],
        cwd=tmp_path,
        env={key: value for key, value in os.environ.items() if key != "PYTHONPATH"},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
