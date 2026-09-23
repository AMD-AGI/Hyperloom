from __future__ import annotations

import argparse
import base64
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest


_SCRIPT = Path(__file__).resolve().parents[1] / "multi_node" / "scripts" / "patch_path_safety.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "_patch_path_safety_transaction_test",
        _SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _make_aiter_jit(tmp_path: Path) -> tuple[Path, Path]:
    aiter = tmp_path / "site-packages" / "aiter"
    jit = aiter / "jit"
    build = jit / "build"
    build.mkdir(parents=True)
    (aiter / "__init__.py").write_text("", encoding="utf-8")
    (jit / "__init__.py").write_text("", encoding="utf-8")
    return aiter, build


def test_pod_jit_transaction_restores_baseline_cache(
    tmp_path,
    monkeypatch,
):
    safety = _load_module()
    _aiter, build = _make_aiter_jit(tmp_path)
    (build / "baseline.so").write_text("baseline", encoding="utf-8")
    backup_root = tmp_path / "backups"
    monkeypatch.setenv(
        "HYPERLOOM_MN_KERNEL_BACKUP_DIR",
        str(backup_root),
    )
    monkeypatch.setenv(
        "INFERENCE_OPTIMIZER_FRAMEWORK_SOURCE_ROOTS",
        str(_aiter),
    )

    record = safety.invalidate_aiter_jit_build(
        build,
        backup_root,
        "kernel_host",
    )

    assert record["status"] == "ok"
    assert not build.exists()
    build.mkdir(parents=True)
    (build / "candidate.so").write_text("candidate", encoding="utf-8")

    restored = safety.restore_aiter_jit_build(record)

    assert restored["status"] == "restored"
    assert (build / "baseline.so").is_file()
    assert not (build / "candidate.so").exists()


@pytest.mark.parametrize("build_exists", [False, True])
def test_pod_jit_transaction_preserves_empty_build_existence(
    tmp_path,
    monkeypatch,
    build_exists,
):
    safety = _load_module()
    _aiter, build = _make_aiter_jit(tmp_path)
    if not build_exists:
        build.rmdir()
    backup_root = tmp_path / "backups"
    monkeypatch.setenv(
        "HYPERLOOM_MN_KERNEL_BACKUP_DIR",
        str(backup_root),
    )
    monkeypatch.setenv(
        "INFERENCE_OPTIMIZER_FRAMEWORK_SOURCE_ROOTS",
        str(_aiter),
    )

    record = safety.invalidate_aiter_jit_build(
        build,
        backup_root,
        "kernel_host",
    )

    assert record["status"] == ("ok" if build_exists else "clean")
    assert record["build_existed"] is build_exists
    build.mkdir(parents=True, exist_ok=True)
    (build / "candidate.so").write_text("candidate", encoding="utf-8")

    restored = safety.restore_aiter_jit_build(record)

    assert restored["status"] == ("restored" if build_exists else "restored_clean")
    assert build.exists() is build_exists
    assert not (build / "candidate.so").exists()


def test_missing_baseline_backup_preserves_candidate_cache(
    tmp_path,
    monkeypatch,
):
    safety = _load_module()
    aiter, build = _make_aiter_jit(tmp_path)
    (build / "baseline.so").write_text("baseline", encoding="utf-8")
    backup_root = tmp_path / "backups"
    monkeypatch.setenv("HYPERLOOM_MN_KERNEL_BACKUP_DIR", str(backup_root))
    monkeypatch.setenv(
        "INFERENCE_OPTIMIZER_FRAMEWORK_SOURCE_ROOTS",
        str(aiter),
    )
    record = safety.invalidate_aiter_jit_build(
        build,
        backup_root,
        "kernel_host",
    )
    shutil.rmtree(record["backup_path"])
    build.mkdir(parents=True)
    candidate = build / "candidate.so"
    candidate.write_text("candidate", encoding="utf-8")

    try:
        safety.restore_aiter_jit_build(record)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("missing baseline backup must fail restore")

    assert candidate.is_file()


def test_finalize_deletes_source_and_jit_backups(
    tmp_path,
    monkeypatch,
):
    safety = _load_module()
    aiter, build = _make_aiter_jit(tmp_path)
    (build / "baseline.so").write_text("baseline", encoding="utf-8")
    backup_root = tmp_path / "backups"
    backup_root.mkdir()
    monkeypatch.setenv("HYPERLOOM_MN_KERNEL_BACKUP_DIR", str(backup_root))
    monkeypatch.setenv(
        "INFERENCE_OPTIMIZER_FRAMEWORK_SOURCE_ROOTS",
        str(aiter),
    )
    jit_record = safety.invalidate_aiter_jit_build(
        build,
        backup_root,
        "kernel_host",
    )
    source_backup = backup_root / "source.bak"
    source_backup.write_text("source", encoding="utf-8")

    result = safety.finalize_patch_records(
        [
            {
                "backup_path": str(source_backup),
                "jit_backup": jit_record,
            }
        ]
    )

    assert result["status"] == "finalized"
    assert not source_backup.exists()
    assert not Path(jit_record["backup_path"]).exists()


def _load_pod_ops(monkeypatch):
    monkeypatch.syspath_prepend(str(_SCRIPT.parent))
    spec = importlib.util.spec_from_file_location("_pod_jit_ops_test", _SCRIPT.with_name("kernel_node_ops.py"))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("action", ["revert", "finalize"])
def test_pod_source_patch_transacts_top_level_modules(tmp_path, monkeypatch, capsys, action):
    pod = _load_pod_ops(monkeypatch)
    aiter, build = _make_aiter_jit(tmp_path)
    baseline = build.parent / "module_gemm.cpython-312-x86_64-linux-gnu.so"
    baseline.write_bytes(b"baseline module")
    (build / "baseline.o").write_bytes(b"baseline build")
    target = aiter / "kernel.py"
    target.write_text("value = 1\n", encoding="utf-8")
    backups = tmp_path / "backups"
    monkeypatch.setenv("HYPERLOOM_MN_KERNEL_BACKUP_DIR", str(backups))
    monkeypatch.setenv("INFERENCE_OPTIMIZER_FRAMEWORK_SOURCE_ROOTS", str(aiter))
    monkeypatch.delenv("AITER_JIT_DIR", raising=False)

    assert (
        pod._do_apply(
            argparse.Namespace(
                target_path=str(target),
                patch_b64=base64.b64encode(b"value = 2\n").decode("ascii"),
                backup_dir=str(backups),
                kernel_id="gemm",
                jit_build_dir=str(build),
            )
        )
        == 0
    )
    record = json.loads(capsys.readouterr().out)
    assert not baseline.exists()
    assert not build.exists()
    assert record["jit_backup"]["module_scope"] is None
    assert record["jit_backup"]["module_names"] == [baseline.name]
    modules_backup = Path(record["jit_backup"]["modules_backup_path"])
    assert (modules_backup / baseline.name).read_bytes() == b"baseline module"
    baseline.write_bytes(b"candidate replacement")
    candidate = build.parent / "module_new.so"
    candidate.write_bytes(b"candidate new")
    build.mkdir()
    (build / "candidate.o").write_bytes(b"candidate build")
    args = argparse.Namespace(records_json=json.dumps([record]), target_path="", backup_path="")

    if action == "revert":
        assert pod._do_revert(args) == 0
        result = json.loads(capsys.readouterr().out)
        assert result["status"] == "restored"
        assert result["jit_restore"]["status"] == "restored"
        assert baseline.read_bytes() == b"baseline module"
        assert not candidate.exists()
        assert (build / "baseline.o").read_bytes() == b"baseline build"
        assert target.read_text(encoding="utf-8") == "value = 1\n"
    else:
        assert pod._do_finalize(args) == 0
        result = json.loads(capsys.readouterr().out)
        assert result["status"] == "finalized"
        assert not modules_backup.exists()
        assert not Path(record["jit_backup"]["backup_path"]).exists()
        assert not Path(record["backup_path"]).exists()
        assert baseline.read_bytes() == b"candidate replacement"
        assert candidate.read_bytes() == b"candidate new"
        assert target.read_text(encoding="utf-8") == "value = 2\n"


def test_pod_finalize_removes_modules_backup_once(tmp_path, monkeypatch):
    safety = _load_module()
    _aiter, build = _make_aiter_jit(tmp_path)
    module = build.parent / "module_gemm.so"
    module.write_bytes(b"baseline")
    backups = tmp_path / "backups"
    monkeypatch.setenv("HYPERLOOM_MN_KERNEL_BACKUP_DIR", str(backups))
    record = safety.invalidate_aiter_jit_build(build, backups, "kernel")
    module_backup = Path(record["modules_backup_path"])

    result = safety.finalize_patch_records([{"jit_backup": record}, {"jit_backup": record}])

    assert result["status"] == "finalized"
    assert result["deleted"].count(str(module_backup)) == 1
    assert not module_backup.exists()


def test_pod_legacy_clean_record_preserves_top_level_modules(tmp_path, monkeypatch):
    safety = _load_module()
    _aiter, build = _make_aiter_jit(tmp_path)
    module = build.parent / "module_gemm.so"
    module.write_bytes(b"unrelated module")
    (build / "candidate.o").write_bytes(b"candidate")
    monkeypatch.setenv("HYPERLOOM_MN_KERNEL_BACKUP_DIR", str(tmp_path / "backups"))

    result = safety.restore_aiter_jit_build({"status": "clean", "src": str(build)})

    assert result["status"] == "restored_clean"
    assert not build.exists()
    assert module.read_bytes() == b"unrelated module"


def test_pod_custom_jit_directory_uses_runtime_owner(tmp_path, monkeypatch):
    safety = _load_module()
    _aiter, _build = _make_aiter_jit(tmp_path)
    jit = tmp_path / "custom-jit"
    jit.mkdir()
    baseline = jit / "module_gemm.so"
    baseline.write_bytes(b"baseline")
    monkeypatch.setenv("AITER_JIT_DIR", str(jit))
    monkeypatch.setenv("HYPERLOOM_MN_KERNEL_BACKUP_DIR", str(tmp_path / "backups"))

    record = safety.invalidate_aiter_jit_build(jit / "build", tmp_path / "backups", "kernel")

    assert not baseline.exists()
    assert safety.restore_aiter_jit_build(record)["status"] == "restored"
    assert baseline.read_bytes() == b"baseline"
    with pytest.raises(ValueError):
        safety.invalidate_aiter_jit_build(tmp_path / "untrusted" / "build", tmp_path / "backups", "other")


def test_pod_rejects_disabled_runtime_jit_override(tmp_path, monkeypatch):
    safety = _load_module()
    _aiter, build = _make_aiter_jit(tmp_path)
    monkeypatch.setenv("AITER_JIT_DIR", "")
    monkeypatch.setenv("HYPERLOOM_MN_KERNEL_BACKUP_DIR", str(tmp_path / "backups"))

    with pytest.raises(ValueError, match="invalid AITER"):
        safety.invalidate_aiter_jit_build(build, tmp_path / "backups", "kernel")


def test_pod_core_invalidation_failure_prevents_source_write(tmp_path, monkeypatch):
    pod = _load_pod_ops(monkeypatch)
    aiter, build = _make_aiter_jit(tmp_path)
    invalid_module = build.parent / "module_gemm.so"
    invalid_module.mkdir()
    target = aiter / "kernel.py"
    target.write_text("value = 1\n", encoding="utf-8")
    monkeypatch.setenv("HYPERLOOM_MN_KERNEL_BACKUP_DIR", str(tmp_path / "backups"))

    with pytest.raises(OSError, match="serving modules"):
        pod._do_apply(
            argparse.Namespace(
                target_path=str(target),
                patch_b64=base64.b64encode(b"value = 2\n").decode(),
                backup_dir=str(tmp_path / "backups"),
                kernel_id="gemm",
                jit_build_dir=str(build),
            )
        )

    assert target.read_text(encoding="utf-8") == "value = 1\n"
    assert invalid_module.is_dir()
    assert build.is_dir()


def test_pod_incomplete_module_inventory_does_not_report_restored(tmp_path, monkeypatch, capsys):
    pod = _load_pod_ops(monkeypatch)
    aiter, build = _make_aiter_jit(tmp_path)
    module = build.parent / "module_gemm.so"
    module.write_bytes(b"baseline module")
    target = aiter / "kernel.py"
    target.write_text("value = 1\n", encoding="utf-8")
    monkeypatch.setenv("HYPERLOOM_MN_KERNEL_BACKUP_DIR", str(tmp_path / "backups"))
    assert (
        pod._do_apply(
            argparse.Namespace(
                target_path=str(target),
                patch_b64=base64.b64encode(b"value = 2\n").decode(),
                backup_dir=str(tmp_path / "backups"),
                kernel_id="gemm",
                jit_build_dir=str(build),
            )
        )
        == 0
    )
    record = json.loads(capsys.readouterr().out)
    saved = Path(record["jit_backup"]["modules_backup_path"]) / module.name
    saved.unlink()
    module.write_bytes(b"candidate")
    build.mkdir()
    (build / "candidate.o").write_bytes(b"candidate build")

    assert pod._do_revert(argparse.Namespace(records_json=json.dumps([record]))) == 1

    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "failed"
    assert "incomplete serving-module backup" in result["error"]
    assert module.read_bytes() == b"candidate"
    assert (build / "candidate.o").read_bytes() == b"candidate build"


@pytest.mark.parametrize("key", ["backup_path", "modules_backup_path"])
def test_pod_jit_backup_paths_stay_within_backup_policy(tmp_path, monkeypatch, key):
    safety = _load_module()
    _aiter, build = _make_aiter_jit(tmp_path)
    backups = tmp_path / "backups"
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "important.txt").write_bytes(b"untouched")
    monkeypatch.setenv("HYPERLOOM_MN_KERNEL_BACKUP_DIR", str(backups))
    record = {"status": "ok", "src": str(build), key: str(outside)}

    with pytest.raises(ValueError, match="not under"):
        safety.restore_aiter_jit_build(record)
    with pytest.raises(ValueError, match="not under"):
        safety.finalize_patch_records([{"jit_backup": record}])

    assert (outside / "important.txt").read_bytes() == b"untouched"


def test_pod_backup_name_cannot_escape_allowed_root(tmp_path, monkeypatch):
    safety = _load_module()
    _aiter, build = _make_aiter_jit(tmp_path)
    monkeypatch.setenv("HYPERLOOM_MN_KERNEL_BACKUP_DIR", str(tmp_path / "backups"))

    with pytest.raises(ValueError, match="not under"):
        safety.invalidate_aiter_jit_build(build, tmp_path / "backups", "../escape")
