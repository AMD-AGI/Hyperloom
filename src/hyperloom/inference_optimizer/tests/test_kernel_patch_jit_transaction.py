from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "multi_node"
    / "scripts"
    / "patch_path_safety.py"
)


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


def test_pod_jit_transaction_clears_candidate_when_baseline_was_clean(
    tmp_path,
    monkeypatch,
):
    safety = _load_module()
    _aiter, build = _make_aiter_jit(tmp_path)
    backup_root = tmp_path / "backups"
    monkeypatch.setenv(
        "HYPERLOOM_MN_KERNEL_BACKUP_DIR",
        str(backup_root),
    )

    record = safety.invalidate_aiter_jit_build(
        build,
        backup_root,
        "kernel_host",
    )

    assert record["status"] == "clean"
    build.mkdir(parents=True, exist_ok=True)
    (build / "candidate.so").write_text("candidate", encoding="utf-8")

    restored = safety.restore_aiter_jit_build(record)

    assert restored["status"] == "restored_clean"
    assert not build.exists()
