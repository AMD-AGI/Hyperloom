# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Revert-time backup-path containment tests for revert_kernel_patch.

The manifest is untrusted at revert time, so every ``copy2`` source
(``backup_path``) must stay under the apply-time backup tree (the manifest's
own directory). A tampered ``backup_path`` pointing at an arbitrary host file
must not be copied onto the restore target, while a legitimate manifest reverts
unchanged.

Note: the restore *target* itself is intentionally not framework-root-gated
here, because apply accepts targets outside the known framework roots; gating
the target would break those legitimate reverts.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
from pathlib import Path

from hyperloom.orchestrator.kernel import apply_kernel_patch as apk


def _write_manifest(backup_dir: Path, manifest: dict) -> Path:
    backup_dir.mkdir(parents=True, exist_ok=True)
    mf = backup_dir / "manifest.json"
    mf.write_text(json.dumps(manifest), encoding="utf-8")
    return mf


def test_legit_manifest_reverts(tmp_path):
    backup_dir = tmp_path / "backup"
    (backup_dir / "source").mkdir(parents=True)
    backup_file = backup_dir / "source" / "orig.py"
    backup_file.write_text("original\n", encoding="utf-8")

    target = tmp_path / "framework" / "mod.py"
    target.parent.mkdir(parents=True)
    target.write_text("patched\n", encoding="utf-8")

    mf = _write_manifest(
        backup_dir,
        {
            "source_backup": {
                "path": str(target),
                "backup_path": str(backup_file),
            }
        },
    )
    res = apk.revert_kernel_patch(mf)
    assert res["status"] == "ok"
    assert "skipped_untrusted_backups" not in res
    assert target.read_text(encoding="utf-8") == "original\n"
    assert str(target) in res["restored_paths"]
    manifest = json.loads(mf.read_text(encoding="utf-8"))
    assert manifest["status"] == "reverted"


def test_tampered_backup_path_is_skipped(tmp_path, caplog):
    """A backup_path pointing outside the backup tree must not be copied."""
    backup_dir = tmp_path / "backup"
    backup_dir.mkdir(parents=True)
    secret = tmp_path / "secret.txt"
    secret.write_text("TOP-SECRET\n", encoding="utf-8")

    target = tmp_path / "framework" / "mod.py"
    target.parent.mkdir(parents=True)
    target.write_text("patched\n", encoding="utf-8")

    mf = _write_manifest(
        backup_dir,
        {
            "source_backup": {
                "path": str(target),
                "backup_path": str(secret),  # outside backup_dir
            }
        },
    )
    with caplog.at_level(logging.WARNING, logger=apk.log.name):
        res = apk.revert_kernel_patch(mf)
    assert res["status"] == "partial"
    assert res["skipped_untrusted_backups"] == [
        {
            "kind": "source_backup",
            "path": str(target),
            "backup_path": str(secret),
        }
    ]
    # Skipped: target not overwritten with the secret file's bytes.
    assert target.read_text(encoding="utf-8") == "patched\n"
    assert str(target) not in res["restored_paths"]
    manifest = json.loads(mf.read_text(encoding="utf-8"))
    assert manifest["status"] == "reverted_partial"
    assert "untrusted backup_path" in caplog.text


def test_tampered_artifact_backup_path_is_skipped(tmp_path):
    """An artifact entry with an out-of-tree backup_path must not be copied."""
    backup_dir = tmp_path / "backup"
    backup_dir.mkdir(parents=True)
    secret = tmp_path / "secret.bin"
    secret.write_text("SECRET-BYTES\n", encoding="utf-8")

    target = tmp_path / "artifacts" / "kernel.so"
    target.parent.mkdir(parents=True)
    target.write_text("real-artifact\n", encoding="utf-8")

    mf = _write_manifest(
        backup_dir,
        {"artifacts": [{"path": str(target), "backup_path": str(secret)}]},
    )
    res = apk.revert_kernel_patch(mf)
    assert res["status"] == "partial"
    assert res["skipped_untrusted_backups"] == [
        {
            "kind": "artifact",
            "path": str(target),
            "backup_path": str(secret),
        }
    ]
    assert target.read_text(encoding="utf-8") == "real-artifact\n"
    assert str(target) not in res["restored_paths"]


def test_source_backups_tampered_backup_path_is_skipped(tmp_path, caplog):
    """source_backups copy source is confined to the backup tree."""
    backup_dir = tmp_path / "backup"
    backup_dir.mkdir(parents=True)
    secret = tmp_path / "secret2.txt"
    secret.write_text("SECRET2\n", encoding="utf-8")

    target = tmp_path / "framework" / "mod2.py"
    target.parent.mkdir(parents=True)
    target.write_text("patched2\n", encoding="utf-8")

    mf = _write_manifest(
        backup_dir,
        {
            "source_backups": [
                {
                    "path": str(target),
                    "backup_path": str(secret),
                    "disposition": "modified",
                }
            ]
        },
    )
    with caplog.at_level(logging.WARNING, logger=apk.log.name):
        res = apk.revert_kernel_patch(mf)
    assert res["status"] == "partial"
    assert res["skipped_untrusted_backups"] == [
        {
            "kind": "source_backups",
            "path": str(target),
            "backup_path": str(secret),
        }
    ]
    assert target.read_text(encoding="utf-8") == "patched2\n"
    assert str(target) not in res["restored_paths"]
    assert "untrusted backup_path" in caplog.text


def test_retrying_partial_revert_invalidates_bytecode_from_unrestored_source(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr(apk.tempfile, "gettempdir", lambda: str(tmp_path / "cache_tmp"))
    backup_dir = tmp_path / "backup"
    backup_dir.mkdir()
    framework = tmp_path / "framework"
    framework.mkdir()
    entries = []
    for name in ("pending.py", "restored.py"):
        target = framework / name
        target.write_text("VALUE = 2\n", encoding="utf-8")
        backup = backup_dir / name
        backup.write_text("VALUE = 1\n", encoding="utf-8")
        # Equal timestamp and size make stale bytecode indistinguishable by Python's normal cache check.
        for path in (target, backup):
            os.utime(path, (1_700_000_000, 1_700_000_000))
        entries.append({"path": str(target), "backup_path": str(backup), "disposition": "modified"})
    valid_backup = entries[0]["backup_path"]
    entries[0]["backup_path"] = str(tmp_path / "untrusted.py")
    manifest_file = _write_manifest(backup_dir, {"source_backups": entries})

    first = apk.revert_kernel_patch(manifest_file)
    assert first["status"] == "partial"
    pending = framework / "pending.py"

    def load_value():
        spec = importlib.util.spec_from_file_location("revert_cache_probe", pending)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.VALUE

    assert load_value() == 2
    bytecode = Path(importlib.util.cache_from_source(str(pending)))
    assert bytecode.is_file()
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    manifest["source_backups"][0]["backup_path"] = valid_backup
    _write_manifest(backup_dir, manifest)

    retried = apk.revert_kernel_patch(manifest_file)

    assert retried["status"] == "ok"
    assert pending.read_text(encoding="utf-8") == "VALUE = 1\n"
    assert load_value() == 1
