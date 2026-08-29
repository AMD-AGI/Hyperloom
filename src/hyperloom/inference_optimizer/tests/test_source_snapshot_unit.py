# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit coverage for the per-KEEP source-layer snapshot contract
(``hyperloom.orchestrator.source_snapshot``): ``_safe_rel`` path sanitation,
``snapshot_source_layer`` capture (upsert/delete/missing), and
``snapshot_is_complete``."""

from __future__ import annotations

import json
from pathlib import Path

from hyperloom.orchestrator.source_snapshot import (
    MANIFEST_NAME,
    _safe_rel,
    snapshot_is_complete,
    snapshot_source_layer,
)


def test_safe_rel_normalizes_leading_slash() -> None:
    assert _safe_rel("/foo/bar.py") == "foo/bar.py"


def test_safe_rel_rejects_empty_or_blank() -> None:
    assert _safe_rel("") is None
    assert _safe_rel("   ") is None
    assert _safe_rel(None) is None


def test_safe_rel_rejects_traversal() -> None:
    assert _safe_rel("../etc/passwd") is None
    assert _safe_rel("a/../../b") is None


def test_safe_rel_passthrough_for_plain_relative_path() -> None:
    assert _safe_rel("vllm/model_executor/layers/foo.py") == ("vllm/model_executor/layers/foo.py")


def test_snapshot_source_layer_upsert_only_is_complete(tmp_path: Path) -> None:
    fw = tmp_path / "framework"
    (fw / "pkg").mkdir(parents=True)
    (fw / "pkg" / "mod.py").write_text("x = 1\n", encoding="utf-8")

    dest = tmp_path / "snap"
    manifest = snapshot_source_layer(
        framework_root=fw,
        base_sha="abc123",
        rel_paths=["pkg/mod.py"],
        dest_dir=dest,
        provenance="integrate_patch",
    )

    assert manifest is not None
    assert manifest["complete"] is True
    ops = {f["rel"]: f["op"] for f in manifest["files"]}
    assert ops == {"pkg/mod.py": "upsert"}
    assert snapshot_is_complete(dest)


def test_snapshot_source_layer_declared_delete_is_complete(tmp_path: Path) -> None:
    fw = tmp_path / "framework"
    (fw / "pkg").mkdir(parents=True)
    kept = fw / "pkg" / "kept.py"
    kept.write_text("print('kept')\n", encoding="utf-8")

    dest = tmp_path / "snap"
    manifest = snapshot_source_layer(
        framework_root=fw,
        base_sha="abc123",
        rel_paths=["pkg/kept.py", "pkg/removed.py"],
        dest_dir=dest,
        provenance="integrate_patch",
        extra={"kernel_id": "k1"},
        declared_ops={"pkg/removed.py": "delete"},
    )

    assert manifest is not None
    assert manifest["complete"] is True
    ops = {f["rel"]: f["op"] for f in manifest["files"]}
    assert ops == {"pkg/kept.py": "upsert", "pkg/removed.py": "delete"}
    on_disk = json.loads((dest / MANIFEST_NAME).read_text(encoding="utf-8"))
    assert on_disk["files"] == manifest["files"]
    assert (dest / "files" / "pkg" / "kept.py").read_text(encoding="utf-8") == "print('kept')\n"


def test_snapshot_source_layer_undeclared_absent_becomes_missing(tmp_path: Path) -> None:
    fw = tmp_path / "framework"
    fw.mkdir(parents=True)

    dest = tmp_path / "snap"
    manifest = snapshot_source_layer(
        framework_root=fw,
        base_sha="abc",
        rel_paths=["pkg/ghost.py"],
        dest_dir=dest,
    )

    assert manifest is not None
    assert manifest["complete"] is False
    ops = {f["rel"]: f["op"] for f in manifest["files"]}
    assert ops == {"pkg/ghost.py": "missing"}
    assert not snapshot_is_complete(dest)


def test_snapshot_source_layer_import_root_recorded(tmp_path: Path) -> None:
    fw = tmp_path / "sglang"
    (fw / "python" / "sglang" / "srt").mkdir(parents=True)
    f = fw / "python" / "sglang" / "srt" / "foo.py"
    f.write_text("x = 1\n", encoding="utf-8")

    dest = tmp_path / "snap"
    manifest = snapshot_source_layer(
        framework_root=fw,
        base_sha="",
        rel_paths=["python/sglang/srt/foo.py"],
        dest_dir=dest,
        import_root="python",
    )

    assert manifest is not None
    assert manifest["import_root"] == "python"
    assert manifest["schema_version"] == 2


def test_snapshot_source_layer_skips_unsafe_paths(tmp_path: Path) -> None:
    fw = tmp_path / "framework"
    fw.mkdir()
    dest = tmp_path / "snap"

    manifest = snapshot_source_layer(
        framework_root=fw,
        base_sha=None,
        rel_paths=["../escape.py", ""],
        dest_dir=dest,
    )

    assert manifest is None
    assert not dest.exists()


def test_snapshot_source_layer_returns_none_when_nothing_capturable(
    tmp_path: Path,
) -> None:
    fw = tmp_path / "framework"
    fw.mkdir()
    dest = tmp_path / "snap"

    manifest = snapshot_source_layer(
        framework_root=fw,
        base_sha="",
        rel_paths=[],
        dest_dir=dest,
    )

    assert manifest is None
    assert not dest.exists()
