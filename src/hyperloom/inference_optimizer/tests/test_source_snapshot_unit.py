# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit coverage for the per-KEEP source-layer snapshot contract
(``hyperloom.orchestrator.source_snapshot``): ``_safe_rel`` path sanitation,
and ``snapshot_source_layer`` capture (upsert/delete/empty)."""

from __future__ import annotations

import json
from pathlib import Path

from hyperloom.orchestrator.source_snapshot import (
    MANIFEST_NAME,
    _safe_rel,
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
    assert _safe_rel("vllm/model_executor/layers/foo.py") == (
        "vllm/model_executor/layers/foo.py"
    )


def test_snapshot_source_layer_captures_upsert_and_delete(tmp_path: Path) -> None:
    fw = tmp_path / "framework"
    (fw / "pkg").mkdir(parents=True)
    kept = fw / "pkg" / "kept.py"
    kept.write_text("print('kept')\n", encoding="utf-8")
    # "removed.py" is a changed path that no longer exists -> recorded as "delete".

    dest = tmp_path / "snap"
    manifest = snapshot_source_layer(
        framework_root=fw,
        base_sha="abc123",
        rel_paths=["pkg/kept.py", "pkg/removed.py"],
        dest_dir=dest,
        provenance="integrate_patch",
        extra={"kernel_id": "k1"},
    )

    assert manifest is not None
    assert manifest["snapshot_dir"] == str(dest)
    assert manifest["base_sha"] == "abc123"
    assert manifest["provenance"] == "integrate_patch"
    assert manifest["extra"] == {"kernel_id": "k1"}
    ops = {f["rel"]: f["op"] for f in manifest["files"]}
    assert ops == {"pkg/kept.py": "upsert", "pkg/removed.py": "delete"}

    # Manifest is persisted alongside a real copy of the file.
    on_disk = json.loads((dest / MANIFEST_NAME).read_text(encoding="utf-8"))
    assert on_disk["files"] == manifest["files"]
    assert (dest / "files" / "pkg" / "kept.py").read_text(encoding="utf-8") == (
        "print('kept')\n"
    )


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
