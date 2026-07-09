# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Unit coverage for the per-KEEP source-layer snapshot/materialize contract
(``hyperloom.orchestrator.source_snapshot``): ``_safe_rel`` path sanitation,
``snapshot_source_layer`` capture (upsert/delete/empty), and
``materialize_source_layer`` reconstruction (missing/corrupt manifest,
symlink-mirror overlay, delete replay, ``mirror_root`` override)."""

from __future__ import annotations

import json
from pathlib import Path

from hyperloom.orchestrator.source_snapshot import (
    MANIFEST_NAME,
    _safe_rel,
    _symlink_mirror,
    materialize_source_layer,
    snapshot_source_layer,
)


# ── _safe_rel ─────────────────────────────────────────────────────────────


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


# ── snapshot_source_layer ────────────────────────────────────────────────


def test_snapshot_source_layer_captures_upsert_and_delete(tmp_path: Path) -> None:
    fw = tmp_path / "framework"
    (fw / "pkg").mkdir(parents=True)
    kept = fw / "pkg" / "kept.py"
    kept.write_text("print('kept')\n", encoding="utf-8")
    # "removed.py" is listed as a changed path but no longer exists post-apply
    # (the KEEP deleted it) -> must be recorded as a "delete" op, not skipped.

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

    # The manifest is durably persisted alongside a real copy of the file.
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


# ── _symlink_mirror ───────────────────────────────────────────────────────


def test_symlink_mirror_links_unchanged_and_skips_changed(tmp_path: Path) -> None:
    mirror_root = tmp_path / "installed"
    (mirror_root / "sub").mkdir(parents=True)
    (mirror_root / "top.py").write_text("top", encoding="utf-8")
    (mirror_root / "sub" / "changed.py").write_text("old", encoding="utf-8")
    (mirror_root / "sub" / "kept.py").write_text("kept", encoding="utf-8")

    tree = tmp_path / "tree"
    _symlink_mirror(mirror_root, {"sub/changed.py"}, tree)

    # Unchanged files are mirrored as symlinks back into the installed tree.
    assert (tree / "top.py").is_symlink()
    assert (tree / "sub" / "kept.py").is_symlink()
    assert (tree / "sub" / "kept.py").read_text(encoding="utf-8") == "kept"
    # The changed file itself is NOT linked (the overlay step owns it).
    assert not (tree / "sub" / "changed.py").exists()
    # Ancestor directories of a changed file are real dirs, not symlinks.
    assert (tree / "sub").is_dir()
    assert not (tree / "sub").is_symlink()


def test_symlink_mirror_rebuilds_existing_tree(tmp_path: Path) -> None:
    mirror_root = tmp_path / "installed"
    mirror_root.mkdir()
    (mirror_root / "a.py").write_text("a", encoding="utf-8")

    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "stale.py").write_text("stale", encoding="utf-8")

    _symlink_mirror(mirror_root, set(), tree)

    assert not (tree / "stale.py").exists()
    assert (tree / "a.py").is_symlink()


def test_symlink_mirror_handles_missing_mirror_root(tmp_path: Path) -> None:
    tree = tmp_path / "tree"
    _symlink_mirror(tmp_path / "does-not-exist", set(), tree)

    assert tree.is_dir()
    assert list(tree.iterdir()) == []


def test_symlink_mirror_tolerates_preexisting_link(
    tmp_path: Path, monkeypatch
) -> None:
    """A ``FileExistsError`` from ``Path.symlink_to`` (e.g. a concurrent writer
    already staged the same name) is swallowed, not raised."""
    mirror_root = tmp_path / "installed"
    mirror_root.mkdir()
    (mirror_root / "a.py").write_text("a", encoding="utf-8")

    from hyperloom.orchestrator import source_snapshot as mod

    real_symlink_to = Path.symlink_to
    calls: list[Path] = []

    def flaky_symlink_to(self: Path, target, *a, **kw):  # noqa: ANN001
        calls.append(self)
        if len(calls) == 1:
            raise FileExistsError(self)
        return real_symlink_to(self, target, *a, **kw)

    monkeypatch.setattr(mod.Path, "symlink_to", flaky_symlink_to)

    tree = tmp_path / "tree"
    _symlink_mirror(mirror_root, set(), tree)  # must not raise
    assert calls  # the flaky path was actually exercised


# ── materialize_source_layer ─────────────────────────────────────────────


def test_materialize_source_layer_missing_manifest_returns_none(
    tmp_path: Path,
) -> None:
    result = materialize_source_layer(
        snapshot_dir=tmp_path / "no-such-snapshot",
        work_root=tmp_path / "work",
    )
    assert result is None


def test_materialize_source_layer_corrupt_manifest_returns_none(
    tmp_path: Path,
) -> None:
    snap = tmp_path / "snap"
    snap.mkdir()
    (snap / MANIFEST_NAME).write_text("{not-json", encoding="utf-8")

    result = materialize_source_layer(
        snapshot_dir=snap,
        work_root=tmp_path / "work",
    )
    assert result is None


def test_materialize_source_layer_round_trip(tmp_path: Path) -> None:
    fw = tmp_path / "framework"
    (fw / "pkg").mkdir(parents=True)
    (fw / "pkg" / "untouched.py").write_text("untouched", encoding="utf-8")
    (fw / "pkg" / "deleted.py").write_text("will be deleted", encoding="utf-8")
    (fw / "pkg" / "edited.py").write_text("old contents", encoding="utf-8")

    dest = tmp_path / "snap"
    # Simulate the post-apply tree: "edited.py" now holds the new contents,
    # "deleted.py" no longer exists.
    (fw / "pkg" / "edited.py").write_text("new contents", encoding="utf-8")
    (fw / "pkg" / "deleted.py").unlink()

    manifest = snapshot_source_layer(
        framework_root=fw,
        base_sha="deadbeef",
        rel_paths=["pkg/edited.py", "pkg/deleted.py"],
        dest_dir=dest,
    )
    assert manifest is not None

    # Independently, the "live" framework tree drifts back to its clean base
    # (e.g. a later candidate's ``git reset --hard``) -- materialization must
    # not depend on it any longer.
    (fw / "pkg" / "edited.py").write_text("old contents", encoding="utf-8")
    (fw / "pkg" / "deleted.py").write_text("resurrected by git reset", encoding="utf-8")

    work_root = tmp_path / "work"
    tree_str = materialize_source_layer(snapshot_dir=dest, work_root=work_root)

    assert tree_str is not None
    tree = Path(tree_str)
    assert tree == work_root / "tree"
    # Overlay applied: the edited file reflects the *realized* (snapshot)
    # contents, not the drifted live tree.
    assert (tree / "pkg" / "edited.py").read_text(encoding="utf-8") == "new contents"
    # Deletion replayed even though the live tree resurrected the file.
    assert not (tree / "pkg" / "deleted.py").exists()
    # Untouched files are still reachable via the mirror symlink.
    assert (tree / "pkg" / "untouched.py").read_text(encoding="utf-8") == "untouched"


def test_materialize_source_layer_delete_op_on_missing_dst_is_noop(
    tmp_path: Path,
) -> None:
    fw = tmp_path / "framework"
    fw.mkdir()
    dest = tmp_path / "snap"
    dest.mkdir()
    (dest / MANIFEST_NAME).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "framework_root": str(fw),
                "base_sha": "",
                "files": [{"rel": "never/existed.py", "op": "delete"}],
            }
        ),
        encoding="utf-8",
    )

    tree_str = materialize_source_layer(snapshot_dir=dest, work_root=tmp_path / "work")

    assert tree_str is not None
    assert not (Path(tree_str) / "never" / "existed.py").exists()


def test_materialize_source_layer_skips_unsafe_manifest_entry(tmp_path: Path) -> None:
    fw = tmp_path / "framework"
    fw.mkdir()
    dest = tmp_path / "snap"
    dest.mkdir()
    (dest / MANIFEST_NAME).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "framework_root": str(fw),
                "base_sha": "",
                "files": [{"rel": "../escape.py", "op": "delete"}],
            }
        ),
        encoding="utf-8",
    )

    tree_str = materialize_source_layer(snapshot_dir=dest, work_root=tmp_path / "work")

    assert tree_str is not None


def test_materialize_source_layer_delete_swallows_os_error(
    tmp_path: Path, monkeypatch
) -> None:
    """An "upsert" followed by a "delete" of the same rel path means the
    overlay copy lands in ``tree`` before the delete record runs; if
    ``Path.unlink()`` then raises ``OSError`` (e.g. a permissions race), it
    must be swallowed rather than propagated."""
    fw = tmp_path / "framework"
    (fw / "pkg").mkdir(parents=True)
    (fw / "pkg" / "file.py").write_text("content", encoding="utf-8")

    dest = tmp_path / "snap"
    dest.mkdir()
    (dest / "files" / "pkg").mkdir(parents=True)
    (dest / "files" / "pkg" / "file.py").write_text("content", encoding="utf-8")
    (dest / MANIFEST_NAME).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "framework_root": str(fw),
                "base_sha": "",
                "files": [
                    {"rel": "pkg/file.py", "op": "upsert"},
                    {"rel": "pkg/file.py", "op": "delete"},
                ],
            }
        ),
        encoding="utf-8",
    )

    from hyperloom.orchestrator import source_snapshot as mod

    real_unlink = Path.unlink

    def flaky_unlink(self: Path, *a, **kw):  # noqa: ANN001
        if self.name == "file.py":
            raise OSError("simulated permission race")
        return real_unlink(self, *a, **kw)

    monkeypatch.setattr(mod.Path, "unlink", flaky_unlink)

    tree_str = materialize_source_layer(snapshot_dir=dest, work_root=tmp_path / "work")

    assert tree_str is not None
    # The delete's OSError was swallowed, so the (stale) overlay copy is
    # still present rather than the call raising.
    assert (Path(tree_str) / "pkg" / "file.py").read_text(encoding="utf-8") == "content"


def test_materialize_source_layer_overlay_replaces_existing_dst(
    tmp_path: Path,
) -> None:
    """Two "upsert" records for the same rel path (duplicate/updated entry):
    the second overlay copy must unlink the first before re-copying."""
    fw = tmp_path / "framework"
    (fw / "pkg").mkdir(parents=True)
    (fw / "pkg" / "file.py").write_text("second", encoding="utf-8")

    dest = tmp_path / "snap"
    dest.mkdir()
    (dest / "files" / "pkg").mkdir(parents=True)
    (dest / "files" / "pkg" / "file.py").write_text("second", encoding="utf-8")
    (dest / MANIFEST_NAME).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "framework_root": str(fw),
                "base_sha": "",
                "files": [
                    {"rel": "pkg/file.py", "op": "upsert"},
                    {"rel": "pkg/file.py", "op": "upsert"},
                ],
            }
        ),
        encoding="utf-8",
    )

    tree_str = materialize_source_layer(snapshot_dir=dest, work_root=tmp_path / "work")

    assert tree_str is not None
    assert (Path(tree_str) / "pkg" / "file.py").read_text(encoding="utf-8") == "second"


def test_materialize_source_layer_overlay_unlink_swallows_os_error(
    tmp_path: Path, monkeypatch
) -> None:
    """Same as above, but the pre-copy ``unlink()`` on the stale overlay
    target itself raises ``OSError`` -- the subsequent ``shutil.copy2`` must
    still run rather than the whole call raising."""
    fw = tmp_path / "framework"
    (fw / "pkg").mkdir(parents=True)
    (fw / "pkg" / "file.py").write_text("second", encoding="utf-8")

    dest = tmp_path / "snap"
    dest.mkdir()
    (dest / "files" / "pkg").mkdir(parents=True)
    (dest / "files" / "pkg" / "file.py").write_text("second", encoding="utf-8")
    (dest / MANIFEST_NAME).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "framework_root": str(fw),
                "base_sha": "",
                "files": [
                    {"rel": "pkg/file.py", "op": "upsert"},
                    {"rel": "pkg/file.py", "op": "upsert"},
                ],
            }
        ),
        encoding="utf-8",
    )

    from hyperloom.orchestrator import source_snapshot as mod

    real_unlink = Path.unlink
    calls: list[Path] = []

    def flaky_unlink(self: Path, *a, **kw):  # noqa: ANN001
        calls.append(self)
        if len(calls) == 1:
            raise OSError("simulated permission race")
        return real_unlink(self, *a, **kw)

    monkeypatch.setattr(mod.Path, "unlink", flaky_unlink)

    tree_str = materialize_source_layer(snapshot_dir=dest, work_root=tmp_path / "work")

    assert tree_str is not None
    assert calls
    assert (Path(tree_str) / "pkg" / "file.py").read_text(encoding="utf-8") == "second"


def test_materialize_source_layer_honors_mirror_root_override(tmp_path: Path) -> None:
    captured_root = tmp_path / "captured-at-apply-time"
    (captured_root / "pkg").mkdir(parents=True)
    (captured_root / "pkg" / "file.py").write_text("v1", encoding="utf-8")

    override_root = tmp_path / "actual-runtime-install"
    (override_root / "pkg").mkdir(parents=True)
    (override_root / "pkg" / "other.py").write_text("from override", encoding="utf-8")

    dest = tmp_path / "snap"
    manifest = snapshot_source_layer(
        framework_root=captured_root,
        base_sha="",
        rel_paths=["pkg/file.py"],
        dest_dir=dest,
    )
    assert manifest is not None

    tree_str = materialize_source_layer(
        snapshot_dir=dest,
        work_root=tmp_path / "work",
        mirror_root=override_root,
    )

    assert tree_str is not None
    tree = Path(tree_str)
    # Mirror comes from the override root, not the manifest's framework_root.
    assert (tree / "pkg" / "other.py").read_text(encoding="utf-8") == "from override"
    # The snapshot overlay is still applied on top of that mirror.
    assert (tree / "pkg" / "file.py").read_text(encoding="utf-8") == "v1"
