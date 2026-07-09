"""Durable, per-KEEP snapshot of the framework *source layer*.

Root-cause this addresses: ``current_best``'s source-code / artifact wins
(``scope=source_patch``) were applied to a *shared* live git tree and, in a
non-cyclic run, left uncommitted. A later candidate's routine ``git
reset --hard`` / ``git clean -fd`` / ``git stash pop`` on that same shared tree
silently discarded them, so the realized best could no longer be relaunched and
the GEAK handoff had no durable source artifact to reference (it fell back
to the stock installed framework).

The fix is to stop treating the mutable live tree as the source of truth: every
KEEP snapshots its *realized* file contents into a session-scoped, self-contained
directory that no later git hygiene can touch. Both the orchestrator (resume /
relaunch) and GEAK (baseline ref) rebuild an identical tree from the
snapshot via :func:`materialize_source_layer`.

The on-disk format is the cross-tool contract (Hyperloom writes it; the GEAK/
GEAK side re-implements the same trivial reader), so neither side needs to
import the other::

    <snapshot_dir>/manifest.json   # {framework_root, base_sha, files:[{rel,op}]}
    <snapshot_dir>/files/<rel>     # full copy of each realized file

Full-file capture (not a fuzzy diff) is deliberate: it reconstructs
byte-for-byte regardless of patch strip levels, generated/untracked files, or
whether the framework root is a git tree at all.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

MANIFEST_NAME = "manifest.json"
SCHEMA_VERSION = 1


def _safe_rel(rel: str) -> str | None:
    """Normalize + reject unsafe (absolute / traversal) repo-relative paths."""
    rel = str(rel or "").strip().lstrip("/")
    if not rel:
        return None
    parts = Path(rel).parts
    if ".." in parts:
        return None
    return rel


def snapshot_source_layer(
    *,
    framework_root: str | Path,
    base_sha: str | None,
    rel_paths: Iterable[str],
    dest_dir: str | Path,
    provenance: str = "",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Capture the realized contents of ``rel_paths`` under ``framework_root``.

    Args:
        framework_root: The framework checkout the KEEP was applied into.
        base_sha: HEAD sha *before* the KEEP was applied (the clean base the
            snapshot is an overlay on top of). Empty/None => materialization
            falls back to copying the current framework tree.
        rel_paths: Repo-relative paths the KEEP created / modified / deleted.
        dest_dir: Session-scoped destination directory (durable; must survive
            any later git hygiene on ``framework_root``).
        provenance: Free-form origin tag (e.g. ``integrate_patch``).
        extra: Optional metadata folded into the manifest.

    Returns:
        The manifest dict augmented with ``snapshot_dir``, or ``None`` when
        nothing capturable was found (so callers can skip recording it).
    """
    framework_root = Path(framework_root)
    dest_dir = Path(dest_dir)
    files_root = dest_dir / "files"

    captured: list[dict[str, str]] = []
    for raw in sorted({str(p) for p in rel_paths}):
        rel = _safe_rel(raw)
        if rel is None:
            continue
        src = framework_root / rel
        if src.is_file():
            dst = files_root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            captured.append({"rel": rel, "op": "upsert"})
        else:
            # Absent post-apply => the KEEP deleted it; record the removal so
            # materialization reproduces the deletion on top of the base tree.
            captured.append({"rel": rel, "op": "delete"})

    if not captured:
        return None

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "framework_root": str(framework_root),
        "base_sha": str(base_sha or ""),
        "provenance": provenance,
        "files": captured,
    }
    if extra:
        manifest["extra"] = extra

    dest_dir.mkdir(parents=True, exist_ok=True)
    (dest_dir / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return {"snapshot_dir": str(dest_dir), **manifest}


def _symlink_mirror(mirror_root: Path, changed_rels: set[str], tree: Path) -> None:
    """Build ``tree`` as a symlink mirror of ``mirror_root``, keeping real dirs
    only along the paths to ``changed_rels``.

    We mirror the *installed* framework so its prebuilt JIT ``.so`` / compiled
    artifacts survive (via symlinks) while only the changed files become real
    copies. A from-scratch ``git worktree`` checkout would be source-only and
    force the engine to JIT-rebuild every kernel at request time, which can fail
    and crash the server.
    """
    changed = {str(PurePosixPath(r)) for r in changed_rels}
    ancestors: set[str] = {"."}
    for r in changed:
        parts = PurePosixPath(r).parts
        for i in range(1, len(parts)):
            ancestors.add(str(PurePosixPath(*parts[:i])))

    def rec(src: Path, dst: Path, rel: str) -> None:
        dst.mkdir(parents=True, exist_ok=True)
        for child in src.iterdir():
            crel = child.name if rel == "." else f"{rel}/{child.name}"
            if crel in changed:
                continue
            link = dst / child.name
            if child.is_dir() and crel in ancestors:
                rec(child, link, crel)
            else:
                try:
                    link.symlink_to(child)
                except FileExistsError:
                    pass

    if tree.exists():
        shutil.rmtree(tree, ignore_errors=True)
    if mirror_root.is_dir():
        rec(mirror_root, tree, ".")
    else:
        tree.mkdir(parents=True, exist_ok=True)


def materialize_source_layer(
    snapshot_dir: str | Path,
    work_root: str | Path,
    mirror_root: str | Path | None = None,
) -> str | None:
    """Rebuild the patched source tree from a snapshot; return its import root.

    Mirrors the installed framework (``mirror_root``, default the snapshot's
    ``framework_root``) via symlinks — preserving prebuilt JIT / compiled
    artifacts — and overlays ONLY the snapshot's changed files as real copies.
    This is robust to the shared live tree having been reset in the meantime:
    the overlay authoritatively pins exactly the files the KEEP changed.

    Args:
        snapshot_dir: A directory produced by :func:`snapshot_source_layer`.
        work_root: Scratch directory the reconstructed tree is built under.
        mirror_root: Installed framework to mirror (defaults to the snapshot's
            ``framework_root``); override when the captured tree differs from the
            runtime install location.

    Returns:
        Absolute path to the reconstructed framework root (prepend to
        ``PYTHONPATH`` so its packages shadow the installed ones), or ``None``
        if the snapshot is unreadable.
    """
    snapshot_dir = Path(snapshot_dir)
    work_root = Path(work_root)
    manifest_path = snapshot_dir / MANIFEST_NAME
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    mirror = Path(str(mirror_root) if mirror_root else str(manifest.get("framework_root") or ""))
    work_root.mkdir(parents=True, exist_ok=True)
    tree = work_root / "tree"

    files_root = snapshot_dir / "files"
    records = manifest.get("files", [])
    changed_rels = {r for r in (_safe_rel(str(x.get("rel") or "")) for x in records) if r}
    _symlink_mirror(mirror, changed_rels, tree)

    for rec in records:
        rel = _safe_rel(str(rec.get("rel") or ""))
        if rel is None:
            continue
        dst = tree / rel
        if str(rec.get("op")) == "delete":
            try:
                if dst.is_symlink() or dst.exists():
                    dst.unlink()
            except OSError:
                pass
            continue
        src = files_root / rel
        if src.is_file():
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.is_symlink() or dst.exists():
                try:
                    dst.unlink()
                except OSError:
                    pass
            shutil.copy2(src, dst)

    return str(tree)
