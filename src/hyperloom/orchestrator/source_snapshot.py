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
directory that no later git hygiene can touch.

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
from pathlib import Path
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

