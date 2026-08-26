"""Durable, per-KEEP snapshot of the framework *source layer*.

Every KEEP snapshots its *realized* file contents into a session-scoped,
self-contained directory that no later git hygiene can touch, rather than
treating the mutable live tree as the source of truth.

On-disk format::

    <snapshot_dir>/manifest.json
        {schema_version, framework_root, base_sha, provenance,
         import_root, complete, files:[{rel, op}], extra}
    <snapshot_dir>/files/<rel>   # full copy of each upserted file

``import_root`` is a path component relative to ``snapshot_dir/files/`` that
locates the Python import root for this framework. GEAK prepends
``snapshot_dir/files/<import_root>`` onto PYTHONPATH so the patched modules
are importable from the snapshot without the ``files/`` indirection.
For a sglang repo checkout, ``import_root`` is ``python``; for a dist-packages
install it is ``""``.

``complete`` is True when every declared file was successfully copied and no
entry carries ``op: "missing"``. Callers derive ``reproducible`` from this
flag rather than from path existence alone.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Iterable

MANIFEST_NAME = "manifest.json"
SCHEMA_VERSION = 2


def _safe_rel(rel: str) -> str | None:
    """Normalize a repo-relative path (strip whitespace and leading slashes);
    return None for empty paths or any containing ``..``."""
    rel = str(rel or "").strip().lstrip("/")
    if not rel:
        return None
    parts = Path(rel).parts
    if ".." in parts:
        return None
    return rel


def snapshot_is_complete(snapshot_dir: str | Path) -> bool:
    """Return True when the snapshot has a manifest and no ``op: "missing"`` entries.

    Args:
        snapshot_dir: Path to the snapshot directory written by
            :func:`snapshot_source_layer`.

    Returns:
        True when the manifest exists, ``complete`` is True (schema ≥ 2), or
        all entries carry ``"upsert"`` or ``"delete"`` (schema 1 fallback).
    """
    try:
        manifest_path = Path(snapshot_dir) / MANIFEST_NAME
        if not manifest_path.is_file():
            return False
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if "complete" in manifest:
            return bool(manifest["complete"])
        files = manifest.get("files") or []
        return all(f.get("op") in ("upsert", "delete") for f in files if isinstance(f, dict))
    except (OSError, json.JSONDecodeError, ValueError):
        return False


def snapshot_source_layer(
    *,
    framework_root: str | Path,
    base_sha: str | None,
    rel_paths: Iterable[str],
    dest_dir: str | Path,
    provenance: str = "",
    extra: dict[str, Any] | None = None,
    declared_ops: dict[str, str] | None = None,
    import_root: str = "",
) -> dict[str, Any] | None:
    """Capture the realized contents of ``rel_paths`` under ``framework_root``.

    Args:
        framework_root: The framework checkout the KEEP was applied into.
        base_sha: HEAD sha *before* the KEEP was applied, recorded verbatim in
            the manifest for downstream provenance. Empty/None is stored as an
            empty string; nothing in this module branches on it.
        rel_paths: Repo-relative paths the KEEP created / modified / deleted.
        dest_dir: Session-scoped destination directory (durable; must survive
            any later git hygiene on ``framework_root``).
        provenance: Free-form origin tag (e.g. ``integrate_patch``).
        extra: Optional metadata folded into the manifest.
        declared_ops: Caller-supplied ``{rel: op}`` mapping.  When a path is
            in this dict with ``op="delete"`` and absent from disk, it is
            recorded as a genuine deletion. Paths absent from disk and not in
            ``declared_ops`` are recorded with ``op="missing"`` and cause
            ``complete=False``.
        import_root: The Python import root relative to ``files/``; placed on
            PYTHONPATH by the GEAK baseline server. ``"python"`` for sglang
            repo checkouts; ``""`` for dist-packages installs.

    Returns:
        The manifest dict augmented with ``snapshot_dir``, or ``None`` when
        nothing capturable was found (so callers can skip recording it).
    """
    framework_root = Path(framework_root)
    dest_dir = Path(dest_dir)
    files_root = dest_dir / "files"
    ops = dict(declared_ops or {})

    captured: list[dict[str, str]] = []
    all_complete = True
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
        elif ops.get(rel) == "delete":
            captured.append({"rel": rel, "op": "delete"})
        else:
            captured.append({"rel": rel, "op": "missing"})
            all_complete = False

    if not captured:
        return None

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "framework_root": str(framework_root),
        "base_sha": str(base_sha or ""),
        "provenance": provenance,
        "import_root": import_root,
        "complete": all_complete,
        "files": captured,
    }
    if extra:
        manifest["extra"] = extra

    dest_dir.mkdir(parents=True, exist_ok=True)
    (dest_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"snapshot_dir": str(dest_dir), **manifest}
