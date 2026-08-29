"""Durable, per-KEEP snapshot of the framework *source layer*.

Every KEEP snapshots its *realized* file contents into a session-scoped,
self-contained directory that no later git hygiene can touch, rather than
treating the mutable live tree as the source of truth.

On-disk format::

    <snapshot_dir>/manifest.json
        {schema_version, framework_root, base_sha, provenance,
         import_root, complete, files:[{rel, op}], extra}
    <snapshot_dir>/files/<rel>   # full copy of each upserted file

Full-file capture (not a fuzzy diff) reconstructs byte-for-byte regardless of
patch strip levels, generated/untracked files, or whether the framework root is
a git tree at all.

The GEAK handoff consumes a snapshot as a PYTHONPATH entry, not as a tree to
rebuild: ``run_e2e`` joins the ``snapshot_dir`` of every entry it considers
reproducible into ``initial_overlay_pythonpath``. Two consequences fix the shape
of everything below.

``import_root`` must name where modules start within the tree -- ``python`` for a
sglang checkout, ``""`` for a dist-packages install -- because the directory that
travels is ``files/<import_root>``, not the snapshot root. Handing over
``files/`` for a checkout puts ``python/sglang/...`` on the path, where
``import sglang`` does not resolve and the stock install answers instead.

``complete`` is False once any declared path could not be accounted for, and
that is what ``reproducible`` reports downstream -- never the mere existence of
the directory. A snapshot that reports False is dropped from the overlay, so an
optimistic answer here is what silently benchmarks an unpatched tree.
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
    """Return whether the snapshot at ``snapshot_dir`` accounts for every path.

    Reads the manifest rather than trusting the directory's existence. Schema-1
    manifests predate ``complete`` and are judged on their ops instead.

    Args:
        snapshot_dir: A directory written by :func:`snapshot_source_layer`.

    Returns:
        True when the manifest is readable and records no unaccounted path.
    """
    manifest_path = Path(snapshot_dir) / MANIFEST_NAME
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if "complete" in manifest:
        return bool(manifest["complete"])
    # Schema 1 only ever recorded these two ops, so this is always True. That is
    # the intended reading: it had no way to express an unaccounted path, and its
    # snapshots have to be taken at face value or not used at all.
    files = manifest.get("files") or []
    return all(f.get("op") in ("upsert", "delete") for f in files if isinstance(f, dict))


def source_layer_reproducible(entry: dict[str, Any]) -> bool:
    """Return whether an ``optimization_stack`` source-patch entry is replayable.

    Prefers the flag recorded at KEEP time; entries written before it existed are
    judged by reading their manifest.

    Args:
        entry: A ``scope=source_patch`` optimization_stack entry.

    Returns:
        True when the entry carries a snapshot that accounts for every path.
    """
    snapshot_dir = str(entry.get("source_snapshot") or "").strip()
    if not snapshot_dir:
        return False
    recorded = entry.get("source_snapshot_complete")
    if recorded is not None:
        return bool(recorded)
    return snapshot_is_complete(snapshot_dir)


def source_layer_overlay_dir(entry: dict[str, Any]) -> str:
    """Return the directory a consumer puts on PYTHONPATH for ``entry``.

    The snapshot root holds the manifest and a ``files/`` tree, neither of which
    is importable; the entry's recorded import root selects the level within it
    that is. Returns ``""`` when the entry carries no snapshot.

    Args:
        entry: A ``scope=source_patch`` optimization_stack entry.

    Returns:
        str: The overlay directory, or ``""``.
    """
    snapshot_dir = str(entry.get("source_snapshot") or "").strip()
    if not snapshot_dir:
        return ""
    overlay = Path(snapshot_dir) / "files"
    import_root = _safe_rel(str(entry.get("source_import_root") or ""))
    return str(overlay / import_root) if import_root else str(overlay)


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
        declared_ops: ``{rel: op}`` from the caller, which is the only party
            that knows a deletion was intended. An absent path it does not
            declare deleted is recorded ``"missing"``, not ``"delete"``.
        import_root: Where modules start within the tree, relative to ``files/``.

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
