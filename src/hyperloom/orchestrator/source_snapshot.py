"""Durable, per-KEEP snapshot of the framework *source layer*."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Iterable

MANIFEST_NAME = "manifest.json"
SCHEMA_VERSION = 2


def _safe_rel(rel: str) -> str | None:
    """Normalize a repo-relative path (strip whitespace and leading slashes); return None for empty paths or any
    containing ``..``.
    """
    rel = str(rel or "").strip().lstrip("/")
    if not rel:
        return None
    parts = Path(rel).parts
    if ".." in parts:
        return None
    return rel


def snapshot_is_complete(snapshot_dir: str | Path) -> bool:
    """Return whether the snapshot at ``snapshot_dir`` accounts for every path."""
    manifest_path = Path(snapshot_dir) / MANIFEST_NAME
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if "complete" in manifest:
        return bool(manifest["complete"])
    # Schema 1 only ever recorded these two ops, so this is always True.
    files = manifest.get("files") or []
    return all(f.get("op") in ("upsert", "delete") for f in files if isinstance(f, dict))


def source_layer_reproducible(entry: dict[str, Any]) -> bool:
    """Return whether an ``optimization_stack`` source-patch entry is replayable."""
    snapshot_dir = str(entry.get("source_snapshot") or "").strip()
    if not snapshot_dir:
        return False
    recorded = entry.get("source_snapshot_complete")
    if recorded is not None:
        return bool(recorded)
    return snapshot_is_complete(snapshot_dir)


def source_layer_overlay_dir(entry: dict[str, Any]) -> str:
    """Return the directory a consumer puts on PYTHONPATH for ``entry``."""
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
    """Capture the realized contents of ``rel_paths`` under ``framework_root``."""
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
