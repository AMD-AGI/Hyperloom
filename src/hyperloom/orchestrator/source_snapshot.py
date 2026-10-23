"""Durable, per-KEEP snapshot of the framework *source layer*."""

from __future__ import annotations

import filecmp
import json
import shutil
from pathlib import Path
from typing import Any, Callable, Iterable, NamedTuple

MANIFEST_NAME = "manifest.json"
SCHEMA_VERSION = 2
_OP_UPSERT = "upsert"
_FILES = "files"
_ENCODING = "utf-8"


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
        manifest = json.loads(manifest_path.read_text(encoding=_ENCODING))
    except (OSError, ValueError):
        return False
    if "complete" in manifest:
        return bool(manifest["complete"])
    # Schema 1 only ever recorded these two ops, so this is always True.
    files = manifest.get(_FILES) or []
    return all(f.get("op") in (_OP_UPSERT, "delete") for f in files if isinstance(f, dict))


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
    overlay = Path(snapshot_dir) / _FILES
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
    files_root = dest_dir / _FILES
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
            captured.append({"rel": rel, "op": _OP_UPSERT})
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
        _FILES: captured,
    }
    if extra:
        manifest["extra"] = extra

    dest_dir.mkdir(parents=True, exist_ok=True)
    (dest_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2), encoding=_ENCODING)
    return {"snapshot_dir": str(dest_dir), **manifest}


class SourceLayerInstall(NamedTuple):
    """What installing the kept source layers on their framework trees did."""

    #: Paths written or removed because the tree did not hold the kept state.
    installed: list[str]
    #: Why a kept state could not be put on its tree, one line each.
    failed: list[str]
    #: Variants whose source patch has no complete snapshot, so there is nothing to install or check.
    unreproducible: list[str]
    #: Every file the kept states cover, relative to the framework root that holds it.
    kept: dict[Path, list[str]]
    #: Paths left alone because the tree holds committed work newer than their kept state.
    superseded: list[str]


def _kept_final_states(
    stack: Iterable[dict[str, Any]], result: SourceLayerInstall
) -> dict[tuple[Path, str], tuple[str, Path]]:
    """Map each ``(framework root, rel)`` a kept source patch touches to its last kept ``(op, snapshot file)``."""
    final: dict[tuple[Path, str], tuple[str, Path]] = {}
    for entry in stack:
        if not isinstance(entry, dict) or entry.get("scope") != "source_patch":
            continue
        variant = str(entry.get("variant_name") or entry.get("name") or "")
        if not source_layer_reproducible(entry):
            result.unreproducible.append(variant)
            continue
        snapshot_dir = Path(str(entry["source_snapshot"]))
        root = str(entry.get("framework_root") or "").strip()
        try:
            manifest = json.loads((snapshot_dir / MANIFEST_NAME).read_text(encoding=_ENCODING))
        except (OSError, ValueError) as exc:
            result.failed.append(f"{variant}: snapshot manifest unreadable: {exc}")
            continue
        if not root or not Path(root).is_dir():
            result.failed.append(f"{variant}: framework root {root or '(none recorded)'} is not a directory")
            continue
        for row in manifest.get(_FILES) or []:
            rel = _safe_rel(str(row.get("rel") or "")) if isinstance(row, dict) else None
            if rel is None:
                result.failed.append(f"{variant}: manifest names an unusable path {row!r}")
                continue
            final[(Path(root), rel)] = (str(row.get("op") or ""), snapshot_dir / _FILES / rel)
            covered = result.kept.setdefault(Path(root), [])
            if rel not in covered:
                covered.append(rel)
    return final


def install_source_layers(
    stack: Iterable[dict[str, Any]],
    superseded: Callable[[Path, str, str, Path], bool] = lambda root, rel, op, source: False,
) -> SourceLayerInstall:
    """Put every kept source patch's final file state on its framework tree, and confirm each file is in place.

    A tree that already holds the kept state is left untouched, and so is a file ``superseded(root, rel, op, source)``
    says the tree has moved past. The checks and writes run in stack order, so a file two patches touch ends in the
    last one's state.
    """
    result = SourceLayerInstall([], [], [], {}, [])
    for (root, rel), (op, source) in _kept_final_states(stack, result).items():
        dest = root / rel
        try:
            if op == "delete":
                if not dest.exists():
                    continue
            elif op != _OP_UPSERT or not source.is_file():
                result.failed.append(f"{dest}: the snapshot holds no {op or 'unrecorded'} state for it")
                continue
            elif dest.is_file() and filecmp.cmp(source, dest, shallow=False):
                continue
            if superseded(root, rel, op, source):
                result.superseded.append(str(dest))
                continue
            if op == "delete":
                dest.unlink()
            else:
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, dest)
                if not filecmp.cmp(source, dest, shallow=False):
                    result.failed.append(f"{dest}: differs from the snapshot after it was written")
                    continue
            result.installed.append(str(dest))
        except OSError as exc:
            result.failed.append(f"{dest}: {exc}")
    return result
