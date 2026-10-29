# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""On-disk recipe-snapshot store selected by ``KNOWLEDGE_STORE_MODE=local``."""

from __future__ import annotations

import fcntl
import json
import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from common.io import WriteOptions, atomic_write_json
from hyperloom.common.jsonio import read_json_strict
from hyperloom.common.timeutil import now_iso

from .canonical_id import (
    InvalidCanonicalIdError,
    canonical_id_for_path,
    cid_to_path_components,
)
from .schema import Recipe


log = logging.getLogger(__name__)


# Filenames + sub-paths within one recipe directory
RECIPE_FILENAME: str = "recipe.json"
HISTORY_DIRNAME: str = "history"
LOCK_FILENAME: str = ".lock"
HISTORY_VERSION_PREFIX: str = "v"
HISTORY_VERSION_SUFFIX: str = ".json"


# List-valued knowledge fields whose pre/post-write sizes ``put_recipe`` reports back, so an audit consumer can derive
# per-write deltas.
_COUNTED_COLLECTIONS: tuple[str, ...] = (
    "lessons",
    "pitfalls",
    "what_worked",
    "what_failed",
    "remaining_gaps",
    "sessions",
)


@dataclass(frozen=True)
class RecipeUpsert:
    """The producer-side fields of one recipe row, before ``put_recipe`` normalises and versions it."""

    canonical_id: str
    # 7-tuple identity (also encoded in canonical_id; stamped at the top
    # level for arbor-compat).
    model: str = ""
    hardware: str = ""
    framework_name: str = ""
    framework_version: str = ""
    precision: str = ""
    # Arbor-aligned payload.
    best_config: dict[str, str] | None = None
    best_throughput: float = 0.0
    what_worked: list[Any] | None = None
    what_failed: list[Any] | None = None
    remaining_gaps: list[Any] | None = None
    pitfalls: list[Any] | None = None
    lessons: list[Any] | None = None
    last_profiled: str = ""
    stack_fingerprint: dict[str, str] | None = None
    sessions: list[Any] | None = None
    # v2 audit / wire-compat fields. provenance is required by the central server so we always stamp something.
    authority: str = "EXPERIENTIAL"
    confidence: float = 0.85
    evidence_refs: list[Any] | None = None
    provenance: dict[str, Any] | None = None
    # Free-form session-level keys preserved verbatim across rewrite.
    extras: dict[str, Any] | None = None


# Errors
class LocalRecipeStoreError(RuntimeError):
    """Raised on unrecoverable failures inside the local KB store."""


def _read_json(path: Path) -> dict[str, Any] | None:
    """Read a JSON file or return ``None`` if it doesn't exist."""
    if not path.is_file():
        return None
    try:
        return read_json_strict(path, require_dict=True)
    except FileNotFoundError:
        # Race: file disappeared between is_file() and open(); treat as missing.
        return None
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise LocalRecipeStoreError(
            f"failed to read {path}: {exc}",
        ) from exc


# Per-lock-file mutexes, shared across ``_CidLock`` instances.
_CID_MUTEXES: dict[str, threading.Lock] = {}
_CID_MUTEX_REGISTRY_LOCK = threading.Lock()


def _cid_mutex(path: Path) -> threading.Lock:
    """Return the process-wide mutex for one ``.lock`` file path."""
    key = os.path.abspath(str(path))
    with _CID_MUTEX_REGISTRY_LOCK:
        mutex = _CID_MUTEXES.get(key)
        if mutex is None:
            mutex = threading.Lock()
            _CID_MUTEXES[key] = mutex
        return mutex


# Per-cid lock
@dataclass
class _CidLock:
    """Exclusive file-lock for one canonical_id directory."""

    path: Path
    _mutex: threading.Lock | None = field(default=None, init=False, repr=False)
    _fd: int | None = field(default=None, init=False, repr=False)

    def __enter__(self) -> _CidLock:
        """Acquire the process mutex then the exclusive file lock."""
        self._mutex = _cid_mutex(self.path)
        self._mutex.acquire()
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
            fcntl.flock(self._fd, fcntl.LOCK_EX)
        except BaseException:
            if self._fd is not None:
                try:
                    os.close(self._fd)
                except OSError:
                    pass
                self._fd = None
            self._mutex.release()
            self._mutex = None
            raise
        return self

    def __exit__(self, *_exc: Any) -> None:
        """Release the file lock then the process mutex."""
        if self._fd is not None:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
            finally:
                try:
                    os.close(self._fd)
                except OSError:
                    pass
                self._fd = None
        if self._mutex is not None:
            self._mutex.release()
            self._mutex = None


# LocalRecipeStore
@dataclass
class LocalRecipeStore:
    """Filesystem-backed recipe-snapshot store."""

    root: Path

    def __post_init__(self) -> None:
        """Coerce ``root`` to a :class:`~pathlib.Path`."""
        self.root = Path(self.root)

    # Path helpers
    def _recipe_dir(self, canonical_id: str) -> Path:
        """Return the 7-level directory holding one cid's files."""
        return self.root.joinpath(*cid_to_path_components(canonical_id))

    def _live_path(self, canonical_id: str) -> Path:
        """Return the path to a cid's live ``recipe.json``."""
        return self._recipe_dir(canonical_id) / RECIPE_FILENAME

    def _history_dir(self, canonical_id: str) -> Path:
        """Return the path to a cid's ``history`` directory."""
        return self._recipe_dir(canonical_id) / HISTORY_DIRNAME

    def _history_version_path(self, canonical_id: str, version: int) -> Path:
        """Return the archive path for a specific recipe version."""
        return self._history_dir(canonical_id) / f"{HISTORY_VERSION_PREFIX}{int(version)}{HISTORY_VERSION_SUFFIX}"

    def _lock_path(self, canonical_id: str) -> Path:
        """Return the path to a cid's advisory ``.lock`` file."""
        return self._recipe_dir(canonical_id) / LOCK_FILENAME

    def _walk_recipe_dirs(self) -> Iterable[Path]:
        """Yield every 7-level directory below ``root`` holding a live ``recipe.json``."""
        if not self.root.is_dir():
            return
        for recipe_path in self.root.rglob(RECIPE_FILENAME):
            if not recipe_path.is_file():
                continue
            recipe_dir = recipe_path.parent
            try:
                rel_parts = recipe_dir.relative_to(self.root).parts
            except ValueError:
                continue
            if len(rel_parts) != 7:
                log.debug(
                    "skipping %s: not at the required 7-level depth",
                    recipe_dir,
                )
                continue
            yield recipe_dir

    def _archive_live_version(
        self,
        *,
        canonical_id: str,
        snapshot: dict[str, Any],
        version: int,
        archived_at: str,
        replaced_by: dict[str, Any] | None,
    ) -> None:
        """Copy the live row into ``history`` under its own version, leaving an identical archive untouched.

        ``replaced_by`` carries the triggering write's provenance, so an audit consumer can
        attribute the archive to the write that displaced it.
        """
        archive_path = self._history_version_path(canonical_id, version)
        try:
            existing_archive = _read_json(archive_path)
        except LocalRecipeStoreError as exc:
            log.warning(
                "put_recipe: unreadable history v%s at %s (%s); rewriting it",
                version,
                archive_path,
                exc,
            )
            existing_archive = None
        if (
            isinstance(existing_archive, dict)
            and isinstance(existing_archive.get("snapshot"), dict)
            and existing_archive["snapshot"] == snapshot
        ):
            log.debug(
                "put_recipe: keeping existing history v%s at %s (crash residue)",
                version,
                archive_path,
            )
            return
        archive_payload: dict[str, Any] = {
            "canonical_id": canonical_id,
            "version": version,
            "archived_at": archived_at,
            "replaced_by": dict(replaced_by or {}),
            "snapshot": snapshot,
        }
        atomic_write_json(archive_path, archive_payload, options=WriteOptions(make_parents=True, fsync=True))

    # put_recipe
    def put_recipe(self, upsert: RecipeUpsert) -> dict[str, Any]:
        """Atomically upsert a recipe row and archive the prior live version."""
        if not upsert.canonical_id:
            raise ValueError("put_recipe requires a non-empty canonical_id")
        recipe_dir = self._recipe_dir(upsert.canonical_id)
        recipe_dir.mkdir(parents=True, exist_ok=True)
        lock = _CidLock(self._lock_path(upsert.canonical_id))
        with lock:
            now = now_iso(timespec="microseconds")
            live = _read_json(self._live_path(upsert.canonical_id))
            created = live is None
            prior_counts = _collection_counts(live)
            prior_version = int(live.get("version", 0)) if isinstance(live, dict) else 0
            new_version = prior_version + 1 if not created else 1

            if live is not None:
                self._archive_live_version(
                    canonical_id=upsert.canonical_id,
                    snapshot=dict(live),
                    version=prior_version,
                    archived_at=now,
                    replaced_by=upsert.provenance,
                )

            # ``Recipe.from_dict`` normalises every list field into the on-disk shape.
            payload_dict: dict[str, Any] = {
                "canonical_id": upsert.canonical_id,
                "version": new_version,
                "created_at": (str(live.get("created_at") or now) if isinstance(live, dict) else now),
                "updated_at": now,
                "model": upsert.model,
                "hardware": upsert.hardware,
                "framework_name": upsert.framework_name,
                "framework_version": upsert.framework_version,
                "precision": upsert.precision,
                "best_config": dict(upsert.best_config or {}),
                "best_throughput": float(upsert.best_throughput),
                # No projection: these rows are stored as the Coordinator wrote them, so a field it adds arrives
                # without a schema here to teach. ``Recipe.from_dict`` rejects an unusable ``extra_envs`` / ``gain_pct``.
                "what_worked": upsert.what_worked,
                "what_failed": upsert.what_failed,
                "remaining_gaps": upsert.remaining_gaps,
                "pitfalls": upsert.pitfalls,
                "lessons": upsert.lessons,
                "last_profiled": upsert.last_profiled
                or (str(live.get("last_profiled") or "") if isinstance(live, dict) else ""),
                "stack_fingerprint": dict(upsert.stack_fingerprint or {}),
                "sessions": upsert.sessions,
                "authority": upsert.authority,
                "confidence": float(upsert.confidence),
                "evidence_refs": list(upsert.evidence_refs or []),
                "provenance": dict(upsert.provenance or {}),
            }
            if upsert.extras:
                # Splat extras at the top level (no nested ``extras`` key on disk).
                for key, val in upsert.extras.items():
                    payload_dict.setdefault(key, val)

            recipe = Recipe.from_dict(payload_dict)
            written = recipe.to_dict()
            atomic_write_json(
                self._live_path(upsert.canonical_id), written, options=WriteOptions(make_parents=True, fsync=True)
            )
            counts = _collection_counts(written)

        return {
            "canonical_id": upsert.canonical_id,
            "version": new_version,
            "created": created,
            "prior_counts": prior_counts,
            "counts": counts,
        }

    # get_recipe
    def get_recipe(self, *, canonical_id: str) -> dict[str, Any] | None:
        """Read the live recipe row."""
        if not canonical_id:
            raise ValueError("get_recipe requires a non-empty canonical_id")
        return _read_json(self._live_path(canonical_id))

    # search
    def search(self, *, label_match: dict[str, Any] | None = None, limit: int = 50) -> list[dict[str, Any]]:
        """Return the live recipes matching ``label_match``, newest first, truncated to ``limit``."""
        clamped_limit = max(1, min(1000, int(limit) if limit else 50))

        rows: list[dict[str, Any]] = []
        for recipe_dir in self._walk_recipe_dirs():
            try:
                cid = canonical_id_for_path(
                    root=self.root,
                    recipe_dir=recipe_dir,
                )
            except InvalidCanonicalIdError as exc:
                log.warning(
                    "skipping malformed recipe dir %s: %s",
                    recipe_dir,
                    exc,
                )
                continue
            payload = _read_json(recipe_dir / RECIPE_FILENAME)
            if not isinstance(payload, dict):
                continue
            # Defensive: stamp the cid if the on-disk payload is missing it.
            payload.setdefault("canonical_id", cid)
            if not _matches_labels(payload, label_match or {}):
                continue
            rows.append(payload)

        # Newest first before truncating, so the rows that survive ``limit`` are the latest; a row without
        # ``updated_at`` sorts last instead of failing the comparison.
        rows.sort(key=lambda r: str(r.get("updated_at") or ""), reverse=True)
        return rows[:clamped_limit]


# search filter helpers
def _matches_labels(payload: dict[str, Any], label_match: dict[str, Any]) -> bool:
    """Key-value match against the top-level identity fields of an arbor-shape recipe."""
    if not label_match:
        return True
    for key, expected in label_match.items():
        if key == "architectures":
            if not _arch_contains(payload.get("architectures"), expected):
                return False
        elif key == "model_type":
            if not _model_type_matches(payload.get("model_type"), expected):
                return False
        elif key == "framework_name":
            # Fall back to the legacy ``framework`` key since search reads raw on-disk JSON without normalizing
            # through ``Recipe.from_dict``.
            actual = payload.get("framework_name") or payload.get("framework")
            if actual != expected:
                return False
        else:
            actual = payload.get(key)
            if actual != expected:
                return False
    return True


def _arch_contains(recipe_arch: Any, query_arch: Any) -> bool:
    """True when the recipe's architectures contain all queried architectures."""
    from hyperloom.inference_optimizer.recipe_snapshot_constants import DEFAULT_ARCHITECTURES_SLUG

    query_slug = _normalize_arch_to_slug(query_arch)
    if not query_slug or query_slug in (DEFAULT_ARCHITECTURES_SLUG, "none"):
        return True
    recipe_slug = _normalize_arch_to_slug(recipe_arch)
    if not recipe_slug or recipe_slug in (DEFAULT_ARCHITECTURES_SLUG, "none"):
        return True
    query_parts = set(query_slug.split("+"))
    recipe_parts = set(recipe_slug.split("+"))
    return query_parts.issubset(recipe_parts)


def _model_type_matches(recipe_mt: Any, query_mt: Any) -> bool:
    """Compare model_type with slug normalization."""
    from hyperloom.inference_optimizer.recipe_snapshot_constants import DEFAULT_MODEL_TYPE_SLUG

    q = str(query_mt or "").strip().lower().replace("/", "_").replace(" ", "_")
    if not q or q in (DEFAULT_MODEL_TYPE_SLUG, "none"):
        return True
    r = str(recipe_mt or "").strip().lower().replace("/", "_").replace(" ", "_")
    if not r or r in (DEFAULT_MODEL_TYPE_SLUG, "none"):
        return True
    return r == q


def _normalize_arch_to_slug(value: Any) -> str:
    """Normalize architectures to a sorted '+'-joined lowercase slug."""
    if isinstance(value, list):
        parts = sorted(
            str(v).strip().lower().replace("/", "_").replace(" ", "_") for v in value if str(v or "").strip()
        )
        return "+".join(parts) if parts else ""
    return str(value or "").strip().lower().replace("/", "_").replace(" ", "_")


def _collection_counts(row: dict[str, Any] | None) -> dict[str, int]:
    """Count the entries in each list-valued knowledge field of a recipe row."""
    out: dict[str, int] = {}
    for key in _COUNTED_COLLECTIONS:
        value = row.get(key) if isinstance(row, dict) else None
        out[key] = len(value) if isinstance(value, list) else 0
    return out


__all__ = [
    "HISTORY_DIRNAME",
    "LOCK_FILENAME",
    "LocalRecipeStore",
    "LocalRecipeStoreError",
    "RECIPE_FILENAME",
]
