# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""On-disk recipe-snapshot store selected by ``KNOWLEDGE_STORE_MODE=local``."""

from __future__ import annotations

import errno
import fcntl
import json
import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from hyperloom.common.io import atomic_write_json
from hyperloom.common.jsonio import read_json
from hyperloom.common.timeutil import now_iso

from .canonical_id import (
    InvalidCanonicalIdError,
    canonical_id_for_path,
    cid_to_path_components,
)
from .schema import Attempt, Recipe


log = logging.getLogger(__name__)


# Filenames + sub-paths within one recipe directory
RECIPE_FILENAME: str = "recipe.json"
HISTORY_DIRNAME: str = "history"
ATTEMPTS_FILENAME: str = "attempts.ndjson"
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


# Order_by whitelist accepted by :meth:`LocalRecipeStore.search`.
_ORDER_BY_KEYS: dict[str, tuple[str, bool]] = {
    "updated_at DESC": ("updated_at", True),
    "updated_at ASC": ("updated_at", False),
    "created_at DESC": ("created_at", True),
    "created_at ASC": ("created_at", False),
    "version DESC": ("version", True),
    "version ASC": ("version", False),
}


# Errors
class LocalRecipeStoreError(RuntimeError):
    """Raised on unrecoverable failures inside the local KB store."""


def _read_json(path: Path) -> dict[str, Any] | None:
    """Read a JSON file or return ``None`` if it doesn't exist."""
    if not path.is_file():
        return None
    try:
        return read_json(path, strict=True, require_dict=True)
    except FileNotFoundError:
        # Race: file disappeared between is_file() and open(); treat as missing.
        return None
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise LocalRecipeStoreError(
            f"failed to read {path}: {exc}",
        ) from exc


def _list_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read every line of an NDJSON file as a JSON dict."""
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                log.warning(
                    "skipping malformed NDJSON row in %s: %s",
                    path,
                    exc,
                )
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


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

    def _attempts_path(self, canonical_id: str) -> Path:
        """Return the path to a cid's append-only attempts log."""
        return self._recipe_dir(canonical_id) / ATTEMPTS_FILENAME

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

    # put_recipe
    def put_recipe(
        self,
        *,
        canonical_id: str,
        # 7-tuple identity (also encoded in canonical_id; stamped at the top
        # level for arbor-compat).
        model: str = "",
        hardware: str = "",
        framework_name: str = "",
        framework_version: str = "",
        precision: str = "",
        # Arbor-aligned payload.
        best_config: dict[str, str] | None = None,
        best_throughput: float = 0.0,
        what_worked: list[Any] | None = None,
        what_failed: list[Any] | None = None,
        remaining_gaps: list[Any] | None = None,
        pitfalls: list[Any] | None = None,
        lessons: list[Any] | None = None,
        last_profiled: str = "",
        stack_fingerprint: dict[str, str] | None = None,
        sessions: list[Any] | None = None,
        # v2 audit / wire-compat fields. provenance is required by the central server so we always stamp something.
        authority: str = "EXPERIENTIAL",
        confidence: float = 0.85,
        evidence_refs: list[Any] | None = None,
        provenance: dict[str, Any] | None = None,
        # Free-form session-level keys preserved verbatim across rewrite.
        extras: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Atomically upsert a recipe row and archive the prior live version."""
        if not canonical_id:
            raise ValueError("put_recipe requires a non-empty canonical_id")
        recipe_dir = self._recipe_dir(canonical_id)
        recipe_dir.mkdir(parents=True, exist_ok=True)
        lock = _CidLock(self._lock_path(canonical_id))
        with lock:
            now = now_iso(timespec="microseconds")
            live = _read_json(self._live_path(canonical_id))
            created = live is None
            prior_counts = _collection_counts(live)
            prior_version = int(live.get("version", 0)) if isinstance(live, dict) else 0
            new_version = prior_version + 1 if not created else 1

            if not created:
                # Archive prior live before overwrite; ``replaced_by`` carries the triggering write's provenance for
                # audit.
                archive_path = self._history_version_path(
                    canonical_id,
                    prior_version,
                )
                try:
                    existing_archive = _read_json(archive_path)
                except LocalRecipeStoreError as exc:
                    log.warning(
                        "put_recipe: unreadable history v%s at %s (%s); rewriting it",
                        prior_version,
                        archive_path,
                        exc,
                    )
                    existing_archive = None
                snapshot = dict(live) if isinstance(live, dict) else {}
                if (
                    isinstance(existing_archive, dict)
                    and isinstance(existing_archive.get("snapshot"), dict)
                    and existing_archive["snapshot"] == snapshot
                ):
                    log.debug(
                        "put_recipe: keeping existing history v%s at %s (crash residue)",
                        prior_version,
                        archive_path,
                    )
                else:
                    archive_payload: dict[str, Any] = {
                        "canonical_id": canonical_id,
                        "version": prior_version,
                        "archived_at": now,
                        "replaced_by": dict(provenance or {}),
                        "snapshot": snapshot,
                    }
                    atomic_write_json(
                        archive_path,
                        archive_payload,
                        indent=2,
                        sort_keys=True,
                        make_parents=True,
                        fsync=True,
                    )

            # Build payload via ``Recipe.from_dict`` so dataclass instances and dicts both round-trip into the same
            # on-disk shape.
            payload_dict: dict[str, Any] = {
                "canonical_id": canonical_id,
                "version": new_version,
                "created_at": (str(live.get("created_at") or now) if isinstance(live, dict) else now),
                "updated_at": now,
                "model": model,
                "hardware": hardware,
                "framework_name": framework_name,
                "framework_version": framework_version,
                "precision": precision,
                "best_config": dict(best_config or {}),
                "best_throughput": float(best_throughput),
                "what_worked": _normalise_str_dicts(what_worked, ("description", "measured_impact")),
                "what_failed": _normalise_str_dicts(what_failed, ("description", "reason")),
                "remaining_gaps": _normalise_str_dicts(remaining_gaps, ("description", "metrics")),
                "pitfalls": _normalise_str_dicts(pitfalls, ("description", "severity")),
                "lessons": _normalise_lessons(lessons),
                "last_profiled": last_profiled
                or (str(live.get("last_profiled") or "") if isinstance(live, dict) else ""),
                "stack_fingerprint": dict(stack_fingerprint or {}),
                "sessions": _normalise_sessions(sessions),
                "authority": authority,
                "confidence": float(confidence),
                "evidence_refs": list(evidence_refs or []),
                "provenance": dict(provenance or {}),
            }
            if extras:
                # Splat extras at the top level (no nested ``extras`` key on disk).
                for key, val in extras.items():
                    payload_dict.setdefault(key, val)

            recipe = Recipe.from_dict(payload_dict)
            written = recipe.to_dict()
            atomic_write_json(
                self._live_path(canonical_id),
                written,
                indent=2,
                sort_keys=True,
                make_parents=True,
                fsync=True,
            )
            counts = _collection_counts(written)

        return {
            "canonical_id": canonical_id,
            "version": new_version,
            "created": created,
            "prior_counts": prior_counts,
            "counts": counts,
        }

    # get_recipe
    def get_recipe(
        self,
        *,
        canonical_id: str,
        version: int | None = None,
    ) -> dict[str, Any] | None:
        """Read live recipe (``version=None``) or an archived version."""
        if not canonical_id:
            raise ValueError("get_recipe requires a non-empty canonical_id")
        if version is None:
            return _read_json(self._live_path(canonical_id))
        # Live version requested explicitly → serve from live.
        live = _read_json(self._live_path(canonical_id))
        if isinstance(live, dict) and int(live.get("version", 0)) == int(version):
            return live
        archive = _read_json(
            self._history_version_path(canonical_id, int(version)),
        )
        if archive is None:
            return None
        # Return the snapshot (the historical Recipe row) so callers see the same shape as a live read.
        snapshot = archive.get("snapshot") if isinstance(archive, dict) else None
        return dict(snapshot) if isinstance(snapshot, dict) else None

    # search
    def search(
        self,
        *,
        label_match: dict[str, Any] | None = None,
        metric_filters: dict[str, Any] | None = None,
        updated_since: str | None = None,
        order_by: str = "updated_at DESC",
        limit: int = 50,
        prefer: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Filter live recipes by labels / metrics / updated_at."""
        del prefer  # client-side rerank lives in RecipeKB
        if order_by not in _ORDER_BY_KEYS:
            raise ValueError(
                f"order_by must be one of {sorted(_ORDER_BY_KEYS)!r}, got {order_by!r}",
            )
        sort_key, descending = _ORDER_BY_KEYS[order_by]
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
            if not _matches_metrics(payload, metric_filters or {}):
                continue
            if not _matches_updated_since(payload, updated_since):
                continue
            rows.append(payload)

        rows.sort(
            key=lambda r: _coerce_sort_value(r.get(sort_key), sort_key),
            reverse=descending,
        )
        return rows[:clamped_limit]

    # Attempts (append-only)
    def append_attempt(
        self,
        *,
        canonical_id: str,
        session_id: str,
        diff: dict[str, Any] | None = None,
        predicted_delta: dict[str, Any] | None = None,
        measured_metrics: dict[str, Any] | None = None,
        fitness: float | None = None,
        outcome: str = "",
        rationale: str = "",
        attempt_at: str | None = None,
    ) -> dict[str, Any]:
        """Append one attempt row."""
        if not canonical_id:
            raise ValueError(
                "append_attempt requires a non-empty canonical_id",
            )
        if not session_id:
            raise ValueError(
                "append_attempt requires a non-empty session_id",
            )
        recipe_dir = self._recipe_dir(canonical_id)
        recipe_dir.mkdir(parents=True, exist_ok=True)
        attempts_path = self._attempts_path(canonical_id)
        # ``id`` is monotonic per cid (existing rows + 1), assigned under the cid lock so concurrent appends don't
        # collide.
        lock = _CidLock(self._lock_path(canonical_id))
        with lock:
            existing = _list_jsonl(attempts_path)
            next_id = len(existing) + 1
            stamped_at = attempt_at or now_iso(timespec="microseconds")
            attempt = Attempt(
                id=next_id,
                recipe_canonical_id=canonical_id,
                session_id=session_id,
                attempt_at=stamped_at,
                diff=dict(diff or {}),
                predicted_delta=dict(predicted_delta or {}),
                measured_metrics=dict(measured_metrics or {}),
                fitness=float(fitness) if fitness is not None else None,
                outcome=str(outcome),
                rationale=str(rationale),
            )
            row = attempt.to_dict()
            line = json.dumps(row, sort_keys=True) + "\n"
            with attempts_path.open("a", encoding="utf-8") as f:
                f.write(line)
                f.flush()
                try:
                    os.fsync(f.fileno())
                except OSError as exc:
                    if exc.errno != errno.EINVAL:
                        log.debug(
                            "fsync skipped on %s: %s",
                            attempts_path,
                            exc,
                        )
        return {
            "id": next_id,
            "recipe_canonical_id": canonical_id,
            "attempt_at": stamped_at,
        }

    def list_attempts(
        self,
        *,
        canonical_id: str,
        session_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return append-only attempts for one Recipe identity."""

        if not canonical_id:
            raise ValueError("list_attempts requires a non-empty canonical_id")
        rows = _list_jsonl(self._attempts_path(canonical_id))
        if session_id is None:
            return rows
        return [row for row in rows if str(row.get("session_id") or "") == str(session_id)]


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


def _matches_metrics(
    payload: dict[str, Any],
    metric_filters: dict[str, Any],
) -> bool:
    """Numeric range filter against arbor-shape metric fields."""
    if not metric_filters:
        return True
    for key, bounds in metric_filters.items():
        # Shorthand alias: ``throughput`` resolves to ``best_throughput``.
        lookup_key = "best_throughput" if key in ("throughput", "best_throughput") else key
        if lookup_key not in payload:
            return False
        try:
            value = float(payload[lookup_key])
        except (TypeError, ValueError):
            return False
        if isinstance(bounds, dict):
            lo = bounds.get("min")
            hi = bounds.get("max")
        else:
            # Scalar shorthand → equality.
            lo = hi = bounds
        if lo is not None:
            try:
                if value < float(lo):
                    return False
            except (TypeError, ValueError):
                return False
        if hi is not None:
            try:
                if value > float(hi):
                    return False
            except (TypeError, ValueError):
                return False
    return True


def _matches_updated_since(
    payload: dict[str, Any],
    updated_since: str | None,
) -> bool:
    """Test whether a row was updated at or after a bound."""
    if not updated_since:
        return True
    raw = payload.get("updated_at") or ""
    return str(raw) >= str(updated_since)


def _coerce_sort_value(value: Any, key: str) -> Any:
    """Stable sort coercion for the order_by whitelist."""
    if key == "version":
        try:
            return int(value or 0)
        except (TypeError, ValueError):
            return 0
    return str(value or "")


# put_recipe input normalisation — accept dataclass OR dict for every list.
def _coerce_dict(item: Any) -> dict[str, Any] | None:
    """Return ``item`` as a dict, or ``None`` when it cannot be coerced."""
    from dataclasses import is_dataclass, asdict

    if item is None:
        return None
    if isinstance(item, dict):
        return item
    if is_dataclass(item):
        return asdict(item)
    if hasattr(item, "to_dict") and callable(item.to_dict):
        out = item.to_dict()
        return out if isinstance(out, dict) else None
    return None


def _collection_counts(row: dict[str, Any] | None) -> dict[str, int]:
    """Count the entries in each list-valued knowledge field of a recipe row."""
    out: dict[str, int] = {}
    for key in _COUNTED_COLLECTIONS:
        value = row.get(key) if isinstance(row, dict) else None
        out[key] = len(value) if isinstance(value, list) else 0
    return out


def _normalise_str_dicts(items: list[Any] | None, keys: tuple[str, ...]) -> list[dict[str, Any]]:
    """Coerce each item to ``{k: str(d.get(k) or "")}`` for *keys*; skip uncoercible entries."""
    out: list[dict[str, Any]] = []
    for it in items or []:
        d = _coerce_dict(it)
        if d is None:
            continue
        out.append({k: str(d.get(k) or "") for k in keys})
    return out


def _normalise_lessons(items: list[Any] | None) -> list[dict[str, Any]]:
    """Coerce lessons into arbor ``{statement, measured_impact}`` dicts."""
    out: list[dict[str, Any]] = []
    for it in items or []:
        d = _coerce_dict(it)
        if d is None:
            continue
        out.append(
            {
                "statement": str(d.get("statement") or ""),
                # Free-form; keep verbatim instead of str()-ing a dict.
                "measured_impact": d.get("measured_impact") or "",
            }
        )
    return out


def _normalise_sessions(items: list[Any] | None) -> list[dict[str, Any]]:
    """Coerce session records into the arbor session dict shape."""
    out: list[dict[str, Any]] = []
    for it in items or []:
        d = _coerce_dict(it)
        if d is None:
            continue
        try:
            tput_before = float(d.get("throughput_before") or 0.0)
        except (TypeError, ValueError):
            tput_before = 0.0
        try:
            tput_after = float(d.get("throughput_after") or 0.0)
        except (TypeError, ValueError):
            tput_after = 0.0
        try:
            gain_pct = float(d.get("gain_pct") or 0.0)
        except (TypeError, ValueError):
            gain_pct = 0.0
        try:
            stack_len = int(d.get("stack_len") or 0)
        except (TypeError, ValueError):
            stack_len = 0
        out.append(
            {
                "date": str(d.get("date") or ""),
                "throughput_before": tput_before,
                "throughput_after": tput_after,
                "actions_taken": list(d.get("actions_taken") or []),
                "session_id": str(d.get("session_id") or ""),
                "gain_pct": gain_pct,
                "stack_len": stack_len,
            }
        )
    return out


__all__ = [
    "ATTEMPTS_FILENAME",
    "HISTORY_DIRNAME",
    "LOCK_FILENAME",
    "LocalRecipeStore",
    "LocalRecipeStoreError",
    "RECIPE_FILENAME",
]
