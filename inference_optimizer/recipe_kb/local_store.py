"""On-disk recipe-snapshot store backing the local-only write path.

Mirrors the wire contract documented in
``primus-cortex-internal/docs/recipe-snapshot-api-reference.md`` so a
caller dispatching reads against either the local store or the central
kb-service sees identical dicts. Every put_recipe is the local
equivalent of the v2 ``PUT /recipes/{cid}`` endpoint:

* prior live row archived to ``history/v{N}.json`` with the incoming
  ``provenance`` recorded in ``replaced_by``;
* new live row written at ``version = N + 1``;
* whole sequence runs under an exclusive file-lock so concurrent
  processes can't tear a write.

Layout (one directory per identity dimension; cid → 5-level path):

::

    <root>/
      <model>/<hardware>/<framework>/<framework_version>/<precision>/
        recipe.json              # current live row
        history/
          v1.json
          v2.json
          ...
        attempts.ndjson          # append-only attempts log
        .lock                    # flock target (separate file so reads
                                 # don't block on the file we're rewriting)

Contracts inherited from Arbor's ``recipes.save_recipe`` proven on
production runs:

* ``fcntl.flock`` (advisory, exclusive) — coordinates writers in the
  same OS namespace; cross-host writes via NFS still need the
  underlying mount to support BSD locking, which both wekafs and
  EFS do.
* tmp + rename — POSIX guarantees rename is atomic on the same
  filesystem, so a concurrent reader either sees the old file or
  the new file, never a half-written one.
* ``os.fsync`` after the rename — best-effort durability; tmpfs /
  some wekafs mounts reject fsync but the rename is still visible.

Anything missing here vs. the central server (audit triggers, GIN
indices on JSONB) is intentional: the local store is the single
source of truth in degraded / offline mode and a "best-effort cache"
in healthy mode. We do not promise SQL-grade analytical queries —
``search`` is a O(N) walk + in-memory filter and that's deliberate
(N is bounded by the number of distinct 5-tuples the optimizer has
ever seen, ~10K is the realistic upper bound).
"""

from __future__ import annotations

import datetime as _dt
import errno
import fcntl
import json
import logging
import os
import shutil
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .canonical_id import (
    InvalidCanonicalIdError,
    canonical_id_for_path,
    cid_to_path_components,
)
from .schema import Attempt, Recipe


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Filenames + sub-paths within one recipe directory
# ---------------------------------------------------------------------------
RECIPE_FILENAME:        str = "recipe.json"
HISTORY_DIRNAME:        str = "history"
ATTEMPTS_FILENAME:      str = "attempts.ndjson"
LOCK_FILENAME:          str = ".lock"
HISTORY_VERSION_PREFIX: str = "v"
HISTORY_VERSION_SUFFIX: str = ".json"


# ---------------------------------------------------------------------------
# Order_by whitelist — mirrors the central /recipes/search contract.
# ---------------------------------------------------------------------------
# Six values total; everything else raises ValueError so a typo'd
# ``order_by`` can't silently emit results in the wrong order.
_ORDER_BY_KEYS: dict[str, tuple[str, bool]] = {
    "updated_at DESC": ("updated_at", True),
    "updated_at ASC":  ("updated_at", False),
    "created_at DESC": ("created_at", True),
    "created_at ASC":  ("created_at", False),
    "version DESC":    ("version",    True),
    "version ASC":     ("version",    False),
}


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
class LocalRecipeStoreError(RuntimeError):
    """Raised on unrecoverable failures inside the local KB store.

    Recoverable cases — missing recipe row, empty history — are
    represented by ``None`` / empty-list returns instead, matching
    the central API's contract for the same situations.
    """


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _utc_now_iso() -> str:
    """ISO-8601 UTC timestamp matching the central server's
    ``created_at`` / ``updated_at`` precision (microsecond + offset)."""
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="microseconds")


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """tmp-file + rename JSON write.

    The tmp file lives in the same directory as ``path`` so the
    rename is atomic on the same filesystem (POSIX guarantee). Any
    other layout would risk a cross-device EXDEV.

    fsync is best-effort: tmpfs and certain wekafs mounts reject the
    syscall, but the rename is already durable on those systems via
    a different path (e.g. journaling). Logging at DEBUG so operators
    aren't spammed by the expected miss on tmpfs CI runners.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_str = tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".tmp",
        dir=str(path.parent),
    )
    tmp = Path(tmp_str)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError as exc:
                log.debug("fsync skipped on %s: %s", tmp, exc)
        os.replace(tmp, path)
    except Exception:
        # Best-effort tmp cleanup so a failed write doesn't leave a
        # ``recipe.json.XXXXX.tmp`` next to the live row.
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


def _read_json(path: Path) -> dict[str, Any] | None:
    """Read a JSON file or return ``None`` if it doesn't exist.

    Other I/O errors (permission, truncated file) propagate as
    :class:`LocalRecipeStoreError` so the caller can decide whether
    to fail the request or fall back to the central read path.
    """
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        # Race: file disappeared between is_file() and open(). Treat
        # as missing — same outcome as if we'd never seen it.
        return None
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise LocalRecipeStoreError(
            f"failed to read {path}: {exc}",
        ) from exc


def _list_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read every line of an NDJSON file as a JSON dict.

    Malformed lines are logged and skipped (matches the dispatcher
    drain behaviour) so a single corrupt row can't take down all
    attempts for a recipe.
    """
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
                    path, exc,
                )
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


# ---------------------------------------------------------------------------
# Per-cid lock
# ---------------------------------------------------------------------------
@dataclass
class _CidLock:
    """Exclusive file-lock for one canonical_id directory.

    Lives on a dedicated ``.lock`` file (NOT ``recipe.json``) so a
    concurrent reader can ``open(recipe.json)`` without contending on
    the lock the writer is currently holding. The reader still gets
    POSIX rename atomicity on the recipe row itself.

    Doubles as a process-local mutex via ``self._mutex`` so two
    threads in the same process don't dead-lock on the same fcntl
    region (BSD/POSIX advisory locks are per-process, not
    per-thread, so we'd otherwise need to serialise this ourselves).
    """

    path: Path
    _mutex: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _fd: int | None = field(default=None, init=False, repr=False)

    def __enter__(self) -> _CidLock:
        self._mutex.acquire()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # ``a+`` so the file is created if missing and the position
        # is at the end (we don't actually write to it; the lock is
        # advisory and the file's contents are irrelevant).
        self._fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(self._fd, fcntl.LOCK_EX)
        except OSError:
            os.close(self._fd)
            self._fd = None
            self._mutex.release()
            raise
        return self

    def __exit__(self, *_exc: Any) -> None:
        if self._fd is not None:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
            finally:
                try:
                    os.close(self._fd)
                except OSError:
                    pass
                self._fd = None
        self._mutex.release()


# ---------------------------------------------------------------------------
# LocalRecipeStore
# ---------------------------------------------------------------------------
@dataclass
class LocalRecipeStore:
    """Filesystem-backed recipe-snapshot store.

    Construction is cheap — no I/O happens until the first
    write/read, so a degraded run that never touches the KB pays
    only the dataclass overhead.

    Args:
        root: store root (typically
            ``$USER_DATA_PATH/recipe_kb/``). Created lazily on first
            write; reads against an absent root return ``None`` /
            empty list.
    """

    root: Path

    def __post_init__(self) -> None:
        self.root = Path(self.root)

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------
    def _recipe_dir(self, canonical_id: str) -> Path:
        components = cid_to_path_components(canonical_id)
        return self.root.joinpath(*components)

    def _live_path(self, canonical_id: str) -> Path:
        return self._recipe_dir(canonical_id) / RECIPE_FILENAME

    def _history_dir(self, canonical_id: str) -> Path:
        return self._recipe_dir(canonical_id) / HISTORY_DIRNAME

    def _history_version_path(self, canonical_id: str, version: int) -> Path:
        return (
            self._history_dir(canonical_id)
            / f"{HISTORY_VERSION_PREFIX}{int(version)}{HISTORY_VERSION_SUFFIX}"
        )

    def _attempts_path(self, canonical_id: str) -> Path:
        return self._recipe_dir(canonical_id) / ATTEMPTS_FILENAME

    def _lock_path(self, canonical_id: str) -> Path:
        return self._recipe_dir(canonical_id) / LOCK_FILENAME

    def _walk_recipe_dirs(self) -> Iterable[Path]:
        """Yield every directory exactly five levels below ``root``
        that contains a live ``recipe.json``.

        Used by :meth:`list_recent` / :meth:`search` — both of which
        only care about live recipe rows. Directories that hold
        attempts but no recipe are intentionally excluded.

        Skips:
        * any malformed depth (operator created an extra subdir or
          put a recipe.json at the wrong level — we log and skip
          rather than indexing it);
        * the ``history`` subdir (six levels deep, the recipe.json
          presence check naturally rules it out).
        """
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
            if len(rel_parts) != 5:
                log.debug(
                    "skipping %s: not at the documented 5-level depth",
                    recipe_dir,
                )
                continue
            yield recipe_dir

    def _walk_cid_dirs(self) -> Iterable[Path]:
        """Yield every directory exactly five levels below ``root``
        that contains EITHER a live ``recipe.json`` OR an
        ``attempts.ndjson``.

        Used by :meth:`list_session_attempts` so attempts written
        against a cid that doesn't (yet) have a parent recipe row
        are still discoverable. Mirrors the central server contract
        that attempts have no FK to the parent recipe.
        """
        if not self.root.is_dir():
            return
        seen: set[Path] = set()
        for filename in (RECIPE_FILENAME, ATTEMPTS_FILENAME):
            for path in self.root.rglob(filename):
                if not path.is_file():
                    continue
                cid_dir = path.parent
                try:
                    rel_parts = cid_dir.relative_to(self.root).parts
                except ValueError:
                    continue
                if len(rel_parts) != 5:
                    continue
                if cid_dir in seen:
                    continue
                seen.add(cid_dir)
                yield cid_dir

    # ------------------------------------------------------------------
    # put_recipe
    # ------------------------------------------------------------------
    def put_recipe(
        self,
        *,
        canonical_id: str,
        # 5-tuple identity (also encoded in canonical_id; stamped at
        # the top level for arbor-compat — arbor's recipe.json has
        # ``model`` / ``hardware`` as top-level fields).
        model: str = "",
        hardware: str = "",
        framework: str = "",
        framework_version: str = "",
        precision: str = "",
        # Arbor-aligned payload. Each list entry can be either an
        # already-shaped dict (the wire representation) or a typed
        # dataclass instance — :meth:`Recipe.from_dict` handles both
        # via the ``payload`` round-trip below.
        best_config: dict[str, str] | None = None,
        best_throughput: float = 0.0,
        what_worked: list[Any] | None = None,
        what_failed: list[Any] | None = None,
        remaining_gaps: list[Any] | None = None,
        prs_tested: list[Any] | None = None,
        pitfalls: list[Any] | None = None,
        lessons: list[Any] | None = None,
        last_profiled: str = "",
        stack_fingerprint: dict[str, str] | None = None,
        sessions: list[Any] | None = None,
        # v2 audit / wire-compat fields (kept so the dispatcher can
        # later push to the central server if we ever re-enable
        # write-through; provenance is REQUIRED by the central server
        # so we always stamp something).
        authority: str = "EXPERIENTIAL",
        confidence: float = 0.85,
        evidence_refs: list[Any] | None = None,
        provenance: dict[str, Any] | None = None,
        # Forward-compat: arbor's existing recipes carry session-level
        # free-form keys (``session_20260515_findings`` etc.); callers
        # can pass them via ``extras`` to avoid losing data on rewrite.
        extras: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Atomically upsert a recipe row in the arbor schema.

        Atomicity is the same as before:

        1. read live ``recipe.json`` (may be missing on first put);
        2. archive prior live to ``history/v{prior_version}.json``,
           stamping ``replaced_by = provenance`` so the archive
           carries the triggering write's audit footprint;
        3. write new live ``recipe.json`` at ``version = prior + 1``
           with refreshed ``updated_at`` (and ``created_at`` carried
           over on update / set to ``now`` on first write).

        Returns ``{"canonical_id", "version", "created"}`` —
        identical to the central server's PUT response shape.

        ``what_worked`` / ``what_failed`` / etc. accept either
        already-shaped dicts (``{"description": ..., "measured_impact":
        ...}``) or arbor dataclass instances; everything is
        normalised through ``Recipe.from_dict`` so the on-disk JSON
        is always the documented arbor shape.
        """
        if not canonical_id:
            raise ValueError("put_recipe requires a non-empty canonical_id")
        recipe_dir = self._recipe_dir(canonical_id)
        recipe_dir.mkdir(parents=True, exist_ok=True)
        lock = _CidLock(self._lock_path(canonical_id))
        with lock:
            now = _utc_now_iso()
            live = _read_json(self._live_path(canonical_id))
            created = live is None
            prior_version = (
                int(live.get("version", 0)) if isinstance(live, dict) else 0
            )
            new_version = prior_version + 1 if not created else 1

            if not created:
                # Archive prior live before overwrite. ``replaced_by``
                # carries the triggering write's provenance so an
                # audit can trace who supplanted v{N-1}.
                archive_path = self._history_version_path(
                    canonical_id, prior_version,
                )
                archive_payload: dict[str, Any] = {
                    "canonical_id": canonical_id,
                    "version":      prior_version,
                    "archived_at":  now,
                    "replaced_by":  dict(provenance or {}),
                    "snapshot":     dict(live) if isinstance(live, dict) else {},
                }
                _atomic_write_json(archive_path, archive_payload)

            # Build payload via ``Recipe.from_dict`` so dataclass
            # instances and dicts both round-trip cleanly (typed
            # callers can pass ``Finding(description=..., measured_impact=...)``
            # or ``{"description": ..., "measured_impact": ...}`` —
            # both end up in the same on-disk shape).
            payload_dict: dict[str, Any] = {
                "canonical_id":      canonical_id,
                "version":           new_version,
                "created_at":        (
                    str(live.get("created_at") or now)
                    if isinstance(live, dict) else now
                ),
                "updated_at":        now,
                "model":             model,
                "hardware":          hardware,
                "framework":         framework,
                "framework_version": framework_version,
                "precision":         precision,
                "best_config":       dict(best_config or {}),
                "best_throughput":   float(best_throughput),
                "what_worked":       _normalise_findings(what_worked),
                "what_failed":       _normalise_failures(what_failed),
                "remaining_gaps":    _normalise_gaps(remaining_gaps),
                "prs_tested":        _normalise_prs(prs_tested),
                "pitfalls":          _normalise_pitfalls(pitfalls),
                "lessons":           _normalise_lessons(lessons),
                "last_profiled":     last_profiled,
                "stack_fingerprint": dict(stack_fingerprint or {}),
                "sessions":          _normalise_sessions(sessions),
                "authority":         authority,
                "confidence":        float(confidence),
                "evidence_refs":     list(evidence_refs or []),
                "provenance":        dict(provenance or {}),
            }
            if extras:
                # Splat extras at the top level so arbor consumers
                # see them where they expect (no nested ``extras``
                # key on disk).
                for key, val in extras.items():
                    payload_dict.setdefault(key, val)

            recipe = Recipe.from_dict(payload_dict)
            _atomic_write_json(
                self._live_path(canonical_id), recipe.to_dict(),
            )

        return {
            "canonical_id": canonical_id,
            "version":      new_version,
            "created":      created,
        }

    # ------------------------------------------------------------------
    # get_recipe / get_history / delete
    # ------------------------------------------------------------------
    def get_recipe(
        self,
        *,
        canonical_id: str,
        version: int | None = None,
    ) -> dict[str, Any] | None:
        """Read live recipe (``version=None``) or an archived version.

        Returns ``None`` for both "canonical_id never existed" and
        "version not in history" — matches the central server's 404
        contract that the dispatcher (Commit 3) maps onto a single
        ``None`` so callers don't have to discriminate.
        """
        if not canonical_id:
            raise ValueError("get_recipe requires a non-empty canonical_id")
        if version is None:
            return _read_json(self._live_path(canonical_id))
        # Live version requested explicitly → serve from live, since
        # history is "everything below the current version".
        live = _read_json(self._live_path(canonical_id))
        if isinstance(live, dict) and int(live.get("version", 0)) == int(version):
            return live
        archive = _read_json(
            self._history_version_path(canonical_id, int(version)),
        )
        if archive is None:
            return None
        # Archive shape is ``{canonical_id, version, archived_at,
        # replaced_by, snapshot}`` — return the snapshot (which IS the
        # historical Recipe row) so callers see the same shape they
        # would for a live read.
        snapshot = archive.get("snapshot") if isinstance(archive, dict) else None
        return dict(snapshot) if isinstance(snapshot, dict) else None

    def get_history(
        self,
        *,
        canonical_id: str,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Return every archived prior version, ascending by version.

        The current (live) row is NOT included — callers fetch that
        via :meth:`get_recipe`. Mirrors the central server's
        ``/history`` contract that returns ``{canonical_id, history:
        [...]}`` with the live row excluded.

        Unknown canonical_id returns ``[]`` (no 404 — matches central
        server behaviour for ``/history``).
        """
        if not canonical_id:
            raise ValueError("get_history requires a non-empty canonical_id")
        history_dir = self._history_dir(canonical_id)
        if not history_dir.is_dir():
            return []
        rows: list[dict[str, Any]] = []
        for entry in sorted(history_dir.iterdir()):
            if not entry.is_file():
                continue
            if not (
                entry.name.startswith(HISTORY_VERSION_PREFIX)
                and entry.name.endswith(HISTORY_VERSION_SUFFIX)
            ):
                continue
            archive = _read_json(entry)
            if isinstance(archive, dict):
                rows.append(archive)
        rows.sort(key=lambda r: int(r.get("version") or 0))
        if limit and len(rows) > int(limit):
            rows = rows[: int(limit)]
        return rows

    def delete_recipe(self, *, canonical_id: str) -> bool:
        """Delete the live row, preserving history.

        Mirrors the central server contract: history rows survive,
        any prior ``GET ?version=N`` still returns the archived
        snapshot. Returns ``True`` iff a live row was actually
        removed; ``False`` when none was present.
        """
        if not canonical_id:
            raise ValueError("delete_recipe requires a non-empty canonical_id")
        live_path = self._live_path(canonical_id)
        lock = _CidLock(self._lock_path(canonical_id))
        with lock:
            try:
                live_path.unlink()
                return True
            except FileNotFoundError:
                return False
            except OSError as exc:
                raise LocalRecipeStoreError(
                    f"failed to delete {live_path}: {exc}",
                ) from exc

    # ------------------------------------------------------------------
    # list_recent / search
    # ------------------------------------------------------------------
    def list_recent(self, *, limit: int = 50) -> list[dict[str, Any]]:
        """Recent live recipes (``updated_at DESC``), no filter.

        Walks the whole store tree — O(N) over distinct cids. Matches
        the central server's ``GET /recipes`` contract; pagination is
        a single ``limit`` because the optimizer uses this only for
        operator dashboards (full search uses :meth:`search`).
        """
        return self.search(order_by="updated_at DESC", limit=int(limit))

    def search(
        self,
        *,
        label_match: dict[str, Any] | None = None,
        metric_filters: dict[str, Any] | None = None,
        updated_since: str | None = None,
        order_by: str = "updated_at DESC",
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Filter live recipes by labels / metrics / updated_at.

        Mirrors the central server's ``POST /recipes/search``:

        * ``label_match``: dict containment — only rows whose
          ``labels`` is a strict superset of every (key, value) pair
          match. Empty/None means no filter.
        * ``metric_filters``: ``{name: {min?, max?}}`` numeric range
          bounds. Rows missing the ``metrics`` key are excluded
          (matches central server semantics — a row without the key
          can't satisfy the bound).
        * ``updated_since``: ISO-8601 string compared lexically (UTC
          ISO-8601 sorts byte-wise the same as chronologically as
          long as the offset is constant, which our ``_utc_now_iso``
          guarantees).
        * ``order_by``: strict whitelist of 6 values, matches the
          server constant. Anything else raises ValueError.
        * ``limit``: ``[1, 1000]`` clamp.
        """
        if order_by not in _ORDER_BY_KEYS:
            raise ValueError(
                f"order_by must be one of {sorted(_ORDER_BY_KEYS)!r}, "
                f"got {order_by!r}",
            )
        sort_key, descending = _ORDER_BY_KEYS[order_by]
        clamped_limit = max(1, min(1000, int(limit) if limit else 50))

        rows: list[dict[str, Any]] = []
        for recipe_dir in self._walk_recipe_dirs():
            try:
                cid = canonical_id_for_path(
                    root=self.root, recipe_dir=recipe_dir,
                )
            except InvalidCanonicalIdError as exc:
                log.warning(
                    "skipping malformed recipe dir %s: %s", recipe_dir, exc,
                )
                continue
            payload = _read_json(recipe_dir / RECIPE_FILENAME)
            if not isinstance(payload, dict):
                continue
            # Defensive: stamp the cid even if the on-disk payload is
            # missing it (older write before Commit 2 might).
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

    # ------------------------------------------------------------------
    # Attempts (append-only)
    # ------------------------------------------------------------------
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
        """Append one attempt row.

        Append-only: never reads / mutates the parent recipe. Matches
        the central server's ``POST /recipes/{cid}/attempts`` —
        attempts are filed even if the parent recipe row doesn't
        exist yet (no FK).

        Returns ``{"id": int, "recipe_canonical_id": str,
        "attempt_at": str}`` mirroring the central response.
        """
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
        # ``id`` is monotonic per cid: count existing rows + 1. We do
        # this under the cid lock so two concurrent appends don't
        # collide on the same id.
        lock = _CidLock(self._lock_path(canonical_id))
        with lock:
            existing = _list_jsonl(attempts_path)
            next_id = len(existing) + 1
            stamped_at = attempt_at or _utc_now_iso()
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
                            "fsync skipped on %s: %s", attempts_path, exc,
                        )
        return {
            "id":                  next_id,
            "recipe_canonical_id": canonical_id,
            "attempt_at":          stamped_at,
        }

    def list_attempts(
        self,
        *,
        canonical_id: str,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """List attempts for one recipe, newest first.

        Mirrors the central server's ``GET /recipes/{cid}/attempts``.
        Empty list for absent canonical_id (no 404 surface).
        """
        if not canonical_id:
            raise ValueError(
                "list_attempts requires a non-empty canonical_id",
            )
        rows = _list_jsonl(self._attempts_path(canonical_id))
        # Newest first — central response is ordered ``attempt_at
        # DESC`` (per spec) but the on-disk file is append-only so
        # iteration order is ascending. Reversing gives the required
        # newest-first contract.
        rows.reverse()
        if limit and len(rows) > int(limit):
            rows = rows[: int(limit)]
        return rows

    def list_session_attempts(
        self,
        *,
        session_id: str,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        """List attempts for one session across all recipes (oldest first).

        Mirrors the central server's
        ``GET /sessions/{session_id}/attempts``. The local store
        achieves the cross-recipe view by walking the tree.
        """
        if not session_id:
            raise ValueError(
                "list_session_attempts requires a non-empty session_id",
            )
        all_rows: list[dict[str, Any]] = []
        for cid_dir in self._walk_cid_dirs():
            attempts_path = cid_dir / ATTEMPTS_FILENAME
            for row in _list_jsonl(attempts_path):
                if str(row.get("session_id") or "") == session_id:
                    all_rows.append(row)
        all_rows.sort(key=lambda r: str(r.get("attempt_at") or ""))
        if limit and len(all_rows) > int(limit):
            all_rows = all_rows[: int(limit)]
        return all_rows

    # ------------------------------------------------------------------
    # Maintenance helpers (used by tests / future cleanup tooling)
    # ------------------------------------------------------------------
    def purge_recipe(self, *, canonical_id: str) -> None:
        """Remove the entire directory tree for one cid (live + history
        + attempts).

        Distinct from :meth:`delete_recipe` which preserves history;
        this is the "obliterate this recipe" escape hatch for tests
        and CLI tooling. Not reachable from the Coordinator hot path.
        """
        if not canonical_id:
            raise ValueError("purge_recipe requires a non-empty canonical_id")
        recipe_dir = self._recipe_dir(canonical_id)
        if recipe_dir.is_dir():
            shutil.rmtree(recipe_dir)


# ---------------------------------------------------------------------------
# search filter helpers
# ---------------------------------------------------------------------------
def _matches_labels(payload: dict[str, Any], label_match: dict[str, Any]) -> bool:
    """Key-value match against the top-level identity fields of an
    arbor-shape recipe.

    Recognised label keys map to top-level fields:

    * ``model`` / ``hardware`` / ``framework`` /
      ``framework_version`` / ``precision`` → the 5-tuple identity
      slots stamped at the top level.

    Any other key is matched against the recipe's free-form
    ``extras`` (preserved arbor session-level keys) so a caller
    that stamps custom labels into ``put_recipe(..., extras={"task":
    "pretrain"})`` can still filter on ``label_match={"task":
    "pretrain"}``.

    Empty filter trivially matches everything.
    """
    if not label_match:
        return True
    for key, expected in label_match.items():
        actual = payload.get(key)
        if actual != expected:
            return False
    return True


def _matches_metrics(
    payload: dict[str, Any], metric_filters: dict[str, Any],
) -> bool:
    """Numeric range filter against arbor-shape metric fields.

    Recognised metric keys:

    * ``best_throughput`` (or shorthand ``throughput``) — read from
      the top-level ``best_throughput`` field;
    * any other key — looked up at the top level (so a caller can
      stamp custom numeric metrics via
      ``put_recipe(..., extras={"mfu": 0.4})``).

    Rows missing the key are excluded (cannot be proven to satisfy
    the bound — same semantics as the central server).
    """
    if not metric_filters:
        return True
    for key, bounds in metric_filters.items():
        # Shorthand alias: ``throughput`` resolves to
        # ``best_throughput`` for arbor-compat (the v2 wire spec
        # uses ``throughput`` as the canonical metric key, so
        # callers might still use that name).
        lookup_key = (
            "best_throughput" if key in ("throughput", "best_throughput")
            else key
        )
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
            # Tolerate ``{"throughput": 10000}`` shorthand → equality.
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
    payload: dict[str, Any], updated_since: str | None,
) -> bool:
    if not updated_since:
        return True
    raw = payload.get("updated_at") or ""
    return str(raw) >= str(updated_since)


def _coerce_sort_value(value: Any, key: str) -> Any:
    """Stable sort coercion for the order_by whitelist.

    ``version`` is integer (defaults to 0 for malformed rows so they
    sink predictably); the timestamp keys stay as strings (ISO-8601
    UTC sorts correctly byte-wise). None / missing always maps to
    "smaller than anything else" so a malformed row falls to the
    bottom of an ASC sort and the top of a DESC sort.
    """
    if key == "version":
        try:
            return int(value or 0)
        except (TypeError, ValueError):
            return 0
    return str(value or "")


# ---------------------------------------------------------------------------
# put_recipe input normalisation — accept dataclass OR dict for every list
# ---------------------------------------------------------------------------
# Each helper coerces a heterogeneous list (None / dataclass / plain dict /
# tuple / scalar) into a list of plain dicts matching the arbor wire shape
# for that field. Errors are propagated only when the input is unambiguously
# malformed (e.g. a string where a Finding was expected). Empty / None
# inputs become empty lists — callers don't have to guard.
def _coerce_dict(item: Any) -> dict[str, Any] | None:
    """Return ``item`` as a dict, or ``None`` when it cannot be coerced.

    Accepts a dataclass instance (``__dict__`` view) or a Mapping.
    Anything else (str / int / None) returns None so the helper-
    specific extractors can decide whether to skip or raise.
    """
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


def _normalise_findings(items: list[Any] | None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for it in (items or []):
        d = _coerce_dict(it)
        if d is None:
            continue
        out.append({
            "description":     str(d.get("description") or ""),
            "measured_impact": str(d.get("measured_impact") or ""),
        })
    return out


def _normalise_failures(items: list[Any] | None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for it in (items or []):
        d = _coerce_dict(it)
        if d is None:
            continue
        out.append({
            "description": str(d.get("description") or ""),
            "reason":      str(d.get("reason") or ""),
        })
    return out


def _normalise_gaps(items: list[Any] | None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for it in (items or []):
        d = _coerce_dict(it)
        if d is None:
            continue
        out.append({
            "description": str(d.get("description") or ""),
            "metrics":     str(d.get("metrics") or ""),
        })
    return out


def _normalise_prs(items: list[Any] | None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for it in (items or []):
        d = _coerce_dict(it)
        if d is None:
            continue
        try:
            number = int(d.get("number") or 0)
        except (TypeError, ValueError):
            number = 0
        out.append({
            "repo":    str(d.get("repo") or ""),
            "number":  number,
            "outcome": str(d.get("outcome") or ""),
            "notes":   str(d.get("notes") or ""),
        })
    return out


def _normalise_pitfalls(items: list[Any] | None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for it in (items or []):
        d = _coerce_dict(it)
        if d is None:
            continue
        out.append({
            "description": str(d.get("description") or ""),
            "severity":    str(d.get("severity") or ""),
        })
    return out


def _normalise_lessons(items: list[Any] | None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for it in (items or []):
        d = _coerce_dict(it)
        if d is None:
            continue
        out.append({
            "statement":       str(d.get("statement") or ""),
            # Free-form (Coordinator writes a structured dict) — keep
            # verbatim instead of str()-ing a dict into a lossy string.
            "measured_impact": d.get("measured_impact") or "",
        })
    return out


def _normalise_sessions(items: list[Any] | None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for it in (items or []):
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
        out.append({
            "date":              str(d.get("date") or ""),
            "throughput_before": tput_before,
            "throughput_after":  tput_after,
            "actions_taken":     list(d.get("actions_taken") or []),
            "session_id":        str(d.get("session_id") or ""),
            "gain_pct":          gain_pct,
            "stack_len":         stack_len,
        })
    return out


__all__ = [
    "ATTEMPTS_FILENAME",
    "HISTORY_DIRNAME",
    "LOCK_FILENAME",
    "LocalRecipeStore",
    "LocalRecipeStoreError",
    "RECIPE_FILENAME",
]
