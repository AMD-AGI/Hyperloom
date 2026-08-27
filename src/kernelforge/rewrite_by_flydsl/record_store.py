"""One backend-agnostic Rewrite record layout, on KB Store or on disk.

A rewrite candidate is a record under a producer-owned ``kernel:`` identity: a
knowledge document describing the port and the ported file itself, kept out of
the document as byte-exact artifact data. Both backends store exactly that, so a
run can move between them without the reader learning a second shape. Reads
first rank metadata, then materialize only the selected candidates as isolated
bundles::

    <destination>/<session-id>/recipe.json
    <destination>/<session-id>/files/**

The identity's ``producer`` owns an independent candidate index and champion;
``backend`` describes the final implementation type. The canonical id carries
both, so the existing ranking and pointer policy needs no producer special case.
The KB Store must accept that producer dimension in its canonical schema; until
it does, remote producer-owned identities remain a live deployment blocker.

The champion is a pointer, not a filter. A correct port that loses to the
source baseline is still the only thing that saves the next run from redoing
PORT, so candidates are recorded whether or not they win; only the pointer is
gated on speedup.

A record's ``speedup`` is what its producer claims, which is not evidence for
any other run: a claim that no consumer reproduced can be arbitrarily inflated
and would otherwise win the ranking forever. ``measured_speedup`` is the value
a consumer measured after actually applying the record, so ranking puts every
measured candidate ahead of every merely claimed one and a consumer amends the
record it measured.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import tempfile
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any, Iterator, Mapping, Protocol

try:
    import fcntl
except ImportError:  # pragma: no cover - exercised only on non-POSIX hosts
    fcntl = None  # type: ignore[assignment]

from kernelforge.knowledge.remote_exp.kb_store_client import KBStoreClient, KBStoreError
from kernelforge.durable_io import fsync_directory

CHAMPION_METRIC = "speedup"
MEASURED_SPEEDUP_KEY = "measured_speedup"
ARTIFACT_KIND = "rewrite"
KNOWLEDGE_FILENAME = "knowledge.json"
CHAMPION_FILENAME = "champion.json"
RECIPE_FILENAME = "recipe.json"
LOCK_FILENAME = ".lock"

_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SEGMENT_RE = re.compile(r"^[a-z0-9_][a-z0-9._+-]*$")
_PROCESS_LOCKS: dict[str, threading.RLock] = {}
_PROCESS_LOCKS_GUARD = threading.Lock()


class RewriteRecordError(RuntimeError):
    """The record layout was violated by a caller or by stored data."""


@dataclass(frozen=True)
class RewriteCandidate:
    """A recorded port, ranked on measured evidence before a bare claim.

    ``speedup`` is what the record's own document claims. ``measured_speedup``
    is present only once a consumer applied this record and measured it, and it
    is the value ranking trusts.
    """

    session_id: str
    knowledge: dict[str, Any]
    speedup: float | None
    is_champion: bool
    envelope: dict[str, Any] | None = None
    measured_speedup: float | None = None

    @property
    def ranked_speedup(self) -> float | None:
        """The speedup this candidate is ranked on, evidence first."""
        return self.speedup if self.measured_speedup is None else self.measured_speedup


class RewriteRecordStore(Protocol):
    """Read and write rewrite candidates under a canonical identity."""

    @property
    def configured(self) -> bool:
        raise NotImplementedError

    def candidates(self, canonical_id: str, *, limit: int) -> list[RewriteCandidate]:
        raise NotImplementedError

    def materialize(
        self,
        canonical_id: str,
        candidate: RewriteCandidate,
        destination: str | Path,
    ) -> Path:
        raise NotImplementedError

    def read_bytes(self, canonical_id: str, session_id: str, rel_path: str) -> bytes:
        """Return byte-exact artifact data, or ``b""`` when it is absent."""
        raise NotImplementedError

    def write(
        self,
        canonical_id: str,
        session_id: str,
        knowledge: Mapping[str, Any],
        files: Mapping[str, Path],
    ) -> None:
        raise NotImplementedError

    def record_measured_speedup(
        self,
        canonical_id: str,
        session_id: str,
        measured_speedup: float,
    ) -> None:
        """Amend one recorded candidate with a speedup a consumer measured."""
        raise NotImplementedError

    def champion_speedup(self, canonical_id: str) -> float | None:
        raise NotImplementedError

    def promote(self, canonical_id: str, session_id: str, speedup: float) -> None:
        raise NotImplementedError


def finite_speedup(value: Any) -> float | None:
    """Coerce a recorded speedup, treating anything unusable as absent."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        return None
    return number


def _checked_measured_speedup(value: Any) -> float:
    """Reject a measurement that cannot stand as evidence for a candidate."""
    measured = finite_speedup(value)
    if measured is None:
        raise RewriteRecordError(f"unusable measured speedup: {value!r}")
    return measured


def _with_preserved_measurement(
    knowledge: Mapping[str, Any],
    *,
    recorded: Any,
) -> dict[str, Any]:
    """Carry a consumer's measurement across a replacing write of one record.

    A producer writes its own claim; a consumer that measured the candidate
    amends the same record with what it actually got, and the ranking then trusts
    the measurement over the claim. Replacing the record would throw that away
    and hand the ranking back the claim that lost, so the measured value is
    carried over unless this write supplies one of its own.

    Ownership stays with the measurer: an unusable recorded value is dropped
    rather than propagated, because a claim is the one thing a record always has
    and a measurement is only worth keeping while it is still a measurement.
    """
    payload = dict(knowledge)
    if payload.get(MEASURED_SPEEDUP_KEY) is not None:
        return payload
    measured = finite_speedup(recorded)
    if measured is not None:
        payload[MEASURED_SPEEDUP_KEY] = measured
    return payload


def validate_session_id(session_id: str) -> str:
    """Reject ids that cannot be both a URL segment and a directory name."""
    raw = str(session_id or "").strip()
    if not _SESSION_ID_RE.fullmatch(raw):
        raise RewriteRecordError(f"unusable session id: {session_id!r}")
    return raw


def safe_rel_path(rel_path: str) -> str:
    """Reject artifact paths that could escape the record's files directory."""
    if not isinstance(rel_path, str):
        raise RewriteRecordError(f"unsafe artifact path: {rel_path!r}")
    raw = rel_path
    parts = raw.split("/")
    if (
        not raw
        or "\0" in raw
        or "\\" in raw
        or raw.startswith("/")
        or PureWindowsPath(raw).drive
        or any(part in {"", "..", "."} for part in parts)
    ):
        raise RewriteRecordError(f"unsafe artifact path: {rel_path!r}")
    return "/".join(parts)


def canonical_relpath(canonical_id: str) -> Path:
    """Render an identity as nested directories, scheme first."""
    parts = str(canonical_id or "").split(":")
    if len(parts) < 2 or any(not _SEGMENT_RE.fullmatch(part) for part in parts):
        raise RewriteRecordError(f"unusable canonical id: {canonical_id!r}")
    return Path(*parts)


def _ranking_key(candidate: RewriteCandidate) -> tuple[int, float, str]:
    """Order measured candidates first, then by value, then by identity.

    A claim no consumer reproduced ranks below every measured candidate however
    large it is; the session id makes the order total so two runs reading the
    same records select the same candidates.
    """
    return (
        1 if candidate.measured_speedup is None else 0,
        -(candidate.ranked_speedup or 0.0),
        candidate.session_id,
    )


def _rank(candidates: list[RewriteCandidate], limit: int) -> list[RewriteCandidate]:
    ordered = sorted(candidates, key=_ranking_key)
    return ordered[: max(0, int(limit))]


def _knowledge_of(document: Any) -> dict[str, Any] | None:
    if not isinstance(document, Mapping):
        return None
    knowledge = document.get("knowledge")
    return dict(knowledge) if isinstance(knowledge, Mapping) else None


def _checked_destination(destination: str | Path) -> Path:
    root = Path(destination)
    if root.is_symlink():
        raise RewriteRecordError(f"destination may not be a symlink: {root}")
    if root.exists() and not root.is_dir():
        raise RewriteRecordError(f"destination is not a directory: {root}")
    root.mkdir(parents=True, exist_ok=True)
    return root


def _bundle_staging(destination: str | Path, session_id: str) -> tuple[Path, Path]:
    root = _checked_destination(destination)
    safe_session_id = validate_session_id(session_id)
    bundle = root / safe_session_id
    if bundle.is_symlink():
        raise RewriteRecordError(f"candidate bundle may not be a symlink: {bundle}")
    staging = Path(tempfile.mkdtemp(prefix=f".{safe_session_id}-", dir=root))
    return bundle, staging


def _safe_files(root: Path) -> set[str]:
    """Validate a files tree and return all regular-file relative paths."""
    if root.is_symlink():
        raise RewriteRecordError(f"files directory may not be a symlink: {root}")
    if not root.exists():
        root.mkdir(parents=True)
        return set()
    if not root.is_dir():
        raise RewriteRecordError(f"files path is not a directory: {root}")

    found: set[str] = set()
    for current, directories, filenames in os.walk(root, followlinks=False):
        current_path = Path(current)
        for name in directories:
            path = current_path / name
            if path.is_symlink():
                raise RewriteRecordError(f"artifact directory may not be a symlink: {path}")
        for name in filenames:
            path = current_path / name
            if path.is_symlink() or not path.is_file():
                raise RewriteRecordError(f"artifact must be a regular file: {path}")
            found.add(safe_rel_path(path.relative_to(root).as_posix()))
    return found


def _recipe(
    canonical_id: str,
    candidate: RewriteCandidate,
    service_fields: Mapping[str, Any],
) -> dict[str, Any]:
    """Merge opaque knowledge under authoritative service-owned fields."""
    recipe = dict(candidate.knowledge)
    recipe.update(service_fields)
    recipe.update(
        {
            "canonical_id": canonical_id,
            "session_id": candidate.session_id,
            "is_champion": candidate.is_champion,
            "champion": candidate.is_champion,
        }
    )
    recipe.setdefault("speedup", candidate.speedup)
    return recipe


def _write_recipe(path: Path, recipe: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(dict(recipe), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _process_lock(path: Path) -> threading.RLock:
    key = str(path.resolve(strict=False))
    with _PROCESS_LOCKS_GUARD:
        return _PROCESS_LOCKS.setdefault(key, threading.RLock())


@contextmanager
def _identity_file_lock(path: Path, *, exclusive: bool) -> Iterator[None]:
    """Hold one POSIX advisory lock for an identity."""
    if fcntl is None:
        raise RewriteRecordError("local rewrite records require POSIX fcntl file locking")
    flags = os.O_RDWR | os.O_CREAT
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as error:
        raise RewriteRecordError(f"could not open rewrite identity lock: {path}") from error
    try:
        mode = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        fcntl.flock(descriptor, mode)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _write_bytes_synced(path: Path, content: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def _write_json_synced(path: Path, document: Mapping[str, Any]) -> None:
    content = json.dumps(
        dict(document),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ).encode("utf-8")
    _write_bytes_synced(path, content)


def _copy_file_synced(source: Path, target: Path) -> None:
    if source.is_symlink() or not source.is_file():
        raise RewriteRecordError(f"artifact source must be a regular file: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as reader, target.open("xb") as writer:
        shutil.copyfileobj(reader, writer)
        writer.flush()
        os.fsync(writer.fileno())


def _fsync_tree_directories(root: Path) -> None:
    directories = [path for path in root.rglob("*") if path.is_dir()]
    for directory in sorted(directories, key=lambda path: len(path.parts), reverse=True):
        fsync_directory(directory)
    fsync_directory(root)


def _replace_directory(staging: Path, destination: Path) -> None:
    """Atomically install staging, restoring the prior directory on failure."""
    parent = destination.parent
    backup: Path | None = None
    displaced: Path | None = None
    try:
        if destination.exists():
            if destination.is_symlink() or not destination.is_dir():
                raise RewriteRecordError(f"existing rewrite session is not a safe directory: {destination}")
            backup = parent / f".{destination.name}.backup-{uuid.uuid4().hex}"
            os.replace(destination, backup)
            fsync_directory(parent)
        os.replace(staging, destination)
        fsync_directory(parent)
    except Exception:
        if backup is not None and backup.exists():
            if destination.exists():
                displaced = parent / f".{destination.name}.failed-{uuid.uuid4().hex}"
                os.replace(destination, displaced)
            os.replace(backup, destination)
            fsync_directory(parent)
        if displaced is not None:
            shutil.rmtree(displaced, ignore_errors=True)
        raise
    if backup is not None:
        shutil.rmtree(backup)
        fsync_directory(parent)


def _commit_bundle(bundle: Path, staging: Path) -> Path:
    if bundle.exists():
        if bundle.is_symlink():
            raise RewriteRecordError(f"candidate bundle may not be a symlink: {bundle}")
        if bundle.is_dir():
            shutil.rmtree(bundle)
        else:
            bundle.unlink()
    staging.replace(bundle)
    return bundle


def _validate_session_envelope(
    envelope: Mapping[str, Any],
    canonical_id: str,
    session_id: str,
) -> None:
    recorded_canonical = str(envelope.get("canonical_id") or "")
    recorded_session = str(envelope.get("session_id") or "")
    if recorded_canonical != canonical_id:
        raise RewriteRecordError(f"session canonical id mismatch: {recorded_canonical!r} != {canonical_id!r}")
    if recorded_session != session_id:
        raise RewriteRecordError(f"session id mismatch: {recorded_session!r} != {session_id!r}")


class KBStoreRewriteRecords:
    """Rewrite records held by the KB Store service."""

    def __init__(self, client: KBStoreClient) -> None:
        self._client = client
        self._download_lock = threading.RLock()

    @property
    def configured(self) -> bool:
        return True

    def candidates(self, canonical_id: str, *, limit: int) -> list[RewriteCandidate]:
        requested = max(0, int(limit))
        if requested == 0:
            return []
        raw_sessions: list[Any] = []
        offset = 0
        while len(raw_sessions) < requested:
            page_limit = min(100, requested - len(raw_sessions))
            ranked = self._client.get_top_sessions(
                canonical_id,
                metric=CHAMPION_METRIC,
                limit=page_limit,
                offset=offset,
            )
            page = ranked.get("sessions")
            if not isinstance(page, list):
                raise RewriteRecordError("ranked session response has no sessions list")
            raw_sessions.extend(page)
            if len(page) < page_limit:
                break
            offset += len(page)
        found: list[RewriteCandidate] = []
        seen: set[str] = set()
        for item in raw_sessions:
            if not isinstance(item, Mapping):
                raise RewriteRecordError("ranked session entry is not an object")
            session_id = validate_session_id(str(item.get("session_id") or ""))
            if session_id in seen:
                raise RewriteRecordError(f"duplicate ranked session id: {session_id}")
            seen.add(session_id)
            envelope = self._client.get_session(canonical_id, session_id)
            if envelope is None:
                continue
            _validate_session_envelope(envelope, canonical_id, session_id)
            knowledge = _knowledge_of(envelope)
            if knowledge is None:
                continue
            found.append(
                RewriteCandidate(
                    session_id=session_id,
                    knowledge=knowledge,
                    speedup=finite_speedup(knowledge.get("speedup")),
                    is_champion=item.get("is_champion") is True,
                    envelope=dict(envelope),
                    measured_speedup=finite_speedup(knowledge.get(MEASURED_SPEEDUP_KEY)),
                )
            )
        return _rank(found, limit)

    def materialize(
        self,
        canonical_id: str,
        candidate: RewriteCandidate,
        destination: str | Path,
    ) -> Path:
        """Download one complete remote session into an isolated recipe bundle."""
        bundle, staging = _bundle_staging(destination, candidate.session_id)
        try:
            envelope = candidate.envelope
            if envelope is None:
                loaded = self._client.get_session(canonical_id, candidate.session_id)
                if loaded is None:
                    raise RewriteRecordError("candidate session disappeared before download")
                envelope = dict(loaded)
            _validate_session_envelope(envelope, canonical_id, candidate.session_id)
            knowledge = _knowledge_of(envelope)
            if knowledge is None:
                raise RewriteRecordError("candidate session has no knowledge document")

            with self._download_lock:
                listing = self._client.list_session_files(canonical_id, candidate.session_id)
                if not isinstance(listing, Mapping):
                    raise RewriteRecordError("session file manifest is not an object")
                raw_files = listing.get("files") or []
                if not isinstance(raw_files, list):
                    raise RewriteRecordError("session file manifest files is not a list")
                expected: set[str] = set()
                for item in raw_files:
                    if not isinstance(item, Mapping):
                        raise RewriteRecordError("session file manifest contains a non-object entry")
                    rel_path = safe_rel_path(item.get("path"))
                    if rel_path in expected:
                        raise RewriteRecordError(f"duplicate session artifact path: {rel_path}")
                    expected.add(rel_path)

                # The upstream SDK lists internally. Pin that call to the
                # validated snapshot so the download neither repeats the
                # request nor observes a different set of paths.
                original_listing = self._client.list_session_files

                def validated_listing(
                    requested_canonical_id: str,
                    requested_session_id: str,
                    *,
                    kind: str = "",
                ) -> dict[str, Any]:
                    if (
                        requested_canonical_id == canonical_id
                        and requested_session_id == candidate.session_id
                        and not kind
                    ):
                        return dict(listing)
                    return original_listing(
                        requested_canonical_id,
                        requested_session_id,
                        kind=kind,
                    )

                self._client.list_session_files = validated_listing  # type: ignore[method-assign]
                try:
                    self._client.download_session(
                        canonical_id,
                        candidate.session_id,
                        staging,
                        include_values=False,
                    )
                finally:
                    self._client.list_session_files = original_listing  # type: ignore[method-assign]
            actual = _safe_files(staging / "files")
            if actual != expected:
                raise RewriteRecordError(
                    f"downloaded session files differ from manifest: {sorted(actual)!r} != {sorted(expected)!r}"
                )
            for generated in list(staging.iterdir()):
                if generated.name == "files":
                    continue
                if generated.is_dir() and not generated.is_symlink():
                    shutil.rmtree(generated)
                else:
                    generated.unlink()
            service_fields = {key: value for key, value in envelope.items() if key != "knowledge"}
            materialized = RewriteCandidate(
                session_id=candidate.session_id,
                knowledge=knowledge,
                speedup=candidate.speedup,
                is_champion=candidate.is_champion,
                envelope=dict(envelope),
                measured_speedup=candidate.measured_speedup,
            )
            _write_recipe(
                staging / RECIPE_FILENAME,
                _recipe(canonical_id, materialized, service_fields),
            )
            return _commit_bundle(bundle, staging)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise

    def read_bytes(self, canonical_id: str, session_id: str, rel_path: str) -> bytes:
        rel = safe_rel_path(rel_path)
        with tempfile.TemporaryDirectory(prefix="rewrite-read-") as temporary:
            destination = Path(temporary)
            self._client.download_session(canonical_id, session_id, destination, include_values=False)
            path = destination / "files" / rel
            if not path.is_file() or path.is_symlink():
                return b""
            return path.read_bytes()

    def write(
        self,
        canonical_id: str,
        session_id: str,
        knowledge: Mapping[str, Any],
        files: Mapping[str, Path],
    ) -> None:
        for rel_path, source in files.items():
            self._client.put_file(
                canonical_id,
                session_id,
                safe_rel_path(rel_path),
                source,
                kind=ARTIFACT_KIND,
                meta={"schema": "kernelforge-rewrite-v1"},
            )
        existing = _knowledge_of(self._client.get_session(canonical_id, session_id)) or {}
        self._client.put_knowledge(
            canonical_id,
            _with_preserved_measurement(knowledge, recorded=existing.get(MEASURED_SPEEDUP_KEY)),
            session_id=session_id,
            mode="replace",
        )

    def record_measured_speedup(
        self,
        canonical_id: str,
        session_id: str,
        measured_speedup: float,
    ) -> None:
        """Merge the measured value into the candidate's own session document.

        Merge mode amends the record the producer wrote instead of rewriting it,
        so the claim, the artifacts and the opaque payload all survive.
        """
        self._client.put_knowledge(
            canonical_id,
            {MEASURED_SPEEDUP_KEY: _checked_measured_speedup(measured_speedup)},
            session_id=validate_session_id(session_id),
            mode="merge",
        )

    def champion_speedup(self, canonical_id: str) -> float | None:
        rollup = self._client.get_rollup(canonical_id) or {}
        champion = rollup.get("champion") or {}
        if not isinstance(champion, Mapping):
            return None
        if str(champion.get("metric") or "") != CHAMPION_METRIC:
            return None
        return finite_speedup(champion.get("value"))

    def promote(self, canonical_id: str, session_id: str, speedup: float) -> None:
        self._client.set_champion(canonical_id, session_id, metric=CHAMPION_METRIC, value=speedup)


class LocalRewriteRecords:
    """Rewrite records held on disk in the same shape the service uses."""

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root).expanduser()

    @property
    def configured(self) -> bool:
        return True

    def _identity_dir(self, canonical_id: str) -> Path:
        return self._root / canonical_relpath(canonical_id)

    def _session_dir(self, canonical_id: str, session_id: str) -> Path:
        return self._identity_dir(canonical_id) / "sessions" / validate_session_id(session_id)

    @contextmanager
    def _identity_lock(self, canonical_id: str, *, exclusive: bool) -> Iterator[None]:
        identity_dir = self._identity_dir(canonical_id)
        identity_dir.mkdir(parents=True, exist_ok=True)
        if identity_dir.is_symlink() or not identity_dir.is_dir():
            raise RewriteRecordError(f"rewrite identity is not a safe directory: {identity_dir}")
        lock_path = identity_dir / LOCK_FILENAME
        with _process_lock(lock_path):
            with _identity_file_lock(lock_path, exclusive=exclusive):
                yield

    def _champion_unlocked(self, canonical_id: str) -> dict[str, Any]:
        path = self._identity_dir(canonical_id) / CHAMPION_FILENAME
        if not path.is_file() or path.is_symlink():
            return {}
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return loaded if isinstance(loaded, dict) else {}

    def candidates(self, canonical_id: str, *, limit: int) -> list[RewriteCandidate]:
        with self._identity_lock(canonical_id, exclusive=False):
            sessions_dir = self._identity_dir(canonical_id) / "sessions"
            if not sessions_dir.is_dir() or sessions_dir.is_symlink():
                return []
            champion_id = str(self._champion_unlocked(canonical_id).get("session_id") or "")
            entries = [path for path in sessions_dir.iterdir() if path.is_dir() and not path.is_symlink()]
            found: list[RewriteCandidate] = []
            for entry in entries:
                validate_session_id(entry.name)
                document = entry / KNOWLEDGE_FILENAME
                if not document.is_file() or document.is_symlink():
                    continue
                try:
                    knowledge = json.loads(document.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                if not isinstance(knowledge, dict):
                    continue
                found.append(
                    RewriteCandidate(
                        session_id=entry.name,
                        knowledge=knowledge,
                        speedup=finite_speedup(knowledge.get("speedup")),
                        is_champion=entry.name == champion_id,
                        measured_speedup=finite_speedup(knowledge.get(MEASURED_SPEEDUP_KEY)),
                    )
                )
            return _rank(found, limit)

    def materialize(
        self,
        canonical_id: str,
        candidate: RewriteCandidate,
        destination: str | Path,
    ) -> Path:
        """Copy one complete local session into the standard recipe bundle."""
        with self._identity_lock(canonical_id, exclusive=False):
            source = self._session_dir(canonical_id, candidate.session_id)
            if source.is_symlink() or not source.is_dir():
                raise RewriteRecordError(f"candidate session is not a safe directory: {source}")
            document = source / KNOWLEDGE_FILENAME
            if document.is_symlink() or not document.is_file():
                raise RewriteRecordError(f"candidate knowledge is not a regular file: {document}")
            try:
                knowledge = json.loads(document.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise RewriteRecordError(f"candidate knowledge is unreadable: {document}") from error
            if not isinstance(knowledge, dict):
                raise RewriteRecordError("candidate knowledge is not an object")

            bundle, staging = _bundle_staging(destination, candidate.session_id)
            try:
                source_files = source / "files"
                rel_paths = _safe_files(source_files)
                target_files = staging / "files"
                target_files.mkdir()
                for rel_path in sorted(rel_paths):
                    target = target_files / rel_path
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source_files / rel_path, target)
                materialized = RewriteCandidate(
                    session_id=candidate.session_id,
                    knowledge=knowledge,
                    speedup=finite_speedup(knowledge.get("speedup")),
                    is_champion=candidate.is_champion,
                    measured_speedup=finite_speedup(knowledge.get(MEASURED_SPEEDUP_KEY)),
                )
                _write_recipe(
                    staging / RECIPE_FILENAME,
                    _recipe(canonical_id, materialized, {}),
                )
                return _commit_bundle(bundle, staging)
            except Exception:
                shutil.rmtree(staging, ignore_errors=True)
                raise

    def read_bytes(self, canonical_id: str, session_id: str, rel_path: str) -> bytes:
        with self._identity_lock(canonical_id, exclusive=False):
            path = self._session_dir(canonical_id, session_id) / "files" / safe_rel_path(rel_path)
            if not path.is_file() or path.is_symlink():
                return b""
            return path.read_bytes()

    @staticmethod
    def _recorded_measurement(session_dir: Path) -> Any:
        """The measured value already on this record, or None when there is none.

        A first write has no record to read, so absence is the ordinary case and
        never an error. A record that exists but cannot be parsed is treated the
        same way: the replacing write is what repairs it, and refusing to write
        would leave the unreadable document in place.
        """
        document = session_dir / KNOWLEDGE_FILENAME
        if document.is_symlink() or not document.is_file():
            return None
        try:
            knowledge = json.loads(document.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(knowledge, dict):
            return None
        return knowledge.get(MEASURED_SPEEDUP_KEY)

    def write(
        self,
        canonical_id: str,
        session_id: str,
        knowledge: Mapping[str, Any],
        files: Mapping[str, Path],
    ) -> None:
        safe_session_id = validate_session_id(session_id)
        normalized_files = {safe_rel_path(rel_path): Path(source) for rel_path, source in files.items()}
        if len(normalized_files) != len(files):
            raise RewriteRecordError("duplicate normalized artifact path")
        with self._identity_lock(canonical_id, exclusive=True):
            identity_dir = self._identity_dir(canonical_id)
            sessions_dir = identity_dir / "sessions"
            sessions_dir.mkdir(parents=True, exist_ok=True)
            if sessions_dir.is_symlink() or not sessions_dir.is_dir():
                raise RewriteRecordError(f"rewrite sessions path is not a safe directory: {sessions_dir}")
            fsync_directory(identity_dir)
            session_dir = sessions_dir / safe_session_id
            payload = _with_preserved_measurement(
                knowledge,
                recorded=self._recorded_measurement(session_dir),
            )
            staging = Path(tempfile.mkdtemp(prefix=f".{safe_session_id}.staging-", dir=sessions_dir))
            try:
                files_root = staging / "files"
                files_root.mkdir()
                for rel_path, source in normalized_files.items():
                    _copy_file_synced(source, files_root / rel_path)
                _write_json_synced(staging / KNOWLEDGE_FILENAME, payload)
                if _safe_files(files_root) != set(normalized_files):
                    raise RewriteRecordError("staged rewrite artifacts failed validation")
                loaded = json.loads((staging / KNOWLEDGE_FILENAME).read_text(encoding="utf-8"))
                if loaded != payload:
                    raise RewriteRecordError("staged rewrite knowledge failed validation")
                _fsync_tree_directories(staging)
                _replace_directory(staging, session_dir)
            finally:
                if staging.exists():
                    shutil.rmtree(staging, ignore_errors=True)

    def record_measured_speedup(
        self,
        canonical_id: str,
        session_id: str,
        measured_speedup: float,
    ) -> None:
        """Amend one session document in place, keeping every other field."""
        measured = _checked_measured_speedup(measured_speedup)
        with self._identity_lock(canonical_id, exclusive=True):
            session_dir = self._session_dir(canonical_id, session_id)
            document_path = session_dir / KNOWLEDGE_FILENAME
            if document_path.is_symlink() or not document_path.is_file():
                raise RewriteRecordError(f"candidate knowledge is not a regular file: {document_path}")
            try:
                knowledge = json.loads(document_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise RewriteRecordError(f"candidate knowledge is unreadable: {document_path}") from error
            if not isinstance(knowledge, dict):
                raise RewriteRecordError("candidate knowledge is not an object")
            knowledge[MEASURED_SPEEDUP_KEY] = measured
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{KNOWLEDGE_FILENAME}.",
                dir=session_dir,
            )
            os.close(descriptor)
            temporary = Path(temporary_name)
            temporary.unlink()
            try:
                _write_json_synced(temporary, knowledge)
                os.replace(temporary, document_path)
                fsync_directory(session_dir)
            finally:
                temporary.unlink(missing_ok=True)

    def champion_speedup(self, canonical_id: str) -> float | None:
        with self._identity_lock(canonical_id, exclusive=False):
            champion = self._champion_unlocked(canonical_id)
            if str(champion.get("metric") or "") != CHAMPION_METRIC:
                return None
            return finite_speedup(champion.get("value"))

    def promote(self, canonical_id: str, session_id: str, speedup: float) -> None:
        document = {
            "session_id": validate_session_id(session_id),
            "metric": CHAMPION_METRIC,
            "value": float(speedup),
        }
        with self._identity_lock(canonical_id, exclusive=True):
            identity_dir = self._identity_dir(canonical_id)
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{CHAMPION_FILENAME}.",
                dir=identity_dir,
            )
            os.close(descriptor)
            temporary = Path(temporary_name)
            temporary.unlink()
            try:
                _write_json_synced(temporary, document)
                os.replace(temporary, identity_dir / CHAMPION_FILENAME)
                fsync_directory(identity_dir)
            finally:
                temporary.unlink(missing_ok=True)


def create_rewrite_record_store(config: Any) -> RewriteRecordStore | None:
    """Pick a backend from the process-wide knowledge configuration.

    Returns ``None`` when remote mode is selected without KB Store
    credentials, which is the same "recorded nothing, cold start" outcome the
    rest of the rewrite path already handles.
    """
    from kernelforge.knowledge.experience_store import (
        KnowledgeStoreMode,
        knowledge_config_from_runtime,
    )

    knowledge = knowledge_config_from_runtime(config)
    if knowledge.mode is KnowledgeStoreMode.LOCAL:
        return LocalRewriteRecords(knowledge.rewrite_root)
    if not knowledge.kb_store_url:
        return None
    try:
        client = KBStoreClient(knowledge.kb_store_url, knowledge.kb_store_token)
    except KBStoreError:
        return None
    return KBStoreRewriteRecords(client)
