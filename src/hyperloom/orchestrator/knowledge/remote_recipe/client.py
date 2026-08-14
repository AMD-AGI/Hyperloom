# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Recipe policy wrapper around the vendored KB Store SDK."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
import threading
from pathlib import Path
from typing import Any

from ._vendor.kb_store_client import KBStoreClient, KBStoreError
from .models import (
    MAX_FILE_BYTES,
    MAX_FILES,
    KnowledgeBundle,
    RemoteRecipeValidationError,
    RemoteWriteResult,
    validate_relative_path,
)
from .sanitize import sanitize_shared_knowledge

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_READ_CHUNK = 1024 * 1024
_STORE_LOCK_INIT = threading.Lock()


class RemoteRecipeConfigurationError(KBStoreError):
    """Remote Recipe KB was only partially configured."""


def _champion(
    rollup: dict[str, Any] | None,
    *,
    validate_metric: bool = False,
    expected_metric: str = "optimized_throughput",
) -> tuple[str, float, dict[str, Any]]:
    """Extract one incumbent, failing closed on an unrecognized rollup."""
    if rollup is None:
        return "", 0.0, {}
    if not isinstance(rollup, dict):
        raise RemoteRecipeValidationError("candidate rollup must be an object")
    if "champion" not in rollup:
        raise RemoteRecipeValidationError("candidate rollup is missing champion")
    raw_champion = rollup.get("champion")
    if raw_champion in (None, {}):
        sessions = rollup.get("sessions")
        if isinstance(sessions, list) and not sessions:
            return "", 0.0, {}
        raise RemoteRecipeValidationError(
            "candidate rollup has sessions but no champion"
        )
    if not isinstance(raw_champion, dict):
        raise RemoteRecipeValidationError("candidate rollup champion must be an object")
    champion = dict(raw_champion)
    metric = str(champion.get("metric") or "").strip()
    if validate_metric and metric != expected_metric:
        raise RemoteRecipeValidationError(
            "cannot compare or replace champion metric "
            f"{metric!r}; expected {expected_metric!r}"
        )
    session_id = str(champion.get("session_id") or "").strip()
    if not session_id:
        raise RemoteRecipeValidationError("candidate rollup champion is missing session_id")
    if "value" not in champion:
        raise RemoteRecipeValidationError("candidate rollup champion is missing value")
    raw_value = champion.get("value")
    if isinstance(raw_value, bool):
        raise RemoteRecipeValidationError(
            f"champion value must be numeric, got {raw_value!r}"
        )
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        raise RemoteRecipeValidationError(
            f"champion value must be numeric, got {raw_value!r}"
        ) from None
    if not math.isfinite(value):
        raise RemoteRecipeValidationError(
            f"champion value must be finite, got {raw_value!r}"
        )
    return session_id, value, champion


def flatten_recipe_document(envelope: dict[str, Any]) -> dict[str, Any]:
    """Merge service-owned envelope fields and opaque knowledge into recipe.json."""
    knowledge = envelope.get("knowledge") or {}
    business = dict(knowledge) if isinstance(knowledge, dict) else {}
    revision = envelope.get("revision", envelope.get("version"))
    raw_selection = envelope.get("selected_by")
    fixed = {
        "schema_version": envelope.get("schema_version"),
        "canonical_id": envelope.get("canonical_id"),
        "session_id": envelope.get("session_id"),
        "record_id": envelope.get("record_id"),
        "revision": revision,
        "version": envelope.get("version", revision),
        "selected_by": dict(raw_selection) if isinstance(raw_selection, dict) else {},
        "view": (
            dict(envelope.get("view") or {})
            if isinstance(envelope.get("view"), dict)
            else {}
        ),
    }
    return {**business, **fixed}


def _validate_session_envelope(
    envelope: Any,
    *,
    canonical_id: str,
    session_id: str,
) -> dict[str, Any]:
    """Validate service-owned identity/version fields before destination cleanup."""
    if not isinstance(envelope, dict):
        raise RemoteRecipeValidationError("session envelope must be an object")
    schema_version = envelope.get("schema_version")
    if isinstance(schema_version, bool) or schema_version != 2:
        raise RemoteRecipeValidationError(
            f"session envelope schema_version must be 2, got {schema_version!r}"
        )
    if str(envelope.get("canonical_id") or "") != canonical_id:
        raise RemoteRecipeValidationError(
            "session envelope canonical_id does not match the requested identity"
        )
    if not session_id:
        raise RemoteRecipeValidationError("session envelope session_id is required")
    if str(envelope.get("session_id") or "") != session_id:
        raise RemoteRecipeValidationError(
            "session envelope session_id does not match the requested session"
        )
    if not str(envelope.get("record_id") or "").strip():
        raise RemoteRecipeValidationError("session envelope record_id is required")
    if (
        ("revision" not in envelope or envelope.get("revision") is None)
        and ("version" not in envelope or envelope.get("version") is None)
    ):
        raise RemoteRecipeValidationError(
            "session envelope revision or version is required"
        )
    if not isinstance(envelope.get("knowledge"), dict):
        raise RemoteRecipeValidationError("session envelope knowledge must be an object")
    return envelope


def _validate_hyperloom_view(envelope: dict[str, Any]) -> dict[str, Any]:
    """Require the service-owned replayability decision on normalized reads."""
    view = envelope.get("view")
    if not isinstance(view, dict):
        raise RemoteRecipeValidationError(
            "Hyperloom Recipe View is missing top-level view metadata"
        )
    source = str(view.get("source") or "")
    if source not in {"current", "legacy_gbrain", "legacy_native"}:
        raise RemoteRecipeValidationError(
            f"Hyperloom Recipe View has invalid source: {source!r}"
        )
    if not isinstance(view.get("replayable"), bool):
        raise RemoteRecipeValidationError(
            "Hyperloom Recipe View replayable must be boolean"
        )
    reason = view.get("replay_disabled_reason")
    if reason is not None and not isinstance(reason, str):
        raise RemoteRecipeValidationError(
            "Hyperloom Recipe View replay_disabled_reason must be a string or null"
        )
    return envelope


def _validate_download_listing(
    listing: Any,
) -> dict[str, tuple[int, str]]:
    """Validate the SDK download manifest before any destination mutation."""
    if not isinstance(listing, dict):
        raise RemoteRecipeValidationError("session file listing must be an object")
    raw_files = listing.get("files") or []
    if not isinstance(raw_files, list):
        raise RemoteRecipeValidationError("session file listing files must be a list")
    if len(raw_files) > MAX_FILES:
        raise RemoteRecipeValidationError(
            f"session file count {len(raw_files)} exceeds KB Store limit {MAX_FILES}"
        )
    seen: set[str] = set()
    manifest: dict[str, tuple[int, str]] = {}
    for index, entry in enumerate(raw_files):
        if not isinstance(entry, dict):
            raise RemoteRecipeValidationError(f"session file entry {index} must be an object")
        path = validate_relative_path(str(entry.get("path") or ""))
        if path in seen:
            raise RemoteRecipeValidationError(f"duplicate session file path: {path}")
        seen.add(path)
        digest = entry.get("sha256")
        if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
            raise RemoteRecipeValidationError(
                f"session file {path!r} sha256 must be 64 lowercase hex characters"
            )
        if "size" not in entry:
            raise RemoteRecipeValidationError(
                f"session file {path!r} size is required"
            )
        raw_size = entry.get("size")
        if isinstance(raw_size, bool) or not isinstance(raw_size, int):
            raise RemoteRecipeValidationError(
                f"session file {path!r} size must be an integer when present"
            )
        if raw_size < 0 or raw_size > MAX_FILE_BYTES:
            raise RemoteRecipeValidationError(
                f"session file {path!r} size {raw_size} is outside 0..{MAX_FILE_BYTES}"
            )
        manifest[path] = (raw_size, digest)
    return manifest


def _verify_downloaded_files(
    files_root: Path,
    manifest: dict[str, tuple[int, str]],
) -> None:
    """Require an exact, regular-file match with size and SHA256 verification."""
    if files_root.is_symlink():
        raise RemoteRecipeValidationError(f"downloaded files root is a symlink: {files_root}")
    actual: set[str] = set()
    if files_root.exists() and not files_root.is_dir():
        raise RemoteRecipeValidationError(f"downloaded files root is not a directory: {files_root}")
    candidates = files_root.rglob("*") if files_root.is_dir() else ()
    for path in candidates:
        if path.is_symlink():
            raise RemoteRecipeValidationError(f"downloaded artifact is a symlink: {path}")
        if path.is_file():
            actual.add(path.relative_to(files_root).as_posix())
    expected = set(manifest)
    missing = expected - actual
    extra = actual - expected
    if missing or extra:
        raise RemoteRecipeValidationError(
            f"downloaded artifact set mismatch: missing={sorted(missing)!r} extra={sorted(extra)!r}"
        )
    for relative_path, (expected_size, expected_digest) in manifest.items():
        path = files_root / relative_path
        if path.is_symlink() or not path.is_file():
            raise RemoteRecipeValidationError(f"downloaded artifact is not a regular file: {relative_path}")
        actual_size = path.stat().st_size
        if actual_size != expected_size:
            raise RemoteRecipeValidationError(
                f"downloaded artifact size mismatch for {relative_path!r}: "
                f"expected={expected_size} actual={actual_size}"
            )
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(_READ_CHUNK):
                digest.update(chunk)
        actual_digest = digest.hexdigest()
        if actual_digest != expected_digest:
            raise RemoteRecipeValidationError(
                f"downloaded artifact sha256 mismatch for {relative_path!r}: "
                f"expected={expected_digest} actual={actual_digest}"
            )


def _deactivate_destination(path: Path) -> None:
    """Remove a destination without following a symlink."""
    if path.is_symlink() or (path.exists() and not path.is_dir()):
        path.unlink(missing_ok=True)
    elif path.is_dir():
        shutil.rmtree(path)


def _view_artifact_manifest(
    envelope: dict[str, Any],
) -> dict[str, tuple[int, str]]:
    """Validate and return the View-embedded artifact snapshot."""
    artifacts = envelope.get("artifacts")
    if not isinstance(artifacts, dict):
        raise RemoteRecipeValidationError(
            "Hyperloom Recipe View artifacts manifest must be an object"
        )
    files = artifacts.get("files")
    if not isinstance(files, list):
        raise RemoteRecipeValidationError(
            "Hyperloom Recipe View artifacts.files must be a list"
        )
    manifest = _validate_download_listing({"files": files})
    file_count = artifacts.get("file_count")
    if (
        isinstance(file_count, bool)
        or not isinstance(file_count, int)
        or file_count != len(manifest)
    ):
        raise RemoteRecipeValidationError(
            "Hyperloom Recipe View artifacts.file_count does not match files"
        )
    return manifest


class RemoteRecipeClient:
    """Recipe-specific read/write policy over the vendored store client."""

    def __init__(self, store: KBStoreClient) -> None:
        self.store = store
        with _STORE_LOCK_INIT:
            lock = getattr(store, "_hyperloom_remote_recipe_lock", None)
            if lock is None:
                lock = threading.RLock()
                setattr(store, "_hyperloom_remote_recipe_lock", lock)
        self._store_lock = lock

    @classmethod
    def from_env_optional(cls) -> "RemoteRecipeClient | None":
        """Feature-detect both required variables; fail fast on partial config."""
        base = os.environ.get("KB_STORE_URL", "").strip()
        token = os.environ.get("KB_STORE_TOKEN", "").strip()
        if not base and not token:
            return None
        if not base or not token:
            missing = "KB_STORE_URL" if not base else "KB_STORE_TOKEN"
            raise RemoteRecipeConfigurationError(
                f"{missing} is required when Remote Recipe KB V2 is enabled"
            )
        return cls(KBStoreClient(base, token))

    def get_view(self, canonical_id: str) -> dict[str, Any] | None:
        """Read and validate one normalized View without downloading files."""
        envelope = self.store.get_hyperloom_recipe_view(canonical_id)
        if envelope is None:
            return None
        session_id = (
            str(envelope.get("session_id") or "").strip()
            if isinstance(envelope, dict)
            else ""
        )
        envelope = _validate_session_envelope(
            envelope,
            canonical_id=canonical_id,
            session_id=session_id,
        )
        envelope = _validate_hyperloom_view(envelope)
        _view_artifact_manifest(envelope)
        return envelope

    def read(
        self,
        canonical_id: str,
        destination: str | Path,
        *,
        envelope: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Activate a fully verified View bundle at the destination."""
        root = Path(destination)
        generation: Path | None = None
        if root.parent.is_dir():
            for stale_generation in root.parent.glob(
                f".{root.name}.generation-*"
            ):
                _deactivate_destination(stale_generation)
        try:
            selected = envelope if envelope is not None else self.get_view(canonical_id)
            if selected is None:
                _deactivate_destination(root)
                return None
            session_id = str(selected.get("session_id") or "").strip()
            selected = _validate_session_envelope(
                selected,
                canonical_id=canonical_id,
                session_id=session_id,
            )
            selected = _validate_hyperloom_view(selected)
            view_manifest = _view_artifact_manifest(selected)
            document = flatten_recipe_document(selected)
            recipe_json = json.dumps(
                document,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
        except (TypeError, ValueError) as exc:
            _deactivate_destination(root)
            raise RemoteRecipeValidationError(
                f"downloaded recipe is not strict JSON: {exc}"
            ) from exc
        except Exception:
            _deactivate_destination(root)
            raise

        try:
            with self._store_lock:
                listing = self.store.list_session_files(
                    canonical_id,
                    session_id,
                )
                listing_manifest = _validate_download_listing(listing)
                if listing_manifest != view_manifest:
                    raise RemoteRecipeValidationError(
                        "session file listing does not match the selected "
                        "Hyperloom Recipe View artifact manifest"
                    )
                root.parent.mkdir(parents=True, exist_ok=True)
                generation = Path(
                    tempfile.mkdtemp(
                        prefix=f".{root.name}.generation-",
                        dir=root.parent,
                    )
                )
                files_root = generation / "files"
                if listing_manifest:
                    # Pin the SDK's internal listing to the compared snapshot.
                    original_listing_method = self.store.list_session_files

                    def validated_listing(
                        requested_canonical_id: str,
                        requested_session_id: str,
                        *,
                        kind: str = "",
                    ) -> dict[str, Any]:
                        if (
                            requested_canonical_id != canonical_id
                            or requested_session_id != session_id
                            or kind
                        ):
                            return original_listing_method(
                                requested_canonical_id,
                                requested_session_id,
                                kind=kind,
                            )
                        return listing

                    self.store.list_session_files = validated_listing  # type: ignore[method-assign]
                    try:
                        self.store.download_session(
                            canonical_id,
                            session_id,
                            generation,
                            include_values=False,
                        )
                    finally:
                        self.store.list_session_files = original_listing_method  # type: ignore[method-assign]
                files_root.mkdir(parents=True, exist_ok=True)
                _verify_downloaded_files(files_root, listing_manifest)
                for generated in list(generation.iterdir()):
                    if generated.name == "files":
                        continue
                    _deactivate_destination(generated)
                (generation / "recipe.json").write_text(
                    recipe_json,
                    encoding="utf-8",
                )
                _deactivate_destination(root)
                generation.replace(root)
                generation = None
            return document
        except Exception:
            _deactivate_destination(root)
            if generation is not None:
                _deactivate_destination(generation)
            raise

    def search_identities(
        self,
        *,
        scheme: str,
        match: dict[str, str],
        hardware_in: list[str] | None = None,
        offset: int = 0,
        limit: int = 50,
    ) -> dict[str, Any]:
        """Return exact identity matches from the KB Store discovery endpoint."""
        return self.store.search_identities(
            scheme=scheme,
            match=match,
            hardware_in=hardware_in,
            offset=offset,
            limit=limit,
        )

    # Compatibility name for callers that explicitly describe the selection.
    read_champion = read

    def write_if_better(
        self,
        canonical_id: str,
        session_id: str,
        bundle: KnowledgeBundle,
        *,
        optimized_throughput: float,
        files_dir: Path,
        metric: str = "optimized_throughput",
    ) -> RemoteWriteResult:
        """Write files, replace knowledge, then promote when the score wins.

        ``metric`` names what the score means. An identity whose records are not
        graded on serving throughput passes its own name so the incumbent is
        only ever compared against a like-for-like reading.
        """
        if not math.isfinite(optimized_throughput):
            raise RemoteRecipeValidationError(
                f"optimized_throughput must be finite, got {optimized_throughput!r}"
            )
        # Defense in depth at the final shared-store boundary. Builders sanitize
        # earlier so their outputs are safe to inspect, but callers can also
        # construct a KnowledgeBundle directly.
        bundle.knowledge = sanitize_shared_knowledge(bundle.knowledge)
        bundle.validate()
        rollup = self.store.get_rollup(canonical_id)
        _, prior, _ = _champion(rollup, validate_metric=True, expected_metric=metric)
        if optimized_throughput <= prior:
            return RemoteWriteResult(
                "skipped",
                "not_better_than_champion",
                canonical_id,
                session_id,
                optimized_throughput,
            )
        expected = {artifact.path for artifact in bundle.artifacts}
        if expected:
            refs = self.store.put_dir(canonical_id, session_id, files_dir)
            returned = set(refs)
            missing = expected - returned
            blank = {path for path in expected if not str(refs.get(path) or "").strip()}
            unexpected = returned - expected
            if missing or blank or unexpected:
                raise RemoteRecipeValidationError(
                    "put_dir refs mismatch: "
                    f"missing={sorted(missing)!r} blank={sorted(blank)!r} "
                    f"unexpected={sorted(unexpected)!r}"
                )
        self.store.put_knowledge(
            canonical_id, bundle.knowledge, session_id=session_id, mode="replace"
        )
        try:
            self.store.set_champion(
                canonical_id,
                session_id,
                metric=metric,
                value=optimized_throughput,
            )
        except KBStoreError as exc:
            if "HTTP 409" not in str(exc):
                raise
            _, winner, _ = _champion(
                self.store.get_rollup(canonical_id),
                validate_metric=True,
                expected_metric=metric,
            )
            if winner < optimized_throughput:
                self.store.set_champion(
                    canonical_id,
                    session_id,
                    metric=metric,
                    value=optimized_throughput,
                )
        return RemoteWriteResult(
            "written",
            "",
            canonical_id,
            session_id,
            optimized_throughput,
        )


__all__ = [
    "KBStoreClient",
    "KBStoreError",
    "RemoteRecipeClient",
    "RemoteRecipeConfigurationError",
    "flatten_recipe_document",
]
