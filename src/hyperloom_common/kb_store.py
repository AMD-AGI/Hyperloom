# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Blocking client for the KB Store service, shared by Hyperloom and KernelForge.

Only small JSON control messages go to the service; artifact bytes travel straight to and
from the object store over presigned URLs. Every method raises :class:`KBStoreError` on
failure, chained (``__cause__``) to the ``urllib`` error that caused it, so a caller can
tell a server error or a dropped connection from a refused request. Producers that treat
the store as a best-effort side channel catch it and carry on.

A downloaded session is laid out as ``<dest>/files/<relative path>``, every file checked
against the sha256 and size its manifest declares before it is moved into place.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections.abc import Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path, PureWindowsPath
from typing import Any

#: Where a downloaded session keeps its artifacts, under the destination directory.
FILES_MEMBER_ROOT = "files"

_TIMEOUT_SEC = 60.0
_PARALLELISM = 8
_READ_CHUNK = 1 << 20
_ERROR_BODY_CHARS = 1024
_NOT_FOUND = "HTTP 404"
_SHA256_HEX = re.compile(r"[0-9a-f]{64}")


class KBStoreError(RuntimeError):
    """Any failure talking to the KB store or the object store."""


@dataclass(frozen=True)
class Upload:
    """One local file to publish at ``path`` under a session's ``files/`` tree."""

    path: str
    source: Path
    kind: str = "other"
    meta: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _Download:
    """One validated entry of a session file manifest."""

    path: str
    sha256: str
    size: int
    url: str


def _quote(value: str) -> str:
    # A canonical id is colon-separated and the service routes on it verbatim.
    return urllib.parse.quote(value, safe=":")


def _scope_query(scope: Mapping[str, Any] | None) -> str:
    return "?" + urllib.parse.urlencode(dict(scope)) if scope else ""


def _text(raw: bytes) -> str:
    return raw.decode("utf-8", errors="replace")


def _sha256_of(path: Path) -> tuple[str, int]:
    """Return ``(hex_digest, size_bytes)`` for a local file."""
    digest = hashlib.sha256()
    size = 0
    with open(path, "rb") as handle:
        while chunk := handle.read(_READ_CHUNK):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _bundle_rel_path(value: Any) -> str:
    """Validate a path relative to the session's ``files/`` directory."""
    if not isinstance(value, str):
        raise KBStoreError(f"artifact path must be a string: {value!r}")
    if not value:
        raise KBStoreError("artifact path is empty")
    if "\0" in value:
        raise KBStoreError(f"artifact path contains NUL: {value!r}")
    if "\\" in value:
        raise KBStoreError(f"artifact path must use forward slashes: {value!r}")
    if value.startswith("/") or PureWindowsPath(value).drive:
        raise KBStoreError(f"artifact path must be relative: {value!r}")
    if any(part in ("", ".", "..") for part in value.split("/")):
        raise KBStoreError(f"artifact path contains an empty or traversal component: {value!r}")
    return value


def _download_entry(index: int, entry: Any) -> _Download:
    if not isinstance(entry, Mapping):
        raise KBStoreError(f"download manifest entry {index} must be an object")
    rel = _bundle_rel_path(entry.get("path"))
    sha256 = entry.get("sha256")
    if not isinstance(sha256, str) or not _SHA256_HEX.fullmatch(sha256):
        raise KBStoreError(f"download manifest entry {rel!r} has invalid sha256")
    size = entry.get("size")
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise KBStoreError(f"download manifest entry {rel!r} has invalid size")
    url = entry.get("download_url")
    if not isinstance(url, str) or not url:
        raise KBStoreError(f"download manifest entry {rel!r} has no download_url")
    return _Download(rel, sha256, size, url)


def _download_manifest(listing: Any) -> list[_Download]:
    """Return the strictly validated entries of a session file manifest."""
    if not isinstance(listing, Mapping):
        raise KBStoreError("download manifest must be an object")
    raw_files = listing.get("files")
    if raw_files is None:
        return []
    if not isinstance(raw_files, list):
        raise KBStoreError("download manifest files must be a list")
    entries = [_download_entry(index, entry) for index, entry in enumerate(raw_files)]
    seen: set[str] = set()
    for entry in entries:
        if entry.path in seen:
            raise KBStoreError(f"duplicate artifact path in download manifest: {entry.path!r}")
        seen.add(entry.path)
    return entries


def _checked_download_target(files_root: Path, rel: str) -> Path:
    """Build a contained target without following an existing parent symlink."""
    target = files_root.joinpath(*rel.split("/"))
    try:
        target.resolve(strict=False).relative_to(files_root.resolve())
    except ValueError as exc:
        raise KBStoreError(f"artifact target escapes files directory: {rel!r}") from exc
    current = target.parent
    while current != files_root:
        if current.is_symlink():
            raise KBStoreError(f"artifact parent directory may not be a symlink: {current}")
        current = current.parent
    return target


def _require_directory(path: Path, role: str) -> None:
    if path.is_symlink():
        raise KBStoreError(f"{role} may not be a symlink: {path}")
    if path.exists() and not path.is_dir():
        raise KBStoreError(f"{role} is not a directory: {path}")


def _prepared_files_root(dest_dir: Path, entries: list[_Download]) -> Path:
    """Create the destination and its ``files/`` tree, refusing any target that escapes it."""
    _require_directory(dest_dir, "download destination")
    dest_dir.mkdir(parents=True, exist_ok=True)
    files_root = dest_dir / FILES_MEMBER_ROOT
    if entries:
        _require_directory(files_root, "files directory")
        files_root.mkdir(exist_ok=True)
        for entry in entries:
            _checked_download_target(files_root, entry.path)
    return files_root


def _fetch(files_root: Path, entry: _Download) -> Path:
    """Download one artifact to a partial file, verify it, then move it into place."""
    partial_path: Path | None = None
    digest = hashlib.sha256()
    size = 0
    try:
        target = _checked_download_target(files_root, entry.path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target = _checked_download_target(files_root, entry.path)
        partial_path = target.with_name(f".{target.name}.{uuid.uuid4().hex}.partial")
        with urllib.request.urlopen(entry.url, timeout=_TIMEOUT_SEC) as resp, open(partial_path, "xb") as out:
            while chunk := resp.read(_READ_CHUNK):
                digest.update(chunk)
                size += len(chunk)
                out.write(chunk)
        actual = digest.hexdigest()
        if actual != entry.sha256:
            raise KBStoreError(f"download of {entry.path!r} sha256 mismatch: expected {entry.sha256}, got {actual}")
        if size != entry.size:
            raise KBStoreError(f"download of {entry.path!r} size mismatch: expected {entry.size}, got {size}")
        _checked_download_target(files_root, entry.path)
        os.replace(partial_path, target)
        partial_path = None
    except KBStoreError:
        raise
    except Exception as exc:
        raise KBStoreError(f"download of {entry.path!r} failed: {exc!r}") from exc
    finally:
        if partial_path is not None:
            partial_path.unlink(missing_ok=True)
    return target


def _upload_one(url: str, path: Path, sha256: str) -> None:
    req = urllib.request.Request(
        url,
        data=path.read_bytes(),
        method="PUT",
        headers={
            "Content-Type": "application/octet-stream",
            # Part of the presigned signature: the URL only accepts bytes declared under this digest.
            "x-amz-meta-sha256": sha256,
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT_SEC) as resp:
            if resp.status not in (200, 201, 204):
                raise KBStoreError(f"upload of {path} returned HTTP {resp.status}")
    except urllib.error.HTTPError as exc:
        detail = _text(exc.read())[:_ERROR_BODY_CHARS]
        raise KBStoreError(f"upload of {path} failed: HTTP {exc.code}: {detail}") from exc
    except KBStoreError:
        raise
    except Exception as exc:
        raise KBStoreError(f"upload of {path} failed: {exc!r}") from exc


def _upload_entries(uploads: Iterable[Upload]) -> list[dict[str, Any]]:
    """Validate every upload, then declare each one's digest, refusing a path named twice."""
    checked: list[Upload] = []
    seen: set[str] = set()
    for upload in uploads:
        rel = _bundle_rel_path(upload.path)
        if rel in seen:
            raise KBStoreError(f"duplicate artifact path for upload: {rel!r}")
        seen.add(rel)
        checked.append(upload)
    entries: list[dict[str, Any]] = []
    for upload in checked:
        if not Path(upload.source).is_file():
            raise KBStoreError(f"not a file: {upload.source}")
        digest, size = _sha256_of(Path(upload.source))
        entries.append(
            {"path": upload.path, "sha256": digest, "size": size, "kind": upload.kind, "meta": dict(upload.meta)}
        )
    return entries


class KBStoreClient:
    """Blocking client for the KB store REST surface."""

    def __init__(self, base_url: str, token: str) -> None:
        if not base_url:
            raise KBStoreError("base_url is required")
        self._base = base_url.rstrip("/")
        self._token = token or ""

    # -- transport ----------------------------------------------------------

    def _request(self, method: str, path: str, payload: Any = None) -> Any:
        headers = {"Accept": "application/json"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        data = None
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(self._base + path, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=_TIMEOUT_SEC) as resp:
                raw = _text(resp.read())
        except urllib.error.HTTPError as exc:
            body = _text(exc.read())[:_ERROR_BODY_CHARS]
            raise KBStoreError(f"{method} {path} -> HTTP {exc.code}: {body}") from exc
        except Exception as exc:
            raise KBStoreError(f"{method} {path} transport error: {exc!r}") from exc
        if not raw.strip():
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise KBStoreError(f"{method} {path}: response was not JSON") from exc

    def _get(self, path: str) -> Any:
        return self._request("GET", path)

    def _post(self, path: str, payload: Mapping[str, Any]) -> Any:
        return self._request("POST", path, payload)

    def _get_or_none(self, path: str) -> Any:
        """GET ``path``, answering ``None`` when the service has nothing there."""
        try:
            return self._get(path)
        except KBStoreError as exc:
            if _NOT_FOUND in str(exc):
                return None
            raise

    @staticmethod
    def _identity(canonical_id: str) -> str:
        return f"/v1/kb/{_quote(canonical_id)}"

    def _session_base(self, canonical_id: str, session_id: str) -> str:
        return f"{self._identity(canonical_id)}/sessions/{_quote(session_id)}"

    # -- identities and sessions ----------------------------------------------

    def search_identities(
        self,
        *,
        scheme: str,
        match: Mapping[str, str] | None = None,
        hardware_in: list[str] | None = None,
        offset: int = 0,
        limit: int = 50,
    ) -> dict[str, Any]:
        """Discover indexed identities matching exact named dimensions, without fuzzy matching."""
        payload: dict[str, Any] = {
            "scheme": str(scheme),
            "match": dict(match or {}),
            "offset": int(offset),
            "limit": int(limit),
        }
        if hardware_in is not None:
            payload["hardware_in"] = [str(value) for value in hardware_in]
        return dict(self._post("/v1/kb/search", payload) or {})

    def get_rollup(self, canonical_id: str, *, scope: Mapping[str, Any] | None = None) -> dict[str, Any] | None:
        """Read the candidate index, or ``None`` when nothing is recorded."""
        return self._get_or_none(f"{self._identity(canonical_id)}/sessions{_scope_query(scope)}")

    def get_top_sessions(
        self,
        canonical_id: str,
        *,
        metric: str = "speedup",
        limit: int = 3,
        offset: int = 0,
    ) -> dict[str, Any]:
        """Rank the scored sessions the identity's rollup index retains."""
        query = urllib.parse.urlencode({"metric": metric, "limit": int(limit), "offset": int(offset)})
        return self._get(f"{self._identity(canonical_id)}/sessions/top?{query}") or {}

    def get_hyperloom_recipe_view(
        self, canonical_id: str, *, scope: Mapping[str, Any] | None = None
    ) -> dict[str, Any] | None:
        """Read the selected record normalized for Hyperloom replay, or ``None`` when there is none."""
        return self._get_or_none(f"{self._identity(canonical_id)}/views/hyperloom-recipe{_scope_query(scope)}")

    def set_champion(
        self,
        canonical_id: str,
        session_id: str,
        *,
        metric: str,
        value: float,
        scope: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Promote a session as the identity's best result."""
        payload: dict[str, Any] = {"session_id": session_id, "metric": metric, "value": value}
        if scope:
            payload["scope"] = dict(scope)
        return self._post(f"{self._identity(canonical_id)}/champion", payload)

    def get_session(self, canonical_id: str, session_id: str) -> dict[str, Any] | None:
        """Read a session document, or ``None`` when it does not exist."""
        return self._get_or_none(self._session_base(canonical_id, session_id))

    # -- knowledge ----------------------------------------------------------

    def replace_knowledge(
        self,
        canonical_id: str,
        session_id: str,
        knowledge: Mapping[str, Any],
        *,
        scope: Mapping[str, Any] | None = None,
        objective_schema: str = "",
    ) -> dict[str, Any]:
        """Overwrite a session's knowledge document."""
        payload: dict[str, Any] = {"knowledge": dict(knowledge), "mode": "replace", "session_id": session_id}
        if scope:
            payload["scope"] = dict(scope)
        if objective_schema:
            payload["objective_schema"] = objective_schema
        return self._post(self._identity(canonical_id), payload)

    def merge_knowledge(self, canonical_id: str, session_id: str, knowledge: Mapping[str, Any]) -> dict[str, Any]:
        """Shallow-merge ``knowledge`` over a session's stored document."""
        payload = {"knowledge": dict(knowledge), "mode": "merge", "session_id": session_id}
        return self._post(self._identity(canonical_id), payload)

    # -- upload -------------------------------------------------------------

    def put_files(self, canonical_id: str, session_id: str, uploads: Iterable[Upload]) -> dict[str, str]:
        """Upload a batch of files and return ``{path: kb:// reference}``.

        Digests are computed locally and declared up front, so the service can skip bytes
        it already holds and pin each object's digest into its presigned signature.
        """
        uploads = list(uploads)
        entries = _upload_entries(uploads)
        if not entries:
            return {}
        sources = {upload.path: Path(upload.source) for upload in uploads}
        digests = {entry["path"]: entry["sha256"] for entry in entries}
        base = self._session_base(canonical_id, session_id)
        grant = self._post(f"{base}/files:grant", {"files": entries}) or {}
        pending = [
            (item["path"], item["upload_url"])
            for item in grant.get("uploads") or []
            if not item.get("skip") and item.get("upload_url")
        ]
        if pending:
            with ThreadPoolExecutor(max_workers=_PARALLELISM) as pool:
                list(pool.map(lambda item: _upload_one(item[1], sources[item[0]], digests[item[0]]), pending))
        commit = self._post(f"{base}/files:commit", {"files": entries, "verify": True}) or {}
        committed = (commit.get("artifacts") or {}).get("files") or []
        refs = {str(item.get("path")): str(item.get("uri") or "") for item in committed}
        return {rel: refs.get(rel, "") for rel in sources}

    def put_dir(self, canonical_id: str, session_id: str, local_dir: str | Path) -> dict[str, str]:
        """Upload a whole directory tree, each file at its path relative to ``local_dir``."""
        root = Path(local_dir)
        if not root.is_dir():
            raise KBStoreError(f"not a directory: {root}")
        uploads = [
            Upload(path.relative_to(root).as_posix(), path) for path in sorted(root.rglob("*")) if path.is_file()
        ]
        return self.put_files(canonical_id, session_id, uploads)

    # -- download -----------------------------------------------------------

    def list_session_files(self, canonical_id: str, session_id: str) -> dict[str, Any]:
        """The session's file manifest, with a short-lived presigned GET URL per file."""
        return self._get(f"{self._session_base(canonical_id, session_id)}/files") or {}

    def download_session(
        self,
        canonical_id: str,
        session_id: str,
        dest_dir: str | Path,
        *,
        listing: Mapping[str, Any] | None = None,
    ) -> list[Path]:
        """Download and verify a session's files into ``dest_dir/files/``.

        ``listing`` is a manifest the caller already fetched with :meth:`list_session_files`
        and checked; the files downloaded are exactly the ones it names. Without it, the
        manifest is fetched here. Bytes come straight from the object store, concurrently.
        """
        if listing is None:
            listing = self.list_session_files(canonical_id, session_id)
        entries = _download_manifest(listing)
        files_root = _prepared_files_root(Path(dest_dir), entries)
        if not entries:
            return []
        with ThreadPoolExecutor(max_workers=_PARALLELISM) as pool:
            return list(pool.map(partial(_fetch, files_root), entries))


__all__ = ["FILES_MEMBER_ROOT", "KBStoreClient", "KBStoreError", "Upload"]
