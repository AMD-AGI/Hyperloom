"""Standalone client for the KB store.

Intentionally stdlib-only so producers (Hyperloom orchestrator, agents,
CLI tools) can vendor this single file without pulling in boto3 or an
async HTTP stack. Uploads and downloads go straight to the object store
over presigned URLs; only small JSON control messages touch the service.

Typical producer flow::

    store = KBStoreClient.from_env()
    store.put_knowledge(cid, {"prs_tested": [...]})
    ref = store.put_file(cid, session_id, "patches/pr-123.patch",
                         local_path, kind="patch",
                         meta={"pr_url": url, "outcome": "integrated"})

Typical consumer flow::

    store.download_session(cid, session_id, Path("/tmp/session"))

Every method raises :class:`KBStoreError` on failure. Producers that treat
the KB as a best-effort side channel should catch it and carry on: losing a
record must never fail the optimization run that produced it.
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections.abc import Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PureWindowsPath
from typing import Any

DEFAULT_TIMEOUT_SEC = 60.0
DEFAULT_PARALLELISM = 8
_READ_CHUNK = 1024 * 1024

#: Bundle layout, kept identical to what the archive endpoint emits so the
#: two download routes are interchangeable for consumers.
VALUES_MEMBER = "values.json"
FILES_MEMBER_ROOT = "files"


class KBStoreError(RuntimeError):
    """Any failure talking to the KB store or the object store."""


#: Must match ``knowledge_base.canonical.RECORD_NAMESPACE``.
RECORD_NAMESPACE = uuid.UUID("0f4d5e6a-8b7c-4d1e-9f2a-3c5b7d9e1f00")


def record_id(canonical_id: str, session_id: str) -> str:
    """Compute a record's UUID locally, without calling the service.

    The id is derived from the identity rather than allocated, so a
    producer can record it (in a session breakdown, a DB row, a log line)
    before the record exists and know the value will match.

    The whole identity is hashed, scheme segment included, which is what
    keeps an ``inference:`` id from colliding with a ``kernel:`` one.
    """
    cid = (canonical_id or "").strip()
    scheme, _, dims = cid.partition(":")
    if not scheme or not dims:
        raise KBStoreError(f"canonical_id {canonical_id!r} is malformed")
    sid = (session_id or "").strip()
    if not sid:
        raise KBStoreError(f"session_id {session_id!r} is malformed")
    return str(uuid.uuid5(RECORD_NAMESPACE, f"{cid}|{sid}"))


def sha256_of(path: str | Path) -> tuple[str, int]:
    """Return ``(hex_digest, size_bytes)`` for a local file."""
    digest = hashlib.sha256()
    size = 0
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(_READ_CHUNK)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _bundle_rel_path(value: Any) -> str:
    """Validate a path relative to the bundle's ``files/`` directory."""
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


def _validated_download_manifest(
    listing: Any,
) -> list[tuple[str, str, int, str]]:
    """Return strictly validated download entries."""
    if not isinstance(listing, Mapping):
        raise KBStoreError("download manifest must be an object")
    raw_files = listing.get("files")
    if raw_files is None:
        return []
    if not isinstance(raw_files, list):
        raise KBStoreError("download manifest files must be a list")

    entries: list[tuple[str, str, int, str]] = []
    seen: set[str] = set()
    for index, entry in enumerate(raw_files):
        if not isinstance(entry, Mapping):
            raise KBStoreError(f"download manifest entry {index} must be an object")
        rel = _bundle_rel_path(entry.get("path"))
        if rel in seen:
            raise KBStoreError(f"duplicate artifact path in download manifest: {rel!r}")
        seen.add(rel)

        expected_sha = entry.get("sha256")
        if (
            not isinstance(expected_sha, str)
            or len(expected_sha) != 64
            or any(char not in "0123456789abcdef" for char in expected_sha)
        ):
            raise KBStoreError(f"download manifest entry {rel!r} has invalid sha256")
        expected_size = entry.get("size")
        if isinstance(expected_size, bool) or not isinstance(expected_size, int) or expected_size < 0:
            raise KBStoreError(f"download manifest entry {rel!r} has invalid size")
        url = entry.get("download_url")
        if not isinstance(url, str) or not url:
            raise KBStoreError(f"download manifest entry {rel!r} has no download_url")
        entries.append((rel, expected_sha, expected_size, url))
    return entries


def _checked_download_target(files_root: Path, rel: str) -> Path:
    """Build a contained target without following an existing parent symlink."""
    target = files_root.joinpath(*rel.split("/"))
    resolved_root = files_root.resolve()
    try:
        target.resolve(strict=False).relative_to(resolved_root)
    except ValueError as exc:
        raise KBStoreError(f"artifact target escapes files directory: {rel!r}") from exc

    current = target.parent
    while current != files_root:
        if current.is_symlink():
            raise KBStoreError(f"artifact parent directory may not be a symlink: {current}")
        current = current.parent
    return target


class KBStoreClient:
    """Blocking client for the KB store REST surface."""

    def __init__(
        self,
        base_url: str,
        token: str,
        *,
        timeout_sec: float = DEFAULT_TIMEOUT_SEC,
        parallelism: int = DEFAULT_PARALLELISM,
    ) -> None:
        if not base_url:
            raise KBStoreError("base_url is required")
        self._base = base_url.rstrip("/")
        self._token = token or ""
        self._timeout = timeout_sec
        self._parallelism = max(1, parallelism)

    @classmethod
    def from_env(cls) -> KBStoreClient:
        """Build from ``KB_STORE_URL`` / ``KB_STORE_TOKEN``."""
        base = (os.environ.get("KB_STORE_URL", "") or "").strip()
        token = (os.environ.get("KB_STORE_TOKEN", "") or "").strip()
        if not base:
            raise KBStoreError("KB_STORE_URL is not set")
        return cls(base, token)

    @classmethod
    def from_env_optional(cls) -> KBStoreClient | None:
        """Build from env, or return ``None`` when unconfigured.

        Lets a producer make KB writes opt-in without wrapping every call
        site in try/except.
        """
        try:
            return cls.from_env()
        except KBStoreError:
            return None

    # -- transport ----------------------------------------------------------

    def _request(self, method: str, path: str, payload: Any = None) -> Any:
        url = self._base + path
        data = None
        headers = {"Accept": "application/json"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"

        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")[:1024]
            raise KBStoreError(f"{method} {path} -> HTTP {exc.code}: {body}") from exc
        except Exception as exc:
            raise KBStoreError(f"{method} {path} transport error: {exc!r}") from exc

        if not raw.strip():
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise KBStoreError(f"{method} {path}: response was not JSON") from exc

    @staticmethod
    def _quote(value: str) -> str:
        # Colons are legal in a path segment and the canonical id relies on
        # them, so they are explicitly kept unescaped.
        return urllib.parse.quote(value, safe=":")

    def _session_base(self, canonical_id: str, session_id: str) -> str:
        return f"/v1/kb/{self._quote(canonical_id)}/sessions/{self._quote(session_id)}"

    # -- knowledge ----------------------------------------------------------

    def put_knowledge(
        self,
        canonical_id: str,
        knowledge: dict[str, Any],
        *,
        session_id: str = "",
        mode: str = "merge",
    ) -> dict[str, Any]:
        """Record what this producer knows about an identity.

        ``session_id`` names a candidate under the identity and is optional;
        omitting it writes to a slot of this producer's own. Pass it to keep
        separate runs comparable — the champion is picked from candidates.
        The resolved id comes back as ``session_id`` in the response.
        """
        payload: dict[str, Any] = {"knowledge": knowledge, "mode": mode}
        if session_id:
            payload["session_id"] = session_id
        return self._request("POST", f"/v1/kb/{self._quote(canonical_id)}", payload)

    def get_session(self, canonical_id: str, session_id: str) -> dict[str, Any] | None:
        """Read a session document, or ``None`` when it does not exist."""
        try:
            return self._request("GET", self._session_base(canonical_id, session_id))
        except KBStoreError as exc:
            if "HTTP 404" in str(exc):
                return None
            raise

    def get_record(self, rid: str) -> dict[str, Any] | None:
        """Fetch a record by UUID alone, or ``None`` when it does not exist."""
        try:
            return self._request("GET", f"/v1/records/{self._quote(rid)}")
        except KBStoreError as exc:
            if "HTTP 404" in str(exc):
                return None
            raise

    def get_best_record(self, canonical_id: str) -> dict[str, Any] | None:
        """The record to act on for an identity, or ``None`` if there is none.

        Answers from the v1 recipe page when an identity predates this store,
        so a caller does not have to know which plane its data lives in.
        """
        try:
            return self._request("GET", f"/v1/kb/{self._quote(canonical_id)}")
        except KBStoreError as exc:
            if "HTTP 404" in str(exc):
                return None
            raise

    def get_rollup(self, canonical_id: str) -> dict[str, Any] | None:
        """Read the candidate index, or ``None`` when nothing is recorded."""
        try:
            return self._request("GET", f"/v1/kb/{self._quote(canonical_id)}/sessions")
        except KBStoreError as exc:
            if "HTTP 404" in str(exc):
                return None
            raise

    def get_top_sessions(
        self,
        canonical_id: str,
        *,
        metric: str = "speedup",
        limit: int = 3,
        offset: int = 0,
    ) -> dict[str, Any]:
        """Rank scored sessions retained by the identity's rollup index."""
        query = urllib.parse.urlencode({"metric": metric, "limit": int(limit), "offset": int(offset)})
        path = f"/v1/kb/{self._quote(canonical_id)}/sessions/top?{query}"
        return self._request("GET", path) or {}

    def list_identity_files(self, canonical_id: str, *, kind: str = "") -> list[dict[str, Any]]:
        """Artifacts across all sessions of an identity, deduped by digest."""
        path = f"/v1/kb/{self._quote(canonical_id)}/files"
        if kind:
            path += "?" + urllib.parse.urlencode({"kind": kind})
        result = self._request("GET", path) or {}
        return list(result.get("files") or [])

    def set_champion(
        self, canonical_id: str, session_id: str, *, metric: str = "throughput", value: float = 0.0
    ) -> dict[str, Any]:
        """Promote a session as the identity's best result."""
        return self._request(
            "POST",
            f"/v1/kb/{self._quote(canonical_id)}/champion",
            {"session_id": session_id, "metric": metric, "value": value},
        )

    # -- upload -------------------------------------------------------------

    def put_file(
        self,
        canonical_id: str,
        session_id: str,
        rel_path: str,
        local_path: str | Path,
        *,
        kind: str = "other",
        meta: dict[str, Any] | None = None,
    ) -> str:
        """Upload one file and return its durable ``kb://`` reference."""
        refs = self.put_files(
            canonical_id,
            session_id,
            [(rel_path, local_path, kind, meta or {})],
        )
        return refs[rel_path]

    def put_files(
        self,
        canonical_id: str,
        session_id: str,
        items: Iterable[tuple[str, str | Path, str, dict[str, Any]]],
    ) -> dict[str, str]:
        """Upload a batch of files; returns ``{rel_path: kb:// reference}``.

        Digests are computed locally and declared up front, so the service
        can skip bytes it already holds and can pin the uploaded object's
        recorded digest into the presigned signature.
        """
        validated: list[tuple[str, str | Path, str, dict[str, Any]]] = []
        seen: set[str] = set()
        for rel_path, local_path, kind, meta in items:
            rel = _bundle_rel_path(rel_path)
            if rel in seen:
                raise KBStoreError(f"duplicate artifact path for upload: {rel!r}")
            seen.add(rel)
            validated.append((rel, local_path, kind, meta))

        entries: list[dict[str, Any]] = []
        sources: dict[str, Path] = {}
        for rel_path, local_path, kind, meta in validated:
            path = Path(local_path)
            if not path.is_file():
                raise KBStoreError(f"not a file: {path}")
            digest, size = sha256_of(path)
            entries.append(
                {
                    "path": rel_path,
                    "sha256": digest,
                    "size": size,
                    "kind": kind,
                    "meta": meta or {},
                }
            )
            sources[rel_path] = path
        if not entries:
            return {}

        grant = self._request(
            "POST",
            self._session_base(canonical_id, session_id) + "/files:grant",
            {"files": entries},
        )

        pending = [
            (u["path"], u["upload_url"])
            for u in (grant.get("uploads") or [])
            if not u.get("skip") and u.get("upload_url")
        ]
        by_path = {e["path"]: e for e in entries}
        if pending:
            with ThreadPoolExecutor(max_workers=self._parallelism) as pool:
                list(
                    pool.map(
                        lambda item: self._upload_one(item[1], sources[item[0]], by_path[item[0]]["sha256"]),
                        pending,
                    )
                )

        commit = self._request(
            "POST",
            self._session_base(canonical_id, session_id) + "/files:commit",
            {"files": entries, "verify": True},
        )
        manifest = {
            str(f.get("path")): str(f.get("uri") or "") for f in (commit.get("artifacts") or {}).get("files") or []
        }
        return {rel: manifest.get(rel, "") for rel in sources}

    def put_dir(
        self,
        canonical_id: str,
        session_id: str,
        local_dir: str | Path,
        *,
        prefix: str = "",
        kind: str = "other",
        meta: dict[str, Any] | None = None,
    ) -> dict[str, str]:
        """Upload a whole directory tree, preserving relative paths."""
        safe_prefix = _bundle_rel_path(prefix) if prefix else ""
        root = Path(local_dir)
        if not root.is_dir():
            raise KBStoreError(f"not a directory: {root}")
        items: list[tuple[str, Path, str, dict[str, Any]]] = []
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(root).as_posix()
            if safe_prefix:
                rel = f"{safe_prefix}/{rel}"
            items.append((rel, path, kind, dict(meta or {})))
        return self.put_files(canonical_id, session_id, items)

    def _upload_one(self, url: str, path: Path, sha256: str) -> None:
        with open(path, "rb") as handle:
            body = handle.read()
        req = urllib.request.Request(
            url,
            data=body,
            method="PUT",
            headers={
                "Content-Type": "application/octet-stream",
                # Part of the presigned signature; the URL is only valid
                # for bytes declared under this digest.
                "x-amz-meta-sha256": sha256,
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                if resp.status not in (200, 201, 204):
                    raise KBStoreError(f"upload of {path} returned HTTP {resp.status}")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:512]
            raise KBStoreError(f"upload of {path} failed: HTTP {exc.code}: {detail}") from exc
        except Exception as exc:
            raise KBStoreError(f"upload of {path} failed: {exc!r}") from exc

    # -- download -----------------------------------------------------------

    def list_session_files(self, canonical_id: str, session_id: str, *, kind: str = "") -> dict[str, Any]:
        """Manifest with a short-lived presigned GET URL per file."""
        path = self._session_base(canonical_id, session_id) + "/files"
        if kind:
            path += "?" + urllib.parse.urlencode({"kind": kind})
        return self._request("GET", path) or {}

    def download_session(
        self,
        canonical_id: str,
        session_id: str,
        dest_dir: str | Path,
        *,
        kind: str = "",
        include_values: bool = True,
    ) -> list[Path]:
        """Download and verify a record in the standard bundle layout.

        Produces the same tree as the archive endpoint::

            values.json                 the knowledge payload
            files/<relative path>       every artifact

        so a consumer can read ``values.json`` and resolve any path it
        references under ``files/`` without caring which of the two
        download routes produced the directory.

        Artifact bytes come straight from the object store over presigned
        URLs, concurrently, and never transit the KB store.
        """
        listing = self.list_session_files(canonical_id, session_id, kind=kind)
        entries = _validated_download_manifest(listing)
        root = Path(dest_dir)
        if root.is_symlink():
            raise KBStoreError(f"download destination may not be a symlink: {root}")
        if root.exists() and not root.is_dir():
            raise KBStoreError(f"download destination is not a directory: {root}")
        root.mkdir(parents=True, exist_ok=True)

        files_root = root / FILES_MEMBER_ROOT
        targets: dict[str, Path] = {}
        if entries:
            if files_root.is_symlink():
                raise KBStoreError(f"files directory may not be a symlink: {files_root}")
            if files_root.exists() and not files_root.is_dir():
                raise KBStoreError(f"files path is not a directory: {files_root}")
            files_root.mkdir(exist_ok=True)
            for rel, _expected_sha, _expected_size, _url in entries:
                targets[rel] = _checked_download_target(files_root, rel)

        if include_values:
            document = self.get_session(canonical_id, session_id) or {}
            values = document.get("knowledge") or {}
            values_target = root / VALUES_MEMBER
            if values_target.is_symlink():
                raise KBStoreError(f"values target may not be a symlink: {values_target}")
            try:
                values_target.resolve(strict=False).relative_to(root.resolve())
            except ValueError as exc:
                raise KBStoreError("values target escapes download destination") from exc
            values_target.write_text(
                json.dumps(values, ensure_ascii=False, indent=2, sort_keys=True),
                encoding="utf-8",
            )

        def fetch(entry: tuple[str, str, int, str]) -> Path:
            rel, expected_sha, expected_size, url = entry
            target = targets[rel]
            partial: Path | None = None
            digest = hashlib.sha256()
            size = 0
            try:
                target = _checked_download_target(files_root, rel)
                target.parent.mkdir(parents=True, exist_ok=True)
                target = _checked_download_target(files_root, rel)
                partial = target.with_name(f".{target.name}.{uuid.uuid4().hex}.partial")
                with urllib.request.urlopen(url, timeout=self._timeout) as resp, open(partial, "xb") as out:
                    while True:
                        chunk = resp.read(_READ_CHUNK)
                        if not chunk:
                            break
                        digest.update(chunk)
                        size += len(chunk)
                        out.write(chunk)
                actual_sha = digest.hexdigest()
                if actual_sha != expected_sha:
                    raise KBStoreError(
                        f"download of {rel!r} sha256 mismatch: expected {expected_sha}, got {actual_sha}"
                    )
                if size != expected_size:
                    raise KBStoreError(f"download of {rel!r} size mismatch: expected {expected_size}, got {size}")
                _checked_download_target(files_root, rel)
                os.replace(partial, target)
                partial = None
            except KBStoreError:
                raise
            except Exception as exc:
                raise KBStoreError(f"download of {rel!r} failed: {exc!r}") from exc
            finally:
                if partial is not None:
                    partial.unlink(missing_ok=True)
            return target

        if not entries:
            return []
        with ThreadPoolExecutor(max_workers=self._parallelism) as pool:
            return list(pool.map(fetch, entries))

    def download_archive(self, canonical_id: str, session_id: str, dest_file: str | Path) -> Path:
        """Download the session directory as a single tar.gz."""
        url = self._base + self._session_base(canonical_id, session_id) + "/archive"
        headers = {}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        req = urllib.request.Request(url, headers=headers, method="GET")
        target = Path(dest_file)
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp, open(target, "wb") as out:
                while True:
                    chunk = resp.read(_READ_CHUNK)
                    if not chunk:
                        break
                    out.write(chunk)
        except Exception as exc:
            raise KBStoreError(f"archive download failed: {exc!r}") from exc
        return target


#: Where a sectioned document keeps its per-section maps. The service treats
#: ``knowledge`` as opaque, so this is a producer-side convention rather than
#: part of the record schema; it is fixed because documents already in the
#: store are written under this key.
SECTION_ROOT = "value"

_DRAFT_DIR_ENV = "KB_DRAFT_DIR"
_WARM_START_DIR_ENV = "KB_WARM_START_DIR"
_SECTIONS_MEMBER = "sections"
_RECIPE_MEMBER = "recipe.json"


def _checked_section(name: str) -> str:
    """Reject a section name that would escape its subtree or collide oddly."""
    section = str(name or "").strip()
    if not section:
        raise KBStoreError("section name is required")
    if section != section.strip("."):
        raise KBStoreError(f"section {name!r} may not start or end with a dot")
    bad = set(section) & set("/\\\0")
    if bad or section in (".", ".."):
        raise KBStoreError(f"section {name!r} may not contain a path separator")
    return section


class SectionContent:
    """One section's knowledge map plus the local files that belong to it."""

    __slots__ = ("files", "knowledge", "section")

    def __init__(
        self,
        section: str,
        knowledge: dict[str, Any],
        files: list[Path] | None = None,
    ) -> None:
        self.section = section
        self.knowledge = knowledge
        self.files = list(files or [])

    def __repr__(self) -> str:
        return f"SectionContent(section={self.section!r}, keys={sorted(self.knowledge)!r}, files={len(self.files)})"


class KnowledgeSections:
    """Section-scoped staging for one knowledge document, backed by a directory.

    A producer is usually several processes: agents that each own one section
    and a publisher that uploads once at the end. They cannot share a client
    object, so the draft is a directory that both sides open by path.

    Layout under ``root``::

        sections/<section>.json     one section's staged knowledge map
        files/<section>/<kind>/...  that section's artifacts, ready for put_dir

    ``files`` is laid out exactly as ``put_dir`` expects, so publishing is
    ``put_dir(cid, sid, sections.files_dir)`` with no repacking.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        warm_start_dir: str | Path | None = None,
    ) -> None:
        self.root = Path(root)
        self.warm_start_dir = Path(warm_start_dir) if warm_start_dir else None

    @classmethod
    def from_env(cls) -> KnowledgeSections | None:
        """Open the draft an orchestrator prepared, or ``None`` when absent.

        Lets an agent stay agnostic about whether this run publishes at all:
        no draft directory means nobody is collecting, so skip the write.
        """
        draft = (os.environ.get(_DRAFT_DIR_ENV, "") or "").strip()
        if not draft:
            return None
        warm = (os.environ.get(_WARM_START_DIR_ENV, "") or "").strip()
        return cls(draft, warm_start_dir=warm or None)

    @property
    def files_dir(self) -> Path:
        """The subtree to hand to :meth:`KBStoreClient.put_dir`."""
        return self.root / FILES_MEMBER_ROOT

    # -- write ---------------------------------------------------------------

    def write(
        self,
        section: str,
        knowledge: dict[str, Any],
        *,
        files: Iterable[str | Path] = (),
        kind: str = "artifacts",
        mode: str = "merge",
    ) -> SectionContent:
        """Stage one section's knowledge map and copy its files into the draft.

        ``mode="merge"`` (the default) shallow-merges over what this section
        already staged and appends to its file list, so an agent that reports
        incrementally does not silently drop its earlier calls. ``"replace"``
        discards the staged map first; staged files always survive because
        they may already be referenced by the map being written.
        """
        name = _checked_section(section)
        if mode not in ("merge", "replace"):
            raise KBStoreError(f"mode must be 'merge' or 'replace', got {mode!r}")
        if not isinstance(knowledge, dict):
            raise KBStoreError(f"knowledge for section {name!r} must be a dict")
        try:
            json.dumps(knowledge, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise KBStoreError(f"section {name!r} is not strict JSON: {exc}") from exc

        staged = self.staged(name)
        merged = dict(knowledge)
        refs: list[str] = []
        if staged is not None:
            refs = [path.relative_to(self.files_dir).as_posix() for path in staged.files]
            if mode == "merge":
                merged = {**staged.knowledge, **knowledge}

        added = [self._copy_in(name, source, kind) for source in files]
        for ref in added:
            if ref and ref not in refs:
                refs.append(ref)

        target = self.root / _SECTIONS_MEMBER / f"{name}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {"knowledge": merged, "files": refs}
        target.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return SectionContent(name, merged, [self.files_dir / ref for ref in refs])

    def _copy_in(self, section: str, source: str | Path, kind: str) -> str:
        raw = str(source or "").strip()
        if not raw:
            return ""
        src = Path(raw)
        if src.is_symlink():
            raise KBStoreError(f"artifact must not be a symlink: {src}")
        if not src.is_file():
            raise KBStoreError(f"artifact is not a readable file: {src}")
        safe_kind = _checked_section(kind)
        rel = f"{section}/{safe_kind}/{src.name}"
        destination = self.files_dir / rel
        if destination.exists() and not _same_bytes(src, destination):
            digest = hashlib.sha256(str(src.resolve()).encode()).hexdigest()[:10]
            rel = f"{section}/{safe_kind}/{src.stem}-{digest}{src.suffix}"
            destination = self.files_dir / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(src.read_bytes())
        return rel

    # -- read ----------------------------------------------------------------

    def staged(self, section: str) -> SectionContent | None:
        """Read back what this draft already holds for ``section``."""
        name = _checked_section(section)
        target = self.root / _SECTIONS_MEMBER / f"{name}.json"
        if not target.is_file():
            return None
        try:
            payload = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise KBStoreError(f"staged section {name!r} is unreadable: {exc}") from exc
        knowledge = payload.get("knowledge")
        refs = payload.get("files") or []
        return SectionContent(
            name,
            dict(knowledge) if isinstance(knowledge, dict) else {},
            [self.files_dir / str(ref) for ref in refs if str(ref).strip()],
        )

    def read(self, section: str) -> SectionContent | None:
        """Return ``section`` from the warm-start record, or ``None``.

        ``None`` means this run has no prior knowledge for the section: either
        nothing was downloaded, or the record predates the section. Callers
        should treat it as a cold start rather than an error.
        """
        name = _checked_section(section)
        if self.warm_start_dir is None:
            return None
        recipe = self.warm_start_dir / _RECIPE_MEMBER
        if not recipe.is_file():
            return None
        try:
            document = json.loads(recipe.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise KBStoreError(f"warm start record is unreadable: {exc}") from exc
        value = document.get(SECTION_ROOT)
        knowledge = (value or {}).get(name) if isinstance(value, dict) else None
        if not isinstance(knowledge, dict):
            return None
        root = self.warm_start_dir / FILES_MEMBER_ROOT / name
        files = sorted(path for path in root.rglob("*") if path.is_file()) if root.is_dir() else []
        return SectionContent(name, dict(knowledge), files)

    def sections(self) -> list[str]:
        """Every section staged in this draft, in a stable order."""
        root = self.root / _SECTIONS_MEMBER
        if not root.is_dir():
            return []
        return sorted(path.stem for path in root.glob("*.json"))

    def document(self) -> dict[str, Any]:
        """The staged ``{section: knowledge}`` map to publish under ``value``."""
        return {name: (self.staged(name) or SectionContent(name, {})).knowledge for name in self.sections()}


def _same_bytes(left: Path, right: Path) -> bool:
    try:
        return left.read_bytes() == right.read_bytes()
    except OSError:
        return False
