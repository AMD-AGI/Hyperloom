# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Durable in-process filesystem backend for Hyperloom's knowledge graph.

``LocalGraphStore`` is a GraphStore backend adapter, not an MCP server.  Its
``call(tool, arguments)`` method intentionally matches the small duck-typed
surface consumed by :class:`recipe_kb.kg_client.KGClient`; every operation is
performed in-process with filesystem I/O only.

The store root contains::

    pages/<slug>.md
    edges/outbound/<slug>.json
    edges/inbound/<slug>.json
    .lock

Slash-separated slugs are represented as nested paths after strict validation.
Edge index updates are serialized by one module-level thread lock per root and
one POSIX ``flock``.  A durable intent journal makes the two edge-index
replacements recoverable as one logical transaction after interruption.
"""

from __future__ import annotations

import json
import os
import re
import stat
import tempfile
import threading
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any, Iterator, Mapping

try:
    import fcntl
except ImportError:  # pragma: no cover - Linux production path
    fcntl = None  # type: ignore[assignment]

from ._path_safety import SLUG_PART_RE as _SLUG_PART_RE
from ._path_safety import assert_within_root as _assert_within_root
from ._path_safety import validated_slug as _validated_slug

_ROOT_LOCKS_GUARD = threading.Lock()
_ROOT_LOCKS: dict[str, threading.RLock] = {}


class LocalGraphStoreError(OSError):
    """The local graph store could not complete a durable operation."""


def _thread_lock_for(root: Path) -> threading.RLock:
    """Return the process-wide lock shared by all instances for *root*."""

    key = os.path.normcase(str(root.resolve()))
    with _ROOT_LOCKS_GUARD:
        lock = _ROOT_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _ROOT_LOCKS[key] = lock
        return lock



def _edge_key(edge: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(edge.get("from_slug") or ""),
        str(edge.get("to_slug") or ""),
        str(edge.get("link_type") or ""),
    )


class LocalGraphStore:
    """In-process filesystem GraphStore backend used by ``KGClient``.

    ``root`` is the graph root itself.  The environment factory supplies
    ``$KNOWLEDGE_LOCAL_ROOT/hyperloom/kg``.
    """

    def __init__(self, root: str | os.PathLike[str]) -> None:
        if fcntl is None:
            raise LocalGraphStoreError("LocalGraphStore requires POSIX fcntl file locking")
        self.root = Path(root).expanduser().resolve()
        self.pages_root = self.root / "pages"
        self.outbound_root = self.root / "edges" / "outbound"
        self.inbound_root = self.root / "edges" / "inbound"
        self._lock_path = self.root / ".lock"
        self._transaction_path = self.root / ".edge-transaction.json"
        self._thread_lock = _thread_lock_for(self.root)
        self._ensure_layout()
        with self._locked():
            pass

    def call(self, tool: str, arguments: Mapping[str, Any] | None = None) -> Any:
        """Dispatch the compatibility surface expected by ``KGClient``.

        This is a local backend adapter call, not an RPC or MCP invocation.
        """

        args = dict(arguments or {})
        handlers = {
            "list_pages": self._list_pages,
            "get_page": self._get_page,
            "put_page": self._put_page,
            "add_link": self._add_link,
            "get_links": self._get_links,
            "get_backlinks": self._get_backlinks,
            "traverse_graph": self._traverse_graph,
            "search": self._search,
        }
        handler = handlers.get(str(tool))
        if handler is None:
            raise ValueError(f"unsupported local graph tool: {tool!r}")
        return handler(args)

    def _ensure_layout(self) -> None:
        for path in (self.pages_root, self.outbound_root, self.inbound_root):
            path.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def _locked(self, *, exclusive: bool = True) -> Iterator[None]:
        """Hold the root lock, sharing it for stable read-only operations."""

        with self._thread_lock:
            flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
            fd: int | None = None
            try:
                fd = os.open(self._lock_path, flags, 0o600)
                if not stat.S_ISREG(os.fstat(fd).st_mode):
                    raise LocalGraphStoreError("local graph lock target is not a regular file")
                fcntl.flock(fd, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
                if exclusive:
                    self._recover_transaction_unlocked()
                elif self._transaction_path.exists():
                    # Upgrade only when an interrupted writer left a journal.
                    # Releasing SH before EX avoids a flock upgrade deadlock.
                    fcntl.flock(fd, fcntl.LOCK_UN)
                    fcntl.flock(fd, fcntl.LOCK_EX)
                    self._recover_transaction_unlocked()
                yield
            except LocalGraphStoreError:
                raise
            except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
                raise LocalGraphStoreError(f"local graph operation failed: {exc}") from exc
            finally:
                if fd is not None:
                    with suppress(OSError):
                        fcntl.flock(fd, fcntl.LOCK_UN)
                    with suppress(OSError):
                        os.close(fd)

    def _path_for(self, base: Path, slug: str, suffix: str) -> Path:
        parts = _validated_slug(slug).split("/")
        path = base.joinpath(*parts[:-1], parts[-1] + suffix)
        try:
            _assert_within_root(path, base)
        except ValueError as exc:
            raise ValueError(f"unsafe graph slug path: {slug!r}") from exc
        return path

    def _page_path(self, slug: str) -> Path:
        return self._path_for(self.pages_root, slug, ".md")

    def _edge_path(self, direction: str, slug: str) -> Path:
        base = self.outbound_root if direction == "outbound" else self.inbound_root
        return self._path_for(base, slug, ".json")

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
        fd = os.open(path, flags)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def _atomic_write(self, path: Path, content: str) -> None:
        """Atomically replace one file after syncing data and its directory."""

        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.chmod(temporary, 0o600)
            os.replace(temporary, path)
            self._fsync_directory(path.parent)
        except BaseException:
            with suppress(OSError):
                os.unlink(temporary)
            raise

    @staticmethod
    def _json_text(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"

    def _relative_transaction_path(self, path: Path) -> str:
        try:
            return path.relative_to(self.root).as_posix()
        except ValueError as exc:
            raise LocalGraphStoreError("transaction target escaped local graph root") from exc

    def _transaction_target(self, relative: Any) -> Path:
        if not isinstance(relative, str) or not relative or relative.startswith("/"):
            raise LocalGraphStoreError("invalid local graph transaction target")
        candidate = self.root.joinpath(*relative.split("/"))
        try:
            _assert_within_root(candidate, self.root)
        except ValueError as exc:
            raise LocalGraphStoreError("local graph transaction target escaped root") from exc
        allowed = (
            self.outbound_root.resolve(),
            self.inbound_root.resolve(),
        )
        resolved = candidate.resolve(strict=False)
        if not any(resolved == base or base in resolved.parents for base in allowed):
            raise LocalGraphStoreError("local graph transaction target is outside edge indexes")
        return candidate

    def _commit_edge_indexes_unlocked(self, writes: list[tuple[Path, str]]) -> None:
        previous = [(path, path.read_text(encoding="utf-8") if path.exists() else None) for path, _ in writes]
        journal = {
            "version": 1,
            "writes": [{"path": self._relative_transaction_path(path), "content": content} for path, content in writes],
        }
        self._atomic_write(self._transaction_path, self._json_text(journal))
        try:
            self._recover_transaction_unlocked()
        except Exception as commit_error:
            # A normal I/O failure rolls the visible indexes back before the
            # lock is released. A hard process interruption bypasses this
            # handler, leaving the durable journal for next-open recovery.
            try:
                for path, content in previous:
                    if content is None:
                        if path.exists():
                            path.unlink()
                            self._fsync_directory(path.parent)
                    else:
                        self._atomic_write(path, content)
                if self._transaction_path.exists():
                    self._transaction_path.unlink()
                    self._fsync_directory(self.root)
            except Exception:
                # Keep the journal when rollback itself cannot complete; the
                # next locked operation deterministically commits both writes.
                raise commit_error
            raise

    def _recover_transaction_unlocked(self) -> None:
        if not self._transaction_path.exists():
            return
        try:
            journal = json.loads(self._transaction_path.read_text(encoding="utf-8"))
            if journal.get("version") != 1 or not isinstance(journal.get("writes"), list):
                raise LocalGraphStoreError("invalid local graph transaction journal")
            writes: list[tuple[Path, str]] = []
            for write in journal["writes"]:
                if not isinstance(write, dict) or not isinstance(write.get("content"), str):
                    raise LocalGraphStoreError("invalid local graph transaction write")
                writes.append((self._transaction_target(write.get("path")), write["content"]))
            for path, content in writes:
                self._atomic_write(path, content)
            self._transaction_path.unlink()
            self._fsync_directory(self.root)
        except LocalGraphStoreError:
            raise
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            raise LocalGraphStoreError(f"could not recover local graph transaction: {exc}") from exc

    def _read_edges_unlocked(self, direction: str, slug: str) -> list[dict[str, Any]]:
        path = self._edge_path(direction, slug)
        if not path.exists():
            return []
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, list) or any(not isinstance(edge, dict) for edge in value):
            raise LocalGraphStoreError(f"invalid {direction} edge index for {slug!r}")
        return [dict(edge) for edge in value]

    def _page_exists_unlocked(self, slug: str) -> bool:
        return self._page_path(slug).is_file()

    def _all_page_slugs_unlocked(self) -> list[str]:
        """List sorted page slugs without reading page bodies."""

        slugs: list[str] = []
        for path in self.pages_root.rglob("*.md"):
            if path.is_symlink() or not path.is_file():
                continue
            relative = path.relative_to(self.pages_root).as_posix()
            slug = relative[:-3]
            _validated_slug(slug)
            slugs.append(slug)
        slugs.sort()
        return slugs

    def _all_pages_unlocked(self) -> list[tuple[str, str]]:
        pages = [
            (slug, self._page_path(slug).read_text(encoding="utf-8"))
            for slug in self._all_page_slugs_unlocked()
        ]
        return pages

    def _list_pages(self, args: Mapping[str, Any]) -> list[dict[str, Any]]:
        limit = max(0, int(args.get("limit", 100)))
        offset = max(0, int(args.get("offset", 0)))
        with self._locked(exclusive=False):
            slugs = self._all_page_slugs_unlocked()[offset : offset + limit]
        return [{"slug": slug} for slug in slugs]

    def _get_page(self, args: Mapping[str, Any]) -> dict[str, Any]:
        slug = _validated_slug(args.get("slug"))
        with self._locked(exclusive=False):
            path = self._page_path(slug)
            if not path.is_file() or path.is_symlink():
                return {"error": "page_not_found", "slug": slug}
            content = path.read_text(encoding="utf-8")
        return {"slug": slug, "content": content, "body": content}

    def _put_page(self, args: Mapping[str, Any]) -> dict[str, Any]:
        slug = _validated_slug(args.get("slug"))
        content = args.get("content", args.get("body"))
        if not isinstance(content, str):
            raise ValueError("put_page requires string content")
        with self._locked():
            created = not self._page_exists_unlocked(slug)
            self._atomic_write(self._page_path(slug), content)
        return {"slug": slug, "status": "created" if created else "updated"}

    def _add_link(self, args: Mapping[str, Any]) -> dict[str, Any]:
        from_slug = _validated_slug(args.get("from", args.get("from_slug")))
        to_slug = _validated_slug(args.get("to", args.get("to_slug")))
        link_type = str(args.get("link_type") or "").strip()
        if not link_type:
            raise ValueError("add_link requires a non-empty link_type")
        context = args.get("context", "")
        edge = {
            "from_slug": from_slug,
            "to_slug": to_slug,
            "link_type": link_type,
            "context": context,
        }
        key = _edge_key(edge)
        with self._locked():
            if not self._page_exists_unlocked(from_slug) or not self._page_exists_unlocked(to_slug):
                return {"error": "page_not_found", "message": "add_link requires both endpoint pages"}
            outbound = self._read_edges_unlocked("outbound", from_slug)
            inbound = self._read_edges_unlocked("inbound", to_slug)
            outbound = [existing for existing in outbound if _edge_key(existing) != key]
            inbound = [existing for existing in inbound if _edge_key(existing) != key]
            outbound.append(edge)
            inbound.append(edge)
            outbound.sort(key=_edge_key)
            inbound.sort(key=_edge_key)
            self._commit_edge_indexes_unlocked(
                [
                    (self._edge_path("outbound", from_slug), self._json_text(outbound)),
                    (self._edge_path("inbound", to_slug), self._json_text(inbound)),
                ]
            )
        return {"status": "ok"}

    def _get_links(self, args: Mapping[str, Any]) -> list[dict[str, Any]]:
        slug = _validated_slug(args.get("slug"))
        link_type = str(args.get("link_type") or "").strip()
        with self._locked(exclusive=False):
            edges = self._read_edges_unlocked("outbound", slug)
        return [edge for edge in edges if not link_type or edge.get("link_type") == link_type]

    def _get_backlinks(self, args: Mapping[str, Any]) -> list[dict[str, Any]]:
        slug = _validated_slug(args.get("slug"))
        link_type = str(args.get("link_type") or "").strip()
        with self._locked(exclusive=False):
            edges = self._read_edges_unlocked("inbound", slug)
        return [edge for edge in edges if not link_type or edge.get("link_type") == link_type]

    def _traverse_graph(self, args: Mapping[str, Any]) -> list[dict[str, Any]]:
        start = _validated_slug(args.get("slug"))
        depth = max(0, min(int(args.get("depth", 5)), 20))
        direction = str(args.get("direction") or "out").strip().lower()
        if direction not in {"out", "in", "both"}:
            raise ValueError("traverse_graph direction must be 'out', 'in', or 'both'")
        link_type = str(args.get("link_type") or "").strip()
        with self._locked(exclusive=False):
            frontier = {start}
            visited = {start}
            seen_edges: set[tuple[str, str, str]] = set()
            result: list[dict[str, Any]] = []
            for hop in range(1, depth + 1):
                next_frontier: set[str] = set()
                for slug in sorted(frontier):
                    candidates: list[tuple[dict[str, Any], str]] = []
                    if direction in {"out", "both"}:
                        candidates.extend(
                            (edge, str(edge.get("to_slug") or ""))
                            for edge in self._read_edges_unlocked("outbound", slug)
                        )
                    if direction in {"in", "both"}:
                        candidates.extend(
                            (edge, str(edge.get("from_slug") or ""))
                            for edge in self._read_edges_unlocked("inbound", slug)
                        )
                    for edge, neighbour in candidates:
                        key = _edge_key(edge)
                        if (link_type and edge.get("link_type") != link_type) or key in seen_edges:
                            continue
                        seen_edges.add(key)
                        result.append({**edge, "depth": hop})
                        if neighbour and neighbour not in visited:
                            next_frontier.add(neighbour)
                if not next_frontier:
                    break
                visited.update(next_frontier)
                frontier = next_frontier
        return result

    def _search(self, args: Mapping[str, Any]) -> list[dict[str, Any]]:
        query = str(args.get("query") or "").casefold()
        limit = max(0, int(args.get("limit", 100)))
        with self._locked(exclusive=False):
            matches = [
                {"slug": slug, "body": content}
                for slug, content in self._all_pages_unlocked()
                if not query or query in slug.casefold() or query in content.casefold()
            ][:limit]
        return matches


__all__ = ["LocalGraphStore", "LocalGraphStoreError"]
