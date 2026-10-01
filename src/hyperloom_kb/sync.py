# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Push this service's own Experiences to a global Experience KB and pull the global KB's into this service."""

from __future__ import annotations

import sqlite3
import threading
from collections import Counter
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Protocol

from hyperloom_kb.remote import RemoteClient, RemoteClientError, RemoteConfig
from hyperloom_kb.schema import Experience, ExperienceDeclaration, JsonValue
from hyperloom_kb.storage import StorageContractError

GLOBAL_URL_ENV = "HYPERLOOM_GLOBAL_KB_URL"
GLOBAL_TOKEN_ENV = "HYPERLOOM_GLOBAL_KB_TOKEN"
SYNC_BATCH = 100
_PUSH = "push"
_PULL = "pull"


class SyncUnavailable(RuntimeError):
    """Raised when this service has no global KB to sync with."""


def global_config_from_env(env: Mapping[str, str]) -> RemoteConfig | None:
    url = str(env.get(GLOBAL_URL_ENV) or "").strip()
    if not url:
        return None
    token = str(env.get(GLOBAL_TOKEN_ENV) or "")
    if not token:
        raise SyncUnavailable(f"{GLOBAL_TOKEN_ENV} must be set with {GLOBAL_URL_ENV}")
    return RemoteConfig(url, token)


class SyncLedger:
    """Per-global-KB push and pull cursors, the Experiences pulled here, which are never pushed back, and the ones
    a push held back because reads here did not see them."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS cursors (
                    global_url TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    position INTEGER NOT NULL,
                    PRIMARY KEY (global_url, direction)
                )
                """
            )
            connection.execute("CREATE TABLE IF NOT EXISTS pulled (experience_id TEXT PRIMARY KEY)")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS held_back (
                    global_url TEXT NOT NULL,
                    experience_id TEXT NOT NULL,
                    PRIMARY KEY (global_url, experience_id)
                )
                """
            )

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30)
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def cursor(self, global_url: str, direction: str) -> int:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT position FROM cursors WHERE global_url = ? AND direction = ?",
                (global_url, direction),
            ).fetchone()
        return int(row[0]) if row else 0

    def advance(self, global_url: str, direction: str, position: int) -> None:
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO cursors(global_url, direction, position) VALUES (?, ?, ?)
                ON CONFLICT(global_url, direction) DO UPDATE SET position = excluded.position
                """,
                (global_url, direction, position),
            )

    def mark_pulled(self, experience_id: str) -> None:
        with self._connection() as connection:
            connection.execute("INSERT OR IGNORE INTO pulled(experience_id) VALUES (?)", (experience_id,))

    def pulled(self, experience_id: str) -> bool:
        with self._connection() as connection:
            row = connection.execute("SELECT 1 FROM pulled WHERE experience_id = ?", (experience_id,)).fetchone()
        return row is not None

    def hold_back(self, global_url: str, experience_id: str) -> None:
        with self._connection() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO held_back(global_url, experience_id) VALUES (?, ?)", (global_url, experience_id)
            )

    def release(self, global_url: str, experience_id: str) -> None:
        with self._connection() as connection:
            connection.execute(
                "DELETE FROM held_back WHERE global_url = ? AND experience_id = ?", (global_url, experience_id)
            )

    def held_back(self, global_url: str) -> tuple[str, ...]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT experience_id FROM held_back WHERE global_url = ? ORDER BY experience_id", (global_url,)
            ).fetchall()
        return tuple(str(row[0]) for row in rows)


def _report(
    status: str,
    global_url: str,
    counts: Counter[str] | None = None,
    rejected: list[JsonValue] | None = None,
    *,
    has_more: bool = False,
    error: str = "",
) -> dict[str, JsonValue]:
    counts = counts or Counter()
    report: dict[str, JsonValue] = {
        "status": status,
        "global_url": global_url,
        "created": counts["created"],
        "unchanged": counts["unchanged"],
        "skipped": counts["skipped"],
        "held_back": counts["held_back"],
        "rejected": rejected or [],
        "has_more": has_more,
    }
    if error:
        report["error"] = error
    return report


def _record_id(item: Mapping[str, Any]) -> str:
    experience = item.get("experience")
    return str(experience.get("id") or "") if isinstance(experience, Mapping) else ""


class SyncedService(Protocol):
    """What sync needs of the Experience service it pushes from and pulls into."""

    @property
    def schema_refs(self) -> tuple[str, ...]:
        """Every schema this service holds."""

    def declaration_for(self, schema_ref: str) -> ExperienceDeclaration:
        """The declaration registered for ``schema_ref``."""

    def write(self, experience: Experience) -> dict[str, JsonValue]:
        """Store one Experience; the result carries its ``status``."""

    def records_after(self, after: int, limit: int) -> tuple[tuple[tuple[int, Experience], ...], int, bool]:
        """Up to ``limit`` records written after sequence ``after``, the next cursor, and whether more remain."""

    def held(self, experience_id: str) -> Experience:
        """The stored Experience ``experience_id``."""

    def is_visible(self, experience: Experience) -> bool:
        """Whether reads on this service see ``experience``."""


class GlobalSync:
    """One service's sync with its global KB; one push or pull batch runs at a time."""

    def __init__(self, service: SyncedService, ledger: SyncLedger, target: RemoteClient | None) -> None:
        self._service = service
        self._ledger = ledger
        self._target = target
        self._lock = threading.Lock()

    def _connected(self) -> RemoteClient:
        if self._target is None:
            raise SyncUnavailable(
                f"this service was started without a global Experience KB ({GLOBAL_URL_ENV}, {GLOBAL_TOKEN_ENV})"
            )
        self._target.health()
        return self._target

    def _send(
        self, target: RemoteClient, experience: Experience, counts: Counter[str], rejected: list[JsonValue]
    ) -> str:
        """Write one Experience to the global KB; returns the error that should stop this push, if any."""

        declaration = self._service.declaration_for(experience.schema_ref)
        try:
            counts[target.write(experience, declaration=declaration).status] += 1
        except RemoteClientError as exc:
            if exc.retryable:
                return str(exc)
            rejected.append({"experience_id": experience.id, "detail": str(exc)})
        return ""

    def push(self) -> dict[str, JsonValue]:
        """Send the Experiences written here that reads here see; one they do not see waits until they do."""

        with self._lock:
            url = self._target.config.base_url if self._target is not None else ""
            try:
                target = self._connected()
            except RemoteClientError as exc:
                return _report("incomplete", url, error=str(exc))
            counts: Counter[str] = Counter()
            rejected: list[JsonValue] = []
            for experience_id in self._ledger.held_back(url):
                experience = self._service.held(experience_id)
                if not self._service.is_visible(experience):
                    continue
                error = self._send(target, experience, counts, rejected)
                if error:
                    return _report("incomplete", url, counts, rejected, error=error)
                self._ledger.release(url, experience_id)
            position = self._ledger.cursor(url, _PUSH)
            records, _, has_more = self._service.records_after(position, SYNC_BATCH)
            for sequence, experience in records:
                if self._ledger.pulled(experience.id):
                    counts["skipped"] += 1
                elif not self._service.is_visible(experience):
                    self._ledger.hold_back(url, experience.id)
                    counts["held_back"] += 1
                else:
                    error = self._send(target, experience, counts, rejected)
                    if error:
                        self._ledger.advance(url, _PUSH, position)
                        return _report("incomplete", url, counts, rejected, error=error)
                position = sequence
            self._ledger.advance(url, _PUSH, position)
            return _report("completed", url, counts, rejected, has_more=has_more)

    def pull(self) -> dict[str, JsonValue]:
        """Pull the global KB's Experiences of every schema this service holds; other schemas stay on the global KB."""

        with self._lock:
            url = self._target.config.base_url if self._target is not None else ""
            try:
                target = self._connected()
            except RemoteClientError as exc:
                return _report("incomplete", url, error=str(exc))
            counts: Counter[str] = Counter()
            rejected: list[JsonValue] = []
            has_more = False
            for schema_ref in self._service.schema_refs:
                direction = f"{_PULL} {schema_ref}"
                try:
                    page = target.export_page(
                        after=self._ledger.cursor(url, direction), limit=SYNC_BATCH, schema_ref=schema_ref
                    )
                except RemoteClientError as exc:
                    return _report("incomplete", url, counts, rejected, error=str(exc))
                for item in page.items:
                    try:
                        experience = Experience.from_dict(item.get("experience"))
                        status = str(self._service.write(experience)["status"])
                    except (KeyError, TypeError, ValueError, StorageContractError) as exc:
                        rejected.append({"experience_id": _record_id(item), "detail": str(exc)})
                        continue
                    if status == "created":
                        self._ledger.mark_pulled(experience.id)
                    counts[status] += 1
                self._ledger.advance(url, direction, page.next_cursor)
                has_more = has_more or page.has_more
            return _report("completed", url, counts, rejected, has_more=has_more)


__all__ = [
    "GLOBAL_TOKEN_ENV",
    "GLOBAL_URL_ENV",
    "SYNC_BATCH",
    "GlobalSync",
    "SyncLedger",
    "SyncUnavailable",
    "SyncedService",
    "global_config_from_env",
]
