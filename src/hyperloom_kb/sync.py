# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Push this service's own Experiences and its Distills' latest versions to a global Experience KB, and pull one
schema of the global KB's, Experiences and Distills, into it."""

from __future__ import annotations

from collections import Counter
from collections.abc import Collection, Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, BinaryIO, Protocol

from hyperloom_kb.database import Database, WriteSource
from hyperloom_kb.distill.model import DistillEntry, FormDeclaration
from hyperloom_kb.local_state import UnknownStateItem
from hyperloom_kb.remote import ExportPage, RemoteClient, RemoteClientError, RemoteConfig
from hyperloom_kb.schema import Experience, ExperienceDeclaration, FileRef, JsonValue
from hyperloom_kb.storage import StorageContractError

GLOBAL_URL_ENV = "HYPERLOOM_GLOBAL_KB_URL"
GLOBAL_TOKEN_ENV = "HYPERLOOM_GLOBAL_KB_TOKEN"
SYNC_BATCH = 100
#: What a sync report counts of the Distill versions it handled: a pull ``skipped`` the ones older than the version
#: here, and a push ``held_back`` the ones a global KB older than Distills cannot take yet.
DISTILL_COUNTS = ("created", "unchanged", "skipped", "held_back")
_PUSH = "push"
_PULL = "pull"
_PER_GLOBAL_TABLES = (
    "sync_identities",
    "sync_cursors",
    "sync_held_back",
    "sync_on_global",
    "sync_pulls_in_progress",
    "sync_pulled_states",
    "sync_distills_on_global",
)


class SyncUnavailable(RuntimeError):
    """Raised when this service has no global KB to sync with."""


class SyncRefused(RuntimeError):
    """Raised when the global KB is not the one this service synced with, or names no identity."""


def global_config_from_env(env: Mapping[str, str]) -> RemoteConfig | None:
    url = str(env.get(GLOBAL_URL_ENV) or "").strip()
    if not url:
        return None
    token = str(env.get(GLOBAL_TOKEN_ENV) or "")
    if not token:
        raise SyncUnavailable(f"{GLOBAL_TOKEN_ENV} must be set with {GLOBAL_URL_ENV}")
    return RemoteConfig(url, token)


class SyncLedger:
    """What one KB knows of each global KB it syncs with: the KB's identity, push and pull cursors, the Experiences
    pulled here, which are never pushed back, the ones a push held back because reads here did not see them, every
    Experience known to be on that global KB, and the state of each schema there as a pull last saw it."""

    def __init__(self, database: Database, kb_id: str) -> None:
        self._database = database
        self.kb_id = kb_id

    @contextmanager
    def exclusive(self) -> Iterator[None]:
        """Hold this KB's sync against every other push or pull of it, in any process sharing the database."""

        with self._database.exclusive(f"hyperloom-kb:sync:{self.kb_id}"):
            yield

    def _execute(self, query: str, *args: Any) -> None:
        with self._database.transaction() as connection:
            connection.execute(query, (self.kb_id, *args))

    def _rows(self, query: str, *args: Any) -> list[dict[str, Any]]:
        with self._database.transaction() as connection:
            return connection.execute(query, (self.kb_id, *args)).fetchall()

    def cursor(self, global_url: str, direction: str) -> tuple[int, str]:
        """The position a push or pull resumes after, and the Experience a pull found there; ``(0, "")`` before."""

        rows = self._rows(
            "SELECT position, anchor FROM sync_cursors WHERE kb_id = %s AND global_url = %s AND direction = %s",
            global_url,
            direction,
        )
        return (int(rows[0]["position"]), str(rows[0]["anchor"])) if rows else (0, "")

    def advance(self, global_url: str, direction: str, position: int, anchor: str = "") -> None:
        self._execute(
            """
            INSERT INTO sync_cursors(kb_id, global_url, direction, position, anchor) VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (kb_id, global_url, direction) DO UPDATE SET position = excluded.position,
                anchor = excluded.anchor
            """,
            global_url,
            direction,
            position,
            anchor,
        )

    def mark_pulled(self, experience_id: str) -> None:
        self._execute(
            "INSERT INTO sync_pulled(kb_id, experience_id) VALUES (%s, %s) ON CONFLICT DO NOTHING", experience_id
        )

    def pulled(self, experience_id: str) -> bool:
        return bool(self._rows("SELECT 1 FROM sync_pulled WHERE kb_id = %s AND experience_id = %s", experience_id))

    def hold_back(self, global_url: str, experience_id: str) -> None:
        self._execute(
            "INSERT INTO sync_held_back(kb_id, global_url, experience_id) VALUES (%s, %s, %s) ON CONFLICT DO NOTHING",
            global_url,
            experience_id,
        )

    def release(self, global_url: str, experience_id: str) -> None:
        self._execute(
            "DELETE FROM sync_held_back WHERE kb_id = %s AND global_url = %s AND experience_id = %s",
            global_url,
            experience_id,
        )

    def held_back(self, global_url: str) -> tuple[str, ...]:
        rows = self._rows(
            "SELECT experience_id FROM sync_held_back WHERE kb_id = %s AND global_url = %s ORDER BY experience_id",
            global_url,
        )
        return tuple(str(row["experience_id"]) for row in rows)

    def identity(self, global_url: str) -> str:
        """The identity of the global KB first synced with at ``global_url``; empty before the first sync."""

        rows = self._rows("SELECT remote_kb_id FROM sync_identities WHERE kb_id = %s AND global_url = %s", global_url)
        return str(rows[0]["remote_kb_id"]) if rows else ""

    def bind(self, global_url: str, kb_id: str) -> None:
        self._execute(
            "INSERT INTO sync_identities(kb_id, global_url, remote_kb_id) VALUES (%s, %s, %s) ON CONFLICT DO NOTHING",
            global_url,
            kb_id,
        )

    def forget(self, global_url: str) -> None:
        """Drop all this KB knows of the global KB at ``global_url``; what it pulled from there stays pulled."""

        with self._database.transaction() as connection:
            for table in _PER_GLOBAL_TABLES:
                connection.execute(
                    f"DELETE FROM {table} WHERE kb_id = %s AND global_url = %s", (self.kb_id, global_url)
                )

    def note_on_global(self, global_url: str, schema_ref: str, experience_ids: Collection[str]) -> None:
        with self._database.transaction() as connection:
            connection.executemany(
                """
                INSERT INTO sync_on_global(kb_id, global_url, schema_ref, experience_id) VALUES (%s, %s, %s, %s)
                ON CONFLICT DO NOTHING
                """,
                [(self.kb_id, global_url, schema_ref, experience_id) for experience_id in experience_ids],
            )

    def on_global(self, global_url: str, schema_ref: str) -> frozenset[str]:
        rows = self._rows(
            "SELECT experience_id FROM sync_on_global WHERE kb_id = %s AND global_url = %s AND schema_ref = %s",
            global_url,
            schema_ref,
        )
        return frozenset(str(row["experience_id"]) for row in rows)

    def note_distill(self, global_url: str, schema_ref: str, entry_id: str, version: int) -> None:
        """Remember that the global KB holds ``version`` of ``entry_id``, or refused it from here, so a push stops
        sending it."""

        self._execute(
            """
            INSERT INTO sync_distills_on_global(kb_id, global_url, schema_ref, entry_id, version)
            VALUES (%s, %s, %s, %s, %s) ON CONFLICT DO NOTHING
            """,
            global_url,
            schema_ref,
            entry_id,
            version,
        )

    def distills_on_global(self, global_url: str, schema_ref: str) -> frozenset[tuple[str, int]]:
        rows = self._rows(
            """
            SELECT entry_id, version FROM sync_distills_on_global
            WHERE kb_id = %s AND global_url = %s AND schema_ref = %s
            """,
            global_url,
            schema_ref,
        )
        return frozenset((str(row["entry_id"]), int(row["version"])) for row in rows)

    def pulled_state(self, global_url: str, schema_ref: str) -> str:
        """The state of ``schema_ref`` on the global KB as the last pull batch saw it; empty before the first."""

        rows = self._rows(
            "SELECT state FROM sync_pulled_states WHERE kb_id = %s AND global_url = %s AND schema_ref = %s",
            global_url,
            schema_ref,
        )
        return str(rows[0]["state"]) if rows else ""

    def note_pulled_state(self, global_url: str, schema_ref: str, state: str) -> None:
        self._execute(
            """
            INSERT INTO sync_pulled_states(kb_id, global_url, schema_ref, state) VALUES (%s, %s, %s, %s)
            ON CONFLICT (kb_id, global_url, schema_ref) DO UPDATE SET state = excluded.state
            """,
            global_url,
            schema_ref,
            state,
        )

    def pull_in_progress(self, global_url: str, schema_ref: str) -> bool:
        return bool(
            self._rows(
                "SELECT 1 FROM sync_pulls_in_progress WHERE kb_id = %s AND global_url = %s AND schema_ref = %s",
                global_url,
                schema_ref,
            )
        )

    def mark_pull_in_progress(self, global_url: str, schema_ref: str, in_progress: bool) -> None:
        if in_progress:
            self._execute(
                """
                INSERT INTO sync_pulls_in_progress(kb_id, global_url, schema_ref) VALUES (%s, %s, %s)
                ON CONFLICT DO NOTHING
                """,
                global_url,
                schema_ref,
            )
        else:
            self._execute(
                "DELETE FROM sync_pulls_in_progress WHERE kb_id = %s AND global_url = %s AND schema_ref = %s",
                global_url,
                schema_ref,
            )


def _report(
    status: str,
    global_url: str,
    counts: Counter[str] | None = None,
    rejected: list[JsonValue] | None = None,
    *,
    distills: Counter[str] | None = None,
    has_more: bool = False,
    error: str = "",
) -> dict[str, JsonValue]:
    counts = counts or Counter()
    distills = distills or Counter()
    report: dict[str, JsonValue] = {
        "status": status,
        "global_url": global_url,
        "created": counts["created"],
        "unchanged": counts["unchanged"],
        "skipped": counts["skipped"],
        "held_back": counts["held_back"],
        "distills": {key: distills[key] for key in DISTILL_COUNTS},
        "rejected": rejected or [],
        "has_more": has_more,
    }
    if error:
        report["error"] = error
    return report


def _record_id(item: Mapping[str, Any]) -> str:
    experience = item.get("experience")
    return str(experience.get("id") or "") if isinstance(experience, Mapping) else ""


class SyncedDistills(Protocol):
    """What sync needs of the Distills of the service it pushes from and pulls into."""

    def pushable(self, schema_ref: str) -> list[tuple[str, int]]:
        """Each visible Distill of ``schema_ref`` that shows its latest version, with that version."""

    def version(self, entry_id: str, version: int) -> tuple[DistillEntry, FormDeclaration]:
        """A stored version and the form it is written in."""

    def latest_version(self, entry_id: str) -> int:
        """The highest version stored of ``entry_id``; 0 for one not held."""

    def receive(self, entry: DistillEntry, form: FormDeclaration, *, source: WriteSource) -> dict[str, JsonValue]:
        """Store a version as another KB wrote it; the result carries its ``status``."""

    def show_at_least(self, schema_ref: str, entry_id: str, version: int) -> None:
        """Make ``entry_id`` show ``version`` when it shows an older one."""


class SyncedService(Protocol):
    """What sync needs of the Experience service it pushes from and pulls into."""

    kb_id: str
    name: str
    distill: SyncedDistills

    @property
    def schema_refs(self) -> tuple[str, ...]:
        """Every schema the service holds."""

    def register(self, declaration: ExperienceDeclaration) -> None:
        """Hold ``declaration``'s schema."""

    def declaration_for(self, schema_ref: str) -> ExperienceDeclaration:
        """The declaration registered for ``schema_ref``."""

    def write(self, experience: Experience, *, source: WriteSource) -> dict[str, JsonValue]:
        """Store one Experience ``source`` sent; the result carries its ``status``."""

    def records_after(self, after: int, limit: int) -> tuple[tuple[tuple[int, Experience], ...], int, bool]:
        """Up to ``limit`` records written after sequence ``after``, the next cursor, and whether more remain."""

    def held(self, experience_id: str) -> Experience:
        """The stored Experience ``experience_id``."""

    def is_pushable(self, experience: Experience) -> bool:
        """Whether a push may send ``experience``: reads here see it and no exclusion still withholds it."""

    def begin_pull(self, schema_ref: str) -> dict[str, JsonValue] | None:
        """Label the current state of ``schema_ref`` when no label holds it; the label made, if any."""

    def bring_in(self, schema_ref: str, experience_ids: Collection[str]) -> None:
        """Put stored Experiences of ``schema_ref`` back into its current state."""

    def file_path(self, ref: FileRef) -> Path:
        """Where the held file ``ref`` names is read from."""

    def missing_files(self, refs: Collection[FileRef]) -> tuple[str, ...]:
        """The sha256 of each of ``refs`` not held here."""

    def put_file(self, sha256: str, size: int, stream: BinaryIO) -> dict[str, JsonValue]:
        """Store ``size`` bytes of ``stream`` as the file ``sha256``."""


@dataclass(frozen=True)
class _Global:
    """The global KB a sync talks to, its identity, and whether it holds Distills: one older than them reports none
    in its health and answers none of their routes."""

    client: RemoteClient
    kb_id: str
    holds_distills: bool


@dataclass(frozen=True)
class _PullPage:
    """A page of the schema a pull batch brings, and the position and Experience on the global KB it follows."""

    page: ExportPage
    position: int
    anchor: str


class GlobalSync:
    """One service's sync with its global KB; one push or pull batch runs at a time."""

    def __init__(self, service: SyncedService, ledger: SyncLedger, target: RemoteClient | None) -> None:
        self._service = service
        self._ledger = ledger
        self._target = target
        if target is not None:
            target.identify(service.kb_id, service.name)

    def _connected(self) -> _Global:
        """The global KB, once it is the one this service synced with before, or the first one."""

        if self._target is None:
            raise SyncUnavailable(
                f"this service was started without a global Experience KB ({GLOBAL_URL_ENV}, {GLOBAL_TOKEN_ENV})"
            )
        url = self._target.config.base_url
        health = self._target.health()
        kb_id = str(health.get("kb_id") or "")
        if not kb_id:
            raise SyncRefused(f"the Experience KB at {url} reports no identity; upgrade it before syncing with it")
        bound = self._ledger.identity(url)
        if bound and bound != kb_id:
            raise SyncRefused(
                f"the Experience KB at {url} is {kb_id}, not {bound} that this service synced with; it is another "
                "global KB. If it replaced that one, rebind this service (hyperloom-kb rebind) to sync with it from "
                "the start"
            )
        self._ledger.bind(url, kb_id)
        return _Global(self._target, kb_id, "distills" in health)

    def rebind(self) -> dict[str, JsonValue]:
        """Forget the global KB this service synced with at its URL, so the next push and pull sync from the start
        with whichever KB answers there; Experiences pulled before stay pulled and are never pushed."""

        with self._ledger.exclusive():
            if self._target is None:
                raise SyncUnavailable(
                    f"this service was started without a global Experience KB ({GLOBAL_URL_ENV}, {GLOBAL_TOKEN_ENV})"
                )
            url = self._target.config.base_url
            forgotten = self._ledger.identity(url)
            self._ledger.forget(url)
            return {"global_url": url, "forgotten_kb_id": forgotten}

    def _send(
        self, target: RemoteClient, experience: Experience, counts: Counter[str], rejected: list[JsonValue]
    ) -> str:
        """Write one Experience to the global KB; returns the error that should stop this push, if any."""

        declaration = self._service.declaration_for(experience.schema_ref)
        files = {ref.sha256: self._service.file_path(ref) for ref in experience.files()}
        try:
            counts[target.write(experience, declaration=declaration, files=files).status] += 1
        except RemoteClientError as exc:
            if exc.retryable:
                return str(exc)
            rejected.append({"experience_id": experience.id, "detail": str(exc)})
            return ""
        self._ledger.note_on_global(target.config.base_url, experience.schema_ref, (experience.id,))
        return ""

    def _unsent_distills(self, url: str) -> Iterator[tuple[str, str, int]]:
        """Each latest Distill version, with its schema, that the global KB at ``url`` does not hold yet."""

        for schema_ref in self._service.schema_refs:
            on_global = self._ledger.distills_on_global(url, schema_ref)
            for entry_id, version in self._service.distill.pushable(schema_ref):
                if (entry_id, version) not in on_global:
                    yield schema_ref, entry_id, version

    def _push_distills(
        self, target: RemoteClient, distills: Counter[str], rejected: list[JsonValue]
    ) -> tuple[str, bool]:
        """Send up to a batch of the latest Distill versions the global KB does not hold yet; the error that should
        stop this push, if any, and whether more remain. A version the global KB refuses is reported once, and a later
        version of the entry is sent again."""

        url = target.config.base_url
        for sent, (schema_ref, entry_id, version) in enumerate(self._unsent_distills(url)):
            if sent == SYNC_BATCH:
                return "", True
            entry, form = self._service.distill.version(entry_id, version)
            try:
                result = target.put_distill_version(
                    entry,
                    form,
                    declaration=self._service.declaration_for(schema_ref),
                    files={ref.sha256: self._service.file_path(ref) for ref in entry.files()},
                )
            except RemoteClientError as exc:
                if exc.retryable:
                    return str(exc), False
                rejected.append({"entry_id": entry_id, "version": version, "detail": str(exc)})
            else:
                distills[str(result["status"])] += 1
            self._ledger.note_distill(url, schema_ref, entry_id, version)
        return "", False

    def push(self) -> dict[str, JsonValue]:
        """Send the Experiences written here that reads here see, then the latest version of each visible Distill;
        an Experience reads do not see waits until they do."""

        with self._ledger.exclusive():
            url = self._target.config.base_url if self._target is not None else ""
            try:
                connected = self._connected()
            except RemoteClientError as exc:
                return _report("incomplete", url, error=str(exc))
            except SyncRefused as exc:
                return _report("refused", url, error=str(exc))
            target = connected.client
            counts: Counter[str] = Counter()
            rejected: list[JsonValue] = []
            for experience_id in self._ledger.held_back(url):
                experience = self._service.held(experience_id)
                if not self._service.is_pushable(experience):
                    continue
                error = self._send(target, experience, counts, rejected)
                if error:
                    return _report("incomplete", url, counts, rejected, error=error)
                self._ledger.release(url, experience_id)
            position, _ = self._ledger.cursor(url, _PUSH)
            records, _, has_more = self._service.records_after(position, SYNC_BATCH)
            for sequence, experience in records:
                if self._ledger.pulled(experience.id):
                    counts["skipped"] += 1
                elif not self._service.is_pushable(experience):
                    self._ledger.hold_back(url, experience.id)
                    counts["held_back"] += 1
                else:
                    error = self._send(target, experience, counts, rejected)
                    if error:
                        self._ledger.advance(url, _PUSH, position)
                        return _report("incomplete", url, counts, rejected, error=error)
                position = sequence
            self._ledger.advance(url, _PUSH, position)
            distills: Counter[str] = Counter()
            if not has_more and not connected.holds_distills:
                # A global KB older than Distills takes them once it is upgraded; until then each waits here.
                distills["held_back"] = sum(1 for _ in self._unsent_distills(url))
            elif not has_more:
                # Distills go once every Experience has, so the evidence they name is on the global KB first.
                error, has_more = self._push_distills(target, distills, rejected)
                if error:
                    return _report("incomplete", url, counts, rejected, distills=distills, error=error)
            return _report("completed", url, counts, rejected, distills=distills, has_more=has_more)

    def _next_page(self, target: RemoteClient, schema_ref: str) -> _PullPage:
        """The page of ``schema_ref`` a pull batch brings, after the cursor, or from the start again when the global KB
        no longer holds what the cursor names or shows the schema differently."""

        url = target.config.base_url
        position, anchor = self._ledger.cursor(url, f"{_PULL} {schema_ref}")
        page = target.export_page(after=position, limit=SYNC_BATCH, schema_ref=schema_ref)
        lost = bool(position) and page.after_id != anchor
        if lost or (position and page.state != self._ledger.pulled_state(url, schema_ref)):
            # Either the global KB no longer holds at the cursor what this service pulled there, as after a
            # restore from an older backup, so what it holds now may sit before the cursor; or its exclusions
            # or restores changed, so Experiences the cursor passed while they were hidden may show now.
            # The ones already here come back unchanged.
            page = target.export_page(after=0, limit=SYNC_BATCH, schema_ref=schema_ref)
            position, anchor = 0, ""
        if lost:
            # What it lost may include what this service pushed, so the next push sends everything again.
            self._ledger.advance(url, _PUSH, 0)
        return _PullPage(page, position, anchor)

    def pull(self, schema_ref: str) -> dict[str, JsonValue]:
        """Bring ``schema_ref``'s state to everything the global KB holds of it, one batch per call.

        The first batch of a pull labels the current state when no label holds it, so the pull can be undone, and
        puts back every Experience of the schema known to be on the global KB that a restore set outside.
        Exclusions stand. Other schemas stay as they are. The cursor remembers the Experience it stopped at; when the
        global KB holds another one there, or none, the pull pages the schema from its start again.
        """

        with self._ledger.exclusive():
            url = self._target.config.base_url if self._target is not None else ""
            try:
                connected = self._connected()
            except RemoteClientError as exc:
                return _report("incomplete", url, error=str(exc))
            except SyncRefused as exc:
                return _report("refused", url, error=str(exc))
            target, global_kb_id = connected.client, connected.kb_id
            direction = f"{_PULL} {schema_ref}"
            try:
                batch = self._next_page(target, schema_ref)
            except RemoteClientError as exc:
                return _report("incomplete", url, error=str(exc))
            page, position, anchor = batch.page, batch.position, batch.anchor
            if page.declaration is not None:
                self._service.register(page.declaration)
            saved: dict[str, JsonValue] | None = None
            if not self._ledger.pull_in_progress(url, schema_ref):
                saved = self._service.begin_pull(schema_ref)
                self._service.bring_in(schema_ref, self._ledger.on_global(url, schema_ref))
            counts: Counter[str] = Counter()
            rejected: list[JsonValue] = []
            fetched: list[str] = []
            # Where the next batch resumes: past this page, or past the last record handled before a file that could
            # not be fetched yet.
            resume, resume_id, error = page.next_cursor, page.next_cursor_id, ""
            reached, reached_id = position, anchor
            for item in page.items:
                try:
                    experience = Experience.from_dict(item.get("experience"))
                    if experience.schema_ref != schema_ref:
                        raise ValueError(f"exported under {schema_ref} but belongs to {experience.schema_ref}")
                    self._fetch_files(target, experience.files())
                    status = str(self._service.write(experience, source=WriteSource(global_kb_id))["status"])
                except RemoteClientError as exc:
                    if exc.retryable:
                        resume, resume_id, error = reached, reached_id, str(exc)
                        break
                    rejected.append({"experience_id": _record_id(item), "detail": str(exc)})
                except (KeyError, TypeError, ValueError, StorageContractError) as exc:
                    rejected.append({"experience_id": _record_id(item), "detail": str(exc)})
                else:
                    if status == "created":
                        self._ledger.mark_pulled(experience.id)
                    fetched.append(experience.id)
                    counts[status] += 1
                sequence = item.get("sequence")
                reached = sequence if isinstance(sequence, int) and not isinstance(sequence, bool) else reached
                reached_id = _record_id(item)
            self._ledger.note_on_global(url, schema_ref, fetched)
            self._service.bring_in(schema_ref, fetched)
            self._ledger.advance(url, direction, resume, resume_id)
            self._ledger.note_pulled_state(url, schema_ref, page.state)
            distills: Counter[str] = Counter()
            if not error and not page.has_more and connected.holds_distills:
                # Distills come once every Experience has, so the evidence they name is here first; a global KB
                # older than Distills holds none.
                error = self._pull_distills(target, schema_ref, WriteSource(global_kb_id), distills, rejected)
            self._ledger.mark_pull_in_progress(url, schema_ref, bool(error) or page.has_more)
            return {
                **_report(
                    "incomplete" if error else "completed",
                    url,
                    counts,
                    rejected,
                    distills=distills,
                    has_more=bool(error) or page.has_more,
                    error=error,
                ),
                "schema_ref": schema_ref,
                "saved": saved,
            }

    def _pull_distills(
        self,
        target: RemoteClient,
        schema_ref: str,
        source: WriteSource,
        distills: Counter[str],
        rejected: list[JsonValue],
    ) -> str:
        """Bring the version each Distill of ``schema_ref`` the global KB shows, and make it shown here unless a later
        version is; the error that should stop this pull, if any.

        A version held here with other content than the global KB's is a conflict, reported and left as it is.
        """

        url = target.config.base_url
        try:
            listed = target.list_distills(schema_ref)["items"]
            forms = {
                str(item["form_ref"]): FormDeclaration.from_dict(item["form"])
                for item in target.distill_forms(schema_ref)["forms"]
            }
        except RemoteClientError as exc:
            return str(exc)
        for item in listed:
            entry_id, version = str(item["entry_id"]), int(item["version"])
            latest = self._service.distill.latest_version(entry_id)
            try:
                if latest > version:
                    distills["skipped"] += 1
                    continue
                if latest == version:
                    held, _ = self._service.distill.version(entry_id, version)
                    if held.content_hash() != item["content_hash"]:
                        raise ValueError(f"version {version} of {entry_id} here differs from the global KB's")
                    distills["unchanged"] += 1
                else:
                    entry = DistillEntry.from_dict(target.distill(entry_id, version=version)["entry"])
                    self._fetch_files(target, entry.files())
                    received = self._service.distill.receive(entry, forms[entry.form_ref], source=source)
                    distills[str(received["status"])] += 1
            except RemoteClientError as exc:
                if exc.retryable:
                    return str(exc)
                rejected.append({"entry_id": entry_id, "version": version, "detail": str(exc)})
                continue
            except (KeyError, TypeError, ValueError, StorageContractError, UnknownStateItem) as exc:
                rejected.append({"entry_id": entry_id, "version": version, "detail": str(exc)})
                continue
            self._service.distill.show_at_least(schema_ref, entry_id, version)
            self._service.bring_in(schema_ref, (entry_id,))
            self._ledger.note_distill(url, schema_ref, entry_id, version)
        return ""

    def _fetch_files(self, target: RemoteClient, refs: Iterable[FileRef]) -> None:
        """Bring each file of ``refs`` that is not held here from the global KB, before the record naming it."""

        for sha256 in self._service.missing_files(tuple(refs)):
            target.fetch_file(sha256, partial(self._service.put_file, sha256))


__all__ = [
    "DISTILL_COUNTS",
    "GLOBAL_TOKEN_ENV",
    "GLOBAL_URL_ENV",
    "SYNC_BATCH",
    "GlobalSync",
    "SyncLedger",
    "SyncRefused",
    "SyncUnavailable",
    "SyncedDistills",
    "SyncedService",
    "global_config_from_env",
]
