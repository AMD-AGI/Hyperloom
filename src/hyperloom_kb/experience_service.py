# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""One Experience KB served over HTTP: writes, reads, lists, labels, exclusions, sync and health."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import uuid
from collections.abc import Collection, Iterable
from dataclasses import dataclass
from datetime import datetime
from functools import cache, partial
from pathlib import Path
from typing import Any, BinaryIO, cast

from hyperloom_kb.database import (
    Connection,
    Database,
    WriteSource,
    record_write,
    utc_now,
)
from hyperloom_kb.files import FileStore, file_sha256
from hyperloom_kb.observability import (
    Metrics,
    event,
)
from hyperloom_kb.local_state import LABEL_BEFORE_PULL, LABEL_MANUAL, LocalState, UnknownStateItem
from hyperloom_kb.knowledge_read import (
    KnowledgeReadService,
    QueryExecutor,
    ReadResult,
    ReadStatus,
    ReadTrace,
)
from hyperloom_kb.query_view import (
    InMemoryQueryViewStore,
    QueryView,
    QueryViewBuilder,
    RetrievalCapability,
)
from hyperloom_kb.remote import QuerySignal, ReadQuery, RemoteClient
from hyperloom_kb.retrieval import LocalRetrievalService, render_complete_experience
from hyperloom_kb.retrieval_policy import (
    LEXICAL_FUZZY_PROVIDER_REF,
    LexicalFuzzyProvider,
    ResultShape,
    RetrievalConfiguration,
)
from hyperloom_kb.records import RecordFiles
from hyperloom_kb.schema import (
    MAX_FILE_BYTES,
    Experience,
    ExperienceDeclaration,
    ExperienceStatus,
    FieldRole,
    FieldValue,
    FileRef,
    JsonValue,
)
from hyperloom_kb.storage import (
    ImmutableExperienceConflict,
    InMemoryExperienceStore,
    InsertStatus,
    StorageContractError,
    canonical_experience_bytes,
)
from hyperloom_kb.sync import GlobalSync, SyncLedger

log = logging.getLogger(__name__)

MIXED_OUTCOME = "mixed"
DEFAULT_READ_LIMIT = 10
MAX_READ_LIMIT = 100
MAX_LIST_LIMIT = 500
# Complete records carry patch content, so an export page stays far smaller than a summary page.
MAX_EXPORT_LIMIT = 100
READ_POLICY_VERSION = "shared-experience-read@v1"
_CONFLICT = "conflict"
_MISSING_FILES = "missing_files"
_AUDIT = "audit"
# Below this a write could fail for space, so the service stops reporting ready.
MIN_FREE_DISK_BYTES = 512 * 1024**2
# How long a probe or a metrics scrape waits on the database: inside the few seconds an orchestrator or a scraper
# waits for the answer, so a database that is gone reads as unready instead of as a probe that timed out.
PROBE_TIMEOUT_SECONDS = 2.0


class HTTPServiceError(ValueError):
    """Raised when a request or service configuration is invalid."""


class MissingFiles(RuntimeError):
    """Raised when a write names files this KB does not hold; send them first."""

    def __init__(self, missing: Collection[str]) -> None:
        super().__init__(f"send the files this record names before it: {', '.join(sorted(missing))}")
        self.missing = tuple(sorted(missing))


@cache
def code_digest() -> str:
    """A fingerprint of this package's code, so a client can tell a service started from other code, such as one
    started before an upgrade."""

    root = Path(__file__).parent
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root)
        if "tests" not in relative.parts:
            digest.update(f"{relative.as_posix()}\0".encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _required_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise HTTPServiceError(f"{name} must be a non-empty string")
    return value.strip()


def _bounded_int(value: Any, name: str, *, minimum: int, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise HTTPServiceError(f"{name} must be an integer")
    if value < minimum or (maximum is not None and value > maximum):
        upper = "" if maximum is None else f" and at most {maximum}"
        raise HTTPServiceError(f"{name} must be at least {minimum}{upper}")
    return value


@dataclass(frozen=True)
class HTTPServiceConfig:
    home: Path
    token: str
    #: How this service names itself in health, metrics and to the KBs it syncs with.
    name: str = ""
    #: Digest of the settings the service was started with, reported by health.
    config_digest: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "home", Path(self.home).expanduser())
        object.__setattr__(self, "token", str(self.token or ""))
        if not self.token:
            raise HTTPServiceError("HYPERLOOM_KB_TOKEN must be configured")


def _role_value(
    declaration: ExperienceDeclaration, experience: Experience, category: str, role: FieldRole
) -> FieldValue | None:
    field = declaration.role_field(category, role)
    return None if field is None else getattr(experience, category).get(field.name)


def _summary(declaration: ExperienceDeclaration, experience: Experience) -> dict[str, JsonValue]:
    """A list line: the record's id and run, the conditions it was measured under, and the values its schema gives
    the summary, decision, and measurement roles."""

    summary = _role_value(declaration, experience, "change", FieldRole.SUMMARY)
    decision = _role_value(declaration, experience, "outcome", FieldRole.DECISION)
    return {
        "experience_id": experience.id,
        "source_run_id": experience.run_id,
        "identity": cast(JsonValue, dict(experience.identity)),
        "change_summary": summary if isinstance(summary, str) else "",
        "decision": decision if isinstance(decision, str) else "",
        "baseline_value": cast(JsonValue, _role_value(declaration, experience, "baseline", FieldRole.MEASUREMENT)),
        "outcome_value": cast(JsonValue, _role_value(declaration, experience, "outcome", FieldRole.MEASUREMENT)),
    }


@dataclass(frozen=True)
class _ReadView:
    """What reads of one schema search, built at one version of the schema's records and state."""

    version: int
    store: InMemoryExperienceStore
    view: QueryView


class ExperienceHTTPService:
    """Write, read, and list one Experience KB holding any number of schemas, kept in ``database`` and its home.

    The service keeps no state of its own beyond caches the database invalidates, so any number of processes may
    serve one KB from one database and one home. ``declaration`` is the schema a read searches when it names none;
    every other registered schema is stored, listed, exported, and synced the same way.
    """

    def __init__(
        self,
        config: HTTPServiceConfig,
        declaration: ExperienceDeclaration,
        *,
        database: Database,
        global_kb: RemoteClient | None = None,
    ) -> None:
        self.config = config
        self.declaration = declaration
        self.name = config.name
        self._database = database
        config.home.mkdir(parents=True, exist_ok=True)
        self.kb_id = database.resolve_kb()
        self._config_digest = config.config_digest
        self._code_digest = code_digest()
        self._records = RecordFiles(config.home, self.kb_id)
        self._files = FileStore(config.home, self.kb_id)
        self._state = LocalState(self.kb_id)
        self._ledger = SyncLedger(database, self.kb_id)
        self._declarations: dict[str, ExperienceDeclaration] = {}
        self._views: dict[str, _ReadView] = {}
        self.metrics = Metrics()
        self.register(declaration)
        self._missing_records = self._verify_records()
        self._missing_files = self._verify_files()
        self._sync = GlobalSync(self, self._ledger, global_kb)

    def _verify_records(self) -> int:
        """How many records the database holds whose file is missing or of another size; reported as unready."""

        with self._database.transaction() as connection:
            rows = connection.execute(
                "SELECT experience_id, bytes FROM experiences WHERE kb_id = %s", (self.kb_id,)
            ).fetchall()
        missing = sum(1 for row in rows if not self._records.holds(str(row["experience_id"]), int(row["bytes"])))
        event(
            log,
            "records_verified",
            level=logging.WARNING if missing else logging.INFO,
            kb_id=self.kb_id,
            records=len(rows),
            missing=missing,
        )
        return missing

    def _verify_files(self) -> int:
        """How many files the database holds that are missing from the home or of another size; reported as
        unready."""

        with self._database.transaction() as connection:
            rows = connection.execute("SELECT sha256, bytes FROM files WHERE kb_id = %s", (self.kb_id,)).fetchall()
        missing = sum(1 for row in rows if not self._files.holds(str(row["sha256"]), int(row["bytes"])))
        event(
            log,
            "files_verified",
            level=logging.WARNING if missing else logging.INFO,
            kb_id=self.kb_id,
            files=len(rows),
            missing=missing,
        )
        return missing

    def _schema_row(self, connection: Connection, schema_ref: str, *, changes_reads: bool) -> None:
        """Lock ``schema_ref`` for this transaction; ``changes_reads`` also moves on what cached reads were built at."""

        # An update locks the row in either database, and an unchanged version leaves cached reads standing.
        version = "version + 1" if changes_reads else "version"
        locked = connection.execute(
            f"UPDATE schemas SET version = {version} WHERE kb_id = %s AND schema_ref = %s", (self.kb_id, schema_ref)
        ).rowcount
        if not locked:
            raise HTTPServiceError(f"schema_ref {schema_ref} is not registered; write it with its declaration")

    def _schema_of(self, connection: Connection, experience_id: str) -> str:
        row = connection.execute(
            "SELECT schema_ref FROM experiences WHERE kb_id = %s AND experience_id = %s", (self.kb_id, experience_id)
        ).fetchone()
        if row is None:
            raise UnknownStateItem(f"Experience {experience_id} is not held here")
        return str(row["schema_ref"])

    def is_visible(self, experience: Experience) -> bool:
        with self._database.transaction() as connection:
            row = connection.execute(
                """
                SELECT NOT EXISTS (
                    SELECT 1 FROM outside WHERE kb_id = %(kb)s AND schema_ref = %(schema)s AND experience_id = %(id)s
                ) AND NOT EXISTS (
                    SELECT 1 FROM exclusions WHERE kb_id = %(kb)s AND schema_ref = %(schema)s AND experience_id = %(id)s
                ) AS visible
                """,
                {"kb": self.kb_id, "schema": experience.schema_ref, "id": experience.id},
            ).fetchone()
        return bool(row and row["visible"])

    def is_pushable(self, experience: Experience) -> bool:
        if not self.is_visible(experience):
            return False
        with self._database.transaction() as connection:
            return experience.id not in self._state.withheld(connection, experience.schema_ref)

    def held(self, experience_id: str) -> Experience:
        with self._database.transaction() as connection:
            row = connection.execute(
                "SELECT content_hash FROM experiences WHERE kb_id = %s AND experience_id = %s",
                (self.kb_id, experience_id),
            ).fetchone()
        if row is None:
            raise UnknownStateItem(f"Experience {experience_id} is not held here")
        return self._records.read(experience_id, str(row["content_hash"]))

    def declaration_for(self, schema_ref: str) -> ExperienceDeclaration:
        declaration = self._declarations.get(schema_ref)
        if declaration is not None:
            return declaration
        with self._database.transaction() as connection:
            row = connection.execute(
                "SELECT declaration FROM schemas WHERE kb_id = %s AND schema_ref = %s", (self.kb_id, schema_ref)
            ).fetchone()
        if row is None:
            raise HTTPServiceError(f"schema_ref {schema_ref} is not registered; write it with its declaration")
        declaration = ExperienceDeclaration.from_dict(json.loads(row["declaration"]))
        self._declarations[schema_ref] = declaration
        return declaration

    def register(self, declaration: ExperienceDeclaration) -> None:
        if declaration.schema_ref in self._declarations:
            return
        with self._database.transaction() as connection:
            registered = connection.execute(
                """
                INSERT INTO schemas(kb_id, schema_ref, declaration, registered_at) VALUES (%s, %s, %s, %s)
                ON CONFLICT DO NOTHING
                """,
                (self.kb_id, declaration.schema_ref, _canonical(declaration.to_dict()), utc_now()),
            ).rowcount
        if registered:
            event(log, _AUDIT, action="register_schema", schema_ref=declaration.schema_ref)
        self._declarations[declaration.schema_ref] = declaration

    def _decision(self, declaration: ExperienceDeclaration, store: InMemoryExperienceStore, experience_id: str) -> str:
        stored = store.get_experience(experience_id)
        if stored is None:
            return ""
        value = _role_value(declaration, stored.experience, "outcome", FieldRole.DECISION)
        return value if isinstance(value, str) else ""

    def _outcome(self, value: Any, declaration: ExperienceDeclaration) -> str:
        """The decision a read restricts to, or ``mixed`` for none; only a schema with a decision field restricts."""

        outcome = MIXED_OUTCOME if value is None else _required_text(value, "outcome")
        allowed = (*declaration.decision_values, MIXED_OUTCOME)
        if outcome not in allowed:
            raise HTTPServiceError(f"outcome must be one of: {', '.join(allowed)}")
        return outcome

    @staticmethod
    def _signals(signals: tuple[QuerySignal, ...], declaration: ExperienceDeclaration) -> tuple[QuerySignal, ...]:
        """``signals`` once every exact one names a field ``declaration`` indexes; a read needs at least one."""

        if not signals:
            raise HTTPServiceError("signals must name at least one signal")
        indexed = declaration.lookup_fields()
        unknown = sorted({signal.field for signal in signals if signal.field and signal.field not in indexed})
        if unknown:
            raise HTTPServiceError(f"signal fields the schema does not index: {', '.join(unknown)}")
        return signals

    def _missing_files_of(self, connection: Connection, refs: Iterable[FileRef]) -> tuple[str, ...]:
        """The sha256 of each of ``refs`` this KB does not hold; a ref of another size than the one held is invalid."""

        wanted = {ref.sha256: ref.bytes for ref in refs}
        if not wanted:
            return ()
        held = {
            str(row["sha256"]): int(row["bytes"])
            for row in connection.execute(
                f"SELECT sha256, bytes FROM files WHERE kb_id = %s AND sha256 IN ({', '.join(['%s'] * len(wanted))})",
                (self.kb_id, *wanted),
            )
        }
        for sha256, size in held.items():
            if wanted[sha256] != size:
                raise HTTPServiceError(f"file {sha256} is {size} bytes, not the {wanted[sha256]} the record names")
        return tuple(sorted(set(wanted) - set(held)))

    def missing_files(self, refs: Iterable[FileRef]) -> tuple[str, ...]:
        with self._database.transaction() as connection:
            return self._missing_files_of(connection, refs)

    def put_file(self, sha256: str, size: int, stream: BinaryIO) -> dict[str, JsonValue]:
        """Store the ``size`` bytes of ``stream`` as the file ``sha256``; storing a held file again changes nothing."""

        sha256 = file_sha256(sha256)
        size = _bounded_int(size, "file size", minimum=0, maximum=MAX_FILE_BYTES)
        self._files.receive(sha256, size, stream)
        with self._database.transaction() as connection:
            created = connection.execute(
                "INSERT INTO files(kb_id, sha256, bytes, stored_at) VALUES (%s, %s, %s, %s) ON CONFLICT DO NOTHING",
                (self.kb_id, sha256, size, utc_now()),
            ).rowcount
        status = InsertStatus.CREATED.value if created else InsertStatus.UNCHANGED.value
        self.metrics.count("hyperloom_kb_file_writes_total", result=status)
        return {"status": status, "sha256": sha256, "bytes": size}

    def held_file(self, sha256: str) -> tuple[Path, int]:
        """Where the held file ``sha256`` is read from, and its size."""

        sha256 = file_sha256(sha256)
        with self._database.transaction() as connection:
            row = connection.execute(
                "SELECT bytes FROM files WHERE kb_id = %s AND sha256 = %s", (self.kb_id, sha256)
            ).fetchone()
        if row is None:
            raise UnknownStateItem(f"file {sha256} is not held here")
        return self._files.path(sha256), int(row["bytes"])

    def file_path(self, ref: FileRef) -> Path:
        """Where an agent on this host reads the file ``ref`` names."""

        return self._files.path(ref.sha256)

    def _existing(self, connection: Connection, experience_id: str, content_hash: str) -> str:
        """``unchanged`` or ``conflict`` for an id already stored with that or other content, ``""`` for a new one."""

        row = connection.execute(
            "SELECT content_hash FROM experiences WHERE kb_id = %s AND experience_id = %s", (self.kb_id, experience_id)
        ).fetchone()
        if row is None:
            return ""
        return InsertStatus.UNCHANGED.value if row["content_hash"] == content_hash else _CONFLICT

    def write(
        self,
        experience: Experience,
        declaration: ExperienceDeclaration | None = None,
        *,
        source: WriteSource = WriteSource(),
    ) -> dict[str, JsonValue]:
        """Store one complete Experience ``source`` sent; ``declaration`` registers its schema when this KB lacks it.

        Every file the record names must be held here first. Every write is recorded with its source and result, a
        refused one included.
        """

        if declaration is not None:
            if declaration.schema_ref != experience.schema_ref:
                raise HTTPServiceError("declaration does not derive the Experience schema_ref")
            self.register(declaration)
        schema = self.declaration_for(experience.schema_ref)
        if experience.status is not ExperienceStatus.COMPLETE:
            raise StorageContractError("a write requires a complete Experience")
        schema.validate(experience)
        data = canonical_experience_bytes(experience)
        content_hash = hashlib.sha256(data).hexdigest()
        missing: tuple[str, ...] = ()
        with self._database.transaction() as connection:
            status = self._existing(connection, experience.id, content_hash)
            if not status:
                missing = self._missing_files_of(connection, experience.files())
                status = _MISSING_FILES if missing else ""
            if not status:
                # Taking the KB's row before looking again serializes writers of the same id across processes.
                connection.execute("UPDATE kbs SET last_sequence = last_sequence WHERE kb_id = %s", (self.kb_id,))
                status = self._existing(connection, experience.id, content_hash)
            if not status:
                connection.execute("UPDATE kbs SET last_sequence = last_sequence + 1 WHERE kb_id = %s", (self.kb_id,))
                position = connection.execute(
                    "SELECT last_sequence FROM kbs WHERE kb_id = %s", (self.kb_id,)
                ).fetchone()
                assert position is not None, "the KB row resolve_kb made is never deleted"
                self._records.write(experience.id, data)
                connection.execute(
                    """
                    INSERT INTO experiences(kb_id, experience_id, schema_ref, sequence, content_hash, bytes, stored_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        self.kb_id,
                        experience.id,
                        experience.schema_ref,
                        position["last_sequence"],
                        content_hash,
                        len(data),
                        utc_now(),
                    ),
                )
                self._schema_row(connection, experience.schema_ref, changes_reads=True)
                status = InsertStatus.CREATED.value
            record_write(connection, self.kb_id, experience, status, source)
        self.metrics.count("hyperloom_kb_writes_total", schema_ref=experience.schema_ref, result=status)
        if status == _CONFLICT:
            raise ImmutableExperienceConflict("Experience id already exists with different content")
        if status == _MISSING_FILES:
            raise MissingFiles(missing)
        return {"status": status, "experience_id": experience.id, "content_hash": content_hash}

    def seed(self, path: Path) -> dict[str, int]:
        counts = {InsertStatus.CREATED.value: 0, InsertStatus.UNCHANGED.value: 0}
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise HTTPServiceError("seed rows must be objects")
                response = self.write(Experience.from_dict(value.get("experience", value)))
                counts[str(response["status"])] += 1
        return counts

    def _read_view(self, declaration: ExperienceDeclaration) -> _ReadView:
        """What reads of ``declaration``'s schema search, rebuilt whenever its records or state moved on."""

        schema_ref = declaration.schema_ref
        with self._database.transaction() as connection:
            row = connection.execute(
                "SELECT version FROM schemas WHERE kb_id = %s AND schema_ref = %s", (self.kb_id, schema_ref)
            ).fetchone()
            version = int(row["version"]) if row else 0
            cached = self._views.get(schema_ref)
            if cached is not None and cached.version == version:
                return cached
            visible = self._state.visible(connection, schema_ref)
            hashes = {
                str(entry["experience_id"]): str(entry["content_hash"])
                for entry in connection.execute(
                    "SELECT experience_id, content_hash FROM experiences WHERE kb_id = %s AND schema_ref = %s",
                    (self.kb_id, schema_ref),
                )
            }
        store = InMemoryExperienceStore()
        for experience_id in sorted(visible):
            store.insert_complete(self._records.read(experience_id, hashes[experience_id]))
        built = _ReadView(
            version,
            store,
            QueryViewBuilder().build(declaration, store.list_experiences(schema_ref), fuzzy_ready=True),
        )
        self._views[schema_ref] = built
        return built

    def _restrict_to_outcome(
        self, declaration: ExperienceDeclaration, store: InMemoryExperienceStore, view: QueryView, outcome: str
    ) -> QueryView:
        """``view`` narrowed to the Experiences whose decision is ``outcome``."""

        return QueryViewBuilder().restrict(
            view,
            (
                experience_id
                for experience_id in view.visible_experience_ids
                if self._decision(declaration, store, experience_id) == outcome
            ),
        )

    def _completed_fields(
        self,
        declaration: ExperienceDeclaration,
        store: InMemoryExperienceStore,
        result: ReadResult,
        traces: list[ReadTrace],
    ) -> dict[str, JsonValue]:
        """The response fields of a completed read, each rendered Experience summarized with its score."""

        groups = {
            experience_id: group
            for group in (traces[-1].execution.groups if traces else ())
            for experience_id in group.representative_ids
        }
        experiences: list[JsonValue] = []
        for reference in result.rendered_refs:
            stored = store.get_experience(reference.id)
            item = _summary(declaration, stored.experience if stored is not None else self.held(reference.id))
            group = groups.get(reference.id)
            item["score"] = group.score if group is not None else 0.0
            item["why_matched"] = [
                contribution.to_dict()
                for contribution in (group.contributions if group is not None else ())
                if contribution.experience_id == reference.id
            ]
            experiences.append(item)
        return {
            "status": result.status.value,
            "prompt_block": result.prompt_block,
            "rendered_refs": [reference.to_dict() for reference in result.rendered_refs],
            "experiences": experiences,
            "rendered_count": len(result.rendered_refs),
            "warnings": list(result.warnings),
        }

    def read(self, query: ReadQuery) -> dict[str, JsonValue]:
        """Search one schema's Experiences, this service's default schema unless ``query.schema_ref`` names another.

        An Experience is recalled by any of ``query.signals`` and ranked by the summed weight of those it matches. Each
        record renders whole; a file field renders as the path on this host its file is read from. With
        ``query.render_budget_chars``, the block carries whole records while they fit and names only those in
        ``rendered_refs``.
        """

        declaration = self.declaration_for(query.schema_ref or self.declaration.schema_ref)
        signals = self._signals(query.signals, declaration)
        selected_outcome = self._outcome(query.outcome, declaration)
        limit = _bounded_int(
            DEFAULT_READ_LIMIT if query.limit is None else query.limit, "limit", minimum=1, maximum=MAX_READ_LIMIT
        )
        render_budget_chars = query.render_budget_chars
        if render_budget_chars is not None:
            render_budget_chars = _bounded_int(render_budget_chars, "render_budget_chars", minimum=1)
        read_view = self._read_view(declaration)
        store, view = read_view.store, read_view.view
        if selected_outcome != MIXED_OUTCOME:
            view = self._restrict_to_outcome(declaration, store, view, selected_outcome)
        eligible_count = len(view.visible_experience_ids)
        response: dict[str, JsonValue] = {
            "read_id": f"read-{uuid.uuid4().hex}",
            "status": ReadStatus.COMPLETED.value,
            "outcome": selected_outcome,
            "limit": limit,
            "prompt_block": "",
            "rendered_refs": [],
            "experiences": [],
            "eligible_count": eligible_count,
            "rendered_count": 0,
            "warnings": [],
        }
        if eligible_count == 0:
            return response

        views = InMemoryQueryViewStore()
        views.publish_view(view)
        provider_refs = {RetrievalCapability.FUZZY: LEXICAL_FUZZY_PROVIDER_REF}
        configuration = RetrievalConfiguration.create(
            declaration.schema_ref,
            READ_POLICY_VERSION,
            limits=dict.fromkeys(RetrievalCapability, eligible_count),
            provider_refs=provider_refs,
            shape=ResultShape("weighted-signal-sum@v1", max_groups=limit, render_budget_chars=render_budget_chars),
        )
        executor = QueryExecutor(
            LocalRetrievalService(
                store,
                views,
                providers=(LexicalFuzzyProvider(store, declaration),),
                renderer=partial(render_complete_experience, file_path=self.file_path),
            ),
            provider_refs=provider_refs,
        )
        traces: list[ReadTrace] = []
        result = KnowledgeReadService(executor, configuration, trace_sink=traces.append).read(signals)
        response.update(self._completed_fields(declaration, store, result, traces))
        return response

    def _listed(self, experience: Experience, include_excluded: bool) -> bool:
        """List and export name what was written here, and of that only what reads see unless ``include_excluded``."""

        return not self._ledger.pulled(experience.id) and (include_excluded or self.is_visible(experience))

    def list_experiences(
        self,
        *,
        after: int = 0,
        limit: int = 100,
        schema_ref: str | None = None,
        include_excluded: bool = False,
    ) -> dict[str, JsonValue]:
        after = _bounded_int(after, "after", minimum=0)
        limit = _bounded_int(limit, "limit", minimum=1, maximum=MAX_LIST_LIMIT)
        records, next_cursor, has_more = self.records_after(after, limit, schema_ref)
        items: list[JsonValue] = [
            {"sequence": sequence, **_summary(self.declaration_for(experience.schema_ref), experience)}
            for sequence, experience in records
            if self._listed(experience, include_excluded)
        ]
        return {"items": items, "next_cursor": next_cursor, "has_more": has_more}

    def records_after(
        self, after: int, limit: int, schema_ref: str | None = None
    ) -> tuple[tuple[tuple[int, Experience], ...], int, bool]:
        """Complete Experiences in write order after the ``after`` sequence, of one schema or of all of them."""

        with self._database.transaction() as connection:
            rows = connection.execute(
                """
                SELECT sequence, experience_id, content_hash FROM experiences
                WHERE kb_id = %s AND (CAST(%s AS TEXT) IS NULL OR schema_ref = %s) AND sequence > %s
                ORDER BY sequence LIMIT %s
                """,
                (self.kb_id, schema_ref, schema_ref, after, limit + 1),
            ).fetchall()
        page = rows[:limit]
        records = tuple(
            (int(row["sequence"]), self._records.read(str(row["experience_id"]), str(row["content_hash"])))
            for row in page
        )
        return records, int(page[-1]["sequence"]) if page else after, len(rows) > limit

    def _head(self, connection: Connection, schema_ref: str | None) -> int:
        """The last write position of one schema or, with ``schema_ref=None``, of every schema; 0 before any write."""

        row = connection.execute(
            """
            SELECT COALESCE(MAX(sequence), 0) AS head FROM experiences
            WHERE kb_id = %s AND (CAST(%s AS TEXT) IS NULL OR schema_ref = %s)
            """,
            (self.kb_id, schema_ref, schema_ref),
        ).fetchone()
        return int(row["head"]) if row else 0

    def _written_at(self, connection: Connection, sequence: int) -> str:
        """The Experience written at ``sequence``; empty when none was, as for 0."""

        row = connection.execute(
            "SELECT experience_id FROM experiences WHERE kb_id = %s AND sequence = %s", (self.kb_id, sequence)
        ).fetchone()
        return str(row["experience_id"]) if row else ""

    def export(
        self,
        *,
        after: int = 0,
        limit: int = MAX_EXPORT_LIMIT,
        schema_ref: str | None = None,
        include_excluded: bool = False,
    ) -> dict[str, JsonValue]:
        after = _bounded_int(after, "after", minimum=0)
        limit = _bounded_int(limit, "limit", minimum=1, maximum=MAX_EXPORT_LIMIT)
        records, next_cursor, has_more = self.records_after(after, limit, schema_ref)
        items: list[JsonValue] = [
            {"sequence": sequence, "experience": experience.to_dict()}
            for sequence, experience in records
            if self._listed(experience, include_excluded)
        ]
        # Resolved before the transaction: a declaration not cached yet is read in a transaction of its own, which on
        # SQLite would wait on this one's write lock.
        declaration = None if schema_ref is None else self.declaration_for(schema_ref)
        with self._database.transaction() as connection:
            page: dict[str, JsonValue] = {
                "items": items,
                "next_cursor": next_cursor,
                "has_more": has_more,
                "head": self._head(connection, schema_ref),
                "after_id": self._written_at(connection, after),
                "next_cursor_id": self._written_at(connection, next_cursor),
            }
            if declaration is not None:
                page["declaration"] = declaration.to_dict()
                page["state"] = self._state.hidden_digest(connection, declaration.schema_ref)
        return page

    def _schema_ref(self, schema_ref: str | None) -> str:
        return self.declaration_for(schema_ref or self.declaration.schema_ref).schema_ref

    def labels(self, schema_ref: str | None = None) -> dict[str, JsonValue]:
        """``schema_ref``'s labels, newest first, its current label, and whether the state changed since it."""

        schema = self._schema_ref(schema_ref)
        with self._database.transaction() as connection:
            current, modified = self._state.current(connection, schema)
            labels = self._state.labels(connection, schema)
        return {
            "schema_ref": schema,
            "current_label_id": None if current is None else current.label_id,
            "modified": modified,
            "labels": [label.to_dict() for label in labels],
        }

    def create_label(self, *, schema_ref: str | None = None, name: str = "") -> dict[str, JsonValue]:
        schema = self._schema_ref(schema_ref)
        with self._database.transaction() as connection:
            self._schema_row(connection, schema, changes_reads=False)
            label = self._state.label(connection, schema, name=name, reason=LABEL_MANUAL)
        event(log, _AUDIT, action="label", schema_ref=schema, label_id=label.label_id, name=label.name)
        return label.to_dict()

    def delete_label(self, label_id: str) -> dict[str, JsonValue]:
        with self._database.transaction() as connection:
            self._state.delete_label(connection, label_id)
        event(log, _AUDIT, action="delete_label", label_id=label_id)
        return {"deleted": label_id}

    def restore(self, label_id: str) -> dict[str, JsonValue]:
        """Make ``label_id``'s state current; a current state no label holds is labelled first and named in
        ``saved``."""

        with self._database.transaction() as connection:
            label = self._state.get_label(connection, label_id)
            self._schema_row(connection, label.schema_ref, changes_reads=True)
            saved = self._state.restore(connection, label)
        event(
            log,
            _AUDIT,
            action="restore",
            schema_ref=label.schema_ref,
            label_id=label.label_id,
            saved_label_id=None if saved is None else saved.label_id,
        )
        return {"restored": label.to_dict(), "saved": None if saved is None else saved.to_dict()}

    def exclude(self, experience_id: str, reason: str) -> dict[str, JsonValue]:
        with self._database.transaction() as connection:
            schema = self._schema_of(connection, experience_id)
            self._schema_row(connection, schema, changes_reads=True)
            self._state.exclude(connection, schema, experience_id, reason)
        event(log, _AUDIT, action="exclude", schema_ref=schema, experience_id=experience_id, reason=reason)
        return {"experience_id": experience_id, "status": "excluded"}

    def include(self, experience_id: str) -> dict[str, JsonValue]:
        with self._database.transaction() as connection:
            schema = self._schema_of(connection, experience_id)
            self._schema_row(connection, schema, changes_reads=True)
            lifted = self._state.include(connection, schema, experience_id)
        event(log, _AUDIT, action="include", schema_ref=schema, experience_id=experience_id, lifted=lifted)
        return {"experience_id": experience_id, "status": "included" if lifted else "not_excluded"}

    def exclusions(self, schema_ref: str | None = None) -> dict[str, JsonValue]:
        schema = self._schema_ref(schema_ref)
        with self._database.transaction() as connection:
            return {
                "schema_ref": schema,
                "exclusions": list(self._state.exclusions(connection, schema)),
                "history": list(self._state.exclusion_history(connection, schema)),
            }

    def _synced(self, direction: str, report: dict[str, JsonValue]) -> dict[str, JsonValue]:
        self.metrics.count("hyperloom_kb_sync_batches_total", direction=direction, status=str(report["status"]))
        counts = {key: report.get(key) for key in ("created", "unchanged", "skipped", "held_back", "has_more")}
        rejected = report.get("rejected")
        event(
            log,
            "sync",
            level=logging.INFO if report["status"] == "completed" else logging.WARNING,
            direction=direction,
            status=report["status"],
            global_url=report.get("global_url"),
            schema_ref=report.get("schema_ref"),
            rejected=len(rejected) if isinstance(rejected, list) else 0,
            error=report.get("error", ""),
            **counts,
        )
        return report

    def push(self) -> dict[str, JsonValue]:
        return self._synced("push", self._sync.push())

    def pull(self, schema_ref: str) -> dict[str, JsonValue]:
        return self._synced("pull", self._sync.pull(schema_ref))

    def rebind(self) -> dict[str, JsonValue]:
        result = self._sync.rebind()
        event(log, _AUDIT, action="rebind", **result)
        return result

    def begin_pull(self, schema_ref: str) -> dict[str, JsonValue] | None:
        with self._database.transaction() as connection:
            self._schema_row(connection, schema_ref, changes_reads=False)
            saved = self._state.save_if_modified(connection, schema_ref, LABEL_BEFORE_PULL)
        return None if saved is None else saved.to_dict()

    def bring_in(self, schema_ref: str, experience_ids: Collection[str]) -> None:
        with self._database.transaction() as connection:
            self._schema_row(connection, schema_ref, changes_reads=True)
            self._state.bring_in(connection, schema_ref, experience_ids)

    def health(self) -> dict[str, JsonValue]:
        with self._database.transaction() as connection:
            rows = connection.execute(
                """
                SELECT schemas.schema_ref, COUNT(experiences.experience_id) AS visible FROM schemas
                LEFT JOIN experiences ON experiences.kb_id = schemas.kb_id
                    AND experiences.schema_ref = schemas.schema_ref
                    AND NOT EXISTS (
                        SELECT 1 FROM outside WHERE outside.kb_id = experiences.kb_id
                            AND outside.schema_ref = experiences.schema_ref
                            AND outside.experience_id = experiences.experience_id
                    )
                    AND NOT EXISTS (
                        SELECT 1 FROM exclusions WHERE exclusions.kb_id = experiences.kb_id
                            AND exclusions.schema_ref = experiences.schema_ref
                            AND exclusions.experience_id = experiences.experience_id
                    )
                WHERE schemas.kb_id = %s GROUP BY schemas.schema_ref ORDER BY schemas.schema_ref
                """,
                (self.kb_id,),
            ).fetchall()
        counts = {str(row["schema_ref"]): int(row["visible"]) for row in rows}
        return {
            "status": "ok",
            "kb_id": self.kb_id,
            "name": self.name,
            "schema_ref": self.declaration.schema_ref,
            "experience_count": sum(counts.values()),
            "schemas": counts,
            "pid": os.getpid(),
            "config_digest": self._config_digest,
            "code_digest": self._code_digest,
            "home": str(self.config.home.resolve()),
        }

    def readiness(self) -> dict[str, bool]:
        """Each check a service must pass to take traffic; the names say what failed, never what the KB holds."""

        home = self.config.home
        return {
            "database": self._database.answers(PROBE_TIMEOUT_SECONDS),
            "home_writable": os.access(home, os.W_OK),
            "disk_space": shutil.disk_usage(home).free >= MIN_FREE_DISK_BYTES,
            "records": self._missing_records == 0,
            "files": self._missing_files == 0,
        }

    def metric_gauges(self) -> list[tuple[str, dict[str, str], float]]:
        """The gauges a metrics scrape samples now: what the KB holds, how large it is, and how ready it is.

        While the database does not answer, the scrape still answers: its readiness gauge says so, and what only the
        database knows is left out.
        """

        readiness = self.readiness()
        gauges: list[tuple[str, dict[str, str], float]] = [
            ("hyperloom_kb_build_info", {"kb_id": self.kb_id, "name": self.name, "code_digest": self._code_digest}, 1),
            ("hyperloom_kb_records_missing", {}, self._missing_records),
            ("hyperloom_kb_files_missing", {}, self._missing_files),
            ("hyperloom_kb_disk_free_bytes", {}, shutil.disk_usage(self.config.home).free),
        ]
        gauges += [("hyperloom_kb_ready", {"check": check}, float(ok)) for check, ok in readiness.items()]
        gauges += [
            ("hyperloom_kb_database_pool", {"stat": stat}, float(value))
            for stat, value in sorted(self._database.connection_stats().items())
            if stat in ("pool_size", "pool_available", "requests_waiting")
        ]
        if readiness["database"]:
            gauges += self._database_gauges()
        return gauges

    def _database_gauges(self) -> list[tuple[str, dict[str, str], float]]:
        gauges: list[tuple[str, dict[str, str], float]] = []
        with self._database.transaction(timeout=PROBE_TIMEOUT_SECONDS) as connection:
            for row in connection.execute(
                """
                SELECT experiences.schema_ref,
                    SUM(CASE WHEN outside.experience_id IS NULL AND exclusions.experience_id IS NULL THEN 1 ELSE 0 END)
                        AS visible,
                    COUNT(exclusions.experience_id) AS excluded,
                    COUNT(outside.experience_id) AS outside
                FROM experiences
                LEFT JOIN outside USING (kb_id, schema_ref, experience_id)
                LEFT JOIN exclusions USING (kb_id, schema_ref, experience_id)
                WHERE experiences.kb_id = %s GROUP BY experiences.schema_ref
                """,
                (self.kb_id,),
            ):
                for state in ("visible", "excluded", "outside"):
                    gauges.append(
                        ("hyperloom_kb_experiences", {"schema_ref": row["schema_ref"], "state": state}, row[state])
                    )
            sizes = connection.execute(
                "SELECT COALESCE(SUM(bytes), 0) AS records, MAX(stored_at) AS last_write FROM experiences WHERE kb_id = %s",
                (self.kb_id,),
            ).fetchone()
            files = connection.execute(
                "SELECT COUNT(*) AS count, COALESCE(SUM(bytes), 0) AS bytes FROM files WHERE kb_id = %s",
                (self.kb_id,),
            ).fetchone()
        if files is not None:
            gauges.append(("hyperloom_kb_files", {}, float(files["count"])))
            gauges.append(("hyperloom_kb_file_bytes", {}, float(files["bytes"])))
        gauges.append(("hyperloom_kb_database_bytes", {}, float(self._database.size_bytes())))
        if sizes is not None:
            gauges.append(("hyperloom_kb_record_bytes", {}, float(sizes["records"])))
            if sizes["last_write"]:
                written = datetime.fromisoformat(str(sizes["last_write"]).replace("Z", "+00:00"))
                gauges.append(("hyperloom_kb_last_write_timestamp_seconds", {}, written.timestamp()))
        return gauges


__all__ = [
    "DEFAULT_READ_LIMIT",
    "MAX_EXPORT_LIMIT",
    "MAX_READ_LIMIT",
    "MIXED_OUTCOME",
    "ExperienceHTTPService",
    "HTTPServiceConfig",
    "HTTPServiceError",
    "MissingFiles",
    "code_digest",
]
