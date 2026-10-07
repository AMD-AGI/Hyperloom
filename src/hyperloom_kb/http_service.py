# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Authenticated HTTP service for one Experience KB, kept in a PostgreSQL database and its home's record files."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import logging
import os
import shutil
import signal
import threading
import time
import uuid
from collections.abc import Collection, Iterator, Mapping
from contextlib import ExitStack, contextmanager, suppress
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import cache, partial
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, TextIO, cast
from urllib.parse import parse_qs, urlsplit

from psycopg.types.json import Jsonb

from hyperloom_kb.config import PACKAGED_DECLARATION, load_declaration
from hyperloom_kb.database import Database, WriteSource
from hyperloom_kb.embedded_postgres import EmbeddedPostgresError, start_embedded_postgres, unavailable_reason
from hyperloom_kb.legacy_home import LegacyHome
from hyperloom_kb.observability import (
    PROBE_ROUTES,
    REQUEST_ID_HEADER,
    JsonLogFormatter,
    Metrics,
    RequestContext,
    current_request,
    event,
    route_of,
)
from hyperloom_kb.local_state import LABEL_BEFORE_PULL, LABEL_MANUAL, Connection, LocalState, UnknownStateItem
from hyperloom_kb.knowledge_read import (
    AnthropicPlannerBackend,
    KnowledgeReadService,
    LLMQueryPlanner,
    PlannerConfiguration,
    PlannerGatewayConfig,
    QueryExecutor,
    ReadStatus,
    ReadTrace,
)
from hyperloom_kb.query_view import (
    InMemoryQueryViewStore,
    QueryView,
    QueryViewBuilder,
    RetrievalCapability,
)
from hyperloom_kb.remote import RemoteClient, RemoteConfig
from hyperloom_kb.retrieval import LocalRetrievalService, render_complete_experience
from hyperloom_kb.retrieval_policy import (
    LEXICAL_FUZZY_PROVIDER_REF,
    LexicalFuzzyProvider,
    RetrievalConfiguration,
)
from hyperloom_kb.records import RecordFiles
from hyperloom_kb.schema import Experience, ExperienceDeclaration, ExperienceStatus, JsonValue
from hyperloom_kb.service import CompleteExperienceRequired
from hyperloom_kb.storage import (
    ImmutableExperienceConflict,
    InMemoryExperienceStore,
    InsertStatus,
    StorageContractError,
    canonical_experience_bytes,
)
from hyperloom_kb.sync import GlobalSync, SyncLedger, SyncUnavailable, global_config_from_env

log = logging.getLogger(__name__)

MIXED_OUTCOME = "mixed"
DEFAULT_READ_LIMIT = 10
MAX_READ_LIMIT = 100
MAX_LIST_LIMIT = 500
# Complete records carry patch content, so an export page stays far smaller than a summary page.
MAX_EXPORT_LIMIT = 100
READ_POLICY_VERSION = "shared-experience-read@v1"
DEFAULT_HOME = Path("~/.local/share/hyperloom-kb").expanduser()
# Held by the one service process that runs a home's embedded database, for as long as it serves it.
SERVICE_LOCK = "service.lock"
DATABASE_URL_ENV = "HYPERLOOM_KB_DATABASE_URL"
_CONFLICT = "conflict"
# Below this a write could fail for space, so the service stops reporting ready.
MIN_FREE_DISK_BYTES = 512 * 1024 * 1024
# A SIGTERM lets requests in flight finish this long; below an orchestrator's usual 30 s grace period.
DRAIN_SECONDS = 25.0
# How long a probe or a metrics scrape waits on the database: inside the few seconds an orchestrator or a scraper
# waits for the answer, so a database that is gone reads as unready instead of as a probe that timed out.
PROBE_TIMEOUT_SECONDS = 2.0
# A transport guard, not a data policy: Experiences of any size are stored.
_MAX_REQUEST_BYTES = 256 * 1024 * 1024
_READ_FIELDS = frozenset(
    {"decision", "context", "outcome", "limit", "schema_ref", "content_inline_limit", "render_budget_chars"}
)
_WRITE_FIELDS = frozenset({"experience", "declaration"})


class HTTPServiceError(ValueError):
    """Raised when a request or service configuration is invalid."""


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


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class ServiceSettings:
    """What the service takes from its environment at start, besides its token."""

    planner: PlannerGatewayConfig | None
    planner_problem: str
    global_kb: RemoteConfig | None
    global_problem: str
    # Empty runs the home's own embedded database.
    database_url: str = ""

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> ServiceSettings:
        planner: PlannerGatewayConfig | None = None
        global_kb: RemoteConfig | None = None
        planner_problem = global_problem = ""
        try:
            planner = PlannerGatewayConfig.from_env(env)
        except ValueError as exc:
            planner_problem = str(exc)
        try:
            global_kb = global_config_from_env(env)
        except SyncUnavailable as exc:
            global_problem = str(exc)
        database_url = str(env.get(DATABASE_URL_ENV) or "").strip()
        return cls(planner, planner_problem, global_kb, global_problem, database_url)

    def digest(self) -> str:
        """Equal digests mean two environments start behaviorally identical services, whatever their spelling."""

        planner = self.planner
        values = {
            "planner": None
            if planner is None
            else [planner.base_url, planner.api_key, planner.model, planner.timeout_seconds, planner.max_output_tokens],
            "global": None if self.global_kb is None else [self.global_kb.base_url, self.global_kb.token],
            "problems": [self.planner_problem, self.global_problem],
            "database": self.database_url,
        }
        return hashlib.sha256(_canonical(values).encode()).hexdigest()


def _required_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise HTTPServiceError(f"{name} must be a non-empty string")
    return value.strip()


def _json_object(value: Any, name: str) -> dict[str, JsonValue]:
    if not isinstance(value, dict):
        raise HTTPServiceError(f"{name} must be an object")
    normalized = json.loads(_canonical(value))
    if not isinstance(normalized, dict):
        raise HTTPServiceError(f"{name} must be an object")
    return normalized


def _bounded_int(value: Any, name: str, *, minimum: int, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise HTTPServiceError(f"{name} must be an integer")
    if value < minimum or (maximum is not None and value > maximum):
        upper = "" if maximum is None else f" and at most {maximum}"
        raise HTTPServiceError(f"{name} must be at least {minimum}{upper}")
    return value


def _query_int(query: Mapping[str, list[str]], name: str, default: int) -> int:
    raw = query.get(name, [str(default)])[0]
    try:
        return int(raw)
    except ValueError as exc:
        raise HTTPServiceError(f"{name} must be an integer") from exc


def _query_text(query: Mapping[str, list[str]], name: str) -> str | None:
    values = query.get(name)
    return _required_text(values[0], name) if values else None


def _query_flag(query: Mapping[str, list[str]], name: str) -> bool:
    raw = query.get(name, ["false"])[0].strip().lower()
    if raw not in ("true", "false"):
        raise HTTPServiceError(f"{name} must be true or false")
    return raw == "true"


def _reject_unknown(body: Mapping[str, JsonValue], allowed: frozenset[str]) -> None:
    unknown = sorted(set(body) - allowed)
    if unknown:
        raise HTTPServiceError(f"unknown request fields: {', '.join(unknown)}")


@dataclass(frozen=True)
class HTTPServiceConfig:
    home: Path
    token: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "home", Path(self.home).expanduser())
        object.__setattr__(self, "token", str(self.token or ""))
        if not self.token:
            raise HTTPServiceError("HYPERLOOM_KB_TOKEN must be configured")


def _summary(experience: Experience) -> dict[str, JsonValue]:
    return {
        "experience_id": experience.id,
        "source_run_id": experience.run_id,
        "change_summary": experience.change.summary if experience.change is not None else "",
        "decision": experience.outcome.decision if experience.outcome is not None else "",
        "baseline_value": experience.baseline_value,
        "outcome_value": experience.outcome.value if experience.outcome is not None else None,
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
        planner: LLMQueryPlanner | None,
        *,
        database: Database,
        global_kb: RemoteClient | None = None,
        config_digest: str = "",
        name: str = "",
    ) -> None:
        self.config = config
        self.declaration = declaration
        self.name = name
        self._database = database
        config.home.mkdir(parents=True, exist_ok=True)
        legacy = LegacyHome.find(config.home)
        self.kb_id = database.resolve_kb(adopt_kb_id=legacy.kb_id if legacy is not None else "")
        self._config_digest = config_digest
        self._code_digest = code_digest()
        self._records = RecordFiles(config.home, self.kb_id)
        self._state = LocalState(self.kb_id)
        self._ledger = SyncLedger(database, self.kb_id)
        self._planner = planner
        self._declarations: dict[str, ExperienceDeclaration] = {}
        self._views: dict[str, _ReadView] = {}
        self.metrics = Metrics()
        self.register(declaration)
        if legacy is not None:
            self._adopt(legacy)
        self._missing_records = self._verify_records()
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

    def _adopt(self, legacy: LegacyHome) -> None:
        """Bring a home an older service kept on disk into the database, once."""

        with self._database.transaction() as connection:
            row = connection.execute("SELECT adopted_home FROM kbs WHERE kb_id = %s", (self.kb_id,)).fetchone()
        if row is not None and row["adopted_home"]:
            return
        for declaration in legacy.schemas():
            self.register(declaration)
        for experience in legacy.experiences():
            self.write(experience)
        for experience_id in legacy.pulled():
            self._ledger.mark_pulled(experience_id)
        with self._database.transaction() as connection:
            connection.execute(
                "UPDATE kbs SET adopted_home = %s WHERE kb_id = %s", (str(legacy.home.resolve()), self.kb_id)
            )

    def _schema_row(self, connection: Connection, schema_ref: str, *, changes_reads: bool) -> None:
        """Lock ``schema_ref`` for this transaction; ``changes_reads`` also moves on what cached reads were built at."""

        if changes_reads:
            query = "UPDATE schemas SET version = version + 1 WHERE kb_id = %s AND schema_ref = %s RETURNING version"
        else:
            query = "SELECT version FROM schemas WHERE kb_id = %s AND schema_ref = %s FOR UPDATE"
        if connection.execute(query, (self.kb_id, schema_ref)).fetchone() is None:
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

    @property
    def schema_refs(self) -> tuple[str, ...]:
        with self._database.transaction() as connection:
            rows = connection.execute(
                "SELECT schema_ref FROM schemas WHERE kb_id = %s ORDER BY schema_ref", (self.kb_id,)
            ).fetchall()
        return tuple(str(row["schema_ref"]) for row in rows)

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
        declaration = ExperienceDeclaration.from_dict(row["declaration"])
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
                (self.kb_id, declaration.schema_ref, Jsonb(declaration.to_dict()), _utc_now()),
            ).rowcount
        if registered:
            event(log, "audit", action="register_schema", schema_ref=declaration.schema_ref)
        self._declarations[declaration.schema_ref] = declaration

    def _decision(self, store: InMemoryExperienceStore, experience_id: str) -> str:
        stored = store.get_experience(experience_id)
        outcome = stored.experience.outcome if stored is not None else None
        return outcome.decision if outcome is not None else ""

    def _outcome(self, value: Any, declaration: ExperienceDeclaration) -> str:
        outcome = MIXED_OUTCOME if value is None else _required_text(value, "outcome")
        allowed = (*declaration.decisions, MIXED_OUTCOME)
        if outcome not in allowed:
            raise HTTPServiceError(f"outcome must be one of: {', '.join(allowed)}")
        return outcome

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

        Every write is recorded with its source and result, a refused one included.
        """

        if declaration is not None:
            if declaration.schema_ref != experience.schema_ref:
                raise HTTPServiceError("declaration does not derive the Experience schema_ref")
            self.register(declaration)
        schema = self.declaration_for(experience.schema_ref)
        if experience.status is not ExperienceStatus.COMPLETE:
            raise CompleteExperienceRequired("a write requires a complete Experience")
        schema.validate(experience)
        data = canonical_experience_bytes(experience)
        content_hash = hashlib.sha256(data).hexdigest()
        with self._database.transaction() as connection:
            status = self._existing(connection, experience.id, content_hash)
            if not status:
                connection.execute("SELECT 1 FROM kbs WHERE kb_id = %s FOR UPDATE", (self.kb_id,))
                status = self._existing(connection, experience.id, content_hash)
            if not status:
                position = connection.execute(
                    "UPDATE kbs SET last_sequence = last_sequence + 1 WHERE kb_id = %s RETURNING last_sequence",
                    (self.kb_id,),
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
                        _utc_now(),
                    ),
                )
                self._schema_row(connection, experience.schema_ref, changes_reads=True)
                status = InsertStatus.CREATED.value
            connection.execute(
                """
                INSERT INTO writes(kb_id, experience_id, schema_ref, result, source_kb_id, source_name, request_id,
                    received_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    self.kb_id,
                    experience.id,
                    experience.schema_ref,
                    status,
                    source.kb_id,
                    source.name,
                    source.request_id,
                    _utc_now(),
                ),
            )
        self.metrics.count("hyperloom_kb_writes_total", schema_ref=experience.schema_ref, result=status)
        if status == _CONFLICT:
            raise ImmutableExperienceConflict("Experience id already exists with different content")
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

    def read(
        self,
        *,
        decision: str,
        context: dict[str, JsonValue],
        outcome: str | None = None,
        limit: int = DEFAULT_READ_LIMIT,
        schema_ref: str | None = None,
        content_inline_limit: int | None = None,
        render_budget_chars: int | None = None,
    ) -> dict[str, JsonValue]:
        """Search one schema's Experiences, this service's default schema unless ``schema_ref`` names another.

        With ``content_inline_limit``, a free-text field over that many bytes is rendered as a reference and its
        text is returned under ``contents``. With ``render_budget_chars``, the block carries whole records while
        they fit and names only those in ``rendered_refs``.
        """

        decision = _required_text(decision, "decision")
        declaration = self.declaration_for(schema_ref or self.declaration.schema_ref)
        selected_outcome = self._outcome(outcome, declaration)
        limit = _bounded_int(limit, "limit", minimum=1, maximum=MAX_READ_LIMIT)
        if content_inline_limit is not None:
            content_inline_limit = _bounded_int(content_inline_limit, "content_inline_limit", minimum=0)
        if render_budget_chars is not None:
            render_budget_chars = _bounded_int(render_budget_chars, "render_budget_chars", minimum=1)
        read_view = self._read_view(declaration)
        store, view = read_view.store, read_view.view
        if selected_outcome != MIXED_OUTCOME:
            view = QueryViewBuilder().restrict(
                view,
                (
                    experience_id
                    for experience_id in view.visible_experience_ids
                    if self._decision(store, experience_id) == selected_outcome
                ),
            )
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
            "contents": [],
        }
        if eligible_count == 0:
            return response
        external: dict[str, str] = {}

        views = InMemoryQueryViewStore()
        views.publish_view(view)
        provider_refs = {RetrievalCapability.FUZZY: LEXICAL_FUZZY_PROVIDER_REF}
        configuration = RetrievalConfiguration.create(
            declaration.schema_ref,
            READ_POLICY_VERSION,
            limits={capability: eligible_count for capability in RetrievalCapability},
            provider_refs=provider_refs,
            ranking_policy_ref="weighted-signal-sum@v1",
            max_groups=limit,
            render_budget_chars=render_budget_chars,
        )
        executor = QueryExecutor(
            LocalRetrievalService(
                store,
                views,
                providers=(LexicalFuzzyProvider(store),),
                renderer=partial(render_complete_experience, inline_limit=content_inline_limit, external=external),
            ),
            provider_refs=provider_refs,
        )
        traces: list[ReadTrace] = []
        result = KnowledgeReadService(
            declaration,
            executor,
            configuration,
            planner=self._planner,
            trace_sink=traces.append,
        ).read(decision, context)
        groups = {
            experience_id: group
            for group in (traces[-1].execution.groups if traces else ())
            for experience_id in group.representative_ids
        }
        experiences: list[JsonValue] = []
        for reference in result.rendered_refs:
            stored = store.get_experience(reference.id)
            item = _summary(stored.experience if stored is not None else self.held(reference.id))
            group = groups.get(reference.id)
            item["score"] = group.score if group is not None else 0.0
            item["why_matched"] = [
                contribution.to_dict()
                for contribution in (group.contributions if group is not None else ())
                if contribution.experience_id == reference.id
            ]
            experiences.append(item)
        # Records the budget left out were rendered too; only what the block references comes back.
        contents: list[JsonValue] = [
            {"ref": ref, "bytes": len(text.encode()), "content": text}
            for ref, text in external.items()
            if ref in result.prompt_block
        ]
        response.update(
            {
                "status": result.status.value,
                "prompt_block": result.prompt_block,
                "rendered_refs": [reference.to_dict() for reference in result.rendered_refs],
                "experiences": experiences,
                "rendered_count": len(result.rendered_refs),
                "warnings": list(result.warnings),
                "contents": contents,
            }
        )
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
            {"sequence": sequence, **_summary(experience)}
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
                WHERE kb_id = %s AND (%s::text IS NULL OR schema_ref = %s) AND sequence > %s
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
            WHERE kb_id = %s AND (%s::text IS NULL OR schema_ref = %s)
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
        with self._database.transaction() as connection:
            page: dict[str, JsonValue] = {
                "items": items,
                "next_cursor": next_cursor,
                "has_more": has_more,
                "head": self._head(connection, schema_ref),
                "after_id": self._written_at(connection, after),
                "next_cursor_id": self._written_at(connection, next_cursor),
            }
            if schema_ref is not None:
                page["declaration"] = self.declaration_for(schema_ref).to_dict()
                page["state"] = self._state.hidden_digest(connection, schema_ref)
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
        event(log, "audit", action="label", schema_ref=schema, label_id=label.label_id, name=label.name)
        return label.to_dict()

    def delete_label(self, label_id: str) -> dict[str, JsonValue]:
        with self._database.transaction() as connection:
            self._state.delete_label(connection, label_id)
        event(log, "audit", action="delete_label", label_id=label_id)
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
            "audit",
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
        event(log, "audit", action="exclude", schema_ref=schema, experience_id=experience_id, reason=reason)
        return {"experience_id": experience_id, "status": "excluded"}

    def include(self, experience_id: str) -> dict[str, JsonValue]:
        with self._database.transaction() as connection:
            schema = self._schema_of(connection, experience_id)
            self._schema_row(connection, schema, changes_reads=True)
            lifted = self._state.include(connection, schema, experience_id)
        event(log, "audit", action="include", schema_ref=schema, experience_id=experience_id, lifted=lifted)
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
        event(log, "audit", action="rebind", **result)
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
            ("hyperloom_kb_disk_free_bytes", {}, shutil.disk_usage(self.config.home).free),
        ]
        gauges += [("hyperloom_kb_ready", {"check": check}, float(ok)) for check, ok in readiness.items()]
        gauges += [
            ("hyperloom_kb_database_pool", {"stat": stat}, float(value))
            for stat, value in sorted(self._database.pool.get_stats().items())
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
                    COUNT(*) FILTER (WHERE outside.experience_id IS NULL AND exclusions.experience_id IS NULL) AS visible,
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
                """
                SELECT COALESCE(SUM(bytes), 0) AS records, MAX(stored_at) AS last_write,
                    pg_database_size(current_database()) AS database
                FROM experiences WHERE kb_id = %s
                """,
                (self.kb_id,),
            ).fetchone()
        if sizes is not None:
            gauges.append(("hyperloom_kb_record_bytes", {}, float(sizes["records"])))
            gauges.append(("hyperloom_kb_database_bytes", {}, float(sizes["database"])))
            if sizes["last_write"]:
                written = datetime.fromisoformat(str(sizes["last_write"]).replace("Z", "+00:00"))
                gauges.append(("hyperloom_kb_last_write_timestamp_seconds", {}, written.timestamp()))
        return gauges


class ExperienceHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    app: ExperienceHTTPService


class RequestHandler(BaseHTTPRequestHandler):
    """JSON HTTP boundary; terminate TLS in front of this process when required."""

    server: ExperienceHTTPServer

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _write(self, status: HTTPStatus, value: Mapping[str, JsonValue]) -> None:
        self._send(status, (_canonical(dict(value)) + "\n").encode(), "application/json")

    def _send(self, status: HTTPStatus, data: bytes, content_type: str) -> None:
        context = current_request.get()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        if context is not None:
            self.send_header(REQUEST_ID_HEADER, context.request_id)
        self.end_headers()
        self.wfile.write(data)
        self._answered = (int(status), len(data))

    def _authorized(self) -> bool:
        supplied = self.headers.get("Authorization", "")
        return hmac.compare_digest(supplied, f"Bearer {self.server.app.config.token}")

    def _body(self) -> dict[str, JsonValue]:
        try:
            length = int(self.headers.get("Content-Length", ""))
        except ValueError as exc:
            raise HTTPServiceError("Content-Length is required") from exc
        if not 0 <= length <= _MAX_REQUEST_BYTES:
            raise HTTPServiceError("request body is too large")
        return _json_object(json.loads(self.rfile.read(length)), "request body")

    def _probe(self, path: str) -> bool:
        """Answer the unauthenticated probes an orchestrator and a metrics scraper send; ``False`` for any other path.

        They carry nothing a KB holds: which checks pass, and counts and sizes.
        """

        app = self.server.app
        if self.command != "GET" or path not in PROBE_ROUTES:
            return False
        if path == "/livez":
            self._write(HTTPStatus.OK, {"status": "alive"})
        elif path == "/readyz":
            checks = app.readiness()
            ready = all(checks.values())
            self._write(
                HTTPStatus.OK if ready else HTTPStatus.SERVICE_UNAVAILABLE,
                {"ready": ready, "checks": {check: "ok" if ok else "failed" for check, ok in checks.items()}},
            )
        else:
            text = app.metrics.render(app.metric_gauges())
            self._send(HTTPStatus.OK, text.encode(), "text/plain; version=0.0.4; charset=utf-8")
        return True

    def _dispatch(self) -> None:
        app = self.server.app
        parsed = urlsplit(self.path)
        if self._probe(parsed.path):
            return
        if not self._authorized():
            app.metrics.count("hyperloom_kb_http_unauthorized_total")
            self._write(HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"})
            return
        query = parse_qs(parsed.query)
        if self.command == "GET" and parsed.path == "/health":
            self._write(HTTPStatus.OK, app.health())
            return
        if self.command == "GET" and parsed.path == "/v1/list":
            self._write(
                HTTPStatus.OK,
                app.list_experiences(
                    after=_query_int(query, "after", 0),
                    limit=_query_int(query, "limit", 100),
                    schema_ref=_query_text(query, "schema_ref"),
                    include_excluded=_query_flag(query, "include_excluded"),
                ),
            )
            return
        if self.command == "GET" and parsed.path == "/v1/export":
            self._write(
                HTTPStatus.OK,
                app.export(
                    after=_query_int(query, "after", 0),
                    limit=_query_int(query, "limit", MAX_EXPORT_LIMIT),
                    schema_ref=_query_text(query, "schema_ref"),
                    include_excluded=_query_flag(query, "include_excluded"),
                ),
            )
            return
        if self._dispatch_state(app, parsed.path, query):
            return
        if self.command == "POST" and parsed.path == "/v1/push":
            _reject_unknown(self._body(), frozenset())
            self._write(HTTPStatus.OK, app.push())
            return
        if self.command == "POST" and parsed.path == "/v1/pull":
            body = self._body()
            _reject_unknown(body, frozenset({"schema_ref"}))
            self._write(HTTPStatus.OK, app.pull(_required_text(body.get("schema_ref"), "schema_ref")))
            return
        if self.command == "POST" and parsed.path == "/v1/rebind":
            _reject_unknown(self._body(), frozenset())
            self._write(HTTPStatus.OK, app.rebind())
            return
        if self.command == "POST" and parsed.path == "/v1/read":
            body = self._body()
            _reject_unknown(body, _READ_FIELDS)
            outcome = body.get("outcome")
            schema_ref = body.get("schema_ref")
            self._write(
                HTTPStatus.OK,
                app.read(
                    decision=_required_text(body.get("decision"), "decision"),
                    context=_json_object(body.get("context", {}), "context"),
                    outcome=None if outcome is None else _required_text(outcome, "outcome"),
                    limit=_bounded_int(
                        body.get("limit", DEFAULT_READ_LIMIT),
                        "limit",
                        minimum=1,
                        maximum=MAX_READ_LIMIT,
                    ),
                    schema_ref=None if schema_ref is None else _required_text(schema_ref, "schema_ref"),
                    content_inline_limit=cast(int | None, body.get("content_inline_limit")),
                    render_budget_chars=cast(int | None, body.get("render_budget_chars")),
                ),
            )
            return
        prefix = "/v1/experiences/"
        if self.command == "PUT" and parsed.path.startswith(prefix):
            body = self._body()
            _reject_unknown(body, _WRITE_FIELDS)
            experience = Experience.from_dict(body.get("experience"))
            if experience.id != parsed.path.removeprefix(prefix):
                raise HTTPServiceError("Experience path id differs from payload")
            declaration = body.get("declaration")
            context = current_request.get() or RequestContext("")
            self._write(
                HTTPStatus.OK,
                app.write(
                    experience,
                    None if declaration is None else ExperienceDeclaration.from_dict(declaration),
                    source=WriteSource(context.client_kb_id, context.client_name, context.request_id),
                ),
            )
            return
        self._write(HTTPStatus.NOT_FOUND, {"error": "not_found"})

    def _dispatch_state(self, app: ExperienceHTTPService, path: str, query: Mapping[str, list[str]]) -> bool:
        """Serve the label and exclusion routes; ``False`` when ``path`` is none of them."""

        labels, exclusions = "/v1/labels", "/v1/exclusions"
        if self.command == "GET" and path == labels:
            self._write(HTTPStatus.OK, app.labels(_query_text(query, "schema_ref")))
        elif self.command == "POST" and path == labels:
            body = self._body()
            _reject_unknown(body, frozenset({"schema_ref", "name"}))
            schema_ref, name = body.get("schema_ref"), body.get("name", "")
            if not isinstance(name, str):
                raise HTTPServiceError("name must be a string")
            self._write(
                HTTPStatus.OK,
                app.create_label(
                    schema_ref=None if schema_ref is None else _required_text(schema_ref, "schema_ref"),
                    name=name.strip(),
                ),
            )
        elif self.command == "DELETE" and path.startswith(f"{labels}/"):
            self._write(HTTPStatus.OK, app.delete_label(path.removeprefix(f"{labels}/")))
        elif self.command == "POST" and path == "/v1/restore":
            body = self._body()
            _reject_unknown(body, frozenset({"label_id"}))
            self._write(HTTPStatus.OK, app.restore(_required_text(body.get("label_id"), "label_id")))
        elif self.command == "GET" and path == exclusions:
            self._write(HTTPStatus.OK, app.exclusions(_query_text(query, "schema_ref")))
        elif self.command == "POST" and path == exclusions:
            body = self._body()
            _reject_unknown(body, frozenset({"experience_id", "reason"}))
            self._write(
                HTTPStatus.OK,
                app.exclude(
                    _required_text(body.get("experience_id"), "experience_id"),
                    _required_text(body.get("reason"), "reason"),
                ),
            )
        elif self.command == "DELETE" and path.startswith(f"{exclusions}/"):
            self._write(HTTPStatus.OK, app.include(path.removeprefix(f"{exclusions}/")))
        else:
            return False
        return True

    def _handle(self) -> None:
        """Answer one request, counted and timed, with its context on every line logged while answering it."""

        token = current_request.set(RequestContext.from_headers(self.headers))
        self._answered = (0, 0)
        started = time.monotonic()
        metrics = self.server.app.metrics
        try:
            with metrics.in_flight():
                self._answer()
        finally:
            route = route_of(urlsplit(self.path).path)
            status, sent = self._answered
            seconds = time.monotonic() - started
            received = int(self.headers.get("Content-Length") or 0) if self.command in ("POST", "PUT") else 0
            metrics.observe_request(
                method=self.command, route=route, status=status, seconds=seconds, bytes_in=received, bytes_out=sent
            )
            if route not in PROBE_ROUTES:
                event(
                    log,
                    "http_request",
                    level=logging.WARNING if status >= 500 else logging.INFO,
                    method=self.command,
                    route=route,
                    status=status,
                    duration_ms=round(seconds * 1000, 3),
                    bytes_in=received,
                    bytes_out=sent,
                )
            current_request.reset(token)

    def _answer(self) -> None:
        try:
            self._dispatch()
        except UnknownStateItem as exc:
            self._write(HTTPStatus.NOT_FOUND, {"error": "not_found", "detail": str(exc)})
        except ImmutableExperienceConflict as exc:
            self._write(HTTPStatus.CONFLICT, {"error": "conflict", "detail": str(exc)})
        except SyncUnavailable as exc:
            self._write(HTTPStatus.CONFLICT, {"error": "sync_unavailable", "detail": str(exc)})
        except (ValueError, StorageContractError) as exc:
            self._write(HTTPStatus.BAD_REQUEST, {"error": "invalid_request", "detail": str(exc)})
        except Exception as exc:
            path = self.path.replace("\r", "").replace("\n", "")
            log.exception("Experience service request failed: %s %s", self.command, path)
            self._write(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": "internal_error", "detail": type(exc).__name__},
            )

    def do_GET(self) -> None:
        self._handle()

    def do_POST(self) -> None:
        self._handle()

    def do_PUT(self) -> None:
        self._handle()

    def do_DELETE(self) -> None:
        self._handle()


def create_http_server(
    app: ExperienceHTTPService,
    host: str,
    port: int,
) -> ExperienceHTTPServer:
    server = ExperienceHTTPServer((host, port), RequestHandler)
    server.app = app
    return server


@contextmanager
def _sole_service(home: Path) -> Iterator[TextIO]:
    """Hold ``home`` for this process, or exit naming the service that does.

    The holder starts the home's embedded database and stops it when it stops serving, so a second service on the
    same home would lose its database under it.
    """

    import fcntl

    home.mkdir(parents=True, exist_ok=True)
    with (home / SERVICE_LOCK).open("a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lock.seek(0)
            holder = lock.read().strip() or "another process"
            raise SystemExit(
                f"another Experience service ({holder}) already serves {home}; start this one with another --home"
            ) from None
        yield lock


def _record_holder(lock: TextIO, port: int) -> None:
    lock.seek(0)
    lock.truncate()
    lock.write(f"pid {os.getpid()}, port {port}\n")
    lock.flush()


def serve_until_stopped(server: ExperienceHTTPServer, stop: threading.Event, drain_seconds: float) -> None:
    """Serve until ``stop`` is set, then stop taking requests and let those in flight finish for ``drain_seconds``."""

    threading.Thread(target=lambda: (stop.wait(), server.shutdown()), daemon=True).start()
    server.serve_forever()
    deadline = time.monotonic() + drain_seconds
    while server.app.metrics.requests_in_flight() and time.monotonic() < deadline:
        time.sleep(0.05)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--declaration", type=Path, default=PACKAGED_DECLARATION)
    parser.add_argument("--home", type=Path, default=DEFAULT_HOME)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--name", default="", help="A display name for this service; its identity stays its kb_id.")
    parser.add_argument(
        "--seed-jsonl",
        type=Path,
        action="append",
        default=[],
        help="Import complete Experiences from JSONL before serving; repeatable.",
    )
    args = parser.parse_args(argv)
    handler = logging.StreamHandler()
    handler.setFormatter(JsonLogFormatter())
    logging.basicConfig(level=logging.INFO, handlers=[handler])

    settings = ServiceSettings.from_env(os.environ)
    planner: LLMQueryPlanner | None = None
    if settings.planner is None:
        log.warning(
            "Experience reads are unavailable until a planner gateway is configured: %s", settings.planner_problem
        )
    else:
        planner = LLMQueryPlanner(
            AnthropicPlannerBackend(settings.planner), PlannerConfiguration.create(settings.planner.model)
        )
    if settings.global_problem:
        log.warning("Push and pull are unavailable: %s", settings.global_problem)
    # A service manager stops the service with SIGTERM: it stops taking requests, lets the ones in flight finish, and
    # stops the database this process started rather than orphaning it.
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda _signum, _frame: stop.set())
    home = args.home.expanduser()
    with ExitStack() as stack:
        lock: TextIO | None = None
        database_url = settings.database_url
        if not database_url:
            reason = unavailable_reason()
            if reason:
                raise SystemExit(f"{reason}; set {DATABASE_URL_ENV} to a PostgreSQL server")
            lock = stack.enter_context(_sole_service(home))
            try:
                embedded = start_embedded_postgres(home)
            except EmbeddedPostgresError as exc:
                raise SystemExit(f"{exc}; or set {DATABASE_URL_ENV} to a PostgreSQL server") from None
            if embedded.started:
                stack.callback(embedded.stop)
            database_url = embedded.conninfo
        database = Database(database_url)
        stack.callback(database.close)
        app = ExperienceHTTPService(
            HTTPServiceConfig(home, os.environ.get("HYPERLOOM_KB_TOKEN", "")),
            load_declaration(args.declaration),
            planner,
            database=database,
            global_kb=None if settings.global_kb is None else RemoteClient(settings.global_kb),
            config_digest=settings.digest(),
            name=args.name,
        )
        for seed in args.seed_jsonl:
            log.info("seeded %s: %s", seed, _canonical(app.seed(seed)))
        server = create_http_server(app, args.host, args.port)
        if lock is not None:
            _record_holder(lock, server.server_address[1])
        event(
            log,
            "listening",
            host=args.host,
            port=server.server_address[1],
            home=str(home),
            kb_id=app.kb_id,
            schema_ref=app.declaration.schema_ref,
            code_digest=code_digest(),
        )
        with server, suppress(KeyboardInterrupt):
            serve_until_stopped(server, stop, DRAIN_SECONDS)
        event(log, "stopped", kb_id=app.kb_id)
    return 0


__all__ = [
    "DEFAULT_READ_LIMIT",
    "MAX_EXPORT_LIMIT",
    "MAX_READ_LIMIT",
    "MIXED_OUTCOME",
    "ExperienceHTTPServer",
    "DATABASE_URL_ENV",
    "ExperienceHTTPService",
    "HTTPServiceConfig",
    "HTTPServiceError",
    "RequestHandler",
    "ServiceSettings",
    "code_digest",
    "create_http_server",
    "main",
    "serve_until_stopped",
]


if __name__ == "__main__":
    raise SystemExit(main())
