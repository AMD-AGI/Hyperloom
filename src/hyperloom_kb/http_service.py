# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Authenticated HTTP service for one shared Experience corpus."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import logging
import os
import sqlite3
import threading
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, TextIO, cast
from urllib.parse import parse_qs, urlsplit

from hyperloom_kb.config import PACKAGED_DECLARATION, load_declaration
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
from hyperloom_kb.schema import Experience, ExperienceDeclaration, JsonValue
from hyperloom_kb.service import ExperienceService
from hyperloom_kb.storage import (
    ImmutableExperienceConflict,
    StorageContractError,
    InMemoryExperienceStore,
    InsertStatus,
    LocalExperienceStore,
    LocalSchemaRegistry,
    StoredExperience,
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
# Held by the one service process that serves a home, for as long as it serves it.
SERVICE_LOCK = "service.lock"
# A transport guard, not a data policy: Experiences of any size are stored.
_MAX_REQUEST_BYTES = 256 * 1024 * 1024
_READ_FIELDS = frozenset(
    {"decision", "context", "outcome", "limit", "schema_ref", "content_inline_limit", "render_budget_chars"}
)
_WRITE_FIELDS = frozenset({"experience", "declaration"})


class HTTPServiceError(ValueError):
    """Raised when a request or service configuration is invalid."""


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
        return cls(planner, planner_problem, global_kb, global_problem)

    def digest(self) -> str:
        """Equal digests mean two environments start behaviorally identical services, whatever their spelling."""

        planner = self.planner
        values = {
            "planner": None
            if planner is None
            else [planner.base_url, planner.api_key, planner.model, planner.timeout_seconds, planner.max_output_tokens],
            "global": None if self.global_kb is None else [self.global_kb.base_url, self.global_kb.token],
            "problems": [self.planner_problem, self.global_problem],
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


class ExperienceIndex:
    """Durable write order for list paging, rebuildable from canonical storage."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        with self._lock, self._connection() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS experiences (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    experience_id TEXT NOT NULL UNIQUE,
                    schema_ref TEXT NOT NULL,
                    indexed_at TEXT NOT NULL
                )
                """
            )

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def register(self, experience_id: str, schema_ref: str) -> None:
        with self._lock, self._connection() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO experiences(experience_id, schema_ref, indexed_at)
                VALUES (?, ?, ?)
                """,
                (experience_id, schema_ref, _utc_now()),
            )

    def indexed_ids(self, schema_ref: str) -> frozenset[str]:
        with self._lock, self._connection() as connection:
            rows = connection.execute(
                "SELECT experience_id FROM experiences WHERE schema_ref = ?",
                (schema_ref,),
            ).fetchall()
        return frozenset(str(row["experience_id"]) for row in rows)

    def page(
        self,
        schema_ref: str | None,
        *,
        after: int,
        limit: int,
    ) -> tuple[tuple[tuple[int, str], ...], int, bool]:
        """One page in write order, of one schema or, with ``schema_ref=None``, of every schema."""

        with self._lock, self._connection() as connection:
            rows = connection.execute(
                """
                SELECT sequence, experience_id FROM experiences
                WHERE (? IS NULL OR schema_ref = ?) AND sequence > ?
                ORDER BY sequence LIMIT ?
                """,
                (schema_ref, schema_ref, after, limit + 1),
            ).fetchall()
        page = tuple((int(row["sequence"]), str(row["experience_id"])) for row in rows[:limit])
        return page, page[-1][0] if page else after, len(rows) > limit


def _summary(experience: Experience) -> dict[str, JsonValue]:
    return {
        "experience_id": experience.id,
        "source_run_id": experience.run_id,
        "change_summary": experience.change.summary if experience.change is not None else "",
        "decision": experience.outcome.decision if experience.outcome is not None else "",
        "baseline_value": experience.baseline_value,
        "outcome_value": experience.outcome.value if experience.outcome is not None else None,
    }


def _completion_order(record: StoredExperience) -> tuple[str, str]:
    experience = record.experience
    return (
        (experience.completed_at or experience.created_at).isoformat(),
        experience.id,
    )


class ExperienceHTTPService:
    """Write, read, and list one authoritative Experience corpus holding any number of schemas.

    ``declaration`` is the schema a read searches when it names none; every other registered schema is stored,
    listed, exported, and synced the same way.
    """

    def __init__(
        self,
        config: HTTPServiceConfig,
        declaration: ExperienceDeclaration,
        planner: LLMQueryPlanner | None,
        *,
        global_kb: RemoteClient | None = None,
        config_digest: str = "",
    ) -> None:
        self.config = config
        self.declaration = declaration
        self._config_digest = config_digest
        canonical_root = config.home / "canonical"
        self._store = LocalExperienceStore(canonical_root)
        self._experience_service = ExperienceService(
            LocalSchemaRegistry(canonical_root),
            self._store,
        )
        self._experience_service.register_schema(declaration)
        self._index = ExperienceIndex(config.home / "kb.sqlite3")
        self._planner = planner
        self._write_lock = threading.RLock()
        self._mirror = InMemoryExperienceStore()
        self._declarations: dict[str, ExperienceDeclaration] = {}
        self._views: dict[str, QueryView] = {}
        for registered in self._experience_service.list_schemas():
            self._load(registered)
        self._sync = GlobalSync(self, SyncLedger(config.home / "sync.sqlite3"), global_kb)

    def _load(self, declaration: ExperienceDeclaration) -> None:
        schema_ref = declaration.schema_ref
        records = self._store.list_experiences(schema_ref)
        indexed = self._index.indexed_ids(schema_ref)
        for record in sorted(records, key=_completion_order):
            self._mirror.insert_complete(record.experience)
            if record.experience.id not in indexed:
                self._index.register(record.experience.id, schema_ref)
        self._declarations[schema_ref] = declaration
        self._views[schema_ref] = self._build_view(declaration)

    def _build_view(self, declaration: ExperienceDeclaration) -> QueryView:
        return QueryViewBuilder().build(
            declaration,
            self._mirror.list_experiences(declaration.schema_ref),
            fuzzy_ready=True,
        )

    @property
    def schema_refs(self) -> tuple[str, ...]:
        return tuple(sorted(self._declarations))

    def declaration_for(self, schema_ref: str) -> ExperienceDeclaration:
        declaration = self._declarations.get(schema_ref)
        if declaration is None:
            raise HTTPServiceError(f"schema_ref {schema_ref} is not registered; write it with its declaration")
        return declaration

    def register(self, declaration: ExperienceDeclaration) -> None:
        with self._write_lock:
            if declaration.schema_ref not in self._declarations:
                self._experience_service.register_schema(declaration)
                self._load(declaration)

    def _stored(self, experience_id: str) -> Experience:
        stored = self._mirror.get_experience(experience_id)
        if stored is None:
            raise RuntimeError(f"Experience {experience_id} is missing from the service corpus")
        return stored.experience

    def _decision(self, experience_id: str) -> str:
        outcome = self._stored(experience_id).outcome
        return outcome.decision if outcome is not None else ""

    def _outcome(self, value: Any, declaration: ExperienceDeclaration) -> str:
        outcome = MIXED_OUTCOME if value is None else _required_text(value, "outcome")
        allowed = (*declaration.decisions, MIXED_OUTCOME)
        if outcome not in allowed:
            raise HTTPServiceError(f"outcome must be one of: {', '.join(allowed)}")
        return outcome

    def write(self, experience: Experience, declaration: ExperienceDeclaration | None = None) -> dict[str, JsonValue]:
        """Store one complete Experience; ``declaration`` registers its schema when this service lacks it."""

        if declaration is not None:
            if declaration.schema_ref != experience.schema_ref:
                raise HTTPServiceError("declaration does not derive the Experience schema_ref")
            self.register(declaration)
        schema = self.declaration_for(experience.schema_ref)
        with self._write_lock:
            result = self._experience_service.submit_complete(experience)
            if self._mirror.get_experience(experience.id) is None:
                self._mirror.insert_complete(result.record.experience)
                self._views[schema.schema_ref] = self._build_view(schema)
            self._index.register(experience.id, experience.schema_ref)
        return {
            "status": result.status.value,
            "experience_id": experience.id,
            "content_hash": result.record.content_hash,
        }

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
        view = self._views[declaration.schema_ref]
        if selected_outcome != MIXED_OUTCOME:
            view = QueryViewBuilder().restrict(
                view,
                (
                    experience_id
                    for experience_id in view.visible_experience_ids
                    if self._decision(experience_id) == selected_outcome
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
                self._mirror,
                views,
                providers=(LexicalFuzzyProvider(self._mirror),),
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
            item = _summary(self._stored(reference.id))
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

    def list_experiences(
        self,
        *,
        after: int = 0,
        limit: int = 100,
        schema_ref: str | None = None,
    ) -> dict[str, JsonValue]:
        after = _bounded_int(after, "after", minimum=0)
        limit = _bounded_int(limit, "limit", minimum=1, maximum=MAX_LIST_LIMIT)
        page, next_cursor, has_more = self._index.page(schema_ref, after=after, limit=limit)
        items: list[JsonValue] = [
            {"sequence": sequence, **_summary(self._stored(experience_id))} for sequence, experience_id in page
        ]
        return {"items": items, "next_cursor": next_cursor, "has_more": has_more}

    def records_after(
        self, after: int, limit: int, schema_ref: str | None = None
    ) -> tuple[tuple[tuple[int, Experience], ...], int, bool]:
        """Complete Experiences in write order after the ``after`` sequence, of one schema or of all of them."""

        page, next_cursor, has_more = self._index.page(schema_ref, after=after, limit=limit)
        return tuple((sequence, self._stored(experience_id)) for sequence, experience_id in page), next_cursor, has_more

    def export(
        self, *, after: int = 0, limit: int = MAX_EXPORT_LIMIT, schema_ref: str | None = None
    ) -> dict[str, JsonValue]:
        after = _bounded_int(after, "after", minimum=0)
        limit = _bounded_int(limit, "limit", minimum=1, maximum=MAX_EXPORT_LIMIT)
        records, next_cursor, has_more = self.records_after(after, limit, schema_ref)
        items: list[JsonValue] = [
            {"sequence": sequence, "experience": experience.to_dict()} for sequence, experience in records
        ]
        return {"items": items, "next_cursor": next_cursor, "has_more": has_more}

    def push(self) -> dict[str, JsonValue]:
        return self._sync.push()

    def pull(self) -> dict[str, JsonValue]:
        return self._sync.pull()

    def health(self) -> dict[str, JsonValue]:
        counts = {schema_ref: len(view.visible_experience_ids) for schema_ref, view in sorted(self._views.items())}
        return {
            "status": "ok",
            "schema_ref": self.declaration.schema_ref,
            "experience_count": sum(counts.values()),
            "schemas": dict(counts),
            "pid": os.getpid(),
            "config_digest": self._config_digest,
            "home": str(self.config.home.resolve()),
        }


class ExperienceHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    app: ExperienceHTTPService


class RequestHandler(BaseHTTPRequestHandler):
    """JSON HTTP boundary; terminate TLS in front of this process when required."""

    server: ExperienceHTTPServer

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _write(self, status: HTTPStatus, value: Mapping[str, JsonValue]) -> None:
        data = (_canonical(dict(value)) + "\n").encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

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

    def _dispatch(self) -> None:
        if not self._authorized():
            self._write(HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"})
            return
        app = self.server.app
        parsed = urlsplit(self.path)
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
                ),
            )
            return
        if self.command == "POST" and parsed.path in ("/v1/push", "/v1/pull"):
            _reject_unknown(self._body(), frozenset())
            self._write(HTTPStatus.OK, app.push() if parsed.path == "/v1/push" else app.pull())
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
            self._write(
                HTTPStatus.OK,
                app.write(experience, None if declaration is None else ExperienceDeclaration.from_dict(declaration)),
            )
            return
        self._write(HTTPStatus.NOT_FOUND, {"error": "not_found"})

    def _handle(self) -> None:
        try:
            self._dispatch()
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

    A service pages its index but answers from an in-memory mirror of the corpus, so a second service on the same
    home would serve, page, and push a different set of Experiences than the first.
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--declaration", type=Path, default=PACKAGED_DECLARATION)
    parser.add_argument("--home", type=Path, default=DEFAULT_HOME)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument(
        "--seed-jsonl",
        type=Path,
        action="append",
        default=[],
        help="Import complete Experiences from JSONL before serving; repeatable.",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

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
    with _sole_service(args.home.expanduser()) as lock:
        app = ExperienceHTTPService(
            HTTPServiceConfig(args.home, os.environ.get("HYPERLOOM_KB_TOKEN", "")),
            load_declaration(args.declaration),
            planner,
            global_kb=None if settings.global_kb is None else RemoteClient(settings.global_kb),
            config_digest=settings.digest(),
        )
        for seed in args.seed_jsonl:
            log.info("seeded %s: %s", seed, _canonical(app.seed(seed)))
        server = create_http_server(app, args.host, args.port)
        _record_holder(lock, server.server_address[1])
        print(
            _canonical(
                {
                    "event": "experience_kb_listening",
                    "host": args.host,
                    "port": server.server_address[1],
                    "home": str(args.home),
                    "schema_ref": app.declaration.schema_ref,
                }
            ),
            flush=True,
        )
        with server, suppress(KeyboardInterrupt):
            server.serve_forever()
    return 0


__all__ = [
    "DEFAULT_READ_LIMIT",
    "MAX_EXPORT_LIMIT",
    "MAX_READ_LIMIT",
    "MIXED_OUTCOME",
    "ExperienceHTTPServer",
    "ExperienceHTTPService",
    "ExperienceIndex",
    "HTTPServiceConfig",
    "HTTPServiceError",
    "RequestHandler",
    "ServiceSettings",
    "create_http_server",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
