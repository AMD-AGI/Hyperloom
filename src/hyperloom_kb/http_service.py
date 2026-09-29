# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Authenticated HTTP service for one shared Experience corpus."""

from __future__ import annotations

import argparse
import hmac
import json
import logging
import os
import sqlite3
import threading
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

from hyperloom_kb.config import PACKAGED_DECLARATION, load_declaration
from hyperloom_kb.knowledge_read import (
    AnthropicPlannerBackend,
    KnowledgeReadService,
    LLMQueryPlanner,
    PlannerConfiguration,
    PlannerGatewayConfig,
    QueryExecutor,
    QueryPlanValidationError,
    ReadStatus,
    ReadTrace,
)
from hyperloom_kb.query_view import (
    InMemoryQueryViewStore,
    QueryView,
    QueryViewBuilder,
    RetrievalCapability,
)
from hyperloom_kb.retrieval import LocalRetrievalService, render_complete_experience
from hyperloom_kb.retrieval_policy import (
    LEXICAL_FUZZY_PROVIDER_REF,
    LexicalFuzzyProvider,
    RetrievalConfiguration,
)
from hyperloom_kb.schema import Experience, ExperienceDeclaration, JsonValue
from hyperloom_kb.service import (
    CompleteExperienceRequired,
    ExperienceService,
    UnknownSchemaRef,
)
from hyperloom_kb.storage import (
    ImmutableExperienceConflict,
    InMemoryExperienceStore,
    InsertStatus,
    LocalExperienceStore,
    LocalSchemaRegistry,
    StoredExperience,
)

log = logging.getLogger(__name__)

MIXED_OUTCOME = "mixed"
DEFAULT_READ_LIMIT = 10
MAX_READ_LIMIT = 100
MAX_LIST_LIMIT = 500
READ_POLICY_VERSION = "shared-experience-read@v1"
DEFAULT_HOME = Path("~/.local/share/hyperloom-kb").expanduser()
_MAX_REQUEST_BYTES = 2 * 1024 * 1024
_READ_FIELDS = frozenset({"decision", "context", "outcome", "limit"})
_WRITE_FIELDS = frozenset({"experience"})


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
        schema_ref: str,
        *,
        after: int,
        limit: int,
    ) -> tuple[tuple[tuple[int, str], ...], int, bool]:
        with self._lock, self._connection() as connection:
            rows = connection.execute(
                """
                SELECT sequence, experience_id FROM experiences
                WHERE schema_ref = ? AND sequence > ?
                ORDER BY sequence LIMIT ?
                """,
                (schema_ref, after, limit + 1),
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
    """Write, read, and list one authoritative shared Experience corpus."""

    def __init__(
        self,
        config: HTTPServiceConfig,
        declaration: ExperienceDeclaration,
        planner: LLMQueryPlanner | None,
    ) -> None:
        self.config = config
        self.declaration = declaration
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
        records = self._store.list_experiences(declaration.schema_ref)
        indexed = self._index.indexed_ids(declaration.schema_ref)
        for record in sorted(records, key=_completion_order):
            self._mirror.insert_complete(record.experience)
            if record.experience.id not in indexed:
                self._index.register(record.experience.id, declaration.schema_ref)
        self._view = self._build_view()

    def _build_view(self) -> QueryView:
        return QueryViewBuilder().build(
            self.declaration,
            self._mirror.list_experiences(self.declaration.schema_ref),
            fuzzy_ready=True,
        )

    def _stored(self, experience_id: str) -> Experience:
        stored = self._mirror.get_experience(experience_id)
        if stored is None:
            raise RuntimeError(f"Experience {experience_id} is missing from the service corpus")
        return stored.experience

    def _decision(self, experience_id: str) -> str:
        outcome = self._stored(experience_id).outcome
        return outcome.decision if outcome is not None else ""

    def _outcome(self, value: Any) -> str:
        outcome = MIXED_OUTCOME if value is None else _required_text(value, "outcome")
        allowed = (*self.declaration.decisions, MIXED_OUTCOME)
        if outcome not in allowed:
            raise HTTPServiceError(f"outcome must be one of: {', '.join(allowed)}")
        return outcome

    def write(self, experience: Experience) -> dict[str, JsonValue]:
        if experience.schema_ref != self.declaration.schema_ref:
            raise HTTPServiceError(
                f"Experience schema_ref must be the service declaration {self.declaration.schema_ref}"
            )
        with self._write_lock:
            result = self._experience_service.submit_complete(experience)
            if self._mirror.get_experience(experience.id) is None:
                self._mirror.insert_complete(result.record.experience)
                self._view = self._build_view()
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
    ) -> dict[str, JsonValue]:
        decision = _required_text(decision, "decision")
        selected_outcome = self._outcome(outcome)
        limit = _bounded_int(limit, "limit", minimum=1, maximum=MAX_READ_LIMIT)
        view = self._view
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
        }
        if eligible_count == 0:
            return response

        views = InMemoryQueryViewStore()
        views.publish_view(view)
        provider_refs = {RetrievalCapability.FUZZY: LEXICAL_FUZZY_PROVIDER_REF}
        configuration = RetrievalConfiguration.create(
            self.declaration.schema_ref,
            READ_POLICY_VERSION,
            limits={capability: eligible_count for capability in RetrievalCapability},
            provider_refs=provider_refs,
            ranking_policy_ref="weighted-signal-sum@v1",
            max_groups=limit,
            render_budget_chars=None,
        )
        executor = QueryExecutor(
            LocalRetrievalService(
                self._mirror,
                views,
                providers=(LexicalFuzzyProvider(self._mirror),),
                renderer=render_complete_experience,
            ),
            provider_refs=provider_refs,
        )
        traces: list[ReadTrace] = []
        result = KnowledgeReadService(
            self.declaration,
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
        response.update(
            {
                "status": result.status.value,
                "prompt_block": result.prompt_block,
                "rendered_refs": [reference.to_dict() for reference in result.rendered_refs],
                "experiences": experiences,
                "rendered_count": len(result.rendered_refs),
                "warnings": list(result.warnings),
            }
        )
        return response

    def list_experiences(
        self,
        *,
        after: int = 0,
        limit: int = 100,
    ) -> dict[str, JsonValue]:
        after = _bounded_int(after, "after", minimum=0)
        limit = _bounded_int(limit, "limit", minimum=1, maximum=MAX_LIST_LIMIT)
        page, next_cursor, has_more = self._index.page(
            self.declaration.schema_ref,
            after=after,
            limit=limit,
        )
        items: list[JsonValue] = [
            {"sequence": sequence, **_summary(self._stored(experience_id))} for sequence, experience_id in page
        ]
        return {"items": items, "next_cursor": next_cursor, "has_more": has_more}

    def health(self) -> dict[str, JsonValue]:
        return {
            "status": "ok",
            "schema_ref": self.declaration.schema_ref,
            "experience_count": len(self._view.visible_experience_ids),
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
                ),
            )
            return
        if self.command == "POST" and parsed.path == "/v1/read":
            body = self._body()
            _reject_unknown(body, _READ_FIELDS)
            outcome = body.get("outcome")
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
            self._write(HTTPStatus.OK, app.write(experience))
            return
        self._write(HTTPStatus.NOT_FOUND, {"error": "not_found"})

    def _handle(self) -> None:
        try:
            self._dispatch()
        except ImmutableExperienceConflict as exc:
            self._write(HTTPStatus.CONFLICT, {"error": "conflict", "detail": str(exc)})
        except (ValueError, UnknownSchemaRef, CompleteExperienceRequired) as exc:
            self._write(HTTPStatus.BAD_REQUEST, {"error": "invalid_request", "detail": str(exc)})
        except Exception as exc:
            log.exception("Experience service request failed: %s %s", self.command, self.path)
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

    planner: LLMQueryPlanner | None
    try:
        gateway = PlannerGatewayConfig.from_env(os.environ)
    except QueryPlanValidationError as exc:
        log.warning("Experience reads are unavailable until a planner gateway is configured: %s", exc)
        planner = None
    else:
        planner = LLMQueryPlanner(AnthropicPlannerBackend(gateway), PlannerConfiguration.create(gateway.model))
    app = ExperienceHTTPService(
        HTTPServiceConfig(args.home, os.environ.get("HYPERLOOM_KB_TOKEN", "")),
        load_declaration(args.declaration),
        planner,
    )
    for seed in args.seed_jsonl:
        log.info("seeded %s: %s", seed, _canonical(app.seed(seed)))
    server = create_http_server(app, args.host, args.port)
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
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


__all__ = [
    "DEFAULT_READ_LIMIT",
    "MAX_READ_LIMIT",
    "MIXED_OUTCOME",
    "ExperienceHTTPServer",
    "ExperienceHTTPService",
    "ExperienceIndex",
    "HTTPServiceConfig",
    "HTTPServiceError",
    "RequestHandler",
    "create_http_server",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
