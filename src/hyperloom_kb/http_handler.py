# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The Experience service's HTTP boundary: request parsing and errors, the request handler, and its server."""

from __future__ import annotations

import hmac
import json
import logging
import shutil
import threading
import time
from collections.abc import Collection, Mapping
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO, cast
from urllib.parse import parse_qs, urlsplit

from hyperloom_kb.database import WriteSource
from hyperloom_kb.local_state import UnknownStateItem
from hyperloom_kb.observability import (
    PROBE_ROUTES,
    REQUEST_ID_HEADER,
    RequestContext,
    ServedRequest,
    current_request,
    event,
    route_of,
)
from hyperloom_kb.remote import ReadQuery
from hyperloom_kb.schema import MAX_FILE_BYTES, Experience, ExperienceDeclaration, FileRef, JsonValue
from hyperloom_kb.storage import ImmutableExperienceConflict, StorageContractError
from hyperloom_kb.sync import SyncUnavailable

if TYPE_CHECKING:
    from hyperloom_kb.http_service import ExperienceHTTPService

log = logging.getLogger(__name__)

DEFAULT_READ_LIMIT = 10
MAX_READ_LIMIT = 100
# Complete records carry patch content, so an export page stays far smaller than a summary page.
MAX_EXPORT_LIMIT = 100
_MISSING_FILES = "missing_files"
# A transport guard, not a data policy: a record's text fields and the size of each file it names are bounded by its
# schema.
_MAX_REQUEST_BYTES = 256 * 1024**2
_READ_FIELDS = frozenset({"decision", "context", "outcome", "limit", "schema_ref", "render_budget_chars"})
_WRITE_FIELDS = frozenset({"experience", "declaration"})
_FILES_PREFIX = "/v1/files/"
_FILES_MISSING = "/v1/files/missing"
_GET, _POST, _PUT, _DELETE = "GET", "POST", "PUT", "DELETE"


class HTTPServiceError(ValueError):
    """Raised when a request or service configuration is invalid."""


class MissingFiles(RuntimeError):
    """Raised when a write names files this KB does not hold; send them first."""

    def __init__(self, missing: Collection[str]) -> None:
        super().__init__(f"send the files this record names before it: {', '.join(sorted(missing))}")
        self.missing = tuple(sorted(missing))


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


def _read_query(body: dict[str, JsonValue]) -> ReadQuery:
    _reject_unknown(body, _READ_FIELDS)
    outcome, schema_ref = body.get("outcome"), body.get("schema_ref")
    return ReadQuery(
        decision=_required_text(body.get("decision"), "decision"),
        context=_json_object(body.get("context", {}), "context"),
        outcome=None if outcome is None else _required_text(outcome, "outcome"),
        limit=_bounded_int(body.get("limit", DEFAULT_READ_LIMIT), "limit", minimum=1, maximum=MAX_READ_LIMIT),
        schema_ref=None if schema_ref is None else _required_text(schema_ref, "schema_ref"),
        render_budget_chars=cast(int | None, body.get("render_budget_chars")),
    )


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

    def _length(self, maximum: int) -> int:
        try:
            length = int(self.headers.get("Content-Length", ""))
        except ValueError as exc:
            raise HTTPServiceError("Content-Length is required") from exc
        if not 0 <= length <= maximum:
            raise HTTPServiceError("request body is too large")
        return length

    def _body(self) -> dict[str, JsonValue]:
        return _json_object(json.loads(self.rfile.read(self._length(_MAX_REQUEST_BYTES))), "request body")

    def _send_file(self, path: Path, size: int) -> None:
        context = current_request.get()
        with path.open("rb") as stream:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(size))
            if context is not None:
                self.send_header(REQUEST_ID_HEADER, context.request_id)
            self.end_headers()
            shutil.copyfileobj(stream, self.wfile)
        self._answered = (int(HTTPStatus.OK), size)

    def _dispatch_transfer(self, app: ExperienceHTTPService, path: str) -> bool:
        """Serve push, pull, rebind and the file routes they move files over; ``False`` for any other path."""

        if self.command == _POST and path == _FILES_MISSING:
            body = self._body()
            _reject_unknown(body, frozenset({"files"}))
            files = body.get("files")
            if not isinstance(files, list):
                raise HTTPServiceError("files must be a list of file references")
            missing = app.missing_files(FileRef.from_dict(item) for item in files)
            self._write(HTTPStatus.OK, {"missing": list(missing)})
        elif self.command == _PUT and path.startswith(_FILES_PREFIX):
            sha256 = path.removeprefix(_FILES_PREFIX)
            self._write(HTTPStatus.OK, app.put_file(sha256, self._length(MAX_FILE_BYTES), cast(BinaryIO, self.rfile)))
        elif self.command == _GET and path.startswith(_FILES_PREFIX):
            self._send_file(*app.held_file(path.removeprefix(_FILES_PREFIX)))
        elif self.command == _POST and path == "/v1/push":
            _reject_unknown(self._body(), frozenset())
            self._write(HTTPStatus.OK, app.push())
        elif self.command == _POST and path == "/v1/pull":
            body = self._body()
            _reject_unknown(body, frozenset({"schema_ref"}))
            self._write(HTTPStatus.OK, app.pull(_required_text(body.get("schema_ref"), "schema_ref")))
        elif self.command == _POST and path == "/v1/rebind":
            _reject_unknown(self._body(), frozenset())
            self._write(HTTPStatus.OK, app.rebind())
        else:
            return False
        return True

    def _probe(self, path: str) -> bool:
        """Answer the unauthenticated probes an orchestrator and a metrics scraper send; ``False`` for any other path.

        They carry nothing a KB holds: which checks pass, and counts and sizes.
        """

        app = self.server.app
        if self.command != _GET or path not in PROBE_ROUTES:
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
        if self.command == _GET and parsed.path == "/health":
            self._write(HTTPStatus.OK, app.health())
            return
        if self.command == _GET and parsed.path == "/v1/list":
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
        if self.command == _GET and parsed.path == "/v1/export":
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
        if self._dispatch_state(app, parsed.path, query) or self._dispatch_transfer(app, parsed.path):
            return
        if self.command == _POST and parsed.path == "/v1/read":
            self._write(HTTPStatus.OK, app.read(_read_query(self._body())))
            return
        prefix = "/v1/experiences/"
        if self.command == _PUT and parsed.path.startswith(prefix):
            body = self._body()
            _reject_unknown(body, _WRITE_FIELDS)
            experience = Experience.from_dict(body.get("experience"))
            if experience.id != parsed.path.removeprefix(prefix):
                raise HTTPServiceError("Experience path id differs from payload")
            declaration = body.get("declaration")
            context = current_request.get()
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
        if self.command == _GET and path == labels:
            self._write(HTTPStatus.OK, app.labels(_query_text(query, "schema_ref")))
        elif self.command == _POST and path == labels:
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
        elif self.command == _DELETE and path.startswith(f"{labels}/"):
            self._write(HTTPStatus.OK, app.delete_label(path.removeprefix(f"{labels}/")))
        elif self.command == _POST and path == "/v1/restore":
            body = self._body()
            _reject_unknown(body, frozenset({"label_id"}))
            self._write(HTTPStatus.OK, app.restore(_required_text(body.get("label_id"), "label_id")))
        elif self.command == _GET and path == exclusions:
            self._write(HTTPStatus.OK, app.exclusions(_query_text(query, "schema_ref")))
        elif self.command == _POST and path == exclusions:
            body = self._body()
            _reject_unknown(body, frozenset({"experience_id", "reason"}))
            self._write(
                HTTPStatus.OK,
                app.exclude(
                    _required_text(body.get("experience_id"), "experience_id"),
                    _required_text(body.get("reason"), "reason"),
                ),
            )
        elif self.command == _DELETE and path.startswith(f"{exclusions}/"):
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
            received = int(self.headers.get("Content-Length") or 0) if self.command in (_POST, _PUT) else 0
            metrics.observe_request(
                ServedRequest(
                    method=self.command, route=route, status=status, seconds=seconds, bytes_in=received, bytes_out=sent
                )
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
        except MissingFiles as exc:
            self._write(
                HTTPStatus.CONFLICT, {"error": _MISSING_FILES, "detail": str(exc), "missing": list(exc.missing)}
            )
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


def serve_until_stopped(server: ExperienceHTTPServer, stop: threading.Event, drain_seconds: float) -> None:
    """Serve until ``stop`` is set, then stop taking requests and let those in flight finish for ``drain_seconds``."""

    threading.Thread(target=lambda: (stop.wait(), server.shutdown()), daemon=True).start()
    server.serve_forever()
    deadline = time.monotonic() + drain_seconds
    while server.app.metrics.requests_in_flight() and time.monotonic() < deadline:
        time.sleep(0.05)


__all__ = [
    "DEFAULT_READ_LIMIT",
    "MAX_EXPORT_LIMIT",
    "MAX_READ_LIMIT",
    "ExperienceHTTPServer",
    "HTTPServiceError",
    "MissingFiles",
    "RequestHandler",
    "create_http_server",
    "serve_until_stopped",
]
