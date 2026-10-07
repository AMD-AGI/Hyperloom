# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What an Experience KB service reports about itself: Prometheus metrics, JSON log lines, and the request each
log line and each write belongs to.

A client names a request with ``X-Request-ID`` and, when it is another KB service syncing, itself with
``X-Hyperloom-KB-Client``; the service answers with the request id, records the client with every write, and puts both
on every line it logs while serving the request.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
import uuid
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote, unquote

REQUEST_ID_HEADER = "X-Request-ID"
CLIENT_HEADER = "X-Hyperloom-KB-Client"
CLIENT_NAME_HEADER = "X-Hyperloom-KB-Client-Name"
_TOKEN = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
_NAME_LIMIT = 128
LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0)

_FIXED_ROUTES = frozenset(
    {
        "/health",
        "/livez",
        "/readyz",
        "/metrics",
        "/v1/read",
        "/v1/list",
        "/v1/export",
        "/v1/push",
        "/v1/pull",
        "/v1/rebind",
        "/v1/labels",
        "/v1/restore",
        "/v1/exclusions",
        "/v1/files/missing",
    }
)
_ROUTE_PATTERNS = (
    (re.compile(r"^/v1/experiences/[^/]+$"), "/v1/experiences/{experience_id}"),
    (re.compile(r"^/v1/files/[^/]+$"), "/v1/files/{sha256}"),
    (re.compile(r"^/v1/labels/[^/]+$"), "/v1/labels/{label_id}"),
    (re.compile(r"^/v1/exclusions/[^/]+$"), "/v1/exclusions/{experience_id}"),
)
# Probed every few seconds by an orchestrator or a scraper; counted, never logged.
PROBE_ROUTES = frozenset({"/livez", "/readyz", "/metrics"})

_HELP = {
    "hyperloom_kb_build_info": ("gauge", "The KB this service serves and the code it runs."),
    "hyperloom_kb_start_time_seconds": ("gauge", "When this service process started, in Unix seconds."),
    "hyperloom_kb_http_requests_total": ("counter", "Requests answered, by method, route, and status."),
    "hyperloom_kb_http_request_duration_seconds": ("histogram", "Time to answer a request, by method and route."),
    "hyperloom_kb_http_requests_in_flight": ("gauge", "Requests being answered now."),
    "hyperloom_kb_http_request_bytes_total": ("counter", "Request body bytes received, by route."),
    "hyperloom_kb_http_response_bytes_total": ("counter", "Response body bytes sent, by route."),
    "hyperloom_kb_http_unauthorized_total": ("counter", "Requests refused for a missing or wrong token."),
    "hyperloom_kb_writes_total": (
        "counter",
        "Experience writes by schema and result: created, unchanged, conflict, missing_files.",
    ),
    "hyperloom_kb_file_writes_total": ("counter", "File writes by result: created, unchanged."),
    "hyperloom_kb_sync_batches_total": ("counter", "Push and pull batches by direction and status."),
    "hyperloom_kb_experiences": ("gauge", "Stored Experiences by schema and state: visible, excluded, outside."),
    "hyperloom_kb_record_bytes": ("gauge", "Bytes of the KB's record files."),
    "hyperloom_kb_database_bytes": ("gauge", "Size of the KB's database."),
    "hyperloom_kb_disk_free_bytes": ("gauge", "Free bytes on the file system of the service's home."),
    "hyperloom_kb_last_write_timestamp_seconds": ("gauge", "When the KB last stored a new Experience."),
    "hyperloom_kb_records_missing": ("gauge", "Records the database holds whose file was missing at start."),
    "hyperloom_kb_files": ("gauge", "Files the KB holds."),
    "hyperloom_kb_file_bytes": ("gauge", "Bytes of the files the KB holds."),
    "hyperloom_kb_files_missing": ("gauge", "Files the database holds that were missing from the home at start."),
    "hyperloom_kb_ready": ("gauge", "Whether each readiness check passes, 1 or 0."),
    "hyperloom_kb_database_pool": ("gauge", "Database connection pool statistics."),
}


@dataclass(frozen=True)
class RequestContext:
    """The request a service is answering: its id, and the KB that sent it when it named itself."""

    request_id: str
    client_kb_id: str = ""
    client_name: str = ""

    @classmethod
    def from_headers(cls, headers: Mapping[str, str]) -> RequestContext:
        supplied = headers.get(REQUEST_ID_HEADER) or ""
        client = headers.get(CLIENT_HEADER) or ""
        return cls(
            supplied if _TOKEN.fullmatch(supplied) else uuid.uuid4().hex,
            client if _TOKEN.fullmatch(client) else "",
            unquote(headers.get(CLIENT_NAME_HEADER) or "")[:_NAME_LIMIT],
        )


current_request: ContextVar[RequestContext | None] = ContextVar("hyperloom_kb_request", default=None)


def client_headers(kb_id: str, name: str) -> dict[str, str]:
    """The headers by which one KB service names itself to another it syncs with."""

    return {CLIENT_HEADER: kb_id, CLIENT_NAME_HEADER: quote(name, safe="")}


def route_of(path: str) -> str:
    if path in _FIXED_ROUTES:
        return path
    for pattern, route in _ROUTE_PATTERNS:
        if pattern.match(path):
            return route
    return "other"


def event(logger: logging.Logger, name: str, /, *, level: int = logging.INFO, **fields: Any) -> None:
    """Log one structured event, with the request being answered when there is one."""

    context = current_request.get()
    if context is not None:
        fields = {
            "request_id": context.request_id,
            "client_kb_id": context.client_kb_id,
            "client_name": context.client_name,
            **fields,
        }
    logger.log(level, name, extra={"fields": {"event": name, **fields}})


class JsonLogFormatter(logging.Formatter):
    """One JSON object per line: time, level, logger, message, and the fields an ``event`` carries."""

    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, timezone.utc).isoformat().replace("+00:00", "Z"),
            "level": record.levelname.lower(),
            "logger": record.name,
            "message": record.getMessage(),
        }
        fields = getattr(record, "fields", None)
        if isinstance(fields, dict):
            entry.update(fields)
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry, ensure_ascii=False, default=str)


Labels = tuple[tuple[str, str], ...]


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _series(name: str, labels: Labels) -> str:
    if not labels:
        return name
    return name + "{" + ",".join(f'{key}="{_escape(value)}"' for key, value in labels) + "}"


def _number(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else repr(float(value))


class Metrics:
    """The counters and request histogram of one service process, rendered with gauges sampled at scrape time."""

    def __init__(self) -> None:
        self.started_at = time.time()
        self._lock = threading.Lock()
        self._counters: dict[tuple[str, Labels], float] = {}
        self._buckets: dict[Labels, list[int]] = {}
        self._sums: dict[Labels, float] = {}
        self._in_flight = 0

    def count(self, name: str, amount: float = 1.0, **labels: str) -> None:
        key = (name, tuple(sorted(labels.items())))
        with self._lock:
            self._counters[key] = self._counters.get(key, 0.0) + amount

    def observe_request(
        self, *, method: str, route: str, status: int, seconds: float, bytes_in: int, bytes_out: int
    ) -> None:
        self.count("hyperloom_kb_http_requests_total", method=method, route=route, status=str(status))
        self.count("hyperloom_kb_http_request_bytes_total", bytes_in, route=route)
        self.count("hyperloom_kb_http_response_bytes_total", bytes_out, route=route)
        labels: Labels = (("method", method), ("route", route))
        with self._lock:
            buckets = self._buckets.setdefault(labels, [0] * (len(LATENCY_BUCKETS) + 1))
            for index, bound in enumerate(LATENCY_BUCKETS):
                if seconds <= bound:
                    buckets[index] += 1
            buckets[-1] += 1
            self._sums[labels] = self._sums.get(labels, 0.0) + seconds

    @contextmanager
    def in_flight(self) -> Iterator[None]:
        with self._lock:
            self._in_flight += 1
        try:
            yield
        finally:
            with self._lock:
                self._in_flight -= 1

    def requests_in_flight(self) -> int:
        with self._lock:
            return self._in_flight

    def render(self, gauges: Iterable[tuple[str, Mapping[str, str], float]]) -> str:
        """The Prometheus text exposition of every metric, with ``gauges`` as this scrape samples them."""

        series: dict[str, list[str]] = {}
        with self._lock:
            for (name, labels), value in sorted(self._counters.items()):
                series.setdefault(name, []).append(f"{_series(name, labels)} {_number(value)}")
            histogram = "hyperloom_kb_http_request_duration_seconds"
            for labels, buckets in sorted(self._buckets.items()):
                lines = series.setdefault(histogram, [])
                for bound, count in zip((*LATENCY_BUCKETS, float("inf")), buckets):
                    le = "+Inf" if bound == float("inf") else repr(bound)
                    lines.append(f"{_series(histogram + '_bucket', (*labels, ('le', le)))} {count}")
                lines.append(f"{_series(histogram + '_sum', labels)} {_number(self._sums[labels])}")
                lines.append(f"{_series(histogram + '_count', labels)} {buckets[-1]}")
            in_flight = self._in_flight
        series.setdefault("hyperloom_kb_http_requests_in_flight", []).append(
            f"hyperloom_kb_http_requests_in_flight {in_flight}"
        )
        series.setdefault("hyperloom_kb_start_time_seconds", []).append(
            f"hyperloom_kb_start_time_seconds {_number(self.started_at)}"
        )
        for name, labels, value in gauges:
            series.setdefault(name, []).append(f"{_series(name, tuple(labels.items()))} {_number(value)}")
        out: list[str] = []
        for name in sorted(series):
            kind, text = _HELP.get(name, ("untyped", ""))
            out.extend((f"# HELP {name} {text}", f"# TYPE {name} {kind}", *series[name]))
        return "\n".join(out) + "\n"


__all__ = [
    "CLIENT_HEADER",
    "CLIENT_NAME_HEADER",
    "LATENCY_BUCKETS",
    "PROBE_ROUTES",
    "REQUEST_ID_HEADER",
    "JsonLogFormatter",
    "Metrics",
    "RequestContext",
    "client_headers",
    "current_request",
    "event",
    "route_of",
]
