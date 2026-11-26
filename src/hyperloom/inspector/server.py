# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""HTTP + Server-Sent Events front of the watcher. Read-only; binds to loopback by default.

Endpoints (all GET):

* ``/api/snapshot``: the full view at the current clock, streams included.
* ``/api/stream``: SSE. ``hello`` first, then ``items`` (stream deltas) and ``view`` (views without
  streams, when something changed).
* ``/api/artifacts``: the artifact index.
* ``/api/artifact?path=&mode=&offset=&limit=``: one artifact's typed payload.
* ``/api/diff?a=&b=``: side-by-side diff of two text artifacts.
* ``/api/conversation?call_id=``: prompt/response of one LLM call.

The Host header must name a loopback address or the bound host, so a page on
another origin can't reach the server through DNS rebinding.
"""

from __future__ import annotations

import ipaddress
import json
import logging
import time
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

from hyperloom.inspector import artifacts
from hyperloom.inspector.watcher import Watcher, WatchOptions

log = logging.getLogger(__name__)

_LOOPBACK = {"localhost", "127.0.0.1", "::1", "[::1]"}
_SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "X-Frame-Options": "DENY",
    "Content-Security-Policy": "default-src 'self'; script-src 'self'; "
    "style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'",
}
_HEARTBEAT_SEC = 15.0
_ENCODING = "utf-8"


class InspectorServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], watcher: Watcher) -> None:
        self.watcher = watcher
        self.allowed_hosts = _LOOPBACK | {address[0]}
        self.closing = False
        super().__init__(address, Handler)


class Handler(BaseHTTPRequestHandler):
    server: InspectorServer
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        log.debug("inspector: %s", fmt % args)

    # ----- plumbing ------------------------------------------------------------------------------------------

    def _host_ok(self) -> bool:
        host = (self.headers.get("Host") or "").strip()
        name = host.rsplit(":", 1)[0] if not host.startswith("[") else host.split("]")[0] + "]"
        return name in self.server.allowed_hosts

    def _send(self, status: int, body: bytes, ctype: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in {"Cache-Control": "no-store", **_SECURITY_HEADERS}.items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, data: Any, status: int = HTTPStatus.OK) -> None:
        self._send(status, json.dumps(data, default=str).encode(_ENCODING), "application/json; charset=utf-8")

    def _error(self, status: int, message: str) -> None:
        self._json({"error": message}, status)

    def _query(self) -> dict[str, str]:
        return {k: v[-1] for k, v in parse_qs(urlsplit(self.path).query).items()}

    # ----- routing -------------------------------------------------------------------------------------------

    def do_GET(self) -> None:
        if not self._host_ok():
            return self._error(HTTPStatus.FORBIDDEN, "host not allowed")
        route = urlsplit(self.path).path
        q = self._query()
        w = self.server.watcher
        try:
            if route == "/api/snapshot":
                return self._json(w.snapshot())
            if route == "/api/stream":
                return self._stream()
            if route == "/api/artifacts":
                return self._json({"artifacts": w.artifacts_at()})
            if route == "/api/artifact":
                return self._json(self._artifact(q))
            if route == "/api/diff":
                return self._json(artifacts.diff(w.session_dir, q.get("a", ""), q.get("b", "")))
            if route == "/api/conversation":
                rec = w.conversation(q.get("call_id", ""), q.get("task_id", ""), q.get("turn", ""))
                return (
                    self._json(rec)
                    if rec is not None
                    else self._error(HTTPStatus.NOT_FOUND, "no conversation recorded")
                )
            return self._error(HTTPStatus.NOT_FOUND, "not found")
        except PermissionError:
            return self._error(HTTPStatus.FORBIDDEN, "path not allowed")
        except FileNotFoundError:
            return self._error(HTTPStatus.NOT_FOUND, "no such file")
        except ValueError as exc:
            return self._error(HTTPStatus.BAD_REQUEST, str(exc))

    def _artifact(self, q: dict[str, str]) -> dict:
        try:
            offset = int(q["offset"]) if q.get("offset") else None
            limit = int(q.get("limit") or 500)
        except ValueError as exc:
            raise ValueError("offset and limit must be integers") from exc
        return artifacts.view(
            self.server.watcher.session_dir, q.get("path", ""), q.get("mode") or None, offset, max(1, min(limit, 5000))
        )

    def _stream(self) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        for k, v in _SECURITY_HEADERS.items():
            self.send_header(k, v)
        self.end_headers()
        self.close_connection = True
        w = self.server.watcher
        cursors: dict[str, int] = {}
        snap = w.snapshot(include_streams=False)
        cursors.update(snap["released"])
        version, last_beat = snap["version"], time.monotonic()
        try:
            self._event("hello", {"clock": snap["clock"], "released": cursors})
            while not self.server.closing:
                time.sleep(w.poll_sec)
                delta = w.delta(cursors)
                sent = False
                if delta["items"]:
                    self._event("items", delta)
                    sent = True
                if delta["version"] != version:
                    version = delta["version"]
                    self._event("view", w.snapshot(include_streams=False))
                    sent = True
                if sent:
                    last_beat = time.monotonic()
                elif time.monotonic() - last_beat > _HEARTBEAT_SEC:
                    self.wfile.write(b": keep-alive\n\n")
                    self.wfile.flush()
                    last_beat = time.monotonic()
        except (BrokenPipeError, ConnectionResetError):
            return

    def _event(self, name: str, data: Any) -> None:
        self.wfile.write(f"event: {name}\ndata: {json.dumps(data, default=str)}\n\n".encode(_ENCODING))
        self.wfile.flush()


def _is_unspecified(host: str) -> bool:
    try:
        return ipaddress.ip_address(host.strip("[]")).is_unspecified
    except ValueError:
        return False


@dataclass(frozen=True)
class ServeOptions:
    """What :func:`serve` listens on (*host*, *port*), and how the watcher reads the session (*watch*)."""

    host: str = "127.0.0.1"
    port: int = 8765
    watch: WatchOptions = field(default_factory=WatchOptions)


def serve(session_dir: Path, options: ServeOptions | None = None) -> None:
    """Run the Inspector until interrupted."""
    options = options or ServeOptions()
    watcher = Watcher(Path(session_dir), options.watch)
    watcher.start()
    httpd = InspectorServer((options.host, options.port), watcher)
    shown = "localhost" if _is_unspecified(options.host) else options.host
    print(f"Hyperloom Inspector (live): http://{shown}:{options.port}/api/snapshot for {session_dir}", flush=True)
    try:
        httpd.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        httpd.closing = True
        watcher.stop()
        httpd.server_close()
