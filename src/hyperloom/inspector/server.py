# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""HTTP + Server-Sent Events front of the watcher. Read-only; binds to loopback by default.

Endpoints (all GET unless noted):

* ``/``: the UI (``static/modern/``, built from ``frontend/inspector``), or a 503 that says how to get it when
  this install has none; ``/assets/<name>``: the build's hashed scripts, styles, icons and fonts.
* ``/api/snapshot``: the full view at the current clock, streams included.
* ``/api/stream``: SSE. ``hello`` first, then ``items`` (stream deltas), ``view`` (views without
  streams, when something changed) and ``reset`` (a replay scrub moved back; refetch the snapshot).
* ``/api/artifacts``: the artifact index (replay: released by mtime).
* ``/api/artifact?path=&mode=&offset=&limit=``: one artifact's typed payload.
* ``/api/diff?a=&b=``: side-by-side diff of two text artifacts.
* ``/api/conversation?call_id=``: prompt/response of one LLM call.
* ``POST /api/replay?t=&speed=&paused=``: move the replay clock.

The Host header must name a loopback address or the bound host, so a page on
another origin can't reach the server through DNS rebinding.
"""

from __future__ import annotations

import ipaddress
import json
import logging
import re
import sys
import time
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs, urlsplit

from hyperloom.inspector import artifacts
from hyperloom.inspector.watcher import Watcher, WatchOptions

if TYPE_CHECKING:
    from importlib.resources.abc import Traversable

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


_ASSET_TYPES = {
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".svg": "image/svg+xml",
    ".woff2": "font/woff2",
}
_ASSET_NAME = re.compile(r"^/assets/([A-Za-z0-9][A-Za-z0-9_-]*)(\.js|\.css|\.svg|\.woff2)$")
# Vite puts a content hash in every asset name, so a cached copy can never go stale.
_IMMUTABLE = {"Cache-Control": "public, max-age=31536000, immutable"}
_NO_BUILD = (
    b"The Inspector UI is not in this install.\n\n"
    b"In a source tree, build it with `npm ci && npm run build` in frontend/inspector.\n"
    b"To run the Inspector in a pod, build it on your machine and copy the tree in, as docs/inspector/README.md\n"
    b'("In a pod") describes.\n'
)


def _modern_root() -> Traversable:
    """Where ``npm run build`` in frontend/inspector writes the UI; absent in a tree where it never ran."""
    return resources.files("hyperloom.inspector").joinpath("static", "modern")


def _modern_page() -> bytes | None:
    res = _modern_root().joinpath("index.html")
    return res.read_bytes() if res.is_file() else None


def _modern_asset(route: str) -> tuple[bytes, str] | None:
    """A file of the build under ``assets/``, by bare name; nested paths and other suffixes are not assets."""
    m = _ASSET_NAME.match(route)
    if not m:
        return None
    res = _modern_root().joinpath("assets", m[1] + m[2])
    if not res.is_file():
        return None
    return res.read_bytes(), _ASSET_TYPES[m[2]]


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

    def _send(self, status: int, body: bytes, ctype: str, extra: dict | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in {"Cache-Control": "no-store", **_SECURITY_HEADERS, **(extra or {})}.items():
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
            if route in ("/", "/index.html"):
                if (page := _modern_page()) is None:
                    return self._send(HTTPStatus.SERVICE_UNAVAILABLE, _NO_BUILD, "text/plain; charset=utf-8")
                return self._send(HTTPStatus.OK, page, "text/html; charset=utf-8")
            if (asset := _modern_asset(route)) is not None:
                return self._send(HTTPStatus.OK, *asset, _IMMUTABLE)
            if route == "/favicon.ico":
                return self._send(HTTPStatus.NO_CONTENT, b"", "image/x-icon")
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

    def do_POST(self) -> None:
        if not self._host_ok():
            return self._error(HTTPStatus.FORBIDDEN, "host not allowed")
        route = urlsplit(self.path).path
        if route != "/api/replay":
            return self._error(HTTPStatus.NOT_FOUND, "not found")
        clock = self.server.watcher.clock
        if not clock.replay:
            return self._error(HTTPStatus.CONFLICT, "not in replay mode")
        q = self._query()
        try:
            t = float(q["t"]) if q.get("t") else None
            speed = float(q["speed"]) if q.get("speed") else None
        except ValueError:
            return self._error(HTTPStatus.BAD_REQUEST, "t and speed must be numbers")
        paused = {"1": True, "true": True, "0": False, "false": False}.get(q.get("paused", "").lower())
        with self.server.watcher.lock:
            clock.seek(t, speed=speed, paused=paused)
        return self._json(clock.info())

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
                if delta.get("reset"):
                    self._event("reset", delta)
                    return
                sent = False
                if delta["items"]:
                    self._event("items", delta)
                    sent = True
                if delta["version"] != version or w.clock.replay:
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
    mode = "replay" if options.watch.replay else "live"
    print(f"Hyperloom Inspector ({mode}): http://{shown}:{options.port}/ for {session_dir}", flush=True)
    if _modern_page() is None:
        print("Inspector UI: not in this install; / says how to get it", file=sys.stderr, flush=True)
    try:
        httpd.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        httpd.closing = True
        watcher.stop()
        httpd.server_close()
