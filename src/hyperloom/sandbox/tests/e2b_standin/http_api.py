# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The stand-in's control-plane HTTP surface: the routes of the E2B API the SDK calls, over :class:`ControlPlane`.

Both the ``/v2/sandboxes`` routes of current SDKs and the ``/sandboxes`` aliases of older ones are served. Every error
is the JSON ``{"code", "message"}`` body E2B sends.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from hyperloom.sandbox.socket import Role
from hyperloom.sandbox.tests.e2b_standin.service import Box, ControlPlane, ServiceError

__all__ = ["serve_api"]


def _iso(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _sandbox(plane: ControlPlane, box: Box) -> dict[str, Any]:
    return {
        "templateID": box.template.name,
        "sandboxID": box.sandbox_id,
        "clientID": "standin",
        "envdVersion": plane.envd_version,
        "alias": box.template.name,
        "envdAccessToken": box.token,
    }


def _detail(plane: ControlPlane, box: Box) -> dict[str, Any]:
    return {
        **_sandbox(plane, box),
        "startedAt": _iso(box.started),
        "endAt": _iso(box.end),
        "cpuCount": 1,
        "memoryMB": 512,
        "diskSizeMB": 1024,
        "state": "running",
        "metadata": box.metadata,
    }


def _metadata_filter(query: dict[str, list[str]]) -> dict[str, str]:
    """The pairs of the SDK's ``metadata=k%3Dv&k2%3Dv2`` parameter, whose keys and values are percent-encoded twice."""
    pairs = (item.split("=", 1) for item in query.get("metadata", [""])[0].split("&") if "=" in item)
    return {unquote(k): unquote(v) for k, v in pairs}


@dataclass(frozen=True)
class Call:
    """What a request asks: the path's segments (after any ``v2``), its query and its JSON body."""

    parts: list[str]
    query: dict[str, list[str]]
    body: dict[str, Any]


def serve_api(plane: ControlPlane, *, page_size: int) -> ThreadingHTTPServer:
    """Build the (not yet serving) API server on a free loopback port."""

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args: object) -> None:
            pass

        def _reply(self, status: int, body: object = None, headers: dict[str, str] | None = None) -> None:
            payload = b"" if body is None else json.dumps(body).encode()
            self.send_response(status)
            if payload:
                self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            for name, value in (headers or {}).items():
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(payload)

        def _json_body(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length") or 0)
            return json.loads(self.rfile.read(length)) if length else {}

        def _dispatch(self) -> None:
            url = urlparse(self.path)
            parts = url.path.strip("/").split("/")
            if parts[0] == "v2":
                parts = parts[1:]
            body = self._json_body() if self.command == "POST" else {}
            call = Call(parts, parse_qs(url.query), body)
            try:
                role = plane.role_of(self.headers.get("X-API-KEY"))
                handler = self._ROUTES.get((self.command, len(parts), parts[2] if len(parts) == 3 else ""))
                if parts[0] != "sandboxes" or handler is None:
                    raise ServiceError(404, f"no route {self.command} {url.path}")
                handler(self, role, call)
            except ServiceError as exc:
                self._reply(exc.status, {"code": exc.status, "message": str(exc)})

        def _create(self, role: Role, call: Call) -> None:
            self._reply(201, _sandbox(plane, plane.create(role, call.body)))

        def _list(self, role: Role, call: Call) -> None:
            boxes = plane.list(role, _metadata_filter(call.query))
            start = int(call.query.get("nextToken", ["0"])[0] or 0)
            limit = min(int(call.query.get("limit", [page_size])[0]), page_size)
            more = start + limit < len(boxes)
            headers = {"x-next-token": str(start + limit)} if more else None
            self._reply(200, [_detail(plane, b) for b in boxes[start : start + limit]], headers)

        def _get(self, role: Role, call: Call) -> None:
            self._reply(200, _detail(plane, plane.get(role, call.parts[1])))

        def _kill(self, role: Role, call: Call) -> None:
            plane.kill(role, call.parts[1])
            self._reply(204)

        def _connect(self, role: Role, call: Call) -> None:
            self._reply(200, _sandbox(plane, plane.connect(role, call.parts[1], call.body.get("timeout"))))

        def _set_timeout(self, role: Role, call: Call) -> None:
            plane.set_timeout(role, call.parts[1], int(call.body["timeout"]))
            self._reply(204)

        _ROUTES = {
            ("POST", 1, ""): _create,
            ("GET", 1, ""): _list,
            ("GET", 2, ""): _get,
            ("DELETE", 2, ""): _kill,
            ("POST", 3, "connect"): _connect,
            ("POST", 3, "timeout"): _set_timeout,
        }

        do_GET = do_POST = do_DELETE = _dispatch

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    return server
