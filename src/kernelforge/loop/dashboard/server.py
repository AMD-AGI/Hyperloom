# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Serve one campaign's dashboard, rendered from its files on every visit, and take operator messages from its form.

The page's form posts to ``say`` beside it; a message lands in the campaign's inbox exactly as ``forge-say`` would
put it there, and the reader is sent back to the page.
"""

from __future__ import annotations

from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from kernelforge.loop.dashboard.render import render_dashboard
from kernelforge.loop.dashboard.snapshot import DASHBOARD_HTML, collect_snapshot
from kernelforge.loop.operator_inbox import CLEAR, post_message

_PAGES = {"/", f"/{DASHBOARD_HTML}"}
# A message is a few sentences; anything far larger is not one.
_MAX_FORM_BYTES = 64 * 1024


def read_form(handler: BaseHTTPRequestHandler) -> str | None:
    """The inbox text a posted form asks for (``CLEAR`` to withdraw every message), or None when it asks nothing."""
    length = int(handler.headers.get("Content-Length") or 0)
    if length > _MAX_FORM_BYTES:
        return None
    form = parse_qs(handler.rfile.read(length).decode("utf-8", errors="replace"))
    if form.get("clear"):
        return CLEAR
    message = " ".join(form.get("message") or []).strip()
    return message or None


class _Handler(BaseHTTPRequestHandler):
    server: _DashboardServer

    def log_message(self, format: str, *args) -> None:  # noqa: A002 -- the stdlib's own parameter name
        return

    def _send(self, status: HTTPStatus, body: str = "", location: str = "") -> None:
        payload = body.encode("utf-8")
        self.send_response(status)
        if location:
            self.send_header("Location", location)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:
        if urlsplit(self.path).path not in _PAGES:
            self._send(HTTPStatus.NOT_FOUND, "not found")
            return
        self._send(HTTPStatus.OK, render_dashboard(collect_snapshot(self.server.campaign_root)))

    def do_POST(self) -> None:
        if urlsplit(self.path).path != "/say":
            self._send(HTTPStatus.NOT_FOUND, "not found")
            return
        text = read_form(self)
        if text is None:
            self._send(HTTPStatus.BAD_REQUEST, "an empty message")
            return
        post_message(self.server.workspace, text)
        self._send(HTTPStatus.SEE_OTHER, location=DASHBOARD_HTML)


class _DashboardServer(ThreadingHTTPServer):
    def __init__(self, workspace: Path, address: tuple[str, int]) -> None:
        self.workspace = workspace
        self.campaign_root = workspace / "forge_experiments"
        super().__init__(address, _Handler)


def make_server(workspace: str | Path, host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    """A server for the campaign in ``workspace``; port 0 picks a free one."""
    return _DashboardServer(Path(workspace).resolve(), (host, port))
