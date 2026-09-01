# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Fake PR Monitor HTTP server fixture (healthz, prs list/detail/files/patches)."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Iterator

import pytest


_FAKE_PR_LIST = {
    "items": [
        {
            "number": 1,
            "title": "Fake winner PR",
            "html_url": "https://example.invalid/pulls/1",
        },
    ]
}

_FAKE_PR_DETAIL = {
    "summary": {
        "number": 1,
        "title": "Fake winner PR",
        "head_sha": "deadbeefcafe000000000000000000000000000a",
        "author_login": "fakebot",
        "labels": [{"name": "perf"}],
        "pr_updated_at": "2026-05-18T00:00:00Z",
        "html_url": "https://example.invalid/pulls/1",
    },
    "files": [
        {"file_path": "python/x/foo.py", "additions": 3, "deletions": 1},
    ],
}

_FAKE_PR_FILES = [
    {"file_path": "python/x/foo.py", "additions": 3, "deletions": 1},
]

_FAKE_PR_PATCHES = [
    {
        "file": {"file_path": "python/x/foo.py", "status": "modified"},
        "patch": "@@ -1,3 +1,5 @@\n-a\n+b\n c\n",
        "patch_truncated": False,
    }
]


class _FakeHandler(BaseHTTPRequestHandler):
    """Minimal handler routing the five endpoints framework-agent uses."""

    def log_message(self, format: str, *args) -> None:  # noqa: A003 - stdlib name
        """Silence the default access log so pytest output stays clean."""
        return

    def do_GET(self) -> None:  # noqa: N802 - stdlib signature
        path = self.path.split("?", 1)[0]
        if path == "/v1/healthz":
            self._send_json({"status": "ok"})
            return
        if path.endswith("/prs"):
            self._send_json(_FAKE_PR_LIST)
            return
        if path.endswith("/prs/1"):
            self._send_json(_FAKE_PR_DETAIL)
            return
        if path.endswith("/prs/1/files"):
            self._send_json(_FAKE_PR_FILES)
            return
        if path.endswith("/prs/1/patches"):
            self._send_json(_FAKE_PR_PATCHES)
            return
        self.send_response(404)
        self.end_headers()

    def _send_json(self, payload) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def fake_pr_monitor() -> Iterator[str]:
    """Start a fake PR Monitor server and yield its base URL."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeHandler)
    host, port = server.server_address[:2]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()
