# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The stand-in's data-plane router: one URL in front of every sandbox's envd, routed by the ``E2b-Sandbox-Id`` header.

This is what E2B's client proxy (and the Claw Router) does: the SDK sends every envd request to ``E2B_SANDBOX_URL`` and
names the target sandbox in a header. Bodies and responses are relayed as streams, so large files and long-running
commands pass through without buffering. A request for a sandbox that does not exist gets the 502 E2B's proxy gives.
"""

from __future__ import annotations

import http.client
import json
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from hyperloom.sandbox.tests.e2b_standin.service import ControlPlane

__all__ = ["serve_proxy"]

_HOP_BY_HOP = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
        "host",
    }
)
_CHUNK = 1 << 20


def serve_proxy(plane: ControlPlane) -> ThreadingHTTPServer:
    """Build the (not yet serving) proxy on a free loopback port."""

    class Proxy(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args: object) -> None:
            pass

        def _route(self) -> None:
            sandbox_id = self.headers.get("E2b-Sandbox-Id")
            box = plane.lookup(sandbox_id)
            if box is None:
                body = json.dumps(
                    {"sandboxId": sandbox_id, "message": f"The sandbox {sandbox_id} was not found", "code": 502}
                ).encode()
                self.send_response(502)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            conn = http.client.HTTPConnection("127.0.0.1", box.port, timeout=3600)
            try:
                self._relay(conn)
            except (http.client.HTTPException, OSError, ValueError):
                self.close_connection = True
            finally:
                conn.close()

        def _body(self) -> Iterator[bytes] | None:
            if "chunked" in (self.headers.get("Transfer-Encoding") or "").lower():
                return self._chunked_body()
            left = int(self.headers.get("Content-Length") or 0)
            return self._sized_body(left) if left else None

        def _sized_body(self, left: int) -> Iterator[bytes]:
            while left > 0:
                data = self.rfile.read(min(left, _CHUNK))
                left -= len(data)
                yield data

        def _chunked_body(self) -> Iterator[bytes]:
            while True:
                size = int(self.rfile.readline().split(b";")[0].strip() or b"0", 16)
                if size == 0:
                    while self.rfile.readline() not in (b"\r\n", b"\n", b""):
                        pass
                    return
                yield self.rfile.read(size)
                self.rfile.readline()

        def _relay(self, conn: http.client.HTTPConnection) -> None:
            headers = {k: v for k, v in self.headers.items() if k.lower() not in _HOP_BY_HOP}
            body = self._body()
            chunked = "chunked" in (self.headers.get("Transfer-Encoding") or "").lower()
            if chunked:
                headers["Transfer-Encoding"] = "chunked"
            conn.request(self.command, self.path, body=body, headers=headers, encode_chunked=chunked)
            response = conn.getresponse()
            self.send_response(response.status)
            length = response.getheader("Content-Length")
            for name, value in response.getheaders():
                if name.lower() not in _HOP_BY_HOP and name.lower() != "content-length":
                    self.send_header(name, value)
            if length is not None or response.status in (204, 304) or self.command == "HEAD":
                if length is not None:
                    self.send_header("Content-Length", length)
                self.end_headers()
                while self.command != "HEAD" and (data := response.read(_CHUNK)):
                    self.wfile.write(data)
                return
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            while data := response.read1(65536):
                self.wfile.write(b"%x\r\n%b\r\n" % (len(data), data))
                self.wfile.flush()
            self.wfile.write(b"0\r\n\r\n")

        do_GET = do_POST = do_PUT = do_DELETE = do_HEAD = do_PATCH = do_OPTIONS = _route

    server = ThreadingHTTPServer(("127.0.0.1", 0), Proxy)
    server.daemon_threads = True
    return server
