# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The one MCP service of a campaign: Streamable HTTP subset, token scopes, tool registration.

The service core knows nothing about the tools it serves. Whoever runs the campaign registers tools with
:meth:`McpServer.register` and issues one token per specialist with :meth:`McpServer.issue_token`; a tool lists the
token kinds that may call it, and its handler receives the :class:`CallContext` of the token that called.

Transport (design 4.12): ``POST /mcp`` takes one JSON-RPC request or a batch and answers ``application/json``;
``GET`` answers 405 because there is no server push stream; ``initialize`` answers an ``Mcp-Session-Id`` header the
client may send back and the server never requires. JSON-RPC dispatch is the transport-independent
``dispatch_envelope`` the stdio servers use.

A handler returns a string (sent as is) or a JSON value, raises :class:`ToolError` to report a failed call and
:class:`ToolForbidden` to refuse the caller.

Authentication: ``X-Hyperloom-Token`` or ``Authorization: Bearer``. The host token is the campaign's
``HYPERLOOM_MCP_TOKEN``; every other token was issued by this server to one specialist and dies with
:meth:`McpServer.revoke`. Issued tokens are kept as digests only, compared in constant time, and a token is never
written to a log or an error message.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import hmac
import inspect
import json
import logging
import os
import secrets
import threading
import uuid
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from hyperloom.hosted.schema_check import first_problem
from kernelforge.tools.mcp_stdio import InvalidParamsError, dispatch_envelope

logger = logging.getLogger(__name__)

JSONRPC_VERSION = "2.0"
MCP_PATH = "/mcp"
HOST_KIND = "host"
SPECIALIST_KIND = "specialist"
TOKEN_KINDS = frozenset({HOST_KIND, SPECIALIST_KIND})
MAX_BODY_BYTES = 1 << 20
MIN_TOKEN_LENGTH = 16
SESSION_HEADER = "Mcp-Session-Id"
ENV_HOST = "HYPERLOOM_MCP_HOST"
ENV_PORT = "HYPERLOOM_MCP_PORT"
ENV_TOKEN = "HYPERLOOM_MCP_TOKEN"

_ERR_PARSE = -32700
_ERR_INVALID_REQUEST = -32600
_ERR_METHOD = -32601
_ERR_PARAMS = -32602
_ERR_INTERNAL = -32603
_ERR_FORBIDDEN = -32001


class ToolForbidden(PermissionError):
    """A handler refuses the call: the token may not do this, whatever the arguments."""


class ToolError(Exception):
    """A handler reports a failed call: the client sees it as an error result, the transport is fine."""


@dataclass(frozen=True)
class McpPrincipal:
    """Who a token speaks for; ``subject`` and ``branch`` are empty for the host."""

    kind: str
    subject: str = ""
    branch: str = ""
    campaign: str = ""
    scopes: frozenset[str] = frozenset()


@dataclass(frozen=True)
class CallContext:
    """What a handler learns about the call: the caller, never the token itself."""

    principal: McpPrincipal
    tool: str


ToolHandler = Callable[[dict[str, Any], CallContext], Any]


@dataclass(frozen=True)
class RegisteredTool:
    name: str
    description: str
    schema: Mapping[str, Any]
    handler: ToolHandler
    scopes: frozenset[str]


@dataclass(frozen=True)
class McpServerConfig:
    host: str = "127.0.0.1"
    port: int = 0
    host_token: str = ""

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> McpServerConfig:
        source = os.environ if env is None else env
        return cls(
            host=source.get(ENV_HOST, "127.0.0.1"),
            port=int(source.get(ENV_PORT, "0")),
            host_token=source.get(ENV_TOKEN, ""),
        )


def _digest(token: str) -> bytes:
    return hashlib.sha256(token.encode()).digest()


class _TokenRegistry:
    """Issued tokens by digest; thread-safe, and a lookup never short-circuits on the first match."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._by_digest: dict[bytes, McpPrincipal] = {}

    def issue(self, principal: McpPrincipal) -> str:
        token = secrets.token_urlsafe(32)
        with self._lock:
            self._by_digest[_digest(token)] = principal
        return token

    def resolve(self, token: str) -> McpPrincipal | None:
        presented = _digest(token)
        with self._lock:
            entries = list(self._by_digest.items())
        found: McpPrincipal | None = None
        for stored, principal in entries:
            if hmac.compare_digest(stored, presented):
                found = principal
        return found

    def revoke_where(self, predicate: Callable[[McpPrincipal], bool]) -> int:
        with self._lock:
            doomed = [key for key, principal in self._by_digest.items() if predicate(principal)]
            for key in doomed:
                del self._by_digest[key]
        return len(doomed)


class McpServer:
    """One MCP service for one campaign."""

    def __init__(
        self,
        config: McpServerConfig,
        *,
        campaign: str,
        server_name: str = "hyperloom",
        heartbeat_seconds: float = 60.0,
    ) -> None:
        if len(config.host_token) < MIN_TOKEN_LENGTH:
            raise ValueError(f"the host token must be at least {MIN_TOKEN_LENGTH} characters")
        if not campaign:
            raise ValueError("a campaign name is required")
        self._config = config
        self._campaign = campaign
        self._server_name = server_name
        self._host_digest = _digest(config.host_token)
        self._tokens = _TokenRegistry()
        self._beats: dict[str, Callable[[], None]] = {}
        self._beats_lock = threading.Lock()
        self._heartbeat_seconds = heartbeat_seconds
        self._tools: dict[str, RegisteredTool] = {}
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    # -- registration and tokens -------------------------------------------------------------------------------

    def register(
        self,
        name: str,
        *,
        schema: Mapping[str, Any],
        handler: ToolHandler,
        scopes: Iterable[str],
        description: str = "",
    ) -> None:
        """Offer ``name`` to the token kinds in ``scopes``; the handler may be a function or a coroutine function."""
        granted = frozenset(scopes)
        if not name or name in self._tools:
            raise ValueError(f"tool name empty or already registered: {name!r}")
        if not granted or not granted <= TOKEN_KINDS:
            raise ValueError(f"scopes must be a non-empty subset of {sorted(TOKEN_KINDS)}")
        self._tools[name] = RegisteredTool(name, description, dict(schema), handler, granted)

    def issue_token(self, specialist: str, branch: str, scopes: Iterable[str] = (SPECIALIST_KIND,)) -> str:
        """A token for one specialist on one branch, carrying ``scopes``; host scope is never issued."""
        granted = frozenset(scopes)
        if not specialist or not branch:
            raise ValueError("a specialist token needs a specialist id and a branch")
        if not granted or not granted <= TOKEN_KINDS - {HOST_KIND}:
            raise ValueError(f"issued scopes must be a non-empty subset of {sorted(TOKEN_KINDS - {HOST_KIND})}")
        principal = McpPrincipal(
            SPECIALIST_KIND, subject=specialist, branch=branch, campaign=self._campaign, scopes=granted
        )
        return self._tokens.issue(principal)

    def attach_heartbeat(self, specialist: str, beat: Callable[[], None]) -> None:
        """Call ``beat()`` every heartbeat interval while a request of one of ``specialist``'s tokens is served.

        One beat per specialist; attaching again replaces it, and :meth:`revoke_specialist` removes it.
        """
        with self._beats_lock:
            self._beats[specialist] = beat

    def revoke_specialist(self, specialist: str) -> int:
        """Revoke every token of ``specialist`` and drop its heartbeat; returns how many tokens were live."""
        with self._beats_lock:
            self._beats.pop(specialist, None)
        return self._tokens.revoke_where(lambda principal: principal.subject == specialist)

    # -- authentication ----------------------------------------------------------------------------------------

    def authenticate(self, token: str) -> McpPrincipal | None:
        if not token:
            return None
        if hmac.compare_digest(self._host_digest, _digest(token)):
            return McpPrincipal(HOST_KIND, campaign=self._campaign, scopes=frozenset({HOST_KIND}))
        principal = self._tokens.resolve(token)
        if principal is None or principal.campaign != self._campaign:
            return None
        return principal

    def scrub(self, text: str, presented: str = "") -> str:
        """``text`` with the host token and the token of this request blanked out."""
        for secret in (self._config.host_token, presented):
            if secret:
                text = text.replace(secret, "[redacted]")
        return text

    # -- JSON-RPC ----------------------------------------------------------------------------------------------

    def _definitions(self, principal: McpPrincipal) -> list[dict[str, Any]]:
        return [
            {"name": tool.name, "description": tool.description, "inputSchema": dict(tool.schema)}
            for tool in self._tools.values()
            if principal.scopes & tool.scopes
        ]

    async def _call_tool(
        self, principal: McpPrincipal, presented: str, name: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        tool = self._tools.get(name)
        if tool is None:
            raise InvalidParamsError("unknown tool")
        if not principal.scopes & tool.scopes:
            raise ToolForbidden("this token may not call this tool")
        problem = first_problem(arguments, tool.schema)
        if problem:
            raise InvalidParamsError(self.scrub(f"invalid arguments: {problem}", presented))
        try:
            outcome = tool.handler(arguments, CallContext(principal, name))
            if inspect.isawaitable(outcome):
                outcome = await outcome
        except (InvalidParamsError, ToolForbidden):
            raise
        except ToolError as exc:
            return _tool_result(self.scrub(str(exc), presented), is_error=True)
        except Exception as exc:  # noqa: BLE001 - a failing tool is a tool error, not a transport error
            logger.warning("tool %s failed: %s", name, type(exc).__name__)
            return _tool_result(self.scrub(f"{type(exc).__name__}: {exc}", presented), is_error=True)
        text = outcome if isinstance(outcome, str) else json.dumps(outcome, default=str)
        return _tool_result(text, is_error=False)

    async def _answer(self, principal: McpPrincipal, presented: str, message: Any) -> dict[str, Any] | None:
        """One JSON-RPC response, or ``None`` for a notification."""
        if not isinstance(message, dict) or message.get("jsonrpc") != JSONRPC_VERSION or "method" not in message:
            return _rpc_error(
                message.get("id") if isinstance(message, dict) else None, _ERR_INVALID_REQUEST, "Invalid Request"
            )
        request_id = message.get("id")
        if request_id is None:
            return None
        params = message.get("params")
        params = {} if params is None else params
        if not isinstance(params, dict):
            return _rpc_error(request_id, _ERR_PARAMS, "params must be an object")

        async def handle(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
            return await self._call_tool(principal, presented, name, arguments)

        try:
            result = await dispatch_envelope(
                str(message["method"]),
                params,
                server_name=self._server_name,
                tool_definitions=self._definitions(principal),
                handle_tool_call=handle,
            )
        except NotImplementedError as exc:
            return _rpc_error(request_id, _ERR_METHOD, str(exc))
        except InvalidParamsError as exc:
            return _rpc_error(request_id, _ERR_PARAMS, str(exc))
        except ToolForbidden as exc:
            return _rpc_error(request_id, _ERR_FORBIDDEN, str(exc))
        except Exception as exc:  # noqa: BLE001 - convert failures to JSON-RPC
            logger.warning("request failed: %s", type(exc).__name__)
            return _rpc_error(request_id, _ERR_INTERNAL, self.scrub(f"{type(exc).__name__}: {exc}", presented))
        return {"jsonrpc": JSONRPC_VERSION, "id": request_id, "result": result}

    async def _answer_all(self, principal: McpPrincipal, presented: str, payload: Any) -> Any:
        if isinstance(payload, list):
            if not payload:
                return _rpc_error(None, _ERR_INVALID_REQUEST, "Invalid Request")
            replies = [await self._answer(principal, presented, item) for item in payload]
            return [reply for reply in replies if reply is not None] or None
        return await self._answer(principal, presented, payload)

    @contextlib.contextmanager
    def _call_heartbeat(self, principal: McpPrincipal) -> Iterator[None]:
        """While a request of a specialist token is being served, call its beat every interval."""
        with self._beats_lock:
            hook = self._beats.get(principal.subject) if principal.kind != HOST_KIND else None
        if hook is None:
            yield
            return
        done = threading.Event()

        def beat() -> None:
            while not done.wait(self._heartbeat_seconds):
                try:
                    hook()
                except Exception as exc:  # noqa: BLE001 - a failing beat must not break the call
                    logger.warning("heartbeat beat failed: %s", type(exc).__name__)

        thread = threading.Thread(target=beat, name="hyperloom-mcp-heartbeat", daemon=True)
        thread.start()
        try:
            yield
        finally:
            done.set()
            thread.join()

    def handle_post(self, principal: McpPrincipal, presented: str, body: bytes) -> tuple[HTTPStatus, Any, bool]:
        """Run one POST body: the HTTP status, the JSON reply (``None`` for no content) and whether it initialized."""
        try:
            payload = json.loads(body.decode())
        except (UnicodeDecodeError, json.JSONDecodeError):
            return HTTPStatus.BAD_REQUEST, _rpc_error(None, _ERR_PARSE, "Parse error"), False
        with self._call_heartbeat(principal):
            reply = asyncio.run(self._answer_all(principal, presented, payload))
        if reply is None:
            return HTTPStatus.ACCEPTED, None, False
        initialized = _mentions_initialize(payload)
        return HTTPStatus.OK, reply, initialized

    # -- lifecycle ---------------------------------------------------------------------------------------------

    def start(self) -> None:
        """Serve on a daemon thread; returns once the socket is listening."""
        if self._httpd is not None:
            raise RuntimeError("the MCP server is already running")
        self._httpd = ThreadingHTTPServer((self._config.host, self._config.port), _make_handler(self))
        self._httpd.daemon_threads = True
        self._thread = threading.Thread(target=self._httpd.serve_forever, name="hyperloom-mcp", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        httpd, thread = self._httpd, self._thread
        self._httpd = self._thread = None
        if httpd is not None:
            httpd.shutdown()
            httpd.server_close()
        if thread is not None:
            thread.join(timeout=5)

    @property
    def port(self) -> int:
        if self._httpd is None:
            raise RuntimeError("the MCP server is not running")
        return int(self._httpd.server_address[1])

    @property
    def url(self) -> str:
        return f"http://{self._config.host}:{self.port}{MCP_PATH}"


def _tool_result(text: str, *, is_error: bool) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": text}], "isError": is_error}


def _rpc_error(request_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": JSONRPC_VERSION, "id": request_id, "error": {"code": code, "message": message}}


def _mentions_initialize(payload: Any) -> bool:
    items = payload if isinstance(payload, list) else [payload]
    return any(isinstance(item, dict) and item.get("method") == "initialize" for item in items)


def _presented_token(headers: Mapping[str, str]) -> str:
    direct = headers.get("X-Hyperloom-Token", "").strip()
    if direct:
        return direct
    scheme, _, value = headers.get("Authorization", "").partition(" ")
    return value.strip() if scheme.lower() == "bearer" else ""


def _make_handler(server: McpServer) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "hyperloom-mcp"

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - signature of the base class
            logger.debug("MCP request handled")

        def _send(self, status: HTTPStatus, body: Any = None, headers: Mapping[str, str] | None = None) -> None:
            raw = b"" if body is None else json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            for key, value in (headers or {}).items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(raw)

        def _on_path(self) -> bool:
            if self.path.split("?", 1)[0] == MCP_PATH:
                return True
            self._send(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return False

        def do_GET(self) -> None:
            if self._on_path():
                self._send(HTTPStatus.METHOD_NOT_ALLOWED, {"error": "no server push stream"}, {"Allow": "POST"})

        def do_POST(self) -> None:
            if not self._on_path():
                return
            presented = _presented_token(self.headers)
            principal = server.authenticate(presented)
            if principal is None:
                self._send(HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"}, {"WWW-Authenticate": "Bearer"})
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                length = -1
            if length < 0 or length > MAX_BODY_BYTES:
                self._send(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "body too large"})
                return
            status, reply, initialized = server.handle_post(principal, presented, self.rfile.read(length))
            headers = {SESSION_HEADER: uuid.uuid4().hex} if initialized else None
            self._send(status, reply, headers)

    return Handler


__all__ = [
    "HOST_KIND",
    "SPECIALIST_KIND",
    "CallContext",
    "InvalidParamsError",
    "McpPrincipal",
    "McpServer",
    "McpServerConfig",
    "ToolError",
    "ToolForbidden",
    "ToolHandler",
]
