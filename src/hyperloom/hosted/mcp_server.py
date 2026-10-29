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

A reply annotator (:meth:`McpServer.attach_reply_annotator`) adds one text block to the tool replies of specialist
tokens. The service keeps one opaque cursor per token for it, starts it empty, and moves it only when the reply
that carries the block has been written to the client.

Authentication: ``X-Hyperloom-Token`` or ``Authorization: Bearer``. The host token is the campaign's
MCP token, which the caller reads from :class:`~hyperloom.hosted.host_info.HostInfo` and passes in
:class:`McpServerConfig`; every other token was issued by this server to one specialist and dies with
:meth:`McpServer.revoke_specialist`. Issued tokens are kept as digests only, compared in constant time, and a token
is never written to a log or an error message.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import hashlib
import hmac
import inspect
import json
import logging
import secrets
import threading
import traceback
import uuid
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from hyperloom.hosted.schema_check import first_problem
from kernelforge.utils import InvalidParamsError, dispatch_envelope

logger = logging.getLogger(__name__)

JSONRPC_VERSION = "2.0"
MCP_PATH = "/mcp"
HOST_KIND = "host"
SPECIALIST_KIND = "specialist"
TOKEN_KINDS = frozenset({HOST_KIND, SPECIALIST_KIND})
MAX_BODY_BYTES = 1 << 20
MIN_TOKEN_LENGTH = 16
SESSION_HEADER = "Mcp-Session-Id"

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
class ReplyCall:
    """What a reply annotator learns about the call it annotates: the caller, the tool, its arguments, and whether
    the tool failed."""

    principal: McpPrincipal
    tool: str
    arguments: Mapping[str, Any]
    is_error: bool


@dataclass(frozen=True)
class ReplyNote:
    """What an annotator adds to one reply: ``text`` (empty to add nothing) and the ``cursor`` it resumes from next."""

    text: str
    cursor: str


#: Called with the call and the token's cursor (``None`` on the token's first call); a function or a coroutine
#: function returning a :class:`ReplyNote`.
ReplyAnnotator = Callable[[ReplyCall, str | None], Any]


@dataclass(frozen=True)
class RegisteredTool:
    name: str
    description: str
    schema: Mapping[str, Any]
    handler: ToolHandler
    scopes: frozenset[str]


@dataclass(frozen=True)
class McpServerConfig:
    """Plain values from the caller; a hosted run takes ``port`` and ``host_token`` from its ``HostInfo``."""

    host: str = "127.0.0.1"
    port: int = 0
    host_token: str = ""


def _digest(token: str) -> bytes:
    return hashlib.sha256(token.encode()).digest()


class _TokenRegistry:
    """Issued tokens by digest; thread-safe, and a lookup never short-circuits on the first match."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._by_digest: dict[bytes, McpPrincipal] = {}
        self._cursors: dict[bytes, str] = {}

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
                self._cursors.pop(key, None)
        return len(doomed)

    def cursor(self, token: str) -> str | None:
        with self._lock:
            return self._cursors.get(_digest(token))

    def move_cursor(self, token: str, was: str | None, now: str | None) -> None:
        """Keep ``now`` (``None`` for no cursor) for a live token whose cursor is still ``was``; a reply that raced
        a newer one loses."""
        key = _digest(token)
        with self._lock:
            if key in self._by_digest and self._cursors.get(key) == was:
                if now is None:
                    del self._cursors[key]
                else:
                    self._cursors[key] = now


class _ReplyNotes:
    """One request's view of its token's annotation cursor: each reply moves it, and it is kept for the replies that
    are being sent."""

    def __init__(self, annotator: ReplyAnnotator, tokens: _TokenRegistry, presented: str) -> None:
        self._annotator, self._tokens, self._presented = annotator, tokens, presented
        self._was = tokens.cursor(presented)
        self._now = self._was

    async def _ask(self, call: ReplyCall) -> ReplyNote | None:
        try:
            note = self._annotator(call, self._now)
            if inspect.isawaitable(note):
                note = await note
        except (OSError, ValueError) as exc:  # the annotator's feed is unreadable; the cursor stays put
            # The stack only: the message may carry a token.
            logger.warning(
                "reply annotator failed: %s\n%s", type(exc).__name__, "".join(traceback.format_tb(exc.__traceback__))
            )
            return None
        self._now = note.cursor
        return note

    async def start(self, call: ReplyCall) -> None:
        """Before the tool runs: a token without a cursor asks the annotator where the feed ends, and the text is
        dropped, so what arrives while its first call runs is still news for that call's reply."""
        if self._now is None:
            await self._ask(call)

    async def annotate(self, call: ReplyCall, result: dict[str, Any]) -> dict[str, Any]:
        """``result`` with the annotator's block after its own content; unchanged when its feed is unreadable."""
        note = await self._ask(call)
        if note is None or not note.text:
            return result
        return {**result, "content": [*result["content"], {"type": "text", "text": note.text}]}

    def keep(self) -> None:
        """Keep the cursor the replies moved to. Called before the first byte is written, so a client that sends its
        next call the moment it has the reply finds the cursor already moved."""
        if self._now is not None and self._now != self._was:
            self._tokens.move_cursor(self._presented, self._was, self._now)

    def drop(self) -> None:
        """The write failed, so the client never got the replies: put the cursor back where this request found it."""
        if self._now is not None and self._now != self._was:
            self._tokens.move_cursor(self._presented, self._now, self._was)


def _nothing_to_keep() -> None:
    return None


@dataclass(frozen=True)
class PostReply:
    """The answer to one POST; call ``keep`` just before ``body`` is written to the client, and ``drop`` if the
    write fails."""

    status: HTTPStatus
    body: Any
    initialized: bool = False
    keep: Callable[[], None] = _nothing_to_keep
    drop: Callable[[], None] = _nothing_to_keep


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
        self._annotator: ReplyAnnotator | None = None
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

    def attach_reply_annotator(self, annotator: ReplyAnnotator | None) -> None:
        """Add the block ``annotator`` returns to every tool reply a specialist token gets; the host gets none.

        The service holds one cursor per token and hands it to the annotator, which returns the text to add (empty
        for none) and the cursor to resume from. The cursor moves when the reply is written to the client, so a reply
        that was never sent leaves it where it was. A token's first call asks the annotator once with the cursor
        ``None``, before the tool runs: the annotator answers where the feed ends and the text is dropped. Attaching
        again replaces the annotator; ``None`` removes it.
        """
        self._annotator = annotator

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
        except Exception as exc:  # noqa: BLE001 - isolation boundary: one tool call, answered as a tool error
            logger.warning(
                "tool %s failed: %s\n%s", name, type(exc).__name__, "".join(traceback.format_tb(exc.__traceback__))
            )
            return _tool_result(self.scrub(f"{type(exc).__name__}: {exc}", presented), is_error=True)
        text = outcome if isinstance(outcome, str) else json.dumps(outcome, default=str)
        return _tool_result(text, is_error=False)

    async def _answer(
        self, principal: McpPrincipal, presented: str, message: Any, notes: _ReplyNotes | None = None
    ) -> dict[str, Any] | None:
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
            call = ReplyCall(principal, name, arguments, is_error=False)
            if notes is not None:
                await notes.start(call)
            result = await self._call_tool(principal, presented, name, arguments)
            if notes is None:
                return result
            return await notes.annotate(dataclasses.replace(call, is_error=bool(result["isError"])), result)

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
        except Exception as exc:  # noqa: BLE001 - outermost per-request boundary: a JSON-RPC internal error
            logger.warning(
                "request failed: %s\n%s", type(exc).__name__, "".join(traceback.format_tb(exc.__traceback__))
            )
            return _rpc_error(request_id, _ERR_INTERNAL, self.scrub(f"{type(exc).__name__}: {exc}", presented))
        return {"jsonrpc": JSONRPC_VERSION, "id": request_id, "result": result}

    async def _answer_all(
        self, principal: McpPrincipal, presented: str, payload: Any, notes: _ReplyNotes | None = None
    ) -> Any:
        if isinstance(payload, list):
            if not payload:
                return _rpc_error(None, _ERR_INVALID_REQUEST, "Invalid Request")
            replies = [await self._answer(principal, presented, item, notes) for item in payload]
            return [reply for reply in replies if reply is not None] or None
        return await self._answer(principal, presented, payload, notes)

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
                except OSError as exc:  # a failing beat must not break the call
                    logger.warning(
                        "heartbeat beat failed: %s\n%s",
                        type(exc).__name__,
                        "".join(traceback.format_tb(exc.__traceback__)),
                    )

        thread = threading.Thread(target=beat, name="hyperloom-mcp-heartbeat", daemon=True)
        thread.start()
        try:
            yield
        finally:
            done.set()
            thread.join()

    def handle_post(self, principal: McpPrincipal, presented: str, body: bytes) -> PostReply:
        """Run one POST body: what to send (status, the JSON reply, ``None`` for no content, whether it initialized)."""
        try:
            payload = json.loads(body.decode())
        except (UnicodeDecodeError, json.JSONDecodeError):
            return PostReply(HTTPStatus.BAD_REQUEST, _rpc_error(None, _ERR_PARSE, "Parse error"))
        notes = None
        if self._annotator is not None and principal.kind == SPECIALIST_KIND:
            notes = _ReplyNotes(self._annotator, self._tokens, presented)
        with self._call_heartbeat(principal):
            reply = asyncio.run(self._answer_all(principal, presented, payload, notes))
        keep, drop = (notes.keep, notes.drop) if notes is not None else (_nothing_to_keep, _nothing_to_keep)
        if reply is None:
            return PostReply(HTTPStatus.ACCEPTED, None, keep=keep, drop=drop)
        return PostReply(HTTPStatus.OK, reply, _mentions_initialize(payload), keep, drop)

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
            reply = server.handle_post(principal, presented, self.rfile.read(length))
            headers = {SESSION_HEADER: uuid.uuid4().hex} if reply.initialized else None
            reply.keep()
            try:
                self._send(reply.status, reply.body, headers)
            except BaseException:
                reply.drop()
                raise

    return Handler


__all__ = [
    "HOST_KIND",
    "SPECIALIST_KIND",
    "CallContext",
    "InvalidParamsError",
    "McpPrincipal",
    "McpServer",
    "McpServerConfig",
    "PostReply",
    "ReplyAnnotator",
    "ReplyCall",
    "ReplyNote",
    "ToolError",
    "ToolForbidden",
    "ToolHandler",
]
