# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A sandboxed specialist's network: a namespace of its own, where it reaches only what its launch names.

bubblewrap's ``--unshare-net`` leaves the sandbox its own loopback and nothing else, so nothing that listens on this
host is reachable from it: not the run's Ray head (whose dashboard Jobs API, client server and workers run what they
are sent as the app), not another agent's server. What the specialist needs comes in over Unix sockets in its scratch
``/tmp``, which a relay inside the sandbox (:data:`RELAY`, started ahead of the agent, which it then becomes) serves
on that loopback:

- each loopback endpoint the launch names (an HTTP MCP server, an LLM gateway on this host), at the same port, relayed
  to that port here and nowhere else;
- one HTTP proxy, named inside by ``HTTPS_PROXY`` and ``HTTP_PROXY`` with loopback exempt (``NO_PROXY``), which this
  process serves (:class:`NetworkBridge`): it connects only to the other hosts and ports the launch names
  (:func:`specialist_reach`), and answers 403 to anything else. A ``CONNECT`` reaches a named host and port; a plain
  ``http`` request only an endpoint the launch names as ``http``. Where the launch's environment names the app's own
  proxy, the bridge itself opens a ``CONNECT`` to the checked target through it and carries the specialist's bytes only
  inside that tunnel, so nothing it sends after its first request reaches the app's proxy as a request of its own.
"""

from __future__ import annotations

import base64
import configparser
import contextlib
import ipaddress
import json
import logging
import re
import socket
import sys
import threading
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence
from urllib.parse import unquote, urlsplit

from hyperloom_common.fs_utils import read_bytes_nofollow

log = logging.getLogger(__name__)

#: Where the bridge's sockets are, in the sandbox's scratch ``/tmp``.
SOCKET_DIR = ".hl-net"
_PROXY_SOCKET = "proxy.sock"
_MCP_FLAG = "--mcp-config"
_BASE_URL_SUFFIX = "_BASE_URL"
#: The one command-line setting that names an LLM endpoint: codex's ``-c model_providers.<name>.base_url=<value>``,
#: whose value this repo writes as a TOML string (:mod:`hyperloom.common.codex_session`).
_CONFIG_FLAGS = frozenset({"-c", "--config"})
_CODEX_BASE_URL = re.compile(r"model_providers\.[A-Za-z0-9_-]+\.base_url=(.+)", re.DOTALL)
_END_OF_OPTIONS = "--"
_BEDROCK = "CLAUDE_CODE_USE_BEDROCK"
#: The region variables a Bedrock-backed claude CLI reads; the first two name its region, the last its small model's.
_AWS_REGION_VARIABLES = ("AWS_REGION", "AWS_DEFAULT_REGION", "ANTHROPIC_SMALL_FAST_MODEL_AWS_REGION")
_AWS_DEFAULT_REGION = "us-east-1"
#: The keys of an AWS config file that name a region: a profile's own, and its SSO login's.
_AWS_CONFIG_REGION_KEYS = ("region", "sso_region")
_AWS_REGION = re.compile(r"[a-z]{2}(?:-[a-z]+)+-\d+")
#: The AWS hosts a Bedrock login reaches in each of its regions: the runtime, STS (an assumed role, a web identity),
#: and IAM Identity Center's portal and OIDC (an SSO profile); and the global STS endpoint.
_AWS_REGIONAL_HOSTS = ("bedrock-runtime", "sts", "portal.sso", "oidc")
_AWS_GLOBAL_STS = "https://sts.amazonaws.com"
_TRUE = frozenset({"1", "true", "yes", "on"})
#: Each provider's API host, reached when the launch names no base URL of its own for it.
_DEFAULT_APIS = {"ANTHROPIC_BASE_URL": "https://api.anthropic.com", "OPENAI_BASE_URL": "https://api.openai.com"}
_DEFAULT_PORTS = {"http": 80, "https": 443}
_HEAD_LIMIT = 64 * 1024
_HEAD_TIMEOUT_SEC = 30.0
_CONNECT_TIMEOUT_SEC = 30.0
_CHUNK = 65536
#: How the bridge reads and writes HTTP heads (byte for byte), and the text it reads from files and logins.
_HEAD_CODEC = "latin-1"
_TEXT_CODEC = "utf-8"
_FIRST_UNPRIVILEGED_PORT = 1024
_ESTABLISHED = b"HTTP/1.1 200 Connection established\r\n\r\n"
#: The app's proxy settings the bridge goes through; nothing else of the launch's environment is kept.
_PROXY_VARIABLES = ("HTTPS_PROXY", "HTTP_PROXY", "NO_PROXY")
#: Taken from the sandbox's environment: the relay names its own proxy, and the app's (with any login in its URL)
#: would show in the relay's ``/proc/<pid>/environ``, which the agent reads.
SANDBOX_DROPPED = (*_PROXY_VARIABLES, "ALL_PROXY", "https_proxy", "http_proxy", "no_proxy", "all_proxy")

#: Run inside the sandbox as ``python -I -S -c RELAY SPEC CMD...``: listen on the sandbox's loopback for each
#: forwarded port (``127.0.0.1``, and ``::1`` where it has one; one it cannot bind, below 1024 without the
#: capability, is said on stderr and left out) and for the proxy, name the proxy in the environment,
#: leave a detached process relaying each connection to its Unix socket, and exec ``CMD``. The relay is bubblewrap's
#: init's child, not the agent's, and ends with the sandbox.
RELAY = r"""
import json, os, socket, sys, threading
spec, cmd = json.loads(sys.argv[1]), sys.argv[2:]
def listen(host, port):
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    sock = socket.socket(family, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    if family == socket.AF_INET6:
        sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
    sock.bind((host, port))
    sock.listen(128)
    return sock
pairs = []
for port, path in spec["forward"]:
    try:
        pairs.append((listen("127.0.0.1", port), path))
    except OSError as exc:
        sys.stderr.write("specialist sandbox: cannot serve 127.0.0.1:%d inside: %s\n" % (port, exc))
        continue
    try:
        pairs.append((listen("::1", port), path))
    except OSError:
        pass
proxy = listen("127.0.0.1", 0)
pairs.append((proxy, spec["proxy"]))
url = "http://127.0.0.1:%d" % proxy.getsockname()[1]
for name in ("HTTPS_PROXY", "HTTP_PROXY", "https_proxy", "http_proxy"):
    os.environ[name] = url
for name in ("ALL_PROXY", "all_proxy"):
    os.environ.pop(name, None)
os.environ["NO_PROXY"] = os.environ["no_proxy"] = "localhost,127.0.0.1,::1"
def pump(src, dst):
    try:
        while True:
            data = src.recv(65536)
            if not data:
                break
            dst.sendall(data)
    except OSError:
        pass
    try:
        dst.shutdown(socket.SHUT_WR)
    except OSError:
        pass
def carry(conn, path):
    up = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        up.connect(path)
        back = threading.Thread(target=pump, args=(up, conn), daemon=True)
        back.start()
        pump(conn, up)
        back.join()
    except OSError:
        pass
    finally:
        conn.close()
        up.close()
def serve(listener, path):
    while True:
        conn, _ = listener.accept()
        threading.Thread(target=carry, args=(conn, path), daemon=True).start()
child = os.fork()
if child == 0:
    if os.fork() == 0:
        null = os.open(os.devnull, os.O_RDWR)
        for fd in (0, 1, 2):
            os.dup2(null, fd)
        threads = [threading.Thread(target=serve, args=pair, daemon=True) for pair in pairs]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    os._exit(0)
os.waitpid(child, 0)
for listener, _ in pairs:
    listener.close()
try:
    os.execvp(cmd[0], cmd)
except OSError as exc:
    sys.stderr.write("specialist sandbox: cannot run %s: %s\n" % (cmd[0], exc))
    os._exit(127)
"""


@dataclass(frozen=True)
class Reach:
    """What a sandboxed specialist reaches: ``loopback`` ports of this host (each with the host name its launch gave,
    ``localhost`` say), served at the same port inside; ``remote`` hosts and ports, tunnelled to by the bridge's proxy;
    and of those, the ``plain`` ones the launch names as ``http``, which the proxy also takes a plain request for."""

    loopback: Mapping[int, str] = field(default_factory=dict)
    remote: frozenset[tuple[str, int]] = frozenset()
    plain: frozenset[tuple[str, int]] = frozenset()


def _endpoint(url: str) -> tuple[str, int] | None:
    """``(host, port)`` of an ``http``/``https`` ``url``, the host lower-cased; ``None`` for anything else."""
    try:
        parts = urlsplit(url.strip())
        host, port = parts.hostname, parts.port or _DEFAULT_PORTS.get(parts.scheme)
    except ValueError:
        return None
    if not host or port is None or parts.scheme not in _DEFAULT_PORTS:
        return None
    return host.lower(), port


def _authority(target: tuple[str, int]) -> str:
    """``host:port`` of ``target`` as a request line or ``Host`` header writes it: an IPv6 host in brackets."""
    host, port = target
    return f"[{host}]:{port}" if ":" in host else f"{host}:{port}"


def _is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _mcp_urls(path: str | None) -> Iterator[str]:
    """The ``url`` of each server in the Claude-style MCP config at ``path`` (read without following a link); none
    when it cannot be read."""
    if not path:
        return
    try:
        payload = json.loads(read_bytes_nofollow(path, limit=1 << 20))
    except (OSError, ValueError):
        return
    servers = payload.get("mcpServers") if isinstance(payload, dict) else None
    for server in servers.values() if isinstance(servers, dict) else ():
        url = server.get("url") if isinstance(server, dict) else None
        if isinstance(url, str):
            yield url


def _options(cmd: Sequence[str]) -> Sequence[str]:
    """``cmd`` up to its first ``--``: after it, its words are operands (a dsh task, say), never settings."""
    return cmd[: cmd.index(_END_OF_OPTIONS)] if _END_OF_OPTIONS in cmd else cmd


def _setting_url(value: str) -> str | None:
    """The URL a codex ``-c`` value gives: a TOML basic string (as this repo writes it), a literal string, or bare."""
    value = value.strip()
    if value.startswith('"'):
        try:
            found = json.loads(value)
        except ValueError:
            return None
        return found if isinstance(found, str) else None
    if len(value) >= 2 and value[0] == value[-1] == "'":
        return value[1:-1]
    return value


def _codex_base_urls(cmd: Sequence[str]) -> Iterator[str]:
    """The value of each codex ``-c model_providers.<name>.base_url=`` setting in ``cmd``'s options; no other word of
    the command line (a prompt, a task text quoting ``base_url=``) names an endpoint."""
    options = _options(cmd)
    for flag, setting in zip(options, options[1:]):
        found = _CODEX_BASE_URL.fullmatch(setting) if flag in _CONFIG_FLAGS else None
        url = _setting_url(found.group(1)) if found else None
        if url:
            yield url


def _aws_config_regions(env: Mapping[str, str]) -> Iterator[str]:
    """The regions the AWS config file ``AWS_CONFIG_FILE`` names (read without following a link): each section's
    ``region`` and ``sso_region``; none when it cannot be read."""
    path = env.get("AWS_CONFIG_FILE") or ""
    if not path:
        return
    parser = configparser.RawConfigParser(strict=False, interpolation=None)
    try:
        parser.read_string(read_bytes_nofollow(path, limit=1 << 20).decode(_TEXT_CODEC))
    except (OSError, UnicodeDecodeError, configparser.Error):
        return
    for section in parser.sections():
        for key in _AWS_CONFIG_REGION_KEYS:
            if parser.has_option(section, key):
                yield parser.get(section, key)


def _bedrock_urls(env: Mapping[str, str]) -> Iterator[str]:
    """What a Bedrock login in ``env`` reaches: in its region (``us-east-1`` where it names none), its small model's
    and each its AWS config file names, the runtime, STS and the SSO portal and OIDC hosts; and the global STS."""
    regions = {env.get(name, "").strip() for name in _AWS_REGION_VARIABLES} | set(_aws_config_regions(env))
    if not (env.get("AWS_REGION") or env.get("AWS_DEFAULT_REGION")):
        regions.add(_AWS_DEFAULT_REGION)
    for region in sorted(region.strip().lower() for region in regions):
        if _AWS_REGION.fullmatch(region):
            for service in _AWS_REGIONAL_HOSTS:
                yield f"https://{service}.{region}.amazonaws.com"
    yield _AWS_GLOBAL_STS


def _launch_urls(cmd: Sequence[str], env: Mapping[str, str], mcp_config: str | None) -> Iterator[str]:
    """Every LLM endpoint and MCP server the launch names: the base URLs its command line sets (codex's
    ``-c model_providers.<name>.base_url=...``; never other text there, a prompt say), its ``*_BASE_URL`` variables
    (each provider's API host where it names none, and what a Bedrock login reaches when ``CLAUDE_CODE_USE_BEDROCK``
    is on), and the servers of the MCP configs it is given (``mcp_config``, and the one ``--mcp-config`` names)."""
    yield from _codex_base_urls(cmd)
    for name, value in env.items():
        if name.endswith(_BASE_URL_SUFFIX) and value:
            yield value
    for name, default in _DEFAULT_APIS.items():
        if not env.get(name):
            yield default
    if (env.get(_BEDROCK) or "").strip().lower() in _TRUE:
        yield from _bedrock_urls(env)
    yield from _mcp_urls(mcp_config)
    options = _options(cmd)
    if _MCP_FLAG in options[:-1]:
        yield from _mcp_urls(options[options.index(_MCP_FLAG) + 1])


def specialist_reach(cmd: Sequence[str], env: Mapping[str, str], mcp_config: str | None) -> Reach:
    """What a specialist launched as ``cmd`` with ``env`` and the operator's ``mcp_config`` reaches from its own
    network: the LLM endpoints and MCP servers its launch names (:func:`_launch_urls`), and nothing else."""
    loopback: dict[int, str] = {}
    remote: set[tuple[str, int]] = set()
    plain: set[tuple[str, int]] = set()
    for url in _launch_urls(cmd, env, mcp_config):
        found = _endpoint(url)
        if found is None:
            continue
        if _is_loopback(found[0]):
            loopback.setdefault(found[1], found[0])
            continue
        remote.add(found)
        if url.strip().lower().startswith("http://"):
            plain.add(found)
    return Reach(loopback, frozenset(remote), frozenset(plain))


def _target(line: str) -> tuple[str, int] | None:
    """The host and port a proxy request line asks for (``CONNECT host:port`` or an absolute ``http`` URL)."""
    words = line.split()
    if len(words) != 3:
        return None
    method, where = words[0].upper(), words[1]
    if method == "CONNECT":
        return _endpoint(f"https://{where}")
    return _endpoint(where) if where.lower().startswith("http://") else None


def _origin_form(data: bytes) -> bytes:
    """A plain proxy request ``data`` with its absolute URL replaced by the path and query, as sent to the server."""
    first, rest = data.split(b"\r\n", 1)
    method, where, version = first.split()
    parts = urlsplit(where.decode(_HEAD_CODEC))
    path = (parts.path or "/") + (f"?{parts.query}" if parts.query else "")
    return b" ".join((method, path.encode(_HEAD_CODEC), version)) + b"\r\n" + rest


def _read_head(conn: socket.socket) -> bytes:
    """What the peer sent up to the end of its head (and any bytes after it); empty when it sent none."""
    data = b""
    while b"\r\n\r\n" not in data and len(data) < _HEAD_LIMIT:
        chunk = conn.recv(_CHUNK)
        if not chunk:
            return b""
        data += chunk
    return data


def _tunnel(remote: socket.socket, upstream: _Upstream, target: tuple[str, int]) -> bytes | None:
    """Open a tunnel to ``target`` through the app's proxy at the other end of ``remote``, with the ``CONNECT`` this
    process writes: the bytes that came after the proxy's ``200``, or ``None`` when it answered anything else."""
    where = _authority(target).encode("ascii")
    remote.sendall(b"CONNECT " + where + b" HTTP/1.1\r\nHost: " + where + b"\r\n" + upstream.authorization + b"\r\n")
    remote.settimeout(_HEAD_TIMEOUT_SEC)
    try:
        answer = _read_head(remote)
    except OSError:
        return None
    remote.settimeout(None)
    if b"\r\n\r\n" not in answer:
        return None
    status = answer.split(b"\r\n", 1)[0].split(b" ")
    if len(status) < 2 or not status[0].startswith(b"HTTP/") or status[1] != b"200":
        return None
    return answer[answer.index(b"\r\n\r\n") + 4 :]


def _refuse(conn: socket.socket, status: str, why: str) -> None:
    body = (why + "\n").encode(_TEXT_CODEC, "replace")
    head = f"HTTP/1.1 {status}\r\nContent-Type: text/plain\r\nContent-Length: {len(body)}\r\nConnection: close\r\n\r\n"
    with contextlib.suppress(OSError):
        conn.sendall(head.encode(_HEAD_CODEC) + body)


def _pump(src: socket.socket, dst: socket.socket) -> None:
    with contextlib.suppress(OSError):
        while chunk := src.recv(_CHUNK):
            dst.sendall(chunk)
    with contextlib.suppress(OSError):
        dst.shutdown(socket.SHUT_WR)


def _carry(conn: socket.socket, remote: socket.socket) -> None:
    """Relay bytes both ways between ``conn`` and ``remote`` until each side has closed, then close both."""
    back = threading.Thread(target=_pump, args=(remote, conn), daemon=True)
    back.start()
    _pump(conn, remote)
    back.join()
    remote.close()


@dataclass(frozen=True)
class _Upstream:
    """The app's own proxy for a request: where it is, and the ``Proxy-Authorization`` its URL's login gives."""

    address: tuple[str, int]
    authorization: bytes = b""


def _upstream(env: Mapping[str, str], target: tuple[str, int], connect: bool) -> _Upstream | None:
    """The proxy ``env`` routes a request for ``target`` through (``HTTPS_PROXY`` for a tunnel, ``HTTP_PROXY`` for
    plain HTTP), unless ``NO_PROXY`` exempts the host; ``None``: connect directly."""
    url = env.get("HTTPS_PROXY" if connect else "HTTP_PROXY") or ""
    if not url or urllib.request.proxy_bypass_environment(target[0], {"no": env.get("NO_PROXY", "")}):
        return None
    parts = urlsplit(url if "://" in url else f"http://{url}")
    try:
        port = parts.port or 80
    except ValueError:
        return None
    if not parts.hostname:
        return None
    login = ""
    if parts.username is not None:
        login = f"{unquote(parts.username)}:{unquote(parts.password or '')}"
    authorization = b""
    if login:
        authorization = b"Proxy-Authorization: Basic " + base64.b64encode(login.encode(_TEXT_CODEC)) + b"\r\n"
    return _Upstream((parts.hostname, port), authorization)


class NetworkBridge:
    """This process's side of a sandbox's network (the module doc): a Unix socket per loopback port of ``reach``, and
    the proxy's, in ``directory``; :meth:`relay` gives the command that serves them inside, where ``directory`` shows
    as ``inside``. :meth:`close` stops serving."""

    def __init__(self, reach: Reach, directory: Path, inside: str, env: Mapping[str, str]) -> None:
        self.reach = reach
        self.directory = directory
        self.inside = inside
        self._env = {name: env[name] for name in _PROXY_VARIABLES if env.get(name)}
        self._listeners: list[socket.socket] = []
        self._open: set[socket.socket] = set()
        self._lock = threading.Lock()
        self._closed = False

    @classmethod
    def open(cls, reach: Reach, directory: Path, inside: str, env: Mapping[str, str]) -> NetworkBridge:
        """Make ``directory`` (mode 0700; it must not exist) and serve ``reach`` there.

        Raises:
            OSError: The directory or a socket could not be made.
        """
        directory.mkdir(mode=0o700)
        privileged = sorted(port for port in reach.loopback if port < _FIRST_UNPRIVILEGED_PORT)
        if privileged:
            log.warning(
                "a sandboxed specialist cannot reach the loopback ports %s it is given: its network is its own, and "
                "it may not listen below %d there",
                ", ".join(map(str, privileged)),
                _FIRST_UNPRIVILEGED_PORT,
            )
        bridge = cls(reach, directory, inside, env)
        try:
            for port, host in reach.loopback.items():
                bridge._serve(f"{port}.sock", lambda conn, address=(host, port): bridge._forward(conn, address))
            bridge._serve(_PROXY_SOCKET, bridge._proxy)
        except OSError:
            bridge.close()
            raise
        return bridge

    def relay(self, cmd: Sequence[str]) -> list[str]:
        """``cmd`` started by :data:`RELAY` under this interpreter, which serves the bridge inside the sandbox."""
        spec = {
            "forward": [[port, f"{self.inside}/{port}.sock"] for port in self.reach.loopback],
            "proxy": f"{self.inside}/{_PROXY_SOCKET}",
        }
        return [sys.executable, "-I", "-S", "-c", RELAY, json.dumps(spec), *cmd]

    def close(self) -> None:
        """Stop accepting, and end every connection still relayed."""
        with self._lock:
            self._closed = True
            sockets = [*self._listeners, *self._open]
            self._listeners, self._open = [], set()
        for sock in sockets:
            with contextlib.suppress(OSError):
                sock.shutdown(socket.SHUT_RDWR)
            sock.close()

    def _serve(self, name: str, handle: Any) -> None:
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            listener.bind(str(self.directory / name))
            listener.listen(128)
        except OSError:
            listener.close()
            raise
        self._listeners.append(listener)
        threading.Thread(target=self._accept, args=(listener, handle), daemon=True, name=f"sandbox-net-{name}").start()

    def _accept(self, listener: socket.socket, handle: Any) -> None:
        while True:
            try:
                conn, _ = listener.accept()
            except OSError:
                return  # closed
            if not self._track(conn):
                conn.close()
                return
            threading.Thread(target=self._handled, args=(conn, handle), daemon=True).start()

    def _track(self, sock: socket.socket) -> bool:
        with self._lock:
            if self._closed:
                return False
            self._open.add(sock)
            return True

    def _handled(self, conn: socket.socket, handle: Any) -> None:
        try:
            handle(conn)
        except OSError:
            log.debug("sandbox network: a relayed connection failed", exc_info=True)
        finally:
            with self._lock:
                self._open.discard(conn)
            conn.close()

    def _connect(self, address: tuple[str, int]) -> socket.socket | None:
        try:
            remote = socket.create_connection(address, timeout=_CONNECT_TIMEOUT_SEC)
        except OSError:
            return None
        remote.settimeout(None)
        if not self._track(remote):
            remote.close()
            return None
        return remote

    def _forward(self, conn: socket.socket, address: tuple[str, int]) -> None:
        remote = self._connect(address)
        if remote is not None:
            self._relayed(conn, remote)

    def _relayed(self, conn: socket.socket, remote: socket.socket) -> None:
        try:
            _carry(conn, remote)
        finally:
            with self._lock:
                self._open.discard(remote)

    def _drop(self, remote: socket.socket) -> None:
        with self._lock:
            self._open.discard(remote)
        remote.close()

    def _proxy(self, conn: socket.socket) -> None:
        """Serve one proxy connection: a ``CONNECT`` to a host and port of the reach, or a plain request to one it
        names as ``http``; 403 for anything else. Through the app's proxy, the bytes go only inside a tunnel this
        process opened to the checked target (:func:`_tunnel`), never to that proxy as requests of their own."""
        conn.settimeout(_HEAD_TIMEOUT_SEC)
        data = _read_head(conn)
        conn.settimeout(None)
        if b"\r\n\r\n" not in data:
            return
        line = data.split(b"\r\n", 1)[0].decode(_HEAD_CODEC)
        target = _target(line)
        connect = line.upper().startswith("CONNECT ")
        if target is None or target not in (self.reach.remote if connect else self.reach.plain):
            shown = _authority(target) if target else line[:200]
            _refuse(conn, "403 Forbidden", f"not reachable from the specialist's sandbox: {shown}")
            return
        upstream = _upstream(self._env, target, connect)
        remote = self._connect(upstream.address if upstream else target)
        early = b"" if remote is None or upstream is None else _tunnel(remote, upstream, target)
        if remote is None or early is None:
            if remote is not None:
                self._drop(remote)
            _refuse(conn, "502 Bad Gateway", f"cannot connect to {_authority(target)}")
            return
        end = data.index(b"\r\n\r\n") + 4
        if connect:
            conn.sendall(_ESTABLISHED + early)
            if data[end:]:
                remote.sendall(data[end:])
        else:
            if early:
                conn.sendall(early)
            remote.sendall(_origin_form(data) if upstream is not None else data)
        self._relayed(conn, remote)


__all__ = ["NetworkBridge", "RELAY", "Reach", "SANDBOX_DROPPED", "SOCKET_DIR", "specialist_reach"]
