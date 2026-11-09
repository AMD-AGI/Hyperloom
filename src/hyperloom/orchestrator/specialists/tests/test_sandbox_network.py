# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What a sandboxed specialist reaches from its own network: the endpoints its launch names, served by the bridge
over Unix sockets, and a 403 for anything else. No namespaces here; the real sandbox is in test_specialist_sandbox."""

from __future__ import annotations

import base64
import contextlib
import json
import socket
import threading
import time
from pathlib import Path
from typing import Callable, Iterator

import pytest

from hyperloom.orchestrator.specialists import dsh_launch
from hyperloom.orchestrator.specialists.sandbox_network import NetworkBridge, Reach, specialist_reach

_INSIDE = "/tmp/.hl-net"


@pytest.fixture
def listeners() -> Iterator[Callable[[Callable[[socket.socket], None]], tuple[int, list[bytes]]]]:
    """Start TCP listeners on 127.0.0.1, each answering every connection with ``handle``; each gives its port and
    the first bytes of every connection it took."""
    started: list[socket.socket] = []

    def start(handle: Callable[[socket.socket], None]) -> tuple[int, list[bytes]]:
        server = socket.create_server(("127.0.0.1", 0))
        started.append(server)
        taken: list[bytes] = []

        def serve() -> None:
            while True:
                try:
                    conn, _ = server.accept()
                except OSError:
                    return
                with conn:
                    taken.append(conn.recv(65536))
                    handle(conn)

        threading.Thread(target=serve, daemon=True).start()
        return server.getsockname()[1], taken

    yield start
    for server in started:
        server.close()


def _echo(conn: socket.socket) -> None:
    conn.sendall(b"echo")


def _tunnel_ok(conn: socket.socket) -> None:
    conn.sendall(b"HTTP/1.1 200 Connection established\r\n\r\n")


def _unix(bridge: NetworkBridge, name: str) -> socket.socket:
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(10)
    sock.connect(str(bridge.directory / name))
    return sock


def _ask(bridge: NetworkBridge, name: str, data: bytes) -> bytes:
    """Send ``data`` through the bridge's socket ``name`` and read until it closes or 1 s of quiet."""
    with _unix(bridge, name) as sock:
        sock.sendall(data)
        got = b""
        sock.settimeout(1)
        with contextlib.suppress(TimeoutError):  # a second of quiet ends a reply on a connection left open
            while chunk := sock.recv(65536):
                got += chunk
        return got


@pytest.fixture
def bridges(tmp_path: Path) -> Iterator[Callable[..., NetworkBridge]]:
    opened: list[NetworkBridge] = []

    def open_bridge(reach: Reach, env: dict[str, str] | None = None) -> NetworkBridge:
        bridge = NetworkBridge.open(reach, tmp_path / f"net-{len(opened)}", _INSIDE, env or {})
        opened.append(bridge)
        return bridge

    yield open_bridge
    for bridge in opened:
        bridge.close()


def test_a_launch_reaches_the_llm_endpoints_and_mcp_servers_it_names_and_no_other_url_it_carries(tmp_path: Path):
    operator = tmp_path / "mcp.json"
    operator.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "pr": {"type": "http", "url": "http://127.0.0.1:7001/mcp"},
                    "far": {"type": "http", "url": "https://mcp.example.test/x"},
                    "local": {"command": "server"},
                }
            }
        ),
        encoding="utf-8",
    )
    staged = tmp_path / "staged.json"
    staged.write_text(json.dumps({"mcpServers": {"t": {"type": "http", "url": "http://localhost:7002/mcp"}}}))
    cmd = [
        "codex",
        "exec",
        "-c",
        'model_providers.gw.base_url="https://gw.example.test:8443/v1"',
        "--mcp-config",
        str(staged),
        "the prompt names https://elsewhere.example.test/ and http://127.0.0.1:8265/api/jobs/",
    ]
    env = {"ANTHROPIC_BASE_URL": "http://127.0.0.1:4000", "HF_ENDPOINT": "https://hf-mirror.example.test"}

    reach = specialist_reach(cmd, env, str(operator))

    assert dict(reach.loopback) == {4000: "127.0.0.1", 7001: "127.0.0.1", 7002: "localhost"}
    assert reach.remote == {("gw.example.test", 8443), ("mcp.example.test", 443), ("api.openai.com", 443)}


def test_a_base_url_in_a_task_text_on_the_command_line_widens_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """dsh takes its task as its last argument; text there (a quoted note, another specialist's summary) is never a
    setting, nor is a codex prompt word, nor a ``-c`` setting after ``--``."""
    for name in ("OPENAI_BASE_URL", "OPENAI_API_KEY", "DEEPSEEK_BASE_URL", "DEEPSEEK_API_KEY", "ANTHROPIC_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    gateway = {"OPENAI_BASE_URL": "https://gateway.example.test/v1", "OPENAI_API_KEY": "key-for-the-test"}
    monkeypatch.setenv("OPENAI_BASE_URL", gateway["OPENAI_BASE_URL"])
    monkeypatch.setenv("OPENAI_API_KEY", gateway["OPENAI_API_KEY"])
    task = "the Ray head says base_url=http://127.0.0.1:8265 and base_url=https://evil.example.test:9 in its notes"
    request = dsh_launch.DshLaunchRequest("/usr/bin/dsh", "m", None, tmp_path / "ws", "system", task)
    launch = dsh_launch.build_dsh_launch(request, gateway)
    assert launch.cmd[-1] == task
    cmd = [*launch.cmd, "-c", "model_providers.x.base_url=https://after.example.test"]
    env = {"OPENAI_BASE_URL": gateway["OPENAI_BASE_URL"], **launch.env_additions}

    dsh = specialist_reach(cmd, env, None)
    codex = specialist_reach(["codex", "exec", "--json", task, "-"], {}, None)

    assert dict(dsh.loopback) == {} and dict(codex.loopback) == {}
    assert dsh.remote == {("gateway.example.test", 443), ("api.anthropic.com", 443)}
    assert codex.remote == {("api.anthropic.com", 443), ("api.openai.com", 443)}


def test_each_providers_api_host_is_reached_where_the_launch_names_no_base_url_and_bedrock_in_its_region():
    reach = specialist_reach(["claude"], {"CLAUDE_CODE_USE_BEDROCK": "1", "AWS_REGION": "eu-west-1"}, None)
    default = specialist_reach(["claude"], {"CLAUDE_CODE_USE_BEDROCK": "1"}, None)

    assert dict(reach.loopback) == {}
    assert reach.remote == {
        ("api.anthropic.com", 443),
        ("api.openai.com", 443),
        *_aws_hosts("eu-west-1"),
        ("sts.amazonaws.com", 443),
    }
    assert ("bedrock-runtime.us-east-1.amazonaws.com", 443) in default.remote
    assert reach.plain == frozenset()


def _aws_hosts(region: str) -> set[tuple[str, int]]:
    """The AWS hosts a Bedrock login reaches in ``region``: the runtime, STS and the SSO portal and OIDC."""
    services = ("bedrock-runtime", "sts", "portal.sso", "oidc")
    return {(f"{service}.{region}.amazonaws.com", 443) for service in services}


def test_a_bedrock_login_reaches_sts_and_sso_in_each_region_it_names_and_its_small_models_runtime(tmp_path: Path):
    """An assumed role or web identity logs in through STS, an SSO profile through IAM Identity Center's portal and
    OIDC in its ``sso_region``, and the small model may run in a second region: each is reached, and nothing a region
    setting could smuggle in as a host."""
    config = tmp_path / "aws-config"
    config.write_text(
        "[profile role]\nregion = eu-central-1\nrole_arn = arn:aws:iam::1:role/r\nsource_profile = base\n"
        "[sso-session corp]\nsso_region = ap-northeast-1\nsso_start_url = https://corp.awsapps.com/start\n"
        "[profile odd]\nregion = evil.example.test/x\n",
        encoding="utf-8",
    )
    env = {
        "CLAUDE_CODE_USE_BEDROCK": "true",
        "AWS_REGION": "us-west-2",
        "ANTHROPIC_SMALL_FAST_MODEL_AWS_REGION": "eu-west-1",
        "AWS_CONFIG_FILE": str(config),
    }

    reach = specialist_reach(["claude"], env, None)

    regions = ("us-west-2", "eu-west-1", "eu-central-1", "ap-northeast-1")
    aws = {found for found in reach.remote if found[0].endswith(".amazonaws.com")}
    assert aws == set().union(*map(_aws_hosts, regions)) | {("sts.amazonaws.com", 443)}
    assert not any("evil" in host for host, _port in reach.remote)
    assert ("bedrock-runtime.us-east-1.amazonaws.com", 443) not in reach.remote  # a region is named


def test_a_loopback_endpoint_is_relayed_to_its_own_port_and_no_other_port_has_a_socket(listeners, bridges):
    port, taken = listeners(_echo)
    bridge = bridges(Reach({port: "127.0.0.1"}))

    assert _ask(bridge, f"{port}.sock", b"hello") == b"echo"
    assert taken == [b"hello"]
    assert sorted(path.name for path in bridge.directory.iterdir()) == sorted([f"{port}.sock", "proxy.sock"])


def test_the_proxy_tunnels_only_to_a_host_and_port_the_launch_names(listeners, bridges):
    named, named_taken = listeners(_echo)
    head, head_taken = listeners(_echo)  # a Ray head's port, say
    tls, tls_taken = listeners(_echo)  # an endpoint the launch names as https
    endpoint = ("127.0.0.1", named)
    bridge = bridges(Reach(remote=frozenset({endpoint, ("127.0.0.1", tls)}), plain=frozenset({endpoint})))

    tunnel = _ask(bridge, "proxy.sock", f"CONNECT 127.0.0.1:{named} HTTP/1.1\r\nHost: x\r\n\r\nafter".encode())
    refused = _ask(bridge, "proxy.sock", f"CONNECT 127.0.0.1:{head} HTTP/1.1\r\n\r\n".encode())
    plain = _ask(bridge, "proxy.sock", f"POST http://127.0.0.1:{head}/api/jobs/ HTTP/1.1\r\n\r\n".encode())
    forwarded = _ask(bridge, "proxy.sock", f"GET http://127.0.0.1:{named}/v1 HTTP/1.1\r\n\r\n".encode())
    plain_to_tls = _ask(bridge, "proxy.sock", f"GET http://127.0.0.1:{tls}/ HTTP/1.1\r\n\r\n".encode())

    assert tunnel == b"HTTP/1.1 200 Connection established\r\n\r\necho"
    assert refused.startswith(b"HTTP/1.1 403 Forbidden") and f"127.0.0.1:{head}".encode() in refused
    assert plain.startswith(b"HTTP/1.1 403 Forbidden")
    assert forwarded == b"echo"
    assert plain_to_tls.startswith(b"HTTP/1.1 403 Forbidden"), "only an endpoint named as http takes plain requests"
    assert named_taken == [b"after", f"GET http://127.0.0.1:{named}/v1 HTTP/1.1\r\n\r\n".encode()]
    assert head_taken == [] and tls_taken == []


def test_the_proxy_goes_through_the_apps_proxy_with_its_login_except_where_no_proxy_exempts(listeners, bridges):
    upstream, upstream_taken = listeners(_tunnel_ok)
    direct, direct_taken = listeners(_echo)
    env = {"HTTPS_PROXY": f"http://agent:p%40ss@127.0.0.1:{upstream}", "NO_PROXY": "127.0.0.1", "PATH": "/x"}
    bridge = bridges(Reach(remote=frozenset({("gw.example.test", 443), ("127.0.0.1", direct)})), env)

    through = _ask(bridge, "proxy.sock", b"CONNECT gw.example.test:443 HTTP/1.1\r\nHost: gw.example.test\r\n\r\n")
    exempt = _ask(bridge, "proxy.sock", f"CONNECT 127.0.0.1:{direct} HTTP/1.1\r\n\r\nx".encode())

    login = base64.b64encode(b"agent:p@ss").decode()
    assert through == b"HTTP/1.1 200 Connection established\r\n\r\n"
    assert upstream_taken == [
        b"CONNECT gw.example.test:443 HTTP/1.1\r\nHost: gw.example.test:443\r\n"
        + f"Proxy-Authorization: Basic {login}\r\n\r\n".encode()
    ]
    assert exempt == b"HTTP/1.1 200 Connection established\r\n\r\necho" and direct_taken == [b"x"]


def test_a_closed_bridge_takes_no_connection(listeners, bridges):
    port, _taken = listeners(_echo)
    bridge = bridges(Reach({port: "127.0.0.1"}))
    bridge.close()

    with pytest.raises(OSError):
        _ask(bridge, f"{port}.sock", b"late")


class _StandInProxy:
    """The app's proxy, as a keep-alive HTTP proxy parses a connection: every request head it reads is a request
    (``requests``); a ``CONNECT`` it answers with ``connect_status``, and after a 200 every later byte of that
    connection is tunnel data (``tunnelled``, one entry per connection, once it ends); a plain request gets a 200."""

    def __init__(self, connect_status: int = 200) -> None:
        self.connect_status = connect_status
        self.requests: list[bytes] = []
        self.tunnelled: list[bytes] = []
        self.ended = 0
        self.server = socket.create_server(("127.0.0.1", 0))
        self.port = self.server.getsockname()[1]
        threading.Thread(target=self._accept, daemon=True).start()

    def _accept(self) -> None:
        while True:
            try:
                conn, _ = self.server.accept()
            except OSError:
                return
            threading.Thread(target=self._serve, args=(conn,), daemon=True).start()

    def _serve(self, conn: socket.socket) -> None:
        buffered, tunnel = b"", None
        with conn:
            while chunk := conn.recv(65536):
                if tunnel is not None:
                    tunnel += chunk
                    continue
                buffered += chunk
                while tunnel is None and b"\r\n\r\n" in buffered:
                    head, buffered = buffered.split(b"\r\n\r\n", 1)
                    line = head.split(b"\r\n", 1)[0]
                    self.requests.append(line)
                    if not line.startswith(b"CONNECT "):
                        conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok")
                    elif self.connect_status == 200:
                        conn.sendall(b"HTTP/1.1 200 Connection established\r\n\r\n")
                        tunnel, buffered = buffered, b""
                    else:
                        conn.sendall(f"HTTP/1.1 {self.connect_status} No\r\nContent-Length: 0\r\n\r\n".encode())
            if tunnel is not None:
                self.tunnelled.append(tunnel)
        self.ended += 1

    def wait_ended(self, count: int) -> None:
        deadline = time.monotonic() + 10
        while self.ended < count and time.monotonic() < deadline:
            time.sleep(0.05)

    def close(self) -> None:
        self.server.close()


@pytest.fixture
def stand_in_proxies() -> Iterator[Callable[..., _StandInProxy]]:
    started: list[_StandInProxy] = []

    def start(connect_status: int = 200) -> _StandInProxy:
        started.append(_StandInProxy(connect_status))
        return started[-1]

    yield start
    for proxy in started:
        proxy.close()


#: Sent after a first request on one connection: the Ray head's Jobs API on the app's loopback, and a tunnel to a
#: host the launch never names.
_SMUGGLED = (
    b"GET http://127.0.0.1:8265/api/jobs/ HTTP/1.1\r\nHost: 127.0.0.1:8265\r\n\r\n"
    b"CONNECT other.example.test:443 HTTP/1.1\r\nHost: other.example.test:443\r\n\r\n"
)


def test_through_the_apps_proxy_a_later_request_on_one_connection_never_reaches_it_as_a_request(
    bridges, stand_in_proxies
):
    """The bridge opens its own ``CONNECT`` to the checked target and carries the specialist's bytes only inside it:
    a plain request to an https host's 443 is refused, and what follows an allowed first request reaches the named
    host through the tunnel, never the app's proxy as a request of its own."""
    upstream = stand_in_proxies()
    proxy_url = f"http://127.0.0.1:{upstream.port}"
    reach = Reach(
        remote=frozenset({("api.example.test", 443), ("gw.example.test", 8000)}),
        plain=frozenset({("gw.example.test", 8000)}),
    )
    bridge = bridges(reach, {"HTTPS_PROXY": proxy_url, "HTTP_PROXY": proxy_url})

    to_443 = _ask(bridge, "proxy.sock", b"GET http://api.example.test:443/ HTTP/1.1\r\n\r\n" + _SMUGGLED)
    tunnel = _ask(bridge, "proxy.sock", b"CONNECT api.example.test:443 HTTP/1.1\r\n\r\n" + _SMUGGLED)
    plain = _ask(
        bridge, "proxy.sock", b"GET http://gw.example.test:8000/v1?a=1 HTTP/1.1\r\nHost: gw\r\n\r\n" + _SMUGGLED
    )
    upstream.wait_ended(2)

    assert to_443.startswith(b"HTTP/1.1 403 Forbidden")
    assert tunnel == b"HTTP/1.1 200 Connection established\r\n\r\n" and plain == b""
    assert upstream.requests == [b"CONNECT api.example.test:443 HTTP/1.1", b"CONNECT gw.example.test:8000 HTTP/1.1"]
    assert sorted(upstream.tunnelled) == sorted([_SMUGGLED, b"GET /v1?a=1 HTTP/1.1\r\nHost: gw\r\n\r\n" + _SMUGGLED])


def test_a_tunnel_the_apps_proxy_refuses_ends_the_connection_and_nothing_sent_after_it_reaches_that_proxy(
    bridges, stand_in_proxies
):
    upstream = stand_in_proxies(connect_status=407)
    bridge = bridges(
        Reach(remote=frozenset({("api.example.test", 443)})), {"HTTPS_PROXY": f"127.0.0.1:{upstream.port}"}
    )

    refused = _ask(bridge, "proxy.sock", b"CONNECT api.example.test:443 HTTP/1.1\r\n\r\n" + _SMUGGLED)
    upstream.wait_ended(1)

    assert refused.startswith(b"HTTP/1.1 502 Bad Gateway")
    assert upstream.requests == [b"CONNECT api.example.test:443 HTTP/1.1"] and upstream.tunnelled == []
