# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What a sandboxed specialist reaches from its own network: the endpoints its launch names, served by the bridge
over Unix sockets, and a 403 for anything else. No namespaces here; the real sandbox is in test_specialist_sandbox."""

from __future__ import annotations

import base64
import json
import socket
import threading
from pathlib import Path
from typing import Callable, Iterator

import pytest

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
        try:
            while chunk := sock.recv(65536):
                got += chunk
        except TimeoutError:
            pass
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


def test_each_providers_api_host_is_reached_where_the_launch_names_no_base_url_and_bedrock_in_its_region():
    reach = specialist_reach(["claude"], {"CLAUDE_CODE_USE_BEDROCK": "1", "AWS_REGION": "eu-west-1"}, None)

    assert dict(reach.loopback) == {}
    assert reach.remote == {
        ("api.anthropic.com", 443),
        ("api.openai.com", 443),
        ("bedrock-runtime.eu-west-1.amazonaws.com", 443),
    }


def test_a_loopback_endpoint_is_relayed_to_its_own_port_and_no_other_port_has_a_socket(listeners, bridges):
    port, taken = listeners(_echo)
    bridge = bridges(Reach({port: "127.0.0.1"}))

    assert _ask(bridge, f"{port}.sock", b"hello") == b"echo"
    assert taken == [b"hello"]
    assert sorted(path.name for path in bridge.directory.iterdir()) == sorted([f"{port}.sock", "proxy.sock"])


def test_the_proxy_tunnels_only_to_a_host_and_port_the_launch_names(listeners, bridges):
    named, named_taken = listeners(_echo)
    head, head_taken = listeners(_echo)  # a Ray head's port, say
    bridge = bridges(Reach(remote=frozenset({("127.0.0.1", named)})))

    tunnel = _ask(bridge, "proxy.sock", f"CONNECT 127.0.0.1:{named} HTTP/1.1\r\nHost: x\r\n\r\nafter".encode())
    refused = _ask(bridge, "proxy.sock", f"CONNECT 127.0.0.1:{head} HTTP/1.1\r\n\r\n".encode())
    plain = _ask(bridge, "proxy.sock", f"POST http://127.0.0.1:{head}/api/jobs/ HTTP/1.1\r\n\r\n".encode())
    forwarded = _ask(bridge, "proxy.sock", f"GET http://127.0.0.1:{named}/v1 HTTP/1.1\r\n\r\n".encode())

    assert tunnel == b"HTTP/1.1 200 Connection established\r\n\r\necho"
    assert refused.startswith(b"HTTP/1.1 403 Forbidden") and f"127.0.0.1:{head}".encode() in refused
    assert plain.startswith(b"HTTP/1.1 403 Forbidden")
    assert forwarded == b"echo"
    assert named_taken == [b"after", f"GET http://127.0.0.1:{named}/v1 HTTP/1.1\r\n\r\n".encode()]
    assert head_taken == []


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
        f"CONNECT gw.example.test:443 HTTP/1.1\r\nProxy-Authorization: Basic {login}\r\nHost: gw.example.test\r\n\r\n".encode()
    ]
    assert exempt == b"HTTP/1.1 200 Connection established\r\n\r\necho" and direct_taken == [b"x"]


def test_a_closed_bridge_takes_no_connection(listeners, bridges):
    port, _taken = listeners(_echo)
    bridge = bridges(Reach({port: "127.0.0.1"}))
    bridge.close()

    with pytest.raises(OSError):
        _ask(bridge, f"{port}.sock", b"late")
