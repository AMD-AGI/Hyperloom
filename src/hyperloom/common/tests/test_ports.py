# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A long-lived server's port is below the ephemeral range and off the blocks Ray and the dev sandboxes bind."""

from __future__ import annotations

import socket
from pathlib import Path

import pytest

from hyperloom.common import ports
from hyperloom.common.ports import DEV_SANDBOX_PORTS, RAY_WORKER_PORTS, pick_server_port


def _ephemeral_from(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, start: int) -> None:
    path = tmp_path / "ip_local_port_range"
    path.write_text(f"{start}\t60999\n", encoding="utf-8")
    monkeypatch.setattr(ports, "_EPHEMERAL_RANGE", path)


def test_the_reserved_blocks_are_the_dev_sandbox_slots_and_rays_default_worker_ports() -> None:
    assert (DEV_SANDBOX_PORTS.start, DEV_SANDBOX_PORTS[-1]) == (32100, 33099)
    assert (RAY_WORKER_PORTS.start, RAY_WORKER_PORTS[-1]) == (10002, 19999)


@pytest.mark.parametrize("start", [32768, 28240])
def test_a_picked_port_is_below_the_ephemeral_range_and_off_the_reserved_blocks(monkeypatch, tmp_path, start) -> None:
    _ephemeral_from(monkeypatch, tmp_path, start)

    for _ in range(20):
        port = pick_server_port()
        assert port < start and port not in DEV_SANDBOX_PORTS and port not in RAY_WORKER_PORTS


def test_the_hosts_own_ephemeral_range_is_the_one_read() -> None:
    start = int(Path("/proc/sys/net/ipv4/ip_local_port_range").read_text(encoding="utf-8").split()[0])

    assert pick_server_port() < start


def test_a_host_that_leaves_no_port_below_its_ephemeral_range_is_an_error_not_a_fixed_port(monkeypatch, tmp_path):
    _ephemeral_from(monkeypatch, tmp_path, 10000)

    with pytest.raises(OSError, match="no server port in 10000-9999 below the ephemeral range"):
        pick_server_port()


def _listening_socket() -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", 0))
    sock.listen()
    return sock


def _no_reserved_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ports, "DEV_SANDBOX_PORTS", range(0))
    monkeypatch.setattr(ports, "RAY_WORKER_PORTS", range(0))


def test_a_port_already_bound_is_skipped(monkeypatch, tmp_path) -> None:
    _no_reserved_blocks(monkeypatch)
    with _listening_socket() as held:
        taken = held.getsockname()[1]
        monkeypatch.setattr(ports, "_LOWEST_SERVER_PORT", taken)
        _ephemeral_from(monkeypatch, tmp_path, taken + 2)

        assert pick_server_port() == taken + 1


def test_every_candidate_bound_raises(monkeypatch, tmp_path) -> None:
    _no_reserved_blocks(monkeypatch)
    with _listening_socket() as held:
        taken = held.getsockname()[1]
        monkeypatch.setattr(ports, "_LOWEST_SERVER_PORT", taken)
        _ephemeral_from(monkeypatch, tmp_path, taken + 1)

        with pytest.raises(OSError, match="could be bound"):
            pick_server_port()
