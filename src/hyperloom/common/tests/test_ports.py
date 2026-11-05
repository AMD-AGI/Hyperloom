# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""``pick_server_port`` stays out of the ephemeral range and skips ports already bound."""

from __future__ import annotations

import socket

import pytest

from hyperloom.common import ports


def _set_ephemeral_start(monkeypatch: pytest.MonkeyPatch, tmp_path, start: int) -> None:
    path = tmp_path / "ip_local_port_range"
    path.write_text(f"{start}\t60999\n")
    monkeypatch.setattr(ports, "_EPHEMERAL_RANGE_PATH", path)


def _listening_socket() -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", 0))
    sock.listen()
    return sock


def test_port_is_below_the_ephemeral_range_and_bindable(monkeypatch, tmp_path):
    _set_ephemeral_start(monkeypatch, tmp_path, 32768)

    port = ports.pick_server_port()

    assert ports._LOWEST_SERVER_PORT <= port < 32768
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("", port))


def test_a_port_already_bound_is_skipped(monkeypatch, tmp_path):
    with _listening_socket() as held:
        taken = held.getsockname()[1]
        monkeypatch.setattr(ports, "_LOWEST_SERVER_PORT", taken)
        _set_ephemeral_start(monkeypatch, tmp_path, taken + 2)

        assert ports.pick_server_port() == taken + 1


def test_every_candidate_bound_raises(monkeypatch, tmp_path):
    with _listening_socket() as held:
        taken = held.getsockname()[1]
        monkeypatch.setattr(ports, "_LOWEST_SERVER_PORT", taken)
        _set_ephemeral_start(monkeypatch, tmp_path, taken + 1)

        with pytest.raises(OSError, match="no free server port"):
            ports.pick_server_port()
