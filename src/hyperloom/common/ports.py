# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Listening ports for servers that bind long after the port is chosen.

A port from ``bind(("", 0))`` is in the ephemeral range every outbound ``connect`` draws from, so a server that
binds it minutes later can find it taken. Ports below that range stay free unless something binds them explicitly.
"""

from __future__ import annotations

import random
import socket
from pathlib import Path

_EPHEMERAL_RANGE_PATH = Path("/proc/sys/net/ipv4/ip_local_port_range")
# Lowest port offered; keeps clear of the well-known and commonly configured service ports below it.
_LOWEST_SERVER_PORT = 10000
_BIND_ATTEMPTS = 64

__all__ = ["pick_server_port"]


def _ephemeral_range_start() -> int:
    """First port of the kernel's ephemeral range."""
    return int(_EPHEMERAL_RANGE_PATH.read_text().split()[0])


def pick_server_port() -> int:
    """Return a free TCP port below the ephemeral range, for a server that binds it later.

    Raises:
        OSError: no candidate port exists below the ephemeral range, or none of
            the sampled candidates could be bound.
    """
    candidates = range(_LOWEST_SERVER_PORT, _ephemeral_range_start())
    for port in random.sample(candidates, min(_BIND_ATTEMPTS, len(candidates))):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("", port))
            except OSError:
                continue
        return port
    raise OSError(f"no free server port among {_BIND_ATTEMPTS} sampled below {candidates.stop}")
