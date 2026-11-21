# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The local ports a long-lived server may listen on, and the ones it leaves alone.

A port from ``bind(("", 0))`` is in the ephemeral range every outbound ``connect`` draws from, so a server that binds
it only after minutes of loading weights can find it taken. Ports below that range stay free unless something binds
them explicitly, as the dev sandboxes do on their own fixed block.
"""

from __future__ import annotations

import random
import socket
from pathlib import Path

__all__ = ["DEV_SANDBOX_PORTS", "pick_server_port"]

#: The ports reserved for dev-sandbox slots: slot ``i`` serves on ``32100 + 10 * i`` up to ``+ 9``, for up to a hundred
#: slots; nothing that measures may listen on a port in it.
DEV_SANDBOX_PORTS = range(32100, 33100)

_EPHEMERAL_RANGE = Path("/proc/sys/net/ipv4/ip_local_port_range")
#: Lowest port offered; keeps clear of the well-known and commonly configured service ports below it.
_LOWEST_SERVER_PORT = 10000
_BIND_ATTEMPTS = 64


def pick_server_port() -> int:
    """A free TCP port below the ephemeral range and outside the dev-sandbox block.

    Raises:
        OSError: When the host leaves no such port, or none of the sampled ones can be bound.
    """
    ephemeral_start = int(_EPHEMERAL_RANGE.read_text(encoding="utf-8").split()[0])
    candidates = [port for port in range(_LOWEST_SERVER_PORT, ephemeral_start) if port not in DEV_SANDBOX_PORTS]
    if not candidates:
        raise OSError(
            f"no server port in {_LOWEST_SERVER_PORT}-{ephemeral_start - 1} below the ephemeral range "
            f"({_EPHEMERAL_RANGE}) outside the dev-sandbox ports {DEV_SANDBOX_PORTS.start}-{DEV_SANDBOX_PORTS.stop - 1}"
        )
    sampled = random.sample(candidates, min(_BIND_ATTEMPTS, len(candidates)))
    for port in sampled:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("", port))
            except OSError:
                continue
        return port
    raise OSError(f"none of {len(sampled)} sampled server ports below {ephemeral_start} could be bound")
