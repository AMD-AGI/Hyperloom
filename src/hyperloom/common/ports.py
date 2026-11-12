# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The local ports a long-lived server may listen on, and the ones it leaves alone.

A port from ``bind(("", 0))`` is in the ephemeral range every outbound ``connect`` draws from, so a server that binds
it only after minutes of loading weights can find it taken. Ports below that range stay free unless something binds
them explicitly, as Ray's workers and the dev sandboxes do on their own fixed blocks.
"""

from __future__ import annotations

import random
import socket
import subprocess
from pathlib import Path

from hyperloom_common.platform import is_windows

__all__ = ["DEV_SANDBOX_PORTS", "RAY_WORKER_PORTS", "pick_server_port"]

#: The ports reserved for dev-sandbox slots: slot ``i`` serves on ``32100 + 10 * i`` up to ``+ 9``, for up to a hundred
#: slots; nothing that measures may listen on a port in it.
DEV_SANDBOX_PORTS = range(32100, 33100)

#: The ports Ray workers bind: ``ray start``'s ``--min-worker-port``/``--max-worker-port`` defaults (10002-19999 in
#: Ray 2.44), which the head this session starts leaves unset.
RAY_WORKER_PORTS = range(10002, 20000)

_EPHEMERAL_RANGE = Path("/proc/sys/net/ipv4/ip_local_port_range")
#: Lowest port offered; keeps clear of the well-known and commonly configured service ports below it.
_LOWEST_SERVER_PORT = 10000
_BIND_ATTEMPTS = 64
#: Windows' own default dynamic range, used when ``netsh`` cannot be read.
_WINDOWS_EPHEMERAL_START = 49152


def _ephemeral_start() -> int:
    """The first port of this host's ephemeral range, whichever host this is."""
    if not is_windows():
        return int(_EPHEMERAL_RANGE.read_text(encoding="utf-8").split()[0])
    try:
        completed = subprocess.run(
            ["netsh", "int", "ipv4", "show", "dynamicport", "tcp"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        for line in completed.stdout.splitlines():
            if "Start Port" in line:
                return int(line.split(":", 1)[1].strip())
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        pass
    return _WINDOWS_EPHEMERAL_START


def pick_server_port() -> int:
    """A free TCP port below the ephemeral range and outside the dev-sandbox and Ray worker blocks.

    Raises:
        OSError: When the host leaves no such port, or none of the sampled ones can be bound.
    """
    ephemeral_start = _ephemeral_start()
    candidates = [
        port
        for port in range(_LOWEST_SERVER_PORT, ephemeral_start)
        if port not in DEV_SANDBOX_PORTS and port not in RAY_WORKER_PORTS
    ]
    if not candidates:
        raise OSError(
            f"no server port in {_LOWEST_SERVER_PORT}-{ephemeral_start - 1} below the ephemeral range "
            f"(starting {ephemeral_start}) outside the dev-sandbox {DEV_SANDBOX_PORTS.start}-"
            f"{DEV_SANDBOX_PORTS.stop - 1} and Ray worker {RAY_WORKER_PORTS.start}-{RAY_WORKER_PORTS.stop - 1} ports"
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
