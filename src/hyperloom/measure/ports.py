# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The ports of the machine that local measurement leaves alone.

A test's server listens on a port the OS found free, which on Linux can be any port from 32768 up. A branch's dev sandbox
serves on a fixed block of ports per slot, and a test's server that took one of them would break the sandbox that owns it.
"""

from __future__ import annotations

from collections.abc import Callable

__all__ = ["DEV_SANDBOX_PORTS", "pick_port_outside_dev_sandboxes"]

#: The ports reserved for dev-sandbox slots: slot ``i`` serves on ``32100 + 10 * i`` up to ``+ 9``, for up to a hundred
#: slots. The consumer is ``hyperloom_multibranch.specialist.dev_sandbox`` (the ports its leases' commands get), which
#: reads this one range so that both sides agree; nothing that measures may listen on a port in it.
DEV_SANDBOX_PORTS = range(32100, 33100)

#: How many ports are asked for before giving up; a source that offers a reserved port this often is not random.
_ATTEMPTS = 100


def pick_port_outside_dev_sandboxes(candidates: Callable[[], int]) -> int:
    """A port ``candidates`` offers that is not one of :data:`DEV_SANDBOX_PORTS`.

    ``candidates`` offers a port each time it is called (the caller's way of finding a free one); an offer in the
    reserved range is dropped and another asked for.

    Raises:
        OSError: When every one of the offers was a reserved port.
    """

    for _ in range(_ATTEMPTS):
        port = candidates()
        if port not in DEV_SANDBOX_PORTS:
            return port
    raise OSError(f"no port outside the dev sandboxes' {DEV_SANDBOX_PORTS.start}-{DEV_SANDBOX_PORTS.stop - 1}")
