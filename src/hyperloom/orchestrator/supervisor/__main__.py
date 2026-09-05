# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Run the out-of-band supervisor as its own process.

Neither failure it catches -- a coordinator that died, a coordinator that
stopped running its loop -- is observable from inside that process.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from hyperloom.orchestrator.supervisor.watch import (
    DEFAULT_POLL_SEC,
    DEFAULT_TICK_STALL_SEC,
    Supervisor,
)


def _parse(argv: list[str] | None) -> argparse.Namespace:
    """Parse the supervisor's arguments; ``None`` reads ``sys.argv``."""
    parser = argparse.ArgumentParser(prog="hyperloom-supervisor", description=__doc__)
    parser.add_argument("--session-dir", required=True, help="the session to watch")
    parser.add_argument("--tick-stall-sec", type=float, default=DEFAULT_TICK_STALL_SEC)
    parser.add_argument("--poll-sec", type=float, default=DEFAULT_POLL_SEC)
    parser.add_argument("--max-polls", type=int, default=0, help="stop after N readings; 0 runs until the session ends")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Watch one session until its coordinator is gone.

    Args:
        argv: Argument list; ``None`` reads ``sys.argv``.

    Returns:
        int: ``0`` once the watch ends.
    """
    args = _parse(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    supervisor = Supervisor(
        Path(args.session_dir),
        tick_stall_sec=args.tick_stall_sec,
        poll_sec=args.poll_sec,
    )
    logging.info(
        "SUPERVISOR: watching %s (stall=%.0fs)",
        args.session_dir,
        supervisor.tick_stall_sec,
    )
    asyncio.run(supervisor.run(max_polls=args.max_polls))
    return 0


if __name__ == "__main__":
    sys.exit(main())
