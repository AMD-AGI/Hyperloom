# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``hyperloom conc-sweep SESSION_DIR``: the concurrency sweep of a finished session, run apart from its optimization.

The sweep holds the session's lock, so it refuses a session another process is running, and SIGINT or SIGTERM cancel
it: the points already measured stay in the report, which then reads ``cancelled``.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from hyperloom.inference_optimizer.cli.bootstrap import hosted_grant_end_unix
from hyperloom.inference_optimizer.session.lock import SessionAlreadyRunning, SessionLock
from hyperloom.orchestrator.actions.cancel_channel import CancelScope, use_cancel_scope

from .report import SETTLED
from .runner import run_conc_sweep


@contextmanager
def _cancelled_by_signals(scope: CancelScope) -> Iterator[None]:
    """Turn SIGINT and SIGTERM into a cancel of ``scope`` for the length of the block."""

    def on_signal(signum: int, _frame: object) -> None:
        scope.cancel(reason=signal.Signals(signum).name)

    previous = {signum: signal.signal(signum, on_signal) for signum in (signal.SIGINT, signal.SIGTERM)}
    try:
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler if handler is not None else signal.SIG_DFL)


async def sweep_session(session_dir: Path) -> dict[str, Any]:
    """Run the sweep of the session in ``session_dir`` under its lock, warned at a hosted run's grant end; raises
    ``SessionAlreadyRunning``."""
    scope = CancelScope()
    with SessionLock(session_dir), use_cancel_scope(scope), _cancelled_by_signals(scope):
        return await run_conc_sweep(session_dir, grant_end_unix=hosted_grant_end_unix())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="hyperloom conc-sweep",
        description="Sweep a finished session's baseline and optimized launch over its recipe's concurrencies.",
    )
    parser.add_argument("session_dir", type=Path, help="The session directory of a finished optimize run.")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        payload = asyncio.run(sweep_session(args.session_dir))
    except SessionAlreadyRunning as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"conc-sweep: {payload['status']} {payload.get('skip_reason', '')}".rstrip())
    return 0 if payload["status"] in SETTLED else 1


__all__ = ["main", "sweep_session"]
