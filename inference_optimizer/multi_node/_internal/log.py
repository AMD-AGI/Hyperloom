# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Tiny stderr logger for the rayjob CLIs.

Avoids :mod:`logging` so the agent reads each stderr line verbatim with no
formatter surprises or global-config conflicts.
"""

from __future__ import annotations

import sys
import time

_LEVELS = ("INFO", "WARN", "ERR")


def log(level: str, msg: str) -> None:
    """Write a single timestamped line to stderr and flush immediately.

    Args:
        level (str): Severity label; one of ``INFO``, ``WARN``, or ``ERR``.
            Any unrecognized value is coerced to ``INFO``.
        msg (str): The message text to emit.
    """
    if level not in _LEVELS:
        level = "INFO"
    ts = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
    sys.stderr.write(f"[{ts}] {level} {msg}\n")
    sys.stderr.flush()


def info(msg: str) -> None:
    """Emit an ``INFO``-level line to stderr.

    Args:
        msg (str): The message text to emit.
    """
    log("INFO", msg)


def warn(msg: str) -> None:
    """Emit a ``WARN``-level line to stderr.

    Args:
        msg (str): The message text to emit.
    """
    log("WARN", msg)


def err(msg: str) -> None:
    """Emit an ``ERR``-level line to stderr.

    Args:
        msg (str): The message text to emit.
    """
    log("ERR", msg)
