"""Tiny stderr logger used by the rayjob CLIs.

Why not :mod:`logging`? The rayjob CLIs are invoked from shell wrappers
inside the Claw sandbox; the orchestrator agent reads each stderr line
verbatim to learn progress. We want zero formatter surprises (no double
prefix, no module name, no level handlers conflicting with anything
imported transitively from :mod:`inference_optimizer`). A 6-line stderr
helper is simpler and avoids the global :mod:`logging` config.
"""

from __future__ import annotations

import sys
import time

_LEVELS = ("INFO", "WARN", "ERR")


def log(level: str, msg: str) -> None:
    """Write a single timestamped line to stderr and flush immediately."""
    if level not in _LEVELS:
        level = "INFO"
    ts = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
    sys.stderr.write(f"[{ts}] {level} {msg}\n")
    sys.stderr.flush()


def info(msg: str) -> None:
    log("INFO", msg)


def warn(msg: str) -> None:
    log("WARN", msg)


def err(msg: str) -> None:
    log("ERR", msg)
