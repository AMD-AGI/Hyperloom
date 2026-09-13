# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tiny stderr logger for the rayjob CLIs."""

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
    """Emit an ``INFO``-level line to stderr."""
    log("INFO", msg)


def warn(msg: str) -> None:
    """Emit a ``WARN``-level line to stderr."""
    log("WARN", msg)


def err(msg: str) -> None:
    """Emit an ``ERR``-level line to stderr."""
    log("ERR", msg)
