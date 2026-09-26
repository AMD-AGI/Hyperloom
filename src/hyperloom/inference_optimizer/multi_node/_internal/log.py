# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tiny stderr logger for the rayjob CLIs."""

from __future__ import annotations

import sys
import time


def _emit(level: str, msg: str) -> None:
    """Write a single timestamped line to stderr and flush immediately."""
    ts = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
    sys.stderr.write(f"[{ts}] {level} {msg}\n")
    sys.stderr.flush()


def info(msg: str) -> None:
    """Emit an ``INFO``-level line to stderr."""
    _emit("INFO", msg)


def warn(msg: str) -> None:
    """Emit a ``WARN``-level line to stderr."""
    _emit("WARN", msg)


def err(msg: str) -> None:
    """Emit an ``ERR``-level line to stderr."""
    _emit("ERR", msg)
