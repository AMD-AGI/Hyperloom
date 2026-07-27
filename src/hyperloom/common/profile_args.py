# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared normalization for server arguments used by profile runs."""

from __future__ import annotations

import shlex

_PROFILE_UNSAFE_BOOL_FLAGS = frozenset(
    {
        "--enable-torch-compile",
    }
)
_PROFILE_UNSAFE_VALUE_FLAGS = frozenset(
    {
        "--torch-compile-max-bs",
    }
)


def sanitize_profile_server_args(args: str) -> str:
    """Drop server flags known to conflict with profiler/shape discovery."""
    raw = str(args or "").strip()
    if not raw:
        return ""
    try:
        tokens = shlex.split(raw, posix=False)
    except ValueError:
        tokens = raw.split()
    out: list[str] = []
    skip_next = False
    for token in tokens:
        if skip_next:
            skip_next = False
            continue
        if token in _PROFILE_UNSAFE_BOOL_FLAGS:
            continue
        if token in _PROFILE_UNSAFE_VALUE_FLAGS:
            skip_next = True
            continue
        if any(token.startswith(f"{flag}=") for flag in _PROFILE_UNSAFE_VALUE_FLAGS):
            continue
        out.append(token)
    return " ".join(out)
