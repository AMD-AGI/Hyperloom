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


def _strip_quotes(token: str) -> str:
    """Remove a single pair of matching surrounding quotes from *token*."""
    if len(token) >= 2 and token[0] in ('"', "'") and token[-1] == token[0]:
        return token[1:-1]
    return token


def sanitize_profile_server_args(args: str) -> str:
    """Drop server flags known to conflict with profiler/shape discovery."""
    raw = str(args or "").strip()
    if not raw:
        return ""
    tokens = shlex.split(raw, posix=False)
    out: list[str] = []
    skip_next = False
    i = 0
    while i < len(tokens):
        token = tokens[i]
        bare = _strip_quotes(token)
        if skip_next:
            skip_next = False
            i += 1
            continue
        if bare in _PROFILE_UNSAFE_BOOL_FLAGS:
            i += 1
            continue
        if bare in _PROFILE_UNSAFE_VALUE_FLAGS:
            next_i = i + 1
            if next_i < len(tokens) and not _strip_quotes(tokens[next_i]).startswith("-"):
                skip_next = True
            i += 1
            continue
        if any(bare.startswith(f"{flag}=") for flag in _PROFILE_UNSAFE_VALUE_FLAGS):
            i += 1
            continue
        out.append(token)
        i += 1
    return " ".join(out)
