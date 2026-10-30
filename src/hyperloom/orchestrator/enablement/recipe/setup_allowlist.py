# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The install-only allowlist: which recorded enablement setup commands the integrate replay may run."""

from __future__ import annotations

import re
from pathlib import PurePosixPath

# Enablement environment-setup replay: allowlist of install-only command shapes.
# A specialist may run arbitrary Bash in its own sandboxed session, but the
# durable *replay* performed here (before applying patches + booting) is limited
# to package/tool installation so a recorded ``setup_commands`` list can never be
# a vector for arbitrary side effects (rm, curl|bash, service restarts, etc.).
# Matched against the command with leading `sudo `/env-assignments stripped.
SETUP_CMD_ALLOWLIST: tuple[str, ...] = (
    r"pip3?\s+install\b",
    r"(?:python3?|uv)\s+-m\s+pip\s+install\b",
    r"uv\s+pip\s+install\b",
    r"pip3?\s+uninstall\s+-y\b",
    # Creating an isolated environment to install INTO. Without these the only
    # spelling that survived the allowlist was installing into the system
    # interpreter (``PIP_BREAK_SYSTEM_PACKAGES=1 pip install``), so the gate was
    # steering repairs toward the less safe of the two options it had to choose
    # between. Creating a venv directory is bounded; breaking the system's
    # package manager is not.
    r"uv\s+venv\b",
    r"(?:python3?|uv)\s+-m\s+venv\b",
    r"apt(?:-get)?\s+(?:install|update)\b",
    r"npm\s+(?:install|i|ci)\b",
    r"pnpm\s+(?:install|add)\b",
    r"yarn\s+(?:add|install)\b",
    r"conda\s+install\b",
    r"mamba\s+install\b",
)


def _installer_invocation(text: str) -> str | None:
    """``text`` with the executable's directory dropped, or ``None`` for a relative one.

    The allowlist decides which KIND of operation may replay, and that is what
    the basename names. A relative path is refused because the replay runs the
    original string from the session directory, where it resolves to a
    different file than where the specialist authored it -- or to none.
    """
    head, *rest = text.split(None, 1)
    if "/" not in head:
        return text
    if not head.startswith("/"):
        return None
    return " ".join([PurePosixPath(head).name, *rest])


def is_allowlisted_setup_command(cmd: str) -> bool:
    """True when ``cmd`` is an install-only command safe to replay.

    Strips a leading ``sudo``, any ``KEY=VALUE`` env-assignment prefixes and the
    executable's directory, then requires the remainder to start with a known
    package/tool installer. Rejects anything with shell control operators that
    could chain an arbitrary payload.
    """
    text = (cmd or "").strip()
    if not text:
        return False
    # Reject command substitution / backticks / newlines outright — these can
    # smuggle an arbitrary payload regardless of tokenization.
    if re.search(r"[`\n]|\$\(", text):
        return False
    # Guard against genuine shell chaining/redirection while allowing pip/pkg
    # version specifiers that legitimately contain ``>``/``<`` (e.g.
    # ``transformers>=4.58``). Neutralise the safe, non-shell uses first, then
    # reject any leftover metacharacter (the replay runs under ``shell=True``).
    scrubbed = text
    # Drop quoted segments (their contents cannot act as shell operators).
    scrubbed = re.sub(r"'[^']*'", " ", scrubbed)
    scrubbed = re.sub(r'"[^"]*"', " ", scrubbed)
    # Drop an unquoted pip-style version comparison only when it is attached to
    # the package token and the version starts with a digit (``pkg>=4.58``).
    # Whitespace-prefixed operators and non-version targets remain visible to
    # the metacharacter check below (``foo >evil``, ``2>evil``, ``foo <evil``).
    scrubbed = re.sub(r"(?<=[0-9A-Za-z_.\]])(?:>=|<=|>|<)(?=\d)", " ", scrubbed)
    # Any remaining shell chaining/redirection metacharacter => unsafe.
    if re.search(r"[;&|<>]", scrubbed):
        return False
    # Strip a leading sudo and leading KEY=VALUE env assignments.
    text = re.sub(r"^\s*sudo\s+", "", text)
    text = re.sub(r"^(?:\s*[A-Za-z_][A-Za-z0-9_]*=[^\s]*\s+)+", "", text)
    invocation = _installer_invocation(text)
    if invocation is None:
        return False
    return any(re.match(pat, invocation) for pat in SETUP_CMD_ALLOWLIST)
