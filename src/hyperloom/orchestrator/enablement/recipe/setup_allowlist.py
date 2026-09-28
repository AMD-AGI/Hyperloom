# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Which recorded enablement setup commands may be replayed, and how a refused one is stored.

The one owner of the install-only allowlist: the integrate replay decides what may run by it, and credential
classification normalises a command the same way before naming its installer.
"""

from __future__ import annotations

import re

from hyperloom.common.env_safety import redact_secret_values

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
    r"npm\s+install\s+-g\b",
    r"pnpm\s+(?:install|add)\b",
    r"yarn\s+(?:add|install)\b",
    r"conda\s+install\b",
    r"mamba\s+install\b",
)
#: Directory prefixes whose basename may stand in for the whole path when the
#: allowlist is matched. Absolute and system-owned on purpose: the replay runs
#: the ORIGINAL command string, so anything a specialist can write to -- a
#: relative ``./pip``, a path under its own workspace -- must not be able to
#: borrow an allowlisted name. ``/opt/venv`` is the canonical ROCm stack this
#: repository installs into; the rest are the standard system bindirs.
#: ``..`` is excluded from the segment class on purpose. With a plain
#: ``[A-Za-z0-9._-]+`` the traversal form ``/usr/bin/../../tmp/x/pip install foo``
#: matches, normalises to an allowlisted ``pip install foo``, and then
#: ``run_setup_commands`` executes the ORIGINAL string -- running /tmp/x/pip,
#: which is exactly the workspace-owned binary the prefix list exists to keep out.
TRUSTED_BIN_PREFIX_RE = re.compile(
    r"^(?:/opt/(?!\.\.?/)[A-Za-z0-9._-]+|/usr(?:/local)?|/bin|/sbin)"
    r"(?:/(?!\.\.?(?:/|$))[A-Za-z0-9._-]+)*/"
)

#: Per-command clip in the rejection summary. Long enough to recognise the
#: command, short enough that twelve of them cannot bury the round's own reason.
SKIPPED_CMD_CHARS = 160


def sanitize_setup_command(cmd: str) -> str:
    """A rejected command in the form it is safe to store and hand back.

    Rejected commands are LLM-written text. They reach the journal, the report
    and the KB, and are read back into the next round's mandate, so a bearer
    token or a credentialed URL in one would outlive the round that produced it.
    Clipped as well, so a single rejected install naming a hundred packages
    cannot crowd out the reason it is reported alongside.
    """
    text = redact_secret_values(str(cmd).strip())
    return text if len(text) <= SKIPPED_CMD_CHARS else text[:SKIPPED_CMD_CHARS] + "..."


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
    # Match on the executable's basename, but ONLY for an absolute path under a
    # system prefix. The patterns below are anchored, so without any
    # normalisation ``/opt/venv/bin/uv pip install X`` was REJECTED while
    # ``uv pip install X`` -- the same operation -- was allowed. Measured: two
    # sessions hit one missing dependency and got opposite outcomes, decided by
    # nothing but how the specialist happened to spell the path.
    #
    # The allowlist is checked against this normalised text, but
    # ``run_setup_commands`` executes the ORIGINAL string under ``shell=True``.
    # So a blanket basename strip would let any binary in: ``./pip install foo``
    # normalises to an allowlisted ``pip install foo`` while running a script
    # the specialist just wrote into its own workspace. Restricting the strip to
    # absolute system prefixes keeps "which KIND of operation may replay" intact
    # -- the property the SETUP_CMD_ALLOWLIST comment promises -- while still treating a venv's own
    # interpreter as the interpreter it is.
    text = TRUSTED_BIN_PREFIX_RE.sub("", text, count=1)
    return any(re.match(pat, text) for pat in SETUP_CMD_ALLOWLIST)
