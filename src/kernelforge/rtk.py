# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""RTK (Rust Token Killer) integration — 60-90% token savings on CLI output."""

from __future__ import annotations

import shutil
from typing import Sequence

# Cache the RTK binary path at import time
_RTK_PATH: str | None = shutil.which("rtk")


def is_available() -> bool:
    """Check if RTK is installed and on PATH."""
    return _RTK_PATH is not None


def unavailable_warning() -> str:
    """Return the one-line warning to surface when RTK is missing, else "".

    RTK degrades silently by design -- a missing binary must never break a
    campaign. But silent and invisible are different things. Measured across
    316 end-to-end Forge runs, roughly three quarters of the LLM bill is
    proportional to how large each agent session's context grows, and what
    grows it is tool output: the average session carries well over 100k
    tokens of context by the time it ends. RTK is the one mechanism that
    filters that output before it lands there, so a deployment that quietly
    lacks it pays for every unfiltered ninja and git dump, twice -- once to
    write the cache and again on every later turn that reads it.

    Returning a string rather than logging keeps the decision of where to
    say it (campaign banner, run summary) with the caller.
    """
    if _RTK_PATH is not None:
        return ""
    return (
        "rtk is not on PATH: verbose tool output (ninja, git, general commands) "
        "enters the agent's context unfiltered, which is the single largest "
        "avoidable driver of LLM spend. Install from https://github.com/rtk-ai/rtk"
    )


def wrap_command(cmd: Sequence[str]) -> list[str]:
    """Wrap a command with RTK if available."""
    if _RTK_PATH is None:
        return list(cmd)
    return [_RTK_PATH, *cmd]


# Commands that should NOT go through RTK (we parse their raw output)
_RTK_SKIP_COMMANDS = {
    "rocprofv3",  # We parse the CSV output directly
    "llvm-objdump",  # We parse register info from disassembly
    "readelf",  # We parse ELF notes
}


def smart_wrap(cmd: Sequence[str]) -> list[str]:
    """Intelligently decide whether to wrap with RTK."""
    if not cmd:
        return list(cmd)

    # Get the base command name (without path)
    base_cmd = cmd[0].rsplit("/", 1)[-1] if "/" in cmd[0] else cmd[0]

    if base_cmd in _RTK_SKIP_COMMANDS:
        return list(cmd)

    return wrap_command(cmd)
