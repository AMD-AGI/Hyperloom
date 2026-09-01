# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""RTK (Rust Token Killer) integration — 60-90% token savings on CLI output.

RTK is a CLI proxy that filters verbose command output to essential information.
When available, the loop routes its build command through RTK for token
savings on:

  - ninja/cmake build output (80-90% reduction)
  - git operations (59-80% reduction)
  - rocprofv3 output (est. 70% reduction via custom filter)
  - general command output (smart summarization)

Install: https://github.com/rtk-ai/rtk
Verify:  rtk --version && rtk gain

Usage in kernelforge is transparent — if rtk is on PATH, it's used
automatically. If not, commands run directly with no RTK overhead.
"""

from __future__ import annotations

import shutil
from typing import Sequence

# Cache the RTK binary path at import time
_RTK_PATH: str | None = shutil.which("rtk")


def is_available() -> bool:
    """Check if RTK is installed and on PATH."""
    return _RTK_PATH is not None


def wrap_command(cmd: Sequence[str]) -> list[str]:
    """Wrap a command with RTK if available.

    RTK automatically detects the command type and applies the appropriate
    filter. If RTK is not available, returns the command unchanged.

    Examples:
        wrap_command(["ninja", "-j4"])     → ["rtk", "ninja", "-j4"]
        wrap_command(["git", "status"])    → ["rtk", "git", "status"]
        wrap_command(["rocprofv3", ...])   → ["rtk", "rocprofv3", ...]

    If RTK is not installed:
        wrap_command(["ninja", "-j4"])     → ["ninja", "-j4"]
    """
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
    """Intelligently decide whether to wrap with RTK.

    Skips RTK for commands whose raw output we parse programmatically.
    Uses RTK for everything else (build output, git, general commands).
    """
    if not cmd:
        return list(cmd)

    # Get the base command name (without path)
    base_cmd = cmd[0].rsplit("/", 1)[-1] if "/" in cmd[0] else cmd[0]

    if base_cmd in _RTK_SKIP_COMMANDS:
        return list(cmd)

    return wrap_command(cmd)
