# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""RTK (Rust Token Killer) integration — trim CLI output before it reaches the context.

RTK is a CLI proxy that replaces a command's output with a condensed form of
it. What that is worth depends entirely on the command, because RTK carries a
per-command filter and passes anything it has no filter for straight through.
Measured against rtk 0.48 on this repository:

  - ``find`` 93%, ``git status`` 78%, ``grep`` 28%, ``git diff`` 20%
  - ``ninja`` / ``cmake`` 0% — upstream ships no filter for either, so a build
    command handed to :func:`smart_wrap` is passed through unchanged

The two subcommands that do not care what the inner command is cover the rest:

  - ``rtk err <cmd>`` keeps only errors and warnings — 99% on a noisy build,
    and the build is where the 0% above would otherwise leave us. See
    :func:`err_wrap`.
  - ``rtk test <cmd>`` keeps only failures — 86% on a real pytest run.

Both preserve the inner command's exit status, emit the trimmed lines on the
stream they came from, and tee the full output to a log file whose path they
print, so nothing is discarded — it just stops being paid for by default.

The rest of the saving is in the shell commands the *agent* issues, which is
why :func:`is_available` also gates the system-prompt paragraph asking for
them (see ``orchestrator/agent.py``).

Install: ``kernelforge install-rtk`` (see :mod:`kernelforge.rtk_install`)
Verify:  ``rtk --version && rtk gain``

Usage in kernelforge is transparent — if rtk is on PATH, it's used
automatically. If not, commands run directly with no RTK overhead.
"""

from __future__ import annotations

import shlex
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
    grows it is tool output: the median session writes ~133k tokens of context
    and re-reads that prefix ~14 times. RTK is the one mechanism this repo has
    for filtering that output before it lands there, so a deployment that
    quietly lacks it pays for every unfiltered listing and test run, twice --
    once to write the cache and again on every later turn that reads it.

    Returning a string rather than logging keeps the decision of where to
    say it (campaign banner, run summary) with the caller.
    """
    if _RTK_PATH is not None:
        return ""
    return (
        "rtk is not on PATH: verbose tool output (find, grep, git, test runs) "
        "enters the agent's context unfiltered, which is the single largest "
        "avoidable driver of LLM spend. Install it with `kernelforge install-rtk`"
    )


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


def err_wrap(cmd: Sequence[str]) -> list[str]:
    """Wrap a command so only its errors and warnings survive.

    For a build. ``rtk`` has no ``ninja``/``cmake`` filter, so :func:`wrap_command`
    leaves a build exactly as noisy as it was; ``rtk err`` is command-agnostic and
    keeps the diagnostic lines out of several thousand lines of progress. It
    preserves the inner command's exit status, so a caller still branches on
    ``returncode``, and tees the untrimmed output to a file whose path it prints.

    Each argument is shell-quoted on the way in. ``rtk err`` joins what it is
    given into one line and hands that to ``sh``, so an argument carrying a space
    or a metacharacter would otherwise be re-split into a different command --
    ``ls "/a b"`` becomes two failed lookups, and ``sh -c "for …; do …; done"``
    becomes a syntax error. Quoting each argument first survives that round trip
    and is a no-op for the ordinary ``["ninja", "-j4"]``.

    Examples:
        err_wrap(["ninja", "-j4"])   → ["rtk", "err", "ninja", "-j4"]

    If RTK is not installed:
        err_wrap(["ninja", "-j4"])   → ["ninja", "-j4"]
    """
    if _RTK_PATH is None or not cmd:
        return list(cmd)
    return [_RTK_PATH, "err", *(shlex.quote(str(arg)) for arg in cmd)]


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
