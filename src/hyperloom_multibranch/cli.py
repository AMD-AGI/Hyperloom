# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``hyperloom-multibranch``: command-line entry for a multi-branch session."""

from __future__ import annotations

import argparse
import sys

PROG = "hyperloom-multibranch"


def main(argv: list[str] | None = None) -> int:
    """Parse the command line and run a multi-branch session."""
    parser = argparse.ArgumentParser(prog=PROG, description="Run a multi-branch optimization session.")
    parser.parse_args(argv)
    print(f"{PROG}: the session runner is not implemented yet", file=sys.stderr)
    return 1
