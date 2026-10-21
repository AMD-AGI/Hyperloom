# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Ave worker receiving its optimization context from Hyperloom."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import subprocess
import sys
import time

from ave.optimizer import AveLoop
from ave.task import read_run


def main(argv: list[str] | None = None) -> int:
    """Execute a caller-prepared run without a user-authored task file."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    worker = commands.add_parser("optimize", help="Read a run context as JSON and optimize its kernel")
    worker.add_argument("context", nargs="?", type=argparse.FileType("r"), default=sys.stdin)
    args = parser.parse_args(argv)
    try:
        with args.context:
            run = read_run(json.load(args.context))
        if not math.isfinite(run.deadline_unix) or run.deadline_unix <= time.time():
            raise ValueError("deadline_unix must be a finite future timestamp")
        result = asyncio.run(AveLoop(run).optimize())
    except (KeyError, TypeError, OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"ave: {error}", file=sys.stderr)
        return 1
    print(f"ave: {result['status']}; result: {run.result_json}")
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
