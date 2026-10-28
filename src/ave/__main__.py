# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Ave worker receiving its optimization context from Hyperloom."""

from __future__ import annotations

import argparse
import asyncio
import subprocess
import sys

from pathlib import Path

from pydantic import TypeAdapter

from ave.optimizer import AveLoop
from ave.task import OptimizationRun


def main(argv: list[str] | None = None) -> int:
    """Execute a caller-prepared run without a user-authored task file."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    worker = commands.add_parser("optimize", help="Read a run context as JSON and optimize its kernel")
    worker.add_argument("context", type=Path, help="Prepared Hyperloom worker context")
    args = parser.parse_args(argv)
    try:
        run = TypeAdapter(OptimizationRun).validate_json(args.context.read_text())
        result = asyncio.run(AveLoop(run).optimize())
    except (KeyError, TypeError, OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"ave: {error}", file=sys.stderr)
        return 1
    print(f"ave: {result['status']}; result: {run.result_json}")
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
