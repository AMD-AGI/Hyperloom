# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``python -m hyperloom.measure.runner run --in /in --out /out``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from hyperloom.measure.runner.run import run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hyperloom.measure.runner")
    sub = parser.add_subparsers(dest="command", required=True)
    cmd = sub.add_parser("run", help="run one test from an instruction sheet")
    cmd.add_argument("--in", dest="in_dir", type=Path, default=Path("/in"))
    cmd.add_argument("--out", dest="out_dir", type=Path, default=Path("/out"))
    cmd.add_argument("--scratch", type=Path, default=Path("/scratch"))
    cmd.add_argument("--spec-wait-sec", type=float, default=600.0)
    args = parser.parse_args(argv)
    return run(args.in_dir, args.out_dir, scratch=args.scratch, spec_wait_sec=args.spec_wait_sec)


if __name__ == "__main__":
    sys.exit(main())
