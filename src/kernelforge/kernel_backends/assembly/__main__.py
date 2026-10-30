# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Command-line entry point for building AMDGPU assembly candidates."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .compiler import AssemblyError, assemble


def main(argv: list[str] | None = None) -> int:
    """Assemble a candidate, returning a nonzero status on validation/build failure."""
    parser = argparse.ArgumentParser(description="Build complete AMDHSA assembly into a loadable HSACO")
    commands = parser.add_subparsers(dest="command", required=True)
    assembly = commands.add_parser("assemble", help="Assemble and link compiler-generated AMDHSA source")
    assembly.add_argument("--source", type=Path, required=True)
    assembly.add_argument("--output", type=Path, required=True)
    assembly.add_argument("--gpu-target", required=True, help="Target ID, e.g. gfx950 or gfx90a:sramecc+:xnack-")
    assembly.add_argument("--toolchain-dir", type=Path, required=True, help="ROCm LLVM bin directory")
    assembly.add_argument("--timeout-sec", type=float, default=60)
    args = parser.parse_args(argv)
    try:
        output = assemble(
            args.source,
            args.output,
            gpu_target=args.gpu_target,
            toolchain_dir=args.toolchain_dir,
            timeout_sec=args.timeout_sec,
        )
    except (AssemblyError, OSError) as exc:
        print(f"Assembly failed: {exc}", file=sys.stderr)
        return 1
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
