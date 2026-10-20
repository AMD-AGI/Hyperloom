# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""`hyperloom sikl` — optimize the kernel of one SIKL schema with KernelForge."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from hyperloom.sikl import forge, handoff, report, workspace
from hyperloom.sikl.schema import SchemaError, find, load

DEFAULT_GIT_BRANCH = "forge-optimize"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hyperloom sikl",
        description="Optimize the kernel of one SIKL schema with KernelForge and score it with `sikl validate`.",
    )
    parser.add_argument("--sikl-root", required=True, type=Path, help="SIKL checkout whose `sikl validate` scores")
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--schema", help="Name of a kernel-less schema in --sikl-root")
    which.add_argument("--schema-file", type=Path, help="A schema file, read with --sikl-root's contracts")
    parser.add_argument("--run-dir", type=Path, help="Run directory (default: ./sikl-runs/<schema>-<stamp>)")
    parser.add_argument(
        "--kernel-backend",
        default="flydsl",
        choices=sorted(workspace.KERNEL_DECORATORS),
        help="Language every kernel a candidate launches is written in (default: flydsl)",
    )
    parser.add_argument(
        "--starting-kernel",
        type=Path,
        help="kernel.py to start from (default: a torch implementation, for gemm-contract schemas only)",
    )
    parser.add_argument("--gpu-target", default="gfx950", help="GPU arch passed to forge-loop")
    parser.add_argument("--max-hours", type=float, default=4.0, help="forge-loop wall-clock budget")
    parser.add_argument("--model", default="", help="LLM model for the implementer")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Materialize the run, check its starting kernel on the GPU, and stop",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        schema = load(args.schema_file or find(args.sikl_root, args.schema))
        starting_kernel = (
            args.starting_kernel.read_text(encoding="utf-8")
            if args.starting_kernel
            else workspace.torch_starting_kernel(schema, args.sikl_root)
        )
        run_dir = args.run_dir or Path.cwd() / "sikl-runs" / f"{schema.name}-{time.strftime('%Y%m%d_%H%M%S')}"
        run = workspace.materialize(
            schema, run_dir, sikl_root=args.sikl_root, language=args.kernel_backend, starting_kernel=starting_kernel
        )
        workspace.check_starting_kernel(run)
    except (SchemaError, FileExistsError, workspace.StartingKernelError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    handoff.write(schema, run.handoff, run.program_md, kernel_backend=args.kernel_backend, sikl_root=args.sikl_root)
    workspace.commit(run)
    print(f"run directory: {run.root}")
    print(f"scored workloads: {len(schema.workloads)}")
    if args.prepare_only:
        return 0

    command = forge.build_command(
        run,
        kernel_backend=args.kernel_backend,
        gpu_target=args.gpu_target,
        max_hours=args.max_hours,
        git_branch=DEFAULT_GIT_BRANCH,
        model=args.model,
    )
    status = forge.dispatch(command, cwd=run.workspace)
    if status != 0:
        print(f"error: forge-loop exited {status}", file=sys.stderr)
        return status
    print(f"summary: {report.finalize(schema, run)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
