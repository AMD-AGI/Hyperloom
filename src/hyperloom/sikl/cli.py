# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""`hyperloom sikl` — optimize one SIKL definition with KernelForge."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from hyperloom.sikl import forge, handoff, report, workspace
from hyperloom.sikl.definition import BundleError, load

DEFAULT_GIT_BRANCH = "forge-optimize"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hyperloom sikl",
        description="Optimize one SIKL definition with KernelForge and score it with SIKL's harness.",
    )
    parser.add_argument("--bundle", required=True, type=Path, help="Exported SIKL schema collection")
    parser.add_argument("--definition", required=True, help="Definition name to optimize")
    parser.add_argument("--sikl-root", required=True, type=Path, help="SIKL checkout that provides sikl.benchmarks")
    parser.add_argument("--run-dir", type=Path, help="Run directory (default: ./sikl-runs/<definition>-<stamp>)")
    parser.add_argument("--kernel-backend", default="triton", help="Candidate language (default: triton)")
    parser.add_argument("--gpu-target", default="gfx950", help="GPU arch passed to forge-loop")
    parser.add_argument("--max-hours", type=float, default=4.0, help="forge-loop wall-clock budget")
    parser.add_argument("--model", default="", help="LLM model for the implementer")
    parser.add_argument("--warmup-runs", type=int, default=20, help="Harness warmup runs per case")
    parser.add_argument("--iterations", type=int, default=100, help="Harness sampling budget per case")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Materialize the run, choose its starting kernel on the GPU, and stop",
    )
    return parser


def _run_dir(args: argparse.Namespace) -> Path:
    if args.run_dir:
        return args.run_dir
    return Path.cwd() / "sikl-runs" / f"{args.definition}-{time.strftime('%Y%m%d_%H%M%S')}"


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        definition = load(args.bundle, args.definition, language=args.kernel_backend)
    except BundleError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    try:
        run = workspace.materialize(definition, args.bundle, _run_dir(args), sikl_root=args.sikl_root)
        starting_point = workspace.choose_starting_kernel(definition, run)
    except (FileExistsError, workspace.StartingKernelError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    handoff.write(
        definition, run.handoff, run.program_md, kernel_backend=args.kernel_backend, starting_point=starting_point
    )
    workspace.commit(run)
    print(f"run directory: {run.root}")
    print(f"scored workloads: {len(definition.workloads)}")
    print(f"starting kernel: {starting_point}")
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

    summary = report.finalize(
        definition, run, sikl_root=args.sikl_root, warmup=args.warmup_runs, iterations=args.iterations
    )
    print(f"summary: {summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
