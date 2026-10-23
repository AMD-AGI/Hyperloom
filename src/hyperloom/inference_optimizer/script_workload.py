# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Prepare a ``--framework custom`` workspace that optimizes an existing model script unchanged.

The script imports ``<module>.<function>`` and passes its model through it once. The
workspace holds that module as the only editable file, and a benchmark directory whose
harness records references from the identity module, then measures every candidate at
that boundary (see ``assets/script_workload/harness.py``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

from hyperloom.common.sandbox import SandboxRoots, interpreter_roots

from .session.paths import asset_root

_HARNESS = "harness.py"
_CONFIG = "script_workload.json"
_IDENTITY = '''"""Optimizations for {script}; the identity until Hyperloom accepts a change."""


def {function}(model):
    return model
'''


def _tolerance(args: argparse.Namespace) -> dict:
    if args.rel_l2 is not None:
        return {"rel_l2": args.rel_l2}
    return {"autocast": args.calibrate_autocast, "headroom": args.headroom, "floor": args.floor}


def _write_benchmark(benchmark: Path, module_file: Path, args: argparse.Namespace) -> None:
    benchmark.mkdir()
    shutil.copy2(asset_root() / "assets" / "script_workload" / _HARNESS, benchmark / _HARNESS)
    script = args.script.resolve()
    config = {
        "script": str(script),
        "script_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
        "arguments": args.arguments,
        "cwd": str(args.cwd.resolve()),
        "python": args.python,
        "module": args.module,
        "function": args.function,
        "method": args.method,
        "record_calls": args.record_calls,
        "held_out": args.held_out,
        "warmup_s": args.warmup_s,
        "warmup_passes": 3,
        "timed_passes": args.timed_passes,
        "tolerance": _tolerance(args),
    }
    (benchmark / _CONFIG).write_text(json.dumps(config, indent=2) + "\n")
    scope = {"schema_version": 2, "file": str(module_file), "function": args.function}
    (benchmark / "optimization_scope.json").write_text(json.dumps(scope, indent=2) + "\n")
    entrypoint = benchmark / f"custom_{args.runner}.sh"
    entrypoint.write_text(
        f'#!/usr/bin/env bash\nset -euo pipefail\nexec {shlex.quote(args.python)} -B "$(dirname "$0")/{_HARNESS}" measure\n'
    )
    entrypoint.chmod(0o755)


def _init_module(root: Path, module_file: Path, args: argparse.Namespace) -> None:
    root.mkdir()
    module_file.write_text(_IDENTITY.format(script=args.script.name, function=args.function))
    git = ["git", "-C", str(root), "-c", "user.name=hyperloom", "-c", "user.email=hyperloom@local"]
    subprocess.run([*git, "init", "-q"], check=True)
    subprocess.run([*git, "add", module_file.name], check=True)
    subprocess.run([*git, "commit", "-qm", "Identity optimization module"], check=True)


def _check_import_order(order: dict, args: argparse.Namespace) -> None:
    """Refuse a script whose import of the module comes too late for its settings; warn on a risky one."""
    statement = f"`from {args.module} import {args.function}`"
    torch_site = f" (torch is first imported at {order['torch_site']})" if order["torch_site"] else ""
    late = [*(["the GPU was initialized"] if order["gpu_initialized"] else []), *order["compiler_imported"]]
    if late:
        raise SystemExit(
            f"{args.script.name} imports {args.module} at {order['module_site']}, after {' and '.join(late)}. "
            f"Runtime settings Hyperloom accepts would not take effect when you deploy the result. "
            f"Move {statement} above `import torch`{torch_site}."
        )
    if order["torch_imported"]:
        print(
            f"warning: {args.script.name} imports {args.module} at {order['module_site']}, after torch{torch_site}. "
            f"The settings checked so far still apply there, but {statement} above `import torch` is safer.",
            file=sys.stderr,
        )


def _record(benchmark: Path, args: argparse.Namespace) -> dict:
    command = [args.python, "-B", str(benchmark / _HARNESS), "record"]
    result = subprocess.run(command, check=True, stdout=subprocess.PIPE, text=True)
    return json.loads(result.stdout.splitlines()[-1])


def _discard_partial_workspace(workspace: Path) -> None:
    """Remove what a failed prepare created so it can run again, keeping the script's recording logs."""
    benchmark = workspace / "benchmark"
    logs = sorted(benchmark.glob("record_*.log"))
    for log in logs:
        shutil.move(log, workspace / log.name)
    shutil.rmtree(workspace / "optimization", ignore_errors=True)
    shutil.rmtree(benchmark, ignore_errors=True)
    if logs:
        print(f"The script's output while recording is kept in {workspace}/record_stderr.log", file=sys.stderr)


def prepare(args: argparse.Namespace) -> Path:
    """Create the workspace and record the references; refuse to touch an existing one."""
    workspace = args.workspace.resolve()
    root, benchmark = workspace / "optimization", workspace / "benchmark"
    if root.exists() or benchmark.exists():
        raise SystemExit(f"{workspace} is already prepared; references are recorded once")
    workspace.mkdir(parents=True, exist_ok=True)
    module_file = root / f"{args.module}.py"
    prepared = False
    try:
        _init_module(root, module_file, args)
        _write_benchmark(benchmark, module_file, args)
        report = _record(benchmark, args)
        _check_import_order(report["import_order"], args)
        prepared = True
    finally:
        if not prepared:
            _discard_partial_workspace(workspace)
    print(f"Recorded {report['calls']} calls; rel-L2 limits {report['rel_l2_limits']}")
    return workspace


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hyperloom script prepare", description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--script", type=Path, required=True, help="the unchanged model script")
    parser.add_argument("--module", help="optimization module the script imports (default: <script>_opt)")
    parser.add_argument("--function", default="apply", help="function the script passes its model through")
    parser.add_argument("--method", default="__call__", help="method of the returned object that runs the model")
    parser.add_argument("--python", default=sys.executable, help="interpreter of the model's environment")
    parser.add_argument("--cwd", type=Path, default=Path.cwd(), help="directory the script is run from")
    parser.add_argument("--runner", default="gfx11", help="runner label of the custom_<runner>.sh entrypoint")
    parser.add_argument("--record-calls", type=int, default=4, help="calls of the model recorded as references")
    parser.add_argument("--held-out", type=int, default=4, help="hidden perturbed inputs per recorded call")
    parser.add_argument("--timed-passes", type=int, default=100)
    parser.add_argument("--warmup-s", type=float, default=5.0)
    tolerance = parser.add_mutually_exclusive_group(required=True)
    tolerance.add_argument("--rel-l2", type=float, help="fixed relative-L2 limit per call")
    tolerance.add_argument(
        "--calibrate-autocast",
        choices=("bfloat16", "float16"),
        help="limit = error of the model under this autocast dtype x headroom",
    )
    parser.add_argument("--headroom", type=float, default=1.15)
    parser.add_argument("--floor", type=float, default=1e-4, help="smallest calibrated limit")
    parser.add_argument("arguments", nargs=argparse.REMAINDER, help="-- followed by the script's own arguments")
    return parser


def _interpreter(python: str) -> str:
    """Absolute, unresolved: a venv's bin/python must keep naming the venv."""
    return os.path.abspath(shutil.which(python) or python)


def benchmark_inputs(benchmark: Path) -> tuple[Path, ...]:
    """What a prepared benchmark's measurements read outside it: the script, its working directory, its interpreter."""
    config = benchmark / _CONFIG
    if not config.is_file():
        return ()
    workload = json.loads(config.read_text())
    return (Path(workload["script"]).parent, Path(workload["cwd"]), *interpreter_roots(Path(workload["python"])))


def sandbox_roots(argv: list[str]) -> SandboxRoots:
    """What ``script prepare`` writes and reads, derived from its flags for the bwrap sandbox."""
    if argv[:1] != ["prepare"] or {"-h", "--help"} & set(argv):
        return SandboxRoots()
    try:
        args, _ = _parser().parse_known_args(argv[1:])
    except SystemExit:
        # A malformed command line; the strict parse inside the sandbox reports it.
        return SandboxRoots()
    readonly = (args.script.resolve().parent, args.cwd.resolve(), *interpreter_roots(Path(_interpreter(args.python))))
    return SandboxRoots(writable=(args.workspace.resolve(),), readonly=readonly)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] != ["prepare"]:
        print("usage: hyperloom script prepare [--help] ...", file=sys.stderr)
        return 2
    args = _parser().parse_args(argv[1:])
    if args.arguments[:1] == ["--"]:
        args.arguments = args.arguments[1:]
    args.module = args.module or f"{args.script.stem}_opt"
    if not (args.module.isidentifier() and args.function.isidentifier()):
        raise SystemExit("--module and --function must be Python identifiers")
    if args.timed_passes < args.held_out:
        raise SystemExit("--timed-passes must be at least --held-out")
    args.python = _interpreter(args.python)
    print(prepare(args))
    return 0
