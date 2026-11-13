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

from hyperloom.common.interpreter_roots import foreign_interpreter_roots
from hyperloom.common.optimization_scope import aoti_config, optimization_scope
from hyperloom.common.reexec import ReexecRoots

from .session.paths import asset_root

_HARNESS = "harness.py"
_CONFIG = "script_workload.json"
_SCORER = "scorer.py"
# The two directories a prepared workspace holds.
_BENCHMARK, _OPTIMIZATION = "benchmark", "optimization"
_SCOPE = "optimization_scope.json"
# The --aoti flag, its dest on the parsed arguments, and the block it writes into the scope.
_AOTI = "aoti"
# One export attempt writes about this much into TMPDIR; see the AOTI section of the how-to.
_EXPORT_BUILD_GB = 10
_IDENTITY = '''"""Optimizations for {script}; the identity until Hyperloom accepts a change."""


def {function}(model):
    return model
'''


def _tolerance(args: argparse.Namespace) -> dict:
    if args.error_limit is not None:
        return {"limit": args.error_limit}
    return {"autocast": args.calibrate_autocast, "headroom": args.headroom, "floor": args.floor}


# Written by --aoti with nulls the operator must replace. Deliberately invalid: the launch
# refuses while a placeholder remains, which is the point -- a skeleton that passed would run
# a campaign against a range nobody chose. Valid JSON, so every reader still parses it and the
# validator can name the offending field.
_AOTI_SKELETON = {
    "_comment": (
        "Declare what must stay dynamic. If every input is fixed, replace dynamic_dims with []. "
        "A range is fixed into the compiled model for good: an input outside it fails in "
        "production. The agent may not change it."
    ),
    "enabled": True,
    "pin_all_others": True,
    "dynamic_dims": [
        {
            "_comment": (
                "input: the name as your script spells it (listed by prepare); dim: 0-indexed "
                "axis; min and max: inclusive; derivation: how you measured the range"
            ),
            "input": None,
            "dim": None,
            "min": None,
            "max": None,
            "derivation": None,
        }
    ],
}


def _aoti_block(declared: object) -> tuple[dict | None, str | None]:
    """The ``aoti`` block to write and the digest of the file it came from.

    ``--aoti`` alone writes the skeleton unvalidated. ``--aoti FILE`` validates now, so a
    typo costs seconds instead of surfacing at the first round of an unattended run.
    """
    if declared is None:
        return None, None
    if declared is True:
        return dict(_AOTI_SKELETON), None
    path = Path(str(declared))
    try:
        body = path.read_bytes()
        block = json.loads(body)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"--aoti {path}: {exc}") from exc
    if not isinstance(block, dict):
        raise SystemExit(f"--aoti {path}: the file must hold the aoti block as a JSON object")
    # Required rather than defaulted: aoti_config defaults it to false, so a file without it
    # would mean "export" here and "inert" when the same block is pasted into the scope by hand.
    if "enabled" not in block:
        raise SystemExit(f"--aoti {path}: the declaration must say whether it is enabled")
    return block, hashlib.sha256(body).hexdigest()


def _validate_aoti(benchmark: Path) -> None:
    """Fail prepare on a declaration the campaign would reject, naming the field."""
    try:
        aoti_config(str(benchmark))
    except ValueError as exc:
        raise SystemExit(f"the declaration in --aoti is not usable: {exc}") from exc


def _copy_scorer(benchmark: Path, scorer: Path | None) -> str | None:
    """Freeze the operator's scorer beside the harness; return its digest, or None."""
    if scorer is None:
        return None
    body = scorer.resolve().read_bytes()
    (benchmark / _SCORER).write_bytes(body)
    return hashlib.sha256(body).hexdigest()


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
        "scorer_sha256": _copy_scorer(benchmark, args.scorer),
    }
    aoti, digest = _aoti_block(getattr(args, _AOTI, None))
    if digest:
        config["aoti_sha256"] = digest
    (benchmark / _CONFIG).write_text(json.dumps(config, indent=2) + "\n")
    # Schema 1 unless an export is declared: ``function`` is read whatever the version, and 2
    # only gates the aoti block. An operator adding one by hand edits both.
    scope = {"schema_version": 2 if aoti else 1, "file": str(module_file), "function": args.function}
    if aoti:
        scope[_AOTI] = aoti
    (benchmark / _SCOPE).write_text(json.dumps(scope, indent=2) + "\n")
    # Not custom_<arch>.sh: prepare runs the model once and never learns the board, so any
    # architecture in the name would be a guess. The launcher's documented fallback takes the
    # directory's only .sh, and a prepared benchmark directory holds exactly this one.
    entrypoint = benchmark / "benchmark.sh"
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
    benchmark = workspace / _BENCHMARK
    logs = sorted(benchmark.glob("record_*.log"))
    for log in logs:
        shutil.move(log, workspace / log.name)
    shutil.rmtree(workspace / _OPTIMIZATION, ignore_errors=True)
    shutil.rmtree(benchmark, ignore_errors=True)
    if logs:
        kept = ", ".join(log.name for log in logs)
        print(f"The script's output while recording is kept in {workspace}: {kept}", file=sys.stderr)


def _recorded_summary(report: dict, args: argparse.Namespace) -> str:
    """The line printed once the references are recorded; it names the units, which a scorer changes."""
    units = f"in the units of {args.scorer.name}" if args.scorer else "relative L2"
    calibrated = f", calibrated from {args.calibrate_autocast} autocast" if args.calibrate_autocast else ""
    return f"Recorded {report['calls']} calls; error limits {report['rel_l2_limits']} ({units}{calibrated})"


def _recorded_inputs(report: dict, workspace: Path) -> str:
    """List what the recorded calls fed the model, so the operator names axes from evidence.

    Observed sizes are reported as observed and never written into the skeleton: the
    recorded calls show the sizes that happened to be exercised, not the sizes that are
    meant to vary. Only the operator knows the latter.
    """
    inputs = report.get("inputs", [])
    lines = [f"Inputs recorded ({report.get('calls', 0)} calls):"]
    varied = False
    for entry in inputs:
        shape = "(" + ", ".join(str(size) for size in entry["shape"]) + ")"
        notes = []
        for axis, (low, high) in sorted(entry["varies"].items()):
            notes.append(f"axis {axis}: observed {low} to {high}")
            varied = True
        lines.append(f"  {entry['name']:<24} {shape:<18} {'; '.join(notes)}".rstrip())
    if not varied:
        lines.append("Nothing varied across the recorded calls, so no range is suggested.")
    lines.append(f"Fill in benchmark/{_SCOPE}, then run: hyperloom script check --workspace {workspace}")
    return "\n".join(lines)


def prepare(args: argparse.Namespace) -> Path:
    """Create the workspace and record the references; refuse to touch an existing one."""
    workspace = args.workspace.resolve()
    root, benchmark = workspace / _OPTIMIZATION, workspace / _BENCHMARK
    if root.exists() or benchmark.exists():
        raise SystemExit(f"{workspace} is already prepared; references are recorded once")
    workspace.mkdir(parents=True, exist_ok=True)
    module_file = root / f"{args.module}.py"
    prepared = False
    try:
        _init_module(root, module_file, args)
        _write_benchmark(benchmark, module_file, args)
        if isinstance(getattr(args, _AOTI, None), str):
            _validate_aoti(benchmark)
        report = _record(benchmark, args)
        _check_import_order(report["import_order"], args)
        prepared = True
    finally:
        if not prepared:
            _discard_partial_workspace(workspace)
    print(_recorded_summary(report, args))
    if getattr(args, _AOTI, None) is True:
        print(_recorded_inputs(report, workspace))
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
    parser.add_argument(
        "--aoti",
        nargs="?",
        const=True,
        metavar="FILE",
        help="declare that the deliverable must be an ahead-of-time compiled package. Alone, writes a "
        "skeleton declaration to fill in and lists the inputs recorded; with a file, merges that "
        "declaration and validates it now",
    )
    parser.add_argument(
        "--scorer",
        type=Path,
        help="your own accuracy measure: a Python file defining score(candidate, reference), "
        "and optionally capture(output). Without it, outputs are compared by relative L2",
    )
    parser.add_argument("--record-calls", type=int, default=4, help="calls of the model recorded as references")
    parser.add_argument("--held-out", type=int, default=4, help="hidden perturbed inputs per recorded call")
    parser.add_argument("--timed-passes", type=int, default=100)
    parser.add_argument("--warmup-s", type=float, default=5.0)
    tolerance = parser.add_mutually_exclusive_group(required=True)
    tolerance.add_argument(
        "--error-limit",
        "--rel-l2",
        dest="error_limit",
        type=float,
        help="fixed error limit per call, in the scorer's units (relative L2 unless --scorer is given)",
    )
    tolerance.add_argument(
        "--calibrate-autocast",
        choices=("bfloat16", "float16"),
        help="limit = the error this autocast dtype costs the unmodified model, x headroom; "
        "measured in the scorer's units, so it reads as 'allow as much error as bfloat16 costs'",
    )
    parser.add_argument("--headroom", type=float, default=1.15)
    parser.add_argument("--floor", type=float, default=1e-4, help="smallest calibrated limit")
    parser.add_argument("arguments", nargs=argparse.REMAINDER, help="-- followed by the script's own arguments")
    return parser


def benchmark_inputs(benchmark: Path) -> tuple[Path, ...]:
    """What a prepared benchmark reads outside itself: the script, its working directory, its interpreter."""
    config = benchmark / _CONFIG
    if not config.is_file():
        return ()
    try:
        workload = json.loads(config.read_text())
        return (
            Path(workload["script"]).parent,
            Path(workload["cwd"]),
            *foreign_interpreter_roots(Path(workload["python"])),
        )
    except (OSError, ValueError, KeyError):
        # A config the sandbox cannot read names nothing; the prepared run reports it.
        return ()


def _interpreter(python: str) -> str:
    """Absolute, unresolved: a venv's bin/python must keep naming the venv."""
    return os.path.abspath(shutil.which(python) or python)


# What a prepared workspace pins by absolute path, and how to name each one to an operator.
_RECORDED_PATHS = (
    ("script", "the model script"),
    ("cwd", "the script's working directory"),
    ("python", "the model environment's interpreter"),
)


def _check_recorded_paths(workload: dict) -> list[str]:
    """Report every pinned path that is gone, and a script that changed under its digest."""
    problems = []
    for key, what in _RECORDED_PATHS:
        recorded = workload.get(key)
        if not recorded:
            problems.append(f"{_CONFIG} records no {key}")
        elif not Path(recorded).exists():
            problems.append(f"{what} is missing: {recorded} (recorded as {key} in {_CONFIG})")
    script = Path(workload.get("script") or "")
    digest = workload.get("script_sha256")
    if script.is_file() and digest and hashlib.sha256(script.read_bytes()).hexdigest() != digest:
        problems.append(f"the model script changed since prepare recorded it: {script}")
    return problems


def _check_entrypoint(benchmark: Path) -> list[str]:
    """The launcher takes custom_<gpu-type>.sh, else the directory's only .sh -- so one of those must hold."""
    scripts = sorted(path.name for path in benchmark.glob("*.sh") if path.is_file())
    if not scripts:
        return [f"{benchmark} holds no .sh entrypoint"]
    if len(scripts) > 1 and not any(name.startswith("custom_") for name in scripts):
        return [
            f"{benchmark} holds {len(scripts)} .sh files ({', '.join(scripts)}) and none names a board, so none resolves"
        ]
    return []


def _check_scope(benchmark: Path, workspace: Path) -> list[str]:
    """The declared module must still be the editable file prepare put in the workspace."""
    try:
        scope = optimization_scope(str(benchmark))
    except ValueError as exc:
        return [str(exc)]
    if scope is None:
        return [f"{benchmark} holds no {_SCOPE}"]
    if scope.file.parent != workspace / _OPTIMIZATION:
        return [f"the optimization module is outside this workspace: {scope.file}"]
    return []


def _placeholders(benchmark: Path) -> list[str]:
    """Skeleton fields nobody filled in; they read as a valid block to anything but the validator."""
    try:
        scope = json.loads((benchmark / _SCOPE).read_text())
    except (OSError, ValueError):
        return []
    return [
        f"aoti.dynamic_dims[{index}].{key} is still the placeholder --aoti wrote"
        for index, entry in enumerate(scope.get(_AOTI, {}).get("dynamic_dims", []))
        if isinstance(entry, dict)
        for key, value in sorted(entry.items())
        if value is None
    ]


def _check_aoti(benchmark: Path) -> list[str]:
    """Validate the declaration exactly as the campaign's own gate will."""
    placeholders = _placeholders(benchmark)
    if placeholders:
        return placeholders
    try:
        aoti_config(str(benchmark))
    except ValueError as exc:
        return [str(exc)]
    return []


def _export_warnings(benchmark: Path) -> list[str]:
    """Host conditions an export needs, reported but never fatal: a workspace may be prepared elsewhere."""
    try:
        config = aoti_config(str(benchmark))
    except ValueError:
        return []
    if config is None or not config.enabled:
        return []
    warnings = []
    tmpdir = Path(os.environ.get("TMPDIR") or "/tmp")
    free_gb = shutil.disk_usage(tmpdir).free / 1e9 if tmpdir.is_dir() else 0
    if free_gb < _EXPORT_BUILD_GB:
        warnings.append(f"TMPDIR={tmpdir} has {free_gb:.0f} GB free; one export attempt writes about 10 GB")
    rocm = Path(os.environ.get("ROCM_PATH") or "/opt/rocm")
    if not (rocm / "include" / "hip" / "hip_runtime.h").is_file():
        warnings.append(f"no ROCm development headers under {rocm}; building an export needs them")
    return warnings


def check(workspace: Path) -> int:
    """Report every problem with a prepared workspace at once, so one run names them all."""
    benchmark = workspace.resolve() / _BENCHMARK
    config = benchmark / _CONFIG
    try:
        workload = json.loads(config.read_text())
    except (OSError, ValueError) as exc:
        print(f"{workspace} is not a prepared workspace: {exc}", file=sys.stderr)
        return 1
    problems = [
        *_check_recorded_paths(workload),
        *_check_entrypoint(benchmark),
        *_check_scope(benchmark, workspace.resolve()),
        *_check_aoti(benchmark),
    ]
    for warning in _export_warnings(benchmark):
        print(f"  warning: {warning}", file=sys.stderr)
    for problem in problems:
        print(f"  {problem}", file=sys.stderr)
    if problems:
        print(f"{len(problems)} problem(s) in {workspace}", file=sys.stderr)
        return 1
    print(f"{workspace} is ready to launch")
    return 0


def _check_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hyperloom script check", description=check.__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    return parser


def reexec_roots(argv: list[str]) -> ReexecRoots:
    """What ``script prepare`` writes and reads, from its own flags (see :mod:`hyperloom.common.reexec`)."""
    if argv[:1] != ["prepare"] or {"-h", "--help"} & set(argv):
        return ReexecRoots()
    try:
        args, _ = _parser().parse_known_args(argv[1:])
    except SystemExit:
        # Malformed; the strict parse inside the sandbox is what reports it.
        return ReexecRoots()
    readonly = (
        args.script.resolve().parent,
        args.cwd.resolve(),
        *foreign_interpreter_roots(Path(_interpreter(args.python))),
    )
    return ReexecRoots(writable=(args.workspace.resolve(),), readonly=readonly)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["check"]:
        return check(_check_parser().parse_args(argv[1:]).workspace)
    if argv[:1] != ["prepare"]:
        print("usage: hyperloom script {prepare,check} [--help] ...", file=sys.stderr)
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
