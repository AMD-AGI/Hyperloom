# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Correctness-only assembly preparation inside a normal Forge campaign."""

from __future__ import annotations

import ast
import asyncio
import hashlib
import json
import math
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING

from kernelforge.assembly.compiler import AssemblyError, _validate_source
from kernelforge.durable_io import atomic_write_text
from kernelforge.loop.canonical_correctness import accept_candidate
from kernelforge.loop.validation import run_validation_pipeline
from kernelforge.mcp_server.tools.bench import bench_wallclock, calculate_mean_case_speedup

if TYPE_CHECKING:
    from kernelforge.config import Config
    from kernelforge.loop.runner import IterationConfig
    from kernelforge.loop.validation import ValidationReport


class AssemblyPreparationError(ValueError):
    """No verified assembly implementation is available for optimization."""


def _git(workspace: Path, *args: str) -> str:
    result = subprocess.run(["git", "-C", str(workspace), *args], capture_output=True, text=True, check=True)
    return result.stdout.strip()


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inside(workspace: Path, path: str | Path) -> Path:
    raw = Path(path)
    if not raw.is_absolute():
        raw = workspace / raw
    resolved = raw.resolve()
    if raw.is_symlink() or not resolved.is_relative_to(workspace):
        raise AssemblyPreparationError(f"assembly inputs must be regular files inside the workspace: {raw}")
    return resolved


def assembly_edit_paths(workspace: str | Path, kernel: str, sources: list[str]) -> list[Path]:
    """Select the explicit assembly surface, never all assembly files in a repository."""
    root = Path(workspace).resolve()
    paths = [_inside(root, path) for path in [kernel, *sources] if Path(path).suffix.lower() in {".s", ".asm"}]
    paths = list(dict.fromkeys(paths))
    if not paths:
        paths = [_inside(root, kernel).with_suffix(".s")]
    if len(paths) != 1:
        raise AssemblyPreparationError("assembly campaigns currently require exactly one assembly source")
    return paths


def frozen_paths(workspace: str | Path, editable: list[str | Path]) -> list[str]:
    """Freeze every tracked file except the campaign's explicitly selected implementation."""
    root = Path(workspace).resolve()
    allowed = {_inside(root, path) for path in editable}
    tracked = _git(root, "ls-files", "-z").split("\0")
    return [str(root / path) for path in tracked if path and (root / path).resolve() not in allowed]


def check_launcher(kernel: Path) -> None:
    """An assembly port exposes Python launch code, not another frontend implementation."""
    if kernel.suffix.lower() in {".s", ".asm"}:
        return
    if kernel.suffix != ".py":
        raise AssemblyPreparationError("assembly PORT currently accepts Python Triton/FlyDSL kernels or an existing .s")
    tree = ast.parse(kernel.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    if any(name.split(".")[0] in {"triton", "flydsl"} for name in modules):
        raise AssemblyPreparationError(
            "the port must execute standalone ASM; remove source-frontend imports from its launcher"
        )
    if "kernelforge.assembly.hip" not in modules or "kernelforge.assembly.compiler" not in modules:
        raise AssemblyPreparationError(
            "the Python launcher must build the .s with assemble() and execute it with HipKernel"
        )


def _timeout(deadline: float, cap: int = 1800) -> int:
    remaining = deadline - time.time()
    if remaining < 1:
        raise AssemblyPreparationError("assembly preparation reached the campaign deadline")
    return max(1, min(cap, int(remaining)))


def _port_score(source: dict, initial: dict) -> float:
    for measurement in (source, initial):
        for latency in [measurement.get("median_ms"), *measurement.get("case_times", {}).values()]:
            if not isinstance(latency, (int, float)) or not math.isfinite(latency) or latency <= 0:
                raise AssemblyPreparationError("assembly preparation needs positive finite benchmark timings")
    score = calculate_mean_case_speedup(
        initial["case_times"], source["case_times"], set(source.get("unscored_cases", []))
    )
    if score is None:
        raise AssemblyPreparationError("assembly preparation has no scored cases")
    return score


def seed_port_baseline(iteration_config: IterationConfig, record: dict) -> None:
    """Adopt the correct port as the incumbent while retaining source-relative scoring."""
    source = record["source_benchmark"]
    initial = record["initial_assembly_benchmark"]
    score = _port_score(source, initial)
    iteration_config.baseline_wall_ms = source["median_ms"]
    iteration_config.pristine_baseline_wall_ms = source["median_ms"]
    iteration_config.baseline_case_times = dict(source["case_times"])
    iteration_config.preloop_baseline_unscored_cases = list(source.get("unscored_cases", []))
    iteration_config.warm_start_wall_ms = initial["median_ms"]
    iteration_config.warm_start_mean_case_speedup = score
    iteration_config.warm_start_bench = initial
    iteration_config.warm_start_commit = record["port_commit"]
    iteration_config.warm_start_solution_slug = "assembly PORT"


async def _validate(driver: str, threshold: float, deadline: float) -> ValidationReport:
    return await run_validation_pipeline(driver, snr_threshold=threshold, timeout_per_stage=_timeout(deadline))


async def verify_assembly(
    kernel: Path, assembly: Path, driver: str, target: str, threshold: float, deadline: float
) -> ValidationReport:
    """Require correctness and prove that a fresh driver consumes the editable source."""
    check_launcher(kernel)
    source = assembly.read_bytes()
    _validate_source(source.decode("utf-8"), target)
    correct = await _validate(driver, threshold, deadline)
    if not correct.results or not correct.all_passed:
        raise AssemblyPreparationError(correct.failed_output or "assembly correctness failed")
    marker = "FORGE_ASSEMBLY_BUILD_PROBE"
    try:
        assembly.write_bytes(source + f'\n.error "{marker}"\n'.encode())
        probe = await _validate(driver, threshold, deadline)
    finally:
        assembly.write_bytes(source)
    if probe.all_passed or marker not in probe.failed_output:
        raise AssemblyPreparationError(
            "driver did not propagate the deliberate assembly build failure; reject cached binaries or source fallback"
        )
    restored = await _validate(driver, threshold, deadline)
    if not restored.results or not restored.all_passed:
        raise AssemblyPreparationError(restored.failed_output or "restored assembly failed correctness")
    return restored


def _load_ready(path: Path, workspace: Path, kernel: Path, assembly: Path, base_commit: str, target: str) -> dict:
    record: dict = json.loads(path.read_text(encoding="utf-8"))
    if (
        record.get("schema_version") != 1
        or record.get("status") != "ready"
        or record.get("kernel") != kernel.relative_to(workspace).as_posix()
        or record.get("assembly") != assembly.relative_to(workspace).as_posix()
        or record.get("source_base_commit") != base_commit
        or record.get("gpu_target") != target
    ):
        raise AssemblyPreparationError("assembly preparation record does not match this campaign")
    if kernel != assembly and record.get("launcher_sha256") != _digest(kernel):
        raise AssemblyPreparationError("the verified Python launcher changed after PORT")
    if record.get("source_sha256") != _digest(path.parent / ("source" + kernel.suffix)):
        raise AssemblyPreparationError("the original source reference changed after PORT")
    _git(workspace, "merge-base", "--is-ancestor", record["port_commit"], "HEAD")
    return record


async def prepare_assembly(
    *,
    config: Config,
    kernel: str,
    driver: str,
    sources: list[str],
    base_commit: str,
    threshold: float,
    deadline: float,
    resume: bool = False,
    program: str = "",
    permission_mode=None,
    usage=None,
) -> dict:
    """Port one source in place, preserving its public Python entry and the original export base."""
    workspace = Path(config.workspace).resolve()
    kernel_path = _inside(workspace, kernel)
    assembly = assembly_edit_paths(workspace, kernel, sources)[0]
    root = workspace / "forge_experiments" / "assembly_port"
    record_path = root / "result.json"
    if record_path.is_file():
        return _load_ready(record_path, workspace, kernel_path, assembly, base_commit, config.gpu_target)
    if resume:
        raise AssemblyPreparationError("assembly resume requires a verified PORT record; start a fresh campaign")
    if kernel_path.suffix.lower() not in {".py", ".s", ".asm"}:
        raise AssemblyPreparationError("assembly PORT currently accepts Python Triton/FlyDSL kernels or an existing .s")
    if _git(workspace, "status", "--porcelain", "--untracked-files=no"):
        raise AssemblyPreparationError("commit tracked workspace changes before assembly preparation")
    start_commit = _git(workspace, "rev-parse", "HEAD")
    original = kernel_path.read_bytes()
    original_assembly = assembly.read_bytes() if assembly.exists() else None
    if not original_assembly and kernel_path == assembly:
        raise AssemblyPreparationError("the assembly input is empty")
    root.mkdir(parents=True, exist_ok=True)
    reference = root / ("source" + kernel_path.suffix)
    reference.write_bytes(original)
    source_report = await _validate(driver, threshold, deadline)
    if not source_report.results or not source_report.all_passed:
        raise AssemblyPreparationError(source_report.failed_output or "original source correctness failed")
    source_acceptance = await accept_candidate(
        str(workspace), timeout_cap_sec=_timeout(deadline), candidate_label="assembly source baseline"
    )
    if not source_acceptance.passed:
        raise AssemblyPreparationError(source_acceptance.detail)
    source_bench = await bench_wallclock(driver, timeout_sec=_timeout(deadline, 600))
    if not source_bench.get("success") or not source_bench.get("case_times"):
        raise AssemblyPreparationError("original source benchmark failed: " + str(source_bench.get("message", "")))

    history = ""
    attempts = 0
    agent = None
    committed = False
    paths = list(
        dict.fromkeys([kernel_path.relative_to(workspace).as_posix(), assembly.relative_to(workspace).as_posix()])
    )
    try:
        for attempts in range(4):
            try:
                report = await verify_assembly(kernel_path, assembly, driver, config.gpu_target, threshold, deadline)
                acceptance = await accept_candidate(
                    str(workspace), timeout_cap_sec=_timeout(deadline), candidate_label="assembly PORT"
                )
                if not acceptance.passed:
                    raise AssemblyPreparationError(acceptance.detail)
                break
            except (AssemblyPreparationError, AssemblyError, OSError, SyntaxError, UnicodeError) as error:
                history = str(error)[-6000:]
            if attempts == 3 or kernel_path == assembly:
                raise AssemblyPreparationError(history)
            if agent is None:
                from kernelforge.orchestrator.agent import make_agent_fn

                if not assembly.exists():
                    assembly.write_text("// Assembly PORT candidate.\n", encoding="utf-8")
                protected = frozen_paths(workspace, [kernel_path, assembly]) + [str(reference)]
                instructions = f"""PORT this operator to standalone AMDGPU assembly for {config.gpu_target}.
Original source (read only): {reference}
Editable Python entry: {kernel_path}
Editable complete AMDHSA source: {assembly}
Protected driver: {driver}
Preserve the Python public API, shapes, dtypes, layouts and numerical semantics exercised by the driver.
Replace the Python implementation with launch glue using kernelforge.assembly.compiler.assemble and
kernelforge.assembly.hip.HipKernel. Match the AMDHSA argument ABI, symbol, grid, block and LDS.
Forward the caller's current stream and validate the supported input domain. Reject unsupported inputs.
Build fresh source outside timing/capture, keep the module alive through graph replay, and propagate failures.
No source-frontend imports, source fallback, binary-only candidates, or computation in the Python wrapper.
Obtain the first .s from compiler output or implement the original math in assembly; a supplied reference
assembly is an attributed seed, not an independently discovered improvement. Only these two files are editable.
Do not commit or alter the driver/reference. This phase requires correctness only: a slower port is acceptable.
The host will verify correctness, deliberately break assembly compilation, and verify restoration before
handing the fixed launcher and editable .s to the performance loop.

Caller context:
{program}
"""
                agent = make_agent_fn(
                    config=config,
                    program_md=instructions,
                    kernel_backend_name="assembly",
                    insession_gate=True,
                    correctness_only=True,
                    driver_script=driver,
                    snr_threshold=threshold,
                    source_files=[str(kernel_path), str(assembly)],
                    extra_protected_paths=protected,
                    commit_new_paths=[assembly.relative_to(workspace).as_posix()],
                    permission_mode=permission_mode,
                    usage=usage,
                    session_timeout_sec=_timeout(deadline),
                )
            print(f"  [assembly PORT] attempt {attempts + 1}/3: {history[-500:]}", flush=True)
            sink: dict = {}
            await asyncio.wait_for(agent(str(kernel_path), history, session_sink=sink), timeout=_timeout(deadline))
            if sink.get("integrity_violation"):
                restore = sink.get("integrity_restore")
                if callable(restore):
                    restore()
                raise AssemblyPreparationError("assembly PORT changed a protected driver/source file")
        initial_bench = await bench_wallclock(driver, timeout_sec=_timeout(deadline, 600))
        if not initial_bench.get("success") or set(initial_bench.get("case_times", {})) != set(
            source_bench["case_times"]
        ):
            raise AssemblyPreparationError("assembly benchmark must cover the original source's complete case set")
        _port_score(source_bench, initial_bench)
        _git(workspace, "add", "--", *paths)
        if _git(workspace, "diff", "--cached", "--name-only"):
            _git(workspace, "commit", "-m", "forge: verified assembly port")
        record = {
            "schema_version": 1,
            "status": "ready",
            "kernel": paths[0],
            "assembly": assembly.relative_to(workspace).as_posix(),
            "gpu_target": config.gpu_target,
            "source_base_commit": base_commit,
            "port_commit": _git(workspace, "rev-parse", "HEAD"),
            "source_sha256": hashlib.sha256(original).hexdigest(),
            "launcher_sha256": _digest(kernel_path),
            "initial_assembly_sha256": _digest(assembly),
            "port_attempts": attempts,
            "source_benchmark": source_bench,
            "initial_assembly_benchmark": initial_bench,
            "correctness": report.summary(),
            "canonical_correctness": acceptance.detail,
            "canonical_unverified_reason": acceptance.unverified_reason,
            "build_failure_probe_passed": True,
        }
        atomic_write_text(record_path, json.dumps(record, indent=2) + "\n")
        committed = True
        print(
            f"  [assembly PORT] PASS; fixed launcher, editable {record['assembly']}; report: {record_path}", flush=True
        )
        return record
    finally:
        if not committed:
            _git(workspace, "reset", "--soft", start_commit)
            _git(workspace, "reset", start_commit, "--", *paths)
            kernel_path.write_bytes(original)
            if kernel_path != assembly:
                if original_assembly is None:
                    assembly.unlink(missing_ok=True)
                else:
                    assembly.write_bytes(original_assembly)
