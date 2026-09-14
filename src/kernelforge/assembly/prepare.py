# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Programmatic compiler-output preparation inside a normal Forge campaign."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING

from kernelforge.assembly.capture import bind_compile
from kernelforge.assembly.compiler import _validate_source
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
        paths = [_inside(root, _inside(root, kernel).with_suffix(".s"))]
    if len(paths) != 1:
        raise AssemblyPreparationError("assembly campaigns currently require exactly one assembly source")
    return paths


def frozen_paths(workspace: str | Path, editable: list[str | Path]) -> list[str]:
    """Freeze every tracked file except the campaign's explicitly selected implementation."""
    root = Path(workspace).resolve()
    allowed = {_inside(root, path) for path in editable}
    tracked = _git(root, "ls-files", "-z").split("\0")
    return [str(root / path) for path in tracked if path and (root / path).resolve() not in allowed]


def _timeout(deadline: float, cap: int = 1800) -> int:
    remaining = deadline - time.time()
    if remaining < 1:
        raise AssemblyPreparationError("assembly preparation reached the campaign deadline")
    return max(1, min(cap, int(remaining)))


def _assembly_score(source: dict, initial: dict) -> float:
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


def seed_source_baseline(iteration_config: IterationConfig, record: dict) -> None:
    """Keep the source as the incumbent; preparation is never a performance KEEP."""
    source = record["source_benchmark"]
    iteration_config.baseline_wall_ms = source["median_ms"]
    iteration_config.pristine_baseline_wall_ms = source["median_ms"]
    iteration_config.baseline_case_times = dict(source["case_times"])
    iteration_config.preloop_baseline_unscored_cases = list(source.get("unscored_cases", []))


def select_result(result: dict, record: dict) -> None:
    """Select the original implementation when the search has no accepted source win."""
    result["assembly_preparation"] = record
    if result["improved"]:
        result["selected_implementation"] = "assembly"
        return
    result["selected_implementation"] = "original"
    result["assembly_search_result"] = {
        key: result.get(key) for key in ("best_ms", "best_commit", "mean_case_speedup", "best_manifest")
    }
    result.update(
        best_ms=record["source_benchmark"]["median_ms"],
        best_commit=record["source_base_commit"],
        best_iteration=0,
        mean_case_speedup=1.0,
        total_speedup=1.0,
        best_manifest="",
    )


async def _validate(driver: str, threshold: float, deadline: float) -> ValidationReport:
    return await run_validation_pipeline(driver, snr_threshold=threshold, timeout_per_stage=_timeout(deadline))


async def verify_assembly(
    kernel: Path, assembly: Path, driver: str, target: str, threshold: float, deadline: float
) -> ValidationReport:
    """Require correctness and prove that a fresh driver consumes the editable source."""
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
    text = source.decode("utf-8")
    symbols = re.findall(r"^\s*\.amdhsa_kernel\s+(\S+)\s*$", text, re.MULTILINE)
    for symbol in symbols:
        text, count = re.subn(
            rf"^([ \t]*{re.escape(symbol)}:)[ \t]*(?://[^\n]*)?$",
            r"\1\n    s_endpgm // FORGE_ASSEMBLY_EXECUTION_PROBE",
            text,
            flags=re.MULTILINE,
        )
        if count != 1:
            raise AssemblyPreparationError("cannot locate an assembly entry for the execution-path negative control")
    if not symbols:
        raise AssemblyPreparationError("assembly execution probe requires AMDHSA kernel entries")
    try:
        assembly.write_text(text, encoding="utf-8")
        execution_probe = await _validate(driver, threshold, deadline)
    finally:
        assembly.write_bytes(source)
    if execution_probe.all_passed or not execution_probe.results:
        raise AssemblyPreparationError(
            "driver accepted no-op assembly; it must test the returned candidate on fresh outputs"
        )
    restored = await _validate(driver, threshold, deadline)
    if not restored.results or not restored.all_passed:
        raise AssemblyPreparationError(restored.failed_output or "restored assembly failed correctness")
    return restored


def _load_ready(path: Path, workspace: Path, kernel: Path, assembly: Path, base_commit: str, target: str) -> dict:
    record: dict = json.loads(path.read_text(encoding="utf-8"))
    if (
        record.get("schema_version") != 2
        or record.get("status") != "ready"
        or record.get("kernel") != kernel.relative_to(workspace).as_posix()
        or record.get("assembly") != assembly.relative_to(workspace).as_posix()
        or record.get("source_base_commit") != base_commit
        or record.get("gpu_target") != target
    ):
        raise AssemblyPreparationError("assembly preparation record does not match this campaign")
    if kernel != assembly and record.get("launcher_sha256") != _digest(kernel):
        raise AssemblyPreparationError("the verified Python launcher changed after preparation")
    if record.get("source_sha256") != _digest(path.parent / ("source" + kernel.suffix)):
        raise AssemblyPreparationError("the original source reference changed after preparation")
    manifest = record.get("binding_manifest")
    if manifest and record.get("binding_manifest_sha256") != _digest(_inside(workspace, manifest)):
        raise AssemblyPreparationError("the compiler binding manifest changed after preparation")
    _git(workspace, "merge-base", "--is-ancestor", record["preparation_commit"], "HEAD")
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
) -> dict:
    """Export and bind compiler output, or verify an explicitly bound assembly input."""
    workspace = Path(config.workspace).resolve()
    kernel_path = _inside(workspace, kernel)
    assembly = assembly_edit_paths(workspace, kernel, sources)[0]
    root = workspace / "forge_experiments" / "assembly_preparation"
    record_path = root / "result.json"
    if record_path.is_file():
        return _load_ready(record_path, workspace, kernel_path, assembly, base_commit, config.gpu_target)
    if resume:
        raise AssemblyPreparationError("assembly resume requires a verified preparation record; start a fresh campaign")
    if _git(workspace, "status", "--porcelain"):
        raise AssemblyPreparationError("commit workspace changes before assembly preparation")
    start_commit = _git(workspace, "rev-parse", "HEAD")
    original = kernel_path.read_bytes()
    capture = not assembly.exists()
    manifest = _inside(workspace, assembly.with_suffix(assembly.suffix + ".json"))
    relative_assembly = Path(os.path.relpath(assembly, kernel_path.parent)).as_posix()
    if capture:
        if kernel_path.suffix != ".py":
            raise AssemblyPreparationError("automatic capture requires a Python FlyDSL compile call")
        if manifest.exists():
            raise AssemblyPreparationError("assembly manifest already exists; start from the original source")
        export_source = bind_compile(original.decode(), relative_assembly, config.gpu_target, export=True)
        candidate_source = bind_compile(original.decode(), relative_assembly, config.gpu_target, export=False)
    original_assembly = assembly.read_bytes() if assembly.exists() else None
    root.mkdir(parents=True, exist_ok=True)
    reference = root / ("source" + kernel_path.suffix)
    reference.write_bytes(original)
    committed = False
    paths = list(
        dict.fromkeys([kernel_path.relative_to(workspace).as_posix(), assembly.relative_to(workspace).as_posix()])
    )
    if capture:
        paths.append(manifest.relative_to(workspace).as_posix())
    try:
        source_report = await _validate(driver, threshold, deadline)
        if not source_report.results or not source_report.all_passed:
            raise AssemblyPreparationError(source_report.failed_output or "original source correctness failed")
        acceptance = await accept_candidate(
            str(workspace), timeout_cap_sec=_timeout(deadline), candidate_label="assembly source baseline"
        )
        if not acceptance.passed:
            raise AssemblyPreparationError(acceptance.detail)
        source_bench = await bench_wallclock(driver, timeout_sec=_timeout(deadline, 600))
        if not source_bench.get("success") or not source_bench.get("case_times"):
            raise AssemblyPreparationError("original source benchmark failed")
        _assembly_score(source_bench, source_bench)
        if capture:
            kernel_path.write_text(export_source, encoding="utf-8")
            export_report = await _validate(driver, threshold, deadline)
            if not export_report.results or not export_report.all_passed or not manifest.is_file():
                raise AssemblyPreparationError(export_report.failed_output or "compiler output was not captured")
            provenance = json.loads(manifest.read_text(encoding="utf-8"))
            if provenance["compiler_assembly_sha256"] != _digest(assembly):
                raise AssemblyPreparationError("initial assembly differs from compiler output")
            kernel_path.write_text(candidate_source, encoding="utf-8")
        report = await verify_assembly(kernel_path, assembly, driver, config.gpu_target, threshold, deadline)
        acceptance = await accept_candidate(
            str(workspace), timeout_cap_sec=_timeout(deadline), candidate_label="assembly compiler roundtrip"
        )
        if not acceptance.passed:
            raise AssemblyPreparationError(acceptance.detail)
        initial_bench = await bench_wallclock(driver, timeout_sec=_timeout(deadline, 600))
        if not initial_bench.get("success") or set(initial_bench.get("case_times", {})) != set(
            source_bench["case_times"]
        ):
            raise AssemblyPreparationError("assembly benchmark must cover the original source's complete case set")
        score = _assembly_score(source_bench, initial_bench)
        _git(workspace, "add", "--", *paths)
        if _git(workspace, "diff", "--cached", "--name-only"):
            _git(workspace, "commit", "-m", "forge: bind compiler assembly to original launcher")
        record = {
            "schema_version": 2,
            "status": "ready",
            "origin": "flydsl_compiler" if capture else "existing_assembly",
            "kernel": paths[0],
            "assembly": assembly.relative_to(workspace).as_posix(),
            "gpu_target": config.gpu_target,
            "source_base_commit": base_commit,
            "preparation_commit": _git(workspace, "rev-parse", "HEAD"),
            "source_sha256": hashlib.sha256(original).hexdigest(),
            "launcher_sha256": _digest(kernel_path),
            "initial_assembly_sha256": _digest(assembly),
            "binding_manifest": manifest.relative_to(workspace).as_posix() if capture else "",
            "binding_manifest_sha256": _digest(manifest) if capture else "",
            "source_benchmark": source_bench,
            "initial_assembly_benchmark": initial_bench,
            "roundtrip_mean_case_speedup": score,
            "correctness": report.summary(),
            "canonical_correctness": acceptance.detail,
            "canonical_unverified_reason": acceptance.unverified_reason,
            "build_failure_probe_passed": True,
            "execution_probe_passed": True,
        }
        atomic_write_text(record_path, json.dumps(record, indent=2) + "\n")
        committed = True
        print(
            f"  [assembly prepare] verified assembly roundtrip: {score:.6f}x; original remains the baseline", flush=True
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
            if capture:
                manifest.unlink(missing_ok=True)
