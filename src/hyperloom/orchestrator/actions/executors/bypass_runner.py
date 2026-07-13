# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Bypass benchmark runner (CLI).

Drop-in alternative to ``python -m Magpie -v benchmark ... --run-mode local``.
It accepts the same CLI flags and the same environment contract, reuses the
InferenceX benchmark scripts (the same source Magpie drives), and writes a
Magpie-compatible workspace + ``benchmark_report.json`` so Hyperloom's
executors and collectors consume bypass runs unchanged.

Scope (Stage 2a): single-node ``--run-mode local`` for sglang/vllm/atom,
including the optional ``RUN_EVAL`` accuracy pass that the InferenceX scripts
already implement. Docker/Ray/scriptable/profile-specific handling is deferred.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from . import bypass_report

# Magpie-generic benchmark scripts shipped in the Magpie checkout. Reused as-is
# so bypass and Magpie resolve to the same InferenceX-driven benchmark.
_MAGPIE_SCRIPTS_SUBDIR = ("scripts", "benchmark")
# InferenceX native script prefixes by framework (mirror Magpie).
_NATIVE_PREFIX = {"sglang": "dsr1", "vllm": "gptoss"}


def _resolve_inferencex_path(bench: dict[str, Any]) -> str:
    """Resolve the InferenceX checkout path.

    Precedence mirrors what Hyperloom already pins: explicit YAML value, then
    ``MAGPIE_INFERENCEX_PATH``, then ``INFERENCEX_PATH``.

    Args:
        bench: The ``benchmark`` section of the config.

    Returns:
        The resolved InferenceX path (may be empty if unresolved).
    """
    return (
        str(bench.get("inferencex_path") or "").strip()
        or os.environ.get("MAGPIE_INFERENCEX_PATH", "").strip()
        or os.environ.get("INFERENCEX_PATH", "").strip()
    )


def _resolve_magpie_scripts_dir() -> Path | None:
    """Locate the Magpie generic benchmark scripts dir via ``MAGPIE_PATH``.

    Returns:
        The scripts dir when present, else None.
    """
    magpie_path = os.environ.get("MAGPIE_PATH", "").strip()
    if not magpie_path:
        return None
    scripts = Path(magpie_path, "Magpie", *_MAGPIE_SCRIPTS_SUBDIR)
    return scripts if scripts.is_dir() else None


def _sync_scripts(inferencex_path: Path) -> None:
    """Copy Magpie generic scripts into ``InferenceX/benchmarks`` (atomic).

    Best-effort and idempotent; no-op when the Magpie scripts dir is absent.

    Args:
        inferencex_path: InferenceX checkout root.
    """
    scripts = _resolve_magpie_scripts_dir()
    if scripts is None:
        return
    target = inferencex_path / "benchmarks"
    target.mkdir(parents=True, exist_ok=True)
    for script in scripts.glob("*.sh"):
        dst = target / script.name
        tmp = dst.with_suffix(dst.suffix + ".tmp")
        shutil.copy2(script, tmp)
        os.chmod(tmp, 0o755)
        os.replace(tmp, dst)


def _find_script(benchmarks_dir: Path, name: str) -> Path | None:
    """Find ``name`` in ``benchmarks_dir`` (top-level first, then recursive)."""
    top = benchmarks_dir / name
    if top.exists():
        return top
    for match in benchmarks_dir.rglob(name):
        if match.is_file():
            return match
    return None


def _resolve_script(bench: dict[str, Any], inferencex_path: Path, runner_type: str) -> str:
    """Resolve the benchmark script path relative to InferenceX (3-tier).

    Mirrors Magpie: explicit benchmark_script, then native
    ``{prefix}_{precision}_{runner}.sh``, then generic ``{framework}_{runner}.sh``.

    Args:
        bench: The ``benchmark`` section of the config.
        inferencex_path: InferenceX checkout root.
        runner_type: Resolved runner type (e.g. mi300x).

    Returns:
        The script path relative to the InferenceX root.

    Raises:
        FileNotFoundError: When no suitable script is found.
    """
    benchmarks_dir = inferencex_path / "benchmarks"
    framework = str(bench.get("framework") or "").lower()
    precision = str(bench.get("precision") or "").lower()

    explicit = str(bench.get("benchmark_script") or "").strip()
    if explicit:
        found = _find_script(benchmarks_dir, explicit)
        if not found:
            raise FileNotFoundError(f"benchmark_script not found: {explicit}")
        return str(found.relative_to(inferencex_path))

    prefix = _NATIVE_PREFIX.get(framework)
    if prefix and precision:
        native = f"{prefix}_{precision}_{runner_type}.sh"
        found = _find_script(benchmarks_dir, native)
        if found:
            return str(found.relative_to(inferencex_path))

    generic = f"{framework}_{runner_type}.sh"
    if (benchmarks_dir / generic).exists():
        return f"benchmarks/{generic}"

    raise FileNotFoundError(
        f"No benchmark script for framework={framework} precision={precision} "
        f"runner={runner_type} under {benchmarks_dir}"
    )


def _build_env(bench: dict[str, Any], runner_type: str, workspace: Path) -> dict[str, str]:
    """Build the subprocess env, mirroring Magpie's local benchmark contract.

    The parent process env is inherited; the benchmark YAML's ``model`` /
    ``precision`` / ``envs`` win, plus the RESULT/SERVER/PROFILE wiring the
    InferenceX scripts expect.

    Args:
        bench: The ``benchmark`` section of the config.
        runner_type: Resolved runner type.
        workspace: Per-run workspace directory.

    Returns:
        The environment mapping for the benchmark subprocess.
    """
    env = os.environ.copy()
    env["MODEL"] = str(bench.get("model") or env.get("MODEL", ""))
    if bench.get("precision"):
        env["PRECISION"] = str(bench["precision"])
    for key, value in (bench.get("envs") or {}).items():
        env[str(key).upper()] = str(value)
    env["RUNNER_TYPE"] = runner_type
    env["RESULT_FILENAME"] = "inferencex_result"
    env["RESULT_DIR"] = str(workspace)
    env["MAGPIE_RUN_PHASE"] = "all"
    env["SERVER_LOG"] = str(workspace / "server.log")

    profiler = (bench.get("profiler") or {}).get("torch_profiler") or {}
    if profiler.get("enabled"):
        trace_dir = workspace / "torch_trace"
        trace_dir.mkdir(parents=True, exist_ok=True)
        env["PROFILE"] = "1"
        env["VLLM_TORCH_PROFILER_DIR"] = str(trace_dir)
        env["SGLANG_TORCH_PROFILER_DIR"] = str(trace_dir)
        env["ATOM_TORCH_PROFILER_DIR"] = str(trace_dir)
    return env


def run_benchmark(config_path: Path, output_dir: Path) -> int:
    """Run one local benchmark by reusing the InferenceX scripts.

    Args:
        config_path: Materialized benchmark config YAML.
        output_dir: Output root for the workspace.

    Returns:
        Process exit code (0 on success).
    """
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    bench = cfg.get("benchmark") or {}
    framework = str(bench.get("framework") or "sglang").lower()
    model = str(bench.get("model") or os.environ.get("MODEL", ""))
    timeout_s = float(bench.get("timeout_seconds") or 3600.0)

    inferencex_path = _resolve_inferencex_path(bench)
    if not inferencex_path or not Path(inferencex_path).is_dir():
        _emit_failure(
            output_dir, framework, model,
            f"InferenceX path not resolvable/usable: {inferencex_path!r}",
        )
        return 2
    inferencex_root = Path(inferencex_path).resolve()

    runner_type = str(bench.get("runner_type") or os.environ.get("RUNNER_TYPE") or "mi300x").lower()

    _sync_scripts(inferencex_root)
    try:
        script_rel = _resolve_script(bench, inferencex_root, runner_type)
    except FileNotFoundError as exc:
        _emit_failure(output_dir, framework, model, str(exc))
        return 2

    workspace = bypass_report.create_workspace(output_dir, framework)
    _snapshot_config(workspace, cfg)
    env = _build_env(bench, runner_type, workspace)

    cmd = ["bash", "-c", f"cd {inferencex_root} && bash {script_rel}"]
    start = time.time()
    try:
        proc = subprocess.run(
            cmd, env=env, capture_output=True, text=True, timeout=timeout_s,
        )
        returncode = proc.returncode
        _save_logs(workspace, proc.stdout or "", proc.stderr or "")
    except subprocess.TimeoutExpired as exc:
        _save_logs(workspace, getattr(exc, "stdout", "") or "", getattr(exc, "stderr", "") or "")
        _write_report(workspace, framework, model, False, start, [f"benchmark timed out after {timeout_s}s"])
        return 124

    raw = _load_raw_result(workspace)
    success = returncode == 0 and raw is not None
    errors: list[str] = []
    if returncode != 0:
        errors.append(f"benchmark process exited {returncode}")
    if raw is None:
        errors.append("inferencex_result.json not produced")
    _write_report(workspace, framework, model, success, start, errors, raw=raw)
    return 0 if success else (returncode or 1)


def _snapshot_config(workspace: Path, cfg: dict[str, Any]) -> None:
    """Persist the effective config into the workspace (best-effort)."""
    try:
        (workspace / "config.yaml").write_text(
            yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8",
        )
    except OSError:
        pass


def _load_raw_result(workspace: Path) -> dict[str, Any] | None:
    """Load ``inferencex_result.json`` from the workspace, if present."""
    path = workspace / "inferencex_result.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _save_logs(workspace: Path, stdout: str, stderr: str) -> None:
    """Persist benchmark stdout/stderr for debugging (best-effort)."""
    try:
        if stdout:
            (workspace / "benchmark_stdout.log").write_text(stdout, encoding="utf-8")
        if stderr:
            (workspace / "benchmark_stderr.log").write_text(stderr, encoding="utf-8")
    except OSError:
        pass


def _write_report(
    workspace: Path,
    framework: str,
    model: str,
    success: bool,
    start: float,
    errors: list[str],
    *,
    raw: dict[str, Any] | None = None,
) -> None:
    """Build and write the Magpie-compatible report."""
    report = bypass_report.build_report(
        raw,
        framework=framework,
        model=model,
        success=success,
        workspace_dir=str(workspace),
        execution_time=time.time() - start,
        errors=errors,
    )
    bypass_report.write_report(workspace, report)


def _emit_failure(output_dir: Path, framework: str, model: str, error: str) -> None:
    """Emit a failing report + workspace for a pre-launch error."""
    workspace = bypass_report.create_workspace(output_dir, framework)
    _write_report(workspace, framework, model, False, time.time(), [error])


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the bypass CLI parser (Magpie-compatible flags)."""
    parser = argparse.ArgumentParser(prog="hyperloom-bypass-benchmark")
    sub = parser.add_subparsers(dest="mode", required=True)
    bench = sub.add_parser("benchmark", help="Run a framework benchmark")
    bench.add_argument("--benchmark-config", required=True)
    bench.add_argument("--output-dir", required=True)
    bench.add_argument("--run-mode", default="local")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Optional argument vector (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code.
    """
    args = _build_arg_parser().parse_args(argv)
    if args.mode != "benchmark":
        print(f"unsupported mode: {args.mode}", file=sys.stderr)
        return 2
    if args.run_mode != "local":
        print(
            f"bypass runner supports --run-mode local only, got {args.run_mode}",
            file=sys.stderr,
        )
        return 2
    return run_benchmark(Path(args.benchmark_config), Path(args.output_dir))


if __name__ == "__main__":
    raise SystemExit(main())