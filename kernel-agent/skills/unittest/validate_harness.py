#!/usr/bin/env python3
"""Validate a GEAK-compatible test harness (static + runtime checks).

Usage:
    python validate_harness.py <harness_path> --static
    python validate_harness.py <harness_path> --run
    python validate_harness.py <harness_path> --all

Exit 0 = valid, exit 1 = invalid.  JSON report on stdout.
"""

import argparse
import ast
import json
import os
import subprocess
import sys
import time

REQUIRED_FLAGS = ("--correctness", "--profile", "--benchmark", "--full-benchmark")
REQUIRED_MARKERS = ("GEAK_SHAPES_USED",)
BENCHMARK_MARKERS = ("GEAK_RESULT_LATENCY_MS",)

GEAK_CONTRACT_KEYWORDS = {
    "GEAK_WORK_DIR": "GEAK places patched candidates in GEAK_WORK_DIR; harness must add it to sys.path first",
    "GEAK_BENCHMARK_ITERATIONS": "GEAK controls iteration count via this env var; harness must read it",
}

WALL_CLOCK_TIMING_PATTERNS = ("time.perf_counter", "time.time()", "time.monotonic")
GPU_EVENT_KEYWORDS = ("torch.cuda.Event", "enable_timing")

MODE_TIMEOUT = int(os.environ.get("HARNESS_VALIDATE_TIMEOUT", "300"))
BENCHMARK_ITERATIONS_OVERRIDE = 5


def _extract_string_literals(source: str) -> set[str]:
    """Extract all string literal values from Python source via AST.

    Args:
        source (str): Python source text to parse.

    Returns:
        set[str]: Every string-constant value found in the AST; an empty
            set when the source fails to parse.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    literals: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            literals.add(node.value)
    return literals


def _has_flag_in_code(source: str, flag: str, string_literals: set[str]) -> bool:
    """Check if a CLI flag appears in actual code (string literals), not just comments.

    Args:
        source (str): The harness source (unused; kept for symmetry).
        flag (str): The CLI flag substring to look for.
        string_literals (set[str]): String literals extracted from the
            source via :func:`_extract_string_literals`.

    Returns:
        bool: True if any string literal contains ``flag``.
    """
    for lit in string_literals:
        if flag in lit:
            return True
    return False


def _has_marker_in_code(source: str, marker: str) -> bool:
    """Check if an output marker appears in non-comment code lines.

    Args:
        source (str): The harness source to scan line by line.
        marker (str): The output-marker substring to look for.

    Returns:
        bool: True if ``marker`` appears on a line that is not a comment.
    """
    for line in source.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("#"):
            continue
        if marker in line:
            return True
    return False


def static_check(harness_path: str) -> tuple[bool, list[str]]:
    """Statically validate a harness against the GEAK contract.

    Parses the file and checks for the required CLI flags, output
    markers, GEAK contract keywords, and GPU-event timing, without
    executing the harness.

    Args:
        harness_path (str): Path to the harness ``.py`` file.

    Returns:
        tuple[bool, list[str]]: ``(ok, errors)`` where ``ok`` is True
            only when ``errors`` is empty; each error string describes a
            missing requirement or a syntax/IO failure.
    """
    errors: list[str] = []

    if not os.path.isfile(harness_path):
        return False, [f"File not found: {harness_path}"]

    with open(harness_path) as f:
        source = f.read()

    try:
        ast.parse(source)
    except SyntaxError as e:
        errors.append(f"Syntax error: {e}")
        return False, errors

    string_literals = _extract_string_literals(source)

    has_argparse = any(
        kw in lit for lit in string_literals
        for kw in ("argparse",)
    ) or "ArgumentParser" in source
    if not has_argparse:
        # Double-check: argparse might be imported but not as a string literal
        if "import argparse" not in source and "ArgumentParser" not in source:
            errors.append("Missing argparse / ArgumentParser")

    for flag in REQUIRED_FLAGS:
        if not _has_flag_in_code(source, flag, string_literals):
            errors.append(f"Missing CLI flag: {flag} (must be in a string literal, not a comment)")

    for marker in REQUIRED_MARKERS:
        if not _has_marker_in_code(source, marker):
            errors.append(f"Missing output marker: {marker}")

    for marker in BENCHMARK_MARKERS:
        if not _has_marker_in_code(source, marker):
            errors.append(f"Missing benchmark marker: {marker}")

    for keyword, reason in GEAK_CONTRACT_KEYWORDS.items():
        if keyword not in source:
            errors.append(f"Missing GEAK contract keyword: {keyword} — {reason}")

    has_gpu_events = any(kw in source for kw in GPU_EVENT_KEYWORDS)
    has_wall_clock = any(pat in source for pat in WALL_CLOCK_TIMING_PATTERNS)
    if not has_gpu_events:
        errors.append(
            "Missing GPU event timing (torch.cuda.Event with enable_timing=True). "
            "Wall-clock timing (time.perf_counter) is inaccurate for async GPU kernels"
        )
    if has_wall_clock and not has_gpu_events:
        errors.append(
            "Uses wall-clock timing for benchmark but no GPU events. "
            "Replace time.perf_counter/time.time with torch.cuda.Event"
        )

    return len(errors) == 0, errors


def _run_mode(harness_path: str, mode: str, env: dict) -> dict:
    """Execute the harness in one mode and inspect its output.

    Runs ``python <harness> --<mode>`` with a timeout, then checks the
    exit code and stdout for the expected GEAK markers.

    Args:
        harness_path (str): Path to the harness ``.py`` file.
        mode (str): Mode flag to pass (e.g. ``correctness``,
            ``benchmark``).
        env (dict): Environment for the subprocess.

    Returns:
        dict: A result mapping with ``mode``, ``ok``, ``returncode``,
            ``elapsed_s``, ``errors``, ``stdout_tail``, and
            ``stderr_tail`` keys.
    """
    cmd = [sys.executable, harness_path, f"--{mode}"]
    start = time.time()
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=MODE_TIMEOUT, env=env,
        )
        elapsed = time.time() - start
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        ok = proc.returncode == 0

        has_shapes = "GEAK_SHAPES_USED=" in stdout
        has_latency = "GEAK_RESULT_LATENCY_MS=" in stdout

        mode_errors = []
        if not ok:
            tail = stderr[-800:] if stderr else stdout[-800:]
            mode_errors.append(f"exit code {proc.returncode}: {tail}")
        if not has_shapes:
            mode_errors.append("Missing GEAK_SHAPES_USED marker in stdout")
        if mode in ("benchmark", "full-benchmark") and not has_latency:
            mode_errors.append("Missing GEAK_RESULT_LATENCY_MS marker in stdout")

        return {
            "mode": mode,
            "ok": ok and not mode_errors,
            "returncode": proc.returncode,
            "elapsed_s": round(elapsed, 2),
            "errors": mode_errors,
            "stdout_tail": stdout[-500:] if stdout else "",
            "stderr_tail": stderr[-500:] if stderr else "",
        }
    except subprocess.TimeoutExpired:
        return {
            "mode": mode,
            "ok": False,
            "returncode": -1,
            "elapsed_s": MODE_TIMEOUT,
            "errors": [f"Timed out after {MODE_TIMEOUT}s"],
            "stdout_tail": "",
            "stderr_tail": "",
        }
    except Exception as e:
        return {
            "mode": mode,
            "ok": False,
            "returncode": -1,
            "elapsed_s": time.time() - start,
            "errors": [str(e)],
            "stdout_tail": "",
            "stderr_tail": "",
        }


def runtime_check(harness_path: str) -> tuple[bool, list[str], list[dict]]:
    """Run the harness in correctness and benchmark modes.

    Forces a small benchmark-iteration count and runs the modes in
    sequence, stopping at the first failing mode.

    Args:
        harness_path (str): Path to the harness ``.py`` file.

    Returns:
        tuple[bool, list[str], list[dict]]: ``(ok, errors, results)``
            where ``ok`` is True only when no mode errored, ``errors``
            are mode-prefixed messages, and ``results`` are the per-mode
            dicts from :func:`_run_mode`.
    """
    env = os.environ.copy()
    env["GEAK_BENCHMARK_ITERATIONS"] = str(BENCHMARK_ITERATIONS_OVERRIDE)

    results = []
    all_errors: list[str] = []

    for mode in ("correctness", "benchmark"):
        r = _run_mode(harness_path, mode, env)
        results.append(r)
        if not r["ok"]:
            all_errors.extend(f"[{mode}] {e}" for e in r["errors"])
            break

    return len(all_errors) == 0, all_errors, results


def main():
    """CLI entry point: validate a harness and print a JSON report.

    Parses ``--static`` / ``--run`` / ``--all``, runs the requested
    checks, prints the report to stdout, and exits 0 when valid or 1
    otherwise (via :func:`sys.exit`).
    """
    parser = argparse.ArgumentParser(description="Validate GEAK test harness")
    parser.add_argument("harness_path", help="Path to harness .py file")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--static", action="store_true", help="Static checks only")
    group.add_argument("--run", action="store_true", help="Runtime checks only")
    group.add_argument("--all", action="store_true", help="Static + runtime")
    args = parser.parse_args()

    report: dict = {"harness": args.harness_path, "valid": False, "errors": []}

    if args.static or args.all:
        ok, errs = static_check(args.harness_path)
        report["static"] = {"ok": ok, "errors": errs}
        report["errors"].extend(errs)

        if not ok and not args.all:
            report["valid"] = False
            print(json.dumps(report, indent=2))
            sys.exit(1)

    if args.run or args.all:
        if args.all and not report.get("static", {}).get("ok", True):
            report["runtime"] = {"ok": False, "errors": ["Skipped: static check failed"]}
        else:
            ok, errs, results = runtime_check(args.harness_path)
            report["runtime"] = {"ok": ok, "errors": errs, "modes": results}
            report["errors"].extend(errs)

    report["valid"] = len(report["errors"]) == 0
    print(json.dumps(report, indent=2))
    sys.exit(0 if report["valid"] else 1)


if __name__ == "__main__":
    main()
