#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""GEAK e2e optimizer submission (whole-pipeline; GEAK@GEAK main)."""

from __future__ import annotations

import ast
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import IO

from hyperloom.common.proctree import signal_group

_TAIL_CHARS = 4000


def _tail(stream: IO[bytes]) -> str:
    """The last :data:`_TAIL_CHARS` characters written to ``stream``."""
    stream.seek(0, os.SEEK_END)
    stream.seek(max(0, stream.tell() - 4 * _TAIL_CHARS))
    return stream.read().decode("utf-8", "replace")[-_TAIL_CHARS:]


def _resolve_runner() -> str:
    """Resolve run_e2e.py from $GEAK_E2E_RUNNER / $GEAK_ROOT (GEAK@GEAK)."""
    runner = os.environ.get("GEAK_E2E_RUNNER", "").strip()
    if runner and Path(runner).is_file():
        return runner
    roots: list[str] = []
    root = os.environ.get("GEAK_ROOT", "").strip()
    if root:
        roots.append(root)
    cache_dir = os.environ.get("HYPERLOOM_CACHE_DIR", "").strip()
    if cache_dir:
        # install.sh clones GEAK per revision as <cache>/GEAK@<sha>; prefer the newest such checkout, then the bare
        # dir.
        pinned = sorted(
            (p for p in Path(cache_dir).glob("GEAK@*") if p.is_dir()),
            key=lambda p: p.stat().st_mtime if p.exists() else 0.0,
            reverse=True,
        )
        roots.extend(str(p) for p in pinned)
        roots.append(str(Path(cache_dir) / "GEAK"))
    for root in dict.fromkeys(roots):
        cand = Path(root) / "interface" / "run_e2e.py"
        if cand.is_file():
            return str(cand)
    raise FileNotFoundError(
        "e2e runner not found. Set GEAK_E2E_RUNNER to "
        "<GEAK checkout>/interface/run_e2e.py (the installer exports it), "
        "or set GEAK_ROOT/HYPERLOOM_CACHE_DIR."
    )


def _source_reader_error(handoff: dict, runner: str) -> dict | None:
    """Refuse source-bearing handoffs an installed reader cannot reproduce."""
    baseline = handoff.get("baseline_env_spec") or {}
    if baseline.get("source_snapshots") and "source_materialization" not in baseline:
        return {
            "status": "error",
            "error_class": "unresolved_baseline_source",
            "error": "Accepted source layers require a complete source materialization before GEAK launches",
        }
    if "source_materialization" in baseline and not _supports_source_materialization(Path(runner)):
        return {
            "status": "error",
            "error_class": "unsupported_source_materialization_reader",
            "error": "GEAK must advertise SOURCE_MATERIALIZATION_SCHEMA_VERSION = 1 for accepted source trees",
        }
    return None


def call_geak(handoff: dict, output_dir: Path, *, timeout_s: int) -> dict:
    """Run GEAK e2e once and return the parsed result.json (+ run metadata)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    handoff_path = output_dir / "handoff.json"
    result_path = output_dir / "result.json"
    handoff_path.write_text(json.dumps(handoff, indent=2), encoding="utf-8")

    runner = _resolve_runner()
    rejected = _source_reader_error(handoff, runner)
    if rejected is not None:
        result_path.write_text(json.dumps(rejected, indent=2), encoding="utf-8")
        return rejected
    cmd = [sys.executable, runner, str(handoff_path), str(result_path)]

    env = dict(os.environ)
    # ``timeout_s`` is authoritative: run_e2e.py reads GEAK_E2E_TIMEOUT_S to self-stop before the outer subprocess
    # kill.
    grace_raw = os.environ.get("GEAK_FLUSH_GRACE_S", "").strip()
    flush_grace = int(grace_raw) if grace_raw.isdigit() and int(grace_raw) > 0 else 180
    inner_timeout = max(60, timeout_s - flush_grace)
    env["GEAK_E2E_TIMEOUT_S"] = str(inner_timeout)  # run_e2e's anyio budget

    started = time.time()
    # Files, not pipes: a process run_e2e leaves behind still holds its stdio, and must not keep this wait open.
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        # start_new_session=True -> run_e2e + its vllm/node children share a process group we can signal as a unit.
        proc = subprocess.Popen(cmd, stdout=stdout, stderr=stderr, env=env, start_new_session=True)

        try:
            returncode = proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            # SIGTERM lets run_e2e flush result.json, then escalate to SIGKILL.
            signal_group(proc.pid, signal.SIGTERM, what="geak")
            try:
                returncode = proc.wait(timeout=flush_grace)
            except subprocess.TimeoutExpired:
                signal_group(proc.pid, signal.SIGKILL, what="geak")
                proc.wait()
                returncode = -1
        stdout_tail = _tail(stdout)
        stderr_tail = _tail(stderr)

    result: dict = {}
    if result_path.is_file():
        try:
            _parsed = json.loads(result_path.read_text(encoding="utf-8"))
            if isinstance(_parsed, dict):
                result = _parsed
        except json.JSONDecodeError:
            pass
    if not result:
        result = {
            "status": "error",
            "error": f"no parseable result.json (rc={returncode})",
        }
    result.update(
        {
            "returncode": returncode,
            "stdout_tail": stdout_tail,
            "stderr_tail": stderr_tail,
            "elapsed_s": round(time.time() - started, 2),
            "handoff_path": str(handoff_path),
            "result_path": str(result_path),
        }
    )
    return result


def _supports_source_materialization(runner: Path) -> bool:
    """Read the consumer's literal protocol capability without importing it."""
    try:
        module = ast.parse(runner.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeError):
        return False
    values = []
    for statement in module.body:
        if isinstance(statement, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "SOURCE_MATERIALIZATION_SCHEMA_VERSION"
            for target in statement.targets
        ):
            values.append(statement.value)
    return (
        len(values) == 1
        and isinstance(values[0], ast.Constant)
        and type(values[0].value) is int
        and values[0].value == 1
    )


def main(argv: list[str]) -> int:
    """CLI: ``python -m hyperloom.orchestrator.kernel.geak_runner <handoff.json> <output_dir> --timeout-s N``."""
    import argparse

    ap = argparse.ArgumentParser(description="Run GEAK e2e once.")
    ap.add_argument("handoff_json")
    ap.add_argument("output_dir")
    ap.add_argument("--timeout-s", type=int, required=True)
    args = ap.parse_args(argv)

    handoff = json.loads(Path(args.handoff_json).read_text(encoding="utf-8"))
    out = call_geak(handoff, Path(args.output_dir), timeout_s=args.timeout_s)
    print(
        json.dumps(
            {
                "status": out.get("status"),
                "speedup": out.get("throughput_speedup"),
                "result_path": out.get("result_path"),
            }
        )
    )
    return 0 if out.get("status") not in ("error", None) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
