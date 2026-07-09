#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""GEAK e2e optimizer submission (whole-pipeline; GEAK@GEAK main).

This is a WHOLE-pipeline e2e optimizer (not a per-kernel backend like the legacy
per-kernel GEAK single-kernel loop, now ``geak_v3``). Its code lives in GEAK
(``interface/run_e2e.py`` + ``e2e_workflow/``). Hyperloom invokes it ONCE at the
KERNEL_AGENT phase via the stable ``interface/run_e2e.py`` contract: we write a
``handoff.json`` (Hyperloom best config + workload), call the runner, and read
back a ``result.json`` (optimized launch script + bench script + throughput +
per-kernel report).

All Claude-SDK / Workflow / ``--effort`` detail lives INSIDE the optimizer's
``interface/run_e2e.py``; this module only marshals the two JSON files and the
subprocess. See GEAK ``interface/run_e2e.md`` for the contract. The
GEAK_* env-var / function names are the stable handle.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path


def _resolve_runner() -> str:
    """Resolve run_e2e.py from $GEAK_E2E_RUNNER / $GEAK_ROOT (GEAK@GEAK)."""
    runner = os.environ.get("GEAK_E2E_RUNNER", "").strip()
    if runner and Path(runner).is_file():
        return runner
    root = os.environ.get("GEAK_ROOT", "").strip()
    if root:
        cand = Path(root) / "interface" / "run_e2e.py"
        if cand.is_file():
            return str(cand)
    raise FileNotFoundError(
        "e2e runner not found. Set GEAK_E2E_RUNNER to "
        "<GEAK checkout>/interface/run_e2e.py (the installer exports it)."
    )


def call_geak(handoff: dict, output_dir: Path, *, timeout_s: int = 43200,
              python_bin: str = "") -> dict:
    """Run GEAK e2e once and return the parsed result.json (+ run metadata).

    Args:
        handoff: handoff.json payload (Hyperloom best config + workload).
        output_dir: where handoff.json / result.json are written.
        timeout_s: subprocess timeout (default 12h).
        python_bin: python interpreter for the runner (default the current one).

    Returns:
        dict: the normalized result.json plus ``returncode`` /
        ``stdout_tail`` / ``stderr_tail`` / ``elapsed_s`` / ``handoff_path`` /
        ``result_path``. On failure ``status == "error"``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    handoff_path = output_dir / "handoff.json"
    result_path = output_dir / "result.json"
    handoff_path.write_text(json.dumps(handoff, indent=2), encoding="utf-8")

    runner = _resolve_runner()
    py = python_bin or shutil.which("python3") or "python3"
    cmd = [py, runner, str(handoff_path), str(result_path)]

    env = dict(os.environ)
    # The resolved ``timeout_s`` is AUTHORITATIVE for the delegated director run:
    # run_e2e.py reads GEAK_E2E_TIMEOUT_S to self-stop (anyio.fail_after)
    # before our outer subprocess kill. Assign (not setdefault) so a value
    # inherited from the parent env (e.g. a stale export / ray passthrough) can
    # never override the caller's budget — when Hyperloom drives, GEAK's
    # time MUST come from Hyperloom (the --timeout-s it passes). Standalone runs
    # resolve timeout_s to the 12h default (or an explicit env, see _main).
    # Split the inner SOFT deadline from the outer HARD kill so run_e2e can
    # self-stop (anyio.fail_after) and FLUSH result.json (recover-from-disk)
    # before we SIGKILL. Previously both were timeout_s, so the flush was killed
    # mid-write -> "no_result_json" and the measured win was lost.
    flush_grace = int(os.environ.get("GEAK_FLUSH_GRACE_S", "180"))
    inner_timeout = max(60, timeout_s - flush_grace)
    env["GEAK_E2E_TIMEOUT_S"] = str(inner_timeout)  # run_e2e's anyio budget

    started = time.time()
    # start_new_session=True -> run_e2e + its vllm/node children share a process
    # group we can signal as a unit (prevents leaked-server orphans).
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        env=env, start_new_session=True,
    )

    def _killpg(sig: int) -> None:
        try:
            os.killpg(os.getpgid(proc.pid), sig)
        except (ProcessLookupError, PermissionError):
            # Process already exited; nothing to signal.
            pass

    try:
        stdout, stderr = proc.communicate(timeout=timeout_s)
        returncode = proc.returncode
    except subprocess.TimeoutExpired:
        # Be polite first: SIGTERM lets run_e2e's handler flush result.json,
        # then escalate to SIGKILL if it overruns the grace window.
        _killpg(signal.SIGTERM)
        try:
            stdout, stderr = proc.communicate(timeout=flush_grace)
            returncode = proc.returncode
        except subprocess.TimeoutExpired:
            _killpg(signal.SIGKILL)
            stdout, stderr = proc.communicate()
            returncode = -1
    stdout_tail = (stdout or "")[-4000:]
    stderr_tail = (stderr or "")[-4000:]

    result: dict = {}
    if result_path.is_file():
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            result = {}
    if not result:
        result = {
            "status": "error",
            "error": f"no parseable result.json (rc={returncode})",
        }
    result.update({
        "returncode": returncode,
        "stdout_tail": stdout_tail,
        "stderr_tail": stderr_tail,
        "elapsed_s": round(time.time() - started, 2),
        "handoff_path": str(handoff_path),
        "result_path": str(result_path),
    })
    return result


def _main(argv: list[str]) -> int:
    """CLI: geak_runner.py <handoff.json> <output_dir> [--timeout-s N]."""
    import argparse

    ap = argparse.ArgumentParser(description="Run GEAK e2e once.")
    ap.add_argument("handoff_json")
    ap.add_argument("output_dir")
    # Sentinel default: an explicit --timeout-s (Hyperloom always passes one) is
    # authoritative. Without it (standalone runner invocation) fall back to an
    # explicitly-exported GEAK_E2E_TIMEOUT_S, else the 12h default.
    ap.add_argument("--timeout-s", type=int, default=None)
    args = ap.parse_args(argv)

    if args.timeout_s is not None:
        timeout_s = args.timeout_s
    else:
        timeout_s = int(os.environ.get("GEAK_E2E_TIMEOUT_S", "43200"))  # 12h

    handoff = json.loads(Path(args.handoff_json).read_text(encoding="utf-8"))
    out = call_geak(handoff, Path(args.output_dir), timeout_s=timeout_s)
    print(json.dumps({"status": out.get("status"),
                      "speedup": out.get("throughput_speedup"),
                      "result_path": out.get("result_path")}))
    return 0 if out.get("status") not in ("error", None) else 1


if __name__ == "__main__":
    import sys

    raise SystemExit(_main(sys.argv[1:]))
