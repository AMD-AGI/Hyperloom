#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""GEAK e2e optimizer submission (whole-pipeline; GEAK@GEAK main).

This is a WHOLE-pipeline e2e optimizer (not a per-kernel backend like
``forge``). Its code lives in GEAK
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
    roots: list[str] = []
    root = os.environ.get("GEAK_ROOT", "").strip()
    if root:
        roots.append(root)
    open_source_root = os.environ.get("HYPERLOOM_OPEN_SOURCE_ROOT", "").strip()
    if open_source_root:
        roots.append(str(Path(open_source_root) / "GEAK"))
    roots.append("/opt/hyperloom/open-source-repos/GEAK")
    for root in dict.fromkeys(roots):
        cand = Path(root) / "interface" / "run_e2e.py"
        if cand.is_file():
            return str(cand)
    raise FileNotFoundError(
        "e2e runner not found. Set GEAK_E2E_RUNNER to "
        "<GEAK checkout>/interface/run_e2e.py (the installer exports it), "
        "or set GEAK_ROOT/HYPERLOOM_OPEN_SOURCE_ROOT."
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
    # ``timeout_s`` is authoritative: run_e2e.py reads GEAK_E2E_TIMEOUT_S to
    # self-stop before the outer subprocess kill. Split the inner SOFT deadline
    # from the outer HARD kill so run_e2e can flush result.json before SIGKILL.
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
        # SIGTERM lets run_e2e flush result.json, then escalate to SIGKILL.
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
    # Explicit --timeout-s wins; else fall back to GEAK_E2E_TIMEOUT_S, else 12h.
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
