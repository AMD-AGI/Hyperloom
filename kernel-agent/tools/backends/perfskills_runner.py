#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""PerfSkills e2e optimizer submission.

PerfSkills is a WHOLE-pipeline e2e optimizer (not a per-kernel backend like
GEAK). Hyperloom invokes it ONCE at the KERNEL phase via the stable
``interface/run_e2e.py`` contract: we write a ``handoff.json`` (Hyperloom best
config + workload), call the runner, and read back a ``result.json`` (optimized
launch script + bench script + throughput + per-kernel report).

All Claude-SDK / Workflow / ``--effort`` detail lives INSIDE PerfSkills'
``interface/run_e2e.py``; this module only marshals the two JSON files and the
subprocess. See PerfSkills/interface/run_e2e.md for the contract.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

HANDOFF_SCHEMA_VERSION = 1


def _resolve_runner() -> str:
    """Resolve PerfSkills' run_e2e.py from $PERFSKILLS_E2E_RUNNER / $PERFSKILLS_ROOT."""
    runner = os.environ.get("PERFSKILLS_E2E_RUNNER", "").strip()
    if runner and Path(runner).is_file():
        return runner
    root = os.environ.get("PERFSKILLS_ROOT", "").strip()
    if root:
        cand = Path(root) / "interface" / "run_e2e.py"
        if cand.is_file():
            return str(cand)
    raise FileNotFoundError(
        "PerfSkills runner not found. Set PERFSKILLS_E2E_RUNNER to "
        "<PerfSkills>/interface/run_e2e.py (the installer exports it)."
    )


def build_handoff(
    *,
    model_path: str,
    framework: str,
    gpu_type: str,
    tp: int,
    workload: dict,
    accepted_flags: str = "",
    accepted_env: str = "",
    launch_recipe: str = "",
    raw_baseline_tput: float = 0.0,
    exp_root: str,
    gpu_ids: str = "",
    bench_client: str = "auto",
    inferencex_path: str = "",
) -> dict:
    """Assemble the stable handoff.json payload (Hyperloom -> PerfSkills)."""
    h = {
        "schema_version": HANDOFF_SCHEMA_VERSION,
        "model_path": model_path,
        "framework": framework or "sglang",
        "gpu_type": gpu_type or "",
        "tp": int(tp or 1),
        "workload": {
            "isl": int(workload.get("isl", 1024)),
            "osl": int(workload.get("osl", 1024)),
            "conc": int(workload.get("conc", 64)),
        },
        "accepted_flags": accepted_flags or "",
        "accepted_env": accepted_env or "",
        "raw_baseline_tput": float(raw_baseline_tput or 0.0),
        "exp_root": exp_root,
        # Use the SAME bench client as Hyperloom (InferenceX benchmark_serving.py)
        # so PerfSkills numbers are cross-harness comparable. "auto" lets the
        # runner pick inferencex when an InferenceX checkout is discoverable.
        "bench_client": bench_client or "auto",
        "inferencex_path": inferencex_path or os.environ.get("INFERENCEX_PATH", ""),
    }
    if launch_recipe:
        h["launch_recipe"] = launch_recipe
    if gpu_ids:
        h["gpu_ids"] = gpu_ids
    return h


def call_perfskills(handoff: dict, output_dir: Path, *, timeout_s: int = 21600,
                    python_bin: str = "") -> dict:
    """Run PerfSkills e2e once and return the parsed result.json (+ run metadata).

    Args:
        handoff: handoff payload (see :func:`build_handoff`).
        output_dir: where handoff.json / result.json are written.
        timeout_s: subprocess timeout (default 6h).
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
    env.setdefault("PERFSKILLS_E2E_TIMEOUT_S", str(timeout_s))

    started = time.time()
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_s, env=env,
        )
        returncode = proc.returncode
        stdout_tail = (proc.stdout or "")[-4000:]
        stderr_tail = (proc.stderr or "")[-4000:]
    except subprocess.TimeoutExpired as e:
        return {
            "status": "error", "error": f"PerfSkills timed out after {timeout_s}s",
            "returncode": -1, "stdout_tail": str(e.stdout or "")[-4000:],
            "stderr_tail": str(e.stderr or "")[-4000:],
            "elapsed_s": round(time.time() - started, 2),
            "handoff_path": str(handoff_path), "result_path": str(result_path),
        }

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
    """CLI: perfskills_runner.py <handoff.json> <output_dir> [--timeout-s N]."""
    import argparse

    ap = argparse.ArgumentParser(description="Run PerfSkills e2e once.")
    ap.add_argument("handoff_json")
    ap.add_argument("output_dir")
    ap.add_argument("--timeout-s", type=int, default=21600)
    args = ap.parse_args(argv)

    handoff = json.loads(Path(args.handoff_json).read_text(encoding="utf-8"))
    out = call_perfskills(handoff, Path(args.output_dir), timeout_s=args.timeout_s)
    print(json.dumps({"status": out.get("status"),
                      "speedup": out.get("throughput_speedup"),
                      "result_path": out.get("result_path")}))
    return 0 if out.get("status") not in ("error", None) else 1


if __name__ == "__main__":
    import sys

    raise SystemExit(_main(sys.argv[1:]))
