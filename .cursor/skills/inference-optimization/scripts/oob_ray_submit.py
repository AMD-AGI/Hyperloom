#!/usr/bin/env python3
"""OOB Ray Submit — schedule oob CLI tasks via local Ray cluster.

Provides GPU isolation for concurrent OOB (Claude Code / Codex) tasks
in local mode. Each task gets an isolated GPU allocated by Ray.

Usage:
    # Single task (blocking, prints result when done)
    python oob_ray_submit.py run -a claude -p "Optimize..." -f kernel.py

    # Status: check Ray cluster resources
    python oob_ray_submit.py status

Extra args after known flags are forwarded to the `oob run` CLI.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

try:
    import ray
except ImportError:
    print("Error: ray is not installed. It should be bundled in the sglang/vllm base image.")
    sys.exit(1)


RAY_ADDRESS = os.environ.get("RAY_ADDRESS", "auto")
DEFAULT_NUM_GPUS = int(os.environ.get("OOB_GPUS_PER_TASK", "1"))

# Env vars that OOB / litellm / providers need; forwarded to every Ray worker.
_FORWARDED_ENV_KEYS = (
    "OOB_API_KEY", "OOB_BASE_URL", "OOB_LOCAL", "OOB_CLI", "OOB_HOME",
    "ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL",
    "OPENAI_API_KEY", "OPENAI_BASE_URL",
    "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES",
    "PATH", "HOME", "LD_LIBRARY_PATH", "AGENT_WORKSPACE_ROOT",
)

_PATH_FLAGS = {"-f", "--file", "--prompt-file", "--files-from", "-o", "--output-dir"}
_PATH_FLAG_PREFIXES = (
    "--file=",
    "--prompt-file=",
    "--files-from=",
    "--output-dir=",
)


def _build_runtime_env() -> dict:
    """Forward OOB/GPU env vars to Ray workers."""
    env = {k: os.environ[k] for k in _FORWARDED_ENV_KEYS if k in os.environ}
    return {"env_vars": env}


def _resolve_path_arg(path_arg: str) -> str:
    """Resolve CLI path args on the driver before sending work to Ray."""
    return str(Path(path_arg).expanduser().resolve())


def _normalize_extra_args(extra_args: list[str]) -> list[str]:
    """Resolve path-bearing CLI arguments so workers can access the same files."""
    normalized: list[str] = []
    expect_path_value = False

    for arg in extra_args:
        if expect_path_value:
            normalized.append(_resolve_path_arg(arg))
            expect_path_value = False
            continue

        replaced = False
        for prefix in _PATH_FLAG_PREFIXES:
            if arg.startswith(prefix):
                normalized.append(f"{prefix}{_resolve_path_arg(arg.split('=', 1)[1])}")
                replaced = True
                break
        if replaced:
            continue

        normalized.append(arg)
        if arg in _PATH_FLAGS:
            expect_path_value = True

    return normalized


@ray.remote
def _run_oob_task(extra_args: list[str], json_mode: bool) -> dict:
    """Execute a single oob task inside a Ray worker with GPU isolation.

    Ray sets CUDA_VISIBLE_DEVICES automatically; we mirror it to HIP for AMD.
    """
    cuda_vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cuda_vis:
        os.environ["HIP_VISIBLE_DEVICES"] = cuda_vis

    oob_bin = os.environ.get("OOB_CLI", "oob")
    cmd = [oob_bin, "run"] + extra_args

    start_ts = time.time()

    if not json_mode:
        print(f"[oob-ray] starting on GPU {cuda_vis or '0'}")
        print(f"[oob-ray] cmd = {' '.join(cmd)}")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )

    elapsed = time.time() - start_ts
    status = "completed" if result.returncode == 0 else "failed"

    return {
        "status": status,
        "returncode": result.returncode,
        "elapsed_s": round(elapsed, 1),
        "gpu": cuda_vis or "0",
        "stdout_tail": result.stdout[-2000:] if result.stdout else "",
        "stderr_tail": result.stderr[-2000:] if result.stderr else "",
        "stdout": result.stdout,
    }


def cmd_run(args):
    """Run a single OOB task via Ray (blocking)."""
    extra_args = _normalize_extra_args(args.extra_args)
    json_mode = "--json" in extra_args
    runtime_env = _build_runtime_env()
    ray.init(address=RAY_ADDRESS, ignore_reinit_error=True, runtime_env=runtime_env)

    num_gpus = args.num_gpus or DEFAULT_NUM_GPUS
    task_fn = _run_oob_task.options(num_gpus=num_gpus, runtime_env=runtime_env)
    ref = task_fn.remote(extra_args, json_mode)

    if not json_mode:
        print(f"[oob-ray] Submitted 1 task, requesting {num_gpus} GPU(s)")
    result = ray.get(ref)

    _print_result(result, json_mode=json_mode)
    sys.exit(0 if result["returncode"] == 0 else 1)


def cmd_status(args):
    """Show Ray cluster GPU resources and running tasks."""
    ray.init(address=RAY_ADDRESS, ignore_reinit_error=True)

    resources = ray.cluster_resources()
    available = ray.available_resources()

    total_gpu = int(resources.get("GPU", 0))
    free_gpu = int(available.get("GPU", 0))
    used_gpu = total_gpu - free_gpu

    print(f"Ray cluster resources:")
    print(f"  GPU: {used_gpu}/{total_gpu} in use ({free_gpu} free)")
    print(f"  CPU: {available.get('CPU', 0):.0f}/{resources.get('CPU', 0):.0f}")
    print(f"  Memory: {available.get('memory', 0) / 1e9:.1f}/{resources.get('memory', 0) / 1e9:.1f} GB")


def _print_result(r: dict, *, json_mode: bool):
    if json_mode:
        raw = (r.get("stdout") or "").strip()
        if raw:
            try:
                json.loads(raw)
                print(raw)
                return
            except json.JSONDecodeError:
                pass
        print(json.dumps({
            "task_id": None,
            "status": r.get("status", "failed"),
            "error_message": r.get("stderr_tail") or r.get("stdout_tail") or "oob_ray_submit produced non-JSON output",
            "workspace": None,
            "log_file": None,
            "usage": None,
        }))
        return

    # Print the actual stdout from the OOB CLI so the user sees the Rich output
    if r.get("stdout"):
        print(r["stdout"])
        
    status_icon = "OK" if r["returncode"] == 0 else "FAIL"
    print(f"\n{'=' * 60}")
    print(f"  [{status_icon}] OOB Task — GPU {r['gpu']} — {r['elapsed_s']}s")
    print(f"{'=' * 60}")
    if r["returncode"] != 0 and r["stderr_tail"]:
        print(f"  stderr: ...{r['stderr_tail'][-500:]}")


def main():
    parser = argparse.ArgumentParser(
        description="OOB Ray Submit — schedule oob CLI tasks via local Ray",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    # run (single task)
    p_run = sub.add_parser("run", help="Run a single OOB task (blocking)")
    p_run.add_argument("--num-gpus", type=int, help=f"GPUs per task (default: {DEFAULT_NUM_GPUS})")

    # status
    sub.add_parser("status", help="Show Ray cluster GPU resources")

    args, extra = parser.parse_known_args()
    args.extra_args = [a for a in extra if a != "--"]

    if not args.command:
        parser.print_help()
        sys.exit(0)

    cmds = {"run": cmd_run, "status": cmd_status}
    cmds[args.command](args)


if __name__ == "__main__":
    main()
