#!/usr/bin/env python3
"""GEAK Ray Submit — schedule geak CLI tasks via local Ray cluster.

Replaces geak_client.py for local mode. Each task gets an isolated
GPU allocated by Ray; multiple tasks run concurrently up to GPU count.

GEAK config: auto-loaded from $GEAK_CONFIG (default: /opt/hyperloom/geak-config/local.yaml).
Override via env: GEAK_MODEL_NAME / GEAK_API_KEY / GEAK_BASE_URL (rendered at container start).

Usage:
    # Single task (blocking, prints result when done)
    python geak_ray_submit.py run -t task.md --yolo

    # Batch: submit multiple kernels concurrently, wait for all
    python geak_ray_submit.py batch -t kernel_a.md -t kernel_b.md -t kernel_c.md --yolo

    # Status: check Ray cluster resources
    python geak_ray_submit.py status

Extra args after known flags are forwarded to the `geak` CLI.
"""

import argparse
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
DEFAULT_NUM_GPUS = int(os.environ.get("GEAK_GPUS_PER_TASK", "1"))

# Env vars that GEAK / litellm / providers need; forwarded to every Ray worker.
_FORWARDED_ENV_KEYS = (
    "AMD_LLM_API_KEY", "LLM_API_KEY", "LLM_API_BASE", "LLM_GATEWAY_KEY",
    "OPENAI_API_KEY", "OPENAI_BASE_URL",
    "ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL",
    "MSWEA_MODEL_NAME", "GEAK_WORK_DIR",
    "GEAK_CONFIG", "GEAK_MODEL_NAME", "GEAK_API_KEY", "GEAK_BASE_URL",
    "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES",
    "PATH", "HOME", "LD_LIBRARY_PATH",
)


def _build_runtime_env() -> dict:
    """Forward LLM/GPU env vars to Ray workers; alias LLM_API_KEY -> AMD_LLM_API_KEY."""
    env = {k: os.environ[k] for k in _FORWARDED_ENV_KEYS if k in os.environ}
    if "AMD_LLM_API_KEY" not in env:
        for fallback in ("LLM_API_KEY", "LLM_GATEWAY_KEY"):
            if fallback in env:
                env["AMD_LLM_API_KEY"] = env[fallback]
                break
    return {"env_vars": env}


def _find_geak_bin() -> str:
    """Locate the geak CLI binary.

    Fallbacks `mini` (upstream mini-swe-agent name) and `geak-gaagent`
    (legacy alias) are kept for backward compatibility with older images.
    """
    import shutil
    for name in ("geak", "mini", "geak-gaagent"):
        path = shutil.which(name)
        if path:
            return path
    return "geak"


def _normalize_task_files(task_files: list[str]) -> list[str]:
    """Resolve task files on the submitter before handing them to Ray workers."""
    return [str(Path(task_file).expanduser().resolve()) for task_file in task_files]


def _ensure_yolo(extra_args: list[str]) -> list[str]:
    """Default to non-interactive GEAK runs unless the caller already chose it."""
    if any(arg in {"-y", "--yolo"} for arg in extra_args):
        return list(extra_args)
    return [*extra_args, "--yolo"]


@ray.remote
def _run_geak_task(task_file: str, extra_args: list[str], num_gpus: int) -> dict:
    """Execute a single geak task inside a Ray worker with GPU isolation.

    Ray sets CUDA_VISIBLE_DEVICES automatically; we mirror it to HIP for AMD.
    GEAK accepts a task file path for `-t`; the submitter resolves it before
    dispatch so Ray workers do not depend on their own current directory.
    """
    cuda_vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cuda_vis:
        os.environ["HIP_VISIBLE_DEVICES"] = cuda_vis

    geak_bin = _find_geak_bin()
    # Auto-inject --config from $GEAK_CONFIG unless user already supplied one
    config_args: list[str] = []
    geak_config = os.environ.get("GEAK_CONFIG", "")
    if geak_config and Path(geak_config).is_file() and "--config" not in extra_args:
        config_args = ["--config", geak_config]
    # Pass the file path directly to the CLI instead of reading its content as a string
    cmd = [geak_bin, "-t", task_file, "--gpu-ids", cuda_vis or "0"] + config_args + extra_args

    start_ts = time.time()
    task_label = Path(task_file).stem

    print(f"[geak-ray] {task_label}: starting on GPU {cuda_vis or '0'}")
    print(f"[geak-ray] {task_label}: cmd = {geak_bin} -t {task_file} --gpu-ids {cuda_vis or '0'} {' '.join(config_args + extra_args)}")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=os.environ.get("GEAK_WORK_DIR", str(Path(task_file).parent)),
    )

    elapsed = time.time() - start_ts
    status = "completed" if result.returncode == 0 else "failed"

    return {
        "task_file": task_file,
        "label": task_label,
        "status": status,
        "returncode": result.returncode,
        "elapsed_s": round(elapsed, 1),
        "gpu": cuda_vis or "0",
        "stdout_tail": result.stdout[-2000:] if result.stdout else "",
        "stderr_tail": result.stderr[-2000:] if result.stderr else "",
    }


def cmd_run(args):
    """Run a single GEAK task via Ray (blocking)."""
    runtime_env = _build_runtime_env()
    ray.init(address=RAY_ADDRESS, ignore_reinit_error=True, runtime_env=runtime_env)

    num_gpus = args.num_gpus or DEFAULT_NUM_GPUS
    task_files = _normalize_task_files(args.task_files)
    extra_args = _ensure_yolo(args.extra_args)
    task_fn = _run_geak_task.options(num_gpus=num_gpus, runtime_env=runtime_env)
    ref = task_fn.remote(task_files[0], extra_args, num_gpus)

    print(f"[geak-ray] Submitted 1 task, requesting {num_gpus} GPU(s)")
    result = ray.get(ref)

    _print_result(result)
    sys.exit(0 if result["returncode"] == 0 else 1)


def cmd_batch(args):
    """Submit multiple GEAK tasks concurrently via Ray."""
    runtime_env = _build_runtime_env()
    ray.init(address=RAY_ADDRESS, ignore_reinit_error=True, runtime_env=runtime_env)

    num_gpus = args.num_gpus or DEFAULT_NUM_GPUS
    task_files = _normalize_task_files(args.task_files)
    extra_args = _ensure_yolo(args.extra_args)
    task_fn = _run_geak_task.options(num_gpus=num_gpus, runtime_env=runtime_env)

    refs = []
    for tf in task_files:
        ref = task_fn.remote(tf, extra_args, num_gpus)
        refs.append(ref)

    n = len(refs)
    avail = ray.cluster_resources().get("GPU", 0)
    print(f"[geak-ray] Submitted {n} tasks, {num_gpus} GPU(s) each, cluster has {int(avail)} GPU(s)")
    if n * num_gpus > avail:
        print(f"[geak-ray] Note: {n * num_gpus} GPUs requested > {int(avail)} available — tasks will queue")

    results = []
    done_ids = set()
    while len(done_ids) < n:
        ready, _ = ray.wait(refs, num_returns=len(refs), timeout=5.0)
        for ref in ready:
            ref_id = ref.hex()
            if ref_id not in done_ids:
                done_ids.add(ref_id)
                result = ray.get(ref)
                results.append(result)
                _print_result(result)
    _print_summary(results)

    failed = [r for r in results if r["returncode"] != 0]
    sys.exit(1 if failed else 0)


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


def _print_result(r: dict):
    status_icon = "OK" if r["returncode"] == 0 else "FAIL"
    print(f"\n{'=' * 60}")
    print(f"  [{status_icon}] {r['label']} — GPU {r['gpu']} — {r['elapsed_s']}s")
    print(f"{'=' * 60}")
    if r["returncode"] != 0 and r["stderr_tail"]:
        print(f"  stderr: ...{r['stderr_tail'][-500:]}")


def _print_summary(results: list[dict]):
    print(f"\n{'=' * 60}")
    print(f"  SUMMARY: {len(results)} tasks")
    print(f"{'=' * 60}")
    for r in sorted(results, key=lambda x: x["task_file"]):
        icon = "OK" if r["returncode"] == 0 else "FAIL"
        print(f"  [{icon}] {r['label']:<30} GPU {r['gpu']}  {r['elapsed_s']:>7.1f}s")
    ok = sum(1 for r in results if r["returncode"] == 0)
    print(f"\n  {ok}/{len(results)} succeeded")


def main():
    parser = argparse.ArgumentParser(
        description="GEAK Ray Submit — schedule geak CLI tasks via local Ray",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    # run (single task)
    p_run = sub.add_parser("run", help="Run a single GEAK task (blocking)")
    p_run.add_argument("-t", "--task", dest="task_files", action="append", required=True,
                       help="Task markdown file")
    p_run.add_argument("--num-gpus", type=int, help=f"GPUs per task (default: {DEFAULT_NUM_GPUS})")

    # batch (multiple tasks)
    p_batch = sub.add_parser("batch", help="Submit multiple GEAK tasks concurrently")
    p_batch.add_argument("-t", "--task", dest="task_files", action="append", required=True,
                         help="Task markdown file (repeat for each kernel)")
    p_batch.add_argument("--num-gpus", type=int, help=f"GPUs per task (default: {DEFAULT_NUM_GPUS})")

    # status
    sub.add_parser("status", help="Show Ray cluster GPU resources")

    args, extra = parser.parse_known_args()
    args.extra_args = [a for a in extra if a != "--"]

    if not args.command:
        parser.print_help()
        sys.exit(0)

    cmds = {"run": cmd_run, "batch": cmd_batch, "status": cmd_status}
    cmds[args.command](args)


if __name__ == "__main__":
    main()
