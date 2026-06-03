#!/usr/bin/env python3
"""Multi-node-aware kernel micro-benchmark runner.

Counterpart to ``kill_multinode.py`` and ``kernel_patch_multinode.py``.
Submitted via Ray Dashboard REST by ``inference_optimizer.multi_node
kernel-bench`` when the workload has ``nodes >= 2``. The sandbox-side
caller is ``kernel_optimization.py`` running on a CPU-only sandbox, so
it cannot itself compile or execute GPU code; we delegate to a single
GPU-bearing Ray actor on the head pod (rank 0) which mirrors how
single-node kernel-agent works locally.

Algorithm:

  1. ``ray.init()`` (no address; in-pod).
  2. Pick THIS pod (the head) as the bench host so the run lands on
     real GPUs; spawn ``@ray.remote(num_gpus=1)`` actor pinned to it
     via ``NodeAffinitySchedulingStrategy(node_id, soft=False)``.
  3. The actor:
     a. ``cd`` into the workspace dir provided by the caller.
     b. base64-decode the bench script + any extra files (test data,
        helper modules) into that workspace; preserve directory
        structure under the workspace root.
     c. Run ``bash bench_command`` with the workspace as CWD, capturing
        stdout/stderr and the exit code.
     d. Glob the workspace for result JSON files matching
        ``result_glob`` and read them back (small artifacts only;
        anything > 1 MiB is skipped with a note so we don't OOM the
        dashboard stdout buffer).
  4. Emit a single JSON document on stdout summarising stdout tail,
     stderr tail, exit code, and the read-back artifact contents.

Why GPU=1 actor instead of GPU=N: kernel micro-benchmarks are by design
single-rank — the per-kernel optimization loop scales by *parallel*
candidate count (handled at a higher layer), not by sharding one bench
across multiple GPUs. ``num_gpus=1`` keeps Ray's scheduler honest and
prevents the bench from being co-scheduled with sglang's TP ranks.

ADDENDUM: runs INSIDE the head pod (sglang/vllm image), not in the Claw
sandbox.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy


# Per-artifact read-back cap so a runaway script can't blow up the
# dashboard stdout buffer with massive JSON dumps.
_MAX_ARTIFACT_BYTES = 1 * 1024 * 1024
# Stdout/stderr tail size returned to the caller.
_STREAM_TAIL_BYTES = 32 * 1024


def _log(msg: str) -> None:
    """Write a timestamped progress line to stderr and flush it.

    Args:
        msg (str): The message text to emit.
    """
    ts = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
    sys.stderr.write(f"[kernel_bench_multinode {ts}] {msg}\n")
    sys.stderr.flush()


def _tail_bytes(s: str | None, limit: int) -> str:
    """Return the trailing portion of a string up to a byte limit.

    Args:
        s (str | None): The source text, or ``None``.
        limit (int): Maximum number of trailing characters to keep.

    Returns:
        str: The last ``limit`` characters of ``s`` (or all of it if
        shorter); an empty string when ``s`` is falsy.
    """
    if not s:
        return ""
    if len(s) <= limit:
        return s
    return s[-limit:]


def _stage_files(workspace: Path, files_b64: dict[str, str]) -> list[str]:
    """Decode each ``{relative_path: base64_content}`` into the workspace.

    Relative paths starting with ``/`` or containing ``..`` are rejected
    so the staging cannot write outside the workspace dir.

    Args:
        workspace (Path): Root directory into which files are written.
        files_b64 (dict[str, str]): Mapping of relative path to
            base64-encoded file content.

    Returns:
        list[str]: Absolute paths of the files that were written.

    Raises:
        ValueError: If any relative path is absolute or contains ``..``.
    """
    staged: list[str] = []
    for rel, b64 in (files_b64 or {}).items():
        if rel.startswith("/") or ".." in Path(rel).parts:
            raise ValueError(
                f"staging path must be relative and free of '..': {rel!r}"
            )
        dst = workspace / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(base64.b64decode(b64.encode("ascii")))
        staged.append(str(dst))
    return staged


def _bench_remote(
    workspace: str,
    bench_command: str,
    files_b64_json: str,
    result_glob: str,
    timeout_sec: int,
) -> dict:
    """Run a kernel micro-benchmark on THIS pod (head).

    The caller pre-pins us to a GPU-bearing node via
    NodeAffinitySchedulingStrategy and ``num_gpus=1``; we don't need to
    query Ray for resources here. Stages helper files, runs the bench
    command in the workspace, and reads back matching result artifacts.

    Args:
        workspace (str): Absolute directory used as the bench CWD.
        bench_command (str): Shell command run via ``bash -lc``.
        files_b64_json (str): JSON object mapping relative paths to
            base64-encoded helper file content.
        result_glob (str): Glob (relative to workspace) of result files to
            read back.
        timeout_sec (int): Hard timeout for the bench command, in seconds.

    Returns:
        dict: Summary with host, workspace, staged files, return code,
        elapsed time, stdout/stderr tails, and read-back artifacts.

    Raises:
        ValueError: If ``files_b64_json`` is not valid JSON.
    """
    host = socket.gethostname()
    ws = Path(workspace)
    ws.mkdir(parents=True, exist_ok=True)

    try:
        files_b64 = json.loads(files_b64_json or "{}")
    except json.JSONDecodeError as exc:
        raise ValueError(f"files_b64_json not valid JSON: {exc}") from exc
    staged = _stage_files(ws, files_b64)

    started = time.time()
    proc = subprocess.run(
        ["bash", "-lc", bench_command],
        cwd=str(ws),
        capture_output=True,
        text=True,
        timeout=timeout_sec,
        env={**os.environ, "WORKSPACE_PATH": str(ws)},
    )
    elapsed = time.time() - started

    artifacts: list[dict[str, Any]] = []
    for path in sorted(ws.glob(result_glob)):
        if not path.is_file():
            continue
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if size > _MAX_ARTIFACT_BYTES:
            artifacts.append({
                "path": str(path),
                "size_bytes": size,
                "content": None,
                "skipped_reason": f"size > {_MAX_ARTIFACT_BYTES} bytes",
            })
            continue
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            artifacts.append({
                "path": str(path), "size_bytes": size, "content": None,
                "skipped_reason": f"read failed: {exc!r}",
            })
            continue
        # Best-effort JSON parse for nicer downstream consumption; fall
        # back to raw text if not JSON.
        parsed: Any = None
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            parsed = content
        artifacts.append({
            "path": str(path), "size_bytes": size, "content": parsed,
        })

    return {
        "host": host,
        "workspace": str(ws),
        "staged_files": staged,
        "bench_command": bench_command,
        "returncode": proc.returncode,
        "elapsed_sec": round(elapsed, 3),
        "stdout_tail": _tail_bytes(proc.stdout, _STREAM_TAIL_BYTES),
        "stderr_tail": _tail_bytes(proc.stderr, _STREAM_TAIL_BYTES),
        "artifacts": artifacts,
    }


def _pick_gpu_node() -> str:
    """Return the head-pod node id to run the bench on.

    Prefers the head pod because (a) it has GPUs and (b) it co-locates
    with the dashboard caller's context. Falls back to any alive GPU node
    if head detection fails.

    Returns:
        str: The Ray ``NodeID`` to pin the bench actor to.

    Raises:
        RuntimeError: If there are no alive nodes, or no alive node with
            at least one GPU when falling back.
    """
    nodes = [n for n in ray.nodes() if n.get("Alive")]
    if not nodes:
        raise RuntimeError("no alive Ray nodes for kernel bench")
    # Head pod hosts the dashboard which dispatched us, so the actor
    # running this script IS on the head — its node id matches.
    my_ip = ""
    try:
        my_ip = ray.util.get_node_ip_address()
    except Exception:  # noqa: BLE001
        pass
    for n in nodes:
        if (n.get("NodeManagerAddress") or "") == my_ip:
            return n["NodeID"]
    # Fallback: any alive node with GPUs > 0.
    for n in nodes:
        if int(n.get("Resources", {}).get("GPU", 0) or 0) >= 1:
            return n["NodeID"]
    raise RuntimeError("no alive Ray node with GPU >= 1 for kernel bench")


def _do_bench(args: argparse.Namespace) -> int:
    """Schedule the bench actor on a GPU node and emit its JSON result.

    Args:
        args (argparse.Namespace): Parsed ``bench`` subcommand arguments
            (``workspace``, ``bench_command``, ``files_b64_json``,
            ``result_glob``, ``timeout_sec``).

    Returns:
        int: ``0`` if the bench command exited 0, otherwise ``1``.
    """
    ray.init(ignore_reinit_error=True, log_to_driver=True)
    node_id = _pick_gpu_node()
    _log(f"bench: node_id={node_id[:16]} workspace={args.workspace}")

    BenchActor = ray.remote(num_cpus=1, num_gpus=1)(_bench_remote)
    ref = BenchActor.options(
        scheduling_strategy=NodeAffinitySchedulingStrategy(
            node_id=node_id, soft=False,
        ),
    ).remote(
        args.workspace, args.bench_command, args.files_b64_json,
        args.result_glob, args.timeout_sec,
    )

    try:
        res = ray.get(ref, timeout=args.timeout_sec + 60)
        ok = (res.get("returncode") == 0)
        payload = {
            "command": "bench",
            "status": "ok" if ok else "failed",
            "result": res,
        }
    except Exception as exc:  # noqa: BLE001
        _log(f"bench actor FAILED: {type(exc).__name__}: {exc}")
        payload = {
            "command": "bench",
            "status": "failed",
            "error": str(exc),
            "error_class": type(exc).__name__,
        }
    sys.stdout.write(json.dumps(payload, indent=2) + "\n")
    sys.stdout.flush()
    return 0 if payload["status"] == "ok" else 1


def main() -> int:
    """Parse CLI arguments and dispatch the ``bench`` subcommand.

    Returns:
        int: Process exit code; the bench result code, or ``2`` if no
        recognized subcommand was given.
    """
    p = argparse.ArgumentParser(
        prog="kernel_bench_multinode.py",
        description=(
            "Run a kernel micro-benchmark on one GPU-bearing pod node. "
            "Designed to be heredoc-embedded into a Ray Dashboard "
            "/api/jobs/ submission by inference_optimizer."
        ),
    )
    sub = p.add_subparsers(dest="command", required=True)

    bp = sub.add_parser("bench", help="compile + run a kernel micro-benchmark on a GPU node")
    bp.add_argument("--workspace", required=True,
                    help="absolute dir on pod that will be CWD for the bench")
    bp.add_argument("--bench-command", required=True,
                    help="shell command to invoke (passed to 'bash -lc')")
    bp.add_argument("--files-b64-json", default="{}",
                    help='JSON {rel_path: base64_content} of helper files to stage into workspace')
    bp.add_argument("--result-glob", default="*.json",
                    help="glob (relative to workspace) of result artifacts to read back")
    bp.add_argument("--timeout-sec", type=int, default=600,
                    help="hard timeout for the bench command (default 600s)")

    args = p.parse_args()
    if args.command == "bench":
        return _do_bench(args)
    p.print_help(sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
