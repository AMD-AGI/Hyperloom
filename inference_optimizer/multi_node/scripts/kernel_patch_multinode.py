#!/usr/bin/env python3
"""Multi-node kernel patch fan-out (apply / revert).

Counterpart to ``kill_multinode.py`` and ``launch_multinode.py``. Submitted
via Ray Dashboard REST by ``inference_optimizer.multi_node apply-patch`` /
``revert-patch`` when the workload has ``nodes >= 2``.

Algorithm (apply):

  1. ``ray.init()`` (no address; in-pod).
  2. Decode the patch payload (base64-encoded source bytes) from
     ``--patch-b64`` once on the head; pass through to every actor.
  3. Enumerate all alive nodes.
  4. For each node, spawn a ``@ray.remote`` actor pinned via
     ``NodeAffinitySchedulingStrategy(node_id, soft=False)``.
  5. Inside each actor:
     a. Copy ``target_path`` to ``<backup_dir>/<safe_name>.<host>.bak``.
     b. Atomic write the decoded patch bytes to ``target_path``
        (write-to-tmp + ``os.replace`` so a half-written file never
        appears on disk visible to sglang loaders).
     c. ``py_compile.compile`` if ``target_path`` ends in ``.py`` to catch
        syntax errors before sglang tries to import it.
  6. Collect per-node results; emit a single JSON document on stdout
     so the caller can parse it from Ray Dashboard job_logs.

Algorithm (revert): same actor fan-out, each actor copies its recorded
backup file back to ``target_path``.

Failure semantics: any actor that raises is treated as a hard failure.
The caller (sandbox-side ``apply_kernel_patch.py``) is expected to
issue a follow-up ``revert`` to roll back the actors that did succeed,
preserving three-way (sandbox + head + worker) source consistency.

ADDENDUM: this script runs INSIDE the RayJob pod (sglang/vllm image),
not in the Claw sandbox. ``import ray`` is the standard in-pod way to
talk to the local GCS; sandbox-side code (``cli.py`` etc.) must NOT
import ray.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import py_compile
import shutil
import socket
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy


def _log(msg: str) -> None:
    """Stderr-only timestamped log line.

    stdout is reserved for the final JSON document the dashboard caller
    parses, so all progress chatter goes to stderr.

    Args:
        msg (str): The message text to emit.
    """
    ts = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
    sys.stderr.write(f"[kernel_patch_multinode {ts}] {msg}\n")
    sys.stderr.flush()


def _safe_name(value: str) -> str:
    """Sanitize a string for use as a filename component.

    Args:
        value (str): The raw string to sanitize.

    Returns:
        str: A filename-safe slug (alnum plus ``._-``), truncated to 80
        characters; ``"patch"`` if the result would be empty.
    """
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)
    return cleaned[:80] or "patch"


def _atomic_write_bytes(target: Path, data: bytes) -> None:
    """Write ``data`` to ``target`` atomically (tmp file + os.replace).

    Args:
        target (Path): Destination file path.
        data (bytes): Bytes to write.

    Raises:
        OSError: If writing the temp file or replacing the target fails.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_str = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent),
    )
    tmp = Path(tmp_str)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.replace(tmp, target)
    except Exception:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
        raise


def _apply_remote(
    target_path: str,
    patch_b64: str,
    backup_dir: str,
    kernel_id: str,
) -> dict:
    """Apply a single patch on THIS pod.

    Backs up the target, atomically writes the decoded patch bytes, and
    (for ``.py`` targets) byte-compiles to catch syntax errors, auto-
    reverting from the backup if compilation fails. Raises on any error so
    the caller sees the failure via ``ray.get`` exception propagation.

    Args:
        target_path (str): Absolute path of the file to overwrite.
        patch_b64 (str): Base64-encoded new file contents.
        backup_dir (str): Directory where the pre-patch original is saved.
        kernel_id (str): Optional identifier used in the backup filename.

    Returns:
        dict: Summary with host, target path, backup path, bytes written,
        and the compile result.

    Raises:
        FileNotFoundError: If ``target_path`` does not exist on the pod.
        ValueError: If ``patch_b64`` is not valid base64, or if the patched
            ``.py`` file fails to compile (after auto-revert).
    """
    host = socket.gethostname()
    target = Path(target_path)
    if not target.is_file():
        raise FileNotFoundError(f"target_path does not exist on pod {host}: {target}")

    bdir = Path(backup_dir)
    bdir.mkdir(parents=True, exist_ok=True)
    backup_name = f"{_safe_name(kernel_id or target.stem)}_{host}_{int(time.time())}.bak"
    backup_path = bdir / backup_name
    shutil.copy2(target, backup_path)

    try:
        data = base64.b64decode(patch_b64.encode("ascii"))
    except Exception as exc:
        raise ValueError(f"patch_b64 not valid base64: {exc!r}") from exc

    _atomic_write_bytes(target, data)

    compile_result: dict[str, Any] = {"status": "skipped", "reason": "non-py target"}
    if target.suffix.lower() == ".py":
        try:
            py_compile.compile(str(target), doraise=True)
            compile_result = {"status": "ok"}
        except py_compile.PyCompileError as exc:
            shutil.copy2(backup_path, target)
            raise ValueError(
                f"py_compile failed on {target} (auto-reverted): {exc.msg}"
            ) from exc

    return {
        "host": host,
        "target_path": str(target),
        "backup_path": str(backup_path),
        "wrote_bytes": len(data),
        "compile": compile_result,
    }


def _revert_remote(
    target_path: str,
    backup_path: str,
) -> dict:
    """Restore ``target_path`` from ``backup_path`` on THIS pod.

    Idempotent when ``backup_path`` is missing (assumes already-reverted
    state).

    Args:
        target_path (str): Absolute path of the file to restore.
        backup_path (str): Path of the backup copy to restore from.

    Returns:
        dict: Summary with host, target path, backup path, and a status of
        ``restored`` or ``noop_missing_backup``.
    """
    host = socket.gethostname()
    target = Path(target_path)
    backup = Path(backup_path)
    if not backup.is_file():
        _log(f"revert noop on {host}: backup missing at {backup}")
        return {
            "host": host,
            "target_path": str(target),
            "backup_path": str(backup),
            "status": "noop_missing_backup",
        }
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(backup, target)
    return {
        "host": host,
        "target_path": str(target),
        "backup_path": str(backup),
        "status": "restored",
    }


def _alive_nodes(min_gpu: int = 0) -> list[dict]:
    """Return the list of currently-alive Ray nodes.

    Each entry is the full ``ray.nodes()`` row so the caller can pick
    per-node IDs and addresses for ``NodeAffinitySchedulingStrategy``.

    Args:
        min_gpu (int): If > 0, only return nodes with at least this many
            GPUs.

    Returns:
        list[dict]: The matching alive node rows from ``ray.nodes()``.
    """
    nodes = [n for n in ray.nodes() if n.get("Alive")]
    if min_gpu > 0:
        nodes = [n for n in nodes if int(n.get("Resources", {}).get("GPU", 0) or 0) >= min_gpu]
    return nodes


def _do_apply(args: argparse.Namespace) -> int:
    """Fan out the patch-apply actor across every alive node and report.

    Args:
        args (argparse.Namespace): Parsed ``apply`` arguments
            (``target_path``, ``patch_b64``, ``backup_dir``, ``kernel_id``,
            ``timeout_sec``).

    Returns:
        int: ``0`` if every node applied successfully, otherwise ``1``.
    """
    ray.init(ignore_reinit_error=True, log_to_driver=True)
    nodes = _alive_nodes()
    _log(f"apply: alive nodes={len(nodes)} target={args.target_path}")
    if not nodes:
        sys.stdout.write(json.dumps({
            "command": "apply",
            "status": "failed",
            "error": "no alive Ray nodes for fan-out",
        }, indent=2) + "\n")
        return 1

    ApplyActor = ray.remote(num_cpus=0, num_gpus=0)(_apply_remote)
    refs = []
    for node in nodes:
        node_id = node["NodeID"]
        ref = ApplyActor.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(
                node_id=node_id, soft=False,
            ),
        ).remote(
            args.target_path, args.patch_b64, args.backup_dir, args.kernel_id,
        )
        refs.append((node_id[:16], ref))

    per_node: list[dict] = []
    failures: list[dict] = []
    for short_id, ref in refs:
        try:
            res = ray.get(ref, timeout=args.timeout_sec)
            per_node.append({"node_id": short_id, **res})
        except Exception as exc:  # noqa: BLE001
            _log(f"node {short_id}: apply FAILED: {type(exc).__name__}: {exc}")
            failures.append({"node_id": short_id, "error": str(exc),
                             "error_class": type(exc).__name__})

    payload = {
        "command": "apply",
        "target_path": args.target_path,
        "kernel_id": args.kernel_id,
        "backup_dir": args.backup_dir,
        "per_node": per_node,
        "failures": failures,
        "status": "ok" if not failures else "partial",
    }
    sys.stdout.write(json.dumps(payload, indent=2) + "\n")
    sys.stdout.flush()
    return 0 if not failures else 1


def _do_revert(args: argparse.Namespace) -> int:
    """Fan out the patch-revert actor to each backed-up host and report.

    Args:
        args (argparse.Namespace): Parsed ``revert`` arguments
            (``target_path``, ``backup_map_json``, ``timeout_sec``).

    Returns:
        int: ``0`` if every reachable host reverted successfully, otherwise
        ``1`` (including when ``backup_map_json`` is empty).
    """
    ray.init(ignore_reinit_error=True, log_to_driver=True)
    backup_map: dict[str, str] = json.loads(args.backup_map_json or "{}")
    if not backup_map:
        sys.stdout.write(json.dumps({
            "command": "revert",
            "status": "failed",
            "error": "empty backup_map_json (expected {hostname: backup_path})",
        }, indent=2) + "\n")
        return 1

    nodes = _alive_nodes()
    by_host: dict[str, str] = {}
    for n in nodes:
        host = (n.get("NodeManagerHostname") or "").strip()
        if host:
            by_host[host] = n["NodeID"]
    _log(f"revert: alive nodes={len(nodes)} target={args.target_path} "
         f"backups={len(backup_map)}")

    RevertActor = ray.remote(num_cpus=0, num_gpus=0)(_revert_remote)
    refs = []
    for host, backup_path in backup_map.items():
        node_id = by_host.get(host)
        if not node_id:
            _log(f"WARN host {host} not currently alive; revert skipped for this host")
            continue
        ref = RevertActor.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(
                node_id=node_id, soft=False,
            ),
        ).remote(args.target_path, backup_path)
        refs.append((host, ref))

    per_node: list[dict] = []
    failures: list[dict] = []
    for host, ref in refs:
        try:
            res = ray.get(ref, timeout=args.timeout_sec)
            per_node.append({"host": host, **res})
        except Exception as exc:  # noqa: BLE001
            _log(f"host {host}: revert FAILED: {type(exc).__name__}: {exc}")
            failures.append({"host": host, "error": str(exc),
                             "error_class": type(exc).__name__})

    payload = {
        "command": "revert",
        "target_path": args.target_path,
        "per_node": per_node,
        "failures": failures,
        "status": "ok" if not failures else "partial",
    }
    sys.stdout.write(json.dumps(payload, indent=2) + "\n")
    sys.stdout.flush()
    return 0 if not failures else 1


def main() -> int:
    """Parse CLI arguments and dispatch the ``apply`` or ``revert`` command.

    Returns:
        int: Process exit code; the subcommand's result code, or ``2`` if
        no recognized subcommand was given.
    """
    p = argparse.ArgumentParser(
        prog="kernel_patch_multinode.py",
        description=(
            "Fan-out kernel patch apply/revert across every node of the "
            "current Ray cluster (one actor per node, NodeAffinity hard-"
            "pinned). Designed to be heredoc-embedded into a Ray "
            "Dashboard /api/jobs/ submission by inference_optimizer."
        ),
    )
    sub = p.add_subparsers(dest="command", required=True)

    ap = sub.add_parser("apply", help="apply a patch to target_path on every pod")
    ap.add_argument("--target-path", required=True,
                    help="absolute file path on the pod (e.g. /sgl-workspace/aiter/aiter/ops/gemm.py)")
    ap.add_argument("--patch-b64", required=True,
                    help="base64-encoded new file contents")
    ap.add_argument("--backup-dir", required=True,
                    help="directory on each pod where the pre-patch original is saved")
    ap.add_argument("--kernel-id", default="",
                    help="optional id used to construct backup filename")
    ap.add_argument("--timeout-sec", type=int, default=120,
                    help="per-actor timeout (default 120s)")

    rp = sub.add_parser("revert", help="restore target_path from per-pod backup")
    rp.add_argument("--target-path", required=True)
    rp.add_argument("--backup-map-json", required=True,
                    help='JSON object mapping pod hostname -> backup file path')
    rp.add_argument("--timeout-sec", type=int, default=60)

    args = p.parse_args()
    if args.command == "apply":
        return _do_apply(args)
    if args.command == "revert":
        return _do_revert(args)
    p.print_help(sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
