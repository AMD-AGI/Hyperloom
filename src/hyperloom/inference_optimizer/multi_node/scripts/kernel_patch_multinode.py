#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Multi-node kernel patch fan-out (apply / revert), run INSIDE the RayJob pod.

One actor per alive node (NodeAffinity hard-pinned): apply backs up
``target_path``, atomically writes the decoded patch bytes, and
``py_compile``s ``.py`` targets (auto-reverting on failure); revert copies
the recorded backup back. Any actor raising is a hard failure (caller
issues a follow-up revert). Emits one JSON summary on stdout.
"""

from __future__ import annotations

import argparse
import base64
import json
import py_compile
import shutil
import socket
import sys
import time
from pathlib import Path
from typing import Any

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

# patch_path_safety.py is shipped beside this script on RayJob pods.
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
from patch_path_safety import (  # noqa: E402
    atomic_write_bytes,
    assert_backup_dir_allowed,
    assert_revert_paths_allowed,
    assert_target_path_allowed,
    invalidate_aiter_jit_build,
    restore_aiter_jit_build,
)


def _log(msg: str) -> None:
    """Stderr-only timestamped log line (stdout is reserved for the final JSON).

    Args:
        msg: The message text to emit.
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


def _apply_remote(
    target_path: str,
    patch_b64: str,
    backup_dir: str,
    kernel_id: str,
    jit_build_dir: str = "",
) -> dict:
    """Apply a single patch on this pod; raises on any error (surfaced via ``ray.get``).

    Args:
        target_path: Absolute path of the file to overwrite on the pod.
        patch_b64: Base64-encoded new file contents.
        backup_dir: Directory where the pre-patch original is saved.
        kernel_id: Optional id used to construct the backup filename.

    Returns:
        dict: Per-host result with the target path, backup path, byte count,
        and compile status.

    Raises:
        ValueError: If ``target_path`` does not exist or resolves outside the
            framework patch roots, ``backup_dir`` is outside the kernel backup
            root, ``patch_b64`` is not valid base64, or a ``.py`` target fails
            to compile (it is auto-reverted first).
    """
    host = socket.gethostname()
    target = Path(target_path)
    assert_target_path_allowed(target, must_exist=True)
    assert_backup_dir_allowed(Path(backup_dir))
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

    jit_backup = invalidate_aiter_jit_build(
        Path(jit_build_dir) if jit_build_dir else None,
        bdir,
        f"{_safe_name(kernel_id or target.stem)}_{host}_{int(time.time())}",
    )

    compile_result: dict[str, Any] = {"status": "skipped", "reason": "non-py target"}
    try:
        atomic_write_bytes(target, data)
        if target.suffix.lower() == ".py":
            py_compile.compile(str(target), doraise=True)
            compile_result = {"status": "ok"}
    except Exception:
        shutil.copy2(backup_path, target)
        restore_aiter_jit_build(jit_backup)
        raise

    return {
        "host": host,
        "target_path": str(target),
        "backup_path": str(backup_path),
        "wrote_bytes": len(data),
        "compile": compile_result,
        "jit_backup": jit_backup,
    }


def _revert_remote(records: list[dict]) -> dict:
    """Restore every source and JIT backup recorded for one pod.

    Args:
        records: Apply records for all files patched on this pod.

    Returns:
        dict: Per-host result with ``status`` of ``restored`` or
        ``noop_missing_backup``.
    """
    host = socket.gethostname()
    restored: list[str] = []
    jit_records: list[dict] = []
    for record in reversed(records):
        target = Path(str(record.get("target_path") or ""))
        backup = Path(str(record.get("backup_path") or ""))
        if not backup.is_file():
            raise FileNotFoundError(f"backup missing on {host}: {backup}")
        assert_revert_paths_allowed(target, backup)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(backup, target)
        restored.append(str(target))
        jit_record = record.get("jit_backup")
        if isinstance(jit_record, dict) and jit_record.get("status") in {"ok", "clean"}:
            jit_records.append(jit_record)
    jit_restore = {"status": "skipped", "reason": "no JIT backup"}
    if jit_records:
        first = jit_records[0]
        if any(record != first for record in jit_records[1:]):
            raise ValueError("conflicting JIT backup records for one pod")
        jit_restore = restore_aiter_jit_build(first)
    return {
        "host": host,
        "status": "restored",
        "restored_targets": restored,
        "jit_restore": jit_restore,
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
        sys.stdout.write(
            json.dumps(
                {
                    "command": "apply",
                    "status": "failed",
                    "error": "no alive Ray nodes for fan-out",
                },
                indent=2,
            )
            + "\n"
        )
        return 1

    ApplyActor = ray.remote(num_cpus=0, num_gpus=0)(_apply_remote)
    refs = []
    for node in nodes:
        node_id = node["NodeID"]
        ref = ApplyActor.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(
                node_id=node_id,
                soft=False,
            ),
        ).remote(
            args.target_path,
            args.patch_b64,
            args.backup_dir,
            args.kernel_id,
            args.jit_build_dir,
        )
        refs.append((node_id, ref))

    per_node: list[dict] = []
    failures: list[dict] = []
    successful_nodes: list[tuple[str, dict]] = []
    for node_id, ref in refs:
        short_id = node_id[:16]
        try:
            res = ray.get(ref, timeout=args.timeout_sec)
            per_node.append({"node_id": short_id, **res})
            successful_nodes.append((node_id, res))
        except Exception as exc:  # noqa: BLE001
            _log(f"node {short_id}: apply FAILED: {type(exc).__name__}: {exc}")
            failures.append({"node_id": short_id, "error": str(exc), "error_class": type(exc).__name__})

    rollback: list[dict] = []
    if failures and successful_nodes:
        RevertActor = ray.remote(num_cpus=0, num_gpus=0)(_revert_remote)
        rollback_refs = [
            (
                node_id,
                RevertActor.options(
                    scheduling_strategy=NodeAffinitySchedulingStrategy(
                        node_id=node_id,
                        soft=False,
                    )
                ).remote([record]),
            )
            for node_id, record in successful_nodes
        ]
        for node_id, ref in rollback_refs:
            try:
                rollback.append(
                    {"node_id": node_id[:16], **ray.get(ref, timeout=args.timeout_sec)}
                )
            except Exception as exc:  # noqa: BLE001
                rollback.append(
                    {
                        "node_id": node_id[:16],
                        "status": "failed",
                        "error": str(exc),
                    }
                )
    payload = {
        "command": "apply",
        "target_path": args.target_path,
        "kernel_id": args.kernel_id,
        "backup_dir": args.backup_dir,
        "per_node": per_node,
        "failures": failures,
        "rollback": rollback,
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
    records_by_host: dict[str, list[dict]] = json.loads(args.records_json or "{}")
    if not records_by_host and args.backup_map_json:
        backup_map: dict[str, str] = json.loads(args.backup_map_json)
        records_by_host = {
            host: [
                {
                    "target_path": args.target_path,
                    "backup_path": backup_path,
                }
            ]
            for host, backup_path in backup_map.items()
        }
    if not records_by_host:
        sys.stdout.write(
            json.dumps(
                {
                    "command": "revert",
                    "status": "failed",
                    "error": "empty records_json",
                },
                indent=2,
            )
            + "\n"
        )
        return 1

    nodes = _alive_nodes()
    by_host: dict[str, str] = {}
    for n in nodes:
        host = (n.get("NodeManagerHostname") or "").strip()
        if host:
            by_host[host] = n["NodeID"]
    _log(f"revert: alive nodes={len(nodes)} hosts={len(records_by_host)}")

    RevertActor = ray.remote(num_cpus=0, num_gpus=0)(_revert_remote)
    refs = []
    failures: list[dict] = []
    for host, records in records_by_host.items():
        node_id = by_host.get(host)
        if not node_id:
            _log(f"WARN host {host} not currently alive; revert skipped for this host")
            failures.append(
                {
                    "host": host,
                    "error": "host is not currently alive",
                    "error_class": "HostUnavailable",
                }
            )
            continue
        ref = RevertActor.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(
                node_id=node_id,
                soft=False,
            ),
        ).remote(records)
        refs.append((host, ref))

    per_node: list[dict] = []
    for host, ref in refs:
        try:
            res = ray.get(ref, timeout=args.timeout_sec)
            per_node.append({"host": host, **res})
        except Exception as exc:  # noqa: BLE001
            _log(f"host {host}: revert FAILED: {type(exc).__name__}: {exc}")
            failures.append({"host": host, "error": str(exc), "error_class": type(exc).__name__})

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
            "Dashboard /api/jobs/ submission by hyperloom.inference_optimizer."
        ),
    )
    sub = p.add_subparsers(dest="command", required=True)

    ap = sub.add_parser("apply", help="apply a patch to target_path on every pod")
    ap.add_argument(
        "--target-path",
        required=True,
        help="absolute file path on the pod (e.g. /sgl-workspace/aiter/aiter/ops/gemm.py)",
    )
    ap.add_argument("--patch-b64", required=True, help="base64-encoded new file contents")
    ap.add_argument("--backup-dir", required=True, help="directory on each pod where the pre-patch original is saved")
    ap.add_argument("--kernel-id", default="", help="optional id used to construct backup filename")
    ap.add_argument("--jit-build-dir", default="")
    ap.add_argument("--timeout-sec", type=int, default=120, help="per-actor timeout (default 120s)")

    rp = sub.add_parser("revert", help="restore target_path from per-pod backup")
    rp.add_argument("--target-path", default="")
    rp.add_argument("--records-json", default="")
    rp.add_argument("--backup-map-json", default="", help="legacy hostname -> backup path map")
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
