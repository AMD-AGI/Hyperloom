#!/usr/bin/env python3
"""Single-pod, Ray-free kernel ops for the Infera backend (SSH control plane).

Ray-free counterpart to ``kernel_patch_multinode.py`` + ``kernel_bench_multinode.py``.
The Infera backend has no Ray cluster, so ``hyperloom.inference_optimizer.multi_node``
ships this script to each GPU pod over SSH and runs ONE subcommand per pod:

  apply   — back up ``--target-path`` then atomically write ``--patch-b64``;
            py_compile-check .py targets (auto-revert on syntax error).
  revert  — restore ``--target-path`` from ``--backup-path``.
  bench   — stage ``--files-b64-json`` into ``--workspace``, run
            ``--bench-command``, read back ``--result-glob`` artifacts.

Each subcommand emits a single JSON document on stdout (stderr is logs only),
matching the per-pod shape the Ray scripts produce so the sandbox-side callers
(including apply_kernel_patch.py) parse it identically. The
sandbox fans this out across pods; this script never enumerates nodes.

Stdlib only — runs in the Infera pod, which has no kernel-agent / ray checkout.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import py_compile
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
from patch_path_safety import (  # noqa: E402
    atomic_write_bytes,
    assert_backup_dir_allowed,
    assert_revert_paths_allowed,
    assert_target_path_allowed,
    finalize_patch_records,
    invalidate_aiter_jit_build,
    restore_aiter_jit_build,
)


def _pod_backup_stem(kernel_id: str, target: Path, host: str) -> str:
    """Name a pod-side backup uniquely per target and per apply.

    A constant ``kernel_id`` does not separate two targets, and a whole-second
    stamp does not separate the files of one multi-file apply.

    Args:
        kernel_id (str): Kernel identifier; falls back to the target's stem.
        target (Path): File being replaced.
        host (str): Pod the backup is taken on.

    Returns:
        str: A collision-free stem for this apply.
    """
    path_hash = hashlib.sha256(str(target).encode("utf-8")).hexdigest()[:16]
    return f"{_safe_name(kernel_id or target.stem)}_{path_hash}_{host}_{time.time_ns()}"


_MAX_ARTIFACT_BYTES = 1 * 1024 * 1024
_STREAM_TAIL_BYTES = 32 * 1024


def _safe_name(value: str) -> str:
    """Sanitize a string into a filesystem-safe filename token.

    Args:
        value (str): the raw string to sanitize.

    Returns:
        str: the sanitized token (alphanumerics and ``._-`` kept, others
            replaced with ``_``), truncated to 80 chars and defaulting to
            ``"patch"`` when empty.
    """
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)
    return cleaned[:80] or "patch"


def _emit(payload: dict) -> int:
    """Print ``payload`` as JSON to stdout and return a process exit code.

    Args:
        payload (dict): the result document to emit (its ``status`` selects the
            exit code).

    Returns:
        int: ``0`` when ``status`` is a success state (``ok`` / ``restored`` /
            ``noop_missing_backup``), else ``1``.
    """
    sys.stdout.write(json.dumps(payload, indent=2) + "\n")
    sys.stdout.flush()
    return 0 if str(payload.get("status", "")).lower() in ("ok", "restored", "noop_missing_backup") else 1


def _do_apply(a: argparse.Namespace) -> int:
    """Back up the target, write the base64 patch, and compile-check .py targets.

    Backs up ``--target-path`` into ``--backup-dir``, atomically writes the
    decoded ``--patch-b64``, and for ``.py`` targets runs ``py_compile`` with
    auto-revert on syntax error. Emits a JSON result document on stdout.

    Args:
        a (argparse.Namespace): parsed ``apply`` arguments (``target_path``,
            ``patch_b64``, ``backup_dir``, ``kernel_id``).

    Returns:
        int: the process exit code from emitting the result (``0`` on success).
    """
    host = socket.gethostname()
    target = Path(a.target_path)
    try:
        assert_target_path_allowed(target, must_exist=True)
        assert_backup_dir_allowed(Path(a.backup_dir))
    except ValueError as exc:
        return _emit({"status": "failed", "host": host, "error": str(exc)})
    if not target.is_file():
        return _emit({"status": "failed", "host": host, "error": f"target_path does not exist: {target}"})
    bdir = Path(a.backup_dir)
    bdir.mkdir(parents=True, exist_ok=True)
    backup_stem = _pod_backup_stem(a.kernel_id, target, host)
    backup_path = bdir / f"{backup_stem}.bak"
    shutil.copy2(target, backup_path)
    try:
        data = base64.b64decode(a.patch_b64.encode("ascii"))
    except Exception as exc:  # noqa: BLE001
        return _emit({"status": "failed", "host": host, "error": f"patch_b64 not valid base64: {exc!r}"})
    jit_build_dir = str(getattr(a, "jit_build_dir", "") or "")
    jit_backup = invalidate_aiter_jit_build(
        Path(jit_build_dir) if jit_build_dir else None,
        bdir,
        backup_stem,
    )
    compile_result: dict[str, Any] = {"status": "skipped", "reason": "non-py target"}
    try:
        atomic_write_bytes(target, data)
        if target.suffix.lower() == ".py":
            py_compile.compile(str(target), doraise=True)
            compile_result = {"status": "ok"}
    except py_compile.PyCompileError as exc:
        shutil.copy2(backup_path, target)
        restore_aiter_jit_build(jit_backup)
        return _emit(
            {
                "status": "failed",
                "host": host,
                "error": (f"py_compile failed (auto-reverted): {exc.msg}"),
            }
        )
    except Exception as exc:  # noqa: BLE001
        shutil.copy2(backup_path, target)
        restore_aiter_jit_build(jit_backup)
        return _emit({"status": "failed", "host": host, "error": str(exc)})
    return _emit(
        {
            "status": "ok",
            "host": host,
            "target_path": str(target),
            "backup_path": str(backup_path),
            "wrote_bytes": len(data),
            "compile": compile_result,
            "jit_backup": jit_backup,
        }
    )


def _do_revert(a: argparse.Namespace) -> int:
    """Restore the target file from its backup copy.

    Emits a JSON result document on stdout (``noop_missing_backup`` when the
    backup is absent, ``restored`` on success).

    Args:
        a (argparse.Namespace): parsed ``revert`` arguments (``target_path``,
            ``backup_path``).

    Returns:
        int: the process exit code from emitting the result.
    """
    host = socket.gethostname()
    records_json = getattr(a, "records_json", "") or ""
    try:
        records = json.loads(records_json or "[]")
    except json.JSONDecodeError as exc:
        return _emit(
            {
                "status": "failed",
                "host": host,
                "error": f"records_json is invalid: {exc}",
                "restored_targets": [],
            }
        )
    if not records_json and getattr(a, "backup_path", "") and not Path(a.backup_path).is_file():
        return _emit(
            {
                "status": "noop_missing_backup",
                "host": host,
                "target_path": str(getattr(a, "target_path", "")),
                "backup_path": str(a.backup_path),
            }
        )
    if not records and a.target_path and a.backup_path:
        records = [
            {
                "target_path": a.target_path,
                "backup_path": a.backup_path,
            }
        ]
    if not records:
        return _emit({"status": "failed", "host": host, "error": "empty revert records"})
    restored: list[str] = []
    jit_records: list[dict] = []
    try:
        for record in reversed(records):
            target = Path(str(record.get("target_path") or ""))
            backup = Path(str(record.get("backup_path") or ""))
            if not backup.is_file():
                raise FileNotFoundError(f"backup missing: {backup}")
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
                raise ValueError("conflicting JIT backup records")
            jit_restore = restore_aiter_jit_build(first)
    except Exception as exc:  # noqa: BLE001
        return _emit(
            {
                "status": "failed",
                "host": host,
                "error": str(exc),
                "restored_targets": restored,
            }
        )
    return _emit(
        {
            "status": "restored",
            "host": host,
            "restored_targets": restored,
            "jit_restore": jit_restore,
        }
    )


def _do_finalize(a: argparse.Namespace) -> int:
    """Delete backups for a transaction that has been accepted."""
    host = socket.gethostname()
    try:
        records = json.loads(getattr(a, "records_json", "") or "[]")
        result = finalize_patch_records(records)
    except (json.JSONDecodeError, OSError, ValueError) as exc:
        return _emit({"status": "failed", "host": host, "error": str(exc)})
    return _emit({"host": host, **result})


def _do_bench(a: argparse.Namespace) -> int:
    """Stage files into the workspace, run the bench command, read back artifacts.

    Decodes ``--files-b64-json`` into ``--workspace`` (rejecting absolute or
    ``..`` paths), runs ``--bench-command`` under bash with a timeout, then
    collects ``--result-glob`` artifacts (skipping oversized ones). Emits a JSON
    result document on stdout.

    Args:
        a (argparse.Namespace): parsed ``bench`` arguments (``workspace``,
            ``bench_command``, ``files_b64_json``, ``result_glob``,
            ``timeout_sec``).

    Returns:
        int: the process exit code from emitting the result (``0`` when the
            bench command exited zero).
    """
    host = socket.gethostname()
    ws = Path(a.workspace)
    ws.mkdir(parents=True, exist_ok=True)
    try:
        files_b64 = json.loads(a.files_b64_json or "{}")
    except json.JSONDecodeError as exc:
        return _emit({"status": "failed", "host": host, "error": f"files_b64_json not valid JSON: {exc}"})
    staged: list[str] = []
    for rel, b64 in (files_b64 or {}).items():
        if rel.startswith("/") or ".." in Path(rel).parts:
            return _emit(
                {"status": "failed", "host": host, "error": f"staging path must be relative + no '..': {rel!r}"}
            )
        dst = ws / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(base64.b64decode(b64.encode("ascii")))
        staged.append(str(dst))

    started = time.time()
    try:
        proc = subprocess.run(
            ["bash", "-lc", a.bench_command],
            cwd=str(ws),
            capture_output=True,
            text=True,
            timeout=a.timeout_sec,
            env={**os.environ, "WORKSPACE_PATH": str(ws)},
        )
        rc = proc.returncode
        out, err = proc.stdout or "", proc.stderr or ""
    except subprocess.TimeoutExpired as exc:
        rc, out = 124, (exc.stdout if isinstance(exc.stdout, str) else "") or ""
        err = f"TimeoutExpired after {a.timeout_sec}s"
    elapsed = time.time() - started

    artifacts: list[dict[str, Any]] = []
    for path in sorted(ws.glob(a.result_glob)):
        if not path.is_file():
            continue
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if size > _MAX_ARTIFACT_BYTES:
            artifacts.append({"path": str(path), "size_bytes": size, "content": None, "skipped_reason": "too large"})
            continue
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            artifacts.append(
                {"path": str(path), "size_bytes": size, "content": None, "skipped_reason": f"read failed: {exc!r}"}
            )
            continue
        try:
            parsed: Any = json.loads(content)
        except json.JSONDecodeError:
            parsed = content
        artifacts.append({"path": str(path), "size_bytes": size, "content": parsed})

    return _emit(
        {
            "status": "ok" if rc == 0 else "failed",
            "host": host,
            "workspace": str(ws),
            "staged_files": staged,
            "bench_command": a.bench_command,
            "returncode": rc,
            "elapsed_sec": round(elapsed, 3),
            "stdout_tail": (out or "")[-_STREAM_TAIL_BYTES:],
            "stderr_tail": (err or "")[-_STREAM_TAIL_BYTES:],
            "artifacts": artifacts,
        }
    )


def main() -> int:
    """Parse the subcommand and dispatch to apply / revert / bench.

    Returns:
        int: the subcommand's exit code, or ``2`` when no known subcommand
            matched (after printing help to stderr).
    """
    p = argparse.ArgumentParser(prog="kernel_node_ops.py")
    sub = p.add_subparsers(dest="command", required=True)

    ap = sub.add_parser("apply")
    ap.add_argument("--target-path", required=True)
    ap.add_argument("--patch-b64", required=True)
    ap.add_argument("--backup-dir", required=True)
    ap.add_argument("--kernel-id", default="")
    ap.add_argument("--jit-build-dir", default="")

    rp = sub.add_parser("revert")
    rp.add_argument("--target-path", default="")
    rp.add_argument("--backup-path", default="")
    rp.add_argument("--records-json", default="")

    fp = sub.add_parser("finalize")
    fp.add_argument("--records-json", required=True)

    bp = sub.add_parser("bench")
    bp.add_argument("--workspace", required=True)
    bp.add_argument("--bench-command", required=True)
    bp.add_argument("--files-b64-json", default="{}")
    bp.add_argument("--result-glob", default="*.json")
    bp.add_argument("--timeout-sec", type=int, default=600)

    a = p.parse_args()
    if a.command == "apply":
        return _do_apply(a)
    if a.command == "revert":
        return _do_revert(a)
    if a.command == "finalize":
        return _do_finalize(a)
    if a.command == "bench":
        return _do_bench(a)
    p.print_help(sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
