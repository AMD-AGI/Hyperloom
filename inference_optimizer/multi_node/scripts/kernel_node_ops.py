#!/usr/bin/env python3
"""Single-pod, Ray-free kernel ops for the Dynamo backend (SSH control plane).

Ray-free counterpart to ``kernel_patch_multinode.py`` + ``kernel_bench_multinode.py``.
The Dynamo backend has no Ray cluster, so ``inference_optimizer.multi_node``
ships this script to each GPU pod over SSH and runs ONE subcommand per pod:

  apply   — back up ``--target-path`` then atomically write ``--patch-b64``;
            py_compile-check .py targets (auto-revert on syntax error).
  revert  — restore ``--target-path`` from ``--backup-path``.
  bench   — stage ``--files-b64-json`` into ``--workspace``, run
            ``--bench-command``, read back ``--result-glob`` artifacts.

Each subcommand emits a single JSON document on stdout (stderr is logs only),
matching the per-pod shape the Ray scripts produce so the sandbox-side callers
(apply_kernel_patch.py / kernel_optimization.py) parse it identically. The
sandbox fans this out across pods; this script never enumerates nodes.

Stdlib only — runs in the Dynamo pod, which has no kernel-agent / ray checkout.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import py_compile
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

_MAX_ARTIFACT_BYTES = 1 * 1024 * 1024
_STREAM_TAIL_BYTES = 32 * 1024


def _safe_name(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)
    return cleaned[:80] or "patch"


def _atomic_write_bytes(target: Path, data: bytes) -> None:
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


def _emit(payload: dict) -> int:
    sys.stdout.write(json.dumps(payload, indent=2) + "\n")
    sys.stdout.flush()
    return 0 if str(payload.get("status", "")).lower() in ("ok", "restored",
                                                            "noop_missing_backup") else 1


def _do_apply(a: argparse.Namespace) -> int:
    host = socket.gethostname()
    target = Path(a.target_path)
    if not target.is_file():
        return _emit({"status": "failed", "host": host,
                      "error": f"target_path does not exist: {target}"})
    bdir = Path(a.backup_dir)
    bdir.mkdir(parents=True, exist_ok=True)
    backup_path = bdir / (
        f"{_safe_name(a.kernel_id or target.stem)}_{host}_{int(time.time())}.bak"
    )
    shutil.copy2(target, backup_path)
    try:
        data = base64.b64decode(a.patch_b64.encode("ascii"))
    except Exception as exc:  # noqa: BLE001
        return _emit({"status": "failed", "host": host,
                      "error": f"patch_b64 not valid base64: {exc!r}"})
    _atomic_write_bytes(target, data)
    compile_result: dict[str, Any] = {"status": "skipped", "reason": "non-py target"}
    if target.suffix.lower() == ".py":
        try:
            py_compile.compile(str(target), doraise=True)
            compile_result = {"status": "ok"}
        except py_compile.PyCompileError as exc:
            shutil.copy2(backup_path, target)
            return _emit({"status": "failed", "host": host,
                          "error": f"py_compile failed (auto-reverted): {exc.msg}"})
    return _emit({
        "status": "ok", "host": host, "target_path": str(target),
        "backup_path": str(backup_path), "wrote_bytes": len(data),
        "compile": compile_result,
    })


def _do_revert(a: argparse.Namespace) -> int:
    host = socket.gethostname()
    target = Path(a.target_path)
    backup = Path(a.backup_path)
    if not backup.is_file():
        return _emit({"status": "noop_missing_backup", "host": host,
                      "target_path": str(target), "backup_path": str(backup)})
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(backup, target)
    return _emit({"status": "restored", "host": host,
                  "target_path": str(target), "backup_path": str(backup)})


def _do_bench(a: argparse.Namespace) -> int:
    host = socket.gethostname()
    ws = Path(a.workspace)
    ws.mkdir(parents=True, exist_ok=True)
    try:
        files_b64 = json.loads(a.files_b64_json or "{}")
    except json.JSONDecodeError as exc:
        return _emit({"status": "failed", "host": host,
                      "error": f"files_b64_json not valid JSON: {exc}"})
    staged: list[str] = []
    for rel, b64 in (files_b64 or {}).items():
        if rel.startswith("/") or ".." in Path(rel).parts:
            return _emit({"status": "failed", "host": host,
                          "error": f"staging path must be relative + no '..': {rel!r}"})
        dst = ws / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(base64.b64decode(b64.encode("ascii")))
        staged.append(str(dst))

    started = time.time()
    try:
        proc = subprocess.run(
            ["bash", "-lc", a.bench_command], cwd=str(ws),
            capture_output=True, text=True, timeout=a.timeout_sec,
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
            artifacts.append({"path": str(path), "size_bytes": size,
                              "content": None, "skipped_reason": "too large"})
            continue
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            artifacts.append({"path": str(path), "size_bytes": size,
                              "content": None, "skipped_reason": f"read failed: {exc!r}"})
            continue
        try:
            parsed: Any = json.loads(content)
        except json.JSONDecodeError:
            parsed = content
        artifacts.append({"path": str(path), "size_bytes": size, "content": parsed})

    return _emit({
        "status": "ok" if rc == 0 else "failed",
        "host": host, "workspace": str(ws), "staged_files": staged,
        "bench_command": a.bench_command, "returncode": rc,
        "elapsed_sec": round(elapsed, 3),
        "stdout_tail": (out or "")[-_STREAM_TAIL_BYTES:],
        "stderr_tail": (err or "")[-_STREAM_TAIL_BYTES:],
        "artifacts": artifacts,
    })


def main() -> int:
    p = argparse.ArgumentParser(prog="kernel_node_ops.py")
    sub = p.add_subparsers(dest="command", required=True)

    ap = sub.add_parser("apply")
    ap.add_argument("--target-path", required=True)
    ap.add_argument("--patch-b64", required=True)
    ap.add_argument("--backup-dir", required=True)
    ap.add_argument("--kernel-id", default="")

    rp = sub.add_parser("revert")
    rp.add_argument("--target-path", required=True)
    rp.add_argument("--backup-path", required=True)

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
    if a.command == "bench":
        return _do_bench(a)
    p.print_help(sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
