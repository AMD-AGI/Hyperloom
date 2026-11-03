#!/usr/bin/env python3
"""Single-pod, Ray-free kernel ops for the Infera backend (SSH control plane)."""

from __future__ import annotations

import argparse
import json
import py_compile
import socket
import sys
from pathlib import Path

from patch_path_safety import apply_patch_on_pod, finalize_patch_records, revert_patch_records


def _emit(payload: dict) -> int:
    """Print ``payload`` as JSON to stdout and return a process exit code."""
    sys.stdout.write(json.dumps(payload, indent=2) + "\n")
    sys.stdout.flush()
    return 0 if str(payload.get("status", "")).lower() in ("ok", "restored", "finalized") else 1


def _do_apply(a: argparse.Namespace) -> int:
    """Back up the target, write the base64 patch, and compile-check .py targets."""
    host = socket.gethostname()
    try:
        record = apply_patch_on_pod(
            Path(a.target_path), a.patch_b64, Path(a.backup_dir), a.kernel_id, getattr(a, "jit_build_dir", "") or ""
        )
    except py_compile.PyCompileError as exc:
        return _emit({"status": "failed", "host": host, "error": f"py_compile failed (auto-reverted): {exc.msg}"})
    except (OSError, ValueError) as exc:
        return _emit({"status": "failed", "host": host, "error": str(exc)})
    return _emit({"status": "ok", **record})


def _do_revert(a: argparse.Namespace) -> int:
    """Restore every recorded target from its backup copy."""
    host = socket.gethostname()
    try:
        result = revert_patch_records(json.loads(a.records_json))
    except (OSError, ValueError) as exc:
        return _emit({"status": "failed", "host": host, "error": str(exc)})
    return _emit({"host": host, **result})


def _do_finalize(a: argparse.Namespace) -> int:
    """Delete backups for a transaction that has been accepted."""
    host = socket.gethostname()
    try:
        records = json.loads(getattr(a, "records_json", "") or "[]")
        result = finalize_patch_records(records)
    except (json.JSONDecodeError, OSError, ValueError) as exc:
        return _emit({"status": "failed", "host": host, "error": str(exc)})
    return _emit({"host": host, **result})


def main() -> int:
    """Parse the subcommand and dispatch to apply / revert / finalize."""
    p = argparse.ArgumentParser(prog="kernel_node_ops.py")
    sub = p.add_subparsers(dest="command", required=True)

    ap = sub.add_parser("apply")
    ap.add_argument("--target-path", required=True)
    ap.add_argument("--patch-b64", required=True)
    ap.add_argument("--backup-dir", required=True)
    ap.add_argument("--kernel-id", default="")
    ap.add_argument("--jit-build-dir", default="")

    rp = sub.add_parser("revert")
    rp.add_argument("--records-json", required=True)

    fp = sub.add_parser("finalize")
    fp.add_argument("--records-json", required=True)

    a = p.parse_args()
    if a.command == "apply":
        return _do_apply(a)
    if a.command == "revert":
        return _do_revert(a)
    if a.command == "finalize":
        return _do_finalize(a)
    p.print_help(sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
