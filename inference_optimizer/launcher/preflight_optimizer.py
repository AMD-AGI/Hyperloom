#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc. All rights reserved.
"""Launcher-side preflight for inference_optimizer.

Usage:
    python inference_optimizer/launcher/preflight_optimizer.py MODEL_PATH
"""
from __future__ import annotations

import argparse
import os
import pathlib
import shutil
import subprocess
import sys


STALE_PROCESS_PATTERNS = (
    "inference_optimizer.cli",
    "Magpie",
    "sglang.launch_server",
    "vllm.entrypoints",
)


def _read_cmdline(pid: str) -> str:
    try:
        raw = pathlib.Path("/proc", pid, "cmdline").read_bytes()
    except OSError:
        return ""
    return raw.replace(b"\0", b" ").decode("utf-8", "ignore")


def _print_torch_visibility() -> bool:
    try:
        import torch  # type: ignore[import-not-found]
    except Exception as exc:
        print("torch_check_error=", type(exc).__name__, str(exc)[:300])
        return False

    available = bool(torch.cuda.is_available())
    count = int(torch.cuda.device_count())
    print("torch_cuda_available=", available)
    print("torch_cuda_device_count=", count)
    return available and count > 0


def _print_rocm_snapshot() -> None:
    rocm_smi = shutil.which("rocm-smi")
    if not rocm_smi:
        print("rocm_smi=missing")
        return

    try:
        result = subprocess.run(
            [rocm_smi, "--showmemuse", "--json"],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
        )
    except Exception as exc:
        print("rocm_smi_error=", type(exc).__name__, str(exc)[:300])
        return

    print("rocm_smi_rc=", result.returncode)
    if result.stdout.strip():
        print(result.stdout.strip()[:2000])
    if result.stderr.strip():
        print("rocm_smi_stderr=", result.stderr.strip()[:500])


def _find_stale_processes() -> list[tuple[str, str]]:
    matches: list[tuple[str, str]] = []
    for pid in filter(str.isdigit, os.listdir("/proc")):
        if int(pid) == os.getpid():
            continue
        text = _read_cmdline(pid)
        if text and any(pattern in text for pattern in STALE_PROCESS_PATTERNS):
            matches.append((pid, text[:300]))
    return matches


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_path", help="Model directory to optimize.")
    args = parser.parse_args()

    model_path = pathlib.Path(args.model_path)
    ok = True

    if not model_path.is_dir():
        print(f"model_path_missing={model_path}", file=sys.stderr)
        ok = False
    else:
        print(f"model_path_ok={model_path}")

    if not _print_torch_visibility():
        ok = False

    _print_rocm_snapshot()

    stale = _find_stale_processes()
    for pid, cmdline in stale:
        print(f"existing_process {pid}: {cmdline}")
    if stale:
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
