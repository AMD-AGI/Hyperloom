#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Launcher-side preflight for hyperloom.inference_optimizer.

Usage:
    python src/hyperloom/inference_optimizer/tools/preflight_optimizer.py MODEL_PATH
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys

from hyperloom.common import rocm_smi


#: Command-line fragments that identify a leftover optimizer or serving
#: process. Both frameworks are matched twice over: by the module path an older
#: launch used, and by the ``setproctitle`` name the current one rewrites argv
#: to. Magpie launches the server as ``vllm serve``, and vLLM then renames its
#: own processes to ``VLLM::APIServer`` / ``VLLM::EngineCore`` /
#: ``VLLM::Worker_TP<n>``, so a scan for ``vllm.entrypoints`` alone sees
#: nothing. That blind spot is not covered by the VRAM check either: an orphan
#: that is still reading weights holds no VRAM yet and reads as idle.
STALE_PROCESS_PATTERNS = (
    "hyperloom.inference_optimizer.cli",
    "Magpie",
    "sglang.launch_server",
    "sglang::",
    "vllm.entrypoints",
    "vllm serve",
    "VLLM::",
)

VRAM_BUSY_FRACTION = 0.01


def _read_cmdline(pid: str) -> str:
    """Read and decode the ``/proc/<pid>/cmdline`` for the given pid.

    Args:
        pid: The process id (as a string) whose command line to read.

    Returns:
        The space-joined command line, or ``""`` when it cannot be read.
    """
    try:
        raw = pathlib.Path("/proc", pid, "cmdline").read_bytes()
    except OSError:
        return ""
    return raw.replace(b"\0", b" ").decode("utf-8", "ignore")


def _print_torch_visibility() -> bool:
    """Print torch CUDA visibility and report whether a device is usable.

    Returns:
        ``True`` when torch imports and reports at least one available CUDA
        device, ``False`` otherwise (including on import failure).
    """
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


def _check_gpu_occupancy() -> bool:
    """Print per-GPU VRAM usage and report whether every GPU is idle.

    A GPU is busy once its used VRAM exceeds :data:`VRAM_BUSY_FRACTION` of its
    own capacity. An unreadable GPU state counts as a failure, so an
    unverifiable launch aborts instead of proceeding.

    Returns:
        ``True`` when every GPU is under the fraction, ``False`` otherwise.
    """
    snapshots = rocm_smi.gpu_vram_usage()
    if snapshots is None:
        print("gpu_vram=unreadable")
        return False

    idle = True
    for idx, snap in enumerate(snapshots):
        busy = snap.used_mib > snap.total_mib * VRAM_BUSY_FRACTION
        print(
            f"gpu{idx}_vram_used={snap.used_mib:.1f}/{snap.total_mib:.1f} MiB"
            f" ({snap.used_mib / snap.total_mib:.2%}) {'BUSY' if busy else 'idle'}"
        )
        idle = idle and not busy
    return idle


def _find_stale_processes() -> list[tuple[str, str]]:
    """Scan ``/proc`` for running processes matching known stale patterns.

    Returns:
        A list of ``(pid, cmdline)`` tuples for processes (other than this one)
        whose command line matches one of :data:`STALE_PROCESS_PATTERNS`.
    """
    matches: list[tuple[str, str]] = []
    for pid in filter(str.isdigit, os.listdir("/proc")):
        if int(pid) == os.getpid():
            continue
        text = _read_cmdline(pid)
        if text and any(pattern in text for pattern in STALE_PROCESS_PATTERNS):
            matches.append((pid, text[:300]))
    return matches


def main() -> int:
    """Run launcher preflight checks and return a process exit code.

    Validates the model path, checks torch/ROCm visibility, verifies GPU VRAM
    occupancy is below the allowed threshold, and reports any stale
    optimizer/server processes.

    Returns:
        ``0`` when every check passes, otherwise ``2``.
    """
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

    if not _check_gpu_occupancy():
        ok = False

    stale = _find_stale_processes()
    for pid, cmdline in stale:
        print(f"existing_process {pid}: {cmdline}")
    if stale:
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
