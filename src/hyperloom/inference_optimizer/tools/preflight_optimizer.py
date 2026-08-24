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


STALE_PROCESS_PATTERNS = (
    "hyperloom.inference_optimizer.cli",
    "Magpie",
    "sglang.launch_server",
    "vllm.entrypoints",
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
    """Check VRAM occupancy via rocm-smi and return whether all GPUs are clean.

    A GPU is considered busy when its used VRAM exceeds ``VRAM_BUSY_FRACTION``
    of its total capacity. rocm-smi absent or non-zero exit is treated as an
    unknown and reported as failure so the caller aborts rather than launching
    into an unverified state.

    Returns:
        ``True`` when every GPU's used VRAM is within the allowed fraction,
        ``False`` on any occupancy violation or when the GPU state is unknown.
    """
    from hyperloom.common.rocm_smi import gpu_vram_usage

    snapshots = gpu_vram_usage()
    if snapshots is None:
        print("rocm_smi=unavailable")
        return False

    clean = True
    for idx, snap in enumerate(snapshots):
        if snap.total_mib is None or snap.total_mib == 0.0:
            print(f"gpu{idx}_vram=unknown (no total reported)")
            clean = False
            continue
        pct = snap.used_mib / snap.total_mib
        threshold_mib = snap.total_mib * VRAM_BUSY_FRACTION
        print(
            f"gpu{idx}_vram_used={snap.used_mib:.1f} MiB  "
            f"total={snap.total_mib:.1f} MiB  "
            f"pct={pct:.3%}  threshold={threshold_mib:.1f} MiB"
        )
        if snap.used_mib > threshold_mib:
            clean = False
    return clean


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
