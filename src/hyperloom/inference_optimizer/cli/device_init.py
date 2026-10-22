# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Device-init detection for ``preflight``'s serving-framework probe: the flag, its stderr match, and the block."""

from __future__ import annotations

import os
import subprocess
import sys
from typing import NamedTuple, NoReturn


class _Probe(NamedTuple):
    """Probe outcome plus the stderr tail needed to diagnose an unclear one."""

    verdict: bool | None
    detail: str = ""
    timed_out: bool = False
    device_init_failed: bool = False


# A probe that reached torch and then could not initialise a single device is not an unclear verdict: the serving
# process dies on the same import. These are the messages torch raises when the ROCm/HIP runtime enumerates no
# usable device (the ROCm build keeps torch's CUDA wording).
_DEVICE_INIT_FAILURE_MARKERS = (
    "no cuda gpus are available",
    "no rocm-capable device is detected",
    "cuda driver initialization failed",
    "invalid device ordinal",
)


def _is_device_init_failure(stderr: str | None) -> bool:
    """Whether a probe's stderr shows torch finding no usable GPU."""
    text = (stderr or "").lower()
    return any(marker in text for marker in _DEVICE_INIT_FAILURE_MARKERS)


def _unclear_probe(detail: str, stderr: str | None) -> _Probe:
    """An unclear verdict, flagged when its traceback shows torch initialising zero devices."""
    return _Probe(None, detail, device_init_failed=_is_device_init_failure(stderr))


def _preferred_unclear(kept: tuple[str, _Probe] | None, candidate: tuple[str, _Probe]) -> tuple[str, _Probe]:
    """Rank unclear candidates: a reproduced device-init failure outranks a merely unclear probe, else the first."""
    if candidate[1].device_init_failed and not (kept and kept[1].device_init_failed):
        return candidate
    return kept or candidate


def _rocm_smi_visible_gpus() -> tuple[int | None, str | None]:
    """GPU count ``rocm-smi`` reports, else ``(None, skip_reason)`` when it cannot answer."""
    try:
        proc = subprocess.run(
            ["rocm-smi", "--showid"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, PermissionError, OSError):
        return None, "rocm_smi_unavailable"
    if proc.returncode != 0:
        return None, "rocm_smi_failed"
    # rocm-smi --showid emits multiple GPU[ lines per GPU; deduplicate by GPU index.
    visible_indices: set[str] = set()
    for line in (proc.stdout or "").splitlines():
        stripped = line.strip()
        if stripped.startswith("GPU["):
            idx, _, _ = stripped[4:].partition("]")
            if idx:
                visible_indices.add(idx)
    return len(visible_indices), None


def _device_visibility_contradiction(python_exe: str, visible: int | None) -> str:
    """Name the rocm-smi-sees-N / torch-sees-none split, with the masks in force."""
    masks = (
        ", ".join(
            f"{name}={os.environ[name]!r}"
            for name in ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")
            if name in os.environ
        )
        or "no visible-device mask set"
    )
    seen = "rocm-smi could not be read" if visible is None else f"rocm-smi reports {visible} GPU(s)"
    return f"{seen} while torch in {python_exe} initialises none; masks in force: {masks}."


def _block_on_device_init(framework: str, python_exe: str, detail: str, visible: int | None, skip_env: str) -> NoReturn:
    """Stop the run on a reproduced device-init failure, naming the traceback and the visibility contradiction."""
    # Not "could not verify": the probe reproduced, in the very interpreter that will launch the server, the
    # device-init failure the server dies on. No TP value, retry or config change repairs it, so the run stops
    # here with the traceback instead of admitting a baseline that cannot start.
    contradiction = _device_visibility_contradiction(python_exe, visible)
    print(
        f"\nERROR: {framework} is importable ({python_exe}) but torch there cannot initialize any GPU."
        f"{detail}\n\n"
        f"{contradiction}\n"
        "The serving process imports the same stack, so it would die the same way\n"
        "before its port is up. This is a node or container device-access problem\n"
        "(device files, ROCr init, visible-device masking or partitioning), not a\n"
        "benchmark-configuration one; restore GPU access on this host, then retry.\n\n"
        f"To proceed anyway, set {skip_env}=1.",
        file=sys.stderr,
    )
    raise SystemExit(2)
