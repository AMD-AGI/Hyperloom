# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Device state for ``preflight``'s serving-framework probe: the probe record, rocm-smi, and the block."""

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


def _device_visibility_split(python_exe: str | None, visible: int | None) -> str:
    """State what rocm-smi and the probe each saw, with the masks in force; draw no conclusion from the split."""
    masks = (
        ", ".join(
            f"{name}={os.environ[name]!r}"
            for name in ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")
            if name in os.environ
        )
        or "no visible-device mask set"
    )
    seen = "rocm-smi could not be read" if visible is None else f"rocm-smi reports {visible} GPU(s)"
    probe = f"the probe in {python_exe} reached no verdict" if python_exe else "the probe reached no verdict"
    return f"{seen} while {probe}; masks in force: {masks}."


def _block_on_unverified_build(
    framework: str, python_exe: str | None, detail: str, visible: int | None, skip_env: str
) -> NoReturn:
    """Stop the run on a probe that reached no verdict, with its stderr and what rocm-smi saw."""
    # Only a clean rc 0 says the stack behind ``framework`` works here; every other outcome -- a traceback, a
    # signal death, a timeout -- is a classification this preflight cannot make, and the serving process imports
    # the same stack. Stop with the evidence rather than guess which failures are harmless.
    state = (
        f"is importable ({python_exe}) but the ROCm probe there reached no verdict"
        if python_exe
        else "could not be probed on this host"
    )
    print(
        f"\nERROR: {framework} {state}.{detail}\n\n"
        f"{_device_visibility_split(python_exe, visible)}\n"
        "The serving process imports the same stack, so it would hit whatever the\n"
        "probe hit, after the benchmark has already been set up. The probe stderr\n"
        "above is the failure to fix; a timeout there means the import is slower\n"
        "than the probe budget or wedged, not that it is healthy.\n\n"
        f"To proceed anyway, set {skip_env}=1.",
        file=sys.stderr,
    )
    raise SystemExit(2)
