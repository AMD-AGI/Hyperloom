# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""GPU-type resolution helpers for the CLI.

Extracted from ``cli.py`` (phase 4). Pure helpers that detect / normalize the
AMD GPU type from args, env, and ``rocm-smi``. Imports stdlib only and must not
import ``cli`` (one-way dependency, mirroring cli_kb / cli_backends).
"""

from __future__ import annotations

import os
import subprocess

_AMD_GPU_TYPES = frozenset({"mi300x", "mi308x", "mi325x", "mi355x"})

_GFX_TO_RUNNER: dict[str, str] = {
    # Mirror Magpie/modes/benchmark/image_selector.py:138-140 so we can log resolved value at session start.
    "gfx942":  "mi300x",
    "gfx950":  "mi355x",
}


def _gpu_runner_type(gpu_type: str) -> str:
    """Return the Magpie runner label for a resolved real GPU type.

    MI308X and MI325X share the gfx942 / CDNA3 die with MI300X and reuse
    the same Magpie benchmark scripts (sglang_mi300x.sh / vllm_mi300x.sh).
    """
    normalized = str(gpu_type or "").strip().lower()
    if normalized in ("mi325x", "mi308x"):
        return "mi300x"
    return normalized

def _resolve_gpu_type(
    user_specified: str,
    probed: str,
) -> tuple[str, list[str]]:
    """Resolve effective gpu_type from a user hint and a hardware probe; pure for unit testing.

    Probe always wins on disagreement (wrong --gpu-type corrupts baseline+KB rows); user value kept
    only on probe failure. Returns ``(effective_gpu_type, warnings)``; warnings go to stderr to keep
    the ``HYPERLOOM_LAUNCH`` stdout sentinel clean.
    """
    warnings: list[str] = []
    if probed and user_specified and probed != user_specified:
        warnings.append(
            f"WARN: --gpu-type={user_specified!r} disagrees with probed "
            f"{probed!r}; using probed {probed!r}. The probe wins because "
            f"Magpie runner_type + KB recipe rows must match the actual "
            f"hardware to keep baseline numbers comparable across sessions."
        )
        return probed, warnings
    return (probed or user_specified), warnings

def _autodetect_gpu_type() -> str | None:
    """Return mi300x|mi308x|mi325x|mi355x or None if undetectable (rocm-smi then torch gcnArchName, best-effort)."""
    import subprocess
    try:
        out = subprocess.run(
            ["rocm-smi", "--showproductname"],
            capture_output=True, text=True, timeout=5,
        ).stdout.upper()
        for tag in ("MI355X", "MI325X", "MI308X", "MI300X"):
            if tag in out:
                return tag.lower()
    except (FileNotFoundError, subprocess.TimeoutExpired, PermissionError, OSError):
        pass
    try:
        import torch
        arch = torch.cuda.get_device_properties(0).gcnArchName
        gfx = arch.split(":", 1)[0].lower()
        return _GFX_TO_RUNNER.get(gfx)
    except Exception:  # noqa: BLE001
        return None

def _resolve_amd_gpu_type(explicit: str | None = None) -> str | None:
    """Resolve the current AMD GPU type, or None when not on AMD/unknown.

    Resolution order (most authoritative first): an explicit ``gpu_type``
    argument, the ``GPU_TYPE`` env, then a best-effort runtime autodetect.
    Returning the resolved value only when it names a known AMD runner lets
    callers gate AMD-specific behaviour on real hardware while still honouring
    a launcher/CI-supplied ``gpu_type`` even if ``rocm-smi``/torch probing is
    unavailable at the call site.
    """
    for cand in (explicit, os.environ.get("GPU_TYPE")):
        norm = str(cand or "").strip().lower()
        if norm in _AMD_GPU_TYPES:
            return norm
    detected = (_autodetect_gpu_type() or "").strip().lower()
    return detected if detected in _AMD_GPU_TYPES else None

