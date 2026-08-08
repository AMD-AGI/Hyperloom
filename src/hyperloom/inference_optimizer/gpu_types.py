# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""AMD GPU-type helpers shared by the CLI and orchestrator runtime."""

from __future__ import annotations

import os
import sys


_AMD_GPU_TYPES = frozenset({
    "mi300x", "mi308x", "mi325x", "mi355x",
    "rx9070xt", "rx9070", "rx9060xt", "r9000"
})

_GFX_TO_RUNNER: dict[str, str] = {
    "gfx942": "mi300x",
    "gfx950": "mi355x",
    "gfx1201": "rx9070xt",
    "gfx1203": "r9000",
    "gfx1206": "rx9060xt",
    "gfx1207": "rx9070",
}

_AMD_GPU_DISPATCH_IDENTITIES: dict[str, tuple[str, int]] = {
    "mi300x": ("gfx942", 304),
    "mi308x": ("gfx942", 304),
    "mi325x": ("gfx942", 304),
    "mi355x": ("gfx950", 256),
    "rx9070xt": ("gfx1201", 64),
    "r9000": ("gfx1203", 64),
    "rx9070": ("gfx1207", 56),
    "rx9060xt": ("gfx1206", 32),
}

# GPU types for which Magpie ships a benchmark runner script (sglang_<runner>.sh).
# MI308X/MI325X are mapped to the MI300X runner by _gpu_runner_type(); every
# RDNA4 SKU resolves to itself. The matching rx9xxx runner scripts are shipped
# by the AMD-AGI/Magpie package (see docs/components/magpie.md).
_SHIPPED_MAGPIE_RUNNERS: frozenset[str] = frozenset({
    "mi300x", "mi355x",
    "rx9070xt", "rx9070", "rx9060xt", "r9000",
})


def _gpu_runner_type(gpu_type: str) -> str:
    """Return the Magpie runner label for a resolved real GPU type.

    MI308X/MI325X collapse to the MI300X runner (shipped as
    ``sglang_mi300x.sh``); every other supported AMD type resolves to itself.
    """
    normalized = str(gpu_type or "").strip().lower()
    if normalized in ("mi300x", "mi308x", "mi325x"):
        return "mi300x"
    return normalized


def _resolve_gpu_type(
    user_specified: str,
    probed: str,
) -> tuple[str, list[str]]:
    """Resolve effective gpu_type from a user hint and a hardware probe."""
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
    """Return mi300x|mi308x|mi325x|mi355x|rx9070xt|rx9070|rx9060xt|r9000 or None."""
    import subprocess

    try:
        if sys.platform == "win32":
            out = subprocess.run(
                ["hipConfig", "--show-device"],
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout.upper()
            # Match whole device names; bare "RX 9070" (no " XT") must still
            # resolve to rx9070 rather than collapsing into the empty set.
            for tag in (
                "RX 9070 XT", "R9000", "RX 9060 XT",
                "RX 9070 ", "RX 9070", "RX 9060",
                "MI355X", "MI300X",
            ):
                if tag in out:
                    return tag.replace("MI", "mi").replace("RX ", "rx").replace(" XT", "xt").replace(" ", "").lower()
            # Fallbacks for "AMD Radeon RX 9070" / "AMD Radeon Pro R9000" (full
            # product strings) so bare non-XT SKUs are detected.
            import re as _re
            bare = _re.search(r"AMD\s+Radeon\s+(Pro\s+R9000|RX\s+\d{3,4}\s*XT|RX\s+\d{3,4})\b", out)
            if bare:
                tok = bare.group(1).replace("Pro ", "Pro_").replace("XT", "xt").replace(" ", "").replace("_", "").replace("RX", "rx").lower()
                return "r9000" if "pro" in tok or "r9000" in tok else tok
        else:
            out = subprocess.run(
                ["rocm-smi", "--showproductname"],
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout.upper()
            for tag in ("MI355X", "MI325X", "MI308X", "MI300X", "RX9070XT", "RX9070", "RX9060XT", "R9000"):
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
    """Resolve the current AMD GPU type, or None when not on AMD/unknown."""
    explicit_norm = str(explicit or "").strip().lower()
    if explicit_norm:
        return explicit_norm if explicit_norm in _AMD_GPU_TYPES else None
    env_norm = os.environ.get("GPU_TYPE", "").strip().lower()
    if env_norm:
        return env_norm if env_norm in _AMD_GPU_TYPES else None
    detected = (_autodetect_gpu_type() or "").strip().lower()
    return detected if detected in _AMD_GPU_TYPES else None


def amd_gpu_dispatch_identity(gpu_type: str | None = None) -> tuple[str, int] | None:
    """Return the AITER dispatch architecture and CU count for an AMD GPU."""
    resolved = _resolve_amd_gpu_type(gpu_type)
    if not resolved:
        return None
    return _AMD_GPU_DISPATCH_IDENTITIES.get(resolved)
