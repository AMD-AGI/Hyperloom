# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""AMD GPU-type helpers shared by the CLI and orchestrator runtime."""

from __future__ import annotations

import os

from hyperloom.common.gpu_identity import AMD_GPU_DISPATCH_IDENTITIES

#: Derived, not typed out again: a board added to the identities table is
#: accepted by the CLI and resolvable in the same commit. Listing it separately
#: let a board have a dispatch identity that ``_resolve_amd_gpu_type`` refused.
_AMD_GPU_TYPES = frozenset(AMD_GPU_DISPATCH_IDENTITIES)

#: rocm-smi product tags, reverse-sorted so a longer tag is tested before any
#: tag that is a prefix of it -- an "MI300XL" must not be claimed by "MI300X".
_PRODUCT_TAGS: tuple[str, ...] = tuple(sorted((t.upper() for t in _AMD_GPU_TYPES), reverse=True))

_GFX_TO_RUNNER: dict[str, str] = {
    # gfx arch -> Magpie runner label, so launchers and runtime materializers
    # agree on the selected benchmark script. Not derived by inverting the
    # identities table: that map is many-to-one (mi308x and mi325x are gfx942
    # too), and this answers "which benchmark script", not "which board", so
    # the arch a runner is reached by is a deliberate choice, not an inverse.
    "gfx942": "mi300x",
    "gfx950": "mi355x",
    "gfx1151": "radeon8060s",
}

#: Re-exported from ``hyperloom.common`` so provenance and this module cannot
#: disagree about which arch a board dispatches to.
_AMD_GPU_DISPATCH_IDENTITIES = AMD_GPU_DISPATCH_IDENTITIES


def _gpu_runner_type(gpu_type: str) -> str:
    """Return the Magpie runner label for a resolved real GPU type."""
    normalized = str(gpu_type or "").strip().lower()
    if normalized in ("mi325x", "mi308x"):
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
    """Return a known AMD board label, or None if undetectable."""
    import subprocess

    try:
        out = subprocess.run(
            ["rocm-smi", "--showproductname"],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.upper()
        for tag in _PRODUCT_TAGS:
            if tag in out:
                return tag.lower()
    except (FileNotFoundError, subprocess.TimeoutExpired, PermissionError, OSError):
        # rocm-smi missing / slow / not permitted; fall through to the torch
        # gcnArchName probe below (autodetect is best-effort).
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
