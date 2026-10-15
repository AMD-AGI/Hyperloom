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

#: Product tags matched against amd-smi's marketing name, reverse-sorted so a longer tag is tested before any tag
#: that is a prefix of it -- an "MI300XL" must not be claimed by "MI300X".
_PRODUCT_TAGS: tuple[str, ...] = tuple(sorted((t.upper() for t in _AMD_GPU_TYPES), reverse=True))

_GFX_TO_RUNNER: dict[str, str] = {
    # gfx arch -> Magpie runner label, so launchers and runtime materializers agree on the selected benchmark script.
    "gfx942": "mi300x",
    "gfx950": "mi355x",
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
    """Return mi300x|mi308x|mi325x|mi355x|radeon8060s|... or None if undetectable.

    ``amd-smi static --asic`` replaces the old ``rocm-smi --showproductname`` plus torch ``gcnArchName`` pair: one
    call returns both the marketing name and the gfx target, where ``rocm-smi`` gave only the former -- and on
    gfx1151 gave nothing usable either way: ``Card Series: N/A`` on ROCm 7.2.3, the generic ``AMD Radeon Graphics``
    on 7.15. Neither matches a product tag, so the board resolved to ``unknown_gpu`` (see PR #1483).

    Note the marketing name is not always resolved on the older stack either: amd-smi 26.2.2 (ROCm 7.2.3) reports
    ``market_name`` as the bare device id ``0x1586`` for this part, which is why the ``target_graphics_version`` tier
    below exists rather than being treated as redundant.
    """
    from hyperloom.common.amd_smi import gpu_asic_info

    asics = gpu_asic_info()
    if asics:
        asic = asics[0]
        # Spaces dropped so a derived tag matches the spelling amd-smi prints: radeon8060s appears as "AMD Radeon
        # 8060S Graphics".
        market = asic.market_name.upper().replace(" ", "")
        for tag in _PRODUCT_TAGS:
            if tag in market:
                return tag.lower()
        # No marketing-name tag matched -- e.g. an unreleased board whose market_name degrades to a bare device id.
        # amd-smi already returned the gfx target in this same call, so try that before falling to torch below.
        gfx = str(asic.target_graphics_version or "").strip().lower()
        if gfx in _GFX_TO_RUNNER:
            return _GFX_TO_RUNNER[gfx]
    # amd-smi is absent/failed, or answered with neither a known tag nor a known gfx target: fall back to torch's
    # own gcnArchName probe. This no longer fills a gap in amd-smi's data (static --asic already carries
    # target_graphics_version), but it is still the only read that works in a container image that ships torch+ROCm
    # without the amd-smi CLI, so it keeps earning its place as a last resort.
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
    env_norm = (os.environ.get("TARGET_GPU_TYPE", "") or os.environ.get("GPU_TYPE", "")).strip().lower()
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
