# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""AMD GPU-type helpers shared by the CLI and orchestrator runtime."""

from __future__ import annotations

import os


# The MI parts. Besides naming Magpie runner scripts these gate the
# CDNA-specific paths in _workload_envs / _grid_server_args / phases.kernel via
# _resolve_amd_gpu_type, so membership must stay CDNA-only.
_AMD_GPU_TYPES = frozenset({"mi300x", "mi308x", "mi325x", "mi355x"})

# gfx11 SKUs, which all share Magpie's single ``{framework}_gfx11.sh`` runner
# but each need their OWN recipe-KB namespace. Without them every gfx11 card
# keyed to ``unknown_gpu``, so a recipe learned on a Strix Halo iGPU was read
# back on any other unrecognised GPU. No CDNA fast path is enabled for them.
_GFX11_GPU_TYPES = frozenset({"8050s", "8060s", "890m", "gfx1150", "gfx1151"})

# Accepted --gpu-type values. Single source of truth shared with the CLI parser
# so a hint the probe can produce is never rejected by argparse.
GPU_TYPE_CHOICES: tuple[str, ...] = tuple(sorted(_AMD_GPU_TYPES | _GFX11_GPU_TYPES))

# Uppercased rocm-smi product-name needles -> gpu_type, most-specific first so
# MI325X is not shadowed by an MI300X substring match. Consumer SKUs carry the
# "RADEON " prefix they are actually printed with ("Card Series: AMD Radeon
# 8060S Graphics") so a bare number cannot collide with a GUID or card model.
_PRODUCT_TAGS: tuple[tuple[str, str], ...] = (
    ("MI355X", "mi355x"),
    ("MI325X", "mi325x"),
    ("MI308X", "mi308x"),
    ("MI300X", "mi300x"),
    ("RADEON 8060S", "8060s"),
    ("RADEON 8050S", "8050s"),
    ("RADEON 890M", "890m"),
)

_GFX_TO_GPU_TYPE: dict[str, str] = {
    # gfx arch -> gpu_type, used when no product name is available (torch
    # gcnArchName) or when the product name names an untabulated SKU.
    "gfx942": "mi300x",
    "gfx950": "mi355x",
    # gfx11 iGPUs resolve to the arch itself: one arch spans several SKUs
    # (gfx1151 is both 8060S and 8050S), so the arch is the most specific
    # identity that a bare gcnArchName can honestly support. rocm-smi's product
    # name is preferred above and yields the finer 8060s/8050s key.
    "gfx1150": "gfx1150",
    "gfx1151": "gfx1151",
}

_GFX_TO_RUNNER: dict[str, str] = {
    # gfx arch -> Magpie runner label, so launchers and runtime materializers
    # agree on the selected benchmark script. Derived from _GFX_TO_GPU_TYPE and
    # MI-only: the remote multi-node probe that reads this resolves MI clusters,
    # and its result is also used as a gpu_type, which "gfx11" is not.
    gfx: gpu_type
    for gfx, gpu_type in _GFX_TO_GPU_TYPE.items()
    if gpu_type in _AMD_GPU_TYPES
}

_AMD_GPU_DISPATCH_IDENTITIES: dict[str, tuple[str, int]] = {
    "mi300x": ("gfx942", 304),
    "mi308x": ("gfx942", 304),
    "mi325x": ("gfx942", 304),
    "mi355x": ("gfx950", 256),
}


def _gpu_runner_type(gpu_type: str) -> str:
    """Return the Magpie runner label for a resolved real GPU type.

    Several types share one script: mi308x/mi325x run the mi300x scripts, and
    every gfx11 SKU runs ``{framework}_gfx11.sh``. The finer gpu_type is kept
    for the recipe KB, which must not merge cards that benchmark differently.
    """
    normalized = str(gpu_type or "").strip().lower()
    if normalized in ("mi325x", "mi308x"):
        return "mi300x"
    if normalized in _GFX11_GPU_TYPES:
        return "gfx11"
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
    """Return a member of :data:`GPU_TYPE_CHOICES`, or None if undetectable.

    Probes in decreasing specificity: rocm-smi's product name (the only signal
    that separates SKUs sharing one arch, e.g. ``AMD Radeon 8060S Graphics`` vs
    an 8050S, both gfx1151), then the ``GFX Version`` line of that same output,
    then torch's ``gcnArchName``. Every step is best-effort; ``None`` means the
    caller falls back to the ``--gpu-type`` / ``$GPU_TYPE`` hint.
    """
    import subprocess

    try:
        out = subprocess.run(
            ["rocm-smi", "--showproductname"],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.upper()
        for needle, gpu_type in _PRODUCT_TAGS:
            if needle in out:
                return gpu_type
        # Product name present but not a SKU we tabulate: rocm-smi also prints
        # "GFX Version: gfx1151", which still beats reporting nothing (and
        # keying the whole run to unknown_gpu) on a host without torch.
        for gfx, gpu_type in _GFX_TO_GPU_TYPE.items():
            if gfx.upper() in out:
                return gpu_type
    except (FileNotFoundError, subprocess.TimeoutExpired, PermissionError, OSError):
        # rocm-smi missing / slow / not permitted; fall through to the torch
        # gcnArchName probe below (autodetect is best-effort).
        pass
    try:
        import torch

        arch = torch.cuda.get_device_properties(0).gcnArchName
        gfx = arch.split(":", 1)[0].lower()
        return _GFX_TO_GPU_TYPE.get(gfx)
    except Exception:  # noqa: BLE001
        return None


def _resolve_amd_gpu_type(explicit: str | None = None) -> str | None:
    """Resolve the current MI GPU type, or None when not on one.

    Deliberately narrower than :data:`GPU_TYPE_CHOICES`: callers use this to gate
    CDNA-specific fast paths (aiter dispatch, sglang FP8/CK switches), so an
    gfx11 type such as ``8060s`` must resolve to ``None`` here even though it
    is a valid KB identity with a Magpie runner of its own.
    """
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
