# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""AMD GPU-type helpers shared by the CLI and orchestrator runtime."""

from __future__ import annotations

import os
import re
import sys

from hyperloom.common.gpu_identity import AMD_GPU_DISPATCH_IDENTITIES, gfx_arch_for_gpu_type

#: Derived, not typed out again: a board added to the identities table is
#: accepted by the CLI and resolvable in the same commit. Listing it separately
#: let a board have a dispatch identity that ``_resolve_amd_gpu_type`` refused.
_AMD_GPU_TYPES = frozenset(AMD_GPU_DISPATCH_IDENTITIES)

#: rocm-smi product tags, reverse-sorted so a longer tag is tested before any
#: tag that is a prefix of it -- an "MI300XL" must not be claimed by "MI300X".
_PRODUCT_TAGS: tuple[str, ...] = tuple(sorted((t.upper() for t in _AMD_GPU_TYPES), reverse=True))

_GFX_TO_RUNNER: dict[str, str] = {
    # gfx arch -> Magpie runner label, so launchers and runtime materializers agree on the selected benchmark script.
    "gfx942": "mi300x",
    "gfx950": "mi355x",
}

_RUNNER_SUPPORTED_FRAMEWORKS = {"gfx12": frozenset({"vllm"})}

_R9700 = "r9700"

#: Boards whose targeted-build (compiled / AITER) path has not been validated.
_TARGETED_BUILD_UNVALIDATED_GPU_TYPES: frozenset[str] = frozenset({_R9700})

#: Boards whose GEAK handoff names the product and ISA instead of leaving them to GEAK's device probe.
_GEAK_PINNED_IDENTITY_GPU_TYPES: frozenset[str] = frozenset({_R9700})

#: Re-exported from ``hyperloom.common`` so provenance and this module cannot
#: disagree about which arch a board dispatches to.
_AMD_GPU_DISPATCH_IDENTITIES = AMD_GPU_DISPATCH_IDENTITIES


def _product_tag_matches(tag: str, text: str) -> bool:
    """Match R9700 exactly while preserving legacy MI product matching."""
    tag = str(tag or "").strip().upper()
    text = str(text or "").upper()
    if tag == "R9700":
        return re.search(r"(?<![A-Z0-9])R9700(?![A-Z0-9])", text) is not None
    if tag == "RADEON8060S":
        # rocm-smi prints "Radeon 8060S Graphics"; the gpu_type cannot hold that
        # space because it doubles as a Magpie script suffix.
        return "RADEON 8060S" in text or "RADEON8060S" in text
    return tag in text


def gpu_runner_type(gpu_type: str) -> str:
    """Return the Magpie runner label for a resolved real GPU type."""
    normalized = str(gpu_type or "").strip().lower()
    if normalized in ("mi325x", "mi308x"):
        return "mi300x"
    if normalized == _R9700:
        return "gfx12"
    return normalized


def targeted_build_validated(gpu_type: str) -> bool:
    """Whether the targeted-build (compiled / AITER) path has been validated on this board."""
    return str(gpu_type or "").strip().lower() not in _TARGETED_BUILD_UNVALIDATED_GPU_TYPES


def geak_identity_pin(gpu_type: str) -> dict[str, str]:
    """The ``expected_target`` / ``expected_gfx`` pair a GEAK handoff carries for this board, if any."""
    product = str(gpu_type or "").strip().lower()
    if product not in _GEAK_PINNED_IDENTITY_GPU_TYPES:
        return {}
    return {"expected_target": product, "expected_gfx": gfx_arch_for_gpu_type(product) or ""}


def _runner_framework_error(gpu_type: str, framework: str) -> str | None:
    """Return why a constrained runner/framework pair is unavailable."""
    runner = gpu_runner_type(gpu_type)
    framework = str(framework or "").strip().lower()
    supported = _RUNNER_SUPPORTED_FRAMEWORKS.get(runner)
    if not framework or not supported or framework in supported:
        return None
    return (
        f"Magpie runner {runner!r} is currently validated only for "
        f"{', '.join(sorted(supported))}; framework {framework!r} has no validated {runner} runner"
    )


def require_runner_framework(gpu_type: str, framework: str) -> None:
    """Exit at launch when the selected runner cannot serve the framework."""
    unsupported = _runner_framework_error(gpu_type, framework)
    if unsupported:
        print(f"ERROR: {unsupported}", file=sys.stderr)
        sys.exit(2)


def warn_family_runner(gpu_type: str, framework: str) -> None:
    """Say when a product launches through its architecture family's Magpie runner."""
    runner = gpu_runner_type(gpu_type)
    if gpu_type and runner != gpu_type:
        print(
            f"WARN: {gpu_type} uses {runner} as Magpie runner_type (architecture-family runner; "
            f"Magpie has no per-SKU {framework}_{gpu_type}.sh)",
            file=sys.stderr,
        )


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
    """Return a canonical supported board name, or ``None`` if undetectable."""
    import subprocess

    try:
        out = subprocess.run(
            ["rocm-smi", "--showproductname"],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.upper()
        for tag in _PRODUCT_TAGS:
            if _product_tag_matches(tag, out):
                return tag.lower()
    except (FileNotFoundError, subprocess.TimeoutExpired, PermissionError, OSError):
        # rocm-smi missing / slow / not permitted; fall through to the torch gcnArchName probe below (autodetect is
        # best-effort).
        pass
    try:
        import torch

        arch = torch.cuda.get_device_properties(0).gcnArchName
        gfx = arch.split(":", 1)[0].lower()
        return _GFX_TO_RUNNER.get(gfx)
    except (ImportError, OSError, RuntimeError, AssertionError, AttributeError):
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
