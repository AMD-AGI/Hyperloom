# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""AMD GPU-type helpers shared by the CLI and orchestrator runtime."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

from hyperloom_common.gpu_identity import AMD_GPU_BOARDS, GpuBoard, board_named_in, gpu_board


@dataclass(frozen=True)
class MagpieRunner:
    """A Magpie benchmark-script family, ``{framework}_{name}.sh``."""

    name: str
    #: Frameworks whose ``{framework}_{name}.sh`` supports Magpie's server_lifecycle reuse protocol.
    reuse_frameworks: frozenset[str]
    #: Frameworks validated on this runner; ``None`` leaves the framework unconstrained.
    validated_frameworks: frozenset[str] | None = None


@dataclass(frozen=True)
class BoardPolicy:
    """What Hyperloom does on one board of ``AMD_GPU_BOARDS``."""

    runner: MagpieRunner
    #: Whether the targeted-build (compiled / AITER) path has been validated on this board.
    targeted_build_validated: bool = True
    #: Whether the GEAK handoff names the product and ISA instead of leaving them to GEAK's device probe.
    geak_identity_pinned: bool = False
    #: Whether a bare arch (``gcnArchName``, rocminfo) stands for this board; other boards need their product name.
    names_its_arch: bool = False


_VLLM_ONLY = frozenset({"vllm"})
_INSTINCT_FRAMEWORKS = _VLLM_ONLY | {"sglang", "atom"}
_MI300X_RUNNER = MagpieRunner("mi300x", _INSTINCT_FRAMEWORKS)
_MI355X_RUNNER = MagpieRunner("mi355x", _INSTINCT_FRAMEWORKS)
_GFX12_RUNNER = MagpieRunner("gfx12", _VLLM_ONLY, validated_frameworks=_VLLM_ONLY)

#: Keyed by ``AMD_GPU_BOARDS``: every board has exactly one policy row.
_BOARD_POLICIES: dict[str, BoardPolicy] = {
    "mi300x": BoardPolicy(_MI300X_RUNNER, names_its_arch=True),
    "mi308x": BoardPolicy(_MI300X_RUNNER),
    "mi325x": BoardPolicy(_MI300X_RUNNER),
    "mi355x": BoardPolicy(_MI355X_RUNNER, names_its_arch=True),
    "r9700": BoardPolicy(_GFX12_RUNNER, targeted_build_validated=False, geak_identity_pinned=True),
    "radeon8060s": BoardPolicy(MagpieRunner("radeon8060s", frozenset())),
}

_RUNNERS: dict[str, MagpieRunner] = {policy.runner.name: policy.runner for policy in _BOARD_POLICIES.values()}

#: A bare arch resolves only to a board marked ``names_its_arch`` (gfx942 to mi300x, gfx950 to mi355x); gfx12 and
#: gfx1151 cover several products, so those resolve through the product name.
_GFX_REFERENCE_BOARDS: dict[str, str] = {
    AMD_GPU_BOARDS[product].gfx: product for product, policy in _BOARD_POLICIES.items() if policy.names_its_arch
}


def _board_policy(gpu_type: str | None) -> BoardPolicy | None:
    return _BOARD_POLICIES.get(str(gpu_type or "").strip().lower())


def gpu_runner_type(gpu_type: str) -> str:
    """Return the Magpie runner label for a resolved real GPU type."""
    policy = _board_policy(gpu_type)
    return policy.runner.name if policy else str(gpu_type or "").strip().lower()


def magpie_reuse_scripts() -> frozenset[str]:
    """The Magpie benchmark scripts that support the server_lifecycle reuse protocol."""
    return frozenset(
        f"{framework}_{runner.name}.sh" for runner in _RUNNERS.values() for framework in runner.reuse_frameworks
    )


def targeted_build_validated(gpu_type: str) -> bool:
    """Whether the targeted-build (compiled / AITER) path has been validated on this board."""
    policy = _board_policy(gpu_type)
    return policy.targeted_build_validated if policy else True


def geak_identity_pin(gpu_type: str) -> dict[str, str]:
    """The ``expected_target`` / ``expected_gfx`` pair a GEAK handoff carries for this board, if any."""
    policy = _board_policy(gpu_type)
    board = gpu_board(gpu_type)
    if not (policy and board and policy.geak_identity_pinned):
        return {}
    return {"expected_target": board.product, "expected_gfx": board.gfx}


def reference_gpu_type_for_gfx(text: str | None) -> str | None:
    """Return the board a bare gfx arch in ``text`` (``gcnArchName``, rocminfo) stands for, or ``None``."""
    lower = str(text or "").lower()
    return next((product for gfx, product in _GFX_REFERENCE_BOARDS.items() if gfx in lower), None)


def parse_gpu_type(text: str | None) -> str | None:
    """Map rocm-smi / gcnArchName output, possibly mixed into other log text, to a supported board name."""
    board = board_named_in(text)
    return board.product if board else reference_gpu_type_for_gfx(text)


def _runner_framework_error(gpu_type: str, framework: str) -> str | None:
    """Return why a constrained runner/framework pair is unavailable."""
    runner_name = gpu_runner_type(gpu_type)
    runner = _RUNNERS.get(runner_name)
    framework = str(framework or "").strip().lower()
    supported = runner.validated_frameworks if runner else None
    if not framework or not supported or framework in supported:
        return None
    return (
        f"Magpie runner {runner_name!r} is currently validated only for "
        f"{', '.join(sorted(supported))}; framework {framework!r} has no validated {runner_name} runner"
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
        ).stdout
        detected = parse_gpu_type(out)
        if detected:
            return detected
    except (FileNotFoundError, subprocess.TimeoutExpired, PermissionError, OSError):
        # rocm-smi missing / slow / not permitted; fall through to the torch gcnArchName probe below (autodetect is
        # best-effort).
        pass
    try:
        import torch

        return parse_gpu_type(str(torch.cuda.get_device_properties(0).gcnArchName))
    except (ImportError, OSError, RuntimeError, AssertionError, AttributeError):
        return None


def _resolve_amd_gpu_type(explicit: str | None = None) -> str | None:
    """Resolve the current AMD GPU type, or None when not on AMD/unknown."""
    explicit_norm = str(explicit or "").strip().lower()
    if explicit_norm:
        return explicit_norm if explicit_norm in AMD_GPU_BOARDS else None
    env_norm = (os.environ.get("TARGET_GPU_TYPE", "") or os.environ.get("GPU_TYPE", "")).strip().lower()
    if env_norm:
        return env_norm if env_norm in AMD_GPU_BOARDS else None
    detected = (_autodetect_gpu_type() or "").strip().lower()
    return detected if detected in AMD_GPU_BOARDS else None


def amd_gpu_board(gpu_type: str | None = None) -> GpuBoard | None:
    """Return the board of an explicit GPU type, ``$TARGET_GPU_TYPE`` / ``$GPU_TYPE``, or the detected GPU."""
    return gpu_board(_resolve_amd_gpu_type(gpu_type))
