# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Single source of truth for inference-framework capabilities."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FrameworkSpec:
    """Static capabilities of one inference framework."""

    name: str
    kind: str
    extra_args_env: str
    repo_url: str | None
    supports_server_reuse: bool
    throughput_unit: str
    has_denoiser_config: bool = False


SERVING = "serving"
SCRIPTABLE = "scriptable"


# The single registry (one entry per framework).
FRAMEWORKS: dict[str, FrameworkSpec] = {
    "sglang": FrameworkSpec(
        name="sglang",
        kind=SERVING,
        extra_args_env="EXTRA_SGLANG_ARGS",
        repo_url="https://github.com/sgl-project/sglang.git",
        supports_server_reuse=True,
        throughput_unit="tok/s",
    ),
    "vllm": FrameworkSpec(
        name="vllm",
        kind=SERVING,
        extra_args_env="EXTRA_VLLM_ARGS",
        repo_url="https://github.com/ROCm/vllm.git",
        supports_server_reuse=True,
        throughput_unit="tok/s",
    ),
    "atom": FrameworkSpec(
        name="atom",
        kind=SERVING,
        extra_args_env="EXTRA_ATOM_ARGS",
        repo_url="https://github.com/ROCm/ATOM.git",
        supports_server_reuse=False,
        throughput_unit="tok/s",
    ),
    "xdit": FrameworkSpec(
        name="xdit",
        kind=SCRIPTABLE,
        extra_args_env="EXTRA_XDIT_ARGS",
        repo_url="https://github.com/xdit-project/xDiT.git",
        supports_server_reuse=False,
        throughput_unit="img/s",
        # A diffusers pipeline: transformer/ + vae/ configs are on disk.
        has_denoiser_config=True,
    ),
    # An operator's own workload.
    "custom": FrameworkSpec(
        name="custom",
        kind=SCRIPTABLE,
        extra_args_env="EXTRA_CUSTOM_ARGS",
        repo_url=None,
        supports_server_reuse=False,
        throughput_unit="unit/s",
    ),
}

DEFAULT_FRAMEWORK = "sglang"


def names() -> tuple[str, ...]:
    """Return the canonical tuple of supported framework names."""
    return tuple(FRAMEWORKS)


def is_supported(framework: str | None) -> bool:
    """Return whether ``framework`` is a registered framework."""
    return str(framework or "").strip().lower() in FRAMEWORKS


def _spec_or_default(framework: str | None) -> FrameworkSpec:
    """Return the spec for ``framework`` or the default's spec when unknown."""
    key = str(framework or "").strip().lower()
    return FRAMEWORKS.get(key, FRAMEWORKS[DEFAULT_FRAMEWORK])


def is_scriptable(framework: str | None) -> bool:
    """Return whether ``framework`` is a server-less scriptable workload."""
    return _spec_or_default(framework).kind == SCRIPTABLE


def has_denoiser_config(framework: str | None) -> bool:
    """Return whether ``framework``'s model can be read as a diffusers denoiser."""
    return _spec_or_default(framework).has_denoiser_config


def extra_args_env(framework: str | None) -> str:
    """Return the Magpie env var used to append backend args."""
    return _spec_or_default(framework).extra_args_env


def server_args_env_name(framework: str | None) -> str:
    """Return the Magpie env var used to append backend server args."""
    name = str(framework or "").strip().lower()
    if is_supported(name):
        return extra_args_env(name)
    for fw in names():
        if fw in name:
            return extra_args_env(fw)
    return extra_args_env(DEFAULT_FRAMEWORK)


def throughput_unit(framework: str | None) -> str:
    """Return the throughput unit string for ``framework``."""
    return _spec_or_default(framework).throughput_unit


def primary_metric_unit(framework: str | None) -> str:
    """Return the human-readable unit for a session's primary display metric."""
    return "ms" if is_scriptable(framework) else "tok/s/GPU"


def primary_metric_name(framework: str | None) -> str:
    """Return the state field name holding a session's primary result metric."""
    return "e2el_mean_ms" if is_scriptable(framework) else "throughput_tok_s_per_gpu"


def primary_metric_value(framework: str | None, tput_per_gpu: float | int | None) -> float | None:
    """Convert stored per-GPU throughput into the value shown for ``framework``."""
    tput = float(tput_per_gpu or 0.0)
    if is_scriptable(framework):
        return (1000.0 / tput) if tput > 0 else None
    return tput


def format_primary_metric(framework: str | None, tput_per_gpu: float | int | None, *, precision: int = 1) -> str:
    """Format a session's primary performance metric for human-readable display."""
    unit = primary_metric_unit(framework)
    value = primary_metric_value(framework, tput_per_gpu)
    if value is None:
        return f"n/a {unit}"
    return f"{value:.{precision}f} {unit}"
