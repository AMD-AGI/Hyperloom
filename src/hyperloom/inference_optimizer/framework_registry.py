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
    # The importable package the framework installs as; ``None`` when it ships none.
    python_package: str | None
    # The image's default checkout of the framework's source; ``None`` when it has none.
    source_root: str | None
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
        python_package="sglang",
        source_root="/sgl-workspace/sglang/",
        throughput_unit="tok/s",
    ),
    "vllm": FrameworkSpec(
        name="vllm",
        kind=SERVING,
        extra_args_env="EXTRA_VLLM_ARGS",
        repo_url="https://github.com/ROCm/vllm.git",
        python_package="vllm",
        source_root="/sgl-workspace/vllm/",
        throughput_unit="tok/s",
    ),
    "atom": FrameworkSpec(
        name="atom",
        kind=SERVING,
        extra_args_env="EXTRA_ATOM_ARGS",
        repo_url="https://github.com/ROCm/ATOM.git",
        python_package="atom",
        source_root="/app/ATOM/atom/",
        throughput_unit="tok/s",
    ),
    "xdit": FrameworkSpec(
        name="xdit",
        kind=SCRIPTABLE,
        extra_args_env="EXTRA_XDIT_ARGS",
        repo_url="https://github.com/xdit-project/xDiT.git",
        python_package="xfuser",
        source_root="/app/xDiT/",
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
        python_package=None,
        source_root=None,
        throughput_unit="unit/s",
    ),
}

DEFAULT_FRAMEWORK = "sglang"

# Server option that turns on expert parallelism: (flag to write, every
# argparse alias of it, whether it takes the EP size). vLLM and ATOM take a
# switch and derive the size from TP x DP. Frameworks absent here have none.
EXPERT_PARALLEL_FLAGS: dict[str, tuple[str, tuple[str, ...], bool]] = {
    "sglang": ("--ep-size", ("--ep-size", "--expert-parallel-size", "--ep"), True),
    "vllm": ("--enable-expert-parallel", ("--enable-expert-parallel", "-ep"), False),
    "atom": ("--enable-expert-parallel", ("--enable-expert-parallel",), False),
}


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


def _registered_spec(framework: str) -> FrameworkSpec:
    """Return the spec for ``framework``; raises ``KeyError`` when it is not registered."""
    return FRAMEWORKS[framework.strip().lower()]


def python_package(framework: str) -> str | None:
    """Return the package ``framework`` imports as, or ``None`` when it ships none."""
    return _registered_spec(framework).python_package


def source_root(framework: str) -> str | None:
    """Return the image's default checkout of ``framework``, or ``None`` when it has none."""
    return _registered_spec(framework).source_root


def repo_url(framework: str) -> str:
    """Return ``framework``'s upstream repo URL, or ``""`` when it has none."""
    return _registered_spec(framework).repo_url or ""


def shipped_config_name(kind: str, framework: str) -> str:
    """Return the shipped ``<kind>_<framework>.yaml`` asset name."""
    return f"{kind}_{_registered_spec(framework).name}.yaml"


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


def expert_parallel_flag(framework: str | None) -> tuple[str, tuple[str, ...], bool] | None:
    """Return the :data:`EXPERT_PARALLEL_FLAGS` row for the server reading ``framework``'s server args."""
    env_name = server_args_env_name(framework)
    return next(
        (row for name, row in EXPERT_PARALLEL_FLAGS.items() if FRAMEWORKS[name].extra_args_env == env_name),
        None,
    )


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
