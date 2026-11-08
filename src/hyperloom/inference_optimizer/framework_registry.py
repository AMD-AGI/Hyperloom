# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Single source of truth for inference-framework capabilities.

The launch facts (kind, package, expert-parallel flag) are ``inference_suite.frameworks``'s; this registry derives
them and holds only Hyperloom's own facts.
"""

from __future__ import annotations

from dataclasses import dataclass

from inference_suite.frameworks import Framework, FrameworkFacts, FrameworkKind, framework_facts


@dataclass(frozen=True)
class FrameworkSpec:
    """Hyperloom's own facts about one inference framework."""

    name: str
    extra_args_env: str
    repo_url: str | None
    # The image's default checkout of the framework's source; ``None`` when it has none.
    source_root: str | None
    throughput_unit: str


SERVING = FrameworkKind.SERVING.value
SCRIPTABLE = FrameworkKind.SCRIPTABLE.value


# The single registry (one entry per ``inference_suite.frameworks.Framework``).
FRAMEWORKS: dict[str, FrameworkSpec] = {
    Framework.SGLANG.value: FrameworkSpec(
        name=Framework.SGLANG.value,
        extra_args_env="EXTRA_SGLANG_ARGS",
        repo_url="https://github.com/sgl-project/sglang.git",
        source_root="/sgl-workspace/sglang/",
        throughput_unit="tok/s",
    ),
    Framework.VLLM.value: FrameworkSpec(
        name=Framework.VLLM.value,
        extra_args_env="EXTRA_VLLM_ARGS",
        repo_url="https://github.com/ROCm/vllm.git",
        source_root="/sgl-workspace/vllm/",
        throughput_unit="tok/s",
    ),
    Framework.ATOM.value: FrameworkSpec(
        name=Framework.ATOM.value,
        extra_args_env="EXTRA_ATOM_ARGS",
        repo_url="https://github.com/ROCm/ATOM.git",
        source_root="/app/ATOM/atom/",
        throughput_unit="tok/s",
    ),
    Framework.XDIT.value: FrameworkSpec(
        name=Framework.XDIT.value,
        extra_args_env="EXTRA_XDIT_ARGS",
        repo_url="https://github.com/xdit-project/xDiT.git",
        source_root="/app/xDiT/",
        throughput_unit="img/s",
    ),
    Framework.CUSTOM.value: FrameworkSpec(
        name=Framework.CUSTOM.value,
        extra_args_env="EXTRA_CUSTOM_ARGS",
        repo_url=None,
        source_root=None,
        throughput_unit="unit/s",
    ),
}

DEFAULT_FRAMEWORK = Framework.SGLANG.value


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
    return _facts(_spec_or_default(framework)).kind is FrameworkKind.SCRIPTABLE


def tracelens_analysis_mode(framework: str | None) -> str:
    """Return the TraceLens analysis mode for ``framework``."""
    spec = _spec_or_default(framework)
    return "default" if spec.name == "custom" else "inference"


def _registered_spec(framework: str) -> FrameworkSpec:
    """Return the spec for ``framework``; raises ``KeyError`` when it is not registered."""
    return FRAMEWORKS[framework.strip().lower()]


def _facts(spec: FrameworkSpec) -> FrameworkFacts:
    return framework_facts(Framework(spec.name))


def python_package(framework: str) -> str | None:
    """Return the package ``framework`` imports as, or ``None`` when it ships none."""
    return _facts(_registered_spec(framework)).python_package


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
    """Return the point env var used to append backend args."""
    return _spec_or_default(framework).extra_args_env


def server_args_env_name(framework: str | None) -> str:
    """Return the point env var used to append backend server args."""
    name = str(framework or "").strip().lower()
    if is_supported(name):
        return extra_args_env(name)
    for fw in names():
        if fw in name:
            return extra_args_env(fw)
    return extra_args_env(DEFAULT_FRAMEWORK)


def expert_parallel_flag(framework: str | None) -> tuple[str, tuple[str, ...], bool] | None:
    """Return ``(flag, aliases, takes_size)`` of the expert-parallel option of the server reading ``framework``'s args."""
    env_name = server_args_env_name(framework)
    spec = next(spec for spec in FRAMEWORKS.values() if spec.extra_args_env == env_name)
    flag = _facts(spec).expert_parallel
    return None if flag is None else (flag.flag, flag.aliases, flag.takes_size)


def throughput_unit(framework: str | None) -> str:
    """Return the throughput unit string for ``framework``."""
    return _spec_or_default(framework).throughput_unit


def primary_metric_unit(framework: str | None) -> str:
    """Return the human-readable unit for a session's primary display metric."""
    return "ms" if is_scriptable(framework) else throughput_unit(framework)


def primary_metric_value(framework: str | None, tput: float | int | None) -> float | None:
    """Convert a session's stored throughput (the server aggregate, not per GPU) into the value shown."""
    tput = float(tput or 0.0)
    if is_scriptable(framework):
        return (1000.0 / tput) if tput > 0 else None
    return tput


def format_primary_metric(framework: str | None, tput: float | int | None, *, precision: int = 1) -> str:
    """Format a session's primary performance metric for human-readable display."""
    unit = primary_metric_unit(framework)
    value = primary_metric_value(framework, tput)
    if value is None:
        return f"n/a {unit}"
    return f"{value:.{precision}f} {unit}"
