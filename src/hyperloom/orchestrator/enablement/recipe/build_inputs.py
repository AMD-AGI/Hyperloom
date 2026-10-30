# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The input set a targeted build actually ran with.

``BuildResult.to_state()`` persists outcomes and no inputs at all, while the
action holding them is cleared on finish, so a succeeded build's own recipe is
unrecoverable. This records it where the action and the result are both in
scope, and records the values the *driver* resolved rather than the action's raw
fields, which are routinely blank where a component default decided the build.
"""

from __future__ import annotations

import hashlib
from typing import Any, Mapping

from hyperloom.common.env_safety import is_secret_shaped_env_name
from hyperloom.orchestrator.enablement.runtime import targeted_build

from .credentials import classify_credential_value, detect_credential_channels, strip_url_userinfo

BUILTIN_PLAN_DRIVER = "builtin_plan"

#: Names a component's own driver overwrites in the merge without first reading
#: them: the inherited value never reaches the build, and the value that does is
#: already carried by ``gpu_arch`` and ``max_jobs``. The split is per driver
#: because the merges differ -- ``PYTORCH_ROCM_ARCH`` is inherited on the
#: sgl-kernel path and overwritten on the other two.
_DRIVER_OVERWRITTEN_ENV: dict[str, frozenset[str]] = {
    "aiter": frozenset(
        {
            "AITER_ROOT_DIR",
            "INFERENCE_OPTIMIZER_AITER_JIT_DIR",
            "AITER_REBUILD",
            "AITER_ROCM_ARCH",
            "PYTORCH_ROCM_ARCH",
            "MAX_JOBS",
        }
    ),
    "sgl_kernel": frozenset({"AMDGPU_TARGET", "MAX_JOBS"}),
    "vllm_source": frozenset({"PYTORCH_ROCM_ARCH", "MAX_JOBS"}),
}

#: Terminal, tty, display, locale-presentation and shell-session names. They
#: carry no build effect and would otherwise make every digest unique.
_IRRELEVANT_ENV: frozenset[str] = frozenset(
    {
        "COLORTERM",
        "COLUMNS",
        "DISPLAY",
        "HISTFILE",
        "LINES",
        "OLDPWD",
        "PS1",
        "PS2",
        "PWD",
        "SHLVL",
        "TERM",
        "TERM_PROGRAM",
        "TERM_PROGRAM_VERSION",
        "TTY",
        "WAYLAND_DISPLAY",
        "_",
    }
)

_IRRELEVANT_ENV_PREFIXES: tuple[str, ...] = ("LC_",)


def _is_irrelevant(name: str) -> bool:
    return name in _IRRELEVANT_ENV or name == "LANG" or name.startswith(_IRRELEVANT_ENV_PREFIXES)


def _digest_pairs(env: Mapping[str, str]) -> str:
    """Digest ``KEY=VALUE`` pairs, reducing a credential-shaped entry to its key.

    Key names alone do not identify a build -- two builds differing only in an
    env *value* would emit byte-identical inputs -- so the digest separates them
    while emitting no value.
    """
    pairs = [f"{key}=" if is_secret_shaped_env_name(key) else f"{key}={value}" for key, value in sorted(env.items())]
    return f"sha256:{hashlib.sha256(chr(10).join(pairs).encode('utf-8')).hexdigest()}"


def _visible_keys(env: Mapping[str, str]) -> list[str]:
    return sorted(key for key in env if not is_secret_shaped_env_name(key))


def ambient_closure(env: Mapping[str, str], *, component: str) -> dict[str, str]:
    """Return the inherited variables that are inputs to ``component``'s build.

    Nothing is judged in or out by supposed relevance: what the driver spawns
    reads its own environment, so a name the driver never mentions can still
    decide which compiler, index or library path the build used.
    """
    driver = "aiter" if component == "framework_ext" else component
    overwritten = _DRIVER_OVERWRITTEN_ENV.get(driver, frozenset())
    return {
        str(key): str(value)
        for key, value in env.items()
        if str(key) not in overwritten and not _is_irrelevant(str(key))
    }


def build_input_record(
    action: Any,
    *,
    installed_versions: Mapping[str, str] | None,
    ambient_env: Mapping[str, str],
    fs_root: str = "/",
) -> dict[str, Any]:
    """Record the resolved input set of one build attempt.

    Args:
        action: The ``TargetedBuildAction`` the attempt ran.
        installed_versions: The attempt's version map, source of the immutable
            commit sha sitting one key from the mutable ref.
        ambient_env: The environment the spawn inherited.
        fs_root: Filesystem root the ambient credential channels are probed under.
    """
    component = str(getattr(action, "component", "") or "")
    default_repo, default_jobs = {
        "aiter": (targeted_build._AITER_DEFAULT_REPO, targeted_build._AITER_DEFAULT_MAX_JOBS),
        "sgl_kernel": (targeted_build._SGLANG_DEFAULT_REPO, targeted_build._SGLANG_DEFAULT_MAX_JOBS),
        "vllm_source": (targeted_build._VLLM_DEFAULT_REPO, targeted_build._VLLM_DEFAULT_MAX_JOBS),
    }.get("aiter" if component == "framework_ext" else component, ("", 0))
    raw_repo = str(getattr(action, "repo_url", "") or "").strip() or default_repo
    versions = dict(installed_versions or {})
    ambient = ambient_closure(ambient_env, component=component)
    return {
        "component": component,
        "repo_url": strip_url_userinfo(raw_repo),
        "ref": str(getattr(action, "ref", "") or ""),
        "resolved_sha": _resolved_sha(versions),
        "gpu_arch": str(getattr(action, "gpu_arch", "") or "") or versions.get("arch", ""),
        "max_jobs": default_jobs,
        "ambient_keys": _visible_keys(ambient),
        "ambient_digest": _digest_pairs(ambient),
        "credential_class": classify_credential_value(raw_repo),
        "credential_channels": detect_credential_channels(ambient_env, fs_root=fs_root),
    }


def _resolved_sha(versions: Mapping[str, str]) -> str:
    """Return the immutable commit sha recorded beside the mutable ref."""
    for key in ("aiter_sha", "sgl_kernel_sha", "vllm_sha"):
        value = str(versions.get(key) or "").strip()
        if value:
            return value
    return ""
