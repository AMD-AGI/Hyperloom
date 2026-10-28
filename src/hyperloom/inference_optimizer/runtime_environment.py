# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Project an isolated framework runtime into a server launch environment."""

from __future__ import annotations

import logging
import os
import re
from typing import Any

from hyperloom.common.env_safety import BLOCKED_CHILD_ENV_NAMES
from hyperloom.common.overlay import is_safe_path_entry

log = logging.getLogger(__name__)


def prepend_launch_path(envs: dict[str, Any], var: str, *entries: str) -> None:
    """Put ``entries`` first on the path-list env ``var`` the materialized config hands the server launch.

    Magpie and the bypass launcher start the server with ``os.environ`` overlaid by ``benchmark.envs``, so a config
    value replaces the inherited one. A config without the key therefore builds on the inherited value, or
    prepending would drop the operator's and the image's entries (import paths, ``PATH``, ``/opt/rocm/lib``); a
    config with the key stays authoritative. An entry already on the path moves to the front, and empty parts are
    dropped.
    """
    base = envs[var] if var in envs else os.environ.get(var)
    parts = [*entries, *str(base or "").strip().split(os.pathsep)]
    envs[var] = os.pathsep.join(dict.fromkeys(part for part in parts if part))


# Reserved keys carried by dedicated override fields; never accepted via runtime_env.
_RUNTIME_ENV_RESERVED: frozenset[str] = frozenset({"PATH", "PYTHONPATH", "LD_LIBRARY_PATH"})


def apply_runtime_override(envs: dict[str, str], override: dict[str, Any]) -> None:
    """Inject an attempt runtime override into the materialized YAML envs dict."""
    if not override:
        return

    path_prefix = str(override.get("path_prefix") or "").strip()
    if path_prefix:
        prepend_launch_path(envs, "PATH", path_prefix)
    # Each list prepends in reverse so its first entry wins, and a later row lands ahead of earlier ones
    # (pythonpath_prefixes ahead of pythonpath_prefix).
    for var, entries, label in (
        ("PATH", _coerce_prefix_list(str(override.get("entrypoint_bin_dir") or "")), "entrypoint_bin_dir"),
        ("PYTHONPATH", _coerce_prefix_list(str(override.get("pythonpath_prefix") or "")), "pythonpath_prefix"),
        ("PYTHONPATH", _coerce_prefix_list(override.get("pythonpath_prefixes")), "pythonpath entry"),
        ("LD_LIBRARY_PATH", _coerce_prefix_list(override.get("ld_library_path_prefix")), "ld_library_path entry"),
    ):
        for entry in reversed(entries):
            if is_safe_path_entry(entry):
                prepend_launch_path(envs, var, entry)
            else:
                log.warning("apply_runtime_override: dropping unsafe %s %r", label, entry)
    _runtime_env = override.get("runtime_env")
    if isinstance(_runtime_env, dict):
        for raw_k, raw_v in _runtime_env.items():
            key = str(raw_k)
            if (
                not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key)
                or key in BLOCKED_CHILD_ENV_NAMES
                or key in _RUNTIME_ENV_RESERVED
            ):
                log.warning("apply_runtime_override: dropping unsafe runtime_env key %r", key)
                continue
            envs[key] = str(raw_v)
    framework_python = str(override.get("framework_python") or "").strip()
    if framework_python:
        envs["HYPERLOOM_FRAMEWORK_PYTHON"] = framework_python
    # runtime_python_exe takes priority over framework_python as the launch interpreter.
    rpe = str(override.get("runtime_python_exe") or "").strip()
    if rpe:
        envs["HYPERLOOM_FRAMEWORK_PYTHON"] = rpe


def _coerce_prefix_list(value: Any) -> list[str]:
    """Normalize a runtime-prefix field (list/tuple/str) to a list of entries."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    return []
