# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Project framework runtime selections onto a benchmark launch environment."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from hyperloom.common.env_safety import BLOCKED_CHILD_ENV_NAMES, valid_env_key

_PATH = "PATH"
_PYTHONPATH = "PYTHONPATH"


log = logging.getLogger(__name__)


def is_safe_path_entry(entry: str) -> bool:
    """A single path entry is safe iff it has no ``:``, no ``..``, no control chars."""
    return (
        bool(entry)
        and ":" not in entry
        and ".." not in Path(entry).parts
        and not any(c in entry for c in ("\n", "\r", "\x00"))
    )


# Reserved keys carried by dedicated override fields; never accepted via runtime_env.
RUNTIME_ENV_RESERVED: frozenset[str] = frozenset({_PATH, _PYTHONPATH, "LD_LIBRARY_PATH"})


def apply_runtime_override(envs: dict[str, str], override: dict[str, Any]) -> None:
    """Inject an attempt runtime override into the materialized YAML envs dict."""
    if not override:
        return
    from hyperloom.common.env import prepend_path

    path_prefix = str(override.get("path_prefix") or "").strip()
    if path_prefix:
        prepend_path(envs, _PATH, path_prefix)
    entrypoint_bin = str(override.get("entrypoint_bin_dir") or "").strip()
    if entrypoint_bin:
        if is_safe_path_entry(entrypoint_bin):
            prepend_path(envs, _PATH, entrypoint_bin)
        else:
            log.warning("apply_runtime_override: dropping unsafe entrypoint_bin_dir %r", entrypoint_bin)
    pp_prefix = str(override.get("pythonpath_prefix") or "").strip()
    if pp_prefix:
        if is_safe_path_entry(pp_prefix):
            prepend_path(envs, _PYTHONPATH, pp_prefix)
        else:
            log.warning("apply_runtime_override: dropping unsafe pythonpath_prefix %r", pp_prefix)
    # Multi-entry prefixes: prepend in order so the first entry wins, ahead of any single-dir pythonpath_prefix and
    # the inherited value.
    for entry in reversed(_coerce_prefix_list(override.get("pythonpath_prefixes"))):
        if is_safe_path_entry(entry):
            prepend_path(envs, _PYTHONPATH, entry)
        else:
            log.warning("apply_runtime_override: dropping unsafe pythonpath entry %r", entry)
    # Native loader path: prepend attempt entries while preserving any inherited entries (e.g. /opt/rocm/lib).
    for entry in reversed(_coerce_prefix_list(override.get("ld_library_path_prefix"))):
        if is_safe_path_entry(entry):
            prepend_path(envs, "LD_LIBRARY_PATH", entry)
        else:
            log.warning("apply_runtime_override: dropping unsafe ld_library_path entry %r", entry)
    _runtime_env = override.get("runtime_env")
    if isinstance(_runtime_env, dict):
        for raw_k, raw_v in _runtime_env.items():
            key = str(raw_k)
            if not valid_env_key(key) or key in BLOCKED_CHILD_ENV_NAMES or key in RUNTIME_ENV_RESERVED:
                log.warning("apply_runtime_override: dropping unsafe runtime_env key %r", key)
                continue
            envs[key] = str(raw_v)
    for key in ("framework_bin", "framework_python", "framework_venv_root"):
        val = str(override.get(key) or "").strip()
        if val:
            envs[f"HYPERLOOM_{key.upper()}"] = val
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
