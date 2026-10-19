# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Normalization of the workload and runtime identity a profile trace is keyed on."""

from __future__ import annotations

import logging
import shlex
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from hyperloom.common.profile_args import sanitize_profile_server_args

log = logging.getLogger(__name__)

__all__ = [
    "PROFILE_OVERRIDABLE_KEYS",
    "identity_fields",
    "merged_runtime_envs",
    "normalized_args_mode",
    "normalized_control_list",
    "normalized_server_args",
    "profile_env_map",
    "reusable_recorded_runtime",
    "runtime_overrides",
    "serving_config_of",
    "shape_fields",
]

_TEXT_KEYS: tuple[str, ...] = ("framework", "precision")
_SHAPE_KEYS: tuple[str, ...] = ("tp", "conc", "isl", "osl", "max_model_len")

# Workload identity a caller may override; ``benchmark_mode`` is stamped from the session instead.
PROFILE_OVERRIDABLE_KEYS: tuple[str, ...] = (*_TEXT_KEYS, "model_path", *_SHAPE_KEYS)

# Runtime fields whose presence means the caller already knows what the server ran with.
_RUNTIME_KEYS: frozenset[str] = frozenset(
    {
        "extra_server_args",
        "extra_envs",
        "remove_args",
        "unset_envs",
        "args_mode",
        "base_extra_args",
        "base_extra_envs",
        "base_remove_args",
        "base_unset_envs",
        "base_args_mode",
    }
)

# Runtime params as ``current_best`` names them, paired with the profile param they feed.
_RUNTIME_SOURCES: tuple[tuple[str, str], ...] = (
    ("extra_server_args", "base_extra_args"),
    ("remove_args", "base_remove_args"),
    ("unset_envs", "base_unset_envs"),
    ("args_mode", "base_args_mode"),
)

# Arms whose ``current_best`` carries no runtime fields of its own.
_BARE_RUNTIME_ACTIONS: frozenset[str] = frozenset({"baseline", "profile"})


def _profile_field(params: Mapping[str, Any], state: Any, name: str, default: Any) -> Any:
    """Read one identity field from the overrides, falling back to the session's own value."""
    value = params.get(name)
    return getattr(state, name, default) if value in (None, "") else value


def _resolved_model_path(raw: str) -> str:
    """Resolve a model path against the filesystem, keeping the raw value when it names nothing readable."""
    if not raw:
        return raw
    path = Path(raw).expanduser()
    if not path.exists():
        return raw
    try:
        return str(path.resolve())
    except OSError:
        log.debug(
            "profile workload path resolution failed for %s; keeping the unresolved path",
            path,
            exc_info=True,
        )
        return raw


def identity_fields(params: Mapping[str, Any], state: Any) -> dict[str, str]:
    """Normalize the framework, precision and model path that name which workload was profiled."""
    fields = {name: str(_profile_field(params, state, name, "") or "").strip().lower() for name in _TEXT_KEYS}
    fields["model_path"] = _resolved_model_path(str(_profile_field(params, state, "model_path", "") or "").strip())
    return fields


def shape_fields(params: Mapping[str, Any], state: Any) -> dict[str, int]:
    """Coerce the request-shape knobs that identify a profiled workload to ints, zero when unparseable."""
    fields: dict[str, int] = {}
    for name in _SHAPE_KEYS:
        try:
            fields[name] = int(_profile_field(params, state, name, 0) or 0)
        except (TypeError, ValueError):
            fields[name] = 0
    return fields


def _override_or_base(params: Mapping[str, Any], direct_name: str, base_name: str) -> Any:
    """Read a runtime param under its direct name, falling back to the baseline-relative one."""
    return params[direct_name] if direct_name in params else params.get(base_name)


def normalized_server_args(params: Mapping[str, Any]) -> str:
    """Canonicalize the profile's extra server args: sanitized and re-quoted, whitespace-collapsed when unparseable."""
    raw = _override_or_base(params, "extra_server_args", "base_extra_args")
    args = sanitize_profile_server_args(str(raw or "").strip())
    try:
        return shlex.join(shlex.split(args)) if args else ""
    except ValueError:
        return " ".join(args.split())


def _sorted_env_map(raw: Any) -> dict[str, str]:
    """Coerce an env mapping to ``str -> str`` ordered by name; anything that is not a mapping reads as empty."""
    if not isinstance(raw, dict):
        return {}
    return {str(key): str(value) for key, value in sorted(raw.items(), key=lambda item: str(item[0]))}


def profile_env_map(params: Mapping[str, Any]) -> dict[str, str]:
    """Normalize the env vars the profile runs with, under either the direct or the baseline-relative name."""
    return _sorted_env_map(_override_or_base(params, "extra_envs", "base_extra_envs"))


def normalized_control_list(params: Mapping[str, Any], direct_name: str, base_name: str) -> list[str]:
    """Normalize one arg/env removal list to a sorted set of non-blank entries."""
    raw = _override_or_base(params, direct_name, base_name)
    values = [raw] if isinstance(raw, str) else list(raw or [])
    return sorted({str(value).strip() for value in values if str(value).strip()})


def normalized_args_mode(params: Mapping[str, Any]) -> str:
    """Normalize how the profile's extra args combine with the server's own."""
    return str(_override_or_base(params, "args_mode", "base_args_mode") or "append").strip().lower()


def serving_config_of(current_best: Mapping[str, Any]) -> dict[str, Any]:
    """Describe the runtime the current best was measured with; empty when it names none."""
    envs = {key: value for key, value in _sorted_env_map(current_best.get("extra_envs")).items() if key.strip()}
    args = str(current_best.get("extra_server_args") or "").strip()
    if not (args or envs):
        return {}
    return {"extra_server_args": args, "extra_envs": envs}


def reusable_recorded_runtime(
    state: Any,
    current_best: Mapping[str, Any],
    incoming: Mapping[str, Any],
) -> dict[str, Any]:
    """Recover the runtime params from the last profile for an arm that records none of its own.

    A bare baseline/profile ``current_best`` carries no runtime fields, so the last profile is the only record of
    what the server actually ran with.

    Returns:
        Baseline-relative runtime params, or empty when the caller or the
        current best already names the runtime.
    """
    if any(key in current_best for key in _RUNTIME_KEYS) or any(key in incoming for key in _RUNTIME_KEYS):
        return {}
    if current_best.get("action") not in _BARE_RUNTIME_ACTIONS:
        return {}
    recorded = state.last_profile_workload if isinstance(state.last_profile_workload, dict) else {}
    # Legacy sessions predate the recorded action; keep reusing them rather than forcing a reprofile on every run.
    recorded_action = str(state.last_profile_workload_action or "").strip()
    if not recorded or recorded_action not in {"", *_BARE_RUNTIME_ACTIONS}:
        return {}
    return {
        "base_extra_args": recorded.get("server_args", ""),
        "base_extra_envs": dict(recorded.get("extra_envs") or {}),
        "base_remove_args": list(recorded.get("remove_args") or []),
        "base_unset_envs": list(recorded.get("unset_envs") or []),
        "base_args_mode": recorded.get("args_mode", "append"),
    }


def runtime_overrides(current_best: Mapping[str, Any], incoming: Mapping[str, Any]) -> dict[str, Any]:
    """Collect the runtime params the current best and the caller's overrides name, the caller winning."""
    params: dict[str, Any] = {}
    for source, target in _RUNTIME_SOURCES:
        for holder, key in ((current_best, source), (incoming, source), (incoming, target)):
            if key in holder:
                params[target] = holder[key]
    return params


def merged_runtime_envs(current_best: Mapping[str, Any], incoming: Mapping[str, Any]) -> dict[str, Any]:
    """Overlay the caller's env overrides onto the env vars the current best was measured with."""
    raw = current_best.get("extra_envs")
    merged = dict(raw) if isinstance(raw, dict) else {}
    for key in ("base_extra_envs", "extra_envs"):
        override = incoming.get(key)
        if isinstance(override, dict):
            merged.update(override)
    return merged
