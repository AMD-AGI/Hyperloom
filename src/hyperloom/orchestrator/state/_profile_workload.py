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
    "reusable_recorded_runtime",
    "runtime_fields",
    "runtime_overrides",
    "serving_config_of",
    "shape_fields",
]

_TEXT_KEYS: tuple[str, ...] = ("framework", "precision")
_SHAPE_KEYS: tuple[str, ...] = ("tp", "conc", "isl", "osl", "max_model_len")

# Workload identity a caller may override; ``benchmark_mode`` is stamped from the session instead.
PROFILE_OVERRIDABLE_KEYS: tuple[str, ...] = (*_TEXT_KEYS, "model_path", *_SHAPE_KEYS)

_EXTRA_SERVER_ARGS = "extra_server_args"
_EXTRA_ENVS = "extra_envs"
_REMOVE_ARGS = "remove_args"
_UNSET_ENVS = "unset_envs"
_ARGS_MODE = "args_mode"
_BASE_EXTRA_ARGS = "base_extra_args"
_BASE_EXTRA_ENVS = "base_extra_envs"
_BASE_REMOVE_ARGS = "base_remove_args"
_BASE_UNSET_ENVS = "base_unset_envs"
_BASE_ARGS_MODE = "base_args_mode"

# Each runtime param as a caller and ``current_best`` name it, paired with the baseline-relative name it feeds.
_ENV_SOURCE: tuple[str, str] = (_EXTRA_ENVS, _BASE_EXTRA_ENVS)
_SERVER_ARGS_SOURCE: tuple[str, str] = (_EXTRA_SERVER_ARGS, _BASE_EXTRA_ARGS)
_REMOVE_ARGS_SOURCE: tuple[str, str] = (_REMOVE_ARGS, _BASE_REMOVE_ARGS)
_UNSET_ENVS_SOURCE: tuple[str, str] = (_UNSET_ENVS, _BASE_UNSET_ENVS)
_ARGS_MODE_SOURCE: tuple[str, str] = (_ARGS_MODE, _BASE_ARGS_MODE)

# Runtime params carried across verbatim; env vars are merged instead, so they travel on their own.
_COPIED_SOURCES: tuple[tuple[str, str], ...] = (
    _SERVER_ARGS_SOURCE,
    _REMOVE_ARGS_SOURCE,
    _UNSET_ENVS_SOURCE,
    _ARGS_MODE_SOURCE,
)

# Runtime fields whose presence means the caller already knows what the server ran with.
_RUNTIME_KEYS: frozenset[str] = frozenset(name for pair in (*_COPIED_SOURCES, _ENV_SOURCE) for name in pair)

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


def _override_or_base(params: Mapping[str, Any], source: tuple[str, str]) -> Any:
    """Read a runtime param under the name a caller uses, falling back to the baseline-relative one."""
    direct_name, base_name = source
    return params[direct_name] if direct_name in params else params.get(base_name)


def _normalized_server_args(params: Mapping[str, Any]) -> str:
    """Canonicalize the profile's extra server args: sanitized and re-quoted, whitespace-collapsed when unparseable."""
    raw = _override_or_base(params, _SERVER_ARGS_SOURCE)
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


def _normalized_control_list(params: Mapping[str, Any], source: tuple[str, str]) -> list[str]:
    """Normalize one arg/env removal list to a sorted set of non-blank entries."""
    raw = _override_or_base(params, source)
    values = [raw] if isinstance(raw, str) else list(raw or [])
    return sorted({str(value).strip() for value in values if str(value).strip()})


def runtime_fields(params: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize the runtime a profile ran under: server args, env vars, the controls that drop them, and the args mode."""
    return {
        "server_args": _normalized_server_args(params),
        _EXTRA_ENVS: _sorted_env_map(_override_or_base(params, _ENV_SOURCE)),
        _REMOVE_ARGS: _normalized_control_list(params, _REMOVE_ARGS_SOURCE),
        _UNSET_ENVS: _normalized_control_list(params, _UNSET_ENVS_SOURCE),
        _ARGS_MODE: str(_override_or_base(params, _ARGS_MODE_SOURCE) or "append").strip().lower(),
    }


def serving_config_of(current_best: Mapping[str, Any]) -> dict[str, Any]:
    """Describe the runtime the current best was measured with; empty when it names none."""
    envs = {key: value for key, value in _sorted_env_map(current_best.get(_EXTRA_ENVS)).items() if key.strip()}
    args = str(current_best.get(_EXTRA_SERVER_ARGS) or "").strip()
    if not (args or envs):
        return {}
    return {_EXTRA_SERVER_ARGS: args, _EXTRA_ENVS: envs}


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
        _BASE_EXTRA_ARGS: recorded.get("server_args", ""),
        _BASE_EXTRA_ENVS: dict(recorded.get(_EXTRA_ENVS) or {}),
        _BASE_REMOVE_ARGS: list(recorded.get(_REMOVE_ARGS) or []),
        _BASE_UNSET_ENVS: list(recorded.get(_UNSET_ENVS) or []),
        _BASE_ARGS_MODE: recorded.get(_ARGS_MODE, "append"),
    }


def runtime_overrides(current_best: Mapping[str, Any], incoming: Mapping[str, Any]) -> dict[str, Any]:
    """Collect the runtime params the current best and the caller's overrides name, the caller winning."""
    params: dict[str, Any] = {}
    for source, target in _COPIED_SOURCES:
        for holder, key in ((current_best, source), (incoming, source), (incoming, target)):
            if key in holder:
                params[target] = holder[key]
    return params


def merged_runtime_envs(current_best: Mapping[str, Any], incoming: Mapping[str, Any]) -> dict[str, Any]:
    """Overlay the caller's env overrides onto the env vars the current best was measured with."""
    raw = current_best.get(_EXTRA_ENVS)
    merged = dict(raw) if isinstance(raw, dict) else {}
    for key in (_BASE_EXTRA_ENVS, _EXTRA_ENVS):
        override = incoming.get(key)
        if isinstance(override, dict):
            merged.update(override)
    return merged
