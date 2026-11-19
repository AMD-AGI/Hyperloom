# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Merge rules for amending a live Recipe KB row with a session's facts and best config."""

from __future__ import annotations

from copy import copy
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Mapping

from hyperloom_common.coerce import recorded_float, to_str_list

if TYPE_CHECKING:
    from ..state.task_registry import Task


#: Recipe fields an amend carries forward when the caller does not override
#: them, each with the empty value ``put_recipe`` falls back to.
PRESERVED_RECIPE_FIELDS: Mapping[str, Any] = MappingProxyType(
    {
        "best_config": {},
        "best_throughput": 0.0,
        "what_worked": [],
        "what_failed": [],
        "remaining_gaps": [],
        "last_profiled": "",
        "stack_fingerprint": {},
        "sessions": [],
        "authority": "EXPERIENTIAL",
        "confidence": 0.85,
        "evidence_refs": [],
    }
)

#: Row keys ``put_recipe`` owns; everything else on a live row is a session
#: extra and survives the amend untouched.
_RECIPE_OWNED_KEYS: frozenset[str] = frozenset(PRESERVED_RECIPE_FIELDS) | frozenset(
    {
        "canonical_id",
        "version",
        "created_at",
        "updated_at",
        "model",
        "hardware",
        "framework",
        "framework_name",
        "framework_version",
        "precision",
        "pitfalls",
        "lessons",
        "provenance",
    }
)


def _extra_server_args(payload: Mapping[str, Any]) -> str:
    return " ".join(to_str_list(payload.get("extra_server_args")))


def rows_not_superseded(live_rows: Any, superseding: Any, *keys: str) -> list[dict[str, Any]]:
    def identity(row: Mapping[str, Any]) -> tuple[str, ...]:
        return tuple(str(row.get(key) or "") for key in keys)

    claimed = {identity(row) for row in superseding or [] if isinstance(row, dict)}
    return [dict(row) for row in live_rows or [] if isinstance(row, dict) and identity(row) not in claimed]


def appended(live_facts: Any, addition: Mapping[str, Any] | None) -> list[Any]:
    facts = list(live_facts or [])
    if addition is not None:
        facts.append(addition)
    return facts


def amended_recipe_fields(live: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    # An absent or empty live value falls back to the field's own default, so
    # ``put_recipe`` never sees ``None`` where it declares a container.
    merged: dict[str, Any] = {}
    for field, default in PRESERVED_RECIPE_FIELDS.items():
        if field in overrides:
            merged[field] = overrides[field]
            continue
        live_value = live.get(field)
        merged[field] = type(default)(live_value) if live_value else copy(default)
    return merged


def amended_recipe_extras(live: Mapping[str, Any], overrides: Mapping[str, Any], state: Any) -> dict[str, Any]:
    # The architecture tags are re-stamped from config.json each time because a
    # row written before the architecture was read carries none.
    extras = {key: value for key, value in live.items() if key not in _RECIPE_OWNED_KEYS}
    extras.update(overrides.get("extras") or {})
    architectures = state.model_architectures or []
    if isinstance(architectures, list):
        named = [str(name).strip() for name in architectures if str(name or "").strip()]
        if named:
            extras["architectures"] = named
    model_type = str(state.model_type or "").strip()
    if model_type:
        extras["model_type"] = model_type
    return extras


def extract_kept_best_config(
    *,
    task: "Task",
    variant_attrs: dict[str, Any] | None = None,
    result_dict: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a replayable ``best_config`` from a KEEP'd task or explore variant."""
    params = task.params if isinstance(getattr(task, "params", None), dict) else {}
    attrs = variant_attrs if isinstance(variant_attrs, dict) else {}

    args = _extra_server_args(attrs)
    if not args.strip():
        args = _extra_server_args(params)
    if not args.strip() and isinstance(result_dict, dict):
        args = _extra_server_args(result_dict)

    envs_raw = attrs.get("extra_envs") or params.get("extra_envs") or {}
    if not envs_raw and isinstance(result_dict, dict):
        envs_raw = result_dict.get("extra_envs") or {}
    envs = {str(k): str(v) for k, v in envs_raw.items()} if isinstance(envs_raw, dict) else {}

    if not args.strip() and not envs:
        return {}

    best_config: dict[str, Any] = {}
    if args.strip():
        best_config["extra_server_args"] = args.strip()
    if envs:
        best_config["extra_envs"] = envs
    return best_config


def kb_best_config_overrides_for_keep(
    *,
    live: Mapping[str, Any],
    best_config_candidate: Mapping[str, Any],
    throughput_after: float | None,
) -> dict[str, Any]:
    """Decide whether a KEEP amend should also stamp ``best_config`` on the recipe row."""
    if not best_config_candidate:
        return {}

    live_bc = live.get("best_config") if isinstance(live.get("best_config"), Mapping) else {}
    live_has_config = bool(
        _extra_server_args(live_bc).strip()
        or (isinstance(live_bc.get("extra_envs"), Mapping) and live_bc.get("extra_envs"))
    )
    live_tput = recorded_float(live.get("best_throughput"))
    new_tput = recorded_float(throughput_after)
    if not live_has_config or (new_tput > 0.0 and new_tput >= live_tput):
        overrides: dict[str, Any] = {
            "best_config": dict(best_config_candidate),
        }
        if new_tput > 0.0:
            overrides["best_throughput"] = new_tput
        return overrides
    return {}
