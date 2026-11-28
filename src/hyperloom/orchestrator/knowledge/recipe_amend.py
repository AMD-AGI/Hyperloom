# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Merge rules for amending a live Recipe KB row with a session's facts and best config."""

from __future__ import annotations

from copy import copy
from types import MappingProxyType
from typing import Any, Mapping

from hyperloom_common.coerce import recorded_float


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


def kb_best_config_overrides_for_keep(
    *,
    live: Mapping[str, Any],
    best_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Decide whether a KEEP amend should also stamp the session ``best_config`` on the recipe row."""
    live_tput = recorded_float(live.get("best_throughput"))
    new_tput = recorded_float(best_config.get("tput"))
    if new_tput > 0.0 and new_tput >= live_tput:
        return {"best_config": dict(best_config), "best_throughput": new_tput}
    return {}
