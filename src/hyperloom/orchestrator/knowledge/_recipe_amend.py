# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Merging one session's result into the live Recipe row.

A recipe row outlives the session amending it, so every write here is a
read-modify-write: the caller's overrides win, each field it left alone keeps
the value the live row carries, and the free-form extras some other session
wrote survive untouched.
"""

from __future__ import annotations

from copy import copy
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Any, Mapping

from hyperloom.common.coerce import to_float

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


def rows_not_superseded(live_rows: Any, superseding: Any, key: str) -> list[dict[str, Any]]:
    """Copies of the live rows whose ``key`` no superseding row already claims."""
    claimed = {str(row.get(key) or "") for row in superseding or [] if isinstance(row, dict)}
    kept: list[dict[str, Any]] = []
    for row in live_rows or []:
        if not isinstance(row, dict):
            continue
        if str(row.get(key) or "") in claimed:
            continue
        kept.append(dict(row))
    return kept


def improved_best_config(state: Any, attrs: Mapping[str, Any], live: Mapping[str, Any]) -> dict[str, Any]:
    """The ``best_config`` override, empty unless this session genuinely beat the live row.

    A session that validated nothing has no standing to overwrite a
    configuration somebody else measured.
    """
    my_throughput = float(attrs.get("best_throughput") or 0.0)
    validated_gain = float(state.cumulative_gain_validated or 0.0)
    current_args = str(state.current_best.get("extra_server_args") or "").strip()
    has_validated_win = bool((state.optimization_stack or []) or validated_gain > 0.0 or current_args)
    if not has_validated_win or my_throughput <= to_float(live.get("best_throughput"), 0.0):
        return {}
    return {
        "best_config": attrs["best_config"],
        "best_throughput": my_throughput,
    }


def appended(live_facts: Any, addition: Mapping[str, Any] | None) -> list[Any]:
    """The live lesson or pitfall list with one more appended, when the caller brought one."""
    facts = list(live_facts or [])
    if addition is not None:
        facts.append(addition)
    return facts


def amend_provenance(details: Mapping[str, Any] | None) -> dict[str, Any]:
    """The provenance block stamped on every amend; the central server requires one."""
    return {
        "source": "hyperloom-inference-optimizer",
        "generator": "coordinator",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
        "details": dict(details or {}),
    }


def amended_recipe_fields(live: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    """The preserved recipe fields for one amend: an override wins, an unset field keeps its live value.

    An absent or empty live value falls back to the field's own default, so
    ``put_recipe`` never sees ``None`` where it declares a container.
    """
    merged: dict[str, Any] = {}
    for field, default in PRESERVED_RECIPE_FIELDS.items():
        if field in overrides:
            merged[field] = overrides[field]
            continue
        live_value = live.get(field)
        merged[field] = type(default)(live_value) if live_value else copy(default)
    return merged


def amended_recipe_extras(live: Mapping[str, Any], overrides: Mapping[str, Any], state: Any) -> dict[str, Any]:
    """The free-form extras an amend writes back: the prior ones, the caller's, then the model's identity.

    The architecture tags are re-stamped from config.json each time because a
    row written before the architecture was read carries none.
    """
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
