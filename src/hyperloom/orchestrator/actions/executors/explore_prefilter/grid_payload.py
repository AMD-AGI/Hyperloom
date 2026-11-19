# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The variants an EXPLORE round is offered: the sent grid, the seeded rewrite-lever arms, or the default grid."""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from typing import Any

from hyperloom_common.coerce import to_int
from ..framework_change import _framework_switch_manifest as _switch_manifest
from ..grid._grid_base import GridVariant, _default_grid_for_framework
from .._proposal_identity import normalize_proposal
from ._remeasure import is_stack_revalidation
from .declared_levers import declared_lever_payload

log = logging.getLogger(__name__)

# ``framework_lever_source`` of an arm that switches one active lever off.
_LEAVE_ONE_OUT = "leave_one_out"


def _action_reasoning(raw: dict[str, Any]) -> tuple[str, str]:
    """Return the action-authored rationale and the payload field that supplied it."""
    for field in ("reasoning", "rationale", "reason", "note"):
        value = str(raw.get(field) or "").strip()
        if value:
            return value, f"action_payload.{field}"
    return "", ""


def grid_variants_from_payload(payload: list[Any]) -> list[GridVariant]:
    """Convert the LLM/specialist grid payload into GridVariant objects."""
    out: list[GridVariant] = []
    for raw in payload or []:
        if not isinstance(raw, dict) or not raw.get("name"):
            continue
        fields = normalize_proposal(raw)
        reasoning, reasoning_origin = _action_reasoning(raw)
        gv = GridVariant(
            name=fields["name"],
            extra_server_args=fields["extra_args"],
            extra_envs=fields["extra_envs"],
            note=reasoning,
            remove_args=fields["remove_args"],
            unset_envs=fields["unset_envs"],
            args_mode=fields["args_mode"],
        )
        # Stash extra metadata on the GridVariant so the ledger writer can pull provenance.
        gv.provenance = str(raw.get("provenance") or "default_grid")  # type: ignore[attr-defined]
        gv.reasoning_origin = reasoning_origin  # type: ignore[attr-defined]
        gv.scope = str(raw.get("scope") or "")  # type: ignore[attr-defined]
        # Authored-kernel overlay dir (PYTHONPATH prefix); "" for env/flag variants.
        gv.overlay_pythonpath = str(raw.get("overlay_pythonpath") or "")
        # Authored kernels this variant's overlay installs.
        gv.accepted_kernels = [  # type: ignore[attr-defined]
            str(k).strip() for k in (raw.get("accepted_kernels") or []) if str(k).strip()
        ]
        # Validated against what the proposing agent was shown before the grid reached this executor.
        gv.experience_citations = [  # type: ignore[attr-defined]
            dict(item) for item in (raw.get("experience_citations") or []) if isinstance(item, dict)
        ]
        out.append(gv)
    return out


def framework_lever_grid(shared_state: Any) -> list[dict[str, Any]]:
    """Build explore variants that attribute each registered rewrite lever."""
    if shared_state is None:
        return []
    rows = [row for row in getattr(shared_state, "authored_framework_levers", None) or [] if isinstance(row, dict)]
    # Arms are built over every registered lever, so attribution only picks which run, never which envs they set.
    pending = {row["switch"] for row in rows if row.get("attributed_gain_pct") is None}
    payload: list[dict[str, Any]] = []
    for variant in _switch_manifest.additive_variants([row for row in rows if not row.get("default_on")]):
        payload.append({**variant, "framework_lever_source": "additive"})
    for variant in _switch_manifest.leave_one_out_variants([row for row in rows if row.get("default_on")]):
        payload.append({**variant, "framework_lever_source": _LEAVE_ONE_OUT})
    return _switch_manifest.unattributed_arms(payload, pending)


def framework_lever_attributions(
    per_variant_outcomes: list[dict[str, Any]],
    lever_payload: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Derive each rewrite lever's own contribution from this round's outcomes."""
    if not lever_payload:
        return []
    by_name = {
        str(v.get("name") or ""): v
        for v in lever_payload
        if isinstance(v, dict) and str(v.get("framework_lever") or "")
    }
    if not by_name:
        return []
    out: list[dict[str, Any]] = []
    for row in per_variant_outcomes:
        seed = by_name.get(str(row.get("variant_name") or ""))
        if seed is None:
            continue
        # A REVERT keeps its measured movement in ``graded_gain_pct``; a regressed leave-one-out arm is that case.
        metrics = row.get("metrics") or {}
        measured = metrics.get("gain_pct", metrics.get("graded_gain_pct"))
        if not isinstance(measured, (int, float)):
            continue
        source = str(seed.get("framework_lever_source") or "")
        gain = -float(measured) if source == _LEAVE_ONE_OUT else float(measured)
        out.append(
            {
                "switch": str(seed["framework_lever"]),
                "gain_pct": round(gain, 4),
                "source": source,
                "variant_name": str(row.get("variant_name") or ""),
                "outcome": str(row.get("outcome") or ""),
            }
        )
    return out


def _lever_seeded_payload(payload: list[Any], levers: list[dict[str, Any]]) -> list[Any]:
    named = {str(v.get("name") or "") for v in payload if isinstance(v, dict)}
    fresh = [v for v in levers if str(v.get("name") or "") not in named]
    if not fresh:
        return payload
    log.info("explore: seeding %d framework-rewrite lever variant(s) for attribution", len(fresh))
    # A leave-one-out arm only attributes a switch already on, so the arms that could KEEP run before it.
    attribution = [v for v in fresh if v.get("framework_lever_source") == _LEAVE_ONE_OUT]
    return [v for v in fresh if v not in attribution] + list(payload) + attribution


def _seed_grid_payload(params: dict[str, Any], framework: str, yaml_envs: dict[str, Any]) -> list[Any]:
    model_class = str(params.get("model_class") or "").strip() or os.environ.get("MODEL_CLASS", "").strip()
    conc = to_int(params.get("conc") or yaml_envs.get("CONC") or os.environ.get("CONC") or 0, 0) or 0
    seed = _default_grid_for_framework(framework, model_class=model_class, conc=conc)
    if not seed:
        return []
    log.info(
        "explore: empty grid for framework=%s; falling through to %d default_grid seed variants "
        "(model_class=%r conc=%d)",
        framework or "?",
        len(seed),
        model_class or "?",
        conc,
    )
    return [
        {
            "name": v.name,
            "extra_server_args": v.extra_server_args,
            "extra_envs": dict(v.extra_envs or {}),
            "note": v.note or "default_grid",
            "provenance": getattr(v, "provenance", "default_grid"),
        }
        for v in seed
    ]


def resolved_grid_payload(
    params: Mapping[str, Any],
    *,
    shared_state: Any,
    framework: str,
    yaml_envs: Mapping[str, Any] | None = None,
) -> tuple[list[Any], list[dict[str, Any]]]:
    """The payload a round over ``params`` measures, and the lever variants seeded into it.

    The sent grid with this session's fresh lever variants in front of it; only when that leaves
    nothing does the framework's default grid stand in. A non-empty sent grid therefore never
    reaches the default grid, however little of it is a variant.
    """
    payload = params.get("grid") or []
    if not isinstance(payload, list):
        payload = []
    # A stack revalidation re-measures the recorded stack on purpose. Seeding a lever
    # into it would validate stack+lever instead, so the declarations sit this round out.
    declared = [] if is_stack_revalidation(params) else declared_lever_payload(shared_state, framework)
    lever_payload = framework_lever_grid(shared_state) + declared
    payload = _lever_seeded_payload(payload, lever_payload)
    if not payload:
        payload = _seed_grid_payload(dict(params), framework, dict(yaml_envs or {}))
    return payload, lever_payload


def runnable_grid_variants(params: Mapping[str, Any] | None, shared_state: Any) -> list[GridVariant]:
    """The variants an explore over ``params`` would have to measure, by the round's own reckoning.

    Admission asks this before spending a dispatch on a round that would refuse itself. The round
    reads its framework back from the materialized YAML, which a proposal has not got yet, so the
    session's framework answers for it; a ``framework`` in ``params`` is not consulted, since the
    round never seeds its grid from one and the gate would otherwise judge a grid the round will not
    run. The config-pinned-env filter that follows in the round needs that YAML, so the variants it
    drops stay the round's to report.
    """
    p = dict(params or {})
    framework = str(getattr(shared_state, "framework", "") or "")
    payload, _ = resolved_grid_payload(p, shared_state=shared_state, framework=framework)
    return grid_variants_from_payload(payload)
