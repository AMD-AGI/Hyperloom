# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The copies of a grid variant an EXPLORE round launches and the rows it records for them."""

from __future__ import annotations

from typing import Any

from hyperloom_common.coerce import to_str_list

from ....state.explore_ledger import variant_knobs
from ....state.failure_evidence import (
    UNMEASURED_OUTCOMES,
    FAILURE_STAGE_DECISION,
    classify_failure_attribution,
    make_failure_id,
)
from .._proposal_identity import controls_of
from ._grid_base import GridVariant, importable_overlay_dir


# Audit/provenance metadata stashed on a GridVariant that must survive being rebuilt into a derived variant.
_CARRIED_VARIANT_ATTRS: tuple[str, ...] = (
    "provenance",
    "reasoning_origin",
    "scope",
    "overlay_pythonpath",
    "accepted_kernels",
    "experience_citations",
)


def decision_fields(gv: Any) -> dict[str, Any]:
    """Why this variant was proposed, as authored at action time, for every row that records it."""
    return {
        "note": gv.note,
        "reasoning_origin": str(getattr(gv, "reasoning_origin", "") or ""),
        "experience_citations": list(getattr(gv, "experience_citations", []) or []),
    }


def variant_with_eval_off(gv: Any) -> Any:
    """Return a copy of ``gv`` with ``RUN_EVAL`` forced off.

    The variant layer is the last one :func:`run_grid` applies over the
    accumulated stack, so forcing the key here holds whatever the stack carries.
    """
    envs = dict(gv.extra_envs)
    envs["RUN_EVAL"] = "false"
    return carry_variant_metadata(
        gv,
        GridVariant(
            name=gv.name,
            extra_server_args=gv.extra_server_args,
            extra_envs=envs,
            note=gv.note,
            remove_args=to_str_list(getattr(gv, "remove_args", [])),
            unset_envs=to_str_list(getattr(gv, "unset_envs", [])),
            args_mode=str(getattr(gv, "args_mode", "append") or "append"),
        ),
    )


def carry_variant_metadata(src: Any, dst: Any) -> Any:
    """Copy the carried audit metadata from ``src`` onto ``dst``."""
    for attr in _CARRIED_VARIANT_ATTRS:
        if hasattr(src, attr):
            setattr(dst, attr, getattr(src, attr))
    return dst


def variant_fingerprint(variant: Any) -> str:
    """The canonical fingerprint stamped on a variant before the round launched it."""
    return str(getattr(variant, "canonical_fp", "") or "")


def staged_variant(gv: GridVariant, envs: dict[str, Any], unset_envs: list[str]) -> GridVariant:
    envs = dict(envs)
    for key in gv.unset_envs:
        envs.pop(key, None)
    envs.update(gv.extra_envs)
    run_gv = GridVariant(
        name=gv.name,
        extra_server_args=gv.extra_server_args,
        extra_envs=envs,
        note=gv.note,
        remove_args=list(gv.remove_args),
        unset_envs=list(dict.fromkeys(unset_envs + to_str_list(getattr(gv, "unset_envs", [])))),
        args_mode=str(getattr(gv, "args_mode", "append") or "append"),
    )
    return carry_variant_metadata(gv, run_gv)


def variant_facts(gv: GridVariant) -> dict[str, Any]:
    return {
        "fingerprint": variant_fingerprint(gv),
        "name": gv.name,
        "extra_server_args": gv.extra_server_args,
        "extra_envs": dict(gv.extra_envs),
        **controls_of(vars(gv)),
        **decision_fields(gv),
        "framework_changed": bool(importable_overlay_dir(gv)),
    }


# The numbers a measured variant produced. ``base_tput`` is carried verbatim rather than left to be back-solved
# from the gain: each KEEP advances the round anchor, so a consumer dividing the gain out of the final throughput
# reconstructs whichever anchor happens to be current, and a FAILED variant has no gain to divide.
_VARIANT_METRIC_KEYS: tuple[str, ...] = ("tput", "gain_pct", "graded_gain_pct", "base_tput")

# The ledgers' views of the row the round records for a variant.
_VARIANT_FIELDS: tuple[str, ...] = (
    "fingerprint",
    "name",
    "extra_server_args",
    "extra_envs",
    "remove_args",
    "unset_envs",
    "args_mode",
)
LOSER_FIELDS: tuple[str, ...] = (*_VARIANT_FIELDS, "provenance", "gain_pct", "tput", "reason", "workspace")


def outcome_row(te: dict[str, Any], fingerprint: str, reason: str, task_id: str) -> dict[str, Any]:
    name = str(te.get("name") or "")
    outcome = str(te.get("outcome") or "")
    return {
        "variant_name": name,
        "outcome": outcome,
        "fingerprint": fingerprint,
        "failure_id": make_failure_id(task_id=task_id, fingerprint=fingerprint, variant_name=name),
        "stage": str(te.get("stage") or FAILURE_STAGE_DECISION),
        "error_excerpt": te.get("error_excerpt"),
        "provenance": str(te.get("provenance") or ""),
        "scope": str(te.get("scope") or ""),
        "metrics": {key: te[key] for key in _VARIANT_METRIC_KEYS if te.get(key) is not None},
        "reason": reason,
        "error_class": str(te.get("error_class") or ""),
        "failure_attribution": classify_failure_attribution(
            error_class=te.get("error_class"),
            framework_changed=bool(te.get("framework_changed")),
            error_excerpt=te.get("error_excerpt"),
            reason=reason,
        )
        if outcome in UNMEASURED_OUTCOMES
        else "",
        "server_log_path": te.get("server_log_path"),
        "workspace": te.get("workspace"),
        "raw_result_path": te.get("raw_result_path"),
        "variant": variant_knobs(te),
        # The verdicts and the stack behind them, as the round ruled. Absent keys mean the variant never got that
        # far: no gate ruled on a warmup failure, and nothing validated one that was killed before it was graded.
        "measured_against": te.get("measured_against") or {},
        "gates": [gate for gate in (te.get("gates") or []) if isinstance(gate, dict)],
        "validation_basis": str(te.get("validation_basis") or ""),
    }
