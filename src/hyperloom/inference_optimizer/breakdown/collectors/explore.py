# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Deterministic collectors for ``session_breakdown.json``.

Each ``collect_<section>`` is a pure function over ``session_dir`` /
``state`` / ``manifest`` returning its schema section (see :mod:`.schema`).
Collectors never mutate state, fabricate values, or raise — failures are
recorded in ``warnings`` and the section returns a best-effort partial.
"""

from __future__ import annotations

from typing import Any

from hyperloom.orchestrator.state.optimization_journal import (
    classify_change_kind,
    operation_kind_for,
    proposer_for,
)

from ._common import (
    _to_float,
)


# Explore search ledger
def _shape_ledger(
    ledger: dict[str, Any] | None,
    *,
    top_n: int = 20,
) -> dict[str, Any]:
    """Normalize a search ledger (params / backends / explore) for the report.

    Shapes the ``accepted`` / ``rejected`` / ``tested`` entries to a stable
    field set and derives a ``top_by_gain`` ranking from the tested entries.

    Args:
        ledger (dict[str, Any] | None): A raw search ledger from state, or
            ``None``.
        top_n (int): Maximum number of entries in ``top_by_gain``. Defaults to
            20.

    Returns:
        dict[str, Any]: ``{"schema_version", "tested_count", "accepted",
        "rejected", "top_by_gain"}``. An empty shell is returned when
        ``ledger`` is not a dict.
    """
    if not isinstance(ledger, dict):
        return {
            "schema_version": 0,
            "tested_count": 0,
            "accepted": [],
            "rejected": [],
            "top_by_gain": [],
        }

    def _shape_entry(e: Any) -> dict[str, Any]:
        """Coerce one ledger entry to the stable field set.

        Args:
            e (Any): A raw ledger entry.

        Returns:
            dict[str, Any]: The shaped entry, or ``{}`` when ``e`` is not a
            dict.
        """
        if not isinstance(e, dict):
            return {}
        args = str(e.get("extra_server_args") or "")
        envs = dict(e.get("extra_envs") or {})
        provenance = str(e.get("provenance") or "")
        # Classify the variant into a stable operation_kind (backend/param/env).
        change_kind = classify_change_kind(
            "explore",
            {"extra_server_args": args, "extra_envs": envs},
        )
        shaped = {
            "name": str(e.get("name") or ""),
            "fingerprint": str(e.get("fingerprint") or ""),
            "extra_server_args": args,
            "extra_envs": envs,
            "output_throughput": _to_float(e.get("output_throughput") or e.get("tput")),
            "gain_pct": _to_float(e.get("gain_pct")),
            "ts": str(e.get("ts") or ""),
            "operation_kind": operation_kind_for("explore", change_kind),
        }
        if provenance:
            shaped["provenance"] = provenance
            shaped["proposer"] = proposer_for(provenance)
        scope = str(e.get("scope") or "")
        if scope:
            shaped["scope"] = scope
        return shaped

    accepted = [_shape_entry(e) for e in ledger.get("accepted") or []]
    rejected = [_shape_entry(e) for e in ledger.get("rejected") or []]
    tested = list((ledger.get("tested") or {}).values()) if isinstance(ledger.get("tested"), dict) else []
    tested_shaped = [_shape_entry(e) for e in tested]
    top_by_gain = sorted(
        (e for e in tested_shaped if e.get("gain_pct") is not None),
        key=lambda e: e.get("gain_pct") or 0.0,
        reverse=True,
    )[:top_n]
    return {
        "schema_version": int(ledger.get("schema_version") or 0),
        "tested_count": len(tested),
        "accepted": accepted,
        "rejected": rejected,
        "top_by_gain": top_by_gain,
    }


def _shape_winners_history(
    explore_search: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Persist ``explore_search.winners_history`` rows with their join key + source.

    Re-emits the fingerprint→provenance map (dropped by ``_shape_ledger``) so
    downstream can reconstruct ``phase_breakdown.explore.by_domain`` offline.

    Args:
        explore_search (dict[str, Any] | None): The ``explore_search`` ledger
            from state, or ``None``.

    Returns:
        list[dict[str, Any]]: The shaped ``winners_history`` rows (round id,
        variant, fingerprint, provenance, scope, gain, args/envs, ts). Empty
        when no history is present.
    """
    if not isinstance(explore_search, dict):
        return []
    rows = explore_search.get("winners_history")
    if not isinstance(rows, list):
        return []
    out: list[dict[str, Any]] = []
    for w in rows:
        if not isinstance(w, dict):
            continue
        out.append(
            {
                "round_id": str(w.get("round_id") or ""),
                "variant_name": str(w.get("variant_name") or w.get("name") or ""),
                "fingerprint": str(w.get("fingerprint") or ""),
                "provenance": str(w.get("provenance") or ""),
                "scope": str(w.get("scope") or ""),
                "gain_pct": _to_float(w.get("gain_pct")),
                "extra_args": str(w.get("extra_args") or w.get("extra_server_args") or ""),
                "extra_envs": dict(w.get("extra_envs") or {}),
                "ts": str(w.get("ts") or ""),
            }
        )
    return out


def collect_explore_search(
    state: dict[str, Any],
    warnings: list[str],
) -> dict[str, Any]:
    """Collect the explore-phase search summary for the breakdown.

    Args:
        state: Session state mapping.
        warnings: Mutable list that collected warnings are appended to.

    Returns:
        A dict summarizing the explore-phase search activity and outcomes.
    """
    # Emit all three ledgers (unified explore + legacy params/backends);
    # unused ones shape to empty shells.
    explore_ledger = _shape_ledger(state.get("explore_search"))
    # Persist provenance+fingerprint winners_history for offline recompute.
    explore_ledger["winners_history"] = _shape_winners_history(state.get("explore_search"))
    explore_ledger["no_promote_streak"] = int(state.get("params_no_promote_streak") or 0)

    params_ledger = _shape_ledger(state.get("params_search"))
    params_ledger["no_promote_streak"] = int(state.get("params_no_promote_streak") or 0)

    backends_ledger = _shape_ledger(state.get("backends_search"))

    return {
        "explore": explore_ledger,
        "params": params_ledger,
        "backends": backends_ledger,
        "synergy_attempted": list((state.get("explore_search") or {}).get("synergy_attempted") or []),
        "discovered_flags": dict(state.get("discovered_flags") or {}),
    }
