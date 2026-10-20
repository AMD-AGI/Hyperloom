# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shaping of the persisted ``explore_search`` ledger when a session is loaded."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

__all__ = ["shape_explore_search"]


def _normalized_control_values(raw: Any) -> list[str]:
    """Normalize one persisted arg/env control to its non-blank entries."""
    if isinstance(raw, str):
        return [raw.strip()] if raw.strip() else []
    if isinstance(raw, (list, tuple, set)):
        return [str(value).strip() for value in raw if str(value).strip()]
    return []


def _winner_controls(entry: Mapping[str, Any]) -> dict[str, Any]:
    """Read the arg/env controls a winner was measured under, in the form the fingerprint takes them."""
    controls: dict[str, Any] = {}
    for key in ("remove_args", "unset_envs"):
        values = _normalized_control_values(entry.get(key))
        if values:
            controls[key] = values
    if str(entry.get("args_mode") or "append").strip().lower() == "replace":
        controls["args_mode"] = "replace"
    return controls


def _normalized_winner(entry: Mapping[str, Any]) -> dict[str, Any]:
    """Project one persisted winner onto the ledger's row shape, fingerprinting it when it carries no fingerprint."""
    from hyperloom.inference_optimizer.canonical_fingerprint import canonical_fingerprint

    controls = _winner_controls(entry)
    fingerprint = entry.get("fingerprint") or canonical_fingerprint(
        str(entry.get("extra_server_args") or ""),
        dict(entry.get("extra_envs") or {}),
        **controls,
    )
    return {
        "round_id": str(entry.get("round_id") or ""),
        "variant_name": str(entry.get("variant_name") or entry.get("name") or ""),
        "fingerprint": str(fingerprint),
        "gain_pct": entry.get("gain_pct"),
        "extra_args": str(entry.get("extra_args") or entry.get("extra_server_args") or ""),
        "extra_envs": dict(entry.get("extra_envs") or {}),
        **controls,
        "provenance": str(entry.get("provenance") or ""),
        "ts": str(entry.get("ts") or ""),
    }


def _winner_order(row: Mapping[str, Any]) -> tuple[str, str]:
    """Order winners by the round they landed in, then by timestamp."""
    return (str(row.get("round_id") or ""), str(row.get("ts") or ""))


def _normalized_winners_history(ledger: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Normalize the persisted winners, dropping anything that is not a row."""
    source = ledger.get("winners_history")
    rows = (
        [_normalized_winner(entry) for entry in source if isinstance(entry, dict)] if isinstance(source, list) else []
    )
    rows.sort(key=_winner_order)
    return rows


def shape_explore_search(existing: Any) -> dict[str, Any]:
    """Shape the unified ``explore_search`` ledger at load time."""
    ledger: dict[str, Any] = dict(existing) if isinstance(existing, dict) else {}
    ledger.setdefault("schema_version", 1)
    ledger.setdefault("tested", {})
    ledger.setdefault("accepted", [])
    ledger.setdefault("rejected", [])
    ledger.setdefault("domains_round_summary", [])
    ledger.setdefault("name_index", {})
    ledger.setdefault("cursor", len(ledger.get("tested") or {}))
    ledger.setdefault("last_round", {})
    ledger["winners_history"] = _normalized_winners_history(ledger)
    return ledger
