# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The rows an explore round writes to the session's explore ledger, and the verdicts it lands before its round settles."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def ledger_view(row: dict[str, Any], fields: tuple[str, ...]) -> dict[str, Any]:
    return {key: row[key] for key in fields if key in row}


def variant_knobs(te: dict[str, Any]) -> dict[str, Any]:
    # Carry the variant knobs so the journal's ``classify_change_kind`` can classify the change kind.
    return {
        "name": str(te.get("name") or ""),
        "extra_server_args": str(te.get("extra_server_args") or ""),
        "extra_envs": dict(te.get("extra_envs") or {}),
        "remove_args": list(te.get("remove_args") or []),
        "unset_envs": list(te.get("unset_envs") or []),
        "args_mode": str(te.get("args_mode") or "append"),
        "note": str(te.get("note") or ""),
        "reasoning_origin": str(te.get("reasoning_origin") or ""),
        "experience_citations": list(te.get("experience_citations") or []),
    }


def land_verdict(
    shared_state: Any, session_dir: Path, update: dict[str, Any], attributions: list[dict[str, Any]]
) -> None:
    """Persist one non-KEEP verdict and its lever attributions while its round still runs.

    Settlement re-merges the rows by fingerprint unchanged. The executor runs on the Coordinator's event loop and
    ``save`` writes atomically, so the mid-round save cannot interleave with a Coordinator write.
    """
    shared_state.apply_explore_search_update(update, advance_round=False)
    for attribution in attributions:
        shared_state.record_framework_lever_attribution(
            attribution["switch"], gain_pct=attribution["gain_pct"], source=str(attribution.get("source") or "")
        )
    shared_state.save(session_dir)
