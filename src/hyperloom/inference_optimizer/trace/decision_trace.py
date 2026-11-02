# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The session's decision trace: one row per decision, with its token cost.

Written after the breakdown, from the same session directory, but not part of
it. The trace's readers are the Langfuse emitter and the backfill tool, which
score a session by joining each decision against the LLM calls that produced
it; the breakdown never read the file it wrote. It lives here, beside those
readers, so a breakdown reader is not left looking for the key it feeds.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from hyperloom_common.coerce import to_int, to_unix
from hyperloom_common.timeutil import iso_z
from hyperloom.inference_optimizer.breakdown.collectors._common import (
    _load_jsonl_safe,
    _to_float,
    phase_at,
)
from hyperloom.inference_optimizer.breakdown.recorder.phase_event import is_phase_transition_row
from hyperloom.inference_optimizer.session.optimization_journal import (
    JournalEntry,
    journal_path,
    operation_kind_for,
    proposer_for,
    read_journal_entries,
)
from hyperloom.inference_optimizer.session.session_paths import (
    decision_trace_path,
    llm_calls_path,
    proposal_task_map_path,
    state_path,
)

from .llm_trace import LLM_STATUS_OK


def _empty_token_bucket() -> dict[str, int]:
    """Return a fresh, zeroed token-rollup bucket."""
    return {
        "total_in": 0,
        "total_out": 0,
        "total_cache_creation": 0,
        "total_cache_read": 0,
        "total_reasoning_out": 0,
        "calls": 0,
    }


def _fold_call_into_bucket(bucket: dict[str, int], call: dict[str, Any]) -> None:
    """Add one call's token counts into a rollup bucket in place."""
    bucket["total_in"] += to_int(call.get("input_tokens"), 0)
    bucket["total_out"] += to_int(call.get("output_tokens"), 0)
    bucket["total_cache_creation"] += to_int(call.get("cache_creation_input_tokens"), 0)
    bucket["total_cache_read"] += to_int(call.get("cache_read_input_tokens"), 0)
    bucket["total_reasoning_out"] += to_int(call.get("reasoning_output_tokens"), 0)
    bucket["calls"] += 1


def _load_llm_calls(
    session_dir: Path,
    warnings: list[str],
) -> list[dict[str, Any]]:
    """Read every *successful* LLM-call row from the trace ledger."""
    rows = _load_jsonl_safe(llm_calls_path(session_dir), warnings)
    return [
        r
        for r in rows
        if isinstance(r, dict) and str(r.get("status") or LLM_STATUS_OK).strip().lower() == LLM_STATUS_OK
    ]


def _load_proposal_task_map(
    session_dir: Path,
    warnings: list[str],
) -> dict[str, str]:
    """Read ``reports/trace/proposal_task_map.jsonl`` into ``{msg_id: task_id}``."""
    rows = _load_jsonl_safe(proposal_task_map_path(session_dir), warnings)
    out: dict[str, str] = {}
    for r in rows:
        if not isinstance(r, dict):
            continue
        mid = str(r.get("proposal_msg_id") or "").strip()
        tid = str(r.get("task_id") or "").strip()
        if mid and tid:
            out[mid] = tid
    return out


def _attribute_critic_calls(
    calls: list[dict[str, Any]],
    msg_to_task: dict[str, str],
) -> None:
    """Backfill ``task_id`` on Critic review calls from the proposal->task map."""
    if not msg_to_task:
        return
    for call in calls:
        if str(call.get("component") or "") != "critic":
            continue
        if str(call.get("task_id") or "").strip():
            continue  # already keyed; respect it
        reviewed = call.get("reviewed_msg_ids")
        if not isinstance(reviewed, list):
            continue
        reviewed_ids = {m for m in reviewed if isinstance(m, str) and m}
        resolved = {msg_to_task[m] for m in reviewed_ids if m in msg_to_task}
        # Single-target review only: a partial mapping (reviewed several, only one materialized) must NOT collapse the
        # batch's cost onto that one.
        if len(reviewed_ids) == 1 and len(resolved) == 1:
            call["task_id"] = next(iter(resolved))


def _build_phase_windows(
    state: dict[str, Any],
) -> list[tuple[float, str]]:
    """Build a sorted ``[(entered_unix, phase), ...]`` timeline."""
    history = state.get("phase_history") or []
    if not isinstance(history, list):
        return []
    windows: list[tuple[float, str]] = []
    for row in history:
        if not isinstance(row, dict):
            continue
        if not is_phase_transition_row(row):
            continue
        to_phase = str(row.get("to_phase") or "").strip()
        ts_unix = row.get("ts_unix")
        if ts_unix is None:
            ts_unix = to_unix(row.get("ts"))
        if ts_unix is None:
            continue
        windows.append((float(ts_unix), to_phase))
    windows.sort(key=lambda w: w[0])
    return windows


def _phase_at(ts: Any, windows: list[tuple[float, str]]) -> str:
    """Return the phase active at ISO-or-numeric ``ts`` per ``windows``."""
    unix = to_unix(ts)
    if unix is None or not windows:
        return ""
    return phase_at(unix, windows)


def _decision_key(task_id: str) -> str | None:
    """Canonical join key for a decision / call: its ``task_id``."""
    t = (task_id or "").strip()
    if t:
        return f"task:{t}"
    return None


def _decision_id(kind: str, *facts: Any) -> str:
    """Name one decision by the facts its source records once, so the id survives every rewrite of the trace."""
    return hashlib.sha256(json.dumps([kind, *facts]).encode("utf-8")).hexdigest()[:16]


def _proposal_scores_by_variant(state: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Index ``specialist_rounds[].ensemble_scores`` by variant name."""
    out: dict[str, list[dict[str, Any]]] = {}
    rounds = state.get("specialist_rounds")
    if not isinstance(rounds, list):
        return out
    for r in rounds:
        if not isinstance(r, dict):
            continue
        ens = r.get("ensemble_scores")
        models = ens.get("models") if isinstance(ens, dict) else None
        if not isinstance(models, dict):
            continue
        for slug, per_model in models.items():
            if not isinstance(per_model, dict):
                continue
            for name, cell in per_model.items():
                if not isinstance(cell, dict) or cell.get("score") is None:
                    continue
                out.setdefault(str(name), []).append(
                    {
                        "rater": str(slug),
                        "score": _to_float(cell.get("score")),
                        "reason": str(cell.get("reason") or ""),
                    }
                )
    return out


def _keep_revert_decision(
    entry: JournalEntry,
    phase_windows: list[tuple[float, str]],
    scores_by_variant: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """The trace row for one KEEP/REVERT journal entry, still carrying its call-join ``key``."""
    ts = iso_z(entry.ts)
    decision: dict[str, Any] = {
        "change": entry.change,
        "outcome": entry.outcome,
        "gain_pct": _to_float(entry.gain_pct),
        "task_id": entry.task_id,
    }
    for field in ("kind", "provenance", "scope", "fingerprint", "variant_name"):
        text = getattr(entry, field)
        if text:
            decision[field] = text
    decision["component"] = proposer_for(decision.get("provenance", ""))
    decision["operation_kind"] = operation_kind_for("", decision.get("kind", ""))
    # Predicted (pre-measurement) gain, when the proposer supplied one.
    predicted_gain = _to_float(entry.predicted_gain_pct)
    if predicted_gain is not None:
        decision["predicted_gain_pct"] = predicted_gain
    if entry.metrics:
        decision["metrics"] = entry.metrics
    scored = scores_by_variant.get(entry.variant_name)
    if scored:
        decision["proposal_scores"] = scored
    return {
        "decision_id": _decision_id("keep_revert", *entry.dedupe_key()),
        "key": _decision_key(entry.task_id),
        "phase": entry.phase.strip() or _phase_at(ts, phase_windows),
        "tick": entry.tick,
        "ts": ts,
        "decision": decision,
    }


def write_decision_trace(
    session_dir: Path,
    state: dict[str, Any],
    warnings: list[str],
) -> None:
    """Join the token ledger to the decision streams and write the timeline.

    Reads the per-call token rows (``reports/trace/llm_calls.jsonl``) and the
    decision rows (``optimization_journal.json`` KEEP/REVERT entries), then
    attaches each decision's LLM calls by the shared ``task_id`` key.

    Every row carries a ``decision_id`` that stays the same across rewrites:
    the journal entry's ``dedupe_key``.

    Writes an empty file when no trace files exist, so a session that ran before
    the trace subsystem landed degrades cleanly. Best-effort: an OSError lands
    in ``warnings``, mutated in place, rather than failing the close-out.
    """
    calls = _load_llm_calls(session_dir, warnings)
    phase_windows = _build_phase_windows(state)
    scores_by_variant = _proposal_scores_by_variant(state)

    _attribute_critic_calls(calls, _load_proposal_task_map(session_dir, warnings))

    # A call without a task_id anchors to no decision row, so it contributes nothing to the timeline.
    calls_by_key: dict[str, list[dict[str, Any]]] = {}
    for call in calls:
        key = _decision_key(str(call.get("task_id") or ""))
        if key is not None:
            calls_by_key.setdefault(key, []).append(call)

    journal_file = journal_path(session_dir)
    entries = read_journal_entries(
        session_dir, on_error=lambda exc: warnings.append(f"failed to parse {journal_file}: {exc!r}")
    )
    decisions = [_keep_revert_decision(e, phase_windows, scores_by_variant) for e in entries]

    # Sorted by time, the first decision that names a key takes that key's calls.
    decisions.sort(key=lambda d: d["ts"])
    for dec in decisions:
        by_component: dict[str, dict[str, int]] = {}
        agg = _empty_token_bucket()
        for call in calls_by_key.pop(dec.pop("key"), []):
            bucket = by_component.setdefault(str(call.get("component") or "unknown"), _empty_token_bucket())
            _fold_call_into_bucket(bucket, call)
            _fold_call_into_bucket(agg, call)
        dec["tokens"] = {
            "by_component": by_component,
            "total_in": agg["total_in"],
            "total_out": agg["total_out"],
            "total_cache": agg["total_cache_creation"] + agg["total_cache_read"],
            "total_reasoning_out": agg["total_reasoning_out"],
            "calls": agg["calls"],
        }

    _write_decision_trace_jsonl(session_dir, decisions, warnings)


def _write_decision_trace_jsonl(
    session_dir: Path,
    decision_trace: list[dict[str, Any]],
    warnings: list[str],
) -> None:
    """Append-free atomic-ish write of ``reports/trace/decision_trace.jsonl``.

    Rewrites the whole file (one JSON object per decision) on each export; the
    collector is the single producer, so a full rewrite is simpler than append
    + dedup and stays consistent with the latest join. Best-effort: OSError is
    recorded in ``warnings`` and swallowed.
    """
    target = decision_trace_path(session_dir)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        lines = [json.dumps(row, sort_keys=True) for row in decision_trace]
        target.write_text(
            ("\n".join(lines) + "\n") if lines else "",
            encoding="utf-8",
        )
    except OSError as exc:
        warnings.append(f"decision_trace: failed to write {target}: {exc!r}")


def write_session_decision_trace(session_dir: Path | str) -> list[str]:
    """Write the session's decision trace, loading ``state.json`` itself.

    The single entry point for callers that only know the session directory, so
    none of them has to know the trace needs ``state.json`` or that the file
    must land before any Langfuse flush reads it. Returns the warnings raised
    while reading the inputs, for the caller to log; empty on a clean write.
    A failure to build the trace propagates to the caller.
    """
    warnings: list[str] = []
    sd = Path(session_dir).resolve()
    state: dict[str, Any] = {}
    path = state_path(sd)
    if path.exists():
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            warnings.append(f"decision_trace: failed to parse state.json: {type(exc).__name__}: {exc}")
            state = {}
    if not isinstance(state, dict):
        state = {}
    write_decision_trace(sd, state, warnings)
    return warnings
