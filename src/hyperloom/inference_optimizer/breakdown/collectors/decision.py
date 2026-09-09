# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Deterministic collectors for ``session_breakdown.json``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from hyperloom.common.timeutil import iso_z
from hyperloom.orchestrator.phases.machine_state import is_phase_transition_row
from hyperloom.orchestrator.state.optimization_journal import (
    operation_kind_for,
    proposer_for,
)

from ._common import (
    _load_jsonl_safe,
    _load_optimization_journal,
    _parse_iso_unix,
    _to_float,
    phase_at,
)


# Unified token + decision timeline.
_TOKEN_IN_KEY = "input_tokens"


_TOKEN_OUT_KEY = "output_tokens"


_TOKEN_CACHE_CREATE_KEY = "cache_creation_input_tokens"


_TOKEN_CACHE_READ_KEY = "cache_read_input_tokens"


# Hidden reasoning output.
_TOKEN_REASONING_KEY = "reasoning_output_tokens"


# Terminal-status key + its success value, re-declared for the same reason as the token keys above.
_STATUS_KEY = "status"


_STATUS_OK = "ok"


_TOKEN_KEYS_ALL: tuple[str, ...] = (
    _TOKEN_IN_KEY,
    _TOKEN_OUT_KEY,
    _TOKEN_CACHE_CREATE_KEY,
    _TOKEN_CACHE_READ_KEY,
    _TOKEN_REASONING_KEY,
)


def _coerce_token(value: Any) -> int:
    """Coerce a token counter to int, treating ``None`` / bad as 0."""
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


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
    bucket["total_in"] += _coerce_token(call.get(_TOKEN_IN_KEY))
    bucket["total_out"] += _coerce_token(call.get(_TOKEN_OUT_KEY))
    bucket["total_cache_creation"] += _coerce_token(call.get(_TOKEN_CACHE_CREATE_KEY))
    bucket["total_cache_read"] += _coerce_token(call.get(_TOKEN_CACHE_READ_KEY))
    bucket["total_reasoning_out"] += _coerce_token(call.get(_TOKEN_REASONING_KEY))
    bucket["calls"] += 1


def _load_llm_calls(
    session_dir: Path,
    warnings: list[str],
) -> list[dict[str, Any]]:
    """Read every *successful* LLM-call row from the trace ledger and ext shards."""
    trace_root = session_dir / "reports" / "trace"
    rows: list[dict[str, Any]] = list(_load_jsonl_safe(trace_root / "llm_calls.jsonl", warnings))
    ext_dir = trace_root / "ext"
    if ext_dir.is_dir():
        try:
            shards = sorted(ext_dir.glob("*.jsonl"))
        except OSError as exc:
            warnings.append(f"decision_trace: failed to scan {ext_dir}: {exc!r}")
            shards = []
        for shard in shards:
            rows.extend(_load_jsonl_safe(shard, warnings))
    return [
        r for r in rows if isinstance(r, dict) and str(r.get(_STATUS_KEY) or _STATUS_OK).strip().lower() == _STATUS_OK
    ]


def _load_proposal_task_map(
    session_dir: Path,
    warnings: list[str],
) -> dict[str, str]:
    """Read ``reports/trace/proposal_task_map.jsonl`` into ``{msg_id: task_id}``."""
    rows = _load_jsonl_safe(
        session_dir / "reports" / "trace" / "proposal_task_map.jsonl",
        warnings,
    )
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


def _load_dispatch_history_all(
    session_dir: Path,
    warnings: list[str],
) -> list[dict[str, Any]]:
    """Read every dynamic_action ``dispatch_history.jsonl`` row."""
    root = session_dir / "agents" / "orchestration" / "dynamic_actions"
    if not root.is_dir():
        return []
    out: list[dict[str, Any]] = []
    try:
        dyn_dirs = sorted(p for p in root.iterdir() if p.is_dir())
    except OSError as exc:
        warnings.append(f"decision_trace: failed to scan {root}: {exc!r}")
        return []
    for dyn_dir in dyn_dirs:
        rows = _load_jsonl_safe(dyn_dir / "dispatch_history.jsonl", warnings)
        for row in rows:
            if isinstance(row, dict):
                row = dict(row)
                row.setdefault("dyn_id", dyn_dir.name)
                out.append(row)
    return out


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
            ts_unix = _parse_iso_unix(row.get("ts"))
        if ts_unix is None:
            continue
        windows.append((float(ts_unix), to_phase))
    windows.sort(key=lambda w: w[0])
    return windows


def _phase_at(ts: Any, windows: list[tuple[float, str]]) -> str:
    """Return the phase active at ISO-or-numeric ``ts`` per ``windows``."""
    unix = _parse_iso_unix(ts)
    if unix is None or not windows:
        return ""
    return phase_at(unix, windows)


# Components whose unjoined LLM spend is legitimately not tied to a single decision (planning / review / monitoring),
# bucketed as ``overhead`` rather than ``unattributed``.
_OVERHEAD_COMPONENTS: frozenset[str] = frozenset(
    {
        "orchestration",
        "critic",
        "robustness",
    }
)


def _decision_key(task_id: str, dyn_id: str) -> str | None:
    """Canonical join key for a decision / call: ``dyn_id`` wins over ``task_id`` (a dynamic_action dispatch owns both)."""
    d = (dyn_id or "").strip()
    if d:
        return f"dyn:{d}"
    t = (task_id or "").strip()
    if t:
        return f"task:{t}"
    return None


def _token_convenience(bucket: dict[str, Any] | None) -> dict[str, Any]:
    """Copy a token bucket and add ``total_in_out``, ``grand_total`` and ``cache_hit_rate``."""
    b = dict(bucket or {})
    ti = int(b.get("total_in", 0) or 0)
    to = int(b.get("total_out", 0) or 0)
    cache = (
        int(b.get("total_cache", 0) or 0)
        + int(b.get("total_cache_creation", 0) or 0)
        + int(b.get("total_cache_read", 0) or 0)
    )
    reasoning = int(b.get("total_reasoning_out", 0) or 0)
    b["total_in_out"] = ti + to
    b["grand_total"] = ti + to + cache + reasoning
    cc = int(b.get("total_cache_creation", 0) or 0)
    cr = int(b.get("total_cache_read", 0) or 0)
    b["cache_hit_rate"] = round(cr / (cc + cr), 4) if (cc + cr) else 0.0
    return b


def collect_token_usage(
    decision_trace: dict[str, Any],
    action_timeline: list[dict[str, Any]],
    warnings: list[str],
) -> dict[str, Any]:
    """Promote the token rollup to a discoverable top-level ``token_usage``."""
    dt = decision_trace if isinstance(decision_trace, dict) else {}
    rollup = dt.get("token_rollup") or {}
    session_total = rollup.get("session_total") or _empty_token_bucket()
    by_component = rollup.get("by_component") or {}
    by_phase = rollup.get("by_phase") or {}
    unattributed = dt.get("unattributed_tokens") or _empty_token_bucket()
    overhead = dt.get("overhead_tokens") or _empty_token_bucket()

    # attributed = session_total - unattributed - overhead, field by field.
    attributed = _empty_token_bucket()
    for k in attributed:
        attributed[k] = (
            int(session_total.get(k, 0) or 0) - int(unattributed.get(k, 0) or 0) - int(overhead.get(k, 0) or 0)
        )
    total_calls = int(session_total.get("calls", 0) or 0)
    attr_calls = int(attributed.get("calls", 0) or 0)
    overhead_calls = int(overhead.get("calls", 0) or 0)
    attributed_calls_pct = round(100.0 * attr_calls / total_calls, 2) if total_calls else 0.0
    overhead_calls_pct = round(100.0 * overhead_calls / total_calls, 2) if total_calls else 0.0

    # Per-task token map from the per-decision view (only decision-bearing task_ids carry tokens — i.e. the attributed
    # subset).
    tokens_by_task: dict[str, dict[str, Any]] = {}
    for entry in dt.get("decision_trace") or []:
        if not isinstance(entry, dict):
            continue
        dec = entry.get("decision") or {}
        tid = str(dec.get("task_id") or dec.get("dyn_id") or "").strip()
        tok = entry.get("tokens") or {}
        if tid and int(tok.get("calls", 0) or 0) > 0:
            tokens_by_task[tid] = tok

    # Annotate the visible action timeline with tokens joined on task_id.
    timeline: list[dict[str, Any]] = []
    for act in action_timeline or []:
        if not isinstance(act, dict):
            continue
        tid = str(act.get("task_id") or "").strip()
        tok = tokens_by_task.get(tid) if tid else None
        timeline.append(
            {
                "task_id": tid or None,
                "action": str(act.get("action") or act.get("change") or ""),
                "phase": str(act.get("phase") or ""),
                "decision": str(act.get("decision") or ""),
                "ts": str(act.get("ts") or ""),
                "tokens": _token_convenience(tok) if tok else None,
            }
        )

    return {
        "session_total": _token_convenience(session_total),
        "by_component": {c: _token_convenience(b) for c, b in by_component.items()},
        "by_phase": {p: _token_convenience(b) for p, b in by_phase.items()},
        "attribution": {
            "attributed_to_decisions": _token_convenience(attributed),
            "overhead": _token_convenience(overhead),
            "unattributed": _token_convenience(unattributed),
            "attributed_calls_pct": attributed_calls_pct,
            "overhead_calls_pct": overhead_calls_pct,
        },
        "timeline": timeline,
        "source": "reports/trace/llm_calls.jsonl",
        "correlation": (
            "timeline[].task_id joins action_timeline[].task_id; components "
            "without a per-decision task_id (orchestration / kernel / critic / "
            "proposal_scorer) are counted in session_total/by_component/by_phase "
            "but appear as tokens=null in timeline (orchestration / critic / "
            "robustness land in attribution.overhead, the rest in "
            "attribution.unattributed)."
        ),
    }


def collect_langfuse(
    session_dir: Path,
    manifest: dict[str, Any],
    warnings: list[str],
) -> dict[str, Any]:
    """Assemble the ``langfuse`` section: was the trace pushed live, and how much."""
    from hyperloom.orchestrator.trace import langfuse_emitter as lfe

    # Tier 1: the persisted post-flush receipt (final counts).
    try:
        receipt = lfe.read_receipt(session_dir)
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"langfuse: read_receipt failed: {type(exc).__name__}: {exc}")
        receipt = None
    if receipt is not None:
        receipt["receipt_source"] = "receipt_file"
        return receipt

    # Tier 2: live read of the emitter singleton (pre-flush / in-process).
    try:
        section = lfe.get_emitter(session_dir).receipt()
        section["receipt_source"] = "live_emitter"
        return section
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"langfuse: live receipt failed: {type(exc).__name__}: {exc}")

    # Tier 3 fallback: config-only view straight from env + manifest, so the breakdown still records whether the
    # feature was configured at all.
    from hyperloom.orchestrator.trace import trace_env as tenv

    creds = tenv.langfuse_credentials()
    return {
        "enabled": False,
        "disabled_reason": "unknown",
        "config": {
            "enable_flag": tenv.langfuse_live_enabled(),
            "host": creds.get(tenv.ENV_LANGFUSE_HOST),
            "public_key_set": tenv.ENV_LANGFUSE_PUBLIC_KEY in creds,
            "secret_key_set": tenv.ENV_LANGFUSE_SECRET_KEY in creds,
            "sdk_available": None,
        },
        "trace_id": None,
        "session_id": str(manifest.get("claw_session_id") or manifest.get("session_id") or ""),
        "correlated_on": (
            "claw_session_id" if str(manifest.get("claw_session_id") or "").strip() else "internal_session_id"
        ),
        "counts": {},
        "counts_final": False,
        "receipt_source": "config_only",
    }


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


def collect_decision_trace(
    session_dir: Path,
    state: dict[str, Any],
    warnings: list[str],
) -> dict[str, Any]:
    """Join the token ledger to the decision streams into one timeline."""
    calls = _load_llm_calls(session_dir, warnings)
    phase_windows = _build_phase_windows(state)
    scores_by_variant = _proposal_scores_by_variant(state)

    # Attribute Critic review calls to the decision their reviewed proposal became.
    _attribute_critic_calls(calls, _load_proposal_task_map(session_dir, warnings))

    # Index calls by decision key; orphans (no key) go to a ts list.
    calls_by_key: dict[str, list[dict[str, Any]]] = {}
    orphan_calls: list[dict[str, Any]] = []
    for call in calls:
        key = _decision_key(
            str(call.get("task_id") or ""),
            str(call.get("dyn_id") or ""),
        )
        if key is None:
            orphan_calls.append(call)
        else:
            calls_by_key.setdefault(key, []).append(call)

    # Gather decisions from the journal + dispatch_history.
    decisions: list[dict[str, Any]] = []
    for e in _load_optimization_journal(session_dir, warnings):
        if not isinstance(e, dict):
            continue
        task_id = str(e.get("task_id") or "")
        key = _decision_key(task_id, "")
        ts = iso_z(e.get("ts"))
        phase = str(e.get("phase") or "").strip() or _phase_at(ts, phase_windows)
        provenance = str(e.get("provenance") or "")
        change_kind = str(e.get("kind") or "")
        # ``component`` is the real proposer derived from provenance.
        decision: dict[str, Any] = {
            "component": proposer_for(provenance) if provenance else "orchestration",
            "change": str(e.get("change") or ""),
            "outcome": str(e.get("outcome") or ""),
            "gain_pct": _to_float(e.get("gain_pct")),
            "task_id": task_id,
            "operation_kind": operation_kind_for("", change_kind),
        }
        # Predicted (pre-measurement) gain, when the proposer supplied one.
        predicted_gain = _to_float(e.get("predicted_gain_pct"))
        if predicted_gain is not None:
            decision["predicted_gain_pct"] = predicted_gain
        if change_kind:
            decision["kind"] = change_kind
        if provenance:
            decision["provenance"] = provenance
        scope = str(e.get("scope") or "")
        if scope:
            decision["scope"] = scope
        fingerprint = str(e.get("fingerprint") or "")
        if fingerprint:
            decision["fingerprint"] = fingerprint
        detail_metrics = e.get("metrics")
        if isinstance(detail_metrics, dict) and detail_metrics:
            decision["metrics"] = detail_metrics
        variant_name = str(e.get("variant_name") or "")
        if variant_name:
            decision["variant_name"] = variant_name
            # Attach the proposal_scorer signal (who rated this proposal, how).
            scored = scores_by_variant.get(variant_name)
            if scored:
                decision["proposal_scores"] = scored
        decisions.append(
            {
                "kind": "keep_revert",
                "key": key,
                "phase": phase,
                "tick": e.get("tick"),
                "ts": ts,
                "decision": decision,
            }
        )
    for row in _load_dispatch_history_all(session_dir, warnings):
        dyn_id = str(row.get("dyn_id") or "")
        key = _decision_key(str(row.get("task_id") or ""), dyn_id)
        ts = iso_z(row.get("ts"))
        phase = _phase_at(ts, phase_windows)
        decisions.append(
            {
                "kind": "dynamic_action",
                "key": key,
                "phase": phase,
                "tick": row.get("tick"),
                "ts": ts,
                "decision": {
                    "component": "dynamic_action",
                    "operation_kind": "dynamic_action",
                    "event": str(row.get("event") or ""),
                    "dyn_id": dyn_id,
                    "verdict": row.get("verdict"),
                    "outcome": str(row.get("integrate_status") or row.get("terminal_state") or ""),
                    "gain_pct": _to_float(row.get("delta_pct")),
                },
            }
        )

    # Attach calls to decisions; build the joined trace.
    consumed_keys: set[str] = set()
    decision_trace: list[dict[str, Any]] = []
    for dec in sorted(decisions, key=lambda d: d.get("ts") or ""):
        key = dec.get("key")
        if key and key in calls_by_key and key not in consumed_keys:
            attached = calls_by_key[key]
            consumed_keys.add(key)
        else:
            attached = []
        by_component: dict[str, dict[str, int]] = {}
        agg = _empty_token_bucket()
        for call in attached:
            comp = str(call.get("component") or "unknown")
            comp_bucket = by_component.setdefault(comp, _empty_token_bucket())
            _fold_call_into_bucket(comp_bucket, call)
            _fold_call_into_bucket(agg, call)
        decision_trace.append(
            {
                "phase": dec.get("phase") or "",
                "tick": dec.get("tick"),
                "ts": dec.get("ts") or "",
                "decision": dec.get("decision") or {},
                "tokens": {
                    "by_component": by_component,
                    "total_in": agg["total_in"],
                    "total_out": agg["total_out"],
                    "total_cache": agg["total_cache_creation"] + agg["total_cache_read"],
                    "total_reasoning_out": agg["total_reasoning_out"],
                    "calls": agg["calls"],
                },
            }
        )

    # Unjoined calls: keyed calls with no matching decision + orphans.
    unattributed = _empty_token_bucket()
    overhead = _empty_token_bucket()

    def _route_unjoined(call: dict[str, Any]) -> dict[str, int]:
        comp = str(call.get("component") or "")
        return overhead if comp in _OVERHEAD_COMPONENTS else unattributed

    for key, key_calls in calls_by_key.items():
        if key in consumed_keys:
            continue
        for call in key_calls:
            _fold_call_into_bucket(_route_unjoined(call), call)
    for call in orphan_calls:
        _fold_call_into_bucket(_route_unjoined(call), call)

    # Rollups: by_phase + by_component + session_total (ALL calls).
    by_phase: dict[str, dict[str, int]] = {}
    by_component_roll: dict[str, dict[str, int]] = {}
    session_total = _empty_token_bucket()
    for call in calls:
        comp = str(call.get("component") or "unknown")
        # Phase: prefer the call's own phase, else ts-window backfill.
        phase = str(call.get("phase") or "").strip() or _phase_at(call.get("ts"), phase_windows) or "unattributed"
        _fold_call_into_bucket(by_phase.setdefault(phase, _empty_token_bucket()), call)
        _fold_call_into_bucket(by_component_roll.setdefault(comp, _empty_token_bucket()), call)
        _fold_call_into_bucket(session_total, call)

    token_rollup = {
        "by_phase": by_phase,
        "by_component": by_component_roll,
        "session_total": session_total,
    }

    # Best-effort side write of the joined timeline.
    _write_decision_trace_jsonl(session_dir, decision_trace, warnings)

    return {
        "decision_trace": decision_trace,
        "token_rollup": token_rollup,
        "unattributed_tokens": unattributed,
        "overhead_tokens": overhead,
    }


def _write_decision_trace_jsonl(
    session_dir: Path,
    decision_trace: list[dict[str, Any]],
    warnings: list[str],
) -> None:
    """Append-free atomic-ish write of ``reports/trace/decision_trace.jsonl``."""
    target = session_dir / "reports" / "trace" / "decision_trace.jsonl"
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        lines = [json.dumps(row, sort_keys=True) for row in decision_trace]
        target.write_text(
            ("\n".join(lines) + "\n") if lines else "",
            encoding="utf-8",
        )
    except OSError as exc:
        warnings.append(f"decision_trace: failed to write {target}: {exc!r}")
