# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Deterministic collectors for ``session_breakdown.json``.

Each ``collect_<section>`` is a pure function over ``session_dir`` /
``state`` / ``manifest`` returning its schema section (see :mod:`.schema`).
Collectors never mutate state, fabricate values, or raise — failures are
recorded in ``warnings`` and the section returns a best-effort partial.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from hyperloom.common.timeutil import iso_z
from hyperloom.orchestrator.state.optimization_journal import (
    operation_kind_for,
    proposer_for,
)

from ._common import (
    _load_jsonl_safe,
    _load_optimization_journal,
    _parse_iso_unix,
    _to_float,
)



# ---------------------------------------------------------------------------
# Full-trace: unified token + decision timeline (FULL_TRACE_DESIGN §6)
# ---------------------------------------------------------------------------
# Token-counter keys, shared with orchestrator.trace.llm_trace. Re-declared
# here (not imported) so the breakdown package stays free of orchestrator
# deps — collectors must run offline against a session_dir tarball.
_TOKEN_IN_KEY = "input_tokens"


_TOKEN_OUT_KEY = "output_tokens"


_TOKEN_CACHE_CREATE_KEY = "cache_creation_input_tokens"


_TOKEN_CACHE_READ_KEY = "cache_read_input_tokens"


_TOKEN_KEYS_ALL: tuple[str, ...] = (
    _TOKEN_IN_KEY,
    _TOKEN_OUT_KEY,
    _TOKEN_CACHE_CREATE_KEY,
    _TOKEN_CACHE_READ_KEY,
)


def _coerce_token(value: Any) -> int:
    """Coerce a token counter to int, treating ``None`` / bad as 0.

    The rollup sums tokens, so a missing counter contributes 0 here (the
    ``None``-vs-0 distinction matters only on the raw per-call rows, which
    we preserve verbatim in ``decision_trace.jsonl``).

    Args:
        value (Any): A raw token-counter value.

    Returns:
        int: The integer count, or ``0`` when ``value`` is ``None`` / not
        numeric.
    """
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _empty_token_bucket() -> dict[str, int]:
    """Return a fresh, zeroed token-rollup bucket.

    Returns:
        A dict with zeroed input/output/cache token totals and call count.
    """
    return {
        "total_in": 0,
        "total_out": 0,
        "total_cache_creation": 0,
        "total_cache_read": 0,
        "calls": 0,
    }


def _fold_call_into_bucket(bucket: dict[str, int], call: dict[str, Any]) -> None:
    """Add one call's token counts into a rollup bucket in place.

    Args:
        bucket: Token bucket to accumulate into (mutated).
        call: Per-call record carrying token counters.
    """
    bucket["total_in"] += _coerce_token(call.get(_TOKEN_IN_KEY))
    bucket["total_out"] += _coerce_token(call.get(_TOKEN_OUT_KEY))
    bucket["total_cache_creation"] += _coerce_token(call.get(_TOKEN_CACHE_CREATE_KEY))
    bucket["total_cache_read"] += _coerce_token(call.get(_TOKEN_CACHE_READ_KEY))
    bucket["calls"] += 1


def _load_llm_calls(
    session_dir: Path,
    warnings: list[str],
) -> list[dict[str, Any]]:
    """Read every LLM-call row from ``reports/trace/llm_calls.jsonl``.

    Merges ``reports/trace/llm_calls.jsonl`` with every
    ``reports/trace/ext/*.jsonl`` shard written by out-of-process children.
    Best-effort: missing files / dirs yield ``[]``; malformed lines are
    skipped by :func:`_load_jsonl_safe`.

    Args:
        session_dir (Path): Absolute session root.
        warnings (list[str]): Shared warnings list (mutated in place on scan
            failures).

    Returns:
        list[dict[str, Any]]: Every well-formed LLM-call row across the ledger
        and ext shards. Empty when no trace files exist.
    """
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
    return [r for r in rows if isinstance(r, dict)]


def aggregate_session_cache_tokens(
    session_dir: Path,
    warnings: list[str] | None = None,
) -> tuple[int, int]:
    """Sum (cache_creation, cache_read) over ``reports/trace/llm_calls.jsonl``.

    Reuses the same ledger read + fold as the token rollup, so the figures match
    ``session_breakdown.json``. Best-effort: missing trace files yield ``(0, 0)``.
    """
    warns = warnings if warnings is not None else []
    bucket = _empty_token_bucket()
    for call in _load_llm_calls(session_dir, warns):
        _fold_call_into_bucket(bucket, call)
    return bucket["total_cache_creation"], bucket["total_cache_read"]


def _load_proposal_task_map(
    session_dir: Path, warnings: list[str],
) -> dict[str, str]:
    """Read ``reports/trace/proposal_task_map.jsonl`` into ``{msg_id: task_id}``.

    Written by the Coordinator when an approved proposal is materialized into a
    task. Lets the join attribute a Critic review call (which only carries the
    reviewed proposal ``msg_id``) to the decision the proposal became. Later
    rows win on duplicate msg_id. Best-effort: missing file yields ``{}``.
    """
    rows = _load_jsonl_safe(
        session_dir / "reports" / "trace" / "proposal_task_map.jsonl", warnings,
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
    calls: list[dict[str, Any]], msg_to_task: dict[str, str],
) -> None:
    """Backfill ``task_id`` on Critic review calls from the proposal→task map.

    A Critic reasoning call records the proposal ``msg_id``s it reviewed but not
    a ``task_id`` (the task is materialized only after approval). We only
    attribute a call that reviewed exactly ONE proposal which resolves to
    exactly one task, so it joins that decision through the normal key path.

    A *batch* review (several reviewed msg_ids) judged all of those proposals
    together, so folding its whole token spend onto a single decision — even
    when only one of the batch was later materialized (the others rejected /
    not materialized, so absent from the map) — would over-attribute that
    decision's cost and under-count overhead. Such batch reviews, plus
    ambiguous (multiple distinct tasks) or unresolvable reviews, are left
    unkeyed (→ overhead). The call dicts are mutated in place. No-op when the
    map is empty.
    """
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
        # Single-target review only: a partial mapping (reviewed several, only
        # one materialized) must NOT collapse the batch's cost onto that one.
        if len(reviewed_ids) == 1 and len(resolved) == 1:
            call["task_id"] = next(iter(resolved))


def _load_dispatch_history_all(
    session_dir: Path,
    warnings: list[str],
) -> list[dict[str, Any]]:
    """Read every dynamic_action ``dispatch_history.jsonl`` row.

    Walks ``agents/orchestration/dynamic_actions/<dyn_id>/`` and stamps
    each row with its owning ``dyn_id`` so the join can key on it. Returns
    ``[]`` when the dynamic_actions tree is absent (no dynamic actions ran).

    Args:
        session_dir (Path): Absolute session root.
        warnings (list[str]): Shared warnings list (mutated in place on scan
            failures).

    Returns:
        list[dict[str, Any]]: Every dispatch-history row, each stamped with its
        owning ``dyn_id``. Empty when no dynamic actions ran.
    """
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
    """Build a sorted ``[(entered_unix, phase), ...]`` timeline.

    Derived from ``state.phase_history`` rows that carry a ``to_phase``
    (real transitions). Used to backfill a call's / decision's phase from
    its ``ts`` when the producer didn't stamp one (out-of-process children,
    sub-agent runners, the proposal_scorer off the dispatch path). Empty when
    phase_history is missing.

    Args:
        state (dict[str, Any]): Parsed ``state.json``.

    Returns:
        list[tuple[float, str]]: ``(entered_unix, phase)`` pairs sorted by
        timestamp. Empty when ``phase_history`` is missing.
    """
    history = state.get("phase_history") or []
    if not isinstance(history, list):
        return []
    windows: list[tuple[float, str]] = []
    for row in history:
        if not isinstance(row, dict):
            continue
        to_phase = str(row.get("to_phase") or "").strip()
        if not to_phase:
            continue
        ts_unix = row.get("ts_unix")
        if ts_unix is None:
            ts_unix = _parse_iso_unix(row.get("ts"))
        if ts_unix is None:
            continue
        windows.append((float(ts_unix), to_phase))
    windows.sort(key=lambda w: w[0])
    return windows


def _phase_at(ts: Any, windows: list[tuple[float, str]]) -> str:
    """Return the phase active at ``ts`` per ``windows`` (latest <= ts).

    Args:
        ts (Any): An ISO-8601 timestamp (or numeric Unix value).
        windows (list[tuple[float, str]]): The ``(entered_unix, phase)``
            timeline from :func:`_build_phase_windows`.

    Returns:
        str: The latest phase whose boundary is ``<= ts``, or ``""`` when ``ts``
        is unparseable or the timeline is empty.
    """
    unix = _parse_iso_unix(ts)
    if unix is None or not windows:
        return ""
    phase = ""
    for entered, name in windows:
        if entered <= unix:
            phase = name
        else:
            break
    return phase


# Components whose unjoined LLM spend is *legitimately* not tied to a single
# decision (planning / review / monitoring), bucketed as ``overhead`` rather
# than ``unattributed`` so the attribution ledger separates "expected shared
# cost" from "a real attribution gap to chase".
_OVERHEAD_COMPONENTS: frozenset[str] = frozenset({
    "orchestration",
    "critic",
    "robustness",
})


def _decision_key(task_id: str, dyn_id: str) -> str | None:
    """Canonical join key for a decision / call: ``dyn_id`` wins over
    ``task_id`` (a dynamic_action dispatch owns both). ``None`` when
    neither is present (the call can only be ts-window bucketed).

    Args:
        task_id (str): The call's / decision's task id.
        dyn_id (str): The owning dynamic-action id.

    Returns:
        str | None: ``"dyn:<dyn_id>"`` when a dyn id is present, else
        ``"task:<task_id>"``, else ``None``.
    """
    d = (dyn_id or "").strip()
    if d:
        return f"dyn:{d}"
    t = (task_id or "").strip()
    if t:
        return f"task:{t}"
    return None


def _token_convenience(bucket: dict[str, Any] | None) -> dict[str, Any]:
    """Copy a token bucket and add ``total_in_out`` + ``grand_total``.

    Handles both bucket shapes: the rollup view (split
    ``total_cache_creation`` / ``total_cache_read``) and the per-decision view
    (pre-summed ``total_cache``). ``total_in_out`` is prompt+completion only;
    ``grand_total`` folds in every cache token too.

    Args:
        bucket (dict[str, Any] | None): A token bucket in either shape, or
            ``None``.

    Returns:
        dict[str, int]: A copy of the bucket with added ``total_in_out`` and
        ``grand_total`` figures.
    """
    b = dict(bucket or {})
    ti = int(b.get("total_in", 0) or 0)
    to = int(b.get("total_out", 0) or 0)
    cache = (
        int(b.get("total_cache", 0) or 0)
        + int(b.get("total_cache_creation", 0) or 0)
        + int(b.get("total_cache_read", 0) or 0)
    )
    b["total_in_out"] = ti + to
    b["grand_total"] = ti + to + cache
    cc = int(b.get("total_cache_creation", 0) or 0)
    cr = int(b.get("total_cache_read", 0) or 0)
    b["cache_hit_rate"] = round(cr / (cc + cr), 4) if (cc + cr) else 0.0
    return b


def collect_token_usage(
    decision_trace: dict[str, Any],
    action_timeline: list[dict[str, Any]],
    warnings: list[str],
) -> dict[str, Any]:
    """Promote the token rollup to a discoverable top-level ``token_usage``.

    Pure / derived: reuses the rollup already computed by
    :func:`collect_decision_trace` (no second ledger read), so the totals here
    always reconcile with ``decision_trace``. Adds:

    * ``session_total`` / ``by_component`` / ``by_phase`` — every call, with
      ``total_in_out`` + ``grand_total`` convenience figures.
    * ``attribution`` — decision-attributed vs unattributed split (most
      orchestration / kernel / critic / proposal_scorer turns carry no
      decision key, so they land in ``unattributed``).
    * ``timeline`` — each ``action_timeline`` row annotated with the tokens
      that join to it on ``task_id`` (``None`` when an action has no LLM spend).

    Empty-but-valid (zeroed ``session_total``) when ``decision_trace`` is empty
    (e.g. a pre-trace session), so downstream readers never KeyError.

    Args:
        decision_trace (dict[str, Any]): The output of
            :func:`collect_decision_trace` (supplies the token rollup).
        action_timeline (list[dict[str, Any]]): The visible action timeline to
            annotate with joined token figures.
        warnings (list[str]): Shared warnings list (kept for a uniform
            collector signature; not mutated here).

    Returns:
        dict[str, Any]: The ``token_usage`` section (session total, per
        component / phase, decision-attribution split, and the annotated
        timeline).
    """
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
            int(session_total.get(k, 0) or 0)
            - int(unattributed.get(k, 0) or 0)
            - int(overhead.get(k, 0) or 0)
        )
    total_calls = int(session_total.get("calls", 0) or 0)
    attr_calls = int(attributed.get("calls", 0) or 0)
    overhead_calls = int(overhead.get("calls", 0) or 0)
    attributed_calls_pct = (
        round(100.0 * attr_calls / total_calls, 2) if total_calls else 0.0
    )
    overhead_calls_pct = (
        round(100.0 * overhead_calls / total_calls, 2) if total_calls else 0.0
    )

    # Per-task token map from the per-decision view (only decision-bearing
    # task_ids carry tokens — i.e. the attributed subset).
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
            "but appear as tokens=null in timeline (see attribution.unattributed)."
        ),
    }


def collect_langfuse(
    session_dir: Path,
    manifest: dict[str, Any],
    warnings: list[str],
) -> dict[str, Any]:
    """Assemble the ``langfuse`` section: was the trace pushed live, and how much.

    Two-tier source (the breakdown is normally written *before* the
    session-end ``flush_session``, so the on-disk receipt may not exist yet):

    1. ``reports/trace/langfuse_receipt.json`` if present -- the post-flush
       receipt with final counts (``receipt_source="receipt_file"``).
    2. Otherwise a live read of the per-session emitter singleton -- reports
       the gating + redacted config + in-process running counts
       (``receipt_source="live_emitter"``, ``counts_final=False``).

    Either way credentials are redacted to host + presence booleans. Never
    raises: any failure degrades to a minimal ``config_only`` view so the
    breakdown still records whether the feature was even on.

    Args:
        session_dir (Path): Absolute session root.
        manifest (dict[str, Any]): Parsed ``manifest.json``.
        warnings (list[str]): Shared warnings list (mutated in place on receipt
            / emitter read failures).

    Returns:
        dict[str, Any]: The ``langfuse`` section, tagged with a
        ``receipt_source`` of ``receipt_file`` / ``live_emitter`` /
        ``config_only`` depending on which tier resolved.
    """
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

    # Tier 3 fallback: config-only view straight from env + manifest, so the
    # breakdown still records whether the feature was configured at all.
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
    """Index ``specialist_rounds[].ensemble_scores`` by variant name.

    Returns ``{variant_name: [{rater, score, reason}, ...]}`` so a decision row
    can show who scored the proposal and how (the proposal_scorer signal that
    fed the KEEP/REVERT). Best-effort: shape drift yields an empty map.

    Args:
        state (dict[str, Any]): Parsed ``state.json``.

    Returns:
        dict[str, list[dict[str, Any]]]: Per-variant rater scores. Empty when
        no ensemble scores are present.
    """
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
    """Join the token ledger to the decision streams into one timeline.

    Implements FULL_TRACE_DESIGN §6: read the per-call token rows
    (``reports/trace/llm_calls.jsonl``), read the
    decision rows (``optimization_journal.json`` KEEP/REVERT entries +
    every dynamic_action ``dispatch_history.jsonl``), then attach each
    decision's LLM calls by the shared ``task_id`` / ``dyn_id`` key, with a
    ``ts``-window phase fallback for calls that carry neither.

    Side effect (best-effort): writes the joined timeline to
    ``reports/trace/decision_trace.jsonl``. A write failure is swallowed —
    the in-breakdown section is the authoritative product and must not be
    lost to a disk error.

    Returns ``{"decision_trace": [...], "token_rollup": {...},
    "unattributed_tokens": {...}}``. All-empty (zeroed rollup) when no
    trace files exist, so a session that ran before the trace subsystem
    landed degrades cleanly.

    Args:
        session_dir (Path): Absolute session root.
        state (dict[str, Any]): Parsed ``state.json``.
        warnings (list[str]): Shared warnings list (mutated in place).

    Returns:
        dict[str, Any]: ``{"decision_trace", "token_rollup",
        "unattributed_tokens"}`` — the joined timeline plus token rollups.
    """
    calls = _load_llm_calls(session_dir, warnings)
    phase_windows = _build_phase_windows(state)
    scores_by_variant = _proposal_scores_by_variant(state)

    # Item 2: attribute Critic review calls to the decision their reviewed
    # proposal became (msg_id -> task_id), so critic spend that served a single
    # materialized proposal joins that decision instead of landing in overhead.
    _attribute_critic_calls(calls, _load_proposal_task_map(session_dir, warnings))

    # ── Index calls by decision key; orphans (no key) go to a ts list ──
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

    # ── Gather decisions from the journal + dispatch_history ──
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
        # ``component`` is now the real proposer (specialist:<domain> / grid /
        # orchestration) derived from provenance, not a hard-coded constant, so
        # the decision timeline answers "who proposed this".
        decision: dict[str, Any] = {
            "component": proposer_for(provenance) if provenance else "orchestration",
            "change": str(e.get("change") or ""),
            "outcome": str(e.get("outcome") or ""),
            "gain_pct": _to_float(e.get("gain_pct")),
            "task_id": task_id,
            "operation_kind": operation_kind_for("", change_kind),
        }
        # Predicted (pre-measurement) gain, when the proposer supplied one, so
        # the decision row carries predicted-vs-realized for calibration.
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
            # Attach the proposal_scorer signal (who rated this proposal, how)
            # so the decision step answers "was it scored, by whom, how high".
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

    # ── Attach calls to decisions; build the joined trace ──
    # A given key's calls attach to exactly ONE decision — the first by ts
    # (e.g. a dynamic_action's DISPATCHED event anchors all its sub-agent
    # turns). Later same-key events (CRITIC_VERDICT / INTEGRATE_RESULT) get
    # empty token buckets so the per-decision sums don't double-count.
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
                    "calls": agg["calls"],
                },
            }
        )

    # ── Unjoined calls: keyed calls with no matching decision + orphans ──
    # These still count toward the session total + phase/component rollup so
    # the books balance, but they don't anchor to a decision row. We split
    # them two ways for an honest attribution ledger:
    #   * ``overhead``      — inherently cross-decision LLM spend (planning /
    #                         review / monitoring) that legitimately has no
    #                         single owning decision (orchestration / critic /
    #                         robustness reactor turns).
    #   * ``unattributed``  — everything else that *should* have carried a
    #                         decision key but didn't (a real gap to chase).
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

    # ── Rollups: by_phase + by_component + session_total (ALL calls) ──
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

    # ── Best-effort side write of the joined timeline (design §6.4) ──
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
    """Append-free atomic-ish write of ``reports/trace/decision_trace.jsonl``.

    Rewrites the whole file (one JSON object per decision) on each export;
    the collector is the single producer, so a full rewrite is simpler than
    append + dedup and stays consistent with the latest join. Best-effort:
    OSError is recorded in ``warnings`` and swallowed.

    Args:
        session_dir (Path): Absolute session root.
        decision_trace (list[dict[str, Any]]): The joined decision-trace rows
            to write (one JSON object per line).
        warnings (list[str]): Shared warnings list (mutated in place on write
            failure).
    """
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

