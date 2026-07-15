# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Deterministic collectors for ``session_breakdown.json``.

Each ``collect_<section>`` is a pure function over ``session_dir`` /
``state`` / ``manifest`` returning its schema section (see :mod:`.schema`).
Collectors never mutate state, fabricate values, or raise — failures are
recorded in ``warnings`` and the section returns a best-effort partial.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


from ._common import (
    _to_float,
)



def _normalize_specialist_key(provenance: str) -> str:
    """Map a raw ``provenance`` to a stable specialist key.

    ``specialist:<domain>`` → bare ``<domain>``; ``legacy:<action>`` →
    ``legacy_<action>``; ``default_grid`` / ``llm_direct`` pass through;
    empty/unknown → ``"unknown"`` (so by_domain is never keyed by ``""``).

    Args:
        provenance (str): The raw provenance label.

    Returns:
        str: The normalized specialist key.
    """
    s = (provenance or "").strip()
    if not s:
        return "unknown"
    if s.startswith("specialist:"):
        # Trust the orchestrator's domain string verbatim (no parallel update needed for new domains).
        return s[len("specialist:") :] or "unknown"
    if s.startswith("legacy:"):
        return f"legacy_{s[len('legacy:') :]}"
    return s


# Ordered ``(predicate, family)`` table for :func:`_action_family`. Matched
# top-to-bottom on the lowercased action label; the FIRST hit wins, so the
# order is load-bearing (e.g. the ``kernel_opt`` prefix must precede the exact
# ``==`` checks below it). Falls through to ``"other"`` when nothing matches.
_ACTION_FAMILY_TABLE: tuple[tuple[Callable[[str], bool], str], ...] = (
    (lambda s: s.startswith("kernel_opt") or s == "integrate", "kernel_agent"),
    # Legacy stack-entry action labels from archived sessions.
    (lambda s: s == "backends", "backends"),
    (lambda s: s == "params", "params"),
    (lambda s: s == "validate_stack", "validate"),
    (lambda s: s == "sweep", "sweep"),
    # merged explore family subsuming the legacy backends + params buckets.
    (lambda s: s == "explore", "explore"),
    # REPLAY_WARM_RECIPE: warm-recipe / cortex best_config replay (a prep action).
    # Its own headline row so its gain reconciles against validated_total_pct
    # instead of vanishing into the non-emitted ``other`` family. The label may
    # carry a tier suffix (``replay_warm_recipe:exact``), so match the base token.
    (lambda s: s.split(":", 1)[0] == "replay_warm_recipe", "replay_warm_recipe"),
    # FRAMEWORK: own headline row so per-source totals reconcile against
    # validated_total_pct (else these KEEPs fell into ``other`` and vanished).
    (lambda s: s == "framework", "framework"),
    # GEMM_TUNING: deterministic FP8 tuner KEEPs, bucketed apart from generic
    # ``kernel`` so the dashboard can split tuner vs source-level rewrite gain.
    (lambda s: s == "gemm_tuning", "gemm_tuning"),
    # GEAK e2e: whole-pipeline KERNEL-phase optimizer, bucketed apart
    # from generic ``kernel`` so its gain gets a dedicated row instead of vanishing into
    # ``other`` or being mis-credited to a backend.
    (lambda s: s == "geak_e2e", "geak"),
)


def _action_family(action: str) -> str:
    """Map an action label to a family for source_breakdown bucketing.

    Args:
        action (str): A stack-entry / gain-ledger action label.

    Returns:
        str: One of ``kernel`` / ``backends`` / ``params`` / ``validate`` /
        ``sweep`` / ``explore`` / ``replay_warm_recipe`` / ``framework`` /
        ``gemm_tuning``, or ``"other"`` when unrecognized.
    """
    s = (action or "").lower()
    for predicate, family in _ACTION_FAMILY_TABLE:
        if predicate(s):
            return family
    return "other"


def _promote_legacy_gain_entries(
    state_entries: list[Any],
    state: dict[str, Any],
) -> list[dict[str, Any]]:
    """Lift a pre-v0.7 ``list[float | None]`` gain ledger into the V1 schema.

    Recovers action/variant/ts/args from the parallel
    ``optimization_stack`` and computes ``delta_pct`` against the prior
    entry; ``None`` ledger entries stay ``None`` to keep index alignment.

    Args:
        state_entries (list[Any]): The legacy numeric gain ledger.
        state (dict[str, Any]): Parsed ``state.json`` (supplies the parallel
            ``optimization_stack``).

    Returns:
        list[dict[str, Any]]: The promoted V1 ``StackGainEntry`` rows.
    """
    stack = state.get("optimization_stack") or []
    out: list[dict[str, Any]] = []
    prev_cum = 0.0
    for i, val in enumerate(state_entries):
        cum_after: float | None
        if isinstance(val, (int, float)):
            cum_after = float(val)
        else:
            cum_after = None
        delta = (cum_after - prev_cum) if cum_after is not None else None
        se = stack[i] if i < len(stack) and isinstance(stack[i], dict) else {}
        promoted: dict[str, Any] = {
            "ts": str(se.get("ts") or ""),
            "action": str(se.get("action") or ""),
            "variant_name": se.get("variant_name") or se.get("kernel_id"),
            "stack_len_before": i,
            "stack_len_after": i + 1,
            "cum_gain_before": prev_cum,
            "cum_gain_after": cum_after,
            "delta_pct": delta,
            "extra_server_args": str(se.get("extra_server_args") or se.get("candidate_extra_server_args") or ""),
        }
        # Carry the explore join key / source forward when the Coordinator
        # stamped them, so phase_breakdown.explore.by_domain can attribute
        # the gain to its specialist provenance instead of ``default_grid``.
        fp = str(se.get("fingerprint") or se.get("variant_fingerprint") or "")
        if fp:
            promoted["fingerprint"] = fp
        prov = str(se.get("provenance") or "").strip()
        if prov:
            promoted["provenance"] = prov
        out.append(promoted)
        if cum_after is not None:
            prev_cum = cum_after
    return out


def collect_attribution(
    state: dict[str, Any],
    geak_invocations: list[dict[str, Any]],
    adopted_kernels: list[dict[str, Any]],
    warnings: list[str],
    forge_invocations: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Attribute end-to-end gains to individual optimization-stack entries.

    Prefers the authoritative ``gain_per_stack_entry`` ledger and falls back to
    reconstructing attribution from the optimization stack.

    Args:
        state: Session state mapping.
        geak_invocations: GEAK backend invocation records.
        adopted_kernels: Kernels adopted into the optimized stack.
        warnings: Mutable list that collected warnings are appended to.
        forge_invocations: Forge backend invocation records (own lane).

    Returns:
        An attribution dict mapping stack entries to their measured gains.
    """
    forge_invocations = forge_invocations or []
    # Prefer the authoritative ``gain_per_stack_entry`` ledger; else reconstruct from optimization_stack.
    state_entries = state.get("gain_per_stack_entry")
    state_provided = isinstance(state_entries, list) and len(state_entries) > 0
    promoted_from_legacy = False
    if state_provided and any(not isinstance(e, dict) for e in state_entries):
        # Older state: bare numeric ledger; promote into the V1 schema.
        entries = _promote_legacy_gain_entries(state_entries, state)
        promoted_from_legacy = True
    elif state_provided:
        entries = list(state_entries)
    else:
        entries = _reconstruct_gain_ledger(state, warnings)

    # Classify the attribution lineage for an honest provenance label.
    stack = state.get("optimization_stack") or []
    stack_len = len(stack) if isinstance(stack, list) else 0
    method: str
    if state_provided:
        if promoted_from_legacy:
            # Lifted post-hoc, not a per-event capture.
            method = "reconstructed"
        else:
            all_deltas_set = all(isinstance(e, dict) and e.get("delta_pct") is not None for e in state_entries)
            method = "validated" if all_deltas_set else "reconstructed"
    elif stack_len == 1:
        # Single-entry stack: one unambiguous source of gain.
        method = "single_source"
    elif stack_len > 1:
        method = "reconstructed"
    else:
        method = "missing"

    # Bucket entries by family for source_breakdown; validated total is the denominator.
    validated_total = _to_float(state.get("cumulative_gain_validated")) or 0.0
    family_totals: dict[str, float] = {
        "kernel_agent": 0.0,
        "sweep": 0.0,
        "other": 0.0,
        # Legacy buckets for archived (pre-merge) sessions.
        "backends": 0.0,
        "params": 0.0,
        "validate": 0.0,
        # unified explore family (subsumes backends + params).
        "explore": 0.0,
        # FRAMEWORK family, kept apart from ``other`` for a dedicated row.
        "framework": 0.0,
        # REPLAY_WARM_RECIPE family: warm-recipe replay, kept apart from ``other``
        # so its gain gets a dedicated headline row.
        "replay_warm_recipe": 0.0,
        # GEMM_TUNING family, kept apart from ``kernel`` (deterministic tuner vs rewrite).
        "gemm_tuning": 0.0,
        # GEAK e2e family: whole-pipeline KERNEL-phase optimizer.
        "geak": 0.0,
    }
    for e in entries:
        if not isinstance(e, dict):
            continue
        delta = _to_float(e.get("delta_pct"))
        if delta is None:
            continue
        fam = _action_family(str(e.get("action") or ""))
        family_totals[fam] = family_totals.get(fam, 0.0) + max(delta, 0.0)

    # Split "kernel_agent" between active per-kernel backends based on adopted KEEP entries.
    forge_kept_kids = {inv.get("kernel_id") for inv in forge_invocations if inv.get("decision") == "KEEP"}
    kernel_total = family_totals.get("kernel_agent", 0.0)
    forge_total = 0.0
    for k in adopted_kernels:
        kid = k.get("kernel_id")
        gain = _to_float(k.get("e2e_gain_pct")) or 0.0
        if kid in forge_kept_kids:
            forge_total += gain
    if forge_total == 0.0 and kernel_total > 0.0:
        forge_total = kernel_total

    notes: list[str] = []
    if not state_provided:
        notes.append(
            "gain_per_stack_entry not written by Coordinator; "
            "attribution reconstructed best-effort from optimization_stack."
        )
    elif promoted_from_legacy:
        notes.append(
            "gain_per_stack_entry was a pre-v0.7 numeric ledger; "
            "promoted to V1 StackGainEntry shape using parallel data from "
            "optimization_stack (delta_pct computed as diff vs prior entry's "
            "cum_gain_after)."
        )

    # per-phase gain breakdown (buckets each KEEP by its phase_history-active phase).
    phase_breakdown = _collect_phase_breakdown(state, entries, warnings)

    return {
        "gain_per_stack_entry": entries,
        "method": method,
        "source_breakdown": {
            "forge_pct_of_total": round(forge_total, 2),
            # primary row.
            "explore_pct_of_total": round(family_totals.get("explore", 0.0), 2),
            # REPLAY_WARM_RECIPE row; always emitted (0.0 when no warm-recipe
            # replay was reproduced/adopted this session).
            "replay_warm_recipe_pct_of_total": round(
                family_totals.get("replay_warm_recipe", 0.0), 2
            ),
            # FRAMEWORK row; always emitted (0.0 when disabled/empty).
            "framework_pct_of_total": round(family_totals.get("framework", 0.0), 2),
            # GEMM_TUNING row; always emitted (0.0 when non-FP8/skipped/no KEEP).
            "gemm_tuning_pct_of_total": round(family_totals.get("gemm_tuning", 0.0), 2),
            # GEAK e2e row (whole-pipeline optimizer, now the canonical ``geak``);
            # always emitted (0.0 when native/no e2e win).
            "geak_pct_of_total": round(family_totals.get("geak", 0.0), 2),
            # Legacy rows, kept so archived-session reports reconcile (0.0 on current sessions).
            "backends_pct_of_total": round(family_totals.get("backends", 0.0), 2),
            "params_pct_of_total": round(family_totals.get("params", 0.0), 2),
            "sweep_pct_of_total": round(family_totals.get("sweep", 0.0), 2),
            "validated_total_pct": round(validated_total, 2),
        },
        "phase_breakdown": phase_breakdown,
        "notes": notes,
    }


def _collect_phase_breakdown(
    state: dict[str, Any],
    entries: list[dict[str, Any]],
    warnings: list[str],
) -> dict[str, Any]:
    """KB_design §3.12 §4.6 + §3.13 M7 §6 — per-phase gain attribution.

    Assigns each KEEP entry to the phase active at its acceptance
    timestamp (explore further splits by domain, kernel by kernel_id).
    Missing phase_history → everything lands under ``unattributed``.

    Args:
        state (dict[str, Any]): Parsed ``state.json``.
        entries (list[dict[str, Any]]): The per-stack gain-ledger entries.
        warnings (list[str]): Shared warnings list (mutated in place when
            ``phase_history`` is empty).

    Returns:
        dict[str, Any]: Per-phase gain buckets (prelude / framework /
        explore / kernel / gemm_tuning / sweep / close, plus a conditional
        ``unattributed``), each with a ``total_gain_pct`` and phase-specific
        sub-breakdowns.
    """
    # phase timeline lookup: for an entry ts, pick the latest row with ts_unix ≤ ts.
    history = state.get("phase_history") or []
    if not isinstance(history, list):
        history = []
    timeline: list[tuple[float, str]] = []
    for row in history:
        if not isinstance(row, dict):
            continue
        try:
            ts = float(row.get("ts_unix") or 0.0)
        except (TypeError, ValueError):
            ts = 0.0
        phase = str(row.get("to_phase") or "").strip().upper()
        if phase:
            timeline.append((ts, phase))
    timeline.sort(key=lambda r: r[0])

    def _phase_for(ts_unix: float) -> str:
        """Resolve which phase owned a given timestamp.

        Args:
            ts_unix (float): Unix timestamp of a gain-ledger entry.

        Returns:
            str: The latest phase whose boundary is ``<= ts_unix``, or ``""``
            when the timeline is empty.
        """
        if not timeline:
            return ""
        current = ""
        for ts, ph in timeline:
            if ts <= ts_unix:
                current = ph
            else:
                break
        return current

    # explore provenance: map fingerprint → provenance from winners_history.
    # ``scope_by_fp`` carries the orthogonal specialist dial (domain / domains
    # / freeform) as an additive analytics tag; absent on legacy sessions.
    explore_search = state.get("explore_search") or {}
    provenance_by_fp: dict[str, str] = {}
    scope_by_fp: dict[str, str] = {}
    if isinstance(explore_search, dict):
        for w in explore_search.get("winners_history") or []:
            if not isinstance(w, dict):
                continue
            fp = str(w.get("fingerprint") or "")
            prov = str(w.get("provenance") or "").strip()
            if fp and prov:
                provenance_by_fp[fp] = prov
            sc = str(w.get("scope") or "").strip()
            if fp and sc:
                scope_by_fp[fp] = sc

    phase_buckets: dict[str, dict[str, Any]] = {
        "prelude": {"total_gain_pct": 0.0},
        # FRAMEWORK: upstream-PR bake-in phase; by_pr keyed per adopted PR.
        "framework": {"total_gain_pct": 0.0, "by_pr": {}},
        "explore": {"total_gain_pct": 0.0, "by_domain": {}},
        "kernel_agent": {"total_gain_pct": 0.0, "by_kernel_id": {}},
        # GEMM_TUNING: KERNEL-entry tuner, bucketed apart; by_tuned_file keyed on the produced CSV.
        "gemm_tuning": {"total_gain_pct": 0.0, "by_tuned_file": {}},
        "sweep": {"total_gain_pct": 0.0},
        "close": {"total_gain_pct": 0.0},
        "unattributed": {"total_gain_pct": 0.0},
    }

    for e in entries:
        if not isinstance(e, dict):
            continue
        delta = _to_float(e.get("delta_pct"))
        if delta is None or delta <= 0:
            continue
        ts = e.get("ts_unix")
        if ts is None:
            ts_str = str(e.get("ts") or "")
            if ts_str:
                try:
                    from datetime import datetime as _dt

                    ts = _dt.fromisoformat(ts_str.replace("Z", "+00:00")).timestamp()
                except (TypeError, ValueError):
                    ts = 0.0
        try:
            ts_f = float(ts or 0.0)
        except (TypeError, ValueError):
            ts_f = 0.0
        phase = _phase_for(ts_f).lower()
        action = str(e.get("action") or "").lower()
        fam = _action_family(action)
        # gemm_tuning runs inside KERNEL but is bucketed separately, so
        # override the coarser phase_history KERNEL label by family.
        if fam == "gemm_tuning":
            phase = "gemm_tuning"
        # Fall back to action family when phase_history isn't usable.
        elif phase not in phase_buckets:
            if fam in ("explore", "backends", "params"):
                phase = "explore"
            elif fam == "kernel_agent":
                phase = "kernel_agent"
            elif fam == "sweep":
                phase = "sweep"
            elif fam == "framework":
                phase = "framework"
            else:
                phase = "unattributed"
        bucket = phase_buckets[phase]
        bucket["total_gain_pct"] = round(
            float(bucket["total_gain_pct"]) + float(delta),
            2,
        )
        if phase == "explore":
            by_domain = bucket.setdefault("by_domain", {})
            fp = str(e.get("fingerprint") or e.get("variant_fingerprint") or "")
            raw_prov = provenance_by_fp.get(fp) or str(e.get("provenance") or "") or "default_grid"
            # Normalize to a bare specialist key (see ``_normalize_specialist_key``).
            domain = _normalize_specialist_key(raw_prov)
            by_domain[domain] = round(
                float(by_domain.get(domain, 0.0)) + float(delta),
                2,
            )
            # Additive scope split (specialist dial); legacy sessions with no
            # ``scope`` recorded collapse into the ``unspecified`` bucket so
            # the totals still reconcile against ``total_gain_pct``.
            by_scope = bucket.setdefault("by_scope", {})
            scope_key = scope_by_fp.get(fp) or str(e.get("scope") or "") or "unspecified"
            by_scope[scope_key] = round(
                float(by_scope.get(scope_key, 0.0)) + float(delta),
                2,
            )
        elif phase == "kernel_agent":
            by_kid = bucket.setdefault("by_kernel_id", {})
            kid = str(e.get("kernel_id") or e.get("action_kernel_id") or "?")
            by_kid[kid] = round(
                float(by_kid.get(kid, 0.0)) + float(delta),
                2,
            )
        elif phase == "framework":
            # Key on the PR ref (variant_name), falling back to ``ref`` then ``?``.
            by_pr = bucket.setdefault("by_pr", {})
            pr_key = str(e.get("variant_name") or "").strip() or str(e.get("ref") or "").strip() or "?"
            by_pr[pr_key] = round(
                float(by_pr.get(pr_key, 0.0)) + float(delta),
                2,
            )
        elif phase == "gemm_tuning":
            # Key on the tuned CSV path, falling back to ``variant_name`` then ``?``.
            by_tuned = bucket.setdefault("by_tuned_file", {})
            tuned_key = str(e.get("tuned_file") or "").strip() or str(e.get("variant_name") or "").strip() or "?"
            by_tuned[tuned_key] = round(
                float(by_tuned.get(tuned_key, 0.0)) + float(delta),
                2,
            )

    # Drop the unattributed bucket when nothing landed there.
    if phase_buckets["unattributed"]["total_gain_pct"] == 0.0:
        phase_buckets.pop("unattributed", None)

    if not timeline:
        warnings.append("attribution.phase_breakdown: phase_history empty; gains bucketed via action family fallback")

    return phase_buckets


def _reconstruct_gain_ledger(
    state: dict[str, Any],
    warnings: list[str],
) -> list[dict[str, Any]]:
    """Approximate per-stack contribution (each entry's ``gain_pct`` as its delta) when Coordinator didn't record it; best-effort, see ``attribution.notes``.

    Args:
        state (dict[str, Any]): Parsed ``state.json``.
        warnings (list[str]): Shared warnings list (kept for signature
            symmetry; not mutated here).

    Returns:
        list[dict[str, Any]]: One reconstructed gain-ledger row per stack
        entry, with cumulative gain and ``delta_pct``. Empty when there is no
        ``optimization_stack``.
    """
    stack = state.get("optimization_stack") or []
    if not isinstance(stack, list):
        return []
    cum_before = 0.0
    out: list[dict[str, Any]] = []
    for i, entry in enumerate(stack):
        if not isinstance(entry, dict):
            continue
        delta = _to_float(entry.get("gain_pct"))
        cum_after = cum_before + (delta or 0.0)
        row: dict[str, Any] = {
            "ts": str(entry.get("ts") or ""),
            "stack_len_before": i,
            "stack_len_after": i + 1,
            "action": str(entry.get("action") or ""),
            "variant_name": str(entry.get("variant_name") or ""),
            "cum_gain_before": round(cum_before, 4),
            "cum_gain_after": round(cum_after, 4),
            "delta_pct": delta,
            "extra_server_args": str(entry.get("extra_server_args") or ""),
        }
        # Preserve the explore join key / source so attribution can resolve the
        # specialist provenance (else by_domain collapses into ``default_grid``).
        fp = str(entry.get("fingerprint") or entry.get("variant_fingerprint") or "")
        if fp:
            row["fingerprint"] = fp
        prov = str(entry.get("provenance") or "").strip()
        if prov:
            row["provenance"] = prov
        out.append(row)
        cum_before = cum_after
    return out

