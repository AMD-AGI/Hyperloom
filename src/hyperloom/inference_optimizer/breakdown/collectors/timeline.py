# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Deterministic collectors for ``session_breakdown.json``.

Each ``collect_<section>`` is a pure function over ``session_dir`` /
``state`` / ``manifest`` returning its schema section (see :mod:`.schema`).
Collectors never mutate state, fabricate values, or raise — failures are
recorded in ``warnings`` and the section returns a best-effort partial.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from hyperloom.common.timeutil import iso_z
from hyperloom.orchestrator.state.optimization_journal import (
    operation_kind_for,
    proposer_for,
)

from ._common import (
    _load_optimization_journal,
    _parse_iso_unix,
    _to_float,
    _to_int,
)



# Phase timeline
# Action labels whose ``<action>_attempts`` lists feed the timeline +
# capability tallies. Carries both the merged ``explore`` and the legacy
# ``backends`` / ``params`` / ``validate_stack`` names; missing lists skip.
_AUDIT_ACTIONS = (
    "baseline",
    "profile",
    "explore",
    "backends",
    "params",
    "validate_stack",
    "sweep",
    "roofline",
)


def _journal_entry_to_event(e: dict[str, Any]) -> dict[str, Any]:
    """Map one optimization_journal entry to a phase_timeline event (keeps the declared ``phase`` for exact bucketing).

    Args:
        e (dict[str, Any]): One ``optimization_journal.json`` entry.

    Returns:
        dict[str, Any]: The event row with normalized timestamp, resolved
        action, metric / decision fields, and a threaded ``extras`` map.
    """
    metric = e.get("throughput_after")
    metric_kind = "output_throughput" if metric is not None else None
    if metric is None and e.get("gain_pct") is not None:
        metric = e.get("gain_pct")
        metric_kind = "gain_pct"
    change = str(e.get("change") or "")
    kind = str(e.get("kind") or "")
    # ``kind == "other"`` is a catch-all; the real op name lives in ``change``.
    if kind and kind.lower() != "other":
        action = kind
    else:
        action = change or kind or "other"
    provenance = str(e.get("provenance") or "")
    extras = {
        k: v
        for k, v in (
            ("variant_name", e.get("variant_name")),
            ("reason", e.get("reason")),
            # Proposer attribution + stable filter label, threaded so the timeline
            # answers "what / how / who" for each step.
            ("provenance", provenance),
            ("proposer", proposer_for(provenance) if provenance else ""),
            ("scope", str(e.get("scope") or "")),
            ("fingerprint", str(e.get("fingerprint") or "")),
            ("operation_kind", operation_kind_for(action, kind)),
            ("metrics", e.get("metrics") if isinstance(e.get("metrics"), dict) else None),
        )
        if v
    }
    return {
        "ts": iso_z(e.get("ts")),
        "action": action,
        "task_id": str(e.get("task_id") or ""),
        "kernel_id": None,
        "status": "",
        "decision": str(e.get("outcome") or ""),
        "key_metric": _to_float(metric),
        "key_metric_kind": metric_kind,
        "workspace": None,
        "error_class": e.get("error_class"),
        "phase": str(e.get("phase") or ""),
        "change": change,
        "extras": extras,
    }


def collect_phase_timeline(
    session_dir: Path | None,
    state: dict[str, Any],
    warnings: list[str],
) -> list[dict[str, Any]]:
    """Flat, chronological action timeline.

    Merges two complementary sources so no action family is dropped:

    * ``reports/optimization_journal.json`` — the canonical decision log
      (target_analysis / baseline / roofline / specialist / explore
      winners / sweep). Carries ``phase`` for exact segment attribution.
    * the per-action ``*_attempts`` audit lists + ``kernel_opt`` /
      ``kernel_integrate`` histories — add per-attempt rows (incl.
      failures) and the kernel lanes the journal records only as a single
      KEEP.

    Events are de-duplicated by ``(action, ts-to-second, change)`` with
    the journal copy winning, then sorted by ``ts``. Passing
    ``session_dir=None`` degrades gracefully to the audit-list scrape
    (used by unit fixtures that have no on-disk journal).

    Args:
        session_dir (Path | None): Absolute session root, or ``None`` to skip
            the on-disk journal source.
        state (dict[str, Any]): Parsed ``state.json``.
        warnings (list[str]): Shared warnings list (mutated in place).

    Returns:
        list[dict[str, Any]]: The merged, de-duplicated, ts-sorted action
        events.
    """
    events: list[dict[str, Any]] = []

    # ── Source 1: canonical journal (preferred; carries phase) ──
    for e in _load_optimization_journal(session_dir, warnings):
        if isinstance(e, dict):
            events.append(_journal_entry_to_event(e))

    # ── Source 2: per-action audit lists (complementary / legacy) ──
    for action in _AUDIT_ACTIONS:
        attempts = state.get(f"{action}_attempts") or []
        if not isinstance(attempts, list):
            continue
        for entry in attempts:
            if not isinstance(entry, dict):
                continue
            events.append(
                {
                    "ts": entry.get("ts") or "",
                    "action": action,
                    "task_id": str(entry.get("task_id") or ""),
                    "kernel_id": None,
                    "status": str(entry.get("status") or ""),
                    "decision": str(entry.get("decision") or ""),
                    "key_metric": _to_float(entry.get("key_metric")),
                    "key_metric_kind": entry.get("key_metric_kind"),
                    "workspace": entry.get("workspace"),
                    "error_class": entry.get("error_class"),
                    "phase": "",
                    "change": action,
                    "extras": dict(entry.get("extras") or {}),
                }
            )

    # Kernel opt attempts (per-kernel history -> flatten to per-attempt rows)
    kernel_opt = state.get("kernel_opt_attempts") or {}
    if isinstance(kernel_opt, dict):
        for kid, ent in kernel_opt.items():
            if not isinstance(ent, dict):
                continue
            for h in ent.get("history") or []:
                if not isinstance(h, dict):
                    continue
                events.append(
                    {
                        "ts": h.get("ts") or ent.get("last_ts") or "",
                        "action": "kernel_opt",
                        "task_id": "",
                        "kernel_id": str(kid),
                        "status": "",
                        "decision": str(h.get("decision") or ""),
                        "key_metric": None,
                        "key_metric_kind": None,
                        "workspace": None,
                        "error_class": None,
                        "phase": "",
                        "change": f"kernel_opt:{kid}",
                        "extras": {},
                    }
                )

    # Integrate attempts (decision history per patch key)
    integ = state.get("kernel_integrate_attempts") or {}
    if isinstance(integ, dict):
        for key, ent in integ.items():
            if not isinstance(ent, dict):
                continue
            for a in ent.get("attempts") or []:
                if not isinstance(a, dict):
                    continue
                kid = str(ent.get("kernel_id") or "")
                events.append(
                    {
                        "ts": a.get("ts") or "",
                        "action": "integrate",
                        "task_id": "",
                        "kernel_id": kid,
                        "status": str(a.get("status") or ""),
                        "decision": str(a.get("decision") or ""),
                        "key_metric": _to_float(a.get("gain_pct")),
                        "key_metric_kind": "gain_pct",
                        "workspace": a.get("workspace"),
                        "error_class": None,
                        "phase": "",
                        "change": f"integrate:{kid}",
                        "extras": {"patch_path": ent.get("patch_path"), "report_path": a.get("report_path")},
                    }
                )

    # Canonicalise every ts to ``...Z`` so mixed-suffix rows dedup and sort consistently.
    for ev in events:
        ev["ts"] = iso_z(ev.get("ts"))

    # De-dup: journal rows are appended first and win on collision.
    seen: set[tuple[str, str, str]] = set()
    deduped: list[dict[str, Any]] = []
    for ev in events:
        key = (
            str(ev.get("action") or ""),
            (str(ev.get("ts") or ""))[:19],
            str(ev.get("change") or ev.get("task_id") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(ev)

    deduped.sort(key=lambda e: e.get("ts") or "")
    return deduped


# Capability summary
def _capability_for_action(
    state: dict[str, Any],
    action: str,
) -> dict[str, Any]:
    """Per-action capability tally from ``<action>_attempts``, with an ``optimization_stack`` KEEP fallback for V1/partial state.

    Args:
        state (dict[str, Any]): Parsed ``state.json``.
        action (str): The action label whose attempts / keeps are tallied.

    Returns:
        dict[str, Any]: ``{"status", "attempts", "keeps"}`` for the action.
    """
    attempts_list = state.get(f"{action}_attempts") or []
    n_attempts = len(attempts_list) if isinstance(attempts_list, list) else 0
    n_keeps = (
        sum(1 for a in attempts_list if isinstance(a, dict) and a.get("decision") in ("promoted", "salvaged"))
        if isinstance(attempts_list, list)
        else 0
    )

    # Fallback from optimization_stack (adopted entries only), counted per action.
    stack = state.get("optimization_stack") or []
    if isinstance(stack, list):
        stack_keeps = sum(1 for e in stack if isinstance(e, dict) and str(e.get("action") or "") == action)
    else:
        stack_keeps = 0
    if stack_keeps > n_keeps:
        n_keeps = stack_keeps
        # Stack keeps imply at least as many attempts.
        if n_attempts < stack_keeps:
            n_attempts = stack_keeps

    status = "kept" if n_keeps > 0 else "tried" if n_attempts > 0 else "not_attempted"
    return {
        "status": status,
        "attempts": n_attempts,
        "keeps": n_keeps,
    }


def collect_capability_summary(
    state: dict[str, Any],
    geak_invocations: list[dict[str, Any]],
    warnings: list[str],
    forge_invocations: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Summarize kernel-optimization capability outcomes for the breakdown.

    Args:
        state: Session state mapping.
        geak_invocations: GEAK backend invocation records.
        warnings: Mutable list that collected warnings are appended to.
        forge_invocations: Forge backend invocation records (own lane).

    Returns:
        A capability-summary dict (per-kernel status, attempt and keep counts).
    """
    forge_invocations = forge_invocations or []
    # Integrate (e2e) outcome per kernel: a kernel-opt KEEP REVERTED at
    # integrate is not a real adoption, so don't inflate the backend tally.
    integ = state.get("kernel_integrate_attempts") or {}
    integ_by_kid: dict[str, dict[str, Any]] = {}
    if isinstance(integ, dict):
        for ent in integ.values():
            if not isinstance(ent, dict):
                continue
            kid = str(ent.get("kernel_id") or "")
            if not kid:
                continue
            integ_by_kid[kid] = {
                "decision": str(ent.get("last_decision") or "").upper(),
                "e2e_gain_pct": _to_float(ent.get("best_gain_pct")),
            }

    # Kernel backends from on-disk invocations, reconciled against the integrate verdict.
    def _from_invocations(invs: list[dict[str, Any]]) -> dict[str, Any]:
        """Reduce one lane's invocations to a capability row.

        Args:
            invs (list[dict[str, Any]]): A lane's invocation records.

        Returns:
            dict[str, Any]: ``{"status", "attempts", "keeps"}`` where
            ``keeps`` counts ``decision == "KEEP"`` entries.
        """
        attempts = len(invs)
        adopted = 0
        reverted = 0
        best_e2e: float | None = None
        for v in invs:
            if v.get("decision") != "KEEP":
                continue
            outcome = integ_by_kid.get(str(v.get("kernel_id") or ""))
            if outcome is None:
                # micro-KEEP with no integrate record stands as kept.
                adopted += 1
                continue
            g = outcome["e2e_gain_pct"]
            if g is not None and (best_e2e is None or g > best_e2e):
                best_e2e = g
            if outcome["decision"] in ("REVERT", "REJECT"):
                reverted += 1
            else:
                adopted += 1
        status = (
            "kept" if adopted > 0 else "reverted" if reverted > 0 else "attempted" if attempts > 0 else "not_attempted"
        )
        row: dict[str, Any] = {
            "status": status,
            "attempts": attempts,
            "keeps": adopted,
        }
        if reverted:
            row["reverts"] = reverted
        if best_e2e is not None:
            row["e2e_gain_pct"] = best_e2e
        return row

    geak_cap = _from_invocations(geak_invocations)
    forge_cap = _from_invocations(forge_invocations)

    # Legacy capability rows for archived (pre-merge) sessions; current
    # sessions leave these not_attempted and carry activity under ``explore``.
    backends = _capability_for_action(state, "backends")
    backends_search = state.get("backends_search") or {}
    if isinstance(backends_search, dict):
        backends["tested"] = len(backends_search.get("tested") or {})
        if backends_search.get("accepted"):
            backends["best_gain_pct"] = max(
                (_to_float(v.get("gain_pct")) or 0.0 for v in backends_search["accepted"] if isinstance(v, dict)),
                default=None,
            )

    params = _capability_for_action(state, "params")
    params_search = state.get("params_search") or {}
    if isinstance(params_search, dict):
        params["tested"] = len(params_search.get("tested") or {})
        if params_search.get("accepted"):
            params["best_gain_pct"] = max(
                (_to_float(v.get("gain_pct")) or 0.0 for v in params_search["accepted"] if isinstance(v, dict)),
                default=None,
            )

    validate = _capability_for_action(state, "validate_stack")
    validate["last_validated_gain_pct"] = _to_float(state.get("cumulative_gain_validated"))

    sweep_cap = _capability_for_action(state, "sweep")
    last_sweep = state.get("last_sweep") or {}
    if isinstance(last_sweep, dict):
        sweep_cap["grid_size"] = _to_int(last_sweep.get("grid_size"))
        bo = last_sweep.get("best_overall")
        if isinstance(bo, dict):
            sweep_cap["best_throughput"] = _to_float(bo.get("output_throughput") or bo.get("tput"))
        if sweep_cap.get("attempts", 0) > 0:
            sweep_cap["status"] = "completed"

    # merged explore row carrying the unified explore_search ledger activity.
    explore = _capability_for_action(state, "explore")
    explore["last_validated_gain_pct"] = _to_float(state.get("cumulative_gain_validated"))
    explore_search = state.get("explore_search") or {}
    if isinstance(explore_search, dict):
        explore["tested"] = len(explore_search.get("tested") or {})
        accepted_entries = [v for v in (explore_search.get("accepted") or []) if isinstance(v, dict)]
        if accepted_entries:
            explore["best_gain_pct"] = max(
                (_to_float(v.get("gain_pct")) or 0.0 for v in accepted_entries),
                default=None,
            )
        keep_unstable_count = sum(
            1
            for entry in (explore_search.get("rejected") or [])
            if isinstance(entry, dict) and entry.get("reason") == "stack_unstable"
        )
        if keep_unstable_count:
            explore["keep_unstable_count"] = keep_unstable_count
        explore["winners_history"] = len(explore_search.get("winners_history") or [])

    # specialist row derived from ``specialist_rounds`` (single source, agrees with specialist_runs).
    specialist_row = _specialist_capability_row(state)
    return {
        "geak": geak_cap,
        "forge": forge_cap,
        # primary post-merge row; backends/params/validate_stack are compat rows.
        "explore": explore,
        "backends": backends,
        "params": params,
        "sweep": sweep_cap,
        "validate_stack": validate,
        "specialist": specialist_row,
    }


def _specialist_capability_row(state: dict[str, Any]) -> dict[str, Any]:
    """Derive ``capability_summary.specialist`` from ``specialist_rounds`` (single source per Inv-12.2).

    Headline counts aggregate all domains; ``by_specialist`` breaks them
    out per SpecialistDomain.key.

    Args:
        state (dict[str, Any]): Parsed ``state.json``.

    Returns:
        dict[str, Any]: The specialist capability row (aggregate status /
        attempts / keeps / tested plus a ``by_specialist`` per-domain map).
    """
    rounds = state.get("specialist_rounds") or []
    if not isinstance(rounds, list) or not rounds:
        return {
            "status": "not_attempted",
            "attempts": 0,
            "keeps": 0,
            "tested": 0,
            "by_specialist": _empty_by_specialist_capability(),
        }

    attempts = 0
    proposals_total = 0
    proposals_kept = 0
    # Per-domain counters seeded with the catalogue for presence-free iteration (unknown domains survive).
    by_specialist_raw: dict[str, dict[str, int]] = {
        d: {"attempts": 0, "keeps": 0, "tested": 0, "rejected": 0} for d in _SPECIALIST_DOMAIN_KEYS
    }

    for r in rounds:
        if not isinstance(r, dict):
            continue
        attempts += 1
        proposals_total += int(r.get("proposals_total") or 0)
        proposals_kept += int(r.get("proposals_kept") or 0)

        # Per-domain tallies: trust ``domain_breakdown`` when present, else
        # split the round totals evenly across ``domains[]``.
        round_breakdown = r.get("domain_breakdown")
        if isinstance(round_breakdown, dict) and round_breakdown:
            for dom, payload in round_breakdown.items():
                if not isinstance(payload, dict):
                    continue
                bucket = by_specialist_raw.setdefault(
                    str(dom),
                    {
                        "attempts": 0,
                        "keeps": 0,
                        "tested": 0,
                        "rejected": 0,
                    },
                )
                bucket["attempts"] += int(payload.get("dispatched") or 0)
                bucket["keeps"] += int(payload.get("proposals_kept") or 0)
                bucket["tested"] += int(payload.get("proposals_total") or 0)
                bucket["rejected"] += int(payload.get("proposals_rejected") or 0)
        else:
            # Legacy round (no ``domain_breakdown``): impute equal share across tags/domains.
            domains = r.get("tags") or r.get("domains") or []
            if isinstance(domains, list) and domains:
                share_total = int(r.get("proposals_total") or 0) // len(domains)
                share_kept = int(r.get("proposals_kept") or 0) // len(domains)
                for dom in domains:
                    bucket = by_specialist_raw.setdefault(
                        str(dom),
                        {
                            "attempts": 0,
                            "keeps": 0,
                            "tested": 0,
                            "rejected": 0,
                        },
                    )
                    bucket["attempts"] += 1
                    bucket["tested"] += share_total
                    bucket["keeps"] += share_kept

    if attempts == 0:
        status = "not_attempted"
    elif proposals_kept > 0:
        status = "kept"
    elif proposals_total > 0:
        status = "tried"
    else:
        status = "attempted"

    by_specialist: dict[str, dict[str, Any]] = {}
    for dom, raw in by_specialist_raw.items():
        if raw["attempts"] == 0:
            dom_status = "not_attempted"
        elif raw["keeps"] > 0:
            dom_status = "kept"
        elif raw["tested"] > 0:
            dom_status = "tried"
        else:
            dom_status = "attempted"
        by_specialist[dom] = {
            "status": dom_status,
            "attempts": raw["attempts"],
            "keeps": raw["keeps"],
            "tested": raw["tested"],
        }

    return {
        "status": status,
        "attempts": attempts,
        "keeps": proposals_kept,
        "tested": proposals_total,
        "by_specialist": by_specialist,
    }


def _empty_by_specialist_capability() -> dict[str, dict[str, Any]]:
    """Seed every catalogue domain with a not_attempted CapabilityEntry (stable shape, no KeyError-guarding).

    Returns:
        dict[str, dict[str, Any]]: One zeroed, ``not_attempted`` capability
        entry per catalogue specialist-domain key.
    """
    return {d: {"status": "not_attempted", "attempts": 0, "keeps": 0, "tested": 0} for d in _SPECIALIST_DOMAIN_KEYS}


# Attribution
# Catalogue of the 7 SpecialistDomain.key strings, inlined (not imported)
# to keep breakdown free of orchestrator deps for offline use.
_SPECIALIST_DOMAIN_KEYS: tuple[str, ...] = (
    "serving_specialist",
    "kernel_switch_specialist",
    "comm_specialist",
    "compiler_specialist",
    "system_specialist",
    "pr_intel_specialist",
    "research_scout_specialist",
)


# Phase segments — phase state machine
def collect_phase_segments(
    state: dict[str, Any],
    phase_timeline: list[dict[str, Any]],
    warnings: list[str],
) -> list[dict[str, Any]]:
    """Group action events into phase segments using ``phase_history`` boundaries.

    Only rows with a non-empty ``to_phase`` are transitions (segment
    boundaries); other rows are folded in as sub-events. Each segment's
    exit comes from the next transition. Actions are attributed by their
    own ``phase`` when present, else by the ``[entered_ts, exit_ts)``
    window. Empty when ``phase_history`` is missing (readers fall back to
    the flat ``phase_timeline``).

    Args:
        state (dict[str, Any]): Parsed ``state.json``.
        phase_timeline (list[dict[str, Any]]): The flat action timeline whose
            events are attributed into segments.
        warnings (list[str]): Shared warnings list (mutated in place when a
            legacy plateau proxy fired).

    Returns:
        list[dict[str, Any]]: One segment per phase transition (with folded
        sub-events and attributed actions). Empty when ``phase_history`` is
        missing.
    """
    history = state.get("phase_history") or []
    if not isinstance(history, list) or not history:
        return []

    rows = [r for r in history if isinstance(r, dict)]
    transitions = [r for r in rows if str(r.get("to_phase") or "")]
    sub_events = [r for r in rows if not str(r.get("to_phase") or "")]

    def _unix(row: dict[str, Any]) -> float | None:
        """Return a row's timestamp as a Unix epoch float.

        Args:
            row: Event row carrying ``ts_unix`` and/or ``ts``.

        Returns:
            The ``ts_unix`` value when numeric, else the parsed ISO ``ts``,
            else ``None``.
        """
        u = row.get("ts_unix")
        if isinstance(u, (int, float)):
            return float(u)
        return _parse_iso_unix(row.get("ts"))

    segments: list[dict[str, Any]] = []
    proxy_seen = False
    for idx, row in enumerate(transitions):
        entered_ts = iso_z(row.get("ts"))
        entered_unix = _unix(row)
        exit_ts = ""
        exit_unix: float | None = None
        exit_reason = ""
        if idx + 1 < len(transitions):
            nxt = transitions[idx + 1]
            exit_ts = iso_z(nxt.get("ts"))
            exit_reason = str(nxt.get("reason") or "")
            exit_unix = _unix(nxt)
        elapsed: float | None = None
        if entered_unix is not None and exit_unix is not None:
            elapsed = max(0.0, float(exit_unix) - float(entered_unix))
        evidence_dict = dict(row.get("evidence") or {})
        segments.append(
            {
                "phase": str(row.get("to_phase") or ""),
                "from_phase": str(row.get("from_phase") or ""),
                "entered_ts": entered_ts,
                "entered_unix": entered_unix,
                "exit_ts": exit_ts,
                "exit_unix": exit_unix,
                "exit_reason": exit_reason,
                "evidence": evidence_dict,
                "events": [],
                "actions": [],
                "elapsed_seconds": elapsed,
            }
        )

    def _owner_by_window(ts: str) -> dict[str, Any] | None:
        """Return the segment whose ``[entered_ts, exit_ts)`` ISO window holds ``ts`` (lexicographic compare).

        Args:
            ts (str): An ISO-8601 timestamp.

        Returns:
            dict[str, Any] | None: The owning segment, the last segment when
            ``ts`` is empty / past the end, or ``None`` when there are no
            segments.
        """
        if not ts:
            return segments[-1] if segments else None
        for s in segments:
            lo_ts, hi_ts = s["entered_ts"], s["exit_ts"]
            if lo_ts and ts < lo_ts:
                continue
            if hi_ts and ts >= hi_ts:
                continue
            return s
        return segments[-1] if segments else None

    # Fold non-transition rows (sub-events) into their containing segment.
    for ev in sub_events:
        ev_evidence = dict(ev.get("evidence") or {})
        if ev_evidence.get("r09_provisional") or (str(ev_evidence.get("evidence") or "") == "m2_proxy"):
            proxy_seen = True
        ev_ts = iso_z(ev.get("ts"))
        s = _owner_by_window(ev_ts)
        if s is not None:
            s["events"].append(
                {
                    "event": str(ev.get("event") or ev.get("reason") or ""),
                    "reason": str(ev.get("reason") or ""),
                    "ts": ev_ts,
                    "evidence": ev_evidence,
                }
            )

    # Attribute timeline actions by declared ``phase``, else the ts window.
    phase_to_segs: dict[str, list[dict[str, Any]]] = {}
    for s in segments:
        phase_to_segs.setdefault(s["phase"], []).append(s)
    for ev in phase_timeline or []:
        if not isinstance(ev, dict):
            continue
        ts = str(ev.get("ts") or "")
        if not ts:
            continue
        target = None
        ev_phase = str(ev.get("phase") or "")
        if ev_phase and ev_phase in phase_to_segs:
            cands = phase_to_segs[ev_phase]
            if len(cands) == 1:
                target = cands[0]
            else:
                for s in cands:
                    lo_ts, hi_ts = s["entered_ts"], s["exit_ts"]
                    if lo_ts and ts < lo_ts:
                        continue
                    if hi_ts and ts >= hi_ts:
                        continue
                    target = s
                    break
                target = target or cands[0]
        if target is None:
            target = _owner_by_window(ts)
        if target is not None:
            target["actions"].append(ev)

    if proxy_seen:
        # Session-level marker for legacy-proxy exits.
        warnings.append(
            "plateau_proxy_provisional: legacy params_no_promote_streak "
            "proxy fired (R-09); set INFERENCE_OPTIMIZER_DISABLE_PLATEAU_PROXY=1 "
            "once the fleet is fully v0.8 to fail closed"
        )
    return segments

