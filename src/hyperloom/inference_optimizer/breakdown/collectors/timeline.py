# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

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
from hyperloom.orchestrator.phases.machine_state import (
    is_phase_transition_row as _is_phase_transition_row,
    phase_history_event_name,
)
from hyperloom.orchestrator.state.optimization_journal import (
    operation_kind_for,
    proposer_for,
)

from ._common import (
    _load_optimization_journal,
    _parse_iso_unix,
    _to_float,
)


# Action labels whose ``<action>_attempts`` lists feed the timeline + capability tallies.
_AUDIT_ACTIONS = (
    "baseline",
    "profile",
    "explore",
    "backends",
    "params",
    "validate_stack",
    "roofline",
)


class TimelineDedup:
    """Decide which timeline rows describe an event already seen.

    Rows collide on ``(action, ts-to-second, change)``. Rows that also carry
    distinct task ids are distinct events and are all kept -- but only while
    every row seen for that triple was itself task-tagged. An untagged row is
    the same event observed from another source (journal vs audit list vs
    recorder fragment), so it keeps the legacy fold rather than duplicating.

    The collector and the exporter merge different mixes of those sources.
    Sharing this decision is what stops them disagreeing about what one event
    is: an exporter with a weaker identity silently drops rows the collector
    would have kept, and nothing downstream can tell that happened.
    """

    def __init__(self) -> None:
        """Start with no events seen."""
        self._seen: dict[tuple[str, str, str], set[str]] = {}

    def is_new(self, ev: dict[str, Any]) -> bool:
        """Record ``ev`` and report whether it is an event not seen before.

        Args:
            ev (dict[str, Any]): One timeline event row.

        Returns:
            bool: ``True`` when the row should be kept.
        """
        base = (
            str(ev.get("action") or ""),
            iso_z(ev.get("ts"))[:19],
            str(ev.get("change") or ev.get("task_id") or ""),
        )
        task_id = str(ev.get("task_id") or "")
        prior = self._seen.get(base)
        if prior is None:
            self._seen[base] = {task_id}
            return True
        if task_id and task_id not in prior and all(prior):
            prior.add(task_id)
            return True
        return False


def _journal_entry_to_event(e: dict[str, Any]) -> dict[str, Any]:
    """Map one optimization_journal entry to a phase_timeline event.

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
            # Proposer attribution + stable filter label.
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

    * ``reports/optimization_journal.json`` — the canonical decision log.
      Carries ``phase`` for exact segment attribution.
    * the per-action ``*_attempts`` audit lists + ``kernel_opt`` /
      ``kernel_integrate`` histories — add per-attempt rows (incl.
      failures) and the kernel lanes.

    Events are de-duplicated by ``(action, ts-to-second, change)`` with
    the journal copy winning, then sorted by ``ts``. Passing
    ``session_dir=None`` degrades gracefully to the audit-list scrape.

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

    # Source 1: canonical journal (carries phase).
    for e in _load_optimization_journal(session_dir, warnings):
        if isinstance(e, dict):
            events.append(_journal_entry_to_event(e))

    # Source 2: per-action audit lists.
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
    kernel_opt = state.get("kernel_opt_task_attempts") or state.get("kernel_opt_attempts") or {}
    if isinstance(kernel_opt, dict):
        for ledger_id, ent in kernel_opt.items():
            if not isinstance(ent, dict):
                continue
            kid = str(ent.get("current_kernel_id") or ent.get("kernel_id") or ledger_id)
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
    dedup = TimelineDedup()
    deduped = [ev for ev in events if dedup.is_new(ev)]

    deduped.sort(key=lambda e: e.get("ts") or "")
    return deduped


# Capability summary
def _capability_for_action(
    state: dict[str, Any],
    action: str,
) -> dict[str, Any]:
    """Per-action capability tally from the real ``<action>_attempts`` ledger.

    Status is derived strictly from recorded attempt evidence (written forward
    by ``SharedState.record_action_attempt``). We deliberately do NOT reverse-
    infer ``kept`` from ``optimization_stack``: the stack is the final adopted
    state and can carry seeded / warm-replayed / cross-harness entries that were
    never a real this-session attempt, so counting them would fabricate KEEPs
    and attempts. No attempt record => ``not_attempted``.

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

    status = "kept" if n_keeps > 0 else "tried" if n_attempts > 0 else "not_attempted"
    return {
        "status": status,
        "attempts": n_attempts,
        "keeps": n_keeps,
    }


def _fold_search_ledger_keeps(row: dict[str, Any], search: dict[str, Any]) -> None:
    """Promote a capability row to ``kept`` from its real ``*_search`` ledger.

    ``<phase>_search.accepted`` is the forward-recorded ledger of variants this
    session actually accepted (KEPT). Unlike ``optimization_stack`` (which can
    carry seeded / warm-replayed entries that were never a real this-session
    KEEP), the search ledger is genuine evidence, so it may set the status to
    ``kept``. No accepted entries => the row's attempt-derived status stands.

    Args:
        row (dict[str, Any]): The capability row to update in place.
        search (dict[str, Any]): The ``<phase>_search`` ledger from state.
    """
    accepted = [v for v in (search.get("accepted") or []) if isinstance(v, dict)]
    if not accepted:
        return
    if row.get("keeps", 0) < len(accepted):
        row["keeps"] = len(accepted)
    if row.get("attempts", 0) < row["keeps"]:
        row["attempts"] = row["keeps"]
    row["status"] = "kept"


# How decided a verdict is. One kernel can carry several integrate rows (the
# ledger is keyed ``<kernel_id>|<patch_path>|<extra_args>``), and folding them
# by kernel must not hand the outcome to whichever row is iterated last: an
# adopted patch is not undone by a reverted sibling. Anything unlisted --
# ``NEEDS_REVIEW``, or no decision recorded yet -- ranks lowest, because it is
# the absence of a verdict rather than a verdict.
_VERDICT_RANK = {"KEEP": 3, "REVERT": 2, "REJECT": 2}


def _stronger_verdict(current: str, candidate: str) -> str:
    """Return whichever of two integrate verdicts is more decided.

    Args:
        current (str): The verdict folded so far.
        candidate (str): The verdict being folded in.

    Returns:
        str: The verdict that should represent the kernel.
    """
    if _VERDICT_RANK.get(candidate, 1) > _VERDICT_RANK.get(current, 1):
        return candidate
    return current


def geak_route_evidence(state: dict[str, Any] | None, geak: dict[str, Any] | None) -> tuple[bool, bool]:
    """Answer whether GEAK's route ran, and whether it was promoted.

    One definition with two readers: the capability-summary fallback below and
    the exporter's consistency warnings. They are only meaningful while they
    agree, so they must not each carry their own copy of the predicate.

    Args:
        state: Session state mapping, read for ``optimization_stack``.
        geak: Normalized GEAK section.

    Returns:
        tuple[bool, bool]: ``(promoted, has_route_evidence)`` -- whether a
        ``geak_e2e`` entry reached the optimization stack, and whether the route
        ran at all. ``engaged`` alone can mean only that GEAK was configured, so
        ``status=missing`` (no result, no disk recovery) is not evidence.
    """
    state = state if isinstance(state, dict) else {}
    geak = geak if isinstance(geak, dict) else {}
    promoted = any(
        isinstance(entry, dict)
        and (
            str(entry.get("action") or "").lower() == "geak_e2e" or str(entry.get("source") or "").lower() == "geak_e2e"
        )
        for entry in state.get("optimization_stack") or []
    )
    has_route_evidence = promoted or (bool(geak.get("engaged")) and str(geak.get("status") or "").lower() != "missing")
    return promoted, has_route_evidence


def collect_capability_summary(
    state: dict[str, Any],
    geak_invocations: list[dict[str, Any]],
    warnings: list[str],
    forge_invocations: list[dict[str, Any]] | None = None,
    geak: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Summarize kernel-optimization capability outcomes for the breakdown.

    Args:
        state: Session state mapping.
        geak_invocations: GEAK backend invocation records.
        warnings: Mutable list that collected warnings are appended to.
        forge_invocations: Forge backend invocation records (own lane).
        geak: Normalized GEAK section. Used as a route-level fallback when
            GEAK ran outside the native kernel-agent run directory layout.

    Returns:
        A capability-summary dict (per-kernel status, attempt and keep counts).
    """
    forge_invocations = forge_invocations or []
    # Integrate (e2e) outcome per kernel: a KEEP reverted at integrate is not a real adoption.
    integ = state.get("kernel_integrate_attempts") or {}
    integ_by_kid: dict[str, dict[str, Any]] = {}
    if isinstance(integ, dict):
        for ent in integ.values():
            if not isinstance(ent, dict):
                continue
            kid = str(ent.get("kernel_id") or "")
            if not kid:
                continue
            decision = str(ent.get("last_decision") or "").upper()
            gain = _to_float(ent.get("best_gain_pct"))
            prior = integ_by_kid.get(kid)
            if prior is None:
                integ_by_kid[kid] = {"decision": decision, "e2e_gain_pct": gain}
                continue
            # Fold, never overwrite: several patches for one kernel each get
            # their own row here.
            prior["decision"] = _stronger_verdict(prior["decision"], decision)
            if gain is not None and (prior["e2e_gain_pct"] is None or gain > prior["e2e_gain_pct"]):
                prior["e2e_gain_pct"] = gain

    # Kernel backends from on-disk invocations, reconciled against the integrate verdict.
    def _from_invocations(invs: list[dict[str, Any]]) -> dict[str, Any]:
        """Reduce one lane's invocations to a capability row.

        Args:
            invs (list[dict[str, Any]]): A lane's invocation records.

        Returns:
            dict[str, Any]: ``{"status", "attempts", "keeps"}`` where ``keeps``
            counts distinct kernels an integrate verdict adopted, plus optional
            ``reverts`` / ``micro_only_keeps`` / ``e2e_gain_pct``.
        """
        attempts = len(invs)
        # Tally distinct kernels, not invocation rows: one kernel re-tried
        # across runs is still one kernel.
        adopted_kids: set[str] = set()
        reverted_kids: set[str] = set()
        pending_kids: set[str] = set()
        micro_only_kids: set[str] = set()
        best_e2e: float | None = None
        for i, v in enumerate(invs):
            if v.get("decision") != "KEEP":
                continue
            kid = str(v.get("kernel_id") or "")
            # A row without a kernel id cannot be folded with any other row.
            ident = kid or f"__row_{i}"
            outcome = integ_by_kid.get(kid) if kid else None
            if outcome is None:
                # A KEEP that never reached integrate cleared the micro
                # benchmark only. It is not an adoption (see CapabilityEntry:
                # ``keeps`` is "kernels adopted at integrate").
                micro_only_kids.add(ident)
                continue
            decision = outcome["decision"]
            if decision == "KEEP":
                adopted_kids.add(ident)
                # Only an adoption contributes to "best gain": a reverted
                # patch's number describes a regression, and an undecided
                # one describes a measurement nobody has ruled on.
                g = outcome["e2e_gain_pct"]
                if g is not None and (best_e2e is None or g > best_e2e):
                    best_e2e = g
            elif decision in ("REVERT", "REJECT"):
                reverted_kids.add(ident)
            else:
                # NEEDS_REVIEW, or no decision recorded. The verdict is not in,
                # and a gain <= 0 NEEDS_REVIEW never gets retried, so calling
                # this an adoption misreports it for the rest of the session.
                pending_kids.add(ident)
        # A decided outcome outranks an undecided one for the same kernel.
        reverted_kids -= adopted_kids
        pending_kids -= adopted_kids | reverted_kids
        micro_only_kids -= adopted_kids | reverted_kids | pending_kids
        adopted = len(adopted_kids)
        reverted = len(reverted_kids)
        pending = len(pending_kids)
        micro_only = len(micro_only_kids)
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
        if pending:
            row["pending_integrate"] = pending
        if micro_only:
            row["micro_only_keeps"] = micro_only
        if best_e2e is not None:
            row["e2e_gain_pct"] = best_e2e
        return row

    geak_cap = _from_invocations(geak_invocations)
    forge_cap = _from_invocations(forge_invocations)

    # GEAK e2e owns its own working-tree layout, so a real run does not
    # necessarily create ``kernel-agent/runs/*/optimization_attempts.jsonl``.
    # Treat the normalized GEAK result as engagement evidence instead of
    # reporting ``not_attempted``. A promoted ``geak_e2e`` stack entry is the
    # route-level KEEP; accepted-kernel count is used when available.
    geak = geak if isinstance(geak, dict) else {}
    promoted, has_route_evidence = geak_route_evidence(state, geak)
    if has_route_evidence:
        attempted_items = geak.get("kernels_attempted")
        accepted_items = geak.get("accepted_kernels")
        accepted_heads = geak.get("accepted_heads")
        attempted_count = len(attempted_items) if isinstance(attempted_items, list) else 0
        accepted_count = len(accepted_items) if isinstance(accepted_items, list) else 0
        accepted_head_count = len(accepted_heads) if isinstance(accepted_heads, list) else 0
        geak_cap["attempts"] = max(
            int(geak_cap.get("attempts") or 0),
            attempted_count,
            accepted_count,
            accepted_head_count,
            1,
        )
        # A revert is decided evidence from the native invocation rows; this
        # fallback exists for the case where those rows are MISSING, so it must
        # not overwrite a verdict they did record — and that means the keep
        # COUNT too. Bumping ``keeps`` while leaving ``status="reverted"``
        # emits a row that says the win was both kept and rolled back.
        if promoted and str(geak_cap.get("status") or "") != "reverted":
            # ONE promotion is ONE keep, whatever it carried. Counting a keep
            # per accepted kernel would contradict the canonical ledger, which
            # books exactly one adoption for the route-level win.
            geak_cap["keeps"] = max(int(geak_cap.get("keeps") or 0), 1)
            geak_cap["status"] = "kept"
        elif geak_cap.get("status") == "not_attempted":
            geak_cap["status"] = "attempted"

    # Legacy capability rows for archived sessions.
    backends = _capability_for_action(state, "backends")
    backends_search = state.get("backends_search") or {}
    if isinstance(backends_search, dict):
        backends["tested"] = len(backends_search.get("tested") or {})
        if backends_search.get("accepted"):
            backends["best_gain_pct"] = max(
                (_to_float(v.get("gain_pct")) or 0.0 for v in backends_search["accepted"] if isinstance(v, dict)),
                default=None,
            )
        _fold_search_ledger_keeps(backends, backends_search)

    params = _capability_for_action(state, "params")
    params_search = state.get("params_search") or {}
    if isinstance(params_search, dict):
        params["tested"] = len(params_search.get("tested") or {})
        if params_search.get("accepted"):
            params["best_gain_pct"] = max(
                (_to_float(v.get("gain_pct")) or 0.0 for v in params_search["accepted"] if isinstance(v, dict)),
                default=None,
            )
        _fold_search_ledger_keeps(params, params_search)

    validate = _capability_for_action(state, "validate_stack")
    validate["last_validated_gain_pct"] = _to_float(state.get("cumulative_gain_validated"))

    # Merged explore row carrying the unified explore_search ledger activity.
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
        _fold_search_ledger_keeps(explore, explore_search)
        # Only a session recorded before the confirmation round was removed
        # carries these rows; the reader stays so its report still renders.
        keep_unstable_count = sum(
            1
            for entry in (explore_search.get("rejected") or [])
            if isinstance(entry, dict) and entry.get("reason") == "stack_unstable"
        )
        if keep_unstable_count:
            explore["keep_unstable_count"] = keep_unstable_count
        explore["winners_history"] = len(explore_search.get("winners_history") or [])

    # Specialist row derived from ``specialist_rounds``.
    specialist_row = _specialist_capability_row(state)
    return {
        "geak": geak_cap,
        "forge": forge_cap,
        # Primary post-merge row; backends/params/validate_stack are compat rows.
        "explore": explore,
        "backends": backends,
        "params": params,
        "validate_stack": validate,
        "specialist": specialist_row,
    }


def _specialist_capability_row(state: dict[str, Any]) -> dict[str, Any]:
    """Derive ``capability_summary.specialist`` from ``specialist_rounds``.

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
    # Per-domain counters seeded with the catalogue.
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
            # No ``domain_breakdown``: impute equal share across tags/domains.
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
    """Seed every catalogue domain with a not_attempted CapabilityEntry.

    Returns:
        dict[str, dict[str, Any]]: One zeroed, ``not_attempted`` capability
        entry per catalogue specialist-domain key.
    """
    return {d: {"status": "not_attempted", "attempts": 0, "keeps": 0, "tested": 0} for d in _SPECIALIST_DOMAIN_KEYS}


# Mirror of the SpecialistDomain.key catalogue (orchestrator/specialists/domains.py),
# inlined to keep breakdown free of orchestrator deps for offline use. Lags the
# catalogue: static_recon / enablement / cross_framework_rewrite are not seeded.
_SPECIALIST_DOMAIN_KEYS: tuple[str, ...] = (
    "serving_specialist",
    "kernel_switch_specialist",
    "comm_specialist",
    "compiler_specialist",
    "system_specialist",
    "candidate_discovery_specialist",
    "research_scout_specialist",
)


def collect_phase_segments(
    state: dict[str, Any],
    phase_timeline: list[dict[str, Any]],
    warnings: list[str],
) -> list[dict[str, Any]]:
    """Group action events into phase segments using ``phase_history`` boundaries.

    Only rows with a real phase change (``to_phase != from_phase``) are
    transitions (segment boundaries); marker rows and legacy rows without
    ``to_phase`` are folded in as sub-events. Each segment's
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
    transitions = [r for r in rows if _is_phase_transition_row(r)]
    sub_events = [r for r in rows if not _is_phase_transition_row(r)]

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
        """Return the segment whose ``[entered_ts, exit_ts)`` ISO window holds ``ts``.

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
                    "event": phase_history_event_name(ev),
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
            "proxy fired (r09_provisional / m2_proxy evidence); treat the "
            "affected plateau exits as provisional, not measured"
        )
    return segments
