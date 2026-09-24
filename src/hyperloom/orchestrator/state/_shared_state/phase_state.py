# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``_PhaseStateMixin`` — the phase ledgers :class:`..shared_state.SharedState` mutates: explore search, attempts, specialist rounds and verdicts, and per-domain counters."""

from __future__ import annotations

import logging
from typing import Any

from hyperloom.common.timeutil import now_iso

log = logging.getLogger(__name__)

# Long-run bounded-growth caps for append-only telemetry ledgers (tail-trim).
_INTERVENTION_MIX_CAP = 500
_SPECIALIST_ROUNDS_CAP = 200
_SEEN_PR_IDS_CAP = 2000
_WINNERS_HISTORY_CAP = 200
# Negative ledger (explore_search["tested"]); oldest insertion-order keys evicted first.
_EXPLORE_TESTED_CAP = 5000


def _cap_tested_ledger(tested: dict[str, Any]) -> dict[str, Any]:
    """Bound the explore_search negative ledger for multi-day runs."""
    if not isinstance(tested, dict) or len(tested) <= _EXPLORE_TESTED_CAP:
        return tested if isinstance(tested, dict) else {}
    keys = list(tested.keys())[-_EXPLORE_TESTED_CAP:]
    return {k: tested[k] for k in keys}


def _stamp_cycle_on_tested(
    tested: dict[str, Any],
    cycle: int,
    bottleneck: str = "",
) -> dict[str, Any]:
    """Bucket negative-ledger entries by macro-cycle + bottleneck (R3)."""
    if not isinstance(tested, dict):
        return {}
    bn = (bottleneck or "").strip()
    for v in tested.values():
        if isinstance(v, dict):
            if "cycle" not in v:
                v["cycle"] = int(cycle)
            if bn and "bottleneck" not in v:
                v["bottleneck"] = bn
    return tested


def _stamp_cycle_on_rejected(
    rejected: list[Any],
    cycle: int,
    bottleneck: str = "",
) -> list[Any]:
    """Bucket rejected entries by macro-cycle + bottleneck (R3)."""
    if not isinstance(rejected, list):
        return []
    bn = (bottleneck or "").strip()
    for v in rejected:
        if isinstance(v, dict):
            if "cycle" not in v:
                v["cycle"] = int(cycle)
            if bn and "bottleneck" not in v:
                v["bottleneck"] = bn
    return rejected


def _merge_rejected(prior: Any, update: Any) -> list[dict[str, Any]]:
    """Merge rejected rows by fingerprint, newest wins. Rows without one are dropped: the fingerprint is what the dedup gate matches on."""
    merged: dict[str, dict[str, Any]] = {}
    for entry in [*(prior or []), *(update or [])]:
        if not isinstance(entry, dict):
            continue
        fingerprint = str(entry.get("fingerprint") or "")
        if fingerprint:
            merged[fingerprint] = entry
    return list(merged.values())


# Gap ``severity`` by urgency; an unknown severity ranks below ``low``.
GAP_SEVERITY_RANK: dict[str, int] = {"high": 3, "medium": 2, "low": 1}


def gap_actionability_key(gap: dict[str, Any]) -> tuple[int, int, str]:
    """Sort key putting the highest-severity, then least-attempted, then oldest gap first."""
    severity = GAP_SEVERITY_RANK.get(str(gap.get("severity") or "").lower(), 0)
    return (-severity, len(gap.get("attempts") or []), str(gap.get("first_seen_ts") or ""))


class _PhaseStateMixin:
    def record_specialist_round(self, entry: dict[str, Any]) -> None:
        """Append one round summary to ``specialist_rounds``; idempotent on ``round_id`` (re-record overwrites)."""
        if not isinstance(entry, dict) or not entry:
            return
        entry = dict(entry)
        entry.setdefault("cycle", int(getattr(self, "macro_cycle", 0) or 0))
        round_id = str(entry.get("round_id") or "").strip()
        if not round_id:
            self.specialist_rounds.append(entry)
        else:
            existing = self.specialist_rounds
            matched = False
            for i, prev in enumerate(existing):
                if isinstance(prev, dict) and str(prev.get("round_id") or "") == round_id:
                    existing[i] = entry
                    matched = True
                    break
            if not matched:
                existing.append(entry)
        self._trim_specialist_rounds()

    def _trim_specialist_rounds(self) -> None:
        """Bound the specialist-round ledger for multi-day runs (keep most recent)."""
        cap = _SPECIALIST_ROUNDS_CAP
        if len(self.specialist_rounds) > cap:
            self.specialist_rounds = self.specialist_rounds[-cap:]

    def record_attempt(self, attempt: dict[str, Any]) -> None:
        """Append one measured attempt; stamps the cycle the dryness judgment filters on."""
        row = dict(attempt)
        row.setdefault("cycle", int(self.macro_cycle or 0))
        self.attempts.append(row)
        cap = _SPECIALIST_ROUNDS_CAP
        if len(self.attempts) > cap:
            self.attempts = self.attempts[-cap:]

    def bump_domain_round_counters(self) -> None:
        """Increment both per-anchor round counters for every knowledge-domain anchor."""
        from ...specialists.domains import KNOWLEDGE_DOMAIN_TAGS

        for anchor in KNOWLEDGE_DOMAIN_TAGS:
            self.rounds_since_last_specialist[anchor] = int(self.rounds_since_last_specialist.get(anchor, 0) or 0) + 1
            self.rounds_since_last_keep[anchor] = int(self.rounds_since_last_keep.get(anchor, 0) or 0) + 1

    @staticmethod
    def _anchor_for(domain_or_anchor: str) -> str:
        """Resolve a domain key or raw tag to its canonical kb_anchor."""
        from ...specialists.domains import domain_for_tag, get_domain

        s = str(domain_or_anchor or "").strip()
        if not s:
            return ""
        d = get_domain(s)
        if d and d.kb_anchor:
            return d.kb_anchor
        dt = domain_for_tag(s)
        if dt and dt.kb_anchor:
            return dt.kb_anchor
        return s

    def note_specialist_dispatched(self, domain_or_anchor: str) -> None:
        """Reset ``rounds_since_last_specialist`` for the dispatched anchor."""
        anchor = self._anchor_for(domain_or_anchor)
        if anchor:
            self.rounds_since_last_specialist[anchor] = 0

    def note_domain_keep(self, domain_or_anchor: str) -> None:
        """Reset ``rounds_since_last_keep`` for the anchor that just KEPT."""
        anchor = self._anchor_for(domain_or_anchor)
        if anchor:
            self.rounds_since_last_keep[anchor] = 0

    def stalled_domains(
        self,
        *,
        specialist_threshold: int,
        keep_threshold: int,
    ) -> list[str]:
        """Return anchors whose ``rounds_since_last_specialist`` ≥ ``specialist_threshold`` OR ``rounds_since_last_keep`` ≥ ``keep_threshold``."""
        anchors = set(self.rounds_since_last_specialist) | set(self.rounds_since_last_keep)
        hits: list[tuple[int, str]] = []
        for anchor in anchors:
            spec = int(self.rounds_since_last_specialist.get(anchor, 0) or 0)
            keep = int(self.rounds_since_last_keep.get(anchor, 0) or 0)
            if spec >= specialist_threshold or keep >= keep_threshold:
                hits.append((max(spec, keep), anchor))
        hits.sort(key=lambda x: (-x[0], x[1]))
        return [anchor for _, anchor in hits]

    def best_gap_for_anchor(self, anchor: str) -> str:
        """Return the canonical_id of the most actionable open gap whose ``domain_hint`` resolves to ``anchor`` (or ``""`` when none)."""
        target = self._anchor_for(anchor)
        if not target:
            return ""
        matches: list[tuple[tuple[int, int, str], str]] = []
        for g in self.gaps:
            if not isinstance(g, dict):
                continue
            cid = str(g.get("canonical_id") or "").strip()
            if not cid:
                continue
            if self._anchor_for(str(g.get("domain_hint") or "")) != target:
                continue
            matches.append((gap_actionability_key(g), cid))
        if not matches:
            return ""
        matches.sort(key=lambda m: m[0])
        return matches[0][1]

    def record_intervention(
        self,
        *,
        change_type: str,
        action: str,
        task_id: str = "",
        delta_pct: float | None = None,
    ) -> None:
        """Append one intervention entry and update config-only counters; the consecutive-config counter advances on ``\"config\"`` and resets on ``\"code_patch\"``."""
        ct = str(change_type or "").strip().lower()
        entry = {
            "change_type": ct,
            "action": str(action or ""),
            "task_id": str(task_id or ""),
            "delta_pct": delta_pct,
            "ts": now_iso(),
        }
        self.intervention_mix.append(entry)
        cap = _INTERVENTION_MIX_CAP
        if len(self.intervention_mix) > cap:
            self.intervention_mix = self.intervention_mix[-cap:]
        if ct == "config":
            self.consecutive_config_only_rounds = int(self.consecutive_config_only_rounds or 0) + 1
        elif ct == "code_patch":
            self.consecutive_config_only_rounds = 0

    def bump_research_scout_runs(self, n: int = 1) -> int:
        """Increment the research-scout dispatch counter; return new total."""
        self.research_scout_runs = int(self.research_scout_runs or 0) + int(n)
        return self.research_scout_runs

    def register_seen_pr_ids(self, pr_ids: Any) -> int:
        """Add PR ids to the shared seen-set (scout + FRAMEWORK dedup); returns count newly added."""
        seen = set(self.research_scout_seen_pr_ids or [])
        added = 0
        for raw in pr_ids or []:
            pid = str(raw or "").strip()
            if not pid or pid in seen:
                continue
            seen.add(pid)
            self.research_scout_seen_pr_ids.append(pid)
            added += 1
        cap = _SEEN_PR_IDS_CAP
        if len(self.research_scout_seen_pr_ids) > cap:
            # FIFO eviction of oldest-seen ids.
            self.research_scout_seen_pr_ids = self.research_scout_seen_pr_ids[-cap:]
        return added

    def reset_explore_plateau_proxy(self) -> None:
        """Reset the explore plateau proxy counter."""
        self.params_no_promote_streak = 0

    def reset_per_cycle_plateau_state(self) -> None:
        """Reset transient plateau and dispatch state for a macro-cycle, plus any stale escalate hint."""
        self.params_no_promote_streak = 0
        self.framework_agent_phase_done = False
        self.framework_agent_discover_failures = 0
        self.framework_agent_empty_discoveries = 0
        self.specialist_domain_empty_streak = {}
        self.rounds_since_last_specialist = {}
        self.rounds_since_last_keep = {}
        self.last_conc_sweep = {}
        self.discard_pending_escalate_hint()

    def note_explore_outcome(self, *, promoted: bool) -> None:
        """Update the plateau proxy after one explore task (KEEP resets, no-promote increments)."""
        if promoted:
            self.reset_explore_plateau_proxy()
        else:
            self.params_no_promote_streak += 1

    def record_specialist_patch_verdict(
        self,
        subject: str,
        verdict: str,
    ) -> None:
        """Record the Critic verdict for a patch's review subject; idempotent (later verdict overwrites), empty ``verdict`` clears the entry to force re-review."""
        sid = str(subject or "").strip()
        if not sid:
            return
        v = str(verdict or "").strip().lower()
        if not v:
            self.specialist_patch_verdicts.pop(sid, None)
            return
        self.specialist_patch_verdicts[sid] = v

    def get_specialist_patch_verdict(
        self,
        subject: str,
    ) -> str:
        """Return the patch verdict, or empty when no Critic decision exists."""
        sid = str(subject or "").strip()
        if not sid:
            return ""
        return self.specialist_patch_verdicts.get(sid, "") or ""

    def update_last_specialist(self, snapshot: dict[str, Any]) -> None:
        """Snapshot the most recent specialist task (parity with last_*)."""
        if isinstance(snapshot, dict):
            self.last_specialist = dict(snapshot)

    def apply_explore_search_update(self, update: dict[str, Any]) -> None:
        """Merge an ExploreExecutor search update into persistent state; :meth:`record_explore_accepted` is the single writer for the ``accepted`` bucket."""
        if not isinstance(update, dict):
            return
        prior = self.explore_search if isinstance(self.explore_search, dict) else {}
        merged = dict(prior)
        merged["schema_version"] = int(update.get("schema_version") or 1)
        cur_cycle = int(getattr(self, "macro_cycle", 0) or 0)
        cur_bottleneck = self.current_top_bottleneck()
        # The executor reports the round it just benched; accumulating it over the
        # durable ledger is this layer's job, and a re-measured fingerprint replaces
        # its earlier row because that is what a fresh measurement means.
        merged["tested"] = _cap_tested_ledger(
            _stamp_cycle_on_tested(
                {**(prior.get("tested") or {}), **(update.get("tested") or {})},
                cur_cycle,
                cur_bottleneck,
            )
        )
        merged["rejected"] = _stamp_cycle_on_rejected(
            _merge_rejected(prior.get("rejected"), update.get("rejected")),
            cur_cycle,
            cur_bottleneck,
        )
        merged["name_index"] = {**(prior.get("name_index") or {}), **(update.get("name_index") or {})}
        # The round ordinal, not the ledger's size: consumers key idempotency and
        # round ids on it, so it has to advance once per benched round.
        merged["cursor"] = int(prior.get("cursor") or 0) + 1
        merged["last_round"] = dict(update.get("last_round") or {})
        # Append-only history fields — merge instead of overwrite.
        wh = list(prior.get("winners_history") or [])
        known = {
            (
                str(row.get("round_id") or ""),
                str(row.get("fingerprint") or row.get("variant_name") or ""),
                str(row.get("ts") or ""),
            )
            for row in wh
            if isinstance(row, dict)
        }
        for entry in update.get("winners_history") or []:
            if not isinstance(entry, dict):
                continue
            row = dict(entry)
            key = (
                str(row.get("round_id") or ""),
                str(row.get("fingerprint") or row.get("variant_name") or ""),
                str(row.get("ts") or ""),
            )
            if key in known:
                continue
            row.setdefault("cycle", cur_cycle)
            known.add(key)
            wh.append(row)
        merged["winners_history"] = wh[-_WINNERS_HISTORY_CAP:]
        merged["domains_round_summary"] = list(
            update.get("domains_round_summary") or prior.get("domains_round_summary") or []
        )
        # Preserve accepted bucket from prior runs (record_explore_accepted is its writer).
        merged["accepted"] = list(prior.get("accepted") or [])
        # Drop so a later load re-runs the legacy union.
        merged.pop("merged_from_legacy_sig", None)
        self.explore_search = merged

    def record_explore_accepted(self, variant: dict[str, Any]) -> None:
        """Append one promoted variant to ``explore_search.accepted``; dedupes by ``fingerprint`` and removes any matching ``rejected`` entry."""
        if not isinstance(variant, dict) or not variant:
            return
        from hyperloom.inference_optimizer.canonical_fingerprint import canonical_fingerprint

        args = str(variant.get("candidate_extra_server_args") or variant.get("extra_server_args") or "")
        envs = dict(variant.get("extra_envs") or {})

        def _list_field(key: str) -> list[str]:
            raw = variant.get(key)
            if isinstance(raw, str):
                return [raw.strip()] if raw.strip() else []
            if isinstance(raw, (list, tuple, set)):
                return [str(v).strip() for v in raw if str(v).strip()]
            return []

        remove_args = _list_field("remove_args")
        unset_envs = _list_field("unset_envs")
        args_mode = str(variant.get("args_mode") or "append").strip().lower()
        control_fields: dict[str, Any] = {}
        if remove_args:
            control_fields["remove_args"] = remove_args
        if unset_envs:
            control_fields["unset_envs"] = unset_envs
        if args_mode == "replace":
            control_fields["args_mode"] = "replace"
        fp = str(
            variant.get("fingerprint")
            or canonical_fingerprint(
                args,
                envs,
                **control_fields,
            )
        )
        entry = {
            "fingerprint": fp,
            "name": str(variant.get("name") or ""),
            "extra_server_args": args,
            "extra_envs": envs,
            **control_fields,
            "note": str(variant.get("note") or ""),
            "tput": variant.get("output_throughput") or variant.get("tput"),
            "gain_pct": variant.get("gain_pct"),
            # Carried through so the ledger records what the KEEP was judged on; ``None`` means the variant was never
            # gated, not that it scored 0.
            "accuracy": variant.get("accuracy"),
            "stack_index": variant.get("stack_index"),
            "accepted_at_round": str(variant.get("accepted_at_round") or ""),
            "ts": str(variant.get("ts") or now_iso()),
            "provenance": str(variant.get("provenance") or "llm_direct"),
            # Attribute the win to the macro-cycle it landed in.
            "cycle": int(getattr(self, "macro_cycle", 0) or 0),
        }
        search = dict(self.explore_search or {})
        search.setdefault("schema_version", 1)
        accepted = [
            v for v in (search.get("accepted") or []) if not (isinstance(v, dict) and v.get("fingerprint") == fp)
        ]
        accepted.append(entry)
        search["accepted"] = accepted
        search["rejected"] = [
            v for v in (search.get("rejected") or []) if not (isinstance(v, dict) and v.get("fingerprint") == fp)
        ]
        name_index = dict(search.get("name_index") or {})
        if entry["name"]:
            name_index[entry["name"]] = fp
        search["name_index"] = name_index
        # Append a winners_history row so plateau judges needn't crawl optimization_stack.
        wh = list(search.get("winners_history") or [])
        wh.append(
            {
                "round_id": entry["accepted_at_round"],
                "variant_name": entry["name"],
                "fingerprint": fp,
                "gain_pct": entry["gain_pct"],
                "extra_args": args,
                "extra_envs": envs,
                **control_fields,
                "provenance": entry["provenance"],
                "ts": entry["ts"],
                "cycle": entry["cycle"],
            }
        )
        search["winners_history"] = wh[-_WINNERS_HISTORY_CAP:]
        self.explore_search = search

    def record_authored_framework_levers(
        self,
        switches: list[dict[str, Any]],
        *,
        default_on: bool,
        specialist_task_id: str = "",
        stack_delta_pct: float | None = None,
    ) -> bool:
        """Register accepted framework-rewrite switches as search levers."""
        if not isinstance(switches, list) or not switches:
            return False
        rows = list(getattr(self, "authored_framework_levers", None) or [])
        by_switch = {str(r.get("switch") or ""): i for i, r in enumerate(rows) if isinstance(r, dict)}
        now = now_iso()
        changed = False
        for entry in switches:
            if not isinstance(entry, dict):
                continue
            name = str(entry.get("switch") or "").strip().upper()
            if not name:
                continue
            row = {
                "switch": name,
                "value": str(entry.get("value") or "1"),
                "category": str(entry.get("category") or ""),
                "target": str(entry.get("target") or ""),
                "evidence": str(entry.get("evidence") or ""),
                "depends_on": list(entry.get("depends_on") or []),
                "enables": list(entry.get("enables") or []),
                "enabler": bool(entry.get("enabler")),
                "default_on": bool(default_on),
                "specialist_task_id": str(specialist_task_id or ""),
                "stack_delta_pct": (float(stack_delta_pct) if isinstance(stack_delta_pct, (int, float)) else None),
                # Filled in by the explore phase once the lever has been measured on its own; ``None`` means
                # "registered, not yet attributed".
                "attributed_gain_pct": None,
                "attribution_source": "",
                "ts": now,
                "cycle": int(getattr(self, "macro_cycle", 0) or 0),
            }
            index = by_switch.get(name)
            if index is None:
                rows.append(row)
                by_switch[name] = len(rows) - 1
                changed = True
                continue
            prior = rows[index] if isinstance(rows[index], dict) else {}
            # Preserve an attribution already measured for this lever; the measurement is more informative than this
            # registration.
            row["attributed_gain_pct"] = prior.get("attributed_gain_pct")
            row["attribution_source"] = str(prior.get("attribution_source") or "")
            if prior != row:
                rows[index] = row
                changed = True
        if changed:
            self.authored_framework_levers = rows
        return changed

    def record_framework_lever_attribution(
        self,
        switch: str,
        *,
        gain_pct: float | None,
        source: str,
    ) -> bool:
        """Record a lever's individually measured contribution."""
        name = str(switch or "").strip().upper()
        if not name:
            return False
        rows = list(getattr(self, "authored_framework_levers", None) or [])
        for index, row in enumerate(rows):
            if not isinstance(row, dict) or str(row.get("switch") or "") != name:
                continue
            updated = dict(row)
            updated["attributed_gain_pct"] = float(gain_pct) if isinstance(gain_pct, (int, float)) else None
            updated["attribution_source"] = str(source or "")
            if updated != row:
                rows[index] = updated
                self.authored_framework_levers = rows
                return True
            return False
        return False
