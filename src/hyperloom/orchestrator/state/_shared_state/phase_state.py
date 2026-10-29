# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``_PhaseStateMixin`` — the phase ledgers :class:`..shared_state.SharedState` mutates: explore search, attempts, specialist rounds and verdicts, and per-domain counters."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, NamedTuple

from common.coerce import to_str_list
from hyperloom.common.timeutil import now_iso

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


def _stamp_cycle(rows: Iterable[Any], cycle: int) -> None:
    """Attribute each ledger row that carries no cycle to the macro-cycle it landed in."""
    for row in rows:
        if isinstance(row, dict):
            row.setdefault("cycle", int(cycle))


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


class _WinnerKey(NamedTuple):
    """What makes a ``winners_history`` row distinct: the round, the variant, and when it landed."""

    round_id: str
    variant: str
    ts: str


def _winner_key(row: dict[str, Any]) -> _WinnerKey:
    """The dedup identity of one winners_history row."""
    return _WinnerKey(
        round_id=str(row.get("round_id") or ""),
        variant=str(row.get("fingerprint") or row.get("variant_name") or ""),
        ts=str(row.get("ts") or ""),
    )


def _merge_winners_history(prior: Any, update: Any, cycle: int) -> list[dict[str, Any]]:
    """Append a round's winners to the durable history, keeping the rows it already holds."""
    merged = list(prior or [])
    known = {_winner_key(row) for row in merged if isinstance(row, dict)}
    for entry in update or []:
        if not isinstance(entry, dict):
            continue
        row = dict(entry)
        key = _winner_key(row)
        if key in known:
            continue
        row.setdefault("cycle", cycle)
        known.add(key)
        merged.append(row)
    return merged[-_WINNERS_HISTORY_CAP:]


class _AcceptedConfig(NamedTuple):
    """The configuration a promoted variant is replayed by, and the fingerprint it is keyed on."""

    args: str
    envs: dict[str, Any]
    control_fields: dict[str, Any]
    fingerprint: str


def _control_fields(variant: dict[str, Any]) -> dict[str, Any]:
    """The arg/env removal and replacement controls a variant carries, omitting the ones left at their default."""
    fields: dict[str, Any] = {}
    remove_args = to_str_list(variant.get("remove_args"))
    if remove_args:
        fields["remove_args"] = remove_args
    unset_envs = to_str_list(variant.get("unset_envs"))
    if unset_envs:
        fields["unset_envs"] = unset_envs
    if str(variant.get("args_mode") or "append").strip().lower() == "replace":
        fields["args_mode"] = "replace"
    return fields


def _accepted_config(variant: dict[str, Any]) -> _AcceptedConfig:
    """Read the config a promoted variant replays and the fingerprint it carries."""
    return _AcceptedConfig(
        args=str(variant.get("candidate_extra_server_args") or variant.get("extra_server_args") or ""),
        envs=dict(variant.get("extra_envs") or {}),
        control_fields=_control_fields(variant),
        fingerprint=str(variant["fingerprint"]),
    )


def _accepted_entry(variant: dict[str, Any], config: _AcceptedConfig, cycle: int) -> dict[str, Any]:
    """The ``explore_search.accepted`` row recording one promoted variant."""
    return {
        "fingerprint": config.fingerprint,
        "name": str(variant.get("name") or ""),
        "extra_server_args": config.args,
        "extra_envs": config.envs,
        **config.control_fields,
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
        "cycle": cycle,
    }


def _lever_row(entry: dict[str, Any], default_on: bool) -> dict[str, Any]:
    """The ``authored_framework_levers`` row for one accepted framework-rewrite switch."""
    return {
        "switch": str(entry.get("switch") or "").strip().upper(),
        "value": str(entry.get("value") or "1"),
        "category": str(entry.get("category") or ""),
        "depends_on": list(entry.get("depends_on") or []),
        "enables": list(entry.get("enables") or []),
        "default_on": default_on,
        # Filled in by the explore phase once the lever has been measured on its own; ``None`` means
        # "registered, not yet attributed".
        "attributed_gain_pct": None,
    }


# Gap ``severity`` by urgency; an unknown severity ranks below ``low``.
GAP_SEVERITY_RANK: dict[str, int] = {"high": 3, "medium": 2, "low": 1}


def gap_actionability_key(gap: dict[str, Any]) -> tuple[int, int, str]:
    """Sort key putting the highest-severity, then least-attempted, then oldest gap first."""
    severity = GAP_SEVERITY_RANK.get(str(gap.get("severity") or "").lower(), 0)
    return (-severity, len(gap.get("attempts") or []), str(gap.get("first_seen_ts") or ""))


class _PhaseStateMixin:
    def record_specialist_round(self, entry: dict[str, Any]) -> dict[str, Any] | None:
        """Append one round summary to ``specialist_rounds``; idempotent on ``round_id`` (re-record overwrites).

        A producer that stamps no ``completed_at`` is stamped now, since the
        phase counts a round as its work by that time. Returns the stored row.
        """
        if not isinstance(entry, dict) or not entry:
            return None
        entry = dict(entry)
        entry.setdefault("cycle", int(getattr(self, "macro_cycle", 0) or 0))
        entry.setdefault("completed_at", now_iso())
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
        return entry

    def _trim_specialist_rounds(self) -> None:
        """Bound the specialist-round ledger for multi-day runs (keep most recent)."""
        cap = _SPECIALIST_ROUNDS_CAP
        if len(self.specialist_rounds) > cap:
            self.specialist_rounds = self.specialist_rounds[-cap:]

    def record_attempt(self, attempt: dict[str, Any]) -> None:
        """Append one resolved attempt; stamps the cycle the dryness judgment filters on."""
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

    def record_intervention(self, *, change_type: str) -> None:
        """Append one intervention entry (capped to the most recent rows)."""
        self.intervention_mix.append({"change_type": str(change_type or "").strip().lower()})
        if len(self.intervention_mix) > _INTERVENTION_MIX_CAP:
            self.intervention_mix = self.intervention_mix[-_INTERVENTION_MIX_CAP:]

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

    def reset_per_cycle_plateau_state(self) -> None:
        """Reset transient plateau and dispatch state for a macro-cycle."""
        self.params_no_promote_streak = 0
        self.framework_agent_phase_done = False
        self.framework_agent_discover_failures = 0
        self.framework_agent_empty_discoveries = 0
        self.rounds_since_last_specialist = {}
        self.rounds_since_last_keep = {}
        self.last_conc_sweep = {}
        self.conc_sweep_granted_until_unix = 0.0

    def open_macro_cycle(self, *, no_gain_cycle_streak: int) -> None:
        """Finish the current macro-cycle's strategy row, start the next cycle, and reset its per-cycle state."""
        finished = self.macro_cycle
        gain_delta = round(self.cumulative_gain_validated - self.gain_at_cycle_start, 6)
        rows = [r for r in self.cycle_strategy_log if isinstance(r, dict)]
        for row in rows:
            if row.get("cycle") == finished and row.get("gain_delta") is None:
                row["gain_delta"] = gain_delta
        self.cycle_strategy_log = rows[-50:]
        self.macro_cycle = finished + 1
        self.no_gain_cycle_streak = no_gain_cycle_streak
        self.gain_at_cycle_start = self.cumulative_gain_validated
        self.reset_per_cycle_plateau_state()

    def note_explore_outcome(self, *, promoted: bool) -> None:
        """Update the plateau proxy after one explore task (KEEP resets, no-promote increments)."""
        if promoted:
            self.params_no_promote_streak = 0
        else:
            self.params_no_promote_streak += 1

    def record_specialist_patch_verdict(
        self,
        subject: str,
        verdict: str,
    ) -> None:
        """Record the Critic verdict for a patch's review subject; idempotent (later verdict overwrites)."""
        sid = str(subject or "").strip()
        if not sid:
            return
        self.specialist_patch_verdicts[sid] = str(verdict or "").strip().lower()

    def get_specialist_patch_verdict(
        self,
        subject: str,
    ) -> str:
        """Return the patch verdict, or empty when no Critic decision exists."""
        sid = str(subject or "").strip()
        if not sid:
            return ""
        return self.specialist_patch_verdicts.get(sid, "") or ""

    def apply_explore_search_update(self, update: dict[str, Any]) -> None:
        """Merge an ExploreExecutor search update into persistent state; :meth:`record_explore_accepted` is the single writer for the ``accepted`` bucket."""
        if not isinstance(update, dict):
            return
        prior = self.explore_search if isinstance(self.explore_search, dict) else {}
        merged = dict(prior)
        cur_cycle = int(getattr(self, "macro_cycle", 0) or 0)
        # The executor reports the round it just benched; accumulating it over the
        # durable ledger is this layer's job, and a re-measured fingerprint replaces
        # its earlier row because that is what a fresh measurement means.
        tested = {**(prior.get("tested") or {}), **(update.get("tested") or {})}
        _stamp_cycle(tested.values(), cur_cycle)
        merged["tested"] = _cap_tested_ledger(tested)
        rejected = _merge_rejected(prior.get("rejected"), update.get("rejected"))
        _stamp_cycle(rejected, cur_cycle)
        merged["rejected"] = rejected
        # The round ordinal, not the ledger's size: consumers key idempotency and
        # round ids on it, so it has to advance once per benched round.
        merged["cursor"] = int(prior.get("cursor") or 0) + 1
        merged["last_round"] = dict(update.get("last_round") or {})
        # Append-only history fields — merge instead of overwrite.
        merged["winners_history"] = _merge_winners_history(
            prior.get("winners_history"),
            update.get("winners_history"),
            cur_cycle,
        )
        # Preserve accepted bucket from prior runs (record_explore_accepted is its writer).
        merged["accepted"] = list(prior.get("accepted") or [])
        self.explore_search = merged

    def record_explore_accepted(self, variant: dict[str, Any]) -> None:
        """Append one promoted variant to ``explore_search.accepted``; dedupes by ``fingerprint`` and removes any matching ``rejected`` entry."""
        if not isinstance(variant, dict) or not variant:
            return
        config = _accepted_config(variant)
        fp = config.fingerprint
        entry = _accepted_entry(variant, config, int(getattr(self, "macro_cycle", 0) or 0))
        search = dict(self.explore_search or {})
        accepted = [
            v for v in (search.get("accepted") or []) if not (isinstance(v, dict) and v.get("fingerprint") == fp)
        ]
        accepted.append(entry)
        search["accepted"] = accepted
        search["rejected"] = [
            v for v in (search.get("rejected") or []) if not (isinstance(v, dict) and v.get("fingerprint") == fp)
        ]
        self.explore_search = search

    def record_authored_framework_levers(
        self,
        switches: list[dict[str, Any]],
        *,
        default_on: bool,
    ) -> bool:
        """Register accepted framework-rewrite switches as search levers."""
        if not isinstance(switches, list) or not switches:
            return False
        rows = list(getattr(self, "authored_framework_levers", None) or [])
        by_switch = {str(r.get("switch") or ""): i for i, r in enumerate(rows) if isinstance(r, dict)}
        changed = False
        for entry in switches:
            if not isinstance(entry, dict):
                continue
            row = _lever_row(entry, bool(default_on))
            name = row["switch"]
            if not name:
                continue
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
            if updated != row:
                rows[index] = updated
                self.authored_framework_levers = rows
                return True
            return False
        return False
