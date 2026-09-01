# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``_ExploreStateMixin`` — explore / gap / specialist-ledger mutators for
:class:`..shared_state.SharedState`.
"""

from __future__ import annotations

import logging
from typing import Any


log = logging.getLogger(__name__)


def _shared_state_module():
    """Import parent shared_state lazily to avoid a module-level cycle."""
    from .. import shared_state

    return shared_state


class _ExploreStateMixin:
    def record_specialist_round(self, entry: dict[str, Any]) -> None:
        """Append one round summary to ``specialist_rounds``; idempotent on ``round_id`` (re-record overwrites).

        Args:
            entry (dict[str, Any]): The round summary; an empty / non-dict
                value is a no-op. A blank ``round_id`` always appends.
        """
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
        # Author-time breakdown capture: one specialist_runs item per round.
        try:
            from hyperloom.inference_optimizer.breakdown.recorder import instrument

            instrument.record_specialist_round(
                getattr(self, "_session_dir", None),
                entry,
                phase=str(getattr(self, "phase", "") or ""),
            )
        except Exception:  # noqa: BLE001 — author-time capture must never block record
            log.debug("record_specialist_round capture failed", exc_info=True)

    def _trim_specialist_rounds(self) -> None:
        """Bound the specialist-round ledger for multi-day runs (keep most recent)."""
        cap = _shared_state_module()._SPECIALIST_ROUNDS_CAP
        if len(self.specialist_rounds) > cap:
            self.specialist_rounds = self.specialist_rounds[-cap:]

    def bump_domain_round_counters(self) -> None:
        """Increment both per-anchor round counters for every knowledge-domain
        anchor. Called once per optimisation round so a long-idle domain's counters
        climb until the Coordinator escalation forces a dispatch."""
        from ...specialists.domains import KNOWLEDGE_DOMAIN_TAGS

        for anchor in KNOWLEDGE_DOMAIN_TAGS:
            self.rounds_since_last_specialist[anchor] = int(self.rounds_since_last_specialist.get(anchor, 0) or 0) + 1
            self.rounds_since_last_keep[anchor] = int(self.rounds_since_last_keep.get(anchor, 0) or 0) + 1

    @staticmethod
    def _anchor_for(domain_or_anchor: str) -> str:
        """Resolve a domain key or raw tag to its canonical kb_anchor.

        Args:
            domain_or_anchor (str): A domain key, raw tag, or anchor string.

        Returns:
            str: The canonical kb_anchor, falling back to the trimmed input
                when no mapping resolves (``""`` for blank input).
        """
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
        """Reset ``rounds_since_last_specialist`` for the dispatched anchor.

        Args:
            domain_or_anchor (str): The dispatched domain / anchor whose
                counter should be reset.
        """
        anchor = self._anchor_for(domain_or_anchor)
        if anchor:
            self.rounds_since_last_specialist[anchor] = 0

    def note_domain_keep(self, domain_or_anchor: str) -> None:
        """Reset ``rounds_since_last_keep`` for the anchor that just KEPT.

        Args:
            domain_or_anchor (str): The domain / anchor whose KEEP counter
                should be reset.
        """
        anchor = self._anchor_for(domain_or_anchor)
        if anchor:
            self.rounds_since_last_keep[anchor] = 0

    def stalled_domains(
        self,
        *,
        specialist_threshold: int,
        keep_threshold: int,
    ) -> list[str]:
        """Return anchors whose ``rounds_since_last_specialist`` ≥
        ``specialist_threshold`` OR ``rounds_since_last_keep`` ≥
        ``keep_threshold``. Deterministically ordered by widest gap first.

        Args:
            specialist_threshold (int): Rounds-since-last-specialist value at
                or above which an anchor counts as stalled.
            keep_threshold (int): Rounds-since-last-keep value at or above
                which an anchor counts as stalled.

        Returns:
            list[str]: Stalled anchors ordered by widest gap first, then by
                anchor name.
        """
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
        """Return the canonical_id of the most actionable open gap whose
        ``domain_hint`` resolves to ``anchor`` (or ``""`` when none). Selection:
        highest severity, then least-attempted, then oldest.

        Args:
            anchor (str): The domain / anchor to match gaps against.

        Returns:
            str: The ``canonical_id`` of the most actionable matching gap, or
                ``""`` when none match.
        """
        target = self._anchor_for(anchor)
        if not target:
            return ""
        severity_rank = {"high": 3, "medium": 2, "low": 1}
        matches: list[tuple[tuple[int, int, str], str]] = []
        for g in self.gaps:
            if not isinstance(g, dict):
                continue
            cid = str(g.get("canonical_id") or "").strip()
            if not cid:
                continue
            if self._anchor_for(str(g.get("domain_hint") or "")) != target:
                continue
            sev = severity_rank.get(str(g.get("severity") or "").lower(), 0)
            attempts = len(g.get("attempts") or [])
            first_seen = str(g.get("first_seen_ts") or "")
            matches.append(((-sev, attempts, first_seen), cid))
        if not matches:
            return ""
        matches.sort(key=lambda m: m[0])
        return matches[0][1]

    def find_gap(self, canonical_id: str) -> dict[str, Any] | None:
        """Return the gap entry matching ``canonical_id`` (or ``None``).

        Args:
            canonical_id (str): The gap's canonical identifier.

        Returns:
            dict[str, Any] | None: The matching gap row, or ``None`` when the
                id is blank or not present.
        """
        if not canonical_id:
            return None
        cid = str(canonical_id)
        for gap in self.gaps:
            if isinstance(gap, dict) and str(gap.get("canonical_id") or "") == cid:
                return gap
        return None

    def upsert_gap(self, entry: dict[str, Any]) -> dict[str, Any]:
        """Insert or update one gap row, keyed by ``canonical_id``. Coordinator-only writer (Inv-1 single-writer + CORE_STATE_FIELDS lock). Returns the merged entry.

        Args:
            entry (dict[str, Any]): The gap row to upsert; must carry a
                non-empty ``canonical_id`` (else a no-op returning ``{}``).

        Returns:
            dict[str, Any]: The inserted-or-merged gap row, or ``{}`` when the
                entry is invalid.
        """
        if not isinstance(entry, dict):
            return {}
        cid = str(entry.get("canonical_id") or "").strip()
        if not cid:
            return {}
        ss = _shared_state_module()
        now = ss._now_iso()
        existing = self.find_gap(cid)
        if existing is None:
            merged: dict[str, Any] = {
                "canonical_id": cid,
                "symptom": str(entry.get("symptom") or ""),
                "layer": str(entry.get("layer") or ""),
                "severity": str(entry.get("severity") or "medium"),
                "domain_hint": str(entry.get("domain_hint") or ""),
                "source": str(entry.get("source") or ""),
                # Optional origin reference (PR/blog URL).
                "provenance": str(entry.get("provenance") or ""),
                "first_seen_ts": str(entry.get("first_seen_ts") or now),
                "last_updated_ts": now,
                "attempts": list(entry.get("attempts") or []),
            }
            if len(merged["attempts"]) > ss._GAPS_ATTEMPTS_HISTORY:
                merged["attempts"] = merged["attempts"][-ss._GAPS_ATTEMPTS_HISTORY :]
            self.gaps.append(merged)
        else:
            # Field-wise merge: incoming non-empty values win except ``first_seen_ts``.
            for key in ("symptom", "layer", "severity", "domain_hint", "source", "provenance"):
                incoming = entry.get(key)
                if incoming:
                    existing[key] = str(incoming)
            existing.setdefault("first_seen_ts", str(entry.get("first_seen_ts") or now))
            existing["last_updated_ts"] = now
            incoming_attempts = list(entry.get("attempts") or [])
            if incoming_attempts:
                merged_attempts = list(existing.get("attempts") or []) + incoming_attempts
                # Capped tail; callers supply newest-last lists (convention).
                if len(merged_attempts) > ss._GAPS_ATTEMPTS_HISTORY:
                    merged_attempts = merged_attempts[-ss._GAPS_ATTEMPTS_HISTORY :]
                existing["attempts"] = merged_attempts
            merged = existing
        # Enforce global cap, trimming oldest after the upsert so the just-touched gap is retained.
        if len(self.gaps) > ss._GAPS_MAX_ENTRIES:
            others = [g for g in self.gaps if g is not merged]

            def _sort_key(g: dict[str, Any]) -> str:
                """Sort key for gap trimming: newest-updated timestamp.

                Args:
                    g (dict[str, Any]): A gap row.

                Returns:
                    str: ``last_updated_ts`` (falling back to
                        ``first_seen_ts`` then ``""``) for chronological sort.
                """
                return str(g.get("last_updated_ts") or g.get("first_seen_ts") or "")

            others.sort(key=_sort_key)
            keep_count = ss._GAPS_MAX_ENTRIES - 1
            others = others[-keep_count:] if keep_count > 0 else []
            self.gaps = others + [merged]
        return merged

    def append_gap_attempt(
        self,
        canonical_id: str,
        attempt: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Append one attempt row to an existing gap; returns the gap or ``None`` when unknown.

        Args:
            canonical_id (str): The gap's canonical identifier.
            attempt (dict[str, Any]): The attempt row to append (a ``ts`` is
                stamped when absent).

        Returns:
            dict[str, Any] | None: The updated gap row, or ``None`` when no
                gap matches ``canonical_id``.
        """
        gap = self.find_gap(canonical_id)
        if gap is None:
            return None
        attempts = list(gap.get("attempts") or [])
        ss = _shared_state_module()
        attempts.append(dict(attempt) | {"ts": str(attempt.get("ts") or ss._now_iso())})
        if len(attempts) > ss._GAPS_ATTEMPTS_HISTORY:
            attempts = attempts[-ss._GAPS_ATTEMPTS_HISTORY :]
        gap["attempts"] = attempts
        gap["last_updated_ts"] = ss._now_iso()
        return gap

    def record_intervention(
        self,
        *,
        change_type: str,
        action: str,
        task_id: str = "",
        delta_pct: float | None = None,
    ) -> None:
        """Append one intervention entry and update config-only counters; the consecutive-config counter advances on ``"config"`` and resets on ``"code_patch"``.

        Args:
            change_type (str): The intervention kind (e.g. ``config`` /
                ``code_patch`` / ``code_patch_attempt``).
            action (str): The action name driving the intervention.
            task_id (str): The originating task id.
            delta_pct (float | None): Optional measured gain delta for the
                intervention.
        """
        ct = str(change_type or "").strip().lower()
        entry = {
            "change_type": ct,
            "action": str(action or ""),
            "task_id": str(task_id or ""),
            "delta_pct": delta_pct,
            "ts": _shared_state_module()._now_iso(),
        }
        self.intervention_mix.append(entry)
        cap = _shared_state_module()._INTERVENTION_MIX_CAP
        if len(self.intervention_mix) > cap:
            self.intervention_mix = self.intervention_mix[-cap:]
        if ct == "config":
            self.consecutive_config_only_rounds = int(self.consecutive_config_only_rounds or 0) + 1
        elif ct == "code_patch":
            self.consecutive_config_only_rounds = 0

    def bump_research_scout_runs(self, n: int = 1) -> int:
        """Increment the research-scout dispatch counter; return new total.

        Args:
            n (int): Amount to add to the scout-run counter (default 1).

        Returns:
            int: The post-increment scout-run total.
        """
        self.research_scout_runs = int(self.research_scout_runs or 0) + int(n)
        return self.research_scout_runs

    def register_seen_pr_ids(self, pr_ids: Any) -> int:
        """Add PR ids to the shared seen-set (scout + FRAMEWORK dedup); returns count newly added.

        Args:
            pr_ids (Any): An iterable of PR id values; blanks and duplicates
                are skipped.

        Returns:
            int: The number of PR ids newly added to the seen-set.
        """
        seen = set(self.research_scout_seen_pr_ids or [])
        added = 0
        for raw in pr_ids or []:
            pid = str(raw or "").strip()
            if not pid or pid in seen:
                continue
            seen.add(pid)
            self.research_scout_seen_pr_ids.append(pid)
            added += 1
        cap = _shared_state_module()._SEEN_PR_IDS_CAP
        if len(self.research_scout_seen_pr_ids) > cap:
            # FIFO eviction of oldest-seen ids.
            self.research_scout_seen_pr_ids = self.research_scout_seen_pr_ids[-cap:]
        return added

    def reset_explore_plateau_proxy(self) -> None:
        """Reset the explore plateau proxy counter."""
        self.params_no_promote_streak = 0

    def reset_per_cycle_plateau_state(self) -> None:
        """Reset transient plateau and dispatch state for a macro-cycle, plus any stale escalate hint.

        ``pending_escalate_hint`` isn't plateau or dispatch state, but it is
        reset here too: a hint set during the prior macro-cycle (e.g. by
        kernel_agent completion) and never claimed by the transition it
        caused must not survive into the next cycle's phases as if they'd
        earned it.
        """
        self.params_no_promote_streak = 0
        self.framework_agent_phase_done = False
        self.framework_agent_discover_failures = 0
        self.framework_agent_empty_discoveries = 0
        self.specialist_domain_empty_streak = {}
        self.rounds_since_last_specialist = {}
        self.rounds_since_last_keep = {}
        self.last_sweep = {}
        self.last_conc_sweep = {}
        self.discard_pending_escalate_hint()

    def note_explore_outcome(self, *, promoted: bool) -> None:
        """Update the plateau proxy after one explore task (KEEP resets, no-promote increments).

        Args:
            promoted (bool): ``True`` when the explore task produced a KEEP
                (resets the proxy); ``False`` increments the no-promote
                streak.
        """
        if promoted:
            self.reset_explore_plateau_proxy()
        else:
            self.params_no_promote_streak += 1

    def record_specialist_patch_verdict(
        self,
        subject: str,
        verdict: str,
    ) -> None:
        """Record the Critic verdict for a patch's review subject; idempotent (later verdict overwrites), empty ``verdict`` clears the entry to force re-review.

        Args:
            subject (str): The specialist task id for an authored patch, the
                candidate id for a PR pre-screen; blank is a no-op.
            verdict (str): The Critic verdict; blank clears the recorded
                verdict to force re-review.
        """
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
        """Return the patch verdict, or empty when no Critic decision exists.

        Args:
            subject (str): The review subject, as
                :meth:`record_specialist_patch_verdict` keys it.

        Returns:
            str: The recorded verdict, or ``""`` when none exists or the id
                is blank.
        """
        sid = str(subject or "").strip()
        if not sid:
            return ""
        return self.specialist_patch_verdicts.get(sid, "") or ""

    def update_last_specialist(self, snapshot: dict[str, Any]) -> None:
        """Snapshot the most recent specialist task (parity with last_*).

        Args:
            snapshot (dict[str, Any]): The specialist task snapshot to copy
                into ``last_specialist``. A non-dict value is ignored.
        """
        if isinstance(snapshot, dict):
            self.last_specialist = dict(snapshot)

    def apply_explore_search_update(self, update: dict[str, Any]) -> None:
        """Merge an ExploreExecutor search update into persistent state; :meth:`record_explore_accepted` is the single writer for the ``accepted`` bucket.

        Args:
            update (dict[str, Any]): The executor's explore-search update
                (tested / rejected / name_index / history fields); a non-dict
                value is a no-op.
        """
        if not isinstance(update, dict):
            return
        prior = self.explore_search if isinstance(self.explore_search, dict) else {}
        merged = dict(prior)
        merged["schema_version"] = int(update.get("schema_version") or 1)
        cur_cycle = int(getattr(self, "macro_cycle", 0) or 0)
        cur_bottleneck = self.current_top_bottleneck()
        ss = _shared_state_module()
        merged["tested"] = ss._cap_tested_ledger(
            ss._stamp_cycle_on_tested(
                dict(update.get("tested") or prior.get("tested") or {}),
                cur_cycle,
                cur_bottleneck,
            )
        )
        merged["rejected"] = ss._stamp_cycle_on_rejected(
            list(update.get("rejected") or prior.get("rejected") or []),
            cur_cycle,
            cur_bottleneck,
        )
        merged["name_index"] = dict(update.get("name_index") or prior.get("name_index") or {})
        merged["cursor"] = int(update.get("cursor") or len(merged["tested"]))
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
        merged["winners_history"] = wh[-ss._WINNERS_HISTORY_CAP :]
        sa: set[tuple[str, ...]] = set()
        for src in (prior.get("synergy_attempted"), update.get("synergy_attempted")):
            for c in src or []:
                if isinstance(c, list):
                    items = tuple(sorted(str(x) for x in c if isinstance(x, str)))
                    if items:
                        sa.add(items)
                elif isinstance(c, str) and c:
                    items = tuple(sorted(c.split("+")))
                    if items:
                        sa.add(items)
        merged["synergy_attempted"] = [list(c) for c in sorted(sa)]
        merged["discovered_flags"] = list(update.get("discovered_flags") or prior.get("discovered_flags") or [])
        merged["domains_round_summary"] = list(
            update.get("domains_round_summary") or prior.get("domains_round_summary") or []
        )
        # Preserve accepted bucket from prior runs (record_explore_accepted is its writer).
        merged["accepted"] = list(prior.get("accepted") or [])
        # Drop so a later load re-runs the legacy union.
        merged.pop("merged_from_legacy_sig", None)
        self.explore_search = merged

    def record_explore_accepted(self, variant: dict[str, Any]) -> None:
        """Append one promoted variant to ``explore_search.accepted``; dedupes by ``fingerprint`` and removes any matching ``rejected`` entry.

        Args:
            variant (dict[str, Any]): The promoted variant to record; an
                empty / non-dict value is a no-op.
        """
        if not isinstance(variant, dict) or not variant:
            return
        from ...actions.executors._canonical_fingerprint import canonical_fingerprint

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
            # Carried through so the ledger records what the KEEP was judged on;
            # ``None`` means the variant was never gated, not that it scored 0.
            "accuracy": variant.get("accuracy"),
            "stack_index": variant.get("stack_index"),
            "accepted_at_round": str(variant.get("accepted_at_round") or ""),
            "ts": str(variant.get("ts") or _shared_state_module()._now_iso()),
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
        search["winners_history"] = wh[-_shared_state_module()._WINNERS_HISTORY_CAP :]
        self.explore_search = search

    def record_authored_framework_levers(
        self,
        switches: list[dict[str, Any]],
        *,
        default_on: bool,
        specialist_task_id: str = "",
        stack_delta_pct: float | None = None,
    ) -> bool:
        """Register accepted framework-rewrite switches as search levers.

        Each switch behind an accepted rewrite becomes a lever the explore phase
        can toggle, which is what turns an authored bundle into per-rewrite
        attribution and a searchable combination space instead of a take-it-or-
        leave-it patch.

        ``default_on`` records whether the switch is part of the running
        configuration. It is ``False`` for an inert KEEP — the patch cleared
        correctness but the bundle did not clear the throughput gate, so the code
        is kept dormant and the levers are registered for the explore phase to
        try one bundle at a time. Keeping default-off code costs nothing, and
        discarding it would throw away the rewrites that do pay along with the
        one that does not.

        Existing rows are updated in place, keyed by switch name, so a re-run of
        the same rewrite does not duplicate its lever.

        Args:
            switches: Parsed manifest entries (see ``_framework_switch_manifest``).
            default_on: Whether these switches are on in the current best config.
            specialist_task_id: The authoring specialist, for provenance.
            stack_delta_pct: Throughput delta measured with the whole bundle on.

        Returns:
            ``True`` when any lever row was added or changed.
        """
        if not isinstance(switches, list) or not switches:
            return False
        rows = list(getattr(self, "authored_framework_levers", None) or [])
        by_switch = {str(r.get("switch") or ""): i for i, r in enumerate(rows) if isinstance(r, dict)}
        now = _shared_state_module()._now_iso()
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
                # Filled in by the explore phase once the lever has been
                # measured on its own; ``None`` means "registered, not yet
                # attributed".
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
            # Preserve an attribution already measured for this lever; the
            # measurement is more informative than this registration.
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
        """Record a lever's individually measured contribution.

        Args:
            switch: Switch name.
            gain_pct: Measured contribution; ``None`` clears a prior value.
            source: How it was measured — ``"additive"`` (lever switched on
                against the running base) or ``"leave_one_out"`` (lever removed
                from a stack that already had it).

        Returns:
            ``True`` when a lever row was updated.
        """
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

    def record_discovered_flags(
        self,
        *,
        framework: str,
        backend_flags: list[str] | None = None,
        param_flags: list[str] | None = None,
        source_path: str = "",
    ) -> None:
        """Persist the AST-discovered flag list for a framework so the prompt can surface the union. Idempotent per-framework.

        Args:
            framework (str): The framework key; blank normalizes to
                ``"unknown"``.
            backend_flags (list[str] | None): Discovered backend flags;
                ``None`` leaves the existing list untouched.
            param_flags (list[str] | None): Discovered parameter flags;
                ``None`` leaves the existing list untouched.
            source_path (str): Optional path to the scanned source file.
        """
        fw = (framework or "").strip().lower() or "unknown"
        entry = dict(self.discovered_flags.get(fw) or {})
        if backend_flags is not None:
            entry["backend_flags"] = sorted(set(str(f) for f in backend_flags))
        if param_flags is not None:
            entry["param_flags"] = sorted(set(str(f) for f in param_flags))
        if source_path:
            entry["source_path"] = str(source_path)
        entry["ts"] = _shared_state_module()._now_iso()
        self.discovered_flags[fw] = entry
