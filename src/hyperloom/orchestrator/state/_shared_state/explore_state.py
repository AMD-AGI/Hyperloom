# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""SharedState — single-writer (Coordinator) persisted session state, backed by atomic JSON at ``$SESSION_DIR/state.json``; enforces CORE_STATE_FIELDS guards.

Fields::

    session_id          str   — set by Coordinator at session creation
    model_name          str   — e.g. "meta-llama/Llama-3.1-8B-Instruct"
    model_path          str   — local NFS path to weights
    model_class         str   — categorical key supplied via --model-class
    model_arch          dict  — advisory architecture profile (hybrid
                                structured + free-text notes) loaded from
                                the launcher's ``$USER_DATA_PATH/model_arch.json``;
                                prompt-context only, no deterministic gating
    model_architectures list  — config.json ``architectures``; stamped into
                                the recipe-snapshot ``extras`` as a KB tag
    model_type          str   — config.json ``model_type``; stamped into
                                the recipe-snapshot ``extras`` as a KB tag
    target_summary      str   — set by `target_analysis` action
    baseline_tput       float — primary throughput after `baseline` action;
                                tok/s/GPU for serving frameworks, img/s for
                                scriptable xDiT (displayed as e2el_mean_ms)
    baseline_accuracy   float — GSM8K score after `baseline`
    current_best        dict  — {action: str, tput: float, accuracy: float}
    cumulative_gain     float — % over baseline
    stop_reason         str   — set when graceful stop fires (§9)
    current_action      str   — what's running right now (set by Orchestration)
    crash_count         int   — incremented by the Coordinator when a tick/agent
                                exception is recorded; also appends to
                                crash_timestamps (Robustness only reads it)
    pruned_families     list[str]  — set by Robustness via PRUNE_BRANCH
    start_ts            str   — ISO timestamp
    max_minutes         int   — wall-clock budget (0 = unlimited)
    last_profile_trace  str   — set by Coordinator when `profile` returns a
                                trace path; consumed by Orch to populate
                                `trace_analyze` REQUEST `trace_input` param
"""

from __future__ import annotations

from typing import Any



from ..shared_state import (
    _GAPS_ATTEMPTS_HISTORY,
    _GAPS_MAX_ENTRIES,
    _INTERVENTION_MIX_CAP,
    _SEEN_PR_IDS_CAP,
    _SPECIALIST_ROUNDS_CAP,
    _WINNERS_HISTORY_CAP,
    _cap_tested_ledger,
    _now_iso,
    _stamp_cycle_on_rejected,
    _stamp_cycle_on_tested,
    log,
)


class _ExploreStateMixin:
    # specialist round bookkeeping
    def record_specialist_round(self, entry: dict[str, Any]) -> None:
        """Append one round summary to ``specialist_rounds``; idempotent on ``round_id`` (re-record overwrites).

        Args:
            entry (dict[str, Any]): The round summary; an empty / non-dict
                value is a no-op. A blank ``round_id`` always appends.
        """
        if not isinstance(entry, dict) or not entry:
            return
        round_id = str(entry.get("round_id") or "").strip()
        if not round_id:
            self.specialist_rounds.append(dict(entry))
        else:
            existing = self.specialist_rounds
            matched = False
            for i, prev in enumerate(existing):
                if isinstance(prev, dict) and str(prev.get("round_id") or "") == round_id:
                    existing[i] = dict(entry)
                    matched = True
                    break
            if not matched:
                existing.append(dict(entry))
        self._trim_specialist_rounds()
        # Author-time breakdown capture: one specialist_runs item per round.
        try:
            from inference_optimizer.breakdown.recorder import instrument

            instrument.record_specialist_round(
                getattr(self, "_session_dir", None),
                entry,
            )
        except Exception:  # noqa: BLE001 — author-time capture must never block record
            log.debug("record_specialist_round capture failed", exc_info=True)

    def _trim_specialist_rounds(self) -> None:
        """Bound the specialist-round ledger for multi-day runs (keep most recent)."""
        if len(self.specialist_rounds) > _SPECIALIST_ROUNDS_CAP:
            self.specialist_rounds = self.specialist_rounds[-_SPECIALIST_ROUNDS_CAP:]

    def bump_specialist_domain_empty_streak(
        self,
        domain: str,
        *,
        empty: bool,
    ) -> int:
        """Increment/reset the per-domain empty-proposal streak; returns new value (escalation threshold lives in ``KB_design §3.9``, not here).

        Args:
            domain (str): The specialist domain; blank normalizes to
                ``"unknown"``.
            empty (bool): ``True`` to increment the streak, ``False`` to reset
                it to zero.

        Returns:
            int: The post-update streak value for the domain.
        """
        d = str(domain or "").strip() or "unknown"
        if empty:
            self.specialist_domain_empty_streak[d] = int(self.specialist_domain_empty_streak.get(d, 0) or 0) + 1
        else:
            self.specialist_domain_empty_streak[d] = 0
        return self.specialist_domain_empty_streak[d]

    # per-anchor coverage counters (point 1)
    def bump_domain_round_counters(self) -> None:
        """Increment both per-anchor round counters for every knowledge-domain
        anchor. Called once per EXPLORE round so a long-idle domain's counters
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
        ``keep_threshold`` (point 1 lower-bound; point 2 hard-trigger reads
        this). Deterministically ordered by widest gap first.

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
        highest severity, then least-attempted, then oldest (point 2 reads this
        to force-dispatch a stalled domain that still has pending work).

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

    # gaps ledger helpers
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
        now = _now_iso()
        existing = self.find_gap(cid)
        if existing is None:
            merged: dict[str, Any] = {
                "canonical_id": cid,
                "symptom": str(entry.get("symptom") or ""),
                "layer": str(entry.get("layer") or ""),
                "severity": str(entry.get("severity") or "medium"),
                "domain_hint": str(entry.get("domain_hint") or ""),
                "source": str(entry.get("source") or ""),
                # Optional origin reference (PR/blog URL) sedimented into the recipe with KEEP/REVERT provenance.
                "provenance": str(entry.get("provenance") or ""),
                "first_seen_ts": str(entry.get("first_seen_ts") or now),
                "last_updated_ts": now,
                "attempts": list(entry.get("attempts") or []),
            }
            if len(merged["attempts"]) > _GAPS_ATTEMPTS_HISTORY:
                merged["attempts"] = merged["attempts"][-_GAPS_ATTEMPTS_HISTORY:]
            self.gaps.append(merged)
        else:
            # Field-wise merge: incoming non-empty values win except ``first_seen_ts`` (preserve oldest).
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
                if len(merged_attempts) > _GAPS_ATTEMPTS_HISTORY:
                    merged_attempts = merged_attempts[-_GAPS_ATTEMPTS_HISTORY:]
                existing["attempts"] = merged_attempts
            merged = existing
        # Enforce global cap, trimming oldest after the upsert so the just-touched gap is retained.
        if len(self.gaps) > _GAPS_MAX_ENTRIES:
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
            keep_count = _GAPS_MAX_ENTRIES - 1
            others = others[-keep_count:] if keep_count > 0 else []
            self.gaps = others + [merged]
        return merged

    def append_gap_attempt(
        self,
        canonical_id: str,
        attempt: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Append one attempt row to an existing gap; returns the gap or ``None`` when unknown (caller may ``upsert_gap`` instead).

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
        attempts.append(dict(attempt) | {"ts": str(attempt.get("ts") or _now_iso())})
        if len(attempts) > _GAPS_ATTEMPTS_HISTORY:
            attempts = attempts[-_GAPS_ATTEMPTS_HISTORY:]
        gap["attempts"] = attempts
        gap["last_updated_ts"] = _now_iso()
        return gap

    def replace_gaps(self, entries: list[dict[str, Any]]) -> None:
        """Bulk-replace ``gaps`` with a fresh dedup'd list (discards stale rows wholesale); idempotent.

        Args:
            entries (list[dict[str, Any]]): The replacement gap rows; deduped
                by ``canonical_id``, per-gap attempts and the list are capped.
                A non-list value is a no-op.
        """
        if not isinstance(entries, list):
            return
        dedup: dict[str, dict[str, Any]] = {}
        order: list[str] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            cid = str(entry.get("canonical_id") or "").strip()
            if not cid:
                continue
            if cid not in dedup:
                order.append(cid)
            dedup[cid] = dict(entry)
        # Apply the per-entry cap on attempts.
        new_list: list[dict[str, Any]] = []
        for cid in order:
            row = dedup[cid]
            attempts = list(row.get("attempts") or [])
            if len(attempts) > _GAPS_ATTEMPTS_HISTORY:
                attempts = attempts[-_GAPS_ATTEMPTS_HISTORY:]
            row["attempts"] = attempts
            new_list.append(row)
        if len(new_list) > _GAPS_MAX_ENTRIES:
            new_list = new_list[-_GAPS_MAX_ENTRIES:]
        self.gaps = new_list

    def record_intervention(
        self,
        *,
        change_type: str,
        action: str,
        task_id: str = "",
        delta_pct: float | None = None,
    ) -> None:
        """Append one intervention entry and update config-only counters. consecutive-config counter advances on ``"config"``, resets on ``"code_patch"``; ``"code_patch_attempt"`` is telemetry-only.

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
            "ts": _now_iso(),
        }
        self.intervention_mix.append(entry)
        if len(self.intervention_mix) > _INTERVENTION_MIX_CAP:
            self.intervention_mix = self.intervention_mix[-_INTERVENTION_MIX_CAP:]
        if ct == "config":
            self.consecutive_config_only_rounds = int(self.consecutive_config_only_rounds or 0) + 1
        elif ct == "code_patch":
            self.consecutive_config_only_rounds = 0

    def get_intervention_mix(self, *, recent_window: int = 5) -> dict[str, Any]:
        """Summarise the intervention-mix ledger as derived counts (config vs code_patch totals, recent window, consecutive_config_only, config_heavy). Read-only; unknown change_types ignored in tallies but break the trailing config-only run.

        Args:
            recent_window (int): Size of the trailing window for the
                ``recent_*`` tallies; non-positive uses the full ledger.

        Returns:
            dict[str, Any]: Derived totals (total/recent config and
                code_patch counts, ``consecutive_config_only``,
                ``config_heavy``).
        """
        ledger = [e for e in (self.intervention_mix or []) if isinstance(e, dict)]

        def _ct(entry: dict[str, Any]) -> str:
            """Return an entry's normalized ``change_type`` string.

            Args:
                entry: A single intervention-mix ledger entry.

            Returns:
                The lowercased, stripped change type (empty if absent).
            """
            return str(entry.get("change_type") or "").strip().lower()

        total_config = sum(1 for e in ledger if _ct(e) == "config")
        total_code_patch = sum(1 for e in ledger if _ct(e) == "code_patch")
        total_code_patch_attempt = sum(1 for e in ledger if _ct(e) in ("code_patch", "code_patch_attempt"))
        window = ledger[-recent_window:] if recent_window > 0 else ledger
        recent_config = sum(1 for e in window if _ct(e) == "config")
        recent_code_patch = sum(1 for e in window if _ct(e) == "code_patch")
        recent_code_patch_attempt = sum(1 for e in window if _ct(e) in ("code_patch", "code_patch_attempt"))

        consecutive_config_only = 0
        for e in reversed(ledger):
            ct = _ct(e)
            if ct == "config":
                consecutive_config_only += 1
            else:
                break

        return {
            "total_config": total_config,
            "total_code_patch": total_code_patch,
            "total_code_patch_attempt": total_code_patch_attempt,
            "recent_config": recent_config,
            "recent_code_patch": recent_code_patch,
            "recent_code_patch_attempt": recent_code_patch_attempt,
            "consecutive_config_only": consecutive_config_only,
            "config_heavy": total_config >= recent_window and total_code_patch == 0,
        }

    def bump_specialist_dispatched(self, n: int = 1) -> int:
        """Increment the per-EXPLORE specialist dispatch counter; returns post-increment value.

        Args:
            n (int): Amount to add to the dispatch counter (default 1).

        Returns:
            int: The post-increment dispatch count.
        """
        self.explore_specialist_dispatched_count = int(self.explore_specialist_dispatched_count or 0) + int(n)
        return self.explore_specialist_dispatched_count

    def reset_specialist_dispatched(self) -> None:
        """Zero the per-EXPLORE specialist dispatch counter (on fresh EXPLORE entry)."""
        self.explore_specialist_dispatched_count = 0

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
        if len(self.research_scout_seen_pr_ids) > _SEEN_PR_IDS_CAP:
            # FIFO eviction of oldest-seen ids; re-surfacing a very old PR is low-harm.
            self.research_scout_seen_pr_ids = self.research_scout_seen_pr_ids[-_SEEN_PR_IDS_CAP:]
        return added

    def has_seen_pr_id(self, pr_id: Any) -> bool:
        """True iff ``pr_id`` was already surfaced by scout / FRAMEWORK.

        Args:
            pr_id (Any): The PR id to check.

        Returns:
            bool: ``True`` when the id is non-blank and already seen.
        """
        pid = str(pr_id or "").strip()
        return bool(pid) and pid in set(self.research_scout_seen_pr_ids or [])

    # Explore plateau proxy
    def reset_explore_plateau_proxy(self) -> None:
        """Reset the legacy explore plateau proxy counter."""
        self.params_no_promote_streak = 0

    def note_explore_outcome(self, *, promoted: bool) -> None:
        """Update the legacy plateau proxy after one explore task (KEEP resets, no-promote increments).

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
        specialist_task_id: str,
        verdict: str,
    ) -> None:
        """Record the Critic verdict for a specialist worktree patch; idempotent (later verdict overwrites), empty ``verdict`` clears the entry to force re-review.

        Args:
            specialist_task_id (str): The specialist task id; blank is a
                no-op.
            verdict (str): The Critic verdict; blank clears the recorded
                verdict to force re-review.
        """
        sid = str(specialist_task_id or "").strip()
        if not sid:
            return
        v = str(verdict or "").strip().lower()
        if not v:
            self.specialist_patch_verdicts.pop(sid, None)
            return
        self.specialist_patch_verdicts[sid] = v

    def get_specialist_patch_verdict(
        self,
        specialist_task_id: str,
    ) -> str:
        """Return the patch verdict, or empty when no Critic decision exists.

        Args:
            specialist_task_id (str): The specialist task id.

        Returns:
            str: The recorded verdict, or ``""`` when none exists or the id
                is blank.
        """
        sid = str(specialist_task_id or "").strip()
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
        """Merge an ExploreExecutor search update into persistent state; executor never writes ``accepted`` directly — :meth:`record_explore_accepted` is the single writer for that bucket.

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
        merged["tested"] = _cap_tested_ledger(
            _stamp_cycle_on_tested(
                dict(update.get("tested") or prior.get("tested") or {}),
                cur_cycle,
                cur_bottleneck,
            )
        )
        merged["rejected"] = _stamp_cycle_on_rejected(
            list(update.get("rejected") or prior.get("rejected") or []),
            cur_cycle,
            cur_bottleneck,
        )
        merged["name_index"] = dict(update.get("name_index") or prior.get("name_index") or {})
        merged["cursor"] = int(update.get("cursor") or len(merged["tested"]))
        merged["last_round"] = dict(update.get("last_round") or {})
        # Append-only history fields — merge instead of overwrite.
        wh = list(prior.get("winners_history") or [])
        for entry in update.get("winners_history") or []:
            if isinstance(entry, dict):
                wh.append(dict(entry))
        merged["winners_history"] = wh[-_WINNERS_HISTORY_CAP:]
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
        # Drop merged_from_legacy_sig so a later load re-runs the legacy union.
        merged.pop("merged_from_legacy_sig", None)
        self.explore_search = merged

    def record_explore_accepted(self, variant: dict[str, Any]) -> None:
        """Append one promoted variant to ``explore_search.accepted``; dedupes by ``fingerprint`` and removes any matching ``rejected`` entry so a variant isn't in both buckets.

        Args:
            variant (dict[str, Any]): The promoted variant to record; an
                empty / non-dict value is a no-op.
        """
        if not isinstance(variant, dict) or not variant:
            return
        from ...actions.executors._canonical_fingerprint import canonical_fingerprint

        args = str(variant.get("candidate_extra_server_args") or variant.get("extra_server_args") or "")
        envs = dict(variant.get("extra_envs") or {})
        fp = str(variant.get("fingerprint") or canonical_fingerprint(args, envs))
        entry = {
            "fingerprint": fp,
            "name": str(variant.get("name") or ""),
            "extra_server_args": args,
            "extra_envs": envs,
            "note": str(variant.get("note") or ""),
            "tput": variant.get("output_throughput") or variant.get("tput"),
            "gain_pct": variant.get("gain_pct"),
            "stack_index": variant.get("stack_index"),
            "accepted_at_round": str(variant.get("accepted_at_round") or ""),
            "ts": str(variant.get("ts") or _now_iso()),
            "provenance": str(variant.get("provenance") or "llm_direct"),
            # R3: attribute the win to the macro-cycle it landed in.
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
                "provenance": entry["provenance"],
                "ts": entry["ts"],
                "cycle": entry["cycle"],
            }
        )
        search["winners_history"] = wh[-_WINNERS_HISTORY_CAP:]
        self.explore_search = search

    # search-space expansion bookkeeping
    def record_discovered_flags(
        self,
        *,
        framework: str,
        backend_flags: list[str] | None = None,
        param_flags: list[str] | None = None,
        source_path: str = "",
    ) -> None:
        """Persist the AST-discovered flag list for a framework; the prompt surfaces the union so the LLM synthesizes new GridVariants beyond DEFAULT_*_GRID. Idempotent per-framework.

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
        entry["ts"] = _now_iso()
        self.discovered_flags[fw] = entry

