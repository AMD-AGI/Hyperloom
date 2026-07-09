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

import os
from datetime import datetime
from typing import Any



def _shared_state_module():
    """Import parent shared_state lazily to avoid a module-level cycle."""
    from .. import shared_state

    return shared_state


class _RenderMixin:
    def to_policy_denial_summary(self, *, top_k: int = 6) -> str:
        """Forwarding shim — implementation in :mod:`.policy`."""
        from ...policy import gate as _m
        return _m.to_policy_denial_summary(self, top_k=top_k)

    def _format_last_kernel_opt(self) -> str:
        """Forwarding shim — implementation in :mod:`.kernel_request_handlers`."""
        from ...kernel import request_handlers as _m
        return _m._format_last_kernel_opt(self)

    def to_intervention_mix_summary(self) -> str:
        """Render the intervention ledger as neutral telemetry (one-line counts summary; ``""`` when empty). No directive emitted — config-vs-patch is the LLM's choice.

        Returns:
            str: A single-line counts summary, or ``""`` when the ledger is
                empty.
        """
        mix = self.intervention_mix or []
        if not mix:
            return ""
        n_config = sum(1 for m in mix if (m or {}).get("change_type") == "config")
        n_patch = sum(1 for m in mix if (m or {}).get("change_type") == "code_patch")
        n_patch_attempt = sum(
            1
            for m in mix
            if (m or {}).get("change_type")
            in (
                "code_patch",
                "code_patch_attempt",
            )
        )
        n_config_attempt = sum(1 for m in mix if (m or {}).get("change_type") == "config_attempt")
        consec = int(self.consecutive_config_only_rounds or 0)
        return (
            f"config_keeps={n_config} config_attempts={n_config_attempt} "
            f"code_patch_keeps={n_patch} code_patch_attempts={n_patch_attempt} "
            f"consecutive_config_only_rounds={consec}"
        )

    def to_mission_summary(self, *, now: datetime | None = None) -> str:
        """Mission-progress block printed at the top of every tick (outcome-shaped state: raw/validated gain, time vs budget, stack staleness); distinct from :meth:`to_prompt_summary`.

        Args:
            now (datetime | None): Reference time for elapsed / remaining
                calculations; defaults to the current UTC time.

        Returns:
            str: The multi-line mission-progress block.
        """
        elapsed = self.elapsed_minutes(now=now)
        remaining = self.remaining_minutes(now=now)
        budget_line = (
            (f"time      : elapsed={elapsed:.1f}min remaining={remaining:.1f}min budget={self.max_minutes}min")
            if remaining is not None
            else (f"time      : elapsed={elapsed:.1f}min budget=unlimited")
        )
        validated_age = ""
        if self.cumulative_gain_validated_ts:
            validated_age = f" (ts={self.cumulative_gain_validated_ts})"
        unvalidated = self.optimization_stack_has_unvalidated_keeps()
        unvalidated_tag = (
            " ⚠ stack changed since last rebench — RUN `explore` (per-KEEP stack rebench is inlined)"
            if unvalidated
            else ""
        )
        resume_revalidation_tag = (
            " ⚠ resume_pending_revalidation=true — recheck current stack before trusting validated gain"
            if bool(getattr(self, "resume_pending_revalidation", False))
            else ""
        )
        perfskills_pending_tag = (
            " ⚠ perfskills candidate awaiting main-flow rebench — NOT in headline until validated"
            if isinstance(getattr(self, "perfskills_pending", None), dict)
            and self.perfskills_pending.get("status") == "awaiting_rebench"
            else ""
        )
        from hyperloom.inference_optimizer import framework_registry

        lines = [
            f"baseline  : {framework_registry.format_primary_metric(self.framework, self.baseline_tput)}",
            f"current   : {self._format_current_best_for_mission()}",
            f"gain      : per-round-sum={self.cumulative_gain:.2f}% "
            f"validated={self.cumulative_gain_validated:.2f}%{validated_age}",
            f"stack     : {len(self.optimization_stack)} entries "
            f"(validated_at_len={self.cumulative_gain_validated_stack_len})"
            f"{unvalidated_tag}{resume_revalidation_tag}{perfskills_pending_tag}",
        ]
        # Surface reusable hot kernels still owing a kernel_opt attempt (visible without a checklist).
        untried_hot = self.untried_hot_reusable_kernels()
        if untried_hot:
            lines.append(f"untried_hot_kernels: {', '.join(untried_hot)}")
        lines.append(budget_line)
        return "\n".join(lines)

    def _format_current_best_for_mission(self) -> str:
        """Render the ``current_best`` one-liner for the mission summary.

        Returns:
            str: ``action=... tput=... variant=...``, or ``"(none)"`` when
                no current best is set.
        """
        if not isinstance(self.current_best, dict) or not self.current_best:
            return "(none)"
        from hyperloom.inference_optimizer import framework_registry

        cb_tput = self.current_best.get("tput")
        perf = (
            framework_registry.format_primary_metric(self.framework, cb_tput)
            if isinstance(cb_tput, (int, float))
            else "?"
        )
        return (
            f"action={self.current_best.get('action', '?')} "
            f"perf={perf} "
            f"variant={self.current_best.get('variant_name', '?')}"
        )

    def to_phase_status_summary(
        self,
        *,
        budget_pct: dict[str, float] | None = None,
        now_unix: float | None = None,
    ) -> str:
        """Render the per-tick ``=== Phase ===`` block; compact (≤6 lines, incl. the ``cycle`` = macro-cycle number). EXPLORE adds a ``force_exit`` line showing runway before the hard force-exit gate.

        Args:
            budget_pct (dict[str, float] | None): Per-phase budget fractions;
                defaults to :attr:`phase_budget_pct`.
            now_unix (float | None): Reference Unix time; defaults to now.

        Returns:
            str: The compact ``=== Phase ===`` block.
        """
        from ...phases.machine_state import (
            DEFAULT_EXPLORE_FORCE_EXIT_BUDGET_PCT,
            DEFAULT_EXPLORE_FORCE_EXIT_HOURS_REMAINING,
            PHASE_EXPLORE,
            llm_proposable_actions_for_with_interleave,
            normalize_budget_pct,
            phase_budget_remaining_seconds,
            phase_elapsed_seconds,
            session_remaining_seconds,
        )

        phase = (self.phase or "").strip().upper() or "UNSET"
        elapsed = int(phase_elapsed_seconds(self, now_unix=now_unix))
        budget = normalize_budget_pct(budget_pct or self.phase_budget_pct)
        budget_pct_for_phase = budget.get(phase, 0.0)
        remaining = phase_budget_remaining_seconds(
            self,
            budget_pct=budget,
            now_unix=now_unix,
        )
        budget_line: str
        if remaining is None:
            budget_line = f"budget    : pct={budget_pct_for_phase:.2f} (unlimited run; no per-phase cap)"
        else:
            budget_line = (
                f"budget    : pct={budget_pct_for_phase:.2f} elapsed_sec={elapsed} remaining_sec={int(remaining)}"
            )
        proposable = tuple(
            sorted(
                llm_proposable_actions_for_with_interleave(
                    phase,
                    explore_enabled=bool(getattr(self, "explore_enabled", True)),
                )
            )
        )
        allowed_line = f"allowed   : {', '.join(proposable) if proposable else '(none)'}"
        lines = [
            f"phase     : {phase}",
            f"cycle     : {int(getattr(self, 'macro_cycle', 0) or 0)}",
            f"entered   : {self.phase_started_ts or '(unset)'}",
            budget_line,
            allowed_line,
        ]
        # EXPLORE-only: distance to hard force-exit alongside the soft budget.
        if phase == PHASE_EXPLORE:
            overrides = self.plateau_overrides or {}
            hours_thresh = float(
                overrides.get(
                    "force_exit_hours_remaining",
                    DEFAULT_EXPLORE_FORCE_EXIT_HOURS_REMAINING,
                )
            )
            pct_thresh = float(
                overrides.get(
                    "force_exit_budget_pct",
                    DEFAULT_EXPLORE_FORCE_EXIT_BUDGET_PCT,
                )
            )
            session_remaining = session_remaining_seconds(
                self,
                now_unix=now_unix,
            )
            session_buffer = int(session_remaining - hours_thresh * 3600.0) if session_remaining is not None else None
            if remaining is not None and budget_pct_for_phase > 0:
                mm = float(self.max_minutes or 0)
                phase_total_sec = mm * 60.0 * budget_pct_for_phase
                phase_remaining_pct = remaining / phase_total_sec if phase_total_sec > 0 else 0.0
            else:
                phase_remaining_pct = None
            force_line = f"force_exit: hours_thresh={hours_thresh:.1f}h pct_thresh={pct_thresh:.2f}"
            if session_buffer is not None:
                force_line += f" session_buffer_sec={session_buffer}"
            if phase_remaining_pct is not None:
                force_line += f" phase_remaining_pct={phase_remaining_pct:.3f}"
            lines.append(force_line)
        return "\n".join(lines)

    def to_phase_budget_telemetry(
        self,
        *,
        budget_pct: dict[str, float] | None = None,
        now_unix: float | None = None,
    ) -> str:
        """Render the per-phase budget telemetry block for Robustness (one ``phase: elapsed=Xs cap=Ys (Z%)`` line per phase) so it can spot budget overruns.

        Args:
            budget_pct (dict[str, float] | None): Per-phase budget fractions;
                defaults to :attr:`phase_budget_pct`.
            now_unix (float | None): Reference Unix time; defaults to now.

        Returns:
            str: One telemetry line per phase, or ``"(no phase history yet)"``
                when no history exists.
        """
        from ...phases.machine_state import (
            DEFAULT_PHASE_BUDGET_PCT,
            PHASE_NAMES,
            normalize_budget_pct,
            phase_elapsed_seconds,
        )

        budget = normalize_budget_pct(budget_pct or self.phase_budget_pct)
        # Aggregate elapsed per phase using phase_history.
        elapsed_per_phase: dict[str, float] = {}
        history = self.phase_history or []
        for idx, row in enumerate(history):
            if not isinstance(row, dict):
                continue
            phase = str(row.get("to_phase") or "").upper()
            entered = float(row.get("ts_unix") or 0.0)
            if not phase or entered <= 0:
                continue
            if idx + 1 < len(history) and isinstance(history[idx + 1], dict):
                exited = float(history[idx + 1].get("ts_unix") or entered)
            else:
                # Currently-active segment — measure to now.
                elapsed_now = phase_elapsed_seconds(self, now_unix=now_unix)
                exited = entered + elapsed_now
            elapsed_per_phase[phase] = elapsed_per_phase.get(phase, 0.0) + max(0.0, exited - entered)
        if not elapsed_per_phase:
            return "(no phase history yet)"
        mm = float(self.max_minutes or 0.0)
        total_budget_sec = mm * 60.0
        lines: list[str] = []
        # Stable order — iterate PHASE_NAMES so new phases render automatically.
        for phase in PHASE_NAMES:
            if phase not in elapsed_per_phase:
                continue
            elapsed = elapsed_per_phase[phase]
            pct = budget.get(phase, DEFAULT_PHASE_BUDGET_PCT.get(phase, 0.0))
            cap_sec = total_budget_sec * pct if total_budget_sec > 0 else 0.0
            used_pct = (elapsed / cap_sec * 100.0) if cap_sec > 0 else 0.0
            cap_line = f"cap={int(cap_sec)}s" if cap_sec > 0 else "cap=unlimited"
            lines.append(f"  {phase}: elapsed={int(elapsed)}s {cap_line} used={used_pct:.0f}%")
        return "\n".join(lines) or "(no phase history yet)"

    def to_warm_start_summary(self, *, max_lines: int = 12) -> str:
        """Render T0 warm-start snapshot for the ``=== Warm start ===`` prompt section; empty when no recipe/pitfalls. Capped; full JSON at runtime/cortex/.kb_warm.json / .kb_pitfalls.json.

        Args:
            max_lines (int): Cap on rendered lines before truncation.

        Returns:
            str: The warm-start summary block, or ``""`` when no recipe /
                pitfalls are present.
        """
        recipe = self.warm_start_recipe or {}
        pitfalls = self.warm_start_pitfalls or []
        if not recipe and not pitfalls:
            return ""
        out: list[str] = []
        workload = str(recipe.get("workload") or "") if isinstance(recipe, dict) else ""
        hw = str(recipe.get("hw") or "") if isinstance(recipe, dict) else ""
        if workload or hw:
            out.append(f"recipe: workload={workload or '?'} hw={hw or '?'}")
        raw = str(recipe.get("raw") or "") if isinstance(recipe, dict) else ""
        # Trim recipe raw text — at most 5 lines, 240 chars each.
        if raw.strip():
            kept = 0
            for line in raw.splitlines():
                stripped = line.strip()
                if not stripped:
                    continue
                out.append(f"  · {stripped[:240]}")
                kept += 1
                if kept >= 5:
                    break
            if kept == 0:
                out.append("  · (recipe present but text was empty)")
        else:
            out.append("  · (no recipe text — first session for this workload/hw)")
        if pitfalls:
            out.append(f"pitfalls ({len(pitfalls)}):")
            for entry in pitfalls[:5]:
                if not isinstance(entry, dict):
                    continue
                snippet = str(entry.get("raw") or entry.get("symptom") or "")
                if not snippet.strip():
                    continue
                first_line = snippet.splitlines()[0].strip()
                out.append(f"  · {first_line[:240]}")
        if max_lines and len(out) > max_lines:
            out = out[:max_lines]
            out.append(f"  · (truncated to {max_lines} lines; see runtime/cortex/.kb_warm.json for full snapshot)")
        return "\n".join(out)

    def to_gaps_summary(self, *, max_entries: int = 10) -> str:
        """Render :attr:`gaps` for prompt injection; empty when no gaps. Capped at ``max_entries`` newest rows.

        Args:
            max_entries (int): Maximum number of newest gap rows to render.

        Returns:
            str: The rendered gaps block, or ``""`` when no gaps exist.
        """
        if not self.gaps:
            return ""
        # Newest first by last_updated_ts (deterministic fallback to first_seen_ts/insertion).
        ordered = list(self.gaps)
        ordered.sort(
            key=lambda g: str(
                g.get("last_updated_ts") or g.get("first_seen_ts") or "",
            ),
            reverse=True,
        )
        rows: list[str] = []
        for gap in ordered[:max_entries]:
            if not isinstance(gap, dict):
                continue
            cid = str(gap.get("canonical_id") or "?")
            layer = str(gap.get("layer") or "?")
            severity = str(gap.get("severity") or "?")
            symptom = str(gap.get("symptom") or "").replace("\n", " ").strip()
            if len(symptom) > 200:
                symptom = symptom[:197] + "..."
            attempts = gap.get("attempts") or []
            attempt_n = len(attempts) if isinstance(attempts, list) else 0
            last_tag = ""
            if isinstance(attempts, list) and attempts:
                last = attempts[-1]
                if isinstance(last, dict):
                    last_tag = f" last={last.get('action', '?')}:{last.get('outcome', '?')}"
            rows.append(f"  - {cid} [{layer}/{severity}] {symptom}\n      attempts={attempt_n}{last_tag}")
        if len(ordered) > max_entries:
            rows.append(f"  · (+{len(ordered) - max_entries} older gaps elided; see state.json `gaps[]`)")
        return "\n".join(rows)

    def to_proposal_scores_summary(self, *, max_rounds: int = 2) -> str:
        """Render advisory multi-model proposal scores for Orchestration. NO mean/sorting (Inv-9.1: no system-side scoreboard); rater identities anonymized to avoid brand bias. Empty when no recent round carries scores.

        Args:
            max_rounds (int): Maximum number of recent scored rounds to
                render.

        Returns:
            str: The anonymized proposal-scores block, or ``""`` when no
                recent round carries scores.
        """
        rounds = [
            r
            for r in (self.specialist_rounds or [])
            if isinstance(r, dict)
            and isinstance(r.get("ensemble_scores"), dict)
            and (r["ensemble_scores"].get("models") or {})
        ]
        if not rounds:
            return ""
        shown = rounds[-max_rounds:]
        # Stable, anonymized rater labels: map each real slug to ``rater_N`` (slug never reaches the prompt).
        all_slugs: set[str] = set()
        for r in shown:
            models = r["ensemble_scores"].get("models") or {}
            all_slugs.update(str(s) for s in models.keys())
            errs = r["ensemble_scores"].get("errors") or {}
            all_slugs.update(str(s) for s in errs.keys())
        rater_label = {slug: f"rater_{i}" for i, slug in enumerate(sorted(all_slugs), start=1)}
        rows: list[str] = [
            "(Advisory only — one reference among many, NOT a ranking "
            "directive. Scores are 0-10 likelihood-of-throughput-gain "
            "priors from independent anonymized raters; weigh on merit "
            "alongside gaps / KB / analysis.md.)",
        ]
        for r in shown:
            ens = r["ensemble_scores"]
            models = ens.get("models") or {}
            scale = str(ens.get("scale") or "0-10")
            round_id = str(r.get("round_id") or "?")
            domain = str(r.get("domain") or "?")
            rows.append(f"round={round_id} domain={domain} scale={scale}")
            # Collect variant names across models, preserving proposal_set order when available.
            ordered_names: list[str] = []
            seen: set[str] = set()
            for variant in r.get("proposal_set") or []:
                if isinstance(variant, dict):
                    nm = str(variant.get("name") or "")
                    if nm and nm not in seen:
                        ordered_names.append(nm)
                        seen.add(nm)
            for per_model in models.values():
                if isinstance(per_model, dict):
                    for nm in per_model:
                        if nm not in seen:
                            ordered_names.append(nm)
                            seen.add(nm)
            # Render raters in stable label order so a column means the same model across rounds.
            ordered_slugs = sorted(
                (s for s in models if s in rater_label),
                key=lambda s: rater_label[s],
            )
            for nm in ordered_names:
                parts: list[str] = []
                for model_slug in ordered_slugs:
                    per_model = models.get(model_slug)
                    if not isinstance(per_model, dict):
                        continue
                    label = rater_label[model_slug]
                    cell = per_model.get(nm)
                    if isinstance(cell, dict) and cell.get("score") is not None:
                        reason = str(cell.get("reason") or "").replace("\n", " ")
                        if len(reason) > 80:
                            reason = reason[:77] + "..."
                        parts.append(f'{label}={float(cell["score"]):.1f} ("{reason}")')
                    else:
                        parts.append(f"{label}=n/a")
                rows.append(f"  - {nm}: " + ", ".join(parts))
            errors = ens.get("errors") or {}
            if errors:
                err_labels = ", ".join(sorted(rater_label.get(str(s), "rater_?") for s in errors))
                rows.append(f"  · raters unavailable this round: {err_labels}")
        return "\n".join(rows)

    def to_prompt_summary(self) -> str:
        """Compact, human-readable snapshot for prompt injection.

        Returns:
            str: A multi-line dump of the session's key fact-layer and
                audit fields (baseline / current best / gains / kernel-opt
                queue / attempts history / failures / phase status).
        """
        lines = [
            f"session_id={self.session_id or '(unset)'}",
            f"model={self.model_name or '(unset)'}  class={self.model_class or '(unset)'}",
        ]
        # Advisory architecture profile; prompt-context only (TraceLens analysis_md is ground truth). Omitted when no profile.
        _arch_line = _shared_state_module().render_model_arch_compact(self.model_arch)
        if _arch_line:
            lines.append(f"model_arch(advisory; subordinate to TraceLens analysis_md)={_arch_line}")
        lines += [
            f"baseline_tput={self.baseline_tput}  baseline_acc={self.baseline_accuracy}",
            f"baseline_failure_streak={self.baseline_failure_streak}",
            f"current_best={self.current_best or '(none)'}",
            f"optimization_stack={self._format_optimization_stack()}",
            f"cumulative_gain={self.cumulative_gain}%",
            (
                f"cumulative_gain_validated={self.cumulative_gain_validated}% "
                f"(stack_len_at_validation={self.cumulative_gain_validated_stack_len}, "
                f"ts={self.cumulative_gain_validated_ts or '(never)'})"
            ),
            f"last_sweep={self._format_last_sweep()}",
            f"current_action={self.current_action or '(idle)'}",
            f"crash_count={self.crash_count}",
            f"pruned_families={self.pruned_families or '(none)'}",
            f"last_profile_trace={self.last_profile_trace or '(none)'}",
            f"last_profile_status={self.last_profile_status or '(none)'}",
            f"last_profile_args='{self.last_profile_args}'",
            f"discovered_flags_error={self.discovered_flags_error or '(none)'}",
            f"last_trace_analyze={self._format_last_trace_analyze()}",
            f"profiler_digest={self._format_profiler_digest()}",
            # Full TraceLens analysis.md so the LLM grounds propose_action in the actual report.
            f"analysis_md={self._format_analysis_md_full()}",
            # Streak counter is a readable fact (KEEP/REVERT counts allowed); plateau judges also consume it on legacy resume.
            f"params_no_promote_streak={self.params_no_promote_streak}",
            f"explore_search={self._format_explore_search()}",
            f"discovered_flags={self._format_discovered_flags()}",
            f"backend_winners_history={self._format_backend_winners_history()}",
            f"synergy_attempted={len(self.synergy_attempted)} combos",
            f"last_kernel_opt={self._format_last_kernel_opt()}",
            # Pending KEEPs the integrate gate will drain, plus per-kernel attempt count.
            (f"pending_keep_kernels={self.pending_keep_kernel_ids() or '(none)'}"),
            (f"has_keep_pending_integrate={'true' if self.has_keep_pending_integrate else 'false'}"),
            f"kernel_opt_attempts_count={self.kernel_opt_attempts_count}",
            f"rejected_kernel_patches={self._format_rejected_kernel_patches()}",
            f"rejected_kernel_ids={self.rejected_kernel_ids or '(none)'}",
            f"last_baseline={self._format_attempt(self.last_baseline)}",
            f"last_profile={self._format_attempt(self.last_profile)}",
            f"last_gemm_tuning={self._format_attempt(self.last_gemm_tuning)}",
            f"last_explore={self._format_attempt(self.last_explore)}",
            f"last_sweep={self._format_attempt(self.last_sweep)}",
            f"attempts_history={self._format_attempts_history()}",
            f"last_action_failures={self._format_last_action_failures()}",
            f"tick={int(self.tick or 0)}  target_gap_pct={float(self.target_gap_pct or 0.0):.2f}",
            f"stop_reason={self.stop_reason or '(none)'}",
            f"closing_phase={self.closing_phase}  "
            f"closing_started_unix={self.closing_started_unix or 0.0}  "
            f"closing_report_task_id={self.closing_report_task_id or '(none)'}",
        ]
        return "\n".join(lines)

    # Audit-trail renderers (per-action attempts + global failure log); compact one-liners.
    @staticmethod
    def _format_attempt(entry: dict[str, Any] | None) -> str:
        """Render one ``last_<action>`` snapshot or ``attempts[-1]`` entry.

        Args:
            entry (dict[str, Any] | None): The attempt snapshot to render.

        Returns:
            str: A compact ``status=... decision=... <metric> err=... ws=...``
                line, or ``"(none)"`` when the entry is empty.
        """
        if not isinstance(entry, dict) or not entry:
            return "(none)"
        metric = entry.get("key_metric")
        metric_kind = entry.get("key_metric_kind") or "metric"
        metric_str = f"{metric_kind}={metric:.2f}" if isinstance(metric, (int, float)) else f"{metric_kind}=N/A"
        err = entry.get("error_class") or "-"
        ws = entry.get("workspace") or "-"
        return (
            f"status={entry.get('status', '?')} "
            f"decision={entry.get('decision', '?')} "
            f"{metric_str} err={err} ws={ws} "
            f"task_id={entry.get('task_id', '?')} ts={entry.get('ts', '?')}"
        )

    def _format_attempts_history(self) -> str:
        """One-line summary across the audit actions (``baseline:total(s<succ>,f<fail>) ...``) so the LLM gauges reliability without 6x20 rows.

        Returns:
            str: A per-action totals summary, or ``"(no attempts recorded)"``
                when no attempts exist.
        """
        parts: list[str] = []
        for action in sorted(_shared_state_module()._AUDIT_ACTIONS):
            attempts_attr = f"{action}_attempts"
            history = getattr(self, attempts_attr, None) or []
            if not history:
                continue
            total = len(history)
            succ = sum(1 for e in history if isinstance(e, dict) and e.get("status") == "succeeded")
            fail = sum(1 for e in history if isinstance(e, dict) and e.get("status") == "failed")
            parts.append(f"{action}:{total}(s{succ},f{fail})")
        return " ".join(parts) if parts else "(no attempts recorded)"

    def _format_last_action_failures(self) -> str:
        """Render up to the 3 most-recent global failures (rich-context companion to crash_count/baseline_failure_streak); full list on disk.

        Returns:
            str: A pipe-joined render of the last 3 failures (with an
                ``[+N earlier]`` suffix when more exist), or ``"(none)"``.
        """
        if not self.last_action_failures:
            return "(none)"
        rows: list[str] = []
        for entry in self.last_action_failures[-3:]:
            if not isinstance(entry, dict):
                continue
            action = entry.get("action") or "?"
            error_class = entry.get("error_class") or "?"
            ts = entry.get("ts") or "?"
            excerpt = entry.get("error_excerpt") or ""
            ws = entry.get("workspace") or "-"
            excerpt_short = excerpt.splitlines()[0][:200] if excerpt else ""
            rows.append(f'[{action}/{error_class}@{ts}] err="{excerpt_short}" ws={ws}')
        suffix = f" [+{len(self.last_action_failures) - 3} earlier]" if len(self.last_action_failures) > 3 else ""
        return " | ".join(rows) + suffix if rows else "(none)"

    def _format_rejected_kernel_patches(self) -> str:
        """Render the most recent rejected kernel patches for the prompt.

        Returns:
            str | list[str]: A list of compact per-patch lines (last 5), or
                ``"(none)"`` when no patches have been rejected.
        """
        if not self.rejected_kernel_patches:
            return "(none)"
        return [
            (
                f"{r.get('kernel_id', '?')}: attempts={r.get('attempt_count', '?')} "
                f"best_gain={r.get('best_gain_pct', '?')} reason={r.get('reason', '?')}"
            )
            for r in self.rejected_kernel_patches[-5:]
            if isinstance(r, dict)
        ] or "(none)"

    def _format_discovered_flags(self) -> str:
        """Render the per-framework discovered-flag counts for the prompt.

        Returns:
            str: ``<framework>:backend=N/param=M`` parts joined by commas,
                or a hint string when no flags have been discovered yet.
        """
        if not self.discovered_flags:
            return "(none — first backends/params round will populate)"
        parts: list[str] = []
        for fw, entry in sorted(self.discovered_flags.items()):
            if not isinstance(entry, dict):
                continue
            n_b = len(entry.get("backend_flags") or [])
            n_p = len(entry.get("param_flags") or [])
            parts.append(f"{fw}:backend={n_b}/param={n_p}")
        return ", ".join(parts) or "(none)"

    @staticmethod
    def _format_variant_line(entry: dict[str, Any]) -> str:
        """One-line render of a search variant for prompt blocks.

        Args:
            entry (dict[str, Any]): A search-variant entry (name, gain_pct,
                tput, extra args / envs).

        Returns:
            str: A single fixed-width line summarizing the variant.
        """
        name = str(entry.get("name") or "?")
        gain = entry.get("gain_pct")
        tput = entry.get("tput") or entry.get("output_throughput")
        gain_s = f"{gain:+.2f}%" if isinstance(gain, (int, float)) else " no_meas"
        tput_s = f" (tput={tput:.1f})" if isinstance(tput, (int, float)) and tput > 0 else ""
        args = str(entry.get("extra_server_args") or "").strip() or "(no-flag)"
        envs = entry.get("extra_envs") or {}
        envs_s = " " + " ".join(f"{k}={v}" for k, v in sorted(envs.items())) if envs else ""
        return f"{name:28s} {gain_s:>9}{tput_s}  {args}{envs_s}"

    @staticmethod
    def _enrich_with_tested_gain(
        entry: dict[str, Any],
        tested: dict[str, Any],
    ) -> dict[str, Any]:
        """Backfill ``gain_pct``/``tput`` from the matching ``tested[fp]`` at render time (some accepted entries don't persist gain_pct; avoids a second writer).

        Args:
            entry (dict[str, Any]): The accepted-variant entry to enrich.
            tested (dict[str, Any]): The negative ledger keyed by fingerprint,
                used to backfill missing gain / tput.

        Returns:
            dict[str, Any]: ``entry`` itself when already complete, otherwise
                a copy with ``gain_pct`` / ``tput`` backfilled where possible.
        """
        if entry.get("gain_pct") is not None and entry.get("tput") is not None:
            return entry
        fp = str(entry.get("fingerprint") or "")
        snap = tested.get(fp) if fp else None
        if not isinstance(snap, dict):
            return entry
        out = dict(entry)
        if out.get("gain_pct") is None:
            out["gain_pct"] = snap.get("gain_pct")
        if out.get("tput") is None:
            result = snap.get("result") if isinstance(snap.get("result"), dict) else {}
            out["tput"] = snap.get("tput") or (result or {}).get("output_throughput")
        return out

    def _format_backend_winners_history(self) -> str:
        """Multi-line render of the explore-round winners history (last 5 rounds: per-winner gain_pct/tput/flags); older rounds collapse to an elision line.

        Returns:
            str: The multi-line winners-history render, or
                ``"(no explore rounds completed)"`` when empty.
        """
        if not self.backend_winners_history:
            return "(no explore rounds completed)"
        last = self.backend_winners_history[-5:]
        out: list[str] = [""]
        for r in last:
            if not isinstance(r, dict):
                continue
            best = r.get("best") if isinstance(r.get("best"), dict) else None
            best_gain = best.get("gain_pct") if best else None
            gain_tag = f" {best_gain:+.2f}%" if isinstance(best_gain, (int, float)) else ""
            base = float(r.get("base_tput", 0.0) or 0.0)
            out.append(
                f"    {r.get('round_id', '?')} ({r.get('action', '?')}): "
                f"base_tput={base:.1f}  "
                f"best={(best.get('name') if best else '(none)')}{gain_tag}"
            )
            winners = [w for w in (r.get("winners") or []) if isinstance(w, dict)]
            if not winners:
                out.append("      (no winners this round)")
                continue
            for w in winners:
                out.append("      • " + _RenderMixin._format_variant_line(w))
        if len(self.backend_winners_history) > 5:
            out.append(f"    [+{len(self.backend_winners_history) - 5} earlier rounds elided]")
        return "\n".join(out)

    def _format_explore_search(self) -> str:
        """Render the unified ``explore_search`` ledger for the prompt.

        Returns:
            str: The :meth:`_format_search_state` render of
                :attr:`explore_search`.
        """
        return self._format_search_state(self.explore_search)

    @staticmethod
    def _format_search_state(search: dict[str, Any] | None) -> str:
        """Multi-line render of a ``*_search`` dedup ledger; each entry surfaces real ``gain_pct``. Counts on the head line; bodies show last 5 per bucket (only the prompt body is truncated).

        Args:
            search (dict[str, Any] | None): The search ledger to render.

        Returns:
            str: The multi-line ledger render, or ``"(none)"`` when empty.
        """
        if not search:
            return "(none)"
        accepted = list(search.get("accepted") or [])
        rejected = list(search.get("rejected") or [])
        tested = search.get("tested") or {}
        cursor = search.get("cursor", 0)
        out: list[str] = [
            "",
            f"    cursor={cursor}  accepted={len(accepted)}  rejected={len(rejected)}  tested={len(tested)}",
        ]
        if accepted:
            out.append("    accepted:")
            for entry in accepted[-5:]:
                if not isinstance(entry, dict):
                    continue
                out.append(
                    "      • " + _RenderMixin._format_variant_line(_RenderMixin._enrich_with_tested_gain(entry, tested))
                )
        if rejected:
            out.append("    rejected (last 5):")
            for entry in rejected[-5:]:
                if not isinstance(entry, dict):
                    continue
                out.append("      • " + _RenderMixin._format_variant_line(entry))
        return "\n".join(out)

    def _format_optimization_stack(self) -> str:
        """Render the optimization stack as ``action:variant`` parts.

        Returns:
            str | list[str]: A list of ``action:variant_name`` strings, or
                ``"(none)"`` when the stack is empty.
        """
        if not self.optimization_stack:
            return "(none)"
        parts = []
        for entry in self.optimization_stack:
            if not isinstance(entry, dict):
                continue
            parts.append(f"{entry.get('action', '?')}:{entry.get('variant_name', '?')}")
        return parts or "(none)"

    @staticmethod
    def _strip_base64_data_urls(text: str) -> str:
        """Drop base64 image payloads before prompt injection (in-memory only; on-disk file intact). Delegates to ``hyperloom.inference_optimizer.tracelens_md``.

        Args:
            text (str): The markdown text to scrub of base64 data URLs.

        Returns:
            str: The text with base64 data URLs stripped (``""`` for falsy
                input).
        """
        if not text:
            return text or ""
        from hyperloom.inference_optimizer.tracelens_md import strip_base64_data_urls

        return strip_base64_data_urls(text)

    def _format_analysis_md_full(self) -> str:
        """Inject TraceLens analysis.md verbatim (no truncation/interpretation) between ``=== TraceLens Analysis ... ===`` bookends; header carries snapshot id + gain. Empty cache → one-line hint to propose ``roofline``.

        Returns:
            str: The verbatim analysis.md wrapped in bookends, or a one-line
                hint when no TraceLens snapshot is cached.
        """
        cached = self.last_trace_analyze or {}
        md_text = cached.get("analysis_md_text") or ""
        if not md_text:
            return (
                "(no TraceLens snapshot yet — analysis is auto-enqueued "
                "by the Coordinator at the end of PRELUDE and on every "
                "+10% validated-gain crossing; wait for the pending "
                "task to land, or continue with specialist / explore "
                "work that does not need analysis.md. `roofline` and "
                "`profile` are Coordinator-managed and absent from "
                "`PHASE_LLM_PROPOSABLE_ACTIONS`, so PolicyGate R1 "
                "denies any LLM-emitted propose_action/delegate "
                "against either name with rule `phase_incompatible`.)"
            )
        md_text = self._strip_base64_data_urls(md_text)
        snap = cached.get("roofline_snapshot_id", "?")
        gain = cached.get("roofline_baseline_gain_at_snapshot", 0.0)
        try:
            gain_str = f"{float(gain):.2f}"
        except (TypeError, ValueError):
            gain_str = "?"
        # When inline injection is disabled, surface the structured digest above
        # (profiler_digest=) and point at the show_analysis_md tool for the full
        # report — saves context on long runs. Default keeps the verbatim md.
        if os.getenv(
            "INFERENCE_OPTIMIZER_PROMPT_ANALYSIS_MD_INLINE",
            "1",
        ).strip().lower() in ("0", "false", "off", "no"):
            return (
                f"(TraceLens snapshot #{snap}, gain at snapshot = {gain_str}% — "
                "full report not inlined; see profiler_digest above or call the "
                "show_analysis_md context tool for the verbatim analysis.md.)"
            )
        return (
            f"\n=== TraceLens Analysis (snapshot #{snap}, "
            f"gain at snapshot = {gain_str}%) ===\n"
            f"{md_text}\n"
            f"=== End TraceLens Analysis ===\n"
        )

    def _format_profiler_digest(self) -> str:
        """Compact bottleneck-focused profiler block (saturation mix + cross-snapshot delta + hot kernels + lever); ``(none)`` until a snapshot lands.

        Returns:
            str: The profiler digest block, or ``"(none)"`` until a roofline
                snapshot lands.
        """
        from ...kernel.roofline_snapshot import build_profiler_digest

        digest = build_profiler_digest(
            self.roofline_snapshots,
            self.last_trace_analyze,
        )
        if not digest:
            return "(none)"
        return f"\n{digest}\n"

    def _format_last_trace_analyze(self) -> str:
        """Render the most recent trace-analyze blob for the prompt.

        Returns:
            A formatted summary of ``last_trace_analyze``.
        """
        return self._format_trace_analyze_blob(self.last_trace_analyze)

    def _format_trace_analyze_blob(self, blob: dict[str, Any] | None) -> str:
        """Render a trace-analyze cache blob as a compact prompt line.

        Surfaces the trace input, candidates path, top kernel ids, reusable
        native kernel ids, and any trace-health routing warnings.

        Args:
            blob (dict[str, Any] | None): A ``last_trace_analyze``-shaped
                dict to render.

        Returns:
            str: The compact one-line render, or ``"(none)"`` when the blob
                is empty.
        """
        if not blob:
            return "(none)"
        ids = [
            str(e.get("kernel_id"))
            for e in blob.get("hot_kernels_top15", [])
            if isinstance(e, dict) and e.get("kernel_id")
        ]
        reusable = list(blob.get("reusable_native_kernel_ids", []))
        base = (
            f"trace={blob.get('trace_input', '?')} "
            f"candidates_path={blob.get('candidates_path', '?')} "
            f"top={ids or []} reusable_native={reusable or []}"
        )
        # With no routable candidates, surface skipped operators so the LLM doesn't echo invalid kernel_ids.
        skipped_suffix = ""
        if not ids:
            sk = blob.get("skipped_kernels_top") or []
            rendered_sk = [
                f"{s.get('kernel_id')}:{s.get('name')}:{s.get('skip_reason') or '?'}"
                for s in sk
                if isinstance(s, dict) and s.get("kernel_id")
            ]
            if rendered_sk:
                skipped_suffix = f" skipped_kernels_top=[{'; '.join(rendered_sk)}]"
        # Surface TraceLens routing signals inline so the LLM grounds the next action; omitted in steady-state.
        warnings = blob.get("trace_health_warnings") or []
        if not warnings:
            return base + skipped_suffix
        rendered: list[str] = []
        for w in warnings:
            if not isinstance(w, dict):
                continue
            code = str(w.get("code") or "unknown")
            extras: list[str] = []
            if "idle_pct" in w and "threshold_pct" in w:
                extras.append(f"idle={w['idle_pct']}%")
                extras.append(f"threshold={w['threshold_pct']}%")
            if "returncode" in w:
                extras.append(f"rc={w['returncode']}")
            if extras:
                rendered.append(f"{code}({','.join(extras)})")
            else:
                rendered.append(code)
        return f"{base}{skipped_suffix} warnings=[{'; '.join(rendered)}]"

    def _format_last_sweep(self) -> str:
        """Render the last workload sweep result for the prompt.

        Returns:
            str: ``grid_size=... best=... tput=... conc/isl/osl=...``, or
                ``"(none)"`` when no sweep has run.
        """
        if not self.last_sweep:
            return "(none)"
        best = self.last_sweep.get("best_overall") or {}
        if not best:
            return f"grid_size={self.last_sweep.get('grid_size', 0)} best=(none)"
        return (
            f"grid_size={self.last_sweep.get('grid_size', 0)} "
            f"best={best.get('name', '?')} "
            f"tput={best.get('output_throughput', '?')} "
            f"conc={best.get('conc', '?')} isl={best.get('isl', '?')} osl={best.get('osl', '?')}"
        )

