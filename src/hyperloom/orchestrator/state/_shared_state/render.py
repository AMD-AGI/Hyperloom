# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``_RenderMixin`` — prompt-facing renderers for :class:`..shared_state.SharedState`
(mission / phase / warm-start / search-ledger blocks).
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import PurePosixPath
from typing import Any

from hyperloom.common.prompt_safety import flatten_for_prompt as _flatten_for_prompt


def _shared_state_module():
    """Import parent shared_state lazily to avoid a module-level cycle."""
    from .. import shared_state

    return shared_state


# Failure rows rendered into the prompt, and per-row excerpt budget.
_FAILURES_RENDERED = 10
_FAILURE_EXCERPT_CHARS = 600

# Latency-budget refusals rendered into the prompt. The most recent few, because
# the signal is the trend rather than the roster: what a router needs is whether
# the SLA is still turning down every gain, not the full list.
_LATENCY_REFUSALS_RENDERED = 5

# Width budget for artifact anchors; sized so a full uuid4 fid still fits ws=.
_VARIANT_ANCHOR_MAX_CHARS = 100

# Numeric fields a ``trace_health_warnings[]`` entry may carry, as
# ``(key, label, suffix)`` in render order. The compact warning line is the part
# of the blob an LLM reads first, so a gate's numbers have to reach it: a router
# tells a comm-bound window from a host-bound one by comparing these, not by the
# code alone. Table-driven because the previous per-field ``if`` chain silently
# dropped every number a newly added gate carried -- a gate now registers its
# field here rather than growing another branch.
_WARNING_EXTRA_FIELDS: tuple[tuple[str, str, str], ...] = (
    ("idle_pct", "idle", "%"),
    ("compute_pct", "compute", "%"),
    ("exposed_comm_pct", "exposed_comm", "%"),
    ("threshold_pct", "threshold", "%"),
    ("returncode", "rc", ""),
)


class _RenderMixin:
    def to_policy_denial_summary(self, *, top_k: int = 6) -> str:
        """Forwarding shim — implementation in :mod:`.policy`."""
        from ...policy import gate as _m

        return _m.to_policy_denial_summary(self, top_k=top_k)

    def to_latency_budget_summary(self) -> str:
        """Render the latency constraint and what it has refused (``""`` when off).

        The refusals are why this reaches the prompt at all. Under a budget a
        throughput gain no longer predicts a KEEP, so a router that cannot see
        the refusals reads a flat ``current_best`` as an exhausted configuration
        space and proposes accordingly -- when what is happening is that the SLA
        is turning down every gain on offer, which calls for the opposite move.
        ``orchestration.md`` tells the model to read this; without it the
        instruction pointed at a field no prompt contained.

        Returns:
            str: A short multi-line block, or ``""`` when no budget is set.
        """
        budget = float(getattr(self, "latency_budget_ms", 0.0) or 0.0)
        if budget <= 0:
            return ""
        refusals = [r for r in (getattr(self, "latency_refusals", None) or []) if isinstance(r, dict)]
        lines = [
            f"budget    : {budget:.0f} ms mean end-to-end, enforced on every KEEP",
            "unmeasured: refused (a constraint that was not measured is not satisfied)",
            f"refused   : {len(refusals)} winner(s) so far",
        ]
        for ref in refusals[-_LATENCY_REFUSALS_RENDERED:]:
            observed = ref.get("e2el_mean_ms")
            shown = f"{float(observed):.0f} ms" if isinstance(observed, (int, float)) else "not measured"
            name = _flatten_for_prompt(str(ref.get("variant_name") or "?"))
            action = _flatten_for_prompt(str(ref.get("action") or "?"))
            lines.append(f"  - {name} ({action}): {shown}")
        if refusals:
            lines.append(
                "A list that keeps growing means the SLA is the binding limit, not an "
                "exhausted search space: propose lower-latency shapes, not more of the same."
            )
        return "\n".join(lines)

    def to_intervention_mix_summary(self) -> str:
        """Render the intervention ledger as a one-line counts summary (``""`` when empty).

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
        """Mission-progress block printed at the top of every tick (raw/validated gain, time vs budget, stack staleness).

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
            " ⚠ stack changed since last validation — RUN `explore` (a KEEP is measured on the stack)"
            if unvalidated
            else ""
        )
        resume_revalidation_tag = (
            " ⚠ resume_pending_revalidation=true — recheck current stack before trusting validated gain"
            if bool(getattr(self, "resume_pending_revalidation", False))
            else ""
        )
        geak_pending_status = (
            str(self.geak_pending.get("status") or "") if isinstance(getattr(self, "geak_pending", None), dict) else ""
        )
        geak_revalidation_status = (
            str(self.geak_result.get("revalidation_status") or "")
            if isinstance(getattr(self, "geak_result", None), dict)
            else ""
        )
        geak_in_stack = any(
            isinstance(entry, dict) and str(entry.get("action") or "") == "geak_e2e"
            for entry in (getattr(self, "optimization_stack", None) or [])
        )
        if geak_pending_status == "awaiting_rebench":
            geak_pending_tag = " ⚠ geak candidate awaiting main-flow rebench — NOT in headline until validated"
        elif geak_pending_status in {"rebench_cancelled", "rebench_unavailable"}:
            geak_pending_tag = f" ⚠ geak candidate dropped unvalidated ({geak_pending_status})"
        elif geak_revalidation_status in {"failed", "fallback_failed"} and not geak_in_stack:
            # A fallback rebench that also failed is the same unjudged drop; only
            # ``no_material`` / ``no_promote`` are verdicts and stay silent here.
            # A ``failed`` 2b is never retracted when the 2a fallback then
            # promotes, so confirm the candidate really is out of the stack
            # before calling it dropped.
            geak_pending_tag = f" ⚠ geak candidate dropped unvalidated (rebench_{geak_revalidation_status})"
        else:
            geak_pending_tag = ""
        from hyperloom.inference_optimizer import framework_registry

        lines = [
            f"baseline  : {framework_registry.format_primary_metric(self.framework, self.baseline_tput)}",
            f"current   : {self._format_current_best_for_mission()}",
            f"gain      : validated={self.cumulative_gain_validated:.2f}%{validated_age}",
            f"stack     : {len(self.optimization_stack)} entries "
            f"(validated_at_len={self.cumulative_gain_validated_stack_len})"
            f"{unvalidated_tag}{resume_revalidation_tag}{geak_pending_tag}",
        ]
        # Surface reusable hot kernels still owing a kernel_opt attempt.
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
        """Render the per-tick ``=== Phase ===`` block (≤7 lines). The mid-chain phases add a ``cycle_reloop`` line showing whether another macro-cycle is still affordable.

        Args:
            budget_pct (dict[str, float] | None): Per-phase budget fractions;
                defaults to :attr:`phase_budget_pct`.
            now_unix (float | None): Reference Unix time; defaults to now.

        Returns:
            str: The compact ``=== Phase ===`` block.
        """
        from ...phases.machine_state import (
            PHASE_FRAMEWORK_AGENT,
            PHASE_KERNEL_AGENT,
            PHASE_SWEEP,
            llm_proposable_actions_for,
            normalize_budget_pct,
            phase_budget_remaining_seconds,
            phase_cumulative_seconds,
            phase_elapsed_seconds,
            session_remaining_seconds,
            should_reloop_to_explore,
        )

        phase = (self.phase or "").strip().upper() or "UNSET"
        elapsed = int(phase_elapsed_seconds(self, now_unix=now_unix))
        # ``remaining`` paces this entry; the absolute cap reads ``cumulative``.
        # A re-entered phase needs both to be legible.
        cumulative = int(phase_cumulative_seconds(self, now_unix=now_unix))
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
                f"budget    : pct={budget_pct_for_phase:.2f} elapsed_sec={elapsed} "
                f"cumulative_sec={cumulative} remaining_sec={int(remaining)}"
            )
        proposable = llm_proposable_actions_for(phase)
        allowed_line = f"allowed   : {', '.join(proposable) if proposable else '(none)'}"
        lines = [
            f"phase     : {phase}",
            f"cycle     : {int(getattr(self, 'macro_cycle', 0) or 0)}",
            f"entered   : {self.phase_started_ts or '(unset)'}",
            budget_line,
            allowed_line,
        ]
        # Whether deferring work to a later cycle is still a real option. Mirrors
        # the SWEEP-exit decision, so it is a projection before SWEEP is reached.
        if phase in (PHASE_FRAMEWORK_AGENT, PHASE_KERNEL_AGENT, PHASE_SWEEP):
            reloop, evidence = should_reloop_to_explore(self, now_unix=now_unix)
            feasible = reloop and self.framework_agent_phase_enabled
            reloop_line = f"reloop    : cycle_reloop_feasible={'true' if feasible else 'false'}"
            threshold = evidence.get("min_remaining_sec_effective")
            if threshold is not None:
                reloop_line += f" threshold_sec={int(threshold)}"
            session_remaining = session_remaining_seconds(self, now_unix=now_unix)
            if session_remaining is not None:
                reloop_line += f" session_remaining_sec={int(session_remaining)}"
            blocked = evidence.get("reloop_blocked")
            if blocked:
                reloop_line += f" blocked={blocked}"
            if phase != PHASE_SWEEP:
                reloop_line += " (projected)"
            lines.append(reloop_line)
        return "\n".join(lines)

    def to_phase_budget_telemetry(
        self,
        *,
        budget_pct: dict[str, float] | None = None,
        now_unix: float | None = None,
    ) -> str:
        """Render the per-phase budget telemetry block for Robustness (one ``phase: elapsed=Xs cap=Ys (Z%)`` line per phase).

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
            is_phase_transition_row,
            normalize_budget_pct,
            phase_elapsed_seconds,
        )

        budget = normalize_budget_pct(budget_pct or self.phase_budget_pct)
        # Aggregate elapsed per phase using real transitions only.
        elapsed_per_phase: dict[str, float] = {}
        history = [row for row in (self.phase_history or []) if is_phase_transition_row(row)]
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
        # Iterate PHASE_NAMES for stable order.
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

    def to_resource_pools_summary(self) -> str:
        """Render the GPU pool / lane capacity block.

        These are the same numbers PolicyGate admits a ``needs_gpu`` dispatch
        against, so a request can be judged schedulable before it is emitted.

        Returns:
            str: One line per pool / lane dimension.
        """
        from ...bus.storage.schema import DEFAULT_LANE_CAPACITIES
        from ...policy.gate import (
            _effective_gpu_specialist_pool_size,
            _serving_tp_for_policy,
            _whole_machine_pool_size,
            gpu_specialist_ceiling,
        )

        lines = [
            f"serving_tp={_serving_tp_for_policy(self)}",
            f"gpu_specialist_capacity={gpu_specialist_ceiling(self)}",
            f"serving_disjoint_gpu_pool={_effective_gpu_specialist_pool_size(self)}"
            "  (non-bench needs_gpu specialists admit against this)",
            f"whole_machine_gpu_pool={_whole_machine_pool_size()}"
            "  (bench / framework-authoring specialists admit against this)",
            f"research_lane_capacity={max(0, int(self.research_lane_capacity or 0))}  (concurrent specialists)",
            f"gpu_research_lane_capacity={DEFAULT_LANE_CAPACITIES['gpu_research_lane']}"
            "  (mutually exclusive with serving / benchmark / profile)",
        ]
        return "\n".join(lines)

    def to_warm_start_summary(self, *, max_lines: int = 12) -> str:
        """Render T0 warm-start snapshot for the ``=== Warm start ===`` prompt section; empty when no recipe/pitfalls.

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
        # Trim recipe raw text to at most 5 lines, 240 chars each.
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
            out.append(f"  · (truncated to {max_lines} lines; see runtime/recipe_kb/.kb_warm.json for full snapshot)")
        return "\n".join(out)

    def to_gaps_summary(self, *, max_entries: int = 10, max_attempts: int = 0) -> str:
        """Render :attr:`gaps` for prompt injection; empty when no gaps. Capped at ``max_entries`` newest rows.

        Args:
            max_entries (int): Maximum number of newest gap rows to render.
            max_attempts (int): When > 0, include the ``max_attempts`` most
                recent attempt rows for each gap.  0 (default) shows the count
                and last action only (prompt-compact mode).

        Returns:
            str: The rendered gaps block, or ``""`` when no gaps exist.
        """
        if not self.gaps:
            return ""
        # Newest first by last_updated_ts (fallback to first_seen_ts/insertion).
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
            if max_attempts > 0 and isinstance(attempts, list):
                for a in attempts[-max_attempts:]:
                    if not isinstance(a, dict):
                        continue
                    fid = a.get("failure_id") or ""
                    rows.append(
                        f"        attempt: {a.get('action', '?')} outcome={a.get('outcome', '?')}"
                        f" err={a.get('error_class', '')}" + (f" fid={fid}" if fid else "")
                    )
        if len(ordered) > max_entries:
            rows.append(f"  · (+{len(ordered) - max_entries} older gaps elided; see state.json `gaps[]`)")
        return "\n".join(rows)

    def _untested_proposal_rows(self) -> list[dict[str, Any]]:
        """Executable proposals from this cycle that no explore round has benched.

        Matched on the proposal's own content. The ledger is keyed on the
        variant folded together with whatever removal controls the stack
        carried at the time, so a KEEP that changes those controls mid-round
        would otherwise make everything benched before it look untried.

        Returns:
            Rows ranked by gap severity then recency, each carrying the
            normalized proposal fields plus ``domain`` / ``severity``.
        """
        from hyperloom.common.coerce import to_int

        from ...actions.executors._proposal_identity import (
            controls_of,
            effective_fingerprint,
            is_executable,
            normalize_proposal,
        )

        def content_fingerprint(fields: dict[str, Any]) -> str:
            return effective_fingerprint(fields["extra_args"], fields["extra_envs"], controls=controls_of(fields))

        cycle = to_int(self.macro_cycle, default=0)
        benched = {
            content_fingerprint(normalize_proposal(row))
            for row in ((self.explore_search or {}).get("tested") or {}).values()
            if isinstance(row, dict)
        }
        severity_of = {
            str(g.get("canonical_id") or ""): str(g.get("severity") or "").strip().lower()
            for g in (self.gaps or [])
            if isinstance(g, dict)
        }
        rank = {"high": 3, "medium": 2, "low": 1}

        ranked: list[tuple[int, int, dict[str, Any]]] = []
        seen: set[str] = set()
        for order, entry in enumerate(self.specialist_rounds or []):
            if not isinstance(entry, dict) or to_int(entry.get("cycle"), default=0) != cycle:
                continue
            domain = str(entry.get("domain") or "?").removesuffix("_specialist")
            severity = severity_of.get(str(entry.get("gap_canonical_id") or ""), "")
            task_id = str(entry.get("task_id") or "")[:8]
            for index, proposal in enumerate(entry.get("proposal_set") or []):
                if not isinstance(proposal, dict):
                    continue
                row = normalize_proposal(proposal)
                if not is_executable(row):
                    continue
                fingerprint = content_fingerprint(row)
                if fingerprint in benched or fingerprint in seen:
                    continue
                seen.add(fingerprint)
                row["name"] = row["name"] or f"{domain or 'specialist'}-{task_id}-{index}"
                row["domain"] = domain
                row["severity"] = severity
                ranked.append((rank.get(severity, 0), order, row))
        ranked.sort(key=lambda r: (-r[0], -r[1]))
        return [row for _, _, row in ranked]

    @staticmethod
    def _untested_proposal_line(row: dict[str, Any]) -> str:
        """Render one queue row, marking each field it carries.

        Args:
            row: One row from :meth:`_untested_proposal_rows`.

        Returns:
            A single ``•``-prefixed line.
        """
        parts = [f"• {row['name']} [{row['domain']}·{row['severity'] or 'sev?'}]"]
        if row["atomic"]:
            parts.append("ATOMIC")
        if row["extra_args"]:
            parts.append(f"+args={row['extra_args']}")
        if row["extra_envs"]:
            parts.append("+envs=" + ",".join(f"{k}={v}" for k, v in sorted(row["extra_envs"].items())))
        if row["remove_args"]:
            parts.append("-args=" + " ".join(row["remove_args"]))
        if row["unset_envs"]:
            parts.append("-envs=" + ",".join(row["unset_envs"]))
        if row["args_mode"] == "replace":
            parts.append("mode=replace")
        reason = row["reason"].replace("\n", " ").strip()[:80].rstrip()
        if reason:
            parts.append(f"why={reason}")
        return _flatten_for_prompt(" ".join(parts))

    def to_untested_proposals_summary(self, *, max_entries: int = 12) -> str:
        """Render the specialist proposals still waiting for a benchmark slot.

        Args:
            max_entries (int): Rows to render before collapsing the rest into a
                count.

        Returns:
            str: The rendered queue, or ``""`` when nothing is waiting.
        """
        rows = self._untested_proposal_rows()
        if not rows:
            return ""
        out = [
            "Executable specialist proposals from this cycle that no explore round has benched.",
            "Ranked by gap severity, then most recent. Compose the next `explore` grid from these;",
            "dispatch an ATOMIC entry verbatim as one variant — never split or re-derive its flags.",
            "",
        ]
        out.extend(self._untested_proposal_line(row) for row in rows[:max_entries])
        if len(rows) > max_entries:
            out.append(f"(+{len(rows) - max_entries} more not shown)")
        return "\n".join(out)

    def to_proposal_scores_summary(self, *, max_rounds: int = 2) -> str:
        """Render advisory multi-model proposal scores for Orchestration; no mean/sorting, rater identities anonymized. Empty when no recent round carries scores.

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
        # Map each real slug to an anonymized ``rater_N`` label.
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
            # Collect variant names across models, preserving proposal_set order.
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
            # Render raters in stable label order.
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

    def _format_last_kernel_opt(self) -> str:
        """Render the latest kernel-opt outcome for prompt injection."""
        if not self.last_kernel_opt:
            return "(none)"
        outcome = self.last_kernel_opt
        kernel_id = str(outcome.get("kernel_id") or "")
        attempts_entry = self.kernel_opt_attempts.get(kernel_id) or {}
        history_tag = ""
        if attempts_entry:
            history_tag = (
                f" history=attempts={attempts_entry.get('attempts', 0)}"
                f"/partial={attempts_entry.get('partial_count', 0)}"
            )
            rejected_reason = attempts_entry.get("rejected_reason")
            if rejected_reason:
                history_tag += f"/retired={rejected_reason}"
        return (
            f"kernel_id={kernel_id or '?'} "
            f"decision={outcome.get('decision', '?')} "
            f"speedup={outcome.get('micro_speedup', '?')}{history_tag}"
        )

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
        # Advisory architecture profile; prompt-context only. Omitted when no profile.
        _arch_line = _shared_state_module().render_model_arch_compact(self.model_arch)
        if _arch_line:
            lines.append(f"model_arch(advisory; subordinate to TraceLens analysis_md)={_arch_line}")
        lines += [
            f"baseline_tput={self.baseline_tput}  baseline_acc={self.baseline_accuracy}",
            f"baseline_failure_streak={self.baseline_failure_streak}",
            f"current_best={self.current_best or '(none)'}",
            f"optimization_stack={self._format_optimization_stack()}",
            (
                f"cumulative_gain_validated={self.cumulative_gain_validated}% "
                f"(stack_len_at_validation={self.cumulative_gain_validated_stack_len}, "
                f"ts={self.cumulative_gain_validated_ts or '(never)'})"
            ),
            f"current_action={self.current_action or '(idle)'}",
            f"crash_count={self.crash_count}",
            f"pruned_families={self.pruned_families or '(none)'}",
            f"last_profile_trace={self.last_profile_trace or '(none)'}",
            f"last_profile_status={self.last_profile_status or '(none)'}",
            f"last_profile_args='{self.last_profile_args}'",
            f"discovered_flags_error={self.discovered_flags_error or '(none)'}",
            f"last_trace_analyze={self._format_trace_analyze_blob(self.last_trace_analyze)}",
            f"profiler_digest={self._format_profiler_digest()}",
            # Full TraceLens analysis.md.
            f"analysis_md={self._format_analysis_md_full()}",
            f"params_no_promote_streak={self.params_no_promote_streak}",
            f"explore_search={self._format_search_state(self.explore_search)}",
            f"discovered_flags={self._format_discovered_flags()}",
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
            f"attempts_history={self._format_attempts_history()}",
            f"last_action_failures={self._format_last_action_failures()}",
            f"tick={int(self.tick or 0)}  target_gap_pct={float(self.target_gap_pct or 0.0):.2f}",
            f"macro_cycle={int(self.macro_cycle or 0)}",
            f"stop_reason={self.stop_reason or '(none)'}",
            f"closing_phase={self.closing_phase}  "
            f"closing_started_unix={self.closing_started_unix or 0.0}  "
            f"closing_report_task_id={self.closing_report_task_id or '(none)'}",
        ]
        return "\n".join(lines)

    # Audit-trail renderers (per-action attempts + global failure log).
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
        """One-line summary across the audit actions (``baseline:total(s<succ>,f<fail>) ...``).

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
        """Render the most-recent global failures, each with its excerpt tail and log path.

        Returns:
            str: A newline-separated render of the last
                :data:`_FAILURES_RENDERED` failures (with an
                ``[+N earlier failures]`` suffix when more exist), or
                ``"(none)"``.
        """
        if not self.last_action_failures:
            return "(none)"
        rows: list[str] = []
        for entry in self.last_action_failures[-_FAILURES_RENDERED:]:
            if not isinstance(entry, dict):
                continue
            action = entry.get("action") or "?"
            error_class = entry.get("error_class") or "?"
            ts = entry.get("ts") or "?"
            header = f"[{action}/{error_class}@{ts}]"
            variant = entry.get("variant_name") or ""
            if variant:
                header += f" variant={variant}"
            header += f" ws={entry.get('workspace') or '-'}"
            log_path = entry.get("stderr_log_path") or ""
            if log_path:
                header += f" log={log_path}"
            # stderr_tail holds the actionable end of the blob; excerpt is its head.
            blob = entry.get("stderr_tail") or entry.get("error_excerpt") or ""
            excerpt = blob[-_FAILURE_EXCERPT_CHARS:].strip()
            rows.append(f"{header}\n  {excerpt}" if excerpt else header)
        earlier = len(self.last_action_failures) - _FAILURES_RENDERED
        suffix = f"\n[+{earlier} earlier failures]" if earlier > 0 else ""
        return "\n".join(rows) + suffix if rows else "(none)"

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
                tput, extra args / envs, and — for rejected rows —
                ``error_class`` / ``reason`` / ``wall_clock_ratio_vs_baseline``).

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
        parts: list[str] = []
        error_class = str(entry.get("error_class") or "").strip()
        if error_class:
            parts.append(f"err={error_class}")
        reason = str(entry.get("reason") or "").strip()
        # Threshold rejections are already conveyed by the gain column. Every other
        # reason must survive: a bare ``no_meas`` is indistinguishable from a
        # measured zero gain, so an overtime kill would otherwise read as "the
        # variant helped nothing" instead of "the variant ran too long to be
        # judged" — opposite follow-up moves. The wall-clock ratio qualifies the
        # reason, so it rides alongside it whenever the executor recorded one.
        if reason and reason not in ("not_keep", "gain_below_threshold"):
            ratio = entry.get("wall_clock_ratio_vs_baseline")
            ratio_s = f" {ratio:.2f}x" if isinstance(ratio, (int, float)) and ratio > 0 else ""
            # error_excerpt on variant rows is tail-1200 (boot assertion at end);
            # flatten+defang prevents section-header injection from untrusted log text.
            body = _flatten_for_prompt(str(entry.get("error_excerpt") or reason))
            parts.append(f"reason={body[-120:]}{ratio_s}")

        # Paths show last two segments; full path available via get_failure(fid).
        # Progressive degradation: drop log=, then ws=, keeping fid= alone.
        fid = str(entry.get("failure_id") or "").strip()
        ws_raw = str(entry.get("workspace") or "").strip()
        log_raw = str(entry.get("server_log_path") or "").strip()
        ws_seg = "/".join(PurePosixPath(ws_raw).parts[-2:]) if ws_raw else ""
        log_seg = "/".join(PurePosixPath(log_raw).parts[-2:]) if log_raw else ""
        anchors = (
            ([f"fid={fid}"] if fid else [])
            + ([f"ws={ws_seg}"] if ws_seg else [])
            + ([f"log={log_seg}"] if log_seg else [])
        )
        anchor_s = ("  " + " ".join(anchors)) if anchors else ""
        if len(anchor_s) > _VARIANT_ANCHOR_MAX_CHARS and log_seg:
            anchors = [a for a in anchors if not a.startswith("log=")]
            anchor_s = ("  " + " ".join(anchors)) if anchors else ""
        if len(anchor_s) > _VARIANT_ANCHOR_MAX_CHARS and ws_seg:
            anchors = [a for a in anchors if not a.startswith("ws=")]
            anchor_s = ("  " + " ".join(anchors)) if anchors else ""

        suffix = "  " + " ".join(parts) if parts else ""
        return f"{name:28s} {gain_s:>9}{tput_s}  {args}{envs_s}{suffix}{anchor_s}"

    @staticmethod
    def _enrich_with_tested_gain(
        entry: dict[str, Any],
        tested: dict[str, Any],
    ) -> dict[str, Any]:
        """Backfill ``gain_pct``/``tput`` from the matching ``tested[fp]`` at render time.

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

    @staticmethod
    def _format_search_state(search: dict[str, Any] | None) -> str:
        """Multi-line render of a ``*_search`` dedup ledger; counts on the head line, bodies show the last 5 accepted / 15 rejected.

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
        head = f"    cursor={cursor}  accepted={len(accepted)}  rejected={len(rejected)}  tested={len(tested)}"
        # Surfaced on the head line because the per-variant bodies are capped,
        # which can hide a whole round reaped by the overtime gate.
        last_round = search.get("last_round")
        n_killed = len((last_round or {}).get("killed_overtime") or []) if isinstance(last_round, dict) else 0
        if n_killed:
            head += f"  killed_overtime(last_round)={n_killed}"
        out: list[str] = ["", head]
        if accepted:
            out.append("    accepted:")
            for entry in accepted[-5:]:
                if not isinstance(entry, dict):
                    continue
                out.append(
                    "      • " + _RenderMixin._format_variant_line(_RenderMixin._enrich_with_tested_gain(entry, tested))
                )
        if rejected:
            out.append("    rejected (last 15):")
            for entry in rejected[-15:]:
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
        """Drop base64 image payloads before prompt injection (in-memory only). Delegates to ``hyperloom.inference_optimizer.tracelens_md``.

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
        """Inject TraceLens analysis.md verbatim between ``=== TraceLens Analysis ... ===`` bookends. Empty cache → one-line hint to propose ``roofline``.

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
        # By default point at the show_analysis_md tool; set
        # INFERENCE_OPTIMIZER_PROMPT_ANALYSIS_MD_INLINE=1 to inline the verbatim md.
        if os.getenv(
            "INFERENCE_OPTIMIZER_PROMPT_ANALYSIS_MD_INLINE",
            "0",
        ).strip().lower() not in ("1", "true", "on", "yes"):
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
        """Compact bottleneck-focused profiler block; ``(none)`` until a snapshot lands.

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
        # With no routable candidates, surface skipped operators.
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
        # Surface TraceLens routing signals inline; omitted in steady-state.
        warnings = blob.get("trace_health_warnings") or []
        if not warnings:
            return base + skipped_suffix
        rendered: list[str] = []
        for w in warnings:
            if not isinstance(w, dict):
                continue
            code = str(w.get("code") or "unknown")
            extras: list[str] = [
                f"{label}={w[key]}{suffix}" for key, label, suffix in _WARNING_EXTRA_FIELDS if key in w
            ]
            if extras:
                rendered.append(f"{code}({','.join(extras)})")
            else:
                rendered.append(code)
        return f"{base}{skipped_suffix} warnings=[{'; '.join(rendered)}]"
