# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``_RenderMixin`` — prompt-facing renderers for :class:`..shared_state.SharedState` (mission / phase / warm-start / search-ledger blocks)."""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any, NamedTuple

from hyperloom.common.coerce import to_float
from hyperloom.common.prompt_safety import flatten_for_prompt as _flatten_for_prompt
from hyperloom.inference_optimizer.breakdown.recorder.kernel_records import GEAK_DROPPED_STATUSES
from hyperloom.inference_optimizer.grid_server_args import dedupe_extra_server_args
from hyperloom.inference_optimizer.session.geak_rebench_verdict import REBENCH_FAILED_STATUSES
from ...mcp_tool_names import qualified_context_tool_name
from . import gap_rows, proposal_scores, variant_line, warm_start
from .attempt_audit import _AUDIT_ACTIONS
from .phase_state import GAP_SEVERITY_RANK

# Failure rows rendered into the prompt, and per-row excerpt budget.
_FAILURES_RENDERED = 10
_FAILURE_EXCERPT_CHARS = 600

_WARM_START_MAX_LINES = 12
_GAPS_MAX_ENTRIES = 10
_UNTESTED_PROPOSALS_MAX_ENTRIES = 12
_PROPOSAL_SCORE_ROUNDS = 2

# Ordered (key, label) projection for advisory ``model_arch``; empty/None keys dropped.
_MODEL_ARCH_STRUCTURED_FIELDS: tuple[tuple[str, str], ...] = (
    ("decoder_type", "decoder"),
    ("attention", "attention"),
    ("layer_mix", "layers"),
    ("kv_cache_per_token", "kv/token"),
    ("active_params", "params"),
    ("num_experts", "experts"),
    ("experts_per_tok", "experts/tok"),
    ("mtp", "mtp"),
    ("swa_window", "swa_window"),
    ("norm", "norm"),
)


def render_model_arch_compact(arch: dict | None) -> str:
    """Render the advisory ``model_arch`` profile as a single compact line (``\"\"`` when empty/not a dict)."""
    if not isinstance(arch, dict) or not arch:
        return ""
    parts: list[str] = []
    for key, label in _MODEL_ARCH_STRUCTURED_FIELDS:
        val = arch.get(key)
        if val is None or val == "":
            continue
        parts.append(f"{label}={val}")
    notes = str(arch.get("notes") or "").strip()
    if notes:
        parts.append(f"notes={notes}")
    return "; ".join(parts)


# Numeric fields a ``trace_health_warnings[]`` entry may carry, as ``(key, label, suffix)`` in render order.
_WARNING_EXTRA_FIELDS: tuple[tuple[str, str, str], ...] = (
    ("idle_pct", "idle", "%"),
    ("compute_pct", "compute", "%"),
    ("exposed_comm_pct", "exposed_comm", "%"),
    ("threshold_pct", "threshold", "%"),
    ("returncode", "rc", ""),
)


class _ProposalRound(NamedTuple):
    """The attribution a specialist round stamps onto every proposal it produced."""

    domain: str
    severity: str
    priority: int
    task_id: str
    provenance: str
    kb_read_id: str
    kb_rendered_refs: list[str]


def _round_attribution(entry: dict[str, Any], severity_of: dict[str, str]) -> _ProposalRound:
    """Read one specialist round's attribution. ``provenance`` defaults to its domain: a proposal cannot choose it."""
    from hyperloom.common.coerce import to_int

    domain = str(entry.get("domain") or "?").removesuffix("_specialist")
    return _ProposalRound(
        domain=domain,
        severity=severity_of.get(str(entry.get("gap_canonical_id") or ""), ""),
        priority=to_int(entry.get("priority"), default=0),
        task_id=str(entry.get("task_id") or "")[:8],
        provenance=str(entry.get("provenance") or f"specialist:{domain}"),
        kb_read_id=str(entry.get("kb_read_id") or ""),
        kb_rendered_refs=list(entry.get("kb_rendered_refs") or []),
    )


def _stamp_round_attribution(
    row: dict[str, Any],
    proposal: dict[str, Any],
    index: int,
    attribution: _ProposalRound,
) -> None:
    """Stamp a normalized proposal with the identity its round gives it, naming it when it named itself nothing."""
    row["name"] = row["name"] or f"{attribution.domain or 'specialist'}-{attribution.task_id}-{index}"
    row["domain"] = attribution.domain
    row["severity"] = attribution.severity
    row["provenance"] = attribution.provenance
    # Already checked against the read this round's dispatch was shown, which travels with them.
    row["experience_citations"] = list(proposal.get("experience_citations") or [])
    row["kb_read_id"] = attribution.kb_read_id
    row["kb_rendered_refs"] = list(attribution.kb_rendered_refs)


def _skipped_kernels_suffix(blob: dict[str, Any]) -> str:
    """The operators TraceLens declined to route, surfaced only when no candidate is routable."""
    rendered = [
        f"{s.get('kernel_id')}:{s.get('name')}:{s.get('skip_reason') or '?'}"
        for s in blob.get("skipped_kernels_top") or []
        if isinstance(s, dict) and s.get("kernel_id")
    ]
    if not rendered:
        return ""
    return f" skipped_kernels_top=[{'; '.join(rendered)}]"


def _trace_health_warnings_suffix(blob: dict[str, Any]) -> str:
    """The TraceLens routing signals on this trace, inline; omitted in steady-state."""
    warnings = blob.get("trace_health_warnings") or []
    if not warnings:
        return ""
    rendered: list[str] = []
    for w in warnings:
        if not isinstance(w, dict):
            continue
        code = str(w.get("code") or "unknown")
        extras = [f"{label}={w[key]}{suffix}" for key, label, suffix in _WARNING_EXTRA_FIELDS if key in w]
        rendered.append(f"{code}({','.join(extras)})" if extras else code)
    return f" warnings=[{'; '.join(rendered)}]"


def untested_proposal_line(row: dict[str, Any], *, why_chars: int | None = 80) -> str:
    """Render one ``untested_proposal_rows`` row, marking each field it carries.

    ``why_chars`` bounds the reason; ``None`` shows a reason its producer has already bounded in full.
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
    reason = row["reason"].replace("\n", " ").strip()[:why_chars].rstrip()
    if reason:
        parts.append(f"why={reason}")
    return _flatten_for_prompt(" ".join(parts))


class _RenderMixin:
    def to_policy_denial_summary(self, *, top_k: int = 6) -> str:
        """Render the most recent PolicyGate denials for prompt injection.

        Args:
            top_k (int): Maximum number of newest denial rows to render.

        Returns:
            str: A ``=== Recent policy denials ===`` block, or ``""`` when
                no denials have been recorded.
        """
        if not self.policy_denial_history:
            return ""
        rows = list(self.policy_denial_history)[-top_k:]
        lines = [f"=== Recent policy denials (newest last, total={len(self.policy_denial_history)}) ==="]
        for r in rows:
            lines.append(
                f"  tick={r.get('tick')} action={r.get('action_name')!r} "
                f"rule={r.get('rule')!r} streak={r.get('streak')} "
                f"hint={str(r.get('hint') or '')[:140]!r}"
            )
        return "\n".join(lines)

    def to_intervention_mix_summary(self) -> str:
        """Render the intervention ledger as a one-line counts summary (``\"\"`` when empty)."""
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
        consec = 0
        for m in reversed(mix):
            change_type = (m or {}).get("change_type")
            if change_type == "code_patch":
                break
            if change_type == "config":
                consec += 1
        return (
            f"config_keeps={n_config} config_attempts={n_config_attempt} "
            f"code_patch_keeps={n_patch} code_patch_attempts={n_patch_attempt} "
            f"consecutive_config_only_rounds={consec}"
        )

    def to_mission_summary(self, *, now: datetime | None = None) -> str:
        """Mission-progress block of every tick's turn section (raw/validated gain, time vs budget, stack staleness)."""
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
        elif geak_pending_status in GEAK_DROPPED_STATUSES:
            geak_pending_tag = f" ⚠ geak candidate dropped unvalidated ({geak_pending_status})"
        elif geak_revalidation_status in REBENCH_FAILED_STATUSES and not geak_in_stack:
            # A fallback rebench that also failed is the same unjudged drop; only ``no_material`` / ``no_promote`` are
            # verdicts and stay silent here.
            geak_pending_tag = f" ⚠ geak candidate dropped unvalidated (rebench_{geak_revalidation_status})"
        else:
            geak_pending_tag = ""
        from hyperloom.inference_optimizer import framework_registry
        from ...kernel import kernel_decisions

        lines = [
            f"baseline  : {framework_registry.format_primary_metric(self.framework, self.baseline_tput)}",
            f"current   : {self._format_current_best_for_mission()}",
            f"gain      : validated={self.cumulative_gain_validated:.2f}%{validated_age}",
            f"stack     : {len(self.optimization_stack)} entries "
            f"(validated_at_len={self.cumulative_gain_validated_stack_len})"
            f"{unvalidated_tag}{geak_pending_tag}",
        ]
        # Surface reusable hot kernels no backend has produced a patch for yet.
        if untried_hot := kernel_decisions.untried_hot_reusable_kernels(self):
            lines.append(f"untried_hot_kernels: {', '.join(untried_hot)}")
        lines.append(budget_line)
        return "\n".join(lines)

    def _format_current_best_for_mission(self) -> str:
        """Render the ``current_best`` one-liner for the mission summary."""
        if not self.current_best:
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

    def to_latency_budget_summary(self) -> str:
        """One line stating the latency budget; empty when unset. Refusals surface in the existing ledgers."""
        budget = float(getattr(self, "latency_budget_ms", 0.0))
        if budget <= 0:
            return ""
        return (
            f"{budget:g} ms mean end-to-end: a KEEP over it, or with no latency reported, is refused "
            "(reason latency_budget_exceeded / latency_unmeasured in explore_search and the journal)"
        )

    def to_constraint_sections(self) -> list[str]:
        """A header and the summary line for each session budget that is set, latency first."""
        sections: list[str] = []
        for title, block in (("Latency", self.to_latency_budget_summary()), ("Power", self.to_power_budget_summary())):
            if block:
                sections += [f"=== {title} budget (constraint) ===", block]
        return sections

    def to_power_budget_summary(self) -> str:
        """One line stating the power budgets and the cap the cards are at; empty when no budget is set."""
        budget = float(getattr(self, "power_budget_w", 0.0) or 0.0)
        per_gpu = dict(getattr(self, "power_budget_per_gpu_w", None) or {})
        if budget <= 0 and not per_gpu:
            return ""
        caps = sorted(
            {
                float(row["power_cap_w"])
                for row in ((getattr(self, "gpu_power_settings", None) or {}).get("observed") or {}).values()
                if isinstance(row, dict) and isinstance(row.get("power_cap_w"), (int, float))
            }
        )
        parts = []
        if budget > 0:
            parts.append(f"{budget:g} W total across the serving GPUs")
        if per_gpu:
            parts.append(
                "per-GPU " + ", ".join(f"GPU {g} {w:g} W" for g, w in sorted(per_gpu.items(), key=lambda i: int(i[0])))
            )
        cap = f", cards capped at {'/'.join(f'{c:g}' for c in caps)} W" if caps else ""
        return (
            f"{'; '.join(parts)}{cap}: a KEEP over a budget, or with no per-GPU power reported, is refused (reason "
            "power_budget_exceeded / gpu_power_budget_exceeded / power_unmeasured in explore_search and the journal)"
        )

    def to_warm_start_summary(self) -> str:
        """Render the ``=== Warm start ===`` prompt section from the T0 warm-start context.

        ``recipe_kb_t0._build_warm_start_context`` already computes the
        model-facing view on every anchor and persists it, so this renders that
        rather than re-deriving a second one off the raw row. The three states
        come from its ``status``; ``match`` supplies tier and confidence, and
        ``recommended_replay`` the config — already split into args and envs, and
        attributed to its donor, which the row itself cannot tell you.

        Empty when T0 never ran (``--degraded-kb``, or a resume from before the
        context existed).
        """
        ctx = self.warm_start_context or {}
        if not isinstance(ctx, dict) or not ctx:
            return ""
        status = str(ctx.get("status") or "").strip()
        if status in warm_start.FIRST_SESSION_LINES:
            return warm_start.FIRST_SESSION_LINES[status]
        head = [f"recipe: {status or 'hit'}", *warm_start.match_facts(ctx.get("match"))]
        out: list[str] = [" ".join(head)]
        out.extend(warm_start.replay_lines(ctx))
        priors = warm_start.priors_line(ctx)
        if priors:
            out.append(priors)
        out.extend(warm_start.pitfall_lines(ctx))
        if len(out) > _WARM_START_MAX_LINES:
            out = out[:_WARM_START_MAX_LINES]
            out.append(f"  · (truncated to {_WARM_START_MAX_LINES} lines)")
        return "\n".join(out)

    def to_gaps_summary(self, *, max_attempts: int = 0) -> str:
        """Render :attr:`gaps` for prompt injection; empty when no gaps. Capped at the newest ``_GAPS_MAX_ENTRIES`` rows."""
        if not self.gaps:
            return ""
        ordered = list(self.gaps)
        ordered.sort(key=gap_rows.recency_key, reverse=True)
        rows: list[str] = []
        for gap in ordered[:_GAPS_MAX_ENTRIES]:
            if not isinstance(gap, dict):
                continue
            rows.append(gap_rows.summary_row(gap))
            rows.extend(gap_rows.attempt_rows(gap, max_attempts))
        if len(ordered) > _GAPS_MAX_ENTRIES:
            rows.append(f"  · (+{len(ordered) - _GAPS_MAX_ENTRIES} older gaps elided; see state.json `gaps[]`)")
        return "\n".join(rows)

    def benched_fingerprints(self) -> set[str]:
        """Content fingerprints of every variant an explore round has benched."""
        from ...actions.executors._proposal_identity import content_fingerprint

        return {
            content_fingerprint(row)
            for row in ((self.explore_search or {}).get("tested") or {}).values()
            if isinstance(row, dict)
        }

    def untested_proposal_rows(self) -> list[dict[str, Any]]:
        """Executable proposals from this cycle that no explore round has benched.

        Ordered by round ``priority`` (predictor rounds set one, specialist rounds
        do not), then gap severity, then most recent. A row's ``provenance`` is its
        round's, ``specialist:<domain>`` unless the round's recorder set one; a
        proposal cannot choose it, since a specialist writes its own proposals.
        """
        from hyperloom.common.coerce import to_int

        from ...actions.executors._proposal_identity import content_fingerprint, is_executable, normalize_proposal

        cycle = to_int(self.macro_cycle, default=0)
        benched = self.benched_fingerprints()
        severity_of = {
            str(g.get("canonical_id") or ""): str(g.get("severity") or "").strip().lower()
            for g in (self.gaps or [])
            if isinstance(g, dict)
        }
        ranked: list[tuple[int, int, int, dict[str, Any]]] = []
        seen: set[str] = set()
        for order, entry in enumerate(self.specialist_rounds or []):
            if not isinstance(entry, dict) or to_int(entry.get("cycle"), default=0) != cycle:
                continue
            attribution = _round_attribution(entry, severity_of)
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
                row["fingerprint"] = fingerprint
                _stamp_round_attribution(row, proposal, index, attribution)
                ranked.append((attribution.priority, GAP_SEVERITY_RANK.get(attribution.severity, 0), order, row))
        ranked.sort(key=lambda r: (-r[0], -r[1], -r[2]))
        return [row for _, _, _, row in ranked]

    def to_untested_proposals_summary(self, *, skip_provenance: str = "") -> str:
        """Render the specialist proposals still waiting for a benchmark slot.

        Args:
            skip_provenance: Leave out the rows a producer with its own section renders.
        """
        rows = [
            row for row in self.untested_proposal_rows() if not skip_provenance or row["provenance"] != skip_provenance
        ]
        if not rows:
            return ""
        out = [
            "Executable specialist proposals from this cycle that no explore round has benched.",
            "Ranked by gap severity, then most recent. The Coordinator benches the head of this",
            "queue whenever no explore is queued or running; dispatch `explore` only for variants not listed here.",
            "",
        ]
        out.extend(untested_proposal_line(row) for row in rows[:_UNTESTED_PROPOSALS_MAX_ENTRIES])
        if len(rows) > _UNTESTED_PROPOSALS_MAX_ENTRIES:
            out.append(f"(+{len(rows) - _UNTESTED_PROPOSALS_MAX_ENTRIES} more not shown)")
        return "\n".join(out)

    def to_proposal_scores_summary(self) -> str:
        """Render advisory multi-model proposal scores for Orchestration; no mean/sorting, rater identities anonymized. Empty when no recent round carries scores."""
        rounds = [
            r
            for r in (self.specialist_rounds or [])
            if isinstance(r, dict)
            and isinstance(r.get("ensemble_scores"), dict)
            and (r["ensemble_scores"].get("models") or {})
        ]
        if not rounds:
            return ""
        shown = rounds[-_PROPOSAL_SCORE_ROUNDS:]
        labels = proposal_scores.rater_labels(shown)
        rows: list[str] = [
            "(Advisory only — one reference among many, NOT a ranking "
            "directive. Scores are 0-10 likelihood-of-throughput-gain "
            "priors from independent anonymized raters; weigh on merit "
            "alongside gaps / KB / analysis.md.)",
        ]
        for round_row in shown:
            rows.extend(proposal_scores.round_rows(round_row, labels))
        return "\n".join(rows)

    def to_prompt_summary(self) -> str:
        """Compact, human-readable snapshot for prompt injection."""
        from ...kernel import kernel_decisions

        lines = [
            f"session_id={self.session_id or '(unset)'}",
            f"model={self.model_name or '(unset)'}  class={self.model_class or '(unset)'}",
        ]
        # Advisory architecture profile; prompt-context only. Omitted when no profile.
        _arch_line = render_model_arch_compact(self.model_arch)
        if _arch_line:
            lines.append(f"model_arch(advisory; subordinate to TraceLens analysis_md)={_arch_line}")
        lines += [
            f"baseline_tput={self.baseline_tput}  baseline_acc={self.baseline_accuracy}",
            f"baseline_failure_streak={self.baseline_failure_streak}",
            f"current_best={self._format_current_best()}",
            f"current_launch_flags(already passed by the running stack)={self._format_current_launch_flags()}",
            f"optimization_stack={self._format_optimization_stack()}",
            (
                f"cumulative_gain_validated={self.cumulative_gain_validated}% "
                f"(stack_len_at_validation={self.cumulative_gain_validated_stack_len}, "
                f"ts={self.cumulative_gain_validated_ts or '(never)'})"
            ),
            f"crash_count={self.crash_count}",
            f"pruned_families={self.pruned_families or '(none)'}",
            f"last_profile_trace={self.last_profile_trace or '(none)'}",
            f"last_profile_status={self.last_profile_status or '(none)'}",
            f"last_profile_args='{self.last_profile_args}'",
            f"last_trace_analyze={self._format_trace_analyze_blob(self.last_trace_analyze)}",
            f"profiler_digest={self._format_profiler_digest()}",
            # Full TraceLens analysis.md.
            f"analysis_md={self._format_analysis_md_pointer()}",
            f"params_no_promote_streak={self.params_no_promote_streak}",
            f"explore_search={self._format_search_state(self.explore_search)}",
            # Pending KEEPs the integrate gate will drain.
            (f"pending_keep_kernels={kernel_decisions.pending_keep_kernel_ids(self) or '(none)'}"),
            (f"has_keep_pending_integrate={'true' if kernel_decisions.has_keep_pending_integrate(self) else 'false'}"),
            f"rejected_kernel_patches={self._format_rejected_kernel_patches()}",
            f"rejected_kernel_ids={self.rejected_kernel_ids or '(none)'}",
            f"last_baseline={self._format_attempt(self.last_baseline)}",
            f"last_profile={self._format_attempt(self.last_profile)}",
            f"last_gemm_tuning={self._format_attempt(self.last_gemm_tuning)}",
            f"last_explore={self._format_attempt(self.last_explore)}",
            f"attempts_history={self._format_attempts_history()}",
            f"last_action_failures={self._format_last_action_failures()}",
            f"gain_gated_action_count={int(self.gain_gated_action_count or 0)}",
            f"macro_cycle={int(self.macro_cycle or 0)}",
            f"stop_reason={self.stop_reason or '(none)'}",
            f"closing_phase={self.closing_phase}  "
            f"closing_started_unix={self.closing_started_unix or 0.0}  "
            f"closing_report_task_id={self.closing_report_task_id or '(none)'}",
        ]
        return "\n".join(lines)

    def to_tick_summary(self) -> str:
        """The per-tick counters of the prompt summary: the tick number and each agent's last pass."""
        return f"tick={int(self.tick or 0)}\nagent_last_active={self._format_agent_last_active()}"

    def _format_agent_last_active(self) -> str:
        """Render each agent's last completed reactor pass as a relative age.

        Returns:
            str: ``orchestration=2s ago, critic=45s ago``, or ``(none)`` when no
                agent has run yet.
        """
        now = time.time()
        parts = [
            f"{agent}={int(max(0.0, now - ts))}s ago"
            for agent, ts in sorted((self.agent_last_active or {}).items())
            if isinstance(ts, (int, float)) and ts > 0
        ]
        return ", ".join(parts) or "(none)"

    # Audit-trail renderers (per-action attempts + global failure log).
    @staticmethod
    def _format_attempt(entry: dict[str, Any] | None) -> str:
        """Render one ``last_<action>`` snapshot or ``attempts[-1]`` entry."""
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
        """One-line summary across the audit actions (``baseline:total(s<succ>,f<fail>) ...``)."""
        parts: list[str] = []
        for action in sorted(_AUDIT_ACTIONS):
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
        """Render the most-recent global failures, each with its excerpt tail and log path."""
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
        """Render the most recent rejected kernel patches for the prompt."""
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

    @staticmethod
    def _format_variant_line(entry: dict[str, Any]) -> str:
        """One-line render of a search variant for prompt blocks."""
        name = str(entry.get("name") or "?")
        return (
            f"{name:28s} {variant_line.format_measurement(entry)}"
            f"  {variant_line.format_config(entry)}"
            f"{variant_line.format_diagnosis(entry)}"
            f"{variant_line.format_anchors(entry)}"
        )

    @staticmethod
    def _format_search_state(search: dict[str, Any] | None) -> str:
        """Multi-line render of a ``*_search`` dedup ledger.

        The head line carries the counts; the body enumerates every ``tested``
        entry, KEEPs first, with no truncation — a variant missing from the
        ledger reads as untried and gets re-proposed.

        Args:
            search (dict[str, Any] | None): The search ledger to render.

        Returns:
            str: The multi-line ledger render, or ``"(none)"`` when empty.
        """
        if not search:
            return "(none)"
        tested: dict[str, Any] = search.get("tested") or {}
        head = (
            f"    cursor={search.get('cursor', 0)}"
            f"  accepted={len(search.get('accepted') or [])}"
            f"  rejected={len(search.get('rejected') or [])}"
            f"  tested={len(tested)}"
        )
        out: list[str] = ["", head]
        rows = [(fp, e) for fp, e in tested.items() if isinstance(e, dict)]
        if not rows:
            return "\n".join(out)
        rows.sort(key=lambda row: str(row[1].get("outcome") or "").upper() != "KEEP")
        out.append("    tested:")
        for fp, entry in rows:
            outcome = str(entry.get("outcome") or "?").upper()
            out.append(f"      {str(fp)[:16]} {outcome:7s} {_RenderMixin._format_variant_line(entry)}")
        return "\n".join(out)

    def _format_current_best(self) -> str:
        """Render ``current_best`` without its ``optimization_stack`` and ``measurement``.

        The stack, which grows with every stacked KEEP, has its own line; the
        measurement is launch evidence (paths, identity hashes) no decision reads.
        """
        if not self.current_best:
            return "(none)"
        return str({k: v for k, v in self.current_best.items() if k not in ("optimization_stack", "measurement")})

    def _format_current_launch_flags(self) -> str:
        """The server flags ``current_best`` launched with, as its measurement recorded them."""
        measurement = getattr(self, "current_best_measurement", None)
        evidence = measurement.get("launch_evidence") if isinstance(measurement, dict) else None
        flags = str(evidence.get("requested_server_args") or "") if isinstance(evidence, dict) else ""
        return _flatten_for_prompt(dedupe_extra_server_args(flags.strip())) or "(unknown)"

    def _format_optimization_stack(self) -> str:
        """Render the optimization stack as ``action:variant`` parts."""
        if not self.optimization_stack:
            return "(none)"
        parts = []
        for entry in self.optimization_stack:
            if not isinstance(entry, dict):
                continue
            parts.append(f"{entry.get('action', '?')}:{entry.get('variant_name', '?')}")
        return parts or "(none)"

    def _format_analysis_md_pointer(self) -> str:
        """Point the prompt at the ``show_analysis_md`` tool once a TraceLens snapshot exists. Empty cache → one-line hint to propose ``roofline``."""
        cached = self.last_trace_analyze or {}
        if not cached.get("analysis_md_text"):
            return (
                "(no TraceLens snapshot yet — analysis is auto-enqueued "
                "by the Coordinator at the end of PRELUDE and on every "
                "+10% validated-gain crossing; wait for the pending "
                "task to land, or continue with specialist / explore "
                "work that does not need analysis.md. `roofline` and "
                "`profile` are Coordinator-managed; you may also propose "
                "them directly if the situation warrants it.)"
            )
        snap = cached.get("roofline_snapshot_id", "?")
        gain = to_float(cached.get("roofline_baseline_gain_at_snapshot", 0.0))
        gain_str = "?" if gain is None else f"{gain:.2f}"
        show_analysis_md = qualified_context_tool_name("show_analysis_md")
        return (
            f"(TraceLens snapshot #{snap}, gain at snapshot = {gain_str}% — "
            "full report not inlined; see profiler_digest above or call the "
            f"{show_analysis_md} context tool for the verbatim analysis.md.)"
        )

    def _format_profiler_digest(self) -> str:
        """Compact bottleneck-focused profiler block; ``(none)`` until a snapshot lands."""
        from hyperloom.inference_optimizer.roofline_snapshot import build_profiler_digest

        digest = build_profiler_digest(
            self.roofline_snapshots,
            self.last_trace_analyze,
        )
        if not digest:
            return "(none)"
        return f"\n{digest}\n"

    def _format_trace_analyze_blob(self, blob: dict[str, Any] | None) -> str:
        """Render a trace-analyze cache blob as a compact prompt line."""
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
        skipped = _skipped_kernels_suffix(blob) if not ids else ""
        return f"{base}{skipped}{_trace_health_warnings_suffix(blob)}"
