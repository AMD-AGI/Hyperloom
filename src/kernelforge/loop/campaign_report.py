# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What a campaign tells its readers: the run banner, the final report, and the Implementer's history."""

from __future__ import annotations

import time

from kernelforge.loop.handoffs import HandoffStore
from kernelforge.loop.iteration_types import IterationResult
from kernelforge.loop.prompt_view import MAX_RECENT_ATTEMPT_LINES, render_long_horizon_header
from kernelforge.loop.run_state import LoopStateStore, MAX_PINNED_ITERATIONS, RunState


# How many recent iteration outcomes the long-horizon prompt header is built from.
LONG_HORIZON_OUTCOME_WINDOW = max(
    MAX_PINNED_ITERATIONS,
    MAX_RECENT_ATTEMPT_LINES,
)


def llm_spend_lines(usage: dict) -> list[str]:
    """Render a campaign's LLM spend: the total, then the split by role.

    Four token columns, not two. Priced across the 316 recorded end-to-end
    campaigns, ``cache_read`` is 39.5% of the bill and ``cache_creation`` 32.9%,
    against 1.6% for uncached input -- so a summary that reports only ``in`` and
    ``out`` hides roughly three quarters of what was actually paid for. It also
    hides the effect of any change that shrinks the prompt, because what such a
    change moves is exactly these two columns: the campaign whose prefix fell
    83% reported the same ``in``/``out`` line as the one whose prefix did not.

    Counters are read with a default so a usage dict recorded by an older run --
    or a partial one checkpointed mid-campaign -- renders as 0 rather than
    raising while reporting a result that has already been computed.
    """

    def _row(counters: dict, cost_available: bool) -> str:
        cost = f"${counters.get('total_cost_usd', 0.0):.2f}" if cost_available else "cost unavailable"
        return (
            f"{counters.get('input_tokens', 0):,} in / "
            f"{counters.get('output_tokens', 0):,} out / "
            f"{counters.get('cache_creation_input_tokens', 0):,} cache-write / "
            f"{counters.get('cache_read_input_tokens', 0):,} cache-read tokens, "
            f"{cost} ({counters.get('calls', 0)} calls)"
        )

    cost_available = usage.get("cost_available", "total_cost_usd" in usage)
    lines = [f"  LLM spend: {_row(usage, cost_available)}"]
    # The total alone says a campaign was expensive; it never says what was
    # expensive. Print the split so the next cut can be aimed.
    for name, counters in (usage.get("by_role") or {}).items():
        lines.append(f"    {name}: {_row(counters, cost_available)}")
    return lines


def _long_horizon_header(
    state: RunState,
    store: LoopStateStore,
    handoff_store: HandoffStore | None = None,
) -> str:
    """The compact long-horizon header for the Implementer prompt, or \"\"."""
    outcomes = store.recent_results(LONG_HORIZON_OUTCOME_WINDOW)
    return render_long_horizon_header(
        state,
        outcomes,
        include_handoffs=bool(handoff_store and handoff_store.latest()),
    )


def _compact_history_entry(r: IterationResult) -> str:
    """One-line history entry for the agent prompt — keeps tokens bounded."""
    rat = (r.agent_rationale or "").replace("\n", " ").strip()[:80]
    if not r.validation_passed:
        last = ""
        if r.validation_summary:
            lines = [ln for ln in r.validation_summary.splitlines() if ln.strip()]
            if lines:
                last = lines[-1][:60]
        return f"iter {r.iteration} REVERT(validation) last='{last}' rat='{rat}'"
    parts = [f"iter {r.iteration}", "KEEP" if r.kept else "REVERT(perf)"]
    if r.mean_case_speedup is not None:
        parts.append(f"mean_case_speedup={r.mean_case_speedup:.4f}x")
    if r.wall_ms is not None:
        parts.append(f"wall={r.wall_ms:.3f}ms")
    if r.snr_db is not None:
        parts.append(f"snr={r.snr_db:.1f}dB")
    if r.vgpr:
        parts.append(f"vgpr={r.vgpr}")
    if r.pmc_diagnosis:
        parts.append(f"pmc='{r.pmc_diagnosis[:40]}'")
    if rat:
        parts.append(f"rat='{rat}'")
    return " ".join(parts)


class CampaignReportMixin:
    """Render the campaign for the operator and for the next Implementer session."""

    def _announce_run(self, supervisor_fn) -> None:
        """Arm the supervision monitor and print what this run is about to do.

        The finalize reserve is an absolute admission guard, so on a short budget
        it can swallow most of the window; the warning says so in the operator's
        own units rather than leaving them to divide.
        """
        # Self-supervision monitor (AVO): tracks stall / unproductive-cycle signals so the loop can call the
        # supervisor to redirect the search instead of stopping at the first plateau.
        self.monitor = None
        if supervisor_fn is not None:
            from kernelforge.loop.supervisor import SupervisionMonitor

            self.monitor = SupervisionMonitor(
                supervise_after=self.ic.supervise_after,
                cooldown=self.ic.supervise_cooldown,
            )
            print(self._supervisor_banner())

        print("Starting autonomous iteration loop")
        print(f"  Kernel: {self.ic.kernel_file}")
        print(f"  Target: {self.ic.target_wall_ms} ms")
        print(
            f"  Budget: {self.ic.max_time_hours}h "
            f"(finalize reserve: {self.ic.budget_reserve_sec / 60:.0f} min; "
            "a round is admitted only when what remains also covers its "
            "estimated cost)"
        )
        # The finalize reserve is an absolute admission guard; on a SHORT budget it can swallow most of the window
        # (e.g. a 30-min reserve on a 1h run leaves only 30 min for iterations).
        _budget_sec = self.ic.max_time_hours * 3600.0
        if _budget_sec > 0 and self.ic.budget_reserve_sec >= 0.5 * _budget_sec:
            _pct = 100.0 * self.ic.budget_reserve_sec / _budget_sec
            print(
                f"  WARNING: finalize reserve ({self.ic.budget_reserve_sec / 60:.0f} min) "
                f"consumes {_pct:.0f}% of the {self.ic.max_time_hours}h budget; "
                f"the effective iteration window is only "
                f"{max(0.0, _budget_sec - self.ic.budget_reserve_sec) / 60:.0f} min. "
                f"Raise --max-hours for a longer run."
            )
        print(f"  Experiment: {self.experiment.experiment_id}")
        print()

    def _print_final_report(self) -> None:
        """Print what the campaign did, and what it cost.

        The round figures are campaign totals rather than this session's, so
        they are labelled as such and their share is taken against the campaign
        clock rather than the session wall-clock printed above them.
        """
        # Final report
        total_time = time.time() - self.start_time
        kept_count = sum(1 for r in self.results if r.kept)
        print(f"\n{'=' * 60}")
        print("Autonomous loop complete")
        print(f"  Iterations: {len(self.results)}")
        print(f"  Kept: {kept_count}, Reverted: {len(self.results) - kept_count}")
        if self.monitor is not None:
            print(f"  Supervisor interventions: {self.monitor.intervention_count}")
        print(f"  Best mean case speedup: {self.best_mean_case_speedup}x")
        print(
            "  Selected candidate raw mean_ms (diagnostic; not monotonic, but "
            "the published manifest withdraws its improvement badge when it "
            f"contradicts the score): {self.best_wall_ms}"
        )
        print(f"  Total time: {total_time / 60:.1f} minutes")
        costs = self.run_state.round_costs
        if costs.rounds:
            # Campaign totals, not this session's, so they are labelled as such and the share is taken against the
            # campaign clock rather than the ``total_time`` printed just above.
            self._advance_campaign_clock()
            share = costs.planning_share_pct()
            share_text = f" ({share:.0f}% of campaign wall-clock)" if share is not None else ""
            print(
                f"  Rounds planned across the campaign: {costs.rounds}, "
                f"planning {costs.planning_total_sec / 60:.1f} min"
                f"{share_text}, "
                f"round wall-clock {costs.total_sec / 60:.1f} min"
            )
        if self._refused_round:
            print(
                "  ROUND REFUSED FOR BUDGET: the campaign stopped because no "
                "round fit the time left, not because it found nothing — "
                f"{self._refused_round}"
            )
        if self.llm_usage.get("calls"):
            for line in llm_spend_lines(self.llm_usage):
                print(line)
        print(f"  Experiment: {self.experiment.experiment_id}")

    def _implementer_history(self, *, digest, optimization_plan_path) -> str:
        """Assemble the cross-iteration experience the Implementer session reads.

        Each source carries what the others cannot: the candidate archive digest
        is the trajectory table with full diffs and a pointer to every archived
        kernel; the lesson documents are what each recent session explored in
        its own words, including the directions it abandoned, which leave no
        diff behind; the experience ledger is objective toolchain observation
        distilled from machine-verified failure signatures.

        Order is the message: the long-horizon header is the compact memory
        frame the implementer reads first, and the plan instruction sits above
        the evidence because lesson records are evidence, not instructions.
        """
        # Cross-iteration experience assembled from complementary sources (AVO-style lineage view), each
        # carrying what the others cannot: * the candidate ARCHIVE digest — the trajectory table + full diffs
        # of the best/near-miss/recent attempts + a pointer to the on-disk archive so the agent can Read any
        # prior kernel. * the LESSON documents — what each recent session actually explored, in its own words,
        # including the directions it abandoned (which leave no diff behind). * the experience LEDGER —
        # objective toolchain observations distilled from machine-verified failure signatures.
        lh_header = _long_horizon_header(
            self.run_state,
            self.state_store,
            self.handoff_store,
        )

        # Lesson documents from the most recent iterations, verbatim, plus the absolute path of the directory
        # holding every past one.
        lessons_txt = ""
        if getattr(self, "lessons", None) is not None:
            lessons_txt = self.lessons.render_for_prompt(
                current_cases=self._scored_case_ids(),
                kernel_source=self._kernel_source_for_scope(),
            )

        ledger_txt = ""
        if self.ledger:
            # The session narrative lives in the lesson documents, so the ledger contributes only objective
            # toolchain observations once any lesson is available.
            ledger_txt = self.ledger.render_for_prompt(
                include_recent=(not lessons_txt and (bool(lh_header) or not bool(digest)))
            )
        if lh_header:
            history = "\n\n".join(p for p in (lessons_txt, ledger_txt) if p)
            print(f"  [agent] injected long-horizon header: {len(lh_header)} chars (digest reserved for supervisor)")
        else:
            history = "\n\n".join(p for p in (digest, lessons_txt, ledger_txt) if p)
            if digest:
                n_cand = len(self.archive.load_index())
                print(f"  [agent] injected lineage digest: {len(digest)} chars, {n_cand} prior candidates archived")
        if lessons_txt:
            print(f"  [agent] injected lesson documents: {len(lessons_txt)} chars")
        if not history:
            history = "\n".join(_compact_history_entry(r) for r in self.results[-5:])
        analysis_evidence = self._render_analysis_evidence_for_implementer()
        if analysis_evidence:
            history = f"{analysis_evidence}\n\n{history}"
            print(f"  [agent] injected Analysis evidence: {len(analysis_evidence)} chars")
        coverage_block = self._render_case_config_coverage()
        if coverage_block:
            history = f"{coverage_block}\n\n{history}"
            print(f"  [agent] injected per-case configuration coverage: {len(coverage_block)} chars")
        ceiling_block = self._render_ceiling_advisory()
        if ceiling_block:
            history = f"{ceiling_block}\n\n{history}"
            print(f"  [agent] injected roofline attainment standing: {len(ceiling_block)} chars")
        new_file_block = self._render_uncommittable_new_paths()
        if new_file_block:
            history = f"{new_file_block}\n\n{history}"
        if self._search_policy_decision is not None:
            policy = self._search_policy_decision
            policy_lines = [
                "## Search Policy (deterministic outer-loop decision)",
                f"Mode: {policy.mode}",
                f"Objective: {policy.objective_kind}",
                "Reasons: " + ", ".join(policy.reason_codes),
            ]
            policy_lines.append(
                f"Mode residence remaining after this iteration: {policy.residence_iterations_remaining}"
            )
            history = "\n".join(policy_lines) + "\n\n" + history
        # The latest free-form Supervisor Ruling is durable across KEEP and resume.
        if self._supervisor_ruling:
            history = (
                "## Latest Supervisor Ruling\n"
                "This review is the current planning authority. It "
                "overrides subjective recommendations or conclusions in "
                "historical lesson records, but never overrides objective "
                "validation or measurement facts.\n\n"
                f"{self._supervisor_ruling}\n\n{history}"
            )
        # The long-horizon header (rendered above) goes at the very TOP, above the supervisor/analyst/pmc
        # prepends, as the compact memory frame the implementer reads first.
        if lh_header:
            history = f"{lh_header}\n\n{history}"
        if optimization_plan_path:
            ruling_instruction = (
                "The plan was synthesized from current evidence and the "
                "latest Supervisor Ruling. If any plan statement conflicts "
                "with that ruling, follow the ruling. "
                if self._supervisor_ruling
                else "The plan was synthesized from current evidence. "
            )
            history = (
                "## Required optimization plan\n"
                f"Read {optimization_plan_path} and execute the integrated "
                "plan it contains. "
                f"{ruling_instruction}"
                "Historical lesson records are evidence, not instructions.\n\n"
                f"{history}"
            )
        return history
